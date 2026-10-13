from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
import requests
import urllib.parse
from rdkit import Chem
from torch_geometric.data import Batch
import sys
import os

# Import models and utilities
from data_pipeline import prepare_dataset, smiles_to_graph, get_canonical_smiles
from gnn_model import SmellGNN
from generative_model import SmilesVAE, latent_search, Vocab

# Ensure Vocab is findable by pickle in __main__
setattr(sys.modules['__main__'], 'Vocab', Vocab)
setattr(sys.modules.setdefault('generative_model', sys.modules[__name__]), 'Vocab', Vocab)

app = FastAPI(title="Mol_Smell API")

# Global variables for models and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = None
vae_model = None
vocab = None
label_classes = None
smiles_to_name = {}
name_to_smiles = {}

def load_models():
    global gnn_model, vae_model, vocab, label_classes, smiles_to_name, name_to_smiles
    print("Loading datasets and models...")
    data = prepare_dataset(min_label_count=20)
    label_classes = data["label_binarizer"].classes_
    node_dim = data["node_dim"]
    edge_dim = data["edge_dim"]
    
    # Build local name <-> smiles cache from dataset
    df = data["df"]
    if "Name" in df.columns:
        for _, row in df.iterrows():
            s = str(row["SMILES"]).strip()
            n = str(row["Name"]).strip()
            if s and n:
                smiles_to_name[s] = n
                name_to_smiles[n.lower()] = s
        print(f"Loaded {len(smiles_to_name)} named molecules into local dictionary.")
    
    # Load GNN
    gnn_model = SmellGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128, n_labels=len(label_classes)).to(device)
    try:
        gnn_model.load_state_dict(torch.load("gnn_model.pt", map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Could not load GNN model weights: {e}")
    gnn_model.eval()

    # Load VAE
    try:
        checkpoint = torch.load("smiles_vae.pt", map_location=device, weights_only=False)
        vocab = checkpoint["vocab"]
        vae_model = SmilesVAE(vocab.vocab_size).to(device)
        vae_model.load_state_dict(checkpoint["model_state"])
        vae_model.eval()
    except Exception as e:
        print(f"Warning: Could not load VAE model weights: {e}")
        
load_models()

def resolve_to_smiles(query: str) -> str:
    """Attempts to parse query as SMILES, check dataset names, or fetch via PubChem/Cactus."""
    q_clean = query.strip()
    mol = Chem.MolFromSmiles(q_clean)
    if mol is not None:
        return get_canonical_smiles(q_clean) or q_clean
    
    # 1. Exact match in local dataset
    q_lower = q_clean.lower()
    if q_lower in name_to_smiles:
        return name_to_smiles[q_lower]
        
    # 2. Substring or comma-separated match in dataset (e.g. 'toluene' in 'toluene')
    for n, s in name_to_smiles.items():
        if q_lower == n or q_lower in [part.strip() for part in n.split(",")]:
            return s
            
    # 3. Try PubChem by name
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(q_clean)}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            smi = resp.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('CanonicalSMILES')
            if smi and Chem.MolFromSmiles(smi):
                return smi
    except Exception:
        pass
    
    # 4. Try resolving name to SMILES using NCI Cactus
    url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(q_clean)}/smiles"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            smiles = resp.text.strip()
            if Chem.MolFromSmiles(smiles) is not None:
                return smiles
    except Exception:
        pass
    
    return None

class PredictRequest(BaseModel):
    molecule: str

class GenerateRequest(BaseModel):
    target: dict
    n_samples: int = 1000

def get_molecule_name(smiles: str) -> str:
    csmi = get_canonical_smiles(smiles) or smiles
    # 1. Local dataset lookup (instant and authoritative for all 7012 compounds)
    if csmi in smiles_to_name:
        return smiles_to_name[csmi]
    if smiles in smiles_to_name:
        return smiles_to_name[smiles]

    # 2. PubChem lookup fallback via Compound Title / IUPACName (fastest & standard)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(csmi)}/property/Title,IUPACName/JSON"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            props = res.json().get('PropertyTable', {}).get('Properties', [{}])[0]
            title = props.get('Title')
            iupac = props.get('IUPACName')
            chosen = title or iupac
            if chosen:
                smiles_to_name[csmi] = chosen
                return chosen
    except Exception:
        pass

    # 3. Cactus lookup fallback
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(csmi)}/iupac_name"
        res = requests.get(url, timeout=3)
        if res.status_code == 200 and res.text.strip() and "<html" not in res.text.lower():
            iupac = res.text.strip()
            smiles_to_name[csmi] = iupac
            return iupac
    except Exception:
        pass

    # 4. Computed formula fallback for novel synthetic molecules
    mol = Chem.MolFromSmiles(csmi)
    if mol is not None:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        formula = CalcMolFormula(mol)
        return f"Novel Molecule ({formula})"
        
    return "Novel Molecule"

@app.post("/api/predict")
def predict_odor(req: PredictRequest):
    query = req.molecule.strip()
    
    # If the user typed a valid SMILES, look up its name
    if Chem.MolFromSmiles(query) is not None:
        smiles = get_canonical_smiles(query) or query
        name = get_molecule_name(smiles)
    else:
        # If the user typed a name, resolve to SMILES and look up official name
        smiles = resolve_to_smiles(query)
        if smiles:
            name = get_molecule_name(smiles)
            if "Novel Molecule" in name:
                name = query
        else:
            name = query
        
    if not smiles:
        raise HTTPException(status_code=400, detail=f"Could not resolve '{req.molecule}' to a valid structure.")
        
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Failed to build graph from SMILES.")
        
    graph = graph.to(device)
    batch = Batch.from_data_list([graph]).to(device)
    
    with torch.no_grad():
        preds = torch.sigmoid(gnn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)).squeeze(0).cpu().numpy()
        
    # Zip classes with predictions and sort
    results = [{"odor": label_classes[i], "score": float(preds[i])} for i in range(len(label_classes))]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    mol = Chem.MolFromSmiles(smiles)
    formula = ""
    mol_weight = 0.0
    if mol is not None:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.Chem.Descriptors import MolWt
        formula = CalcMolFormula(mol)
        mol_weight = round(float(MolWt(mol)), 2)
        
    return {
        "smiles": smiles,
        "name": name,
        "formula": formula,
        "mol_weight": mol_weight,
        "predictions": results[:10] # Return top 10
    }

@app.post("/api/generate")
def generate_molecules(req: GenerateRequest):
    if vae_model is None:
        raise HTTPException(status_code=500, detail="Generative model is not loaded.")
        
    # Run the optimized latent search
    results = latent_search(req.target, gnn_model, vae_model, vocab, label_classes, n_samples=req.n_samples)
    
    formatted_results = [{"smiles": res[0], "score": res[1], "name": get_molecule_name(res[0])} for res in results]
    return {"candidates": formatted_results}

@app.get("/api/molecule_image")
def get_molecule_image(smiles: str, width: int = 340, height: int = 220):
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception:
            mol = None
            
    if mol is None:
        # High quality fallback SVG card so image NEVER fails or breaks
        fallback_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" rx="10"/>
  <g transform="translate({width/2}, {height/2 - 15})">
    <polygon points="0,-25 22,-12 22,12 0,25 -22,12 -22,-12" fill="none" stroke="#94a3b8" stroke-width="2.5"/>
    <circle cx="0" cy="0" r="10" fill="none" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="3,3"/>
  </g>
  <text x="50%" y="{height/2 + 32}" font-family="Plus Jakarta Sans, sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#64748b">Structure Preview</text>
  <text x="50%" y="{height/2 + 48}" font-family="JetBrains Mono, monospace" font-size="10" text-anchor="middle" fill="#94a3b8">Complex/Novel Structure</text>
</svg>'''
        return Response(content=fallback_svg, media_type="image/svg+xml")
        
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
        
    d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2.4
    opts.minFontSize = 13
    opts.maxFontSize = 22
    opts.centreMoleculesBeforeDrawing = True
    opts.addStereoAnnotation = True
    opts.additionalAtomLabelPadding = 0.08
    
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol)
    except Exception:
        try:
            d2d.DrawMolecule(mol)
        except Exception:
            pass
            
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()
    return Response(content=svg_text, media_type="image/svg+xml")

# Mount static files for the frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
