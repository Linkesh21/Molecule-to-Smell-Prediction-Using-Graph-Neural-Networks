from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
import requests
from rdkit import Chem
from torch_geometric.data import Batch

# Import models and utilities
from data_pipeline import prepare_dataset, smiles_to_graph
from gnn_model import SmellGNN
from generative_model import SmilesVAE, latent_search

app = FastAPI(title="Mol_Smell API")

# Global variables for models and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = None
vae_model = None
vocab = None
label_classes = None

def load_models():
    global gnn_model, vae_model, vocab, label_classes
    print("Loading datasets and models...")
    data = prepare_dataset(min_label_count=20)
    label_classes = data["label_binarizer"].classes_
    node_dim = data["node_dim"]
    edge_dim = data["edge_dim"]
    
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
    """Attempts to parse query as SMILES, else fetches SMILES from NCI Cactus by name."""
    mol = Chem.MolFromSmiles(query)
    if mol is not None:
        return query # It's a valid SMILES
    
    # Try resolving name to SMILES using NCI Cactus
    url = f"https://cactus.nci.nih.gov/chemical/structure/{query}/smiles"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            smiles = resp.text.strip()
            # Verify the returned SMILES is valid
            if Chem.MolFromSmiles(smiles) is not None:
                return smiles
    except Exception as e:
        pass
    
    return None

class PredictRequest(BaseModel):
    molecule: str

class GenerateRequest(BaseModel):
    target: dict
    n_samples: int = 1000

@app.post("/api/predict")
def predict_odor(req: PredictRequest):
    smiles = resolve_to_smiles(req.molecule)
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
    
    return {
        "smiles": smiles,
        "predictions": results[:10] # Return top 10
    }

@app.post("/api/generate")
def generate_molecules(req: GenerateRequest):
    if vae_model is None:
        raise HTTPException(status_code=500, detail="Generative model is not loaded.")
        
    # Run the optimized latent search
    results = latent_search(req.target, gnn_model, vae_model, vocab, label_classes, n_samples=req.n_samples)
    
    formatted_results = [{"smiles": res[0], "score": res[1]} for res in results]
    return {"candidates": formatted_results}

# Mount static files for the frontend
import os
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
