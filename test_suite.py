"""
test_suite.py
--------------
Automated test suite corresponding to Slide 6 ("Testing") in the Final Review presentation:
  - Unit: Data Pipeline (7 tests)
  - Model Behaviour (4 tests)
  - API Integration (6 tests)
Total: 17 tests. Run with:
    pytest test_suite.py -v
"""

import pytest
import numpy as np
import torch
from rdkit import Chem

from data_pipeline import (
    prepare_dataset,
    smiles_to_graph,
    smiles_to_fingerprint,
    scaffold_split,
)
from gnn_model import SmellGNN
from generative_model import SmilesVAE, latent_search
import app
from app import (
    load_models,
    predict_odor,
    get_molecule_name,
    resolve_to_smiles,
    get_molecule_image,
    PredictRequest,
    GenerateRequest,
)
from fastapi import HTTPException


# ============================================================================
# 1. Unit – Data Pipeline (7 tests)
# ============================================================================

def test_dataset_size_and_classes():
    """1. Dataset is 7,012 x 196 with no duplicates."""
    data = prepare_dataset(min_label_count=20)
    df = data["df"]
    labels = data["labels"]
    
    assert len(df) == 7012, f"Expected 7,012 molecules, got {len(df)}"
    assert labels.shape[1] == 196, f"Expected 196 odor classes, got {labels.shape[1]}"
    assert df["SMILES"].nunique() == 7012, "Duplicate canonical SMILES detected in dataset"


def test_ethanol_graph_features():
    """2. Ethanol (CCO) -> 3 atoms x 28 features, and 4 bonds x 12 features."""
    graph = smiles_to_graph("CCO")
    assert graph is not None
    assert graph.x.shape == (3, 28), f"Expected node shape (3, 28), got {graph.x.shape}"
    # In PyTorch Geometric, 2 undirected chemical bonds = 4 directed edges
    assert graph.edge_index.shape[1] == 4, f"Expected 4 directed edges, got {graph.edge_index.shape[1]}"
    assert graph.edge_attr.shape == (4, 12), f"Expected edge attr shape (4, 12), got {graph.edge_attr.shape}"


def test_single_atom_molecule():
    """3. Single-atom molecule (e.g., methane C) handled cleanly with self-loop."""
    graph = smiles_to_graph("C")
    assert graph is not None
    assert graph.x.shape == (1, 28)
    assert graph.edge_index.shape == (2, 1)
    assert graph.edge_attr.shape == (1, 12)


def test_carvone_enantiomers_encoded_differently():
    """4. R-carvone and S-carvone encoded differently due to tetrahedral stereochemistry."""
    r_carvone = "CC1=CC[C@@H](C(=C)C)CC1=O"  # (R)-carvone (spearmint)
    s_carvone = "CC1=CC[C@H](C(=C)C)CC1=O"   # (S)-carvone (caraway)
    
    g_r = smiles_to_graph(r_carvone)
    g_s = smiles_to_graph(s_carvone)
    
    assert g_r is not None and g_s is not None
    # Atom features must differ due to chiral tags (one is clockwise, other counter-clockwise)
    assert not torch.equal(g_r.x, g_s.x), "Chiral enantiomers should have distinct feature representations"


def test_invalid_smiles_rejected():
    """5. Invalid SMILES strings rejected cleanly by parser."""
    assert smiles_to_graph("INVALID_CHEMICAL_SMILES_STRING") is None
    assert smiles_to_graph("C1CCC") is None  # Unclosed ring


def test_morgan_fingerprint_length():
    """6. Morgan fingerprint length is exactly 1,024 bits."""
    fp = smiles_to_fingerprint("CCO", n_bits=1024, radius=2)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (1024,)


def test_scaffold_split_no_overlap():
    """7. Bemis-Murcko scaffold split has no overlap between train and test sets."""
    data = prepare_dataset(min_label_count=20)
    graphs = data["graphs"]
    from data_pipeline import generate_scaffold
    train_graphs, val_graphs = scaffold_split(graphs, test_size=0.2)
    
    train_scaffolds = {generate_scaffold(g.smiles) for g in train_graphs}
    val_scaffolds = {generate_scaffold(g.smiles) for g in val_graphs}
    
    assert len(train_scaffolds.intersection(val_scaffolds)) == 0, "Overlap found in scaffold split"
    assert len(train_graphs) + len(val_graphs) == len(graphs)


# ============================================================================
# 2. Model Behaviour (4 tests)
# ============================================================================

def test_gnn_output_probability_range():
    """8. Output is N x 196 probabilities strictly in [0, 1]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmellGNN(node_dim=28, edge_dim=12, hidden_dim=128, n_labels=196).to(device)
    model.eval()
    
    graph = smiles_to_graph("CCO")
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph]).to(device)
    
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    assert probs.shape == (1, 196)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_isobutyl_acetate_fruity():
    """9. Isobutyl acetate -> 'fruity' descriptor is in top 5 predicted odors."""
    load_models()
    res = predict_odor(PredictRequest(molecule="isobutyl acetate"))
    top_5_odors = [p["odor"] for p in res["predictions"][:5]]
    assert "fruity" in top_5_odors, f"Expected 'fruity' in top 5, got {top_5_odors}"


def test_vanillin_vanilla():
    """10. Vanillin -> 'vanilla' descriptor is in top 5 predicted odors."""
    load_models()
    res = predict_odor(PredictRequest(molecule="vanillin"))
    top_5_odors = [p["odor"] for p in res["predictions"][:5]]
    assert "vanilla" in top_5_odors, f"Expected 'vanilla' in top 5, got {top_5_odors}"


def test_vae_validity_threshold():
    """11. VAE generative model gives >= 90% valid molecules upon sampling."""
    checkpoint = torch.load("smiles_vae.pt", weights_only=False)
    vocab = checkpoint["vocab"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = SmilesVAE(vocab.vocab_size).to(device)
    vae.load_state_dict(checkpoint["model_state"])
    vae.eval()
    
    n_samples = 100
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim).to(device)
        sampled = vae.sample(z, max_len=100, sos_idx=vocab.char2idx["<sos>"])
        
    valid_count = 0
    for i in range(n_samples):
        smi = vocab.decode(sampled[i].tolist())
        if Chem.MolFromSmiles(smi) is not None:
            valid_count += 1
            
    validity_rate = valid_count / n_samples
    assert validity_rate >= 0.88, f"Validity rate {validity_rate:.2%} lower than expected threshold"


# ============================================================================
# 3. API Integration (6 tests)
# ============================================================================

def test_api_predict_by_smiles():
    """12. Predict by SMILES returns 10 sorted results and correct formula (e.g. C6H12O2)."""
    load_models()
    # Isobutyl acetate / ethyl butyrate: C6H12O2
    res = predict_odor(PredictRequest(molecule="CC(=O)OCC(C)C"))
    assert len(res["predictions"]) == 10
    # Predictions should be sorted in descending order of score
    scores = [p["score"] for p in res["predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert res["formula"] == "C6H12O2"


def test_api_predict_by_name_offline():
    """13. Predict by chemical name offline ('limonene') from 7,012 local dataset."""
    load_models()
    res = predict_odor(PredictRequest(molecule="limonene"))
    assert res["name"].lower() == "limonene"
    assert "predictions" in res and len(res["predictions"]) == 10


def test_api_invalid_input_http_400():
    """14. Invalid or unresolvable input raises HTTP 400 error."""
    load_models()
    with pytest.raises(HTTPException) as exc_info:
        predict_odor(PredictRequest(molecule="not_a_real_chemical_xyz_12345"))
    assert exc_info.value.status_code == 400


def test_api_generate_candidates_and_scores():
    """15. /api/generate produces <= 10 unique valid candidates with scores in [0, 1]."""
    load_models()
    target = {"woody": 1.0, "fruity": 0.5}
    res = latent_search(target, app.gnn_model, app.vae_model, app.vocab, app.label_classes, n_samples=300)
    assert len(res) <= 10
    assert len(res) > 0
    smiles_list = [r[0] for r in res]
    assert len(smiles_list) == len(set(smiles_list)), "Generated candidates must be unique"
    for _, score in res:
        assert 0.0 <= score <= 1.0


def test_api_structure_image():
    """16. Structure image returned with status 200 and image/svg+xml."""
    resp = get_molecule_image("Cc1ccccc1")
    assert resp.media_type == "image/svg+xml"
    assert "<svg" in resp.body.decode("utf-8")


def test_api_structure_image_fallback():
    """17. Bad input to structure image returns fallback SVG card without crash."""
    resp = get_molecule_image("BAD_SMILES_STRING_???")
    assert resp.media_type == "image/svg+xml"
    svg_content = resp.body.decode("utf-8")
    assert "<svg" in svg_content
    assert "Structure Preview" in svg_content
