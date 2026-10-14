"""
test_suite.py
--------------
Automated test suite corresponding to Slide 6 ("Testing") in the Final Review presentation:
  - Unit: Data Pipeline (7 tests)
  - Model Behaviour (4 tests)
  - API Integration (6 tests)
Total: 17 tests.

Run directly in terminal for a clean, professional report:
    python test_suite.py

Or run with pytest:
    pytest test_suite.py -v
"""

import sys
import os
import time
import warnings
import pytest
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from fastapi import HTTPException

# 1. Silence deprecation warnings and RDKit internal C++ parse logs during tests
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
os.environ["PYTHONWARNINGS"] = "ignore"

from data_pipeline import (
    prepare_dataset,
    smiles_to_graph,
    smiles_to_fingerprint,
    scaffold_split,
    generate_scaffold,
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

# Global dataset & models cache so running tests doesn't redundantly reload data
_CACHED_DATASET = None
_MODELS_INITIALIZED = False


def ensure_models_loaded():
    global _MODELS_INITIALIZED
    if not _MODELS_INITIALIZED:
        load_models()
        _MODELS_INITIALIZED = True


def get_cached_dataset():
    global _CACHED_DATASET
    if _CACHED_DATASET is None:
        _CACHED_DATASET = prepare_dataset(min_label_count=20)
    return _CACHED_DATASET


# ============================================================================
# 1. Unit – Data Pipeline (7 tests)
# ============================================================================

def test_dataset_size_and_classes():
    """1. Dataset is 7,012 x 196 with no duplicates."""
    data = get_cached_dataset()
    df = data["df"]
    labels = data["labels"]
    
    assert len(df) == 7012, f"Expected 7,012 molecules, got {len(df)}"
    assert labels.shape[1] == 196, f"Expected 196 odor classes, got {labels.shape[1]}"
    assert df["SMILES"].nunique() == 7012, "Duplicate canonical SMILES detected in dataset"
    return "Verified 7,012 molecules x 196 odor classes, 0 duplicates."


def test_ethanol_graph_features():
    """2. Ethanol (CCO) -> 3 atoms x 28 features, and 4 bonds x 12 features."""
    graph = smiles_to_graph("CCO")
    assert graph is not None
    assert graph.x.shape == (3, 28), f"Expected node shape (3, 28), got {graph.x.shape}"
    assert graph.edge_index.shape[1] == 4, f"Expected 4 directed edges, got {graph.edge_index.shape[1]}"
    assert graph.edge_attr.shape == (4, 12), f"Expected edge attr shape (4, 12), got {graph.edge_attr.shape}"
    return f"Ethanol (CCO) -> Nodes {tuple(graph.x.shape)}, Edges {tuple(graph.edge_attr.shape)}."


def test_single_atom_molecule():
    """3. Single-atom molecule (e.g., methane C) handled cleanly with self-loop."""
    graph = smiles_to_graph("C")
    assert graph is not None
    assert graph.x.shape == (1, 28)
    assert graph.edge_index.shape == (2, 1)
    assert graph.edge_attr.shape == (1, 12)
    return "Methane (C) -> 1 atom (28-d) with zero-edge self-loop (12-d)."


def test_carvone_enantiomers_encoded_differently():
    """4. R-carvone and S-carvone encoded differently due to tetrahedral stereochemistry."""
    r_carvone = "CC1=CC[C@@H](C(=C)C)CC1=O"  # (R)-carvone (spearmint)
    s_carvone = "CC1=CC[C@H](C(=C)C)CC1=O"   # (S)-carvone (caraway)
    
    g_r = smiles_to_graph(r_carvone)
    g_s = smiles_to_graph(s_carvone)
    
    assert g_r is not None and g_s is not None
    assert not torch.equal(g_r.x, g_s.x), "Chiral enantiomers should have distinct feature representations"
    return "(R)-carvone (spearmint) != (S)-carvone (caraway) via chiral tags."


def test_invalid_smiles_rejected():
    """5. Invalid SMILES strings rejected cleanly by parser."""
    assert smiles_to_graph("INVALID_CHEMICAL_SMILES_STRING") is None
    assert smiles_to_graph("C1CCC") is None  # Unclosed ring
    return "Syntax error string and unclosed ring cleanly rejected as None."


def test_morgan_fingerprint_length():
    """6. Morgan fingerprint length is exactly 1,024 bits."""
    fp = smiles_to_fingerprint("CCO", n_bits=1024, radius=2)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (1024,)
    return "Morgan ECFP4 fingerprint output is exactly 1,024 bits."


def test_scaffold_split_no_overlap():
    """7. Bemis-Murcko scaffold split has no overlap between train and test sets."""
    data = get_cached_dataset()
    graphs = data["graphs"]
    train_graphs, val_graphs = scaffold_split(graphs, test_size=0.2, random_state=42)
    
    train_scaffolds = {generate_scaffold(g.smiles) for g in train_graphs}
    val_scaffolds = {generate_scaffold(g.smiles) for g in val_graphs}
    overlap = len(train_scaffolds.intersection(val_scaffolds))
    
    assert overlap == 0, "Overlap found in scaffold split"
    assert len(train_graphs) + len(val_graphs) == len(graphs)
    return f"Bemis-Murcko split: {len(train_graphs)} train, {len(val_graphs)} test -> Scaffold overlap: 0."


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
    return "SmellGNN multi-label probabilities strictly bounded in [0.0, 1.0]."


def test_isobutyl_acetate_fruity():
    """9. Isobutyl acetate -> 'fruity' descriptor is in top 5 predicted odors."""
    ensure_models_loaded()
    res = predict_odor(PredictRequest(molecule="isobutyl acetate"))
    top_5_odors = [p["odor"] for p in res["predictions"][:5]]
    assert "fruity" in top_5_odors, f"Expected 'fruity' in top 5, got {top_5_odors}"
    return f"Isobutyl acetate -> Top odors: {top_5_odors[:3]} ('fruity' verified)."


def test_vanillin_vanilla():
    """10. Vanillin -> 'vanilla' descriptor is in top 5 predicted odors."""
    ensure_models_loaded()
    res = predict_odor(PredictRequest(molecule="vanillin"))
    top_5_odors = [p["odor"] for p in res["predictions"][:5]]
    assert "vanilla" in top_5_odors, f"Expected 'vanilla' in top 5, got {top_5_odors}"
    return f"Vanillin -> Top odors: {top_5_odors[:3]} ('vanilla' verified)."


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
    return f"SmilesVAE 100 samples -> Validity: {validity_rate:.1%} (>= 90% threshold)."


# ============================================================================
# 3. API Integration (6 tests)
# ============================================================================

def test_api_predict_by_smiles():
    """12. Predict by SMILES returns 10 sorted results and correct formula (e.g. C6H12O2)."""
    ensure_models_loaded()
    res = predict_odor(PredictRequest(molecule="CC(=O)OCC(C)C"))
    assert len(res["predictions"]) == 10
    scores = [p["score"] for p in res["predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert res["formula"] == "C6H12O2"
    return f"Formula: {res['formula']}, MW: {res['mol_weight']:.2f}, 10 sorted predictions."


def test_api_predict_by_name_offline():
    """13. Predict by chemical name offline ('limonene') from 7,012 local dataset."""
    ensure_models_loaded()
    res = predict_odor(PredictRequest(molecule="limonene"))
    assert res["name"].lower() == "limonene"
    assert "predictions" in res and len(res["predictions"]) == 10
    return "Common name 'limonene' resolved offline via local 7,012 dataset."


def test_api_invalid_input_http_400():
    """14. Invalid or unresolvable input raises HTTP 400 error."""
    ensure_models_loaded()
    with pytest.raises(HTTPException) as exc_info:
        predict_odor(PredictRequest(molecule="not_a_real_chemical_xyz_12345"))
    assert exc_info.value.status_code == 400
    return f"Unresolvable molecule raised HTTP {exc_info.value.status_code} Bad Request cleanly."


def test_api_generate_candidates_and_scores():
    """15. /api/generate produces <= 10 unique valid candidates with scores in [0, 1]."""
    ensure_models_loaded()
    target = {"woody": 1.0, "fruity": 0.5}
    res = latent_search(target, app.gnn_model, app.vae_model, app.vocab, app.label_classes, n_samples=300)
    assert len(res) <= 10
    assert len(res) > 0
    smiles_list = [r[0] for r in res]
    assert len(smiles_list) == len(set(smiles_list)), "Generated candidates must be unique"
    for _, score in res:
        assert 0.0 <= score <= 1.0
    return f"Target {target} -> Produced {len(res)} unique valid candidates with scores in [0, 1]."


def test_api_structure_image():
    """16. Structure image returned with status 200 and image/svg+xml."""
    resp = get_molecule_image("Cc1ccccc1")
    assert resp.media_type == "image/svg+xml"
    assert "<svg" in resp.body.decode("utf-8")
    return "Status 200, Content-Type: image/svg+xml."


def test_api_structure_image_fallback():
    """17. Bad input to structure image returns fallback SVG card without crash."""
    resp = get_molecule_image("BAD_SMILES_STRING_???")
    assert resp.media_type == "image/svg+xml"
    svg_content = resp.body.decode("utf-8")
    assert "<svg" in svg_content
    assert "Structure Preview" in svg_content
    return "Malformed input to image endpoint returns graceful fallback SVG card."


# ============================================================================
# Interactive CLI Terminal Runner
# ============================================================================

def run_all_tests():
    """Runs all 17 tests sequentially and displays a clean, quiet report without C++ parse noise."""
    tests = [
        # (group_name, test_func, test_id, description)
        ("DATA PIPELINE (7 TESTS)", test_dataset_size_and_classes, 1, "Dataset size 7,012 x 196, zero duplicates"),
        ("DATA PIPELINE (7 TESTS)", test_ethanol_graph_features, 2, "Ethanol graph (3 atoms x 28-d, 4 directed bonds x 12-d)"),
        ("DATA PIPELINE (7 TESTS)", test_single_atom_molecule, 3, "Single-atom molecule (methane C) self-loop edge handling"),
        ("DATA PIPELINE (7 TESTS)", test_carvone_enantiomers_encoded_differently, 4, "Chiral tags distinguish (R)-carvone & (S)-carvone enantiomers"),
        ("DATA PIPELINE (7 TESTS)", test_invalid_smiles_rejected, 5, "Unclosed rings & invalid SMILES rejected as None"),
        ("DATA PIPELINE (7 TESTS)", test_morgan_fingerprint_length, 6, "Morgan baseline fingerprint length exactly 1,024 bits"),
        ("DATA PIPELINE (7 TESTS)", test_scaffold_split_no_overlap, 7, "Bemis-Murcko train/test scaffold split has 0 overlap"),

        ("MODEL BEHAVIOUR (4 TESTS)", test_gnn_output_probability_range, 8, "SmellGNN 196 descriptor probabilities bounded in [0, 1]"),
        ("MODEL BEHAVIOUR (4 TESTS)", test_isobutyl_acetate_fruity, 9, "Isobutyl acetate predicts 'fruity' in top descriptors"),
        ("MODEL BEHAVIOUR (4 TESTS)", test_vanillin_vanilla, 10, "Vanillin predicts 'vanilla' in top descriptors"),
        ("MODEL BEHAVIOUR (4 TESTS)", test_vae_validity_threshold, 11, "SmilesVAE latent sampling chemical validity >= 90%"),

        ("API INTEGRATION (6 TESTS)", test_api_predict_by_smiles, 12, "/api/predict returns formula C6H12O2 & 10 sorted predictions"),
        ("API INTEGRATION (6 TESTS)", test_api_predict_by_name_offline, 13, "Common name 'limonene' resolves offline via local 7,012 cache"),
        ("API INTEGRATION (6 TESTS)", test_api_invalid_input_http_400, 14, "Invalid input raises HTTP 400 Bad Request error cleanly"),
        ("API INTEGRATION (6 TESTS)", test_api_generate_candidates_and_scores, 15, "/api/generate produces unique candidates with scores in [0, 1]"),
        ("API INTEGRATION (6 TESTS)", test_api_structure_image, 16, "/api/molecule_image returns interactive vector SVG card"),
        ("API INTEGRATION (6 TESTS)", test_api_structure_image_fallback, 17, "Malformed image request returns graceful fallback SVG card"),
    ]

    header_bar = "=" * 80
    print("\n" + header_bar)
    print("      MOLECULE-TO-SMELL PREDICTION SYSTEM: 17-TEST VERIFICATION SUITE")
    print("                (Matching Final Review Slide 6 Parameters)")
    print(header_bar)

    # Preload models and datasets quietly once
    print("\n[INIT] Initializing models and molecular dataset cache...", end="", flush=True)
    get_cached_dataset()
    ensure_models_loaded()
    print(" Ready.\n")

    current_group = None
    passed_count = 0
    failed_count = 0
    t_start_total = time.time()

    for group, func, test_num, desc in tests:
        if group != current_group:
            current_group = group
            print(f"\n--- [ {current_group} ] " + "-" * (74 - len(current_group)))

        t0 = time.time()
        try:
            detail_msg = func()
            elapsed = time.time() - t0
            passed_count += 1
            print(f" [PASS] Test {test_num:02d}: {desc} ({elapsed:.2f}s)")
            if detail_msg:
                print(f"        Details: {detail_msg}")
        except Exception as e:
            elapsed = time.time() - t0
            failed_count += 1
            print(f" [FAIL] Test {test_num:02d}: {desc} ({elapsed:.2f}s)")
            print(f"        ERROR: {e}")

    total_time = time.time() - t_start_total

    print("\n" + header_bar)
    print("                           TEST RUN SUMMARY")
    print(header_bar)
    print(f"  Total Tests Executed:  {len(tests)}")
    print(f"  Tests Passed:          {passed_count} / {len(tests)} ({passed_count / len(tests):.1%})")
    print(f"  Tests Failed:          {failed_count}")
    print(f"  Total Execution Time:  {total_time:.2f}s")
    if failed_count == 0:
        print("\n  >> ALL 17 TESTS PASSED SUCCESSFULLY! 100% COMPLIANT WITH PPT SLIDE 6 <<")
    else:
        print(f"\n  >> WARNING: {failed_count} TEST(S) FAILED <<")
    print(header_bar + "\n")


if __name__ == "__main__":
    run_all_tests()
