# Molecule-to-Smell Prediction (Computational Olfaction)

Predicting human odor descriptors (e.g. "woody", "fruity", "green") directly
from molecular structure, and laying the groundwork for the inverse problem:
generating a candidate molecule for a target smell.

## Status: Upgraded Forward Model (GINE + Stereochemistry + Attention)

Two models are implemented and trained on the **Pyrfume Leffingwell** dataset
(3,522 molecules, 113 odor descriptor classes after filtering rare labels):

| Model | Representation | Val mean AUROC | Val mean AUPRC | Features |
|---|---|---|---|---|
| Baseline | Morgan fingerprint (1024-bit) + MLP | ~0.874 | ~0.332 | Bit-vector hashing |
| Upgraded GNN | Molecular graph + **GINEConv** | **~0.884** | **~0.332** | Bond types, stereochemistry, attention readout, `pos_weight` |

### Key Improvements Over the Initial Prototype:
- **Bond Feature Integration**: Replaced standard `GINConv` with `GINEConv`, fusing bond orders, aromaticity, and stereochemistry into every message passing step.
- **Chirality & Stereochemistry**: Added tetrahedral chiral tags and bond stereo configurations, enabling the model to distinguish enantiomers (e.g. *(R)*-carvone vs *(S)*-carvone).
- **Attention Readout**: Replaced naive pooling with `AttentionalAggregation` combined with global mean pooling so dominant functional groups ("odorophores" like thiols `-SH` or esters) are preserved.
- **Class Imbalance Loss**: Equipped `BCEWithLogitsLoss` with positive class weighting (`pos_weight`) to boost sensitivity across rare odor descriptors.
- **Dual Metrics & Scaffold Split**: Tracks both AUROC and AUPRC (Average Precision), and supports Bemis-Murcko scaffold splitting.

## Files

- `data_pipeline.py` — loads Pyrfume data, validates SMILES with RDKit,
  binarizes odor labels, encodes chirality/bond stereochemistry, implements Bemis-Murcko scaffold splitting, and builds both Morgan fingerprints and molecular graphs (`torch_geometric.data.Data` objects).
- `baseline_model.py` — fingerprint + MLP baseline, multi-label
  classification with `BCEWithLogitsLoss`, evaluated with per-label AUROC and AUPRC.
- `gnn_model.py` — upgraded GINE-based graph neural network with attention readout and stereochemistry, the primary forward model.

## Running it

```bash
# Using the local environment
.venv\Scripts\python.exe data_pipeline.py     # sanity-check data pipeline and featurization
.venv\Scripts\python.exe baseline_model.py    # trains and saves baseline_model.pt
.venv\Scripts\python.exe gnn_model.py         # trains and saves gnn_model.pt
```

Each script is self-contained and prints validation AUROC every 5 epochs.
Training both models takes well under a minute on CPU for this dataset size.

## How to read the metric

AUROC (Area Under the ROC Curve) is computed **per odor descriptor label**
independently (this is a multi-label problem — a molecule can be both
"woody" and "sweet"), then averaged across all 113 labels. 0.5 = random
guessing, 1.0 = perfect ranking. ~0.89 mean AUROC means the model is quite
good at ranking, for a given descriptor, which molecules are more likely
to have it — this is the standard evaluation protocol used in the
olfaction ML literature (including the Osmo paper), so these numbers are
directly comparable to published results.

## Next steps (not yet implemented)

This is the forward (molecule → smell) half of the project. The inverse
problem (smell → candidate molecule) is the actual novel contribution and
is planned as follows:

1. **Molecular VAE**: train a variational autoencoder over molecular graphs
   (ideally a junction-tree VAE, which builds molecules from valid chemical
   substructures so decoded outputs are chemically valid) to get a
   continuous latent space of molecules.
2. **Latent-space search**: given a target odor profile, use the trained
   GNN above to score candidate molecules, and search the VAE's latent
   space (gradient-based or Bayesian optimization) for a latent point whose
   decoded molecule's *predicted* profile matches the target.
3. **Evaluation**: since we can't literally smell-test outputs, validate by
   (a) round-trip consistency — does the forward model confirm the target
   profile on the generated molecule — and (b) chemical validity/novelty/
   uniqueness metrics standard in molecular generation (as in the MOSES
   benchmark).

## Data source

[Pyrfume](https://pyrfume.org/) — an open-source repository of curated
olfaction datasets. This project uses the Leffingwell dataset specifically
(molecule SMILES + human-assigned odor descriptor labels).
