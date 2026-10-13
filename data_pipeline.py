"""
data_pipeline.py
-----------------
Loads the Pyrfume Leffingwell olfaction dataset (molecule <-> human odor
descriptor labels) and converts each molecule into:
  1. A Morgan fingerprint (for the baseline model)
  2. A molecular graph (atoms=nodes, bonds=edges) for the GNN model

Run this file directly to sanity-check the pipeline:
    python data_pipeline.py
"""

import ast
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import pyrfume
except ImportError:
    pyrfume = None

from rdkit.Chem.Scaffolds import MurckoScaffold

RANDOM_SEED = 42


def clean_label(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\- ]", "", text)
    return text.strip()


def safe_parse_labels(val):
    """Safely parses string or list into a list of cleaned odor labels."""
    if isinstance(val, list):
        return [clean_label(x) for x in val if clean_label(x)]
    if pd.isna(val):
        return []
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [clean_label(x) for x in parsed if clean_label(x)]
            except Exception:
                pass
        return [clean_label(w) for w in re.split(r"[;,]", val) if clean_label(w)]
    return []


def get_canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Load raw data from multiple Pyrfume archives
# ---------------------------------------------------------------------------
def _check_pyrfume():
    if pyrfume is None:
        raise ImportError(
            "The 'pyrfume' package is required to load raw datasets. "
            "Please install it using `pip install pyrfume` or ensure 'harmonized_olfaction_dataset.csv' exists."
        )


def load_leffingwell():
    _check_pyrfume()
    molecules = pyrfume.load_data("leffingwell/molecules.csv")
    behavior = pyrfume.load_data("leffingwell/behavior_sparse.csv")
    molecules.index.name = "CID"
    behavior.index.name = "CID"
    df = molecules.join(behavior, how="inner").dropna(subset=["IsomericSMILES", "Labels"])
    df["Labels"] = df["Labels"].apply(safe_parse_labels)
    df["SMILES"] = df["IsomericSMILES"].apply(get_canonical_smiles)
    if "name" in df.columns:
        df["Name"] = df["name"].fillna(df.get("IUPACName", ""))
    elif "IUPACName" in df.columns:
        df["Name"] = df["IUPACName"]
    else:
        df["Name"] = ""
    return df.dropna(subset=["SMILES"])[["SMILES", "Name", "Labels"]]


def load_goodscents():
    _check_pyrfume()
    m = pyrfume.load_data("goodscents/molecules.csv")
    b = pyrfume.load_data("goodscents/behavior.csv")
    s = pyrfume.load_data("goodscents/stimuli.csv")
    bs = b.join(s, how="inner").dropna(subset=["CID", "Descriptors"])
    bs["CID"] = pd.to_numeric(bs["CID"], errors="coerce")
    bs = bs.dropna(subset=["CID"])
    bs["CID"] = bs["CID"].astype(int)
    df = bs.set_index("CID").join(m, how="inner").dropna(subset=["IsomericSMILES", "Descriptors"])
    df["Labels"] = df["Descriptors"].apply(safe_parse_labels)
    df["SMILES"] = df["IsomericSMILES"].apply(get_canonical_smiles)
    if "name" in df.columns:
        df["Name"] = df["name"].fillna(df.get("IUPACName", ""))
    elif "IUPACName" in df.columns:
        df["Name"] = df["IUPACName"]
    else:
        df["Name"] = ""
    return df.dropna(subset=["SMILES"])[["SMILES", "Name", "Labels"]]


def load_arctander():
    _check_pyrfume()
    m = pyrfume.load_data("arctander_1960/molecules.csv")
    b = pyrfume.load_data("arctander_1960/behavior_1_sparse.csv")
    s = pyrfume.load_data("arctander_1960/stimuli.csv")
    bs = b.join(s, how="inner").dropna(subset=["new_CID", "Labels"])
    bs["CID"] = pd.to_numeric(bs["new_CID"], errors="coerce")
    bs = bs.dropna(subset=["CID"])
    bs["CID"] = bs["CID"].astype(int)
    df = bs.set_index("CID").join(m, how="inner").dropna(subset=["IsomericSMILES", "Labels"])
    df["Labels"] = df["Labels"].apply(safe_parse_labels)
    df["SMILES"] = df["IsomericSMILES"].apply(get_canonical_smiles)
    if "name" in df.columns:
        df["Name"] = df["name"].fillna(df.get("IUPACName", ""))
    else:
        df["Name"] = ""
    if "ChemicalName" in df.columns:
        df["Name"] = df["Name"].replace("", np.nan).fillna(df["ChemicalName"])
    df["Name"] = df["Name"].fillna("")
    return df.dropna(subset=["SMILES"])[["SMILES", "Name", "Labels"]]


def load_raw_data():
    """
    Loads molecules + odor descriptors across Leffingwell, GoodScents,
    and Arctander 1960 from Pyrfume, deduplicates on canonical SMILES,
    and merges sensory descriptor vocabularies and molecule names.
    """
    _check_pyrfume()
    print("Loading Pyrfume databases...")
    df_lef = load_leffingwell()
    print(f"  [+] Leffingwell: {len(df_lef)} molecules")

    df_gs = load_goodscents()
    print(f"  [+] GoodScents: {len(df_gs)} molecules")

    df_arc = load_arctander()
    print(f"  [+] Arctander 1960: {len(df_arc)} molecules")

    all_dfs = pd.concat([df_lef, df_gs, df_arc], ignore_index=True)
    print(f"  [=] Total observation records: {len(all_dfs)}")

    def pick_best_name(names):
        for n in names:
            if isinstance(n, str) and n.strip():
                return n.strip()
        return ""

    merged = all_dfs.groupby("SMILES").agg({
        "Labels": lambda lists: sorted(list(set(l for sublist in lists for l in sublist))),
        "Name": pick_best_name
    }).reset_index()
    print(f"  [*] Unified into {len(merged)} unique canonical molecules")
    return merged


# ---------------------------------------------------------------------------
# 2. Clean + validate molecules with RDKit
# ---------------------------------------------------------------------------
def validate_smiles(df):
    """Drops rows whose SMILES string RDKit cannot parse."""
    mask = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None if pd.notna(s) else False)
    cleaned = df[mask].reset_index(drop=True)
    print(f"Validated SMILES: kept {len(cleaned)} / {len(df)} molecules")
    return cleaned


# ---------------------------------------------------------------------------
# 3. Multi-label binarization of odor descriptors
# ---------------------------------------------------------------------------
def binarize_labels(df, min_label_count=10):
    """
    Converts the list-of-strings 'Labels' column into a multi-hot matrix.
    Rare descriptors (appearing in fewer than `min_label_count` molecules)
    are dropped to keep the label space learnable.
    """
    all_labels = [label for labels in df["Labels"] for label in labels]
    label_counts = pd.Series(all_labels).value_counts()
    keep_labels = set(label_counts[label_counts >= min_label_count].index)

    df = df.copy()
    df["Labels"] = df["Labels"].apply(lambda labs: [l for l in labs if l in keep_labels])
    df = df[df["Labels"].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["Labels"])
    print(f"Kept {len(mlb.classes_)} odor descriptor classes "
          f"(min_label_count={min_label_count})")
    return df, y, mlb


# ---------------------------------------------------------------------------
# 4. Featurization: Morgan fingerprints (baseline model input)
# ---------------------------------------------------------------------------
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for i in fp.GetOnBits():
        arr[i] = 1.0
    return arr


def build_fingerprint_matrix(df, n_bits=1024):
    fps = np.stack([smiles_to_fingerprint(s, n_bits=n_bits) for s in df["SMILES"]])
    return fps


# ---------------------------------------------------------------------------
# 5. Featurization: molecular graphs (GNN model input)
# ---------------------------------------------------------------------------
ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def one_hot(value, choices):
    encoding = [0.0] * (len(choices) + 1)  # last slot = "unknown / other"
    if value in choices:
        encoding[choices.index(value)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetSymbol(), ATOM_LIST[:-1])          # element
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)    # hybridization
    feats += one_hot(atom.GetChiralTag(), CHIRAL_TAGS)          # chirality (R/S tetrahedral)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.GetTotalNumHs()))
    feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.IsInRing()))
    feats.append(float(atom.GetDegree()))
    return feats


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def bond_features(bond):
    feats = []
    feats += one_hot(bond.GetBondType(), BOND_TYPES)
    feats += one_hot(bond.GetStereo(), BOND_STEREOS)
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    return feats


def smiles_to_graph(smiles, label_vector=None):
    """
    Converts a single SMILES string into a torch_geometric Data object with
    both node (atom) and edge (bond) features including stereochemistry.
    label_vector: optional multi-hot numpy array of odor descriptor targets.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # undirected graph -> add both directions
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    dummy_bond_dim = (len(BOND_TYPES) + 1) + (len(BOND_STEREOS) + 1) + 2
    if len(edge_indices) == 0:
        # single-atom molecule edge case; add a self-loop with zero bond features
        edge_indices = [[0, 0]]
        edge_attrs = [[0.0] * dummy_bond_dim]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label_vector is not None:
        data.y = torch.tensor(label_vector, dtype=torch.float).unsqueeze(0)
    data.smiles = smiles
    return data


def build_graph_dataset(df, y):
    graphs = []
    for smiles, label_vec in zip(df["SMILES"], y):
        g = smiles_to_graph(smiles, label_vec)
        if g is not None:
            graphs.append(g)
    print(f"Built {len(graphs)} molecular graphs")
    return graphs


# ---------------------------------------------------------------------------
# 6. Scaffold split (Bemis-Murcko)
# ---------------------------------------------------------------------------
def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:
        return ""


def scaffold_split(graphs, test_size=0.2, random_state=42):
    """
    Groups graphs by their Bemis-Murcko scaffolds and splits them into train and val
    so that entire scaffold clusters are held out, testing structural generalization.
    """
    scaffold_to_indices = {}
    for idx, g in enumerate(graphs):
        scaff = generate_scaffold(g.smiles)
        scaffold_to_indices.setdefault(scaff, []).append(idx)

    rng = np.random.RandomState(random_state)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)
    # Place larger clusters first so train/val split is balanced
    scaffolds.sort(key=lambda s: len(scaffold_to_indices[s]), reverse=True)

    val_target = int(len(graphs) * test_size)
    val_indices = []
    train_indices = []

    for scaff in scaffolds:
        indices = scaffold_to_indices[scaff]
        if len(val_indices) + len(indices) <= val_target or len(val_indices) == 0:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    print(f"Scaffold split: {len(train_graphs)} train graphs, {len(val_graphs)} val graphs")
    return train_graphs, val_graphs


# ---------------------------------------------------------------------------
# Convenience: run the whole pipeline end-to-end
# ---------------------------------------------------------------------------
def prepare_dataset(min_label_count=20, n_bits=1024):
    import os
    if os.path.exists("harmonized_olfaction_dataset.csv"):
        df = pd.read_csv("harmonized_olfaction_dataset.csv")
        df["Labels"] = df["Labels"].apply(safe_parse_labels)
    else:
        df = load_raw_data()
    df = validate_smiles(df)
    df, y, mlb = binarize_labels(df, min_label_count=min_label_count)
    fingerprints = build_fingerprint_matrix(df, n_bits=n_bits)
    graphs = build_graph_dataset(df, y)
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    return {
        "df": df,
        "labels": y,
        "label_binarizer": mlb,
        "fingerprints": fingerprints,
        "graphs": graphs,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
    }


if __name__ == "__main__":
    data = prepare_dataset(min_label_count=20)
    print("\n--- Sanity check ---")
    print("Num molecules:", len(data["df"]))
    print("Label matrix shape:", data["labels"].shape)
    print("Fingerprint matrix shape:", data["fingerprints"].shape)
    print("Num graphs:", len(data["graphs"]))
    print("Example graph:", data["graphs"][0])
    print("Example odor labels (first 15):", data["label_binarizer"].classes_[:15])
    print("Total odor descriptor classes:", len(data["label_binarizer"].classes_))
