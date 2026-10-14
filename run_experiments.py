"""Final-review experiments on the user's latest code (exported to ../final).
Mirrors train_baseline / train_gnn exactly (same split, optimiser, epochs, loss),
but repeats each run with 3 seeds and adds ablation variants."""
import sys, os, json, random, warnings, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")

FINAL = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(FINAL, "experiments_results.json")
sys.path.insert(0, FINAL)
os.chdir(FINAL)

print(f"============================================================")
print(f"           RUNNING MODEL EXPERIMENTS & ABLATION SUITE       ")
print(f"============================================================")
print(f"Project directory: {FINAL}")
print(f"Output results:    {OUT}\n")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader as TDL, TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GINConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import data_pipeline as dp
from baseline_model import FingerprintMLP, compute_mean_metrics
from gnn_model import SmellGNN

torch.set_num_threads(4)
data = dp.prepare_dataset(min_label_count=20)
G, Y, FP = data["graphs"], data["labels"], data["fingerprints"]
CLASSES = list(data["label_binarizer"].classes_)
ND, ED, NL = data["node_dim"], data["edge_dim"], Y.shape[1]
SEEDS = [0, 1, 2]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, "w"), indent=1)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class NoBondGNN(SmellGNN):
    """Same as SmellGNN but GINConv (bond features ignored)."""
    def __init__(self, node_dim, edge_dim, hidden_dim, n_labels, num_layers=3, dropout=0.2):
        super().__init__(node_dim, edge_dim, hidden_dim, n_labels, num_layers, dropout)
        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp)); in_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, edge_index))), p=self.dropout, training=self.training)
        return self.classifier(torch.cat([self.att_pool(x, batch), global_mean_pool(x, batch)], 1))


class NoAttnGNN(SmellGNN):
    """Same as SmellGNN but mean pooling only (no attention readout)."""
    def __init__(self, node_dim, edge_dim, hidden_dim, n_labels, num_layers=3, dropout=0.2):
        super().__init__(node_dim, edge_dim, hidden_dim, n_labels, num_layers, dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, n_labels))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, edge_index, edge_attr))), p=self.dropout, training=self.training)
        return self.classifier(global_mean_pool(x, batch))


def eval_gnn(model, loader):
    model.eval(); L, T = [], []
    with torch.no_grad():
        for b in loader:
            L.append(model(b.x, b.edge_index, b.edge_attr, b.batch).numpy()); T.append(b.y.numpy())
    L, T = np.concatenate(L), np.concatenate(T)
    a, p, _ = compute_mean_metrics(T, L)
    return float(a), float(p), L, T


def run_gnn(key, cls, split, seed, pos_weight=True, epochs=60):
    if key in res: return
    seed_all(seed)
    if split == "scaffold":
        tr, va = dp.scaffold_split(G, test_size=0.2, random_state=42)
    else:
        tr, va = train_test_split(G, test_size=0.2, random_state=42)
    tl = DataLoader(tr, batch_size=64, shuffle=True); vl = DataLoader(va, batch_size=64)
    model = cls(ND, ED, 128, NL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if pos_weight:
        yt = torch.cat([g.y for g in tr]); pc = yt.sum(0)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.clamp((len(tr) - pc) / (pc + 1e-5), 1.0, 15.0))
    else:
        crit = nn.BCEWithLogitsLoss()
    curve = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).backward(); opt.step()
        if ep % 5 == 0:
            a, p, L, T = eval_gnn(model, vl); curve.append([ep, a, p])
    out = {"auroc": a, "auprc": p, "curve": curve, "secs": time.time() - t0, "n_val": len(va)}
    if key.startswith("gnn_full_random"):
        per = {}
        for i, c in enumerate(CLASSES):
            if len(np.unique(T[:, i])) > 1:
                per[c] = [float(roc_auc_score(T[:, i], 1 / (1 + np.exp(-L[:, i])))), int(T[:, i].sum())]
        out["per_label"] = per
    res[key] = out; save(); print(key, round(a, 4), round(p, 4), flush=True)


def run_baseline(key, split, seed, epochs=40):
    if key in res: return
    seed_all(seed)
    idx = np.arange(len(G))
    if split == "scaffold":
        tr_g, va_g = dp.scaffold_split(G, test_size=0.2, random_state=42)
        pos = {id(g): i for i, g in enumerate(G)}
        tri = np.array([pos[id(g)] for g in tr_g]); vai = np.array([pos[id(g)] for g in va_g])
    else:
        tri, vai = train_test_split(idx, test_size=0.2, random_state=42)
    Xt, Yt = torch.tensor(FP[tri]), torch.tensor(Y[tri], dtype=torch.float32)
    Xv = torch.tensor(FP[vai])
    model = FingerprintMLP(FP.shape[1], 512, NL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    tl = TDL(TensorDataset(Xt, Yt), batch_size=64, shuffle=True); curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad(): lg = model(Xv).numpy()
            a, p, _ = compute_mean_metrics(Y[vai], lg); curve.append([ep, float(a), float(p)])
    res[key] = {"auroc": float(a), "auprc": float(p), "curve": curve, "n_val": int(len(vai))}
    save(); print(key, round(a, 4), round(p, 4), flush=True)


for s in SEEDS:
    run_baseline(f"base_random_s{s}", "random", s)
    run_gnn(f"gnn_full_random_s{s}", SmellGNN, "random", s)
for s in SEEDS:
    run_gnn(f"gnn_nopw_random_s{s}", SmellGNN, "random", s, pos_weight=False)
    run_gnn(f"gnn_nobond_random_s{s}", NoBondGNN, "random", s)
    run_gnn(f"gnn_noattn_random_s{s}", NoAttnGNN, "random", s)
for s in SEEDS:
    run_baseline(f"base_scaffold_s{s}", "scaffold", s)
    run_gnn(f"gnn_full_scaffold_s{s}", SmellGNN, "scaffold", s)
print("ALL DONE", flush=True)
