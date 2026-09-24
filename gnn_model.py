"""
gnn_model.py
------------
Graph Neural Network (GINE-based) that predicts odor descriptors directly
from molecular graph structure, incorporating both atom-level and bond-level
features (including chirality and stereochemistry).

Upgraded features:
  - GINEConv: fuses rich bond features (type, stereochemistry, conjugation, rings)
  - AttentionalAggregation + global mean pooling: captures both salient odorophores and overall scaffold
  - Class-imbalanced BCEWithLogitsLoss (pos_weight): balances sparse odor descriptors
  - Dual metrics: tracks both AUROC and mean AUPRC (Average Precision)
  - Supports both Random and Bemis-Murcko Scaffold splitting

Run directly:
    python gnn_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from data_pipeline import prepare_dataset, scaffold_split
from baseline_model import compute_mean_metrics, compute_mean_auroc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmellGNN(nn.Module):
    """
    Stack of GINE layers over the molecular graph (incorporating atom and bond
    features with stereochemistry), pooled via AttentionalAggregation and
    global mean pool, then an MLP head predicts multi-label odor descriptor logits.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, n_labels, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout = dropout

        # Attention pooling to attend to salient functional groups / odorophores
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.att_pool = AttentionalAggregation(gate_nn=gate_nn)

        # Classifier combines attention-pooled and mean-pooled representations
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        att_pooled = self.att_pool(x, batch)
        mean_pooled = global_mean_pool(x, batch)
        pooled = torch.cat([att_pooled, mean_pooled], dim=1)
        return self.classifier(pooled)  # raw logits

    def embed(self, x, edge_index, edge_attr, batch):
        """Returns the pooled molecule embedding (used for VAE / generation step)."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        att_pooled = self.att_pool(x, batch)
        mean_pooled = global_mean_pool(x, batch)
        return torch.cat([att_pooled, mean_pooled], dim=1)


def train_gnn(
    graphs,
    hidden_dim=128,
    num_layers=3,
    epochs=60,
    batch_size=64,
    lr=1e-3,
    split_method="random",
    use_pos_weight=True,
):
    if split_method == "scaffold":
        train_graphs, val_graphs = scaffold_split(graphs, test_size=0.2, random_state=42)
    else:
        train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    n_labels = graphs[0].y.shape[1]

    model = SmellGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        n_labels=n_labels,
        num_layers=num_layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    if use_pos_weight:
        y_train_all = torch.cat([g.y for g in train_graphs], dim=0)
        pos_counts = y_train_all.sum(dim=0)
        neg_counts = len(train_graphs) - pos_counts
        pos_weights = torch.clamp(neg_counts / (pos_counts + 1e-5), min=1.0, max=15.0).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_auc, best_ap = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 5 == 0 or epoch == epochs:
            mean_auc, mean_ap = evaluate_gnn(model, val_loader)
            if mean_auc > best_auc:
                best_auc, best_ap = mean_auc, mean_ap
            print(
                f"Epoch {epoch:3d} | train loss {total_loss/len(train_loader.dataset):.4f} "
                f"| val AUROC {mean_auc:.4f} | val AUPRC {mean_ap:.4f}"
            )

    return model, val_graphs


def evaluate_gnn(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_auc, mean_ap, _ = compute_mean_metrics(all_labels, all_logits)
    return mean_auc, mean_ap


if __name__ == "__main__":
    print("Loading and featurizing unified data (Leffingwell + GoodScents + Arctander)...")
    data = prepare_dataset(min_label_count=20)

    print(f"\nTraining upgraded SmellGNN on {len(data['graphs'])} molecular graphs ({data['labels'].shape[1]} odor labels)...")
    model, val_graphs = train_gnn(data["graphs"], epochs=60, split_method="random")

    torch.save(model.state_dict(), "gnn_model.pt")
    print("\nSaved upgraded GNN model to gnn_model.pt")
