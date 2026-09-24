"""
baseline_model.py
------------------
Baseline model: Morgan fingerprint -> MLP -> multi-label odor descriptor prediction.
This is the benchmark the GNN model needs to beat.

Run directly:
    python baseline_model.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from data_pipeline import prepare_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FingerprintMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_labels, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, x):
        return self.net(x)  # raw logits; use BCEWithLogitsLoss


def compute_mean_metrics(y_true, logits):
    """
    Computes AUROC and AUPRC (Average Precision) per label, skipping labels with
    fewer than 2 classes in the validation split.
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    aucs, aps = [], []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(col, probs[:, i]))
        except ValueError:
            pass
        try:
            aps.append(average_precision_score(col, probs[:, i]))
        except ValueError:
            pass
    mean_auc = np.mean(aucs) if aucs else float("nan")
    mean_ap = np.mean(aps) if aps else float("nan")
    return mean_auc, mean_ap, aucs


def compute_mean_auroc(y_true, logits):
    mean_auc, _, aucs = compute_mean_metrics(y_true, logits)
    return mean_auc, aucs


def train_baseline(fingerprints, labels, hidden_dim=512, epochs=40, batch_size=64, lr=1e-3):
    X_train, X_val, y_train, y_val = train_test_split(
        fingerprints, labels, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = FingerprintMLP(input_dim=fingerprints.shape[1], hidden_dim=hidden_dim,
                            n_labels=labels.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).cpu().numpy()
            mean_auc, mean_ap, _ = compute_mean_metrics(y_val, val_logits)
            print(f"Epoch {epoch:3d} | train loss {total_loss/len(train_loader.dataset):.4f} "
                  f"| val AUROC {mean_auc:.4f} | val AUPRC {mean_ap:.4f}")

    return model, (X_val, y_val)


if __name__ == "__main__":
    print("Loading and featurizing unified data...")
    data = prepare_dataset(min_label_count=20, n_bits=1024)

    print(f"\nTraining baseline fingerprint + MLP model on {len(data['df'])} molecules ({data['labels'].shape[1]} labels)...")
    model, (X_val, y_val) = train_baseline(
        data["fingerprints"], data["labels"], epochs=40
    )

    torch.save(model.state_dict(), "baseline_model.pt")
    print("\nSaved baseline model to baseline_model.pt")
