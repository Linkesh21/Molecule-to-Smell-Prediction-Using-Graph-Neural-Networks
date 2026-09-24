"""
generative_model.py
-------------------
Implements a SMILES-based sequence-to-sequence Variational Autoencoder (VAE)
to learn a continuous latent space of molecules.
It also includes a search routine to find a molecule in the latent space
whose predicted odor profile (using SmellGNN) matches a target profile.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import os

from data_pipeline import prepare_dataset, smiles_to_graph
from gnn_model import SmellGNN

# Define special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

class Vocab:
    def __init__(self, smiles_list):
        self.char2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.idx2char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN}
        
        # Collect unique characters
        chars = set()
        for s in smiles_list:
            chars.update(list(s))
            
        for char in sorted(list(chars)):
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            
        self.vocab_size = len(self.char2idx)
        
    def encode(self, smiles):
        return [self.char2idx[SOS_TOKEN]] + [self.char2idx[c] for c in smiles] + [self.char2idx[EOS_TOKEN]]
    
    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx == self.char2idx[EOS_TOKEN]:
                break
            if idx not in [self.char2idx[SOS_TOKEN], self.char2idx[PAD_TOKEN]]:
                chars.append(self.idx2char[idx])
        return "".join(chars)

class SmilesDataset(Dataset):
    def __init__(self, smiles_list, vocab, max_len=120):
        self.vocab = vocab
        self.max_len = max_len
        
        self.encoded_smiles = []
        for s in smiles_list:
            encoded = vocab.encode(s)
            if len(encoded) <= max_len:
                # Pad sequence
                padded = encoded + [vocab.char2idx[PAD_TOKEN]] * (max_len - len(encoded))
                self.encoded_smiles.append(padded)
                
        self.encoded_smiles = torch.tensor(self.encoded_smiles, dtype=torch.long)
        
    def __len__(self):
        return len(self.encoded_smiles)
    
    def __getitem__(self, idx):
        return self.encoded_smiles[idx]

class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_rnn = nn.GRU(emb_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
    def encode(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder_rnn(embedded)
        # hidden is [2, batch, hidden_dim] (for bidirectional)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1) # [batch, hidden_dim * 2]
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, x):
        batch_size = z.size(0)
        seq_len = x.size(1)
        
        embedded = self.embedding(x) # [batch, seq_len, emb_dim]
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1) # [batch, seq_len, latent_dim]
        
        rnn_input = torch.cat((embedded, z_expanded), dim=-1) # [batch, seq_len, emb_dim + latent_dim]
        
        hidden = self.latent_to_hidden(z).unsqueeze(0) # [1, batch, hidden_dim]
        
        output, _ = self.decoder_rnn(rnn_input, hidden)
        logits = self.fc_out(output) # [batch, seq_len, vocab_size]
        
        return logits
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Use teacher forcing (shift inputs for decoder: remove last char, add SOS if necessary)
        # But here x already has <sos> at the start and <eos> at the end.
        # Decoder input: x[:, :-1], Target: x[:, 1:]
        decoder_input = x[:, :-1]
        logits = self.decode(z, decoder_input)
        return logits, mu, logvar

    def sample(self, z, max_len, sos_idx):
        batch_size = z.size(0)
        device = z.device
        
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        current_input = torch.tensor([[sos_idx]] * batch_size, device=device) # [batch, 1]
        
        outputs = []
        for _ in range(max_len):
            embedded = self.embedding(current_input) # [batch, 1, emb_dim]
            z_expanded = z.unsqueeze(1) # [batch, 1, latent_dim]
            rnn_input = torch.cat((embedded, z_expanded), dim=-1) # [batch, 1, emb_dim + latent_dim]
            
            out, hidden = self.decoder_rnn(rnn_input, hidden)
            logits = self.fc_out(out) # [batch, 1, vocab_size]
            
            # Greedy decoding
            current_input = logits.argmax(dim=-1) # [batch, 1]
            outputs.append(current_input)
            
        return torch.cat(outputs, dim=1) # [batch, max_len]

def vae_loss_fn(recon_logits, target, mu, logvar, kl_weight=0.1):
    # recon_logits: [batch, seq_len, vocab_size]
    # target: [batch, seq_len]
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, recon_logits.size(-1)), 
        target.reshape(-1), 
        ignore_index=0 # PAD_TOKEN
    )
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def train_vae():
    print("Preparing dataset for VAE...")
    data = prepare_dataset(min_label_count=20)
    df = data["df"]
    
    # Extract unique valid SMILES up to length 100
    smiles_list = df["SMILES"].tolist()
    
    vocab = Vocab(smiles_list)
    print(f"Vocab size: {vocab.vocab_size}")
    
    dataset = SmilesDataset(smiles_list, vocab, max_len=100)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmilesVAE(vocab.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 15
    print(f"Training VAE for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            target = batch[:, 1:] # Target shifted by 1
            
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            loss, recon, kl = vae_loss_fn(logits, target, mu, logvar, kl_weight=0.01)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} "
              f"| Recon: {total_recon/len(dataloader):.4f} | KL: {total_kl/len(dataloader):.4f}")
        
    torch.save({"model_state": model.state_dict(), "vocab": vocab}, "smiles_vae.pt")
    print("Saved VAE to smiles_vae.pt")
    return model, vocab

def evaluate_generation(model, vocab, n_samples=100):
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        sampled_indices = model.sample(z, max_len=100, sos_idx=vocab.char2idx[SOS_TOKEN])
    
    valid_smiles = []
    for i in range(n_samples):
        s = vocab.decode(sampled_indices[i].tolist())
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_smiles.append(s)
            
    print(f"Generation Validity: {len(valid_smiles)}/{n_samples} ({len(valid_smiles)/n_samples*100:.1f}%)")
    return valid_smiles

def latent_search(target_profile, gnn_model, vae_model, vocab, label_classes, n_samples=5000):
    """
    Random search in latent space: sample N points, decode, evaluate validity, 
    score with GNN, and return the best ones.
    target_profile: dict of {descriptor: weight} (e.g., {'woody': 1.0, 'fruity': 0.5})
    """
    device = next(vae_model.parameters()).device
    vae_model.eval()
    gnn_model.eval()
    
    print(f"Sampling {n_samples} latent vectors for search...")
    with torch.no_grad():
        z = torch.randn(n_samples, vae_model.latent_dim).to(device)
        sampled_indices = vae_model.sample(z, max_len=100, sos_idx=vocab.char2idx[SOS_TOKEN])
        
    # Decode and filter valid
    valid_candidates = []
    valid_z = []
    
    for i in range(n_samples):
        s = vocab.decode(sampled_indices[i].tolist())
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_candidates.append(s)
            valid_z.append(z[i])
            
    if not valid_candidates:
        print("No valid molecules generated.")
        return []
        
    print(f"Found {len(valid_candidates)} valid candidates. Scoring with GNN...")
    
    # Score with GNN
    target_indices = {np.where(label_classes == k)[0][0]: v for k, v in target_profile.items() if k in label_classes}
    
    scores = []
    graphs = []
    valid_smiles = []
    
    for smi in valid_candidates:
        graph = smiles_to_graph(smi)
        if graph is not None:
            graphs.append(graph)
            valid_smiles.append(smi)
            
    if not graphs:
        return []
        
    from torch_geometric.data import Batch
    
    chunk_size = 500
    preds_list = []
    
    with torch.no_grad():
        for i in range(0, len(graphs), chunk_size):
            chunk = graphs[i:i+chunk_size]
            batch = Batch.from_data_list(chunk).to(device)
            preds = torch.sigmoid(gnn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)).cpu().numpy()
            # If batch size is 1, ensure it's 2D
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)
            preds_list.append(preds)
            
    all_preds = np.vstack(preds_list)
    
    scores = np.zeros(len(valid_smiles))
    for idx, weight in target_indices.items():
        scores += all_preds[:, idx] * weight
        
    # Sort and return top 10
    top_indices = np.argsort(scores)[::-1][:10]
    
    results = []
    for idx in top_indices:
        results.append((valid_smiles[idx], float(scores[idx])))
        
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_vae()
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        if not os.path.exists("smiles_vae.pt"):
            print("Please run `python generative_model.py train` first.")
            sys.exit(1)
            
        print("Loading VAE...")
        checkpoint = torch.load("smiles_vae.pt", weights_only=False)
        vocab = checkpoint["vocab"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vae_model = SmilesVAE(vocab.vocab_size).to(device)
        vae_model.load_state_dict(checkpoint["model_state"])
        
        evaluate_generation(vae_model, vocab, n_samples=100)
        
        print("Loading GNN...")
        data = prepare_dataset(min_label_count=20)
        node_dim = data["node_dim"]
        edge_dim = data["edge_dim"]
        label_classes = data["label_binarizer"].classes_
        
        gnn_model = SmellGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128, n_labels=len(label_classes)).to(device)
        gnn_model.load_state_dict(torch.load("gnn_model.pt", map_location=device, weights_only=True))
        
        target = {"woody": 1.0, "fruity": 0.5}
        print(f"Searching for target profile: {target}")
        
        best_candidates = latent_search(target, gnn_model, vae_model, vocab, label_classes, n_samples=1000)
        
        print("\n--- Top Candidates ---")
        for i, (smi, score) in enumerate(best_candidates):
            print(f"{i+1}. SMILES: {smi} | Score: {score:.4f}")
            
    else:
        print("Usage: python generative_model.py [train|search]")
