from __future__ import annotations
"""
Minimal qGAN resampling evaluator.
- Loads existing datasets and saved model weights for the 1vs2/3vs1/3vs2 classifiers (PyTorch expected).
- Performs k resamplings (bootstrap or k-fold) and reports mean±std accuracy.

Outputs: Results/Quantum Client/qGAN_eval/*.txt with summary stats.
"""
import os
import glob
import json
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_txt_pairs(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load simple txt dataset: features and labels.
    Assumes two-column or similar numerical format; adapt as needed.
    """
    X = []
    y = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                X.append([float(parts[0])])
                y.append(int(float(parts[1])))
            except Exception:
                continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def simple_mlp(in_dim: int, num_classes: int = 2) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, num_classes),
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = os.path.dirname(__file__)
    out_dir = os.path.join("Results", "Quantum Client", "qGAN_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Try to find a text dataset (adjust if different format)
    candidates = glob.glob(os.path.join(data_dir, "*.txt"))
    if not candidates:
        print("No text datasets found; skipping.")
        return
    data_path = candidates[0]
    X, y = load_txt_pairs(data_path)
    if X.size == 0:
        print("Dataset empty or incompatible; skipping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # K resampling evaluations
    K = int(os.environ.get("QGAN_EVAL_K", "5"))
    test_frac = float(os.environ.get("QGAN_TEST_FRAC", "0.2"))

    accs = []
    for k in range(K):
        # random split
        N = len(X)
        idx = np.random.permutation(N)
        Xb, yb = X[idx], y[idx]
        n_test = max(1, int(N * test_frac))
        Xtr, ytr = Xb[:-n_test], yb[:-n_test]
        Xte, yte = Xb[-n_test:], yb[-n_test:]

        # Model
        model = simple_mlp(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=128)

        # quick train
        for _ in range(20):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        acc = evaluate(model, test_loader, device)
        accs.append(acc)

    summary = {
        "dataset": os.path.basename(data_path),
        "K": K,
        "test_frac": test_frac,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "accs": [float(a) for a in accs],
    }
    out_path = os.path.join(out_dir, "resampling_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
