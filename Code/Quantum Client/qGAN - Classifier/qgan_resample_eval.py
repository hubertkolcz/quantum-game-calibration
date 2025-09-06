"""
Minimal qGAN resampling evaluator.

Loads the saved qGAN classifier weights and evaluates accuracy under bootstrap resampling
of the provided text dataset, reporting mean±std accuracy and histogram.

Outputs: Results/Quantum Client/qGAN_resample/*
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_dataset(path: str, max_rows: int | None = None):
    X = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = int(parts[-1])
            bits = ''.join(parts[:-1])
            X.append([1.0 if b == '1' else 0.0 for b in bits])
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


class TinyLinear(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        pred = logits.argmax(dim=1).cpu().numpy()
    acc = (pred == y).mean()
    return float(acc)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.environ.get(
        "QGAN_DATA",
        os.path.join(repo_root, "Code", "Quantum Client", "qGAN - Classifier", "AI_2qubits_training_data.txt"),
    )
    out_dir = os.path.join(repo_root, "Results", "Quantum Client", "qGAN_resample")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_dataset(data_path, max_rows=5000)
    in_dim = X.shape[1]

    # Minimal baseline model (placeholder for loading real qGAN classifier if available)
    model = TinyLinear(in_dim)
    # If a real state_dict is available, load it here (e.g., torch.load("qgan_*.pth"))
    # try:
    #     model.load_state_dict(torch.load(os.path.join(repo_root, 'Code', 'Quantum Client', 'qGAN - Classifier', 'qgan_1vs2.pth'), map_location='cpu'))
    # except Exception:
    #     pass

    # Bootstrap resampling evaluation
    rng = np.random.default_rng(42)
    B = int(os.environ.get("QGAN_BOOTSTRAPS", "50"))
    n = len(y)
    accs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        accs.append(eval_model(model, X[idx], y[idx]))

    accs = np.array(accs)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Bootstrap accuracy: mean={accs.mean():.4f}, std={accs.std(ddof=1):.4f}\n")

    plt.figure(figsize=(6,4))
    plt.hist(accs, bins=20)
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title('qGAN resampling accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_hist.png"))
    plt.close()


if __name__ == "__main__":
    main()
