"""
Sweep over (alpha, beta) rotation angles for the computation round and record P(m2=0).

Outputs:
- CSV with alpha,beta,frac0 for selected device
- Heatmap PNG under Results/Quantum Server/plots/

Env:
- BQC_NUM_TIMES: shots per point (default 200)
- BQC_DEVICE: one of {nv, ions} (default nv)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from bqc import load_configurations, computation_round


def ensure_out() -> str:
    out = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out, exist_ok=True)
    return out


def main():
    set_qstate_formalism(QFormalism.DM)
    shots = int(os.environ.get("BQC_NUM_TIMES", "200"))
    device = os.environ.get("BQC_DEVICE", "nv").lower()
    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    if device not in {"nv", "ions"}:
        device = "nv"
    cfg = cfg_nv if device == "nv" else cfg_ions

    # grid over [0, pi] for alpha,beta (periodicity allows this range)
    n = int(os.environ.get("BQC_ANGLE_STEPS", "13"))
    alphas = np.linspace(0.0, np.pi, n)
    betas = np.linspace(0.0, np.pi, n)

    rows = []
    for a in alphas:
        for b in betas:
            frac0 = computation_round(cfg, num_times=shots, alpha=float(a), beta=float(b), theta1=0.0, theta2=0.0)
            rows.append({"alpha": float(a), "beta": float(b), "frac0": frac0})

    df = pd.DataFrame(rows)
    out_dir = ensure_out()
    csv_path = os.path.join(out_dir, f"angle_sweep_{device}_{shots}.csv")
    df.to_csv(csv_path, index=False)

    # heatmap
    pivot = df.pivot(index="alpha", columns="beta", values="frac0")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, origin="lower", extent=[betas.min(), betas.max(), alphas.min(), alphas.max()], aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label="P(m2=0)")
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title(f"Computation round: P(m2=0) over (alpha,beta) [{device}, shots={shots}]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"angle_sweep_{device}_{shots}.png"))
    plt.close()


if __name__ == "__main__":
    main()
