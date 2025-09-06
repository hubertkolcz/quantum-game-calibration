"""
Compare trap success for dummy=1 vs dummy=2 across key parameter sweeps for both NV and ions.

Outputs:
- CSVs per parameter per device under Results/Quantum Server/plots/
- Plots comparing the two dummy settings

Env:
- BQC_NUM_TIMES: shots per point (default 200)
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from bqc import load_configurations, setup_parameter_ranges, trap_round, PI_OVER_2


def ensure_out() -> str:
    out = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out, exist_ok=True)
    return out


def sweep_one(stack, cfg, label: str, sweep: np.ndarray, shots: int) -> pd.DataFrame:
    rows = []
    old = getattr(stack, label)
    for i, v in enumerate(sweep):
        setattr(stack, label, v)
        s1 = trap_round(cfg, num_times=shots, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=1)
        s2 = trap_round(cfg, num_times=shots, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=2)
        rows.append({"i": i, label: float(v), "trap_succ_dummy1": s1, "trap_succ_dummy2": s2})
    setattr(stack, label, old)
    return pd.DataFrame(rows)


def plot_compare(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, xlabel: str, title: str, outpath: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label="dummy=1")
    plt.plot(x, y2, label="dummy=2")
    plt.xlabel(xlabel)
    plt.ylabel("Trap success")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    set_qstate_formalism(QFormalism.DM)
    shots = int(os.environ.get("BQC_NUM_TIMES", "200"))
    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    params = setup_parameter_ranges()
    out_dir = ensure_out()

    ts = time.strftime("%Y%m%d-%H%M%S")

    # NV
    for label, (_base, sweep) in params["color_centers"].items():
        df = sweep_one(nv_stack, cfg_nv, label, sweep, shots)
        csv_path = os.path.join(out_dir, f"{ts}_nv_{label}_trap_dummy_compare.csv")
        df.to_csv(csv_path, index=False)
        plot_compare(df[label].to_numpy(), df["trap_succ_dummy1"].to_numpy(), df["trap_succ_dummy2"].to_numpy(), label, f"NV trap success: dummy=1 vs 2 across {label}", os.path.join(out_dir, f"nv_{label}_trap_dummy_compare.png"))

    # Ions
    for label, (_base, sweep) in params["trapped_ions"].items():
        df = sweep_one(ions_stack, cfg_ions, label, sweep, shots)
        csv_path = os.path.join(out_dir, f"{ts}_ions_{label}_trap_dummy_compare.csv")
        df.to_csv(csv_path, index=False)
        plot_compare(df[label].to_numpy(), df["trap_succ_dummy1"].to_numpy(), df["trap_succ_dummy2"].to_numpy(), label, f"Ions trap success: dummy=1 vs 2 across {label}", os.path.join(out_dir, f"ions_{label}_trap_dummy_compare.png"))


if __name__ == "__main__":
    main()
