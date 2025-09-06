"""
Trap experiments for BQC NetSquid setup.

This script reuses bqc.py configuration utilities to:
- run trap rounds (dummy state checks) to estimate trap success probability
- run computation rounds to estimate output bit distribution
- sweep key parameters (mirroring bqc.py sweeps) and save results

Outputs:
- CSV logs per parameter under current working dir
- Plots under Results/Quantum Server/plots/
"""

from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import netsquid as ns
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
from matplotlib import pyplot as plt

from bqc import (
    load_configurations,
    setup_parameter_ranges,
    ClientProgram,
    ServerProgram,
    computation_round,
    trap_round,
    PI_OVER_2,
)


def ensure_results_dir() -> str:
    out_dir = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_trap_sweep(stack, cfg, label: str, base_val: float, sweep: np.ndarray, num_times: int) -> pd.DataFrame:
    old = getattr(stack, label)
    rows = []
    for i, v in enumerate(sweep):
        setattr(stack, label, v)
        succ = trap_round(cfg, num_times=num_times, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0, theta2=0, dummy=1)
        rows.append({"i": i, label: v, "trap_succ_rate": succ})
    setattr(stack, label, old)
    return pd.DataFrame(rows)


def run_comp_sweep(stack, cfg, label: str, base_val: float, sweep: np.ndarray, num_times: int) -> pd.DataFrame:
    old = getattr(stack, label)
    rows = []
    for i, v in enumerate(sweep):
        setattr(stack, label, v)
        frac0 = computation_round(cfg, num_times=num_times, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0, theta2=0)
        rows.append({"i": i, label: v, "frac0": frac0, "frac1": 1 - frac0})
    setattr(stack, label, old)
    return pd.DataFrame(rows)


def plot_pair(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, xlabel: str, y1label: str, y2label: str, title: str, outpath: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=y1label)
    plt.plot(x, y2, label=y2label)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.savefig(outpath)
    plt.close()


def main():
    set_qstate_formalism(QFormalism.DM)
    num_times = int(os.environ.get("BQC_NUM_TIMES", "100"))  # faster default for explorations

    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    param_sets = setup_parameter_ranges()

    out_dir = ensure_results_dir()

    # NV experiments
    for label, (base, sweep) in param_sets["color_centers"].items():
        print(f"NV sweep: {label}")
        trap_df = run_trap_sweep(nv_stack, cfg_nv, label, base, sweep, num_times)
        comp_df = run_comp_sweep(nv_stack, cfg_nv, label, base, sweep, num_times)
        merged = trap_df.merge(comp_df, on=["i", label])
        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(out_dir, f"{ts}_nv_{label}_trap_vs_frac.csv")
        merged.to_csv(csv_path, index=False)
        plot_pair(
            merged[label].to_numpy(),
            merged["trap_succ_rate"].to_numpy(),
            merged["frac0"].to_numpy(),
            xlabel=label,
            y1label="trap success",
            y2label="P(m2=0)",
            title=f"NV: trap success vs output fraction across {label}",
            outpath=os.path.join(out_dir, f"nv_{label}_trap_vs_frac.png"),
        )

    # Trapped ions experiments
    for label, (base, sweep) in param_sets["trapped_ions"].items():
        print(f"Ions sweep: {label}")
        trap_df = run_trap_sweep(ions_stack, cfg_ions, label, base, sweep, num_times)
        comp_df = run_comp_sweep(ions_stack, cfg_ions, label, base, sweep, num_times)
        merged = trap_df.merge(comp_df, on=["i", label])
        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(out_dir, f"{ts}_ions_{label}_trap_vs_frac.csv")
        merged.to_csv(csv_path, index=False)
        plot_pair(
            merged[label].to_numpy(),
            merged["trap_succ_rate"].to_numpy(),
            merged["frac0"].to_numpy(),
            xlabel=label,
            y1label="trap success",
            y2label="P(m2=0)",
            title=f"Ions: trap success vs output fraction across {label}",
            outpath=os.path.join(out_dir, f"ions_{label}_trap_vs_frac.png"),
        )


if __name__ == "__main__":
    main()
