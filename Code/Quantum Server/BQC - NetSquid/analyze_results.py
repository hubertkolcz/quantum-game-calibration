"""
Aggregate existing BQC result CSVs, fix throughput units, and add confidence intervals.

Inputs:
  Results/Quantum Server/results/*.csv  (emitted by bqc.py)

For each CSV, we compute:
  - exec_ms_per_run = exec_per_sec (as stored by bqc.py; actually ms per run)
  - execs_per_sec = 1000.0 / exec_ms_per_run
  - succes_per_sec_corrected = execs_per_sec * succ_rate
  - Wilson 95% CI for succ_rate given num_times parsed from filename

Outputs:
  - Results/Quantum Server/summary/aggregated_metrics.csv
  - Results/Quantum Server/summary/throughput_summary.txt (quick ranges)
  - Optional: plots per parameter showing corrected successes/sec vs parameter
"""
from __future__ import annotations
import os
import re
import math
from typing import Tuple
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


RESULTS_DIR = os.path.join("Results", "Quantum Server", "results")
OUT_DIR = os.path.join("Results", "Quantum Server", "summary")


def parse_num_times(name: str) -> int:
    # filenames end with ..._<num_times>.csv
    m = re.search(r"_(\d+)\.csv$", name)
    return int(m.group(1)) if m else 0


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return (max(0.0, low), min(1.0, high))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    if not os.path.isdir(RESULTS_DIR):
        raise SystemExit(f"Missing directory: {RESULTS_DIR}")

    csvs = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    for fname in sorted(csvs):
        num_times = parse_num_times(fname)
        path = os.path.join(RESULTS_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue
        # Column holding the swept parameter (last column)
        param_col = df.columns[-1]
        # Correct throughput units
        exec_ms_per_run = df["exec_per_sec"].astype(float)
        execs_per_sec = 1000.0 / exec_ms_per_run.replace(0.0, np.nan)
        succ_rate = df["succ_rate"].astype(float)
        succ_per_sec_fixed = execs_per_sec * succ_rate
        # Compute Wilson CI for succ_rate
        if num_times <= 0:
            # try infer from cardinality if unknown
            num_times = 1000
        k = (succ_rate * num_times).round().astype(int)
        ci = k.apply(lambda x: wilson_ci(int(x), num_times))
        low = [c[0] for c in ci]
        high = [c[1] for c in ci]

        out_df = pd.DataFrame({
            "source_file": fname,
            "param": param_col,
            param_col: df[param_col].values,
            "num_times": num_times,
            "succ_rate": succ_rate,
            "succ_low95": low,
            "succ_high95": high,
            "exec_ms_per_run": exec_ms_per_run,
            "execs_per_sec": execs_per_sec,
            "succ_per_sec_corrected": succ_per_sec_fixed,
        })
        rows.append(out_df)

    # Plot corrected successes/sec vs parameter
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(df[param_col].values, succ_per_sec_fixed, marker='o')
            plt.xlabel(param_col)
            plt.ylabel('successes per second (corrected)')
            plt.title(fname.replace('.csv',''))
            plt.tight_layout()
            base = os.path.splitext(fname)[0]
            plt.savefig(os.path.join(OUT_DIR, f"{base}_succ_per_sec_corrected.png"))
            plt.close()
        except Exception as e:
            print(f"Plot failed for {fname}: {e}")

    if not rows:
        print("No CSVs aggregated.")
        return
    agg = pd.concat(rows, ignore_index=True)
    agg_path = os.path.join(OUT_DIR, "aggregated_metrics.csv")
    agg.to_csv(agg_path, index=False)

    # Quick ranges summary
    lines = ["Throughput summary (corrected):"]
    for src, g in agg.groupby("source_file"):
        v = g["succ_per_sec_corrected"].replace([np.inf, -np.inf], np.nan).dropna()
        if v.empty:
            continue
        lines.append(f"  {src}: min={v.min():.3f}, median={v.median():.3f}, max={v.max():.3f}")
    with open(os.path.join(OUT_DIR, "throughput_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {agg_path}")

    # Pareto envelopes: group by device inferred from filename prefix
    def device_of(s: str) -> str:
        s = s.lower()
        if s.startswith("nvqdeviceconfig") or "nv" in s:
            return "nv"
        if s.startswith("genericqdeviceconfig") or "generic" in s or "ions" in s:
            return "ions"
        return "other"

    plt.figure(figsize=(7,5))
    for dev, g in agg.groupby(agg["source_file"].apply(device_of)):
        # Build Pareto: for each succ_rate, take max successes/sec
        gg = g.copy()
        gg = gg.sort_values(["succ_rate", "succ_per_sec_corrected"], ascending=[True, False])
        pareto = gg.groupby("succ_rate")["succ_per_sec_corrected"].max().reset_index()
        plt.plot(pareto["succ_rate"], pareto["succ_per_sec_corrected"], label=dev)
    plt.xlabel("success probability")
    plt.ylabel("successes per second (corrected)")
    plt.title("Pareto envelope by device")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pareto_by_device.png"))
    plt.close()

    # Pareto envelopes: corrected successes/sec (maximize) vs succ_rate (maximize)
    def pareto_front(points: np.ndarray) -> np.ndarray:
        # points: Nx2 array (x=succ_rate, y=succ_per_sec_corrected)
        idx = np.argsort(points[:, 0])  # sort by x asc
        best = -np.inf
        keep = []
        for i in idx:
            y = points[i, 1]
            if y > best:
                best = y
                keep.append(i)
        return np.array(keep, dtype=int)

    # Per-file pareto plots
    for src, g in agg.groupby("source_file"):
        pts = g[["succ_rate", "succ_per_sec_corrected"]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if pts.shape[0] < 3:
            continue
        keep = pareto_front(pts)
        plt.figure(figsize=(7,5))
        plt.scatter(pts[:,0], pts[:,1], s=15, alpha=0.5, label="points")
        plt.plot(pts[keep,0], pts[keep,1], color='C1', label="Pareto envelope")
        plt.xlabel("success probability")
        plt.ylabel("successes per second (corrected)")
        plt.title(f"Pareto: {src}")
        plt.legend()
        plt.tight_layout()
        base = os.path.splitext(src)[0]
        plt.savefig(os.path.join(OUT_DIR, f"pareto_{base}.png"))
        plt.close()



if __name__ == "__main__":
    main()
