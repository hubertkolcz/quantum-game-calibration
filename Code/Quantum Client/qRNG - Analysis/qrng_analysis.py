"""
QRNG analysis utilities: bias, min-entropy, and cross-talk risk from IBM calibration data.

Inputs:
- CSV calibration file exported from IBM system (example in Others/ibm_sherbrooke_calibrations_*.csv)

Outputs (written under Results/Quantum Client/qRNG):
- qrng_metrics.csv: per-qubit metrics (bias, min-entropy estimate, readout error, neighbors, cross-talk score)
- Plots: histograms and scatterplots for T1/T2, min-entropy, bias heatmap, and optional coupling graph (if networkx available)

Notes:
- This script does not require device access; it analyzes static calibrations.
- Cross-talk risk is a heuristic using two-qubit ECR error and readout error disparity across neighbors (see comments below).
- Inspired by common practices and open-source qRNG repos (e.g., dorahacksglobal/quantum-randomness-generator) but reimplemented here.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Parsing helpers
# --------------------------

PAIR_RE = re.compile(r"(\d+)_(\d+):([0-9.eE+-]+)")


def parse_ecr_pairs(ecr_field: str) -> List[Tuple[int, int, float]]:
    """Parse an ECR error field like "7_6:0.00536;7_8:0.00727" into (a, b, error) tuples.
    Returns empty list if no pairs available.
    """
    if not isinstance(ecr_field, str) or not ecr_field:
        return []
    pairs = []
    for piece in ecr_field.split(";"):
        m = PAIR_RE.match(piece.strip())
        if m:
            a, b, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
            pairs.append((a, b, val))
    return pairs


def _canon(s: str) -> str:
    return " ".join(s.strip().lower().replace("\u00a0", " ").split())


def _colmap(cols: List[str]) -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    for c in cols:
        cmap[_canon(c)] = c
    return cmap


def _resolve(cmap: Dict[str, str], candidates: List[str]) -> str | None:
    for cand in candidates:
        key = _canon(cand)
        if key in cmap:
            return cmap[key]
    # try substring search
    for key, orig in cmap.items():
        for cand in candidates:
            if _canon(cand) in key:
                return orig
    return None


def load_calibrations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace("\u00a0", " ") for c in df.columns]
    cmap = _colmap(list(df.columns))
    q_col = _resolve(cmap, ["Qubit"])
    if q_col:
        df[q_col] = df[q_col].astype(int)
        df.rename(columns={q_col: "Qubit"}, inplace=True)
    return df


# --------------------------
# Metrics
# --------------------------

@dataclass
class QRNGMetrics:
    qubit: int
    readout_err: float
    p_meas0_prep1: float
    p_meas1_prep0: float
    bias_est: float
    min_entropy_est: float
    neighbors: List[int]
    ecr_avg_error_to_neighbors: float
    readout_err_neighbor_dispersion: float
    crosstalk_risk: float


def estimate_bias_and_entropy(e0: float, e1: float) -> Tuple[float, float]:
    """Estimate bias and min-entropy for a QRNG measuring a nominally unbiased state.

    Given asymmetric readout assignment errors:
    - e0 = Prob(meas1 | prep0)
    - e1 = Prob(meas0 | prep1)

    If the underlying bit is truly unbiased (p(0)=p(1)=0.5), the observed distribution is:
        P(0) = 0.5 * (1 - e0) + 0.5 * e1
        P(1) = 1 - P(0)
    Bias = |P(1) - P(0)| = |(e0 - e1)| / 2
    Min-entropy H_inf = -log2(max(P(0), P(1)))
    """
    p0 = 0.5 * ((1.0 - e0) + e1)
    p1 = 1.0 - p0
    bias = abs(p1 - p0)
    pmax = max(p0, p1)
    min_entropy = float(-np.log2(max(pmax, 1e-12)))
    return bias, min_entropy


def build_coupling_graph(df: pd.DataFrame) -> Dict[int, List[Tuple[int, float]]]:
    """Build an undirected neighbor map from ECR error field.
    Returns: {qubit: [(neighbor, ecr_error), ...], ...}
    """
    graph: Dict[int, List[Tuple[int, float]]] = {}
    cmap = _colmap(list(df.columns))
    ecr_col = _resolve(cmap, ["ECR error"]) or _resolve(cmap, ["ECR errors", "ECR"]) or "ECR error "
    for _, row in df.iterrows():
        q = int(row["Qubit"]) if "Qubit" in row else None
        if q is None:
            continue
        ecr_field = str(row.get(ecr_col, "")).strip()
        pairs = parse_ecr_pairs(ecr_field)
        for a, b, val in pairs:
            # Add both directions to make it undirected
            graph.setdefault(a, []).append((b, val))
            graph.setdefault(b, []).append((a, val))
    # Deduplicate neighbors by keeping lowest error when duplicated
    dedup_graph: Dict[int, List[Tuple[int, float]]] = {}
    for q, nbrs in graph.items():
        best: Dict[int, float] = {}
        for n, e in nbrs:
            best[n] = min(e, best.get(n, e))
        dedup_graph[q] = sorted(best.items())
    return dedup_graph


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    graph = build_coupling_graph(df)
    cmap = _colmap(list(df.columns))
    readout_col = _resolve(cmap, ["Readout assignment error"]) or "Readout assignment error "
    p01_col = _resolve(cmap, ["Prob meas0 prep1"]) or "Prob meas0 prep1 "
    p10_col = _resolve(cmap, ["Prob meas1 prep0"]) or "Prob meas1 prep0 "

    rows: List[QRNGMetrics] = []
    for _, row in df.iterrows():
        q = int(row["Qubit"]) if "Qubit" in row else None
        if q is None:
            continue

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        e_readout = _to_float(row.get(readout_col, np.nan))
        e0 = _to_float(row.get(p10_col, np.nan))  # P(meas1|prep0)
        e1 = _to_float(row.get(p01_col, np.nan))  # P(meas0|prep1)

        bias, hmin = estimate_bias_and_entropy(e0 if not np.isnan(e0) else 0.0, e1 if not np.isnan(e1) else 0.0)

        nbrs = graph.get(q, [])
        nbr_ids = [n for n, _ in nbrs]
        nbr_ecr = np.array([e for _, e in nbrs]) if nbrs else np.array([])
        ecr_avg = float(np.mean(nbr_ecr)) if nbr_ecr.size else 0.0

        # Dispersion of readout error across neighbors (difference indicates potential cross-impact when measured together)
        if nbr_ids:
            neighbor_readout = []
            for n in nbr_ids:
                n_row = df[df["Qubit"] == n]
                if not n_row.empty:
                    neighbor_readout.append(_to_float(n_row.iloc[0].get(readout_col, np.nan)))
            neighbor_readout = np.array([x for x in neighbor_readout if not np.isnan(x)])
            dispersion = float(np.std(np.append(neighbor_readout, e_readout))) if neighbor_readout.size else 0.0
        else:
            dispersion = 0.0

        # Crosstalk risk heuristic: scale of neighbor ECR error and readout dispersion, weighted by bias
        crosstalk_risk = float((ecr_avg + dispersion) * (0.5 + bias))

        rows.append(
            QRNGMetrics(
                qubit=q,
                readout_err=e_readout,
                p_meas0_prep1=e1,
                p_meas1_prep0=e0,
                bias_est=bias,
                min_entropy_est=hmin,
                neighbors=nbr_ids,
                ecr_avg_error_to_neighbors=ecr_avg,
                readout_err_neighbor_dispersion=dispersion,
                crosstalk_risk=crosstalk_risk,
            )
        )

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values("qubit")
    return out


def ensure_output_dir() -> str:
    out_dir = os.path.join("Results", "Quantum Client", "qRNG")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_distributions(calib: pd.DataFrame, metrics: pd.DataFrame, out_dir: str) -> None:
    # T1/T2 histograms and scatter
    if "T1 (us)" in calib.columns and "T2 (us)" in calib.columns:
        plt.figure(figsize=(10, 5))
        plt.hist(calib["T1 (us)"].astype(float), bins=30, alpha=0.7, label="T1 (us)")
        plt.hist(calib["T2 (us)"].astype(float), bins=30, alpha=0.7, label="T2 (us)")
        plt.legend()
        plt.title("T1/T2 distributions")
        plt.savefig(os.path.join(out_dir, "t1_t2_hist.png"))
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.scatter(calib["T1 (us)"].astype(float), calib["T2 (us)"].astype(float), s=10)
        plt.xlabel("T1 (us)")
        plt.ylabel("T2 (us)")
        plt.title("T1 vs T2")
        plt.savefig(os.path.join(out_dir, "t1_vs_t2.png"))
        plt.close()

    # Min-entropy and bias
    plt.figure(figsize=(10, 5))
    plt.hist(metrics["min_entropy_est"], bins=30)
    plt.xlabel("Estimated min-entropy per bit")
    plt.title("QRNG min-entropy estimate")
    plt.savefig(os.path.join(out_dir, "min_entropy_hist.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics["qubit"], metrics["bias_est"])  # index as x
    plt.xlabel("Qubit index")
    plt.ylabel("Bias estimate")
    plt.title("QRNG bias estimate per qubit")
    plt.savefig(os.path.join(out_dir, "bias_per_qubit.png"))
    plt.close()

    # Crosstalk risk
    plt.figure(figsize=(10, 5))
    plt.bar(metrics["qubit"], metrics["crosstalk_risk"])  # index as x
    plt.xlabel("Qubit index")
    plt.ylabel("Crosstalk risk (heuristic)")
    plt.title("Crosstalk risk per qubit (heuristic)")
    plt.savefig(os.path.join(out_dir, "crosstalk_risk_per_qubit.png"))
    plt.close()

    # Correlation: crosstalk risk vs min-entropy
    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore
        x = metrics["crosstalk_risk"].astype(float)
        y = metrics["min_entropy_est"].astype(float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        plt.figure(figsize=(6,5))
        plt.scatter(x, y, s=15, alpha=0.7)
        plt.xlabel("crosstalk_risk (heuristic)")
        plt.ylabel("min_entropy_est")
        plt.title(f"Correlation (Pearson {pr:.3f}, p={pp:.3g}; Spearman {sr:.3f}, p={sp:.3g})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "crosstalk_vs_minentropy.png"))
        plt.close()
        with open(os.path.join(out_dir, "correlation.txt"), "w", encoding="utf-8") as f:
            f.write(f"Pearson r={pr:.6f}, p={pp:.6g}\nSpearman r={sr:.6f}, p={sp:.6g}\n")
    except Exception:
        # scipy not available; skip correlation
        pass

    # Optional: coupling graph visualization if networkx is available
    try:
        import networkx as nx  # type: ignore

        G = nx.Graph()
        for _, row in calib.iterrows():
            q = int(row["Qubit"]) if "Qubit" in row else None
            if q is not None:
                G.add_node(q)
            pairs = parse_ecr_pairs(str(row.get("ECR error ", "")))
            for a, b, val in pairs:
                G.add_edge(a, b, weight=val)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        node_vals = metrics.set_index("qubit")["crosstalk_risk"].to_dict()
        v = np.array([node_vals.get(n, 0.0) for n in G.nodes])
        nodes = nx.draw_networkx_nodes(G, pos, node_color=v, cmap="viridis", node_size=80)
        edges = nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.colorbar(nodes, label="Crosstalk risk")
        plt.axis("off")
        plt.title("Coupling graph (ECR) colored by crosstalk risk")
        plt.savefig(os.path.join(out_dir, "coupling_graph_crosstalk.png"))
        plt.close()
    except Exception:
        # networkx not installed or visualization failed; skip
        pass


def _find_repo_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, ".git")) or os.path.isfile(os.path.join(cur, "readme.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_path)


def main():
    # Resolve repo root robustly
    script_dir = os.path.dirname(__file__)
    repo_root = _find_repo_root(script_dir)

    # Pick default CSV under Others/, fallback to any *calibrations*.csv
    default_csv = os.path.join(repo_root, "Others", "ibm_sherbrooke_calibrations_2024-09-12T05_28_56Z.csv")
    csv_path = os.environ.get("QRNG_CALIB_CSV", default_csv)
    if not os.path.isfile(csv_path):
        # try any calibrations CSV
        others_dir = os.path.join(repo_root, "Others")
        candidates = []
        if os.path.isdir(others_dir):
            for name in os.listdir(others_dir):
                if name.lower().endswith(".csv") and "calibrations" in name.lower():
                    candidates.append(os.path.join(others_dir, name))
        if candidates:
            csv_path = sorted(candidates)[0]
        else:
            raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    calib = load_calibrations(csv_path)
    metrics = compute_metrics(calib)

    out_dir = ensure_output_dir()
    metrics.to_csv(os.path.join(out_dir, "qrng_metrics.csv"), index=False)

    plot_distributions(calib, metrics, out_dir)

    # Print short summary
    avg_hmin = metrics["min_entropy_est"].mean()
    worst_hmin = metrics["min_entropy_est"].min()
    print(f"Avg min-entropy: {avg_hmin:.4f} bits; Worst min-entropy: {worst_hmin:.4f} bits")

    # Write a summary file with top-N biased/crosstalk qubits
    top_n = 10
    worst_bias = metrics.sort_values("bias_est", ascending=False).head(top_n)[["qubit", "bias_est", "readout_err", "p_meas0_prep1", "p_meas1_prep0"]]
    worst_xt = metrics.sort_values("crosstalk_risk", ascending=False).head(top_n)[["qubit", "crosstalk_risk", "ecr_avg_error_to_neighbors", "readout_err_neighbor_dispersion"]]
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Avg min-entropy: {avg_hmin:.4f}\n")
        f.write(f"Worst min-entropy: {worst_hmin:.4f}\n\n")
        f.write("Top biased qubits (by bias_est):\n")
        f.write(worst_bias.to_string(index=False))
        f.write("\n\nTop cross-talk risk qubits (heuristic):\n")
        f.write(worst_xt.to_string(index=False))


if __name__ == "__main__":
    main()
