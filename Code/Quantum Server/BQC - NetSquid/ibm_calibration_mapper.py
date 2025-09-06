"""
Map IBM calibration CSV to NetSquid-like parameter suggestions and quick success-per-second proxy.

This does NOT modify YAML files automatically; it emits a CSV/JSON with suggested values and a short report
to help set bqc.py ranges or pick realistic parameters when simulating superconducting qubits.

Inputs:
- Others/ibm_sherbrooke_calibrations_*.csv (default path can be overridden via IBM_CALIB_CSV env)

Outputs:
- Results/Quantum Server/ibm_mapping/suggested_params.csv
- Results/Quantum Server/ibm_mapping/summary.txt
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd


def ensure_dir() -> str:
    out_dir = os.path.join("Results", "Quantum Server", "ibm_mapping")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_ibm(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace("\u00a0", " ") for c in df.columns]
    return df


def suggest_params(df: pd.DataFrame) -> pd.DataFrame:
    # We derive rough proxies that could map into generic gate depolarization probabilities and timings.
    # Readout length ns is present; errors for sx/x and ECR are also present.
    # We'll propose:
    # - single_qubit_gate_depolar_prob ≈ median of {sx error, Pauli-X error, rz error}
    # - two_qubit_gate_depolar_prob ≈ median of ECR error edges
    # - measure_time ≈ median Readout length (ns)
    # - init_time ≈ a fixed conservative value (since IBM API doesn't expose it)
    # For per-qubit mapping we keep all stats, plus global medians.

    # Resolve likely single-qubit error columns via lowercase contains checks
    all_cols = list(df.columns)
    low = {c: c.lower() for c in all_cols}
    s_candidates = []
    for c in all_cols:
        lc = low[c]
        if ("error" in lc) and ("ecr" not in lc) and ("readout" not in lc) and ("id error" not in lc):
            # pick rz/sx/x specific ones if present, otherwise keep generic
            if "rz" in lc or "sx" in lc or "pauli-x" in lc or lc.endswith(" error ") or lc.endswith(" error"):
                s_candidates.append(c)
    s_candidates = list(dict.fromkeys(s_candidates))  # dedupe
    if s_candidates:
        df_s = df[s_candidates].apply(pd.to_numeric, errors="coerce")
        df["single_qubit_error_proxy"] = df_s.replace(0.0, np.nan).median(axis=1, skipna=True)
    else:
        df["single_qubit_error_proxy"] = np.nan

    # Readout length/time
    m_col = None
    for c in all_cols:
        if "readout" in low[c] and "length" in low[c] and "ns" in low[c]:
            m_col = c
            break
    df["measure_time_ns"] = pd.to_numeric(df[m_col], errors="coerce") if m_col else np.nan

    # Two-qubit ECR median per-qubit: parse and take median of connected edges
    # Detect ECR error column
    ecr_col = None
    for c in all_cols:
        lc = low[c]
        if ("ecr" in lc) and ("error" in lc):
            ecr_col = c
            break

    def ecr_median(row) -> float:
        if not ecr_col:
            return np.nan
        val = row.get(ecr_col, "")
        if isinstance(val, float) and np.isnan(val):
            return np.nan
        val = str(val).strip()
        if not val or val.lower() == "nan":
            return np.nan
        parts = []
        for piece in val.split(";"):
            if ":" in piece:
                try:
                    parts.append(float(piece.split(":", 1)[1]))
                except Exception:
                    pass
        return float(np.median(parts)) if parts else np.nan

    df["two_qubit_error_proxy"] = df.apply(ecr_median, axis=1)

    # Global medians
    global_single = float(np.nanmedian(df["single_qubit_error_proxy"]))
    global_two = float(np.nanmedian(df["two_qubit_error_proxy"]))
    global_measure = float(np.nanmedian(df["measure_time_ns"]))

    # Suggest a conservative init_time (ns). For superconducting, reset ~ several 100ns - 1us.
    suggested_init_ns = 1_000.0

    # Assemble per-qubit suggestions
    out = df[["Qubit", "single_qubit_error_proxy", "two_qubit_error_proxy", "measure_time_ns"]].copy()
    out.rename(columns={
        "single_qubit_error_proxy": "suggest_single_qubit_gate_depolar_prob",
        "two_qubit_error_proxy": "suggest_two_qubit_gate_depolar_prob",
        "measure_time_ns": "suggest_measure_time_ns",
    }, inplace=True)

    # Append global row with qubit = -1
    out_global = pd.DataFrame([
        {
            "Qubit": -1,
            "suggest_single_qubit_gate_depolar_prob": global_single,
            "suggest_two_qubit_gate_depolar_prob": global_two,
            "suggest_measure_time_ns": global_measure,
        }
    ])

    out_all = pd.concat([out, out_global], ignore_index=True)
    return out_all


def summarize(out_dir: str, df_suggest: pd.DataFrame) -> None:
    g = df_suggest[df_suggest["Qubit"] == -1].iloc[0].to_dict()
    txt = [
        "Suggested global parameters (medians):",
        f"  single_qubit_gate_depolar_prob ≈ {g['suggest_single_qubit_gate_depolar_prob']:.4g}",
        f"  two_qubit_gate_depolar_prob ≈ {g['suggest_two_qubit_gate_depolar_prob']:.4g}",
        f"  measure_time ≈ {g['suggest_measure_time_ns']:.1f} ns",
        f"  init_time ≈ 1000 ns (fixed heuristic)",
        "",
        "Note: map these into config_trapped_ions.yaml or a new superconducting config if desired.",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt))


def _find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, ".git")) or os.path.isfile(os.path.join(cur, "readme.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start)


def main():
    repo_root = _find_repo_root(os.path.dirname(__file__))
    csv_default = os.path.join(repo_root, "Others", "ibm_sherbrooke_calibrations_2024-09-12T05_28_56Z.csv")
    csv_path = os.environ.get("IBM_CALIB_CSV", csv_default)
    if not os.path.isfile(csv_path):
        # try to find any calibrations csv in Others/
        others_dir = os.path.join(repo_root, "Others")
        candidates = []
        if os.path.isdir(others_dir):
            for name in os.listdir(others_dir):
                if name.lower().endswith(".csv") and "calibrations" in name.lower():
                    candidates.append(os.path.join(others_dir, name))
        if candidates:
            csv_path = sorted(candidates)[0]
        else:
            raise FileNotFoundError(csv_path)
    out_dir = ensure_dir()
    df = load_ibm(csv_path)
    df_suggest = suggest_params(df)
    csv_out = os.path.join(out_dir, "suggested_params.csv")
    df_suggest.to_csv(csv_out, index=False)
    summarize(out_dir, df_suggest)
    print(f"Wrote suggestions to {csv_out}")


if __name__ == "__main__":
    main()
