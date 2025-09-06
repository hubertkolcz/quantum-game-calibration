"""
Aggregate results from generated scripts into a concise validation report (markdown).

Inputs expected (optional if missing):
- Results/Quantum Client/qRNG/summary.txt
- Results/Quantum Server/ibm_mapping/summary.txt
- Results/Quantum Server/plots/heuristic_succ_per_sec_vs_fidelity.png

Output:
- Results/validation_report.md
"""
from __future__ import annotations
import os


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return "(not available)"


def main():
    out_dir = os.path.join("Results")
    os.makedirs(out_dir, exist_ok=True)

    qrng_summary = read_text(os.path.join("Results", "Quantum Client", "qRNG", "summary.txt"))
    ibm_map_summary = read_text(os.path.join("Results", "Quantum Server", "ibm_mapping", "summary.txt"))

    lines = []
    lines.append("# Validation report\n")
    lines.append("## QRNG analysis\n")
    lines.append("" + qrng_summary + "\n")
    lines.append("## IBM calibration mapping\n")
    lines.append("" + ibm_map_summary + "\n")
    lines.append("## Heuristic fidelity-throughput plot\n")
    lines.append("See Results/Quantum Server/plots/heuristic_succ_per_sec_vs_fidelity.png\n")

    out_path = os.path.join(out_dir, "validation_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
