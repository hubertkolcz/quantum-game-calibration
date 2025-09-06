"""
Sweep client rotation angles (alpha, beta) and record P(m2=0).

Uses NV color center stack by default; controllable via ANGLES_STACK env: nv|ions.
Shots: BQC_NUM_TIMES (default 200).

Outputs:
  Results/Quantum Server/results/angles_sweep_<stack>_<num_times>.csv
  Results/Quantum Server/plots/angles_sweep_<stack>_<num_times>.png (heatmap)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from bqc import load_configurations, computation_round


def ensure_dirs():
    out_res = os.path.join("Results", "Quantum Server", "results")
    out_plot = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_plot, exist_ok=True)
    return out_res, out_plot


def main():
    set_qstate_formalism(QFormalism.DM)
    num_times = int(os.environ.get("BQC_NUM_TIMES", "200"))
    stack_sel = os.environ.get("ANGLES_STACK", "nv").lower()

    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    if stack_sel.startswith("nv"):
        cfg = cfg_nv
        stack_name = "nv"
    else:
        cfg = cfg_ions
        stack_name = "ions"

    # angles grid
    vals = np.linspace(0.0, np.pi, 9)
    rows = []
    for a in vals:
        for b in vals:
            frac0 = computation_round(cfg, num_times=num_times, alpha=a, beta=b, theta1=0.0, theta2=0.0)
            rows.append({"alpha": float(a), "beta": float(b), "frac0": float(frac0)})

    out_res, out_plot = ensure_dirs()
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_res, f"angles_sweep_{stack_name}_{num_times}.csv")
    df.to_csv(csv_path, index=False)

    # heatmap
    pivot = df.pivot(index="alpha", columns="beta", values="frac0")
    plt.figure(figsize=(6,5))
    im = plt.imshow(pivot.values, origin='lower', extent=[0, np.pi, 0, np.pi], aspect='auto', vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, label='P(m2=0)')
    plt.xlabel('beta [rad]')
    plt.ylabel('alpha [rad]')
    plt.title(f'Angles sweep ({stack_name}), shots={num_times}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_plot, f"angles_sweep_{stack_name}_{num_times}.png"))
    plt.close()


if __name__ == "__main__":
    main()
"""
Sweep client angles (alpha, beta, theta1, theta2) and record P(m2=0) for baseline configs.
Requires NetSquid/SquidASM. Run inside WSL venv.

Outputs: CSV and plots under Results/Quantum Server/plots/
"""
from __future__ import annotations
import os
import math
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from bqc import load_configurations, computation_round, PI_OVER_2


def ensure_out() -> str:
    out_dir = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def grid(vals: List[float]) -> List[Tuple[float, float]]:
    pts = []
    for a in vals:
        for b in vals:
            pts.append((a, b))
    return pts


def main():
    set_qstate_formalism(QFormalism.DM)
    out_dir = ensure_out()
    # angles to test
    angs = [0.0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
    tests = grid(angs)
    num_times = int(os.environ.get("BQC_NUM_TIMES", "200"))

    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()

    rows = []
    for tech, cfg in [("NV", cfg_nv), ("Generic", cfg_ions)]:
        for (alpha, beta) in tests:
            frac0 = computation_round(cfg, num_times=num_times, alpha=alpha, beta=beta, theta1=0.0, theta2=0.0)
            rows.append({
                "tech": tech,
                "alpha": alpha,
                "beta": beta,
                "theta1": 0.0,
                "theta2": 0.0,
                "num_times": num_times,
                "frac0": frac0,
                "frac1": 1 - frac0,
            })

    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(out_dir, f"{ts}_angles_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Plot P(m2=0) vs alpha for each beta and tech
    for tech, g in df.groupby("tech"):
        plt.figure(figsize=(8,5))
        for beta, gb in g.groupby("beta"):
            # sort by alpha
            gb = gb.sort_values("alpha")
            plt.plot(gb["alpha"].values, gb["frac0"].values, marker='o', label=f"beta={beta:.2f}")
        plt.xlabel("alpha (rad)")
        plt.ylabel("P(m2=0)")
        plt.title(f"Angle sweep {tech}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"angles_sweep_{tech}.png"))
        plt.close()

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
