"""
Compare trap success for dummy=1 vs dummy=2 across key parameter sweeps.

Outputs CSVs and plots under Results/Quantum Server/plots/.
Control shots via BQC_NUM_TIMES (default 200).
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from bqc import load_configurations, setup_parameter_ranges, trap_round, PI_OVER_2


def ensure_dir() -> str:
    out_dir = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sweep_symmetry(stack, cfg, label: str, base: float, sweep: np.ndarray, shots: int) -> pd.DataFrame:
    old = getattr(stack, label)
    rows = []
    for i, v in enumerate(sweep):
        setattr(stack, label, v)
        s1 = trap_round(cfg, num_times=shots, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=1)
        s2 = trap_round(cfg, num_times=shots, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=2)
        rows.append({"i": i, label: v, "trap_succ_dummy1": s1, "trap_succ_dummy2": s2, "diff": s1 - s2})
    setattr(stack, label, old)
    return pd.DataFrame(rows)


def main():
    set_qstate_formalism(QFormalism.DM)
    shots = int(os.environ.get("BQC_NUM_TIMES", "200"))
    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    params = setup_parameter_ranges()
    out_dir = ensure_dir()

    for label, (base, sweep) in params["color_centers"].items():
        print(f"NV symmetry sweep: {label}")
        df = sweep_symmetry(nv_stack, cfg_nv, label, base, sweep, shots)
        ts = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(os.path.join(out_dir, f"{ts}_nv_{label}_trap_symmetry.csv"), index=False)
        plt.figure(figsize=(8,5))
        plt.plot(df[label], df["trap_succ_dummy1"], label="dummy=1")
        plt.plot(df[label], df["trap_succ_dummy2"], label="dummy=2")
        plt.xlabel(label)
        plt.ylabel("trap success")
        plt.title(f"NV trap symmetry: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"nv_{label}_trap_symmetry.png"))
        plt.close()

    for label, (base, sweep) in params["trapped_ions"].items():
        print(f"Ions symmetry sweep: {label}")
        df = sweep_symmetry(ions_stack, cfg_ions, label, base, sweep, shots)
        ts = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(os.path.join(out_dir, f"{ts}_ions_{label}_trap_symmetry.csv"), index=False)
        plt.figure(figsize=(8,5))
        plt.plot(df[label], df["trap_succ_dummy1"], label="dummy=1")
        plt.plot(df[label], df["trap_succ_dummy2"], label="dummy=2")
        plt.xlabel(label)
        plt.ylabel("trap success")
        plt.title(f"Ions trap symmetry: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ions_{label}_trap_symmetry.png"))
        plt.close()


if __name__ == "__main__":
    main()
"""
Compare trap success for dummy=1 vs dummy=2 across parameter sweeps.
Outputs CSV and plots under Results/Quantum Server/plots/
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
    out_dir = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_symmetry(stack, cfg, label: str, base_val: float, sweep: np.ndarray, num_times: int) -> pd.DataFrame:
    old = getattr(stack, label)
    rows = []
    for i, v in enumerate(sweep):
        setattr(stack, label, v)
        s1 = trap_round(cfg, num_times=num_times, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=1)
        s2 = trap_round(cfg, num_times=num_times, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0, dummy=2)
        rows.append({"i": i, label: v, "trap_succ_dummy1": s1, "trap_succ_dummy2": s2, "diff": s1 - s2})
    setattr(stack, label, old)
    return pd.DataFrame(rows)


def main():
    set_qstate_formalism(QFormalism.DM)
    out_dir = ensure_out()
    num_times = int(os.environ.get("BQC_NUM_TIMES", "200"))
    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()
    param_sets = setup_parameter_ranges()

    for tech, (stack, cfg) in {"NV": (nv_stack, cfg_nv), "Generic": (ions_stack, cfg_ions)}.items():
        for label, (base, sweep) in param_sets["color_centers" if tech=="NV" else "trapped_ions"].items():
            print(f"{tech} symmetry sweep: {label}")
            df = run_symmetry(stack, cfg, label, base, sweep, num_times)
            ts = time.strftime("%Y%m%d-%H%M%S")
            csv_path = os.path.join(out_dir, f"{ts}_{tech}_{label}_trap_symmetry.csv")
            df.to_csv(csv_path, index=False)
            # plot
            plt.figure(figsize=(8,5))
            x = df[label].to_numpy()
            plt.plot(x, df["trap_succ_dummy1"].to_numpy(), label="dummy=1")
            plt.plot(x, df["trap_succ_dummy2"].to_numpy(), label="dummy=2")
            plt.plot(x, df["diff"].to_numpy(), label="diff", linestyle='--')
            plt.xlabel(label)
            plt.ylabel("trap success")
            plt.title(f"Trap symmetry {tech}: {label}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"trap_symmetry_{tech}_{label}.png"))
            plt.close()


if __name__ == "__main__":
    main()
