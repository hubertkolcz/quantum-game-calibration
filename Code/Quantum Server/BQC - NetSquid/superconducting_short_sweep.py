"""
Run short sweeps around IBM-derived medians using a superconducting config.

Outputs CSVs and plots alongside existing results.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
from squidasm.run.stack.config import StackNetworkConfig, StackConfig, LinkConfig, HeraldedLinkConfig, GenericQDeviceConfig

from bqc import computation_round, ClientProgram, ServerProgram, PI_OVER_2


def load_superconducting_cfg() -> tuple[StackNetworkConfig, GenericQDeviceConfig]:
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "config_superconducting.yaml")
    cfg = StackNetworkConfig.from_file(cfg_path)
    # Extract the qdevice config for sweeping
    # In this minimal setup, reuse a GenericQDeviceConfig from file
    stack_cfg = GenericQDeviceConfig.from_file(os.path.join(os.path.dirname(__file__), "configs", "config_stack_trapped_ions.yaml"))
    return cfg, stack_cfg


def main():
    set_qstate_formalism(QFormalism.DM)
    shots = int(os.environ.get("BQC_NUM_TIMES", "200"))
    out_res = os.path.join("Results", "Quantum Server", "results")
    out_plot = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_plot, exist_ok=True)

    cfg, stack = load_superconducting_cfg()

    # Base medians
    sq = 0.000215
    tq = 0.007289
    mt = 1244.4
    init = 1000.0

    scans = {
        "single_qubit_gate_depolar_prob": np.linspace(sq*0.5, sq*1.5, 7),
        "two_qubit_gate_depolar_prob": np.linspace(tq*0.5, tq*1.5, 7),
        "measure_time": np.linspace(mt*0.8, mt*1.2, 7),
        "init_time": np.linspace(init*0.8, init*1.2, 7),
    }

    for label, sweep in scans.items():
        old = getattr(stack, label)
        rows = []
        for i, v in enumerate(sweep):
            setattr(stack, label, float(v))
            frac0 = computation_round(cfg, num_times=shots, alpha=PI_OVER_2, beta=PI_OVER_2, theta1=0.0, theta2=0.0)
            # Approximate timing: use measure_time + gate times as rough per-run ms proxy
            exec_ms_per_run = max(1.0, stack.measure_time / 1000.0)
            execs_per_sec = 1000.0 / exec_ms_per_run
            succ_per_sec = frac0 * execs_per_sec
            rows.append({"i": i, label: float(v), "succ_rate": float(frac0), "exec_ms_per_run": exec_ms_per_run, "succ_per_sec_corrected": succ_per_sec})
        setattr(stack, label, old)
        df = pd.DataFrame(rows)
        base = f"SC_{label}_{shots}"
        df.to_csv(os.path.join(out_res, f"{base}.csv"), index=False)
        plt.figure(figsize=(7,5))
        plt.plot(df[label], df["succ_per_sec_corrected"], marker='o')
        plt.xlabel(label)
        plt.ylabel("successes per second (corrected)")
        plt.title(base)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plot, f"{base}.png"))
        plt.close()


if __name__ == "__main__":
    main()
