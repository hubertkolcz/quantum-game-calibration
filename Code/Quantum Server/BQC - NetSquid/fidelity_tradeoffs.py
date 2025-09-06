"""
Heuristic fidelity vs successes-per-second trade-off explorer.

This script avoids lengthy simulations by combining known timings with proxy error/fidelity models
to quickly draw qualitative trade-off curves, complementing bqc.py's full simulation.

Outputs:
- Results/Quantum Server/plots/heuristic_succ_per_sec_vs_fidelity.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def succ_per_sec(exec_time_ns: float) -> float:
    # NetSquid timings are in ns; convert to seconds per execution, then flip
    sec = max(exec_time_ns, 1.0) * 1e-9
    return 1.0 / sec


def circuit_time_ns(measure_ns: float, init_ns: float, single_ns: float, two_qubit_ns: float, n_single: int, n_two: int) -> float:
    # Simple serial model: init + single + two + measure
    return init_ns + n_single * single_ns + n_two * two_qubit_ns + measure_ns


def fidelity_proxy(single_p: float, two_p: float, readout_err: float, n_single: int, n_two: int) -> float:
    # Product of (1 - error) factors (simplified); readout error applied once
    return (1 - single_p) ** n_single * (1 - two_p) ** n_two * (1 - readout_err)


def main():
    out_dir = os.path.join("Results", "Quantum Server", "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Base parameters inspired by configs and IBM calibrations
    init_ns = 1_000.0
    single_ns = 25.0
    two_ns = 250.0
    measure_ns = 1_200.0

    # Error ranges
    single_ps = np.linspace(0.0001, 0.01, 30)
    two_ps = np.linspace(0.005, 0.1, 30)
    readout_err = 0.02

    n_single = 6
    n_two = 1

    # Sweep mixes of single/two errors
    F = []
    S = []
    for sp in single_ps:
        for tp in two_ps:
            t_ns = circuit_time_ns(measure_ns, init_ns, single_ns, two_ns, n_single, n_two)
            S.append(succ_per_sec(t_ns))
            F.append(fidelity_proxy(sp, tp, readout_err, n_single, n_two))

    F = np.array(F)
    S = np.array(S)

    plt.figure(figsize=(8, 6))
    plt.scatter(F, S, s=10, alpha=0.5)
    plt.xlabel("Fidelity (proxy)")
    plt.ylabel("Successes per second (proxy)")
    plt.title("Heuristic trade-off: successes/sec vs fidelity")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(out_dir, "heuristic_succ_per_sec_vs_fidelity.png"))
    plt.close()

    # Print quick summary
    print(f"Fidelity proxy range: [{F.min():.3f}, {F.max():.3f}]")
    print(f"Successes/sec proxy range: [{S.min():.1f}, {S.max():.1f}]")


if __name__ == "__main__":
    main()
