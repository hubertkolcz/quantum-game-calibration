from __future__ import annotations
"""
Run a short sweep around superconducting-like parameters suggested by ibm_mapping.
Outputs plots into Results/Quantum Server/plots/ and CSVs into working directory.
"""
import os
import yaml
import numpy as np
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
from bqc import load_configurations, experiment


def main():
    set_qstate_formalism(QFormalism.DM)
    num_times = int(os.environ.get("BQC_NUM_TIMES", "200"))
    cfg_ions, cfg_nv, ions_stack, nv_stack = load_configurations()

    # Load superconducting suggestions
    cfg_path = os.path.join(os.path.dirname(__file__), "config_superconducting.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        sc = yaml.safe_load(f)["superconducting"]

    # Sweep +/- 50% around suggested medians
    factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

    # Map onto the generic stack (ions_stack used as GenericQDeviceConfig in bqc)
    for key in [
        "init_time",
        "single_qubit_gate_depolar_prob",
        "two_qubit_gate_depolar_prob",
        "measure_time",
    ]:
        base = float(sc[key])
        sweep = base * factors
        experiment(ions_stack, cfg_ions, base, sweep, key, num_times)


if __name__ == "__main__":
    main()
