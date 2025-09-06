## Article coverage map

This document tracks which numerical results from the article are implemented and where, and what was added.

- BQC simulations (success rate, executions per second, successes per second vs hardware parameters)
  - Implemented by: `Code/Quantum Server/BQC - NetSquid/bqc.py`
  - Plots present under `Results/Quantum Server/plots/*_vs_succ_per_sec.png` (for multiple `num_times`).
  - New: `experiments_trap.py` adds trap-round success vs computation outputs across parameter sweeps, outputting CSV and plots.

- Trade-off between fidelity and throughput
  - Previously: `*_succ_per_sec_vs_fidelity.png` present as static outputs.
  - New: `fidelity_tradeoffs.py` provides a fast heuristic explorer to reproduce and adjust the fidelity-throughput scatter.

- Quantum client classifier via qGAN
  - Implemented by notebooks under `Code/Quantum Client/qGAN - Classifier/*.ipynb` with saved weights. Results in `Results/Quantum Client`.

- Classical models (LLE numerical and PINN)
  - Implemented by notebooks under `Code/Classical Client` with figures in `Results/Classical Client`.

- QRNG aspects (bias, cross-talk risks, device suitability)
  - New: `Code/Quantum Client/qRNG - Analysis/qrng_analysis.py` parses IBM calibration CSV and computes per-qubit
    bias and min-entropy estimates, neighbor ECR-derived cross-talk heuristic, and produces plots. Outputs saved under
    `Results/Quantum Client/qRNG/`.

- Mapping real calibrations to simulation parameters
  - New: `Code/Quantum Server/BQC - NetSquid/ibm_calibration_mapper.py` computes suggested single/two-qubit error and measurement time
    proxies from IBM calibration CSV and writes a summary suitable for adjusting NetSquid configs.

If a result is missing from the article or requires a more detailed experimental protocol, consider extending:
- Add superconducting-specific NetSquid configs and integrate suggested parameters from the mapper.
- Add GPU-accelerated sweeps and confidence intervals around success probabilities (Wilson intervals) in both bqc and trap experiments.
