The project consists of the following folders:
- Results: charts and spreadsheets with network calibration results.
- Code: neural network models, numerical solvers—including PINN.
- Other: technical documents.

A detailed description of the contents can be found in the master's thesis, in the Attachments section.

To run the BQC simulator, which outputs charts and simulation results in xls files, you need to install the SquidASM and NetSquid simulators on a Unix or WSL environment, following the instructions:
- NetSquid: https://netsquid.org/
- SquidASM: https://squidasm.readthedocs.io/en/latest/installation.html
	- Windows users: see Docs/WSL_NetSquid_Setup.md for a step-by-step WSL guide.

After installing both packages, run the following commands:
- cd Code/Quantum Server/BQC - NetSquid
- python bqc.py  (or set env BQC_NUM_TIMES=100 for a quick run)

The script should produce a set of measurements. Due to the default value, the simulation may take several hours. To reduce the time required for measurements (at the cost of accuracy), either set the environment variable BQC_NUM_TIMES (e.g., 100) or change the value of num_times in the script.

Previously generated experiment results are available in the main folder and in "Code/Quantum Server/BQC - NetSquid/plots".


To run the qGAN network, go to the folder "Code/Quantum Client/qGAN - Classifier". Running the models there requires installing the packages listed in the first cell of the program.

The AMD code and other programs in the Code/ folder can be run analogously to the above instructions.


Additional scripts (new):

- QRNG analysis and cross-talk (no device required):
	- Location: `Code/Quantum Client/qRNG - Analysis/qrng_analysis.py`
	- Uses: `Others/ibm_sherbrooke_calibrations_*.csv`
	- Output: `Results/Quantum Client/qRNG/` with metrics and plots
	- Run: set env `QRNG_CALIB_CSV` if using a different CSV; otherwise it picks the provided file.

- BQC trap experiments (requires NetSquid/SquidASM):
	- Location: `Code/Quantum Server/BQC - NetSquid/experiments_trap.py`
	- Output: CSV and plots under `Results/Quantum Server/plots/` comparing trap success vs computation outcomes.
	- Run similarly to `bqc.py` (same environment requirements). Use env `BQC_NUM_TIMES` to shorten runs.

- IBM calibration mapper to NetSquid-like params (no NetSquid needed):
	- Location: `Code/Quantum Server/BQC - NetSquid/ibm_calibration_mapper.py`
	- Output: `Results/Quantum Server/ibm_mapping/` with suggested depolarization and timing medians.
	- Run with optional env `IBM_CALIB_CSV` to point to a different calibration export.

- Heuristic fidelity-throughput explorer (no NetSquid needed):
	- Location: `Code/Quantum Server/BQC - NetSquid/fidelity_tradeoffs.py`
	- Output: `Results/Quantum Server/plots/heuristic_succ_per_sec_vs_fidelity.png`

See `Results/article_coverage.md` for a mapping between article claims and available scripts/figures.