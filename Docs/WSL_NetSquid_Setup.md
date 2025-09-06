# NetSquid on Windows via WSL: Setup and Run Guide

This guide shows how to install NetSquid/SquidASM on Windows using WSL, then run the NetSquid scripts in this repository (BQC experiments under `Code/Quantum Server/BQC - NetSquid`).

Note: NetSquid officially supports Linux and macOS. On Windows, use WSL2 (Ubuntu recommended).

## 0) Prerequisites
- Windows 10/11 with WSL2
- An account on the NetSquid forum (required to download the `netsquid` wheel)
  - Register: https://forum.netsquid.org/ucp.php?mode=register
  - You’ll need the username/password in step 5.

## 1) Install WSL (Windows PowerShell)
Run PowerShell as Administrator and install Ubuntu 22.04:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted. Launch “Ubuntu” from Start Menu to finish first-time setup.

## 2) Update Ubuntu and install tools (inside Ubuntu)
```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git build-essential python3.10 python3.10-venv python3-pip
```

Tip: NetSquid is usually tested on Python 3.10. If you prefer 3.11 and it’s supported for your wheel, replace `python3.10` with `python3.11` consistently below.

## 3) Choose a working folder (inside Ubuntu)
You can run directly from your Windows repo via `/mnt/c` or copy the code into Linux for better FS performance.

Option A: Use your existing Windows workspace (simple)
```bash
cd /mnt/c/Users/Hubert/Desktop/Projects/quantum-game-calibration
```

Option B: Copy the NetSquid subfolder to Linux home (faster)
```bash
mkdir -p ~/quantum-game-calibration
cp -r /mnt/c/Users/Hubert/Desktop/Projects/quantum-game-calibration/Code/Quantum\ Server/BQC\ -\ NetSquid ~/quantum-game-calibration/
cd ~/quantum-game-calibration/BQC\ -\ NetSquid
```

The rest of the commands assume you’re in the project root (`quantum-game-calibration`).

## 4) Create and activate a Python venv (inside Ubuntu)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip
```

## 5) Configure NetSquid credentials (inside Ubuntu)
Export your NetSquid forum credentials to env vars so pip (and SquidASM) can fetch from the private wheel index:
```bash
export NETSQUIDPYPI_USER="your_forum_username"
export NETSQUIDPYPI_PWD="your_forum_password"
```
Alternatively, put the password in a file and set `NETSQUIDPYPI_PWD_FILEPATH` instead of `NETSQUIDPYPI_PWD`.

To make these persistent, add them to `~/.bashrc`.

## 6) Install NetSquid + SquidASM (inside Ubuntu)
Two supported paths—use one.

A) Minimal pip-based install
```bash
pip install --extra-index-url https://pypi.netsquid.org netsquid
pip install squidasm netqasm pydynaa numpy pandas matplotlib
```

B) Recommended SquidASM repo install (runs an install script)
```bash
# Ensure credentials from step 5 are exported in this shell
cd ~
git clone https://github.com/QuTech-Delft/squidasm.git
cd squidasm
make install
make verify  # optional self-check
```

If `make install` fails to download `netsquid`, re-check the credentials and that `NETSQUIDPYPI_*` are exported in this shell.

## 7) Verify imports (inside Ubuntu)
```bash
python - << 'PY'
import sys
print(sys.version)
import netsquid as ns
import squidasm, netqasm, pydynaa
print('OK: NetSquid', ns.__version__)
PY
```

## 8) Run this repo’s NetSquid scripts
From the repo root in Ubuntu shell (`quantum-game-calibration`):

```bash
source .venv/bin/activate  # if not already active
cd "Code/Quantum Server/BQC - NetSquid"
mkdir -p graphs  # bqc.py saves plots here
# quick run
BQC_NUM_TIMES=100 python bqc.py
# trap vs. computation sweeps
BQC_NUM_TIMES=100 python experiments_trap.py
```

Notes:
- `bqc.py` uses env var `BQC_NUM_TIMES` (default 1000). For a quick smoke test, set it lower (e.g., 50–200).
- Config YAMLs under `configs/` are loaded via paths relative to this file, so run from this folder.
- Outputs: CSVs in the working directory and PNG plots in `./graphs/` (plus additional plots under `Results/Quantum Server/plots/` for `experiments_trap.py`).

## 9) Common pitfalls
- “Could not find a version that satisfies the requirement netsquid”
  - Ensure step 5 was done in the same shell. Use the exact extra index URL and your forum creds.
- Python version mismatch
  - Prefer Python 3.10 unless your NetSquid wheel supports newer versions. If using `python3.11`, ensure compatibility.
- Headless plotting errors
  - This guide uses `pyplot.savefig`; no display required. If needed, set `MPLBACKEND=Agg`.
- Slow file I/O under `/mnt/c`
  - Large sweeps generate many points; consider Option B (copy code into `~`) for speed.

## 10) VS Code (optional)
Use the “Remote - WSL” extension to open the repo in Ubuntu directly:
- In Windows VS Code: “WSL: Connect to WSL”, then `File > Open Folder` and select the repo path inside WSL.

---
If you hit issues, capture the exact error and the output of:
```bash
python -V; pip -V; python -c "import netsquid, squidasm, netqasm, pydynaa; print(netsquid.__version__)"
```
