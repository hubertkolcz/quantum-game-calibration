The project consists of the following folders:
- Results: charts and spreadsheets with network calibration results.
- Code: neural network models, numerical solvers—including PINN.
- Other: technical documents.

A detailed description of the contents can be found in the master's thesis, in the Attachments section.

To run the BQC simulator, which outputs charts and simulation results in xls files, you need to install the SquidASM and NetSquid simulators on a Unix or WSL environment, following the instructions:
- NetSquid: https://netsquid.org/
- SquidASM: https://squidasm.readthedocs.io/en/latest/installation.html

After installing both packages, run the following commands:
- cd Code/Quantum Server/BQC - NetSquid
- python bqc.py

The script should produce a set of measurements. Due to the default value, the simulation may take several hours. To reduce the time required for measurements (at the cost of accuracy), change the value num_times = 1000 in line 241 to e.g. 10.

Previously generated experiment results are available in the main folder and in "Code/Quantum Server/BQC - NetSquid/plots".


To run the qGAN network, go to the folder "Code/Quantum Client/qGAN - Classifier". Running the models there requires installing the packages listed in the first cell of the program.

The AMD code and other programs in the Code/ folder can be run analogously to the above instructions.