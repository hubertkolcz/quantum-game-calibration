from __future__ import annotations
import time
import math
import os
import statistics
import pandas as pd
from typing import Any, Dict, Generator

import numpy as np
import netsquid as ns
from matplotlib import pyplot

from netqasm.lang.ir import BreakpointAction
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

from pydynaa import EventExpression
from squidasm.run.stack.config import  StackNetworkConfig, StackConfig, LinkConfig, HeraldedLinkConfig, GenericQDeviceConfig, NVQDeviceConfig
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

# Example Blind Quantum Computation application.
# See the README.md for the computation that is performed.

class ClientProgram(Program):
    PEER = "server"

    def __init__(
        self,
        alpha: float,
        beta: float,
        trap: bool,
        dummy: int,
        theta1: float,
        theta2: float,
        r1: int,
        r2: int,
    ):
        self._alpha = alpha
        self._beta = beta
        self._trap = trap
        self._dummy = dummy
        self._theta1 = theta1
        self._theta2 = theta2
        self._r1 = r1
        self._r2 = r2

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="client_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=2,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]

        # Create EPR pair
        epr1 = epr_socket.create_keep()[0]

        # RSP
        if self._trap and self._dummy == 2:
            # remotely-prepare a dummy state
            p2 = epr1.measure(store_array=False)
        else:
            epr1.rot_Z(angle=self._theta2)
            epr1.H()
            p2 = epr1.measure(store_array=False)

        # Create EPR pair
        epr2 = epr_socket.create_keep()[0]

        # RSP
        if self._trap and self._dummy == 1:
            # remotely-prepare a dummy state
            p1 = epr2.measure(store_array=False)
        else:
            epr2.rot_Z(angle=self._theta1)
            epr2.H()
            p1 = epr2.measure(store_array=False)

        yield from conn.flush()

        p1 = int(p1)
        p2 = int(p2)

        if self._trap and self._dummy == 2:
            delta1 = -self._theta1 + (p1 + self._r1) * math.pi
        else:
            delta1 = self._alpha - self._theta1 + (p1 + self._r1) * math.pi
        csocket.send_float(delta1)

        m1 = yield from csocket.recv_int()
        if self._trap and self._dummy == 1:
            delta2 = -self._theta2 + (p2 + self._r2) * math.pi
        else:
            delta2 = (
                math.pow(-1, (m1 + self._r1)) * self._beta
                - self._theta2
                + (p2 + self._r2) * math.pi
            )
        csocket.send_float(delta2)

        return {"p1": p1, "p2": p2}

class ServerProgram(Program):
    PEER = "client"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="server_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=2,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]
        csocket: ClassicalSocket = context.csockets[self.PEER]

        # Create EPR Pair
        epr1 = epr_socket.recv_keep()[0]
        epr2 = epr_socket.recv_keep()[0]
        epr2.cphase(epr1)

        yield from conn.flush()

        delta1 = yield from csocket.recv_float()

        epr2.rot_Z(angle=delta1)
        epr2.H()
        m1 = epr2.measure(store_array=False)
        yield from conn.flush()

        m1 = int(m1)

        csocket.send_int(m1)

        delta2 = yield from csocket.recv_float()

        epr1.rot_Z(angle=delta2)
        epr1.H()
        conn.insert_breakpoint(BreakpointAction.DUMP_LOCAL_STATE)
        m2 = epr1.measure(store_array=False)
        yield from conn.flush()

        m2 = int(m2)
        return {"m1": m1, "m2": m2}


PI = math.pi
PI_OVER_2 = math.pi / 2


def computation_round(
    cfg: StackNetworkConfig,
    num_times: int = 1,
    alpha: float = 0.0,
    beta: float = 0.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
) -> None:
    client_program = ClientProgram(
        alpha=alpha,
        beta=beta,
        trap=False,
        dummy=-1,
        theta1=theta1,
        theta2=theta2,
        r1=0,
        r2=0,
    )
    server_program = ServerProgram()

    _, server_results = run(
        cfg, {"client": client_program, "server": server_program}, num_times=num_times
    )

    m2s = [result["m2"] for result in server_results]
    num_zeros = len([m for m in m2s if m == 0])
    frac0 = round(num_zeros / num_times, 2)
    frac1 = 1 - frac0
    print(f"dist (0, 1) = ({frac0}, {frac1})")
    return frac0


def trap_round(
    cfg: StackNetworkConfig,
    num_times: int = 1,
    alpha: float = 0.0,
    beta: float = 0.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
    dummy: int = 1,
) -> None:
    client_program = ClientProgram(
        alpha=alpha,
        beta=beta,
        trap=True,
        dummy=dummy,
        theta1=theta1,
        theta2=theta2,
        r1=0,
        r2=0,
    )
    server_program = ServerProgram()

    client_results, server_results = run(
        cfg, {"client": client_program, "server": server_program}, num_times=num_times
    )

    p1s = [result["p1"] for result in client_results]
    p2s = [result["p2"] for result in client_results]
    m1s = [result["m1"] for result in server_results]
    m2s = [result["m2"] for result in server_results]

    assert dummy in [1, 2]
    if dummy == 1:
        num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
    else:
        num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

    frac_fail = round(num_fails / num_times, 2)
    frac_succ = 1 - frac_fail
    print(f"succ rate: {frac_succ}")
    return frac_succ

PI_OVER_2 = np.pi / 2

def load_configurations():
    """Load configurations for trapped ions and color centers."""
    # Trapped Ions
    cfg_trapped_ions = StackNetworkConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_trapped_ions.yaml")
    )
    ions_stack = GenericQDeviceConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_stack_trapped_ions.yaml")
    )
    ions_stack_alice = StackConfig(name="client", qdevice_typ="generic", qdevice_cfg=ions_stack)
    ions_stack_bob = StackConfig(name="server", qdevice_typ="generic", qdevice_cfg=ions_stack)
    trapped_link = HeraldedLinkConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_link_trapped_ions.yaml")
    )
    ions_link = LinkConfig(stack1="client", stack2="server", typ="heralded", cfg=trapped_link)

    cfg_network_trapped_ions = StackNetworkConfig(stacks=[ions_stack_alice, ions_stack_bob], links=[ions_link])

    # Color Centers - NV
    cfg_colors_centers = StackNetworkConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_color_centers.yaml")
    )
    colors_stack = NVQDeviceConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_stack_color_centers.yaml")
    )
    stack_alice = StackConfig(name="client", qdevice_typ="nv", qdevice_cfg=colors_stack)
    stack_bob = StackConfig(name="server", qdevice_typ="nv", qdevice_cfg=colors_stack)
    colors_link = HeraldedLinkConfig.from_file(
        os.path.join(os.path.dirname(__file__), "configs/config_link_color_centers.yaml")
    )
    nv_link = LinkConfig(stack1="client", stack2="server", typ="heralded", cfg=colors_link)

    cfg_network_colors_centers = StackNetworkConfig(stacks=[stack_alice, stack_bob], links=[nv_link])

    return cfg_network_trapped_ions, cfg_network_colors_centers, ions_stack, colors_stack

def setup_parameter_ranges():
    """Setup parameter ranges for experiments."""
    # Trapped Ions - General
    init_time = 40000
    single_qubit_gate_depolar_prob = 0.015
    two_qubit_gate_depolar_prob = 0.08
    measure_time = 0.12

    range_init_time = np.arange(init_time * 0.4, init_time * 2, init_time * 0.1)
    range_single_qubit_gate_depolar_prob = np.arange(single_qubit_gate_depolar_prob * 0.5, single_qubit_gate_depolar_prob * 2.5, single_qubit_gate_depolar_prob * 0.1)
    range_two_qubit_gate_depolar_prob = np.arange(two_qubit_gate_depolar_prob * 0.5, two_qubit_gate_depolar_prob * 2.5, two_qubit_gate_depolar_prob * 0.1)
    range_measure_time = np.arange(measure_time * 0.4, measure_time * 2, measure_time * 0.1)

    # Color Centers - NV
    carbon_init = 350000
    electron_init = 2500
    measure = 4000
    electron_single_qubit_depolar_prob = 0.008
    carbon_z_rot_depolar_prob = 0.0015
    ec_gate_depolar_prob = 0.05

    range_nv_carbon_init = np.arange(carbon_init * 0.4, carbon_init * 2, carbon_init * 0.1)
    range_nv_electron_init = np.arange(electron_init * 0.4, electron_init * 2, electron_init * 0.1)
    range_nv_measure = np.arange(measure * 0.4, measure * 2, measure * 0.1)
    range_nv_electron_single_qubit_depolar_prob = np.arange(electron_single_qubit_depolar_prob * 0.5, electron_single_qubit_depolar_prob * 2.5, electron_single_qubit_depolar_prob * 0.1)
    range_nv_carbon_z_rot_depolar_prob = np.arange(carbon_z_rot_depolar_prob * 0.5, carbon_z_rot_depolar_prob * 2.5, carbon_z_rot_depolar_prob * 0.1)
    range_nv_ec_gate_depolar_prob = np.arange(ec_gate_depolar_prob * 0.5, ec_gate_depolar_prob * 2.5, ec_gate_depolar_prob * 0.1)

    return {
        "trapped_ions": {
            "init_time": (init_time, range_init_time),
            "single_qubit_gate_depolar_prob": (single_qubit_gate_depolar_prob, range_single_qubit_gate_depolar_prob),
            "two_qubit_gate_depolar_prob": (two_qubit_gate_depolar_prob, range_two_qubit_gate_depolar_prob),
            "measure_time": (measure_time, range_measure_time),
        },
        "color_centers": {
            "carbon_init": (carbon_init, range_nv_carbon_init),
            "electron_init": (electron_init, range_nv_electron_init),
            "measure": (measure, range_nv_measure),
            "electron_single_qubit_depolar_prob": (electron_single_qubit_depolar_prob, range_nv_electron_single_qubit_depolar_prob),
            "carbon_z_rot_depolar_prob": (carbon_z_rot_depolar_prob, range_nv_carbon_z_rot_depolar_prob),
            "ec_gate_depolar_prob": (ec_gate_depolar_prob, range_nv_ec_gate_depolar_prob),
        }
    }

def experiment(stack, cfg, parameter, parameters_set, label, num_times):
    """Conduct an experiment with varying parameters."""
    avg_exec_per_sec = []
    avg_succ_rate = []
    avg_combined = []

    logger = {
        'id': [],
        'exec_per_sec': [],
        'succ_rate': [],
        'succ_per_sec': [],
        'm1': [],
        'm2': [],
        'p1': [],
        'p2': [],
        label: []
    }

    old_parameter = getattr(stack, label)

    for i, element in enumerate(parameters_set):
        setattr(stack, label, element)
        start_time = ns.sim_time(magnitude=ns.MILLISECOND)
        client_results, server_results = run(cfg, {"client": ClientProgram(alpha=PI_OVER_2, beta=PI_OVER_2, trap=False, dummy=-1, theta1=0.0, theta2=0.0, r1=0, r2=0), "server": ServerProgram()}, num_times=num_times)
        succ_rate = computation_round(cfg, num_times, alpha=PI_OVER_2, beta=PI_OVER_2)
        end_time = ns.sim_time(magnitude=ns.MILLISECOND)

        time_result = round(end_time - start_time, 5)
        exec_per_sec = time_result / num_times

        # Extract m1, m2, p1, p2 from results
        p1 = client_results[0]["p1"]
        p2 = client_results[0]["p2"]
        m1 = server_results[0]["m1"]
        m2 = server_results[0]["m2"]

        avg_exec_per_sec.append(exec_per_sec)
        avg_succ_rate.append(succ_rate)
        avg_combined.append(succ_rate * exec_per_sec)

        logger['id'].append(i)
        logger['exec_per_sec'].append(exec_per_sec)
        logger['succ_rate'].append(succ_rate)
        logger['succ_per_sec'].append(succ_rate * exec_per_sec)
        logger['m1'].append(m1)
        logger['m2'].append(m2)
        logger['p1'].append(p1)
        logger['p2'].append(p2)
        logger[label].append(element)

        print(f'Executions per second: {exec_per_sec}')
        print(f'Success rate: {succ_rate}')
        print(f'Successes per second: {succ_rate * exec_per_sec}')
        print(f'm1: {m1}, m2: {m2}, p1: {p1}, p2: {p2}')

    plot_results(parameters_set, avg_succ_rate, avg_exec_per_sec, avg_combined, label, num_times)
    save_results(logger, stack, label, num_times)
    setattr(stack, label, old_parameter)

def plot_results(parameters_set, avg_succ_rate, avg_exec_per_sec, avg_combined, label, num_times):
    """Plot the results of the experiments."""
    fig, ax = pyplot.subplots(1, 3, figsize=(45, 15))

    ax[0].plot(parameters_set, avg_succ_rate)
    ax[0].set_xlabel(label)
    ax[0].set_ylabel('Success probability')

    ax[1].ticklabel_format(useOffset=False, style='plain', axis='y')
    ax[1].plot(parameters_set, avg_exec_per_sec)
    ax[1].set_xlabel(label)
    ax[1].set_ylabel('Executions per second')

    ax[2].ticklabel_format(useOffset=False, style='plain', axis='y')
    ax[2].plot(parameters_set, avg_combined)
    ax[2].set_xlabel(label)
    ax[2].set_ylabel('Successes per second')

    pyplot.savefig(f'./graphs/{num_times}_{label}_vs_succ_per_sec.png')
    pyplot.close()

def save_results(logger, stack, label, num_times):
    """Save the experiment results to a CSV file."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestr}_{stack.__class__.__name__}_{label}_{num_times}.csv"
    df = pd.DataFrame(logger)
    df.to_csv(filename)

def main():
    set_qstate_formalism(QFormalism.DM)
    num_times = 1000
    cfg_network_trapped_ions, cfg_network_colors_centers, ions_stack, colors_stack = load_configurations()
    parameters = setup_parameter_ranges()

    # Experiment with Color Centers - NV
    for label, (parameter, parameters_set) in parameters["color_centers"].items():
        experiment(colors_stack, cfg_network_colors_centers, parameter, parameters_set, label, num_times)

    # Experiment with Trapped Ions
    for label, (parameter, parameters_set) in parameters["trapped_ions"].items():
        experiment(ions_stack, cfg_network_trapped_ions, parameter, parameters_set, label, num_times)

if __name__ == "__main__":
    main()