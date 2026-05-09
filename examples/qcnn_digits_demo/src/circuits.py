"""QCNN 量子电路构件。

接口与 docs/tutorials/11_quantum_convolutional_neural_networks.ipynb 完全一致：
  - conv_circuit(params): 2 比特参数化幺正
  - conv_layer(num_qubits, prefix): 卷积层（环形耦合）
  - pool_circuit(params): 2 比特池化幺正
  - pool_layer(sources, sinks, prefix): 把 sources 比特"折叠"到 sinks 比特

参考: A. Cong, S. Choi, and M. D. Lukin, "Quantum convolutional neural networks",
Nature Physics 15, 1273-1278 (2019).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def conv_circuit(params) -> QuantumCircuit:
    """两比特参数化卷积单元，含 3 个可训练参数。"""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """卷积层：先对偶数对应用 conv_circuit，再对奇数对环形应用一次。

    总参数数 = num_qubits * 3。
    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    wrapped = QuantumCircuit(num_qubits)
    wrapped.append(qc_inst, qubits)
    return wrapped


def pool_circuit(params) -> QuantumCircuit:
    """两比特池化单元：把 source 信息编码进 sink，然后忽略 source。"""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """池化层：将 sources 中的比特折叠到 sinks 上。

    总参数数 = len(sources) * 3 = (num_qubits // 2) * 3。
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    wrapped = QuantumCircuit(num_qubits)
    wrapped.append(qc_inst, range(num_qubits))
    return wrapped
