"""组装 8 比特 QCNN 并封装成 EstimatorQNN。

电路拓扑：
    z_feature_map(8, reps=2)     # 角度编码（默认; 教程 11 同款）
    -> conv1 (8) -> pool1 8->4
    -> conv2 (4) -> pool2 4->2
    -> conv3 (2) -> pool3 2->1
    -> 测量末位比特的 Z 期望

输出范围 [-1, 1]，正好可以让 label {-1, +1} 直接做平方误差回归式分类。
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN

from .circuits import conv_layer, pool_layer


@dataclass
class QCNNBundle:
    """build_qcnn 的返回包，包括 QNN 与各层引用，便于可视化和调试。"""

    qnn: EstimatorQNN
    full_circuit: QuantumCircuit
    feature_map: QuantumCircuit
    ansatz: QuantumCircuit
    observable: SparsePauliOp


def build_qcnn(
    num_qubits: int = 8,
    feature_map_kind: str = "z",
) -> QCNNBundle:
    """构建一个 num_qubits 比特的 QCNN（默认 8）。

    Args:
        num_qubits: 比特数，必须是 2 的幂（8 -> 4 -> 2 -> 1 这样能折叠到 1）。
        feature_map_kind: ``"z"`` (默认, 教程 11 用的纯单比特角度编码) 或
            ``"zz"`` (二阶纠缠编码，表达力更强但参数空间更复杂，对收敛不一定友好)。
    """
    if num_qubits != 8:
        raise NotImplementedError(
            "本 demo 的 QCNN 拓扑硬编码为 8 比特 (8->4->2->1)。"
            "若要换比特数，请同步修改 ansatz 的层结构。"
        )

    if feature_map_kind == "zz":
        feature_map = zz_feature_map(num_qubits, reps=2, entanglement="linear")
    elif feature_map_kind == "z":
        feature_map = z_feature_map(num_qubits, reps=2)
    else:
        raise ValueError(f"未知 feature_map_kind: {feature_map_kind!r}")

    ansatz = QuantumCircuit(num_qubits, name="QCNN_ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, range(num_qubits), inplace=True)
    full_circuit.compose(ansatz, range(num_qubits), inplace=True)

    # 末位比特上的 Z 算符：8 比特拼成 "Z" + "I"*7（Qiskit 里左 high-index）
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)])

    qnn = EstimatorQNN(
        circuit=full_circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return QCNNBundle(
        qnn=qnn,
        full_circuit=full_circuit,
        feature_map=feature_map,
        ansatz=ansatz,
        observable=observable,
    )
