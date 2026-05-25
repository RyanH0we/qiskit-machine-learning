"""VQE-VRPTW 示例的共用工具。

本文件只放轻量级基础设施：路径常量、JSON 读写、任务 banner、QUBO 到
Ising Hamiltonian 的转换，以及本地 estimator 创建函数。真实量子后端的
替换入口也集中在这里。
"""

from __future__ import annotations

import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEMO_ROOT / "artifacts"
INSTANCE_DIR = ARTIFACTS_DIR / "instance"
ROUTES_DIR = ARTIFACTS_DIR / "routes"
QUBO_DIR = ARTIFACTS_DIR / "qubo"
ANSATZ_DIR = ARTIFACTS_DIR / "ansatz"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

INSTANCE_JSON = INSTANCE_DIR / "instance.json"
ROUTES_JSON = ROUTES_DIR / "routes.json"
ROUTES_CSV = ROUTES_DIR / "routes.csv"
QUBO_JSON = QUBO_DIR / "qubo.json"
HAMILTONIAN_JSON = QUBO_DIR / "hamiltonian.json"
HAMILTONIAN_DILL = QUBO_DIR / "hamiltonian.dill"
REFERENCE_JSON = RESULTS_DIR / "reference.json"
ANSATZ_DILL = ANSATZ_DIR / "ansatz.dill"
ANSATZ_JSON = ANSATZ_DIR / "ansatz.json"
INITIAL_POINT_NPY = ANSATZ_DIR / "initial_point.npy"
INITIAL_RESULT_JSON = RESULTS_DIR / "initial_quantum.json"
VQE_RESULT_JSON = RESULTS_DIR / "vqe_result.json"
VQE_RESULT_DILL = RESULTS_DIR / "vqe_result.dill"
VQE_TRACE_CSV = RESULTS_DIR / "vqe_trace.csv"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"


class TaskKind(str, Enum):
    """任务类型：经典、量子、混合。"""

    CLASSICAL = "CLASSICAL"
    QUANTUM = "QUANTUM"
    HYBRID = "HYBRID"


_COLORS = {
    TaskKind.CLASSICAL: "\033[94m",
    TaskKind.QUANTUM: "\033[95m",
    TaskKind.HYBRID: "\033[93m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def print_banner(kind: TaskKind, name: str) -> None:
    """打印任务类型 banner，帮助新手区分当前步骤属于哪类计算。"""

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}")
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}")
    print(f"{color}{bold}{bar}{reset}")


class Timer:
    """简单计时器。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        print(f"  开始: {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._t0
        status = "完成" if exc_type is None else "失败"
        print(f"  [{status}: {self.label}, 用时 {self.elapsed:.2f}s]")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike, payload: dict[str, Any]) -> Path:
    out = ensure_parent_dir(path)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def create_estimator(seed: int | None = None):
    """创建量子期望值估计器。

    默认使用本地 ``StatevectorEstimator``，它是精确、无噪声的状态向量模拟器。
    以后要改成真实量子硬件时，可在这里替换为 IBM Runtime 的 EstimatorV2，
    并把 backend/session/shots 等参数从 CLI 传进来。
    """

    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator(seed=seed)


def bits_to_qiskit_bitstring(bits: list[int]) -> str:
    """把变量顺序 x0..xn-1 转成 Qiskit 概率字典使用的 bitstring。"""

    return "".join(str(bits[i]) for i in range(len(bits) - 1, -1, -1))


def qiskit_bitstring_to_bits(bitstring: str) -> list[int]:
    """把 Qiskit bitstring 转回变量顺序 x0..xn-1。"""

    return [int(bitstring[i]) for i in range(len(bitstring) - 1, -1, -1)]


def bits_label(bits: list[int]) -> str:
    """面向人阅读的变量顺序 bitstring。"""

    return "".join(str(int(b)) for b in bits)


def qubo_energy(
    bits: list[int],
    constant: float,
    linear: list[float],
    quadratic: list[list[float]],
) -> float:
    """计算 QUBO 能量：constant + sum a_i x_i + sum b_ij x_i x_j。"""

    value = float(constant)
    for i, bit in enumerate(bits):
        value += float(linear[i]) * bit
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            value += float(quadratic[i][j]) * bits[i] * bits[j]
    return value


def _pauli_label(num_qubits: int, z_indices: tuple[int, ...]) -> str:
    chars = ["I"] * num_qubits
    for index in z_indices:
        chars[num_qubits - 1 - index] = "Z"
    return "".join(chars)


def qubo_to_sparse_pauli(
    constant: float,
    linear: list[float],
    quadratic: list[list[float]],
):
    """把 QUBO 映射为 Ising Hamiltonian ``SparsePauliOp``。

    使用二进制变量和 Pauli-Z 的标准关系 ``x_i = (1 - Z_i) / 2``。
    返回值包含 identity offset，因此 Hamiltonian 的期望值直接等于 QUBO 目标值。
    """

    from qiskit.quantum_info import SparsePauliOp

    num_qubits = len(linear)
    coeffs: dict[str, float] = {}

    def add_term(z_indices: tuple[int, ...], coefficient: float) -> None:
        if abs(coefficient) < 1e-12:
            return
        label = _pauli_label(num_qubits, z_indices)
        coeffs[label] = coeffs.get(label, 0.0) + float(coefficient)

    add_term((), constant)
    for i, a_i in enumerate(linear):
        add_term((), a_i / 2.0)
        add_term((i,), -a_i / 2.0)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            b_ij = float(quadratic[i][j])
            if abs(b_ij) < 1e-12:
                continue
            add_term((), b_ij / 4.0)
            add_term((i,), -b_ij / 4.0)
            add_term((j,), -b_ij / 4.0)
            add_term((i, j), b_ij / 4.0)

    terms = [(label, coeff) for label, coeff in coeffs.items() if abs(coeff) > 1e-12]
    terms.sort(key=lambda item: (item[0] != "I" * num_qubits, item[0]))
    return SparsePauliOp.from_list(terms).simplify(atol=1e-12)


def pauli_records(operator, limit: int | None = None) -> list[dict[str, Any]]:
    """把 ``SparsePauliOp`` 转成 JSON 友好的 Pauli 项列表。"""

    rows: list[dict[str, Any]] = []
    for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs):
        rows.append(
            {
                "pauli": label,
                "coefficient": {"real": float(coeff.real), "imag": float(coeff.imag)},
                "abs_coefficient": float(abs(coeff)),
            }
        )
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows if limit is None else rows[:limit]


def summarize_state_probabilities(
    ansatz,
    parameters,
    qubo: dict[str, Any],
    routes: list[dict[str, Any]],
    top_n: int = 12,
) -> dict[str, Any]:
    """从参数化线路得到 bitstring 概率、路线概率和最高概率状态。"""

    import numpy as np
    from qiskit.quantum_info import Statevector

    bound = ansatz.assign_parameters(np.asarray(parameters, dtype=float), inplace=False)
    statevector = Statevector.from_instruction(bound)
    probabilities = statevector.probabilities_dict()

    rows: list[dict[str, Any]] = []
    route_probabilities = {route["route_id"]: 0.0 for route in routes}
    invalid_probability = 0.0
    for qiskit_bitstring, probability in probabilities.items():
        bits = qiskit_bitstring_to_bits(qiskit_bitstring)
        selected = [i for i, bit in enumerate(bits) if bit == 1]
        energy = qubo_energy(bits, qubo["constant"], qubo["linear"], qubo["quadratic"])
        route_id = None
        route_label = None
        if len(selected) == 1:
            route = routes[selected[0]]
            route_id = route["route_id"]
            route_label = route["label"]
            route_probabilities[route_id] += float(probability)
        else:
            invalid_probability += float(probability)
        rows.append(
            {
                "bits": bits,
                "bitstring": bits_label(bits),
                "qiskit_bitstring": qiskit_bitstring,
                "probability": float(probability),
                "energy": float(energy),
                "num_selected_routes": len(selected),
                "is_one_hot": len(selected) == 1,
                "selected_route_id": route_id,
                "selected_route_label": route_label,
            }
        )

    rows.sort(key=lambda row: row["probability"], reverse=True)
    route_rows = [
        {
            "route_id": route["route_id"],
            "route_label": route["label"],
            "probability": route_probabilities[route["route_id"]],
            "route_cost": float(route["cost"]),
            "is_time_window_feasible": bool(route["is_time_window_feasible"]),
        }
        for route in routes
    ]
    route_rows.sort(key=lambda row: row["probability"], reverse=True)

    return {
        "top_states": rows[:top_n],
        "all_states": rows,
        "route_probabilities": route_rows,
        "invalid_probability": invalid_probability,
        "best_probability_state": rows[0],
        "best_probability_route": next((row for row in route_rows if row["probability"] > 0), None),
    }
