"""[CLASSICAL] Task 02 -- 手写 JSSP 的 QUBO 并映射到 Ising Hamiltonian。

这一步把调度问题翻译成量子比特可以处理的能量函数。所有变量、惩罚项和
Pauli 项都显式保存，方便新手检查“经典建模”到底做了什么。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    complex_to_record,
    ensure_parent_dir,
    print_banner,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--penalty", type=float, default=10.0, help="违反约束时加入的 QUBO 惩罚权重")
    p.add_argument("--qubo-output", type=Path, required=True, help="输出 qubo.json")
    p.add_argument("--hamiltonian-output", type=Path, required=True, help="输出 hamiltonian.dill")
    p.add_argument("--metadata-output", type=Path, required=True, help="输出 hamiltonian.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 QUBO 矩阵热力图 PNG")
    return p.parse_args()


class QuboBuilder:
    """小型 QUBO 构造器，统一处理 x_i^2 = x_i。"""

    def __init__(self) -> None:
        self.offset = 0.0
        self.linear: defaultdict[int, float] = defaultdict(float)
        self.quadratic: defaultdict[tuple[int, int], float] = defaultdict(float)

    def add_linear(self, i: int, coeff: float) -> None:
        self.linear[int(i)] += float(coeff)

    def add_quadratic(self, i: int, j: int, coeff: float) -> None:
        i = int(i)
        j = int(j)
        coeff = float(coeff)
        if i == j:
            self.add_linear(i, coeff)
            return
        if i > j:
            i, j = j, i
        self.quadratic[(i, j)] += coeff

    def add_square_constraint(self, expr: dict[int, float], rhs: float, weight: float) -> None:
        """加入 ``weight * (sum_i a_i x_i - rhs)^2``。"""

        rhs = float(rhs)
        weight = float(weight)
        self.offset += weight * rhs * rhs
        for i, a_i in expr.items():
            self.add_linear(i, weight * (a_i * a_i - 2.0 * rhs * a_i))
        for (i, a_i), (j, a_j) in combinations(expr.items(), 2):
            self.add_quadratic(i, j, weight * 2.0 * a_i * a_j)


def _operation_windows(instance: dict[str, Any]) -> dict[tuple[int, int], tuple[int, int]]:
    horizon = int(instance["horizon"])
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    for job in instance["jobs"]:
        ops = job["operations"]
        prefix = 0
        suffix_by_op: dict[int, int] = {}
        remaining = 0
        for op in reversed(ops):
            remaining += int(op["duration"])
            suffix_by_op[int(op["operation"])] = remaining
        for op in ops:
            key = (int(job["job"]), int(op["operation"]))
            earliest = prefix
            latest = horizon - suffix_by_op[int(op["operation"])]
            if latest < earliest:
                raise ValueError(f"工序 J{key[0]}-O{key[1]} 没有可行开始窗口")
            windows[key] = (earliest, latest)
            prefix += int(op["duration"])
    return windows


def _machine_load_lower_bound(instance: dict[str, Any]) -> int:
    loads = {int(m): 0 for m in instance["machines"]}
    for op in instance["operations"]:
        loads[int(op["machine"])] += int(op["duration"])
    return max(loads.values())


def _job_duration_lower_bound(instance: dict[str, Any]) -> int:
    return max(sum(int(op["duration"]) for op in job["operations"]) for job in instance["jobs"])


def _make_variables(instance: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int], int], list[int]]:
    variables: list[dict[str, Any]] = []
    start_index: dict[tuple[int, int, int], int] = {}
    windows = _operation_windows(instance)

    for op in instance["operations"]:
        job = int(op["job"])
        operation = int(op["operation"])
        earliest, latest = windows[(job, operation)]
        for start in range(earliest, latest + 1):
            idx = len(variables)
            start_index[(job, operation, start)] = idx
            variables.append(
                {
                    "index": idx,
                    "name": f"x_j{job}_o{operation}_t{start}",
                    "kind": "start",
                    "job": job,
                    "operation": operation,
                    "machine": int(op["machine"]),
                    "duration": int(op["duration"]),
                    "start": start,
                    "finish": start + int(op["duration"]),
                }
            )

    lower_bound = max(_job_duration_lower_bound(instance), _machine_load_lower_bound(instance))
    cmax_values = list(range(lower_bound, int(instance["horizon"]) + 1))
    for makespan in cmax_values:
        idx = len(variables)
        variables.append(
            {
                "index": idx,
                "name": f"cmax_{makespan}",
                "kind": "cmax",
                "makespan": makespan,
            }
        )
    return variables, start_index, cmax_values


def _indices_for_operation(
    variables: list[dict[str, Any]],
    job: int,
    operation: int,
) -> list[int]:
    return [
        int(v["index"])
        for v in variables
        if v["kind"] == "start" and int(v["job"]) == job and int(v["operation"]) == operation
    ]


def _intervals_overlap(a_start: int, a_duration: int, b_start: int, b_duration: int) -> bool:
    return a_start < b_start + b_duration and b_start < a_start + a_duration


def _build_qubo(instance: dict[str, Any], penalty: float) -> dict[str, Any]:
    variables, _, cmax_values = _make_variables(instance)
    builder = QuboBuilder()
    constraint_counts: defaultdict[str, int] = defaultdict(int)

    # 目标：选择的 Cmax 越小，能量越低。
    for var in variables:
        if var["kind"] == "cmax":
            builder.add_linear(int(var["index"]), float(var["makespan"]))

    # 每道工序必须且只能选择一个开始时间。
    for op in instance["operations"]:
        indices = _indices_for_operation(variables, int(op["job"]), int(op["operation"]))
        builder.add_square_constraint({i: 1.0 for i in indices}, rhs=1.0, weight=penalty)
        constraint_counts["operation_starts_once"] += 1

    # Cmax 也必须且只能选择一个。
    cmax_indices = [int(v["index"]) for v in variables if v["kind"] == "cmax"]
    builder.add_square_constraint({i: 1.0 for i in cmax_indices}, rhs=1.0, weight=penalty)
    constraint_counts["cmax_chosen_once"] += 1

    # 同一作业内部，后一工序不能早于前一工序完成。
    by_index = {int(v["index"]): v for v in variables}
    for job in instance["jobs"]:
        ops = job["operations"]
        for prev, nxt in zip(ops, ops[1:]):
            prev_indices = _indices_for_operation(variables, int(job["job"]), int(prev["operation"]))
            next_indices = _indices_for_operation(variables, int(job["job"]), int(nxt["operation"]))
            for i in prev_indices:
                prev_start = int(by_index[i]["start"])
                prev_finish = prev_start + int(prev["duration"])
                for j in next_indices:
                    next_start = int(by_index[j]["start"])
                    if next_start < prev_finish:
                        builder.add_quadratic(i, j, penalty)
                        constraint_counts["precedence_invalid_pairs"] += 1

    # 同一机器上的任意两道工序不能重叠。
    ops = instance["operations"]
    for left, right in combinations(ops, 2):
        if int(left["machine"]) != int(right["machine"]):
            continue
        left_indices = _indices_for_operation(variables, int(left["job"]), int(left["operation"]))
        right_indices = _indices_for_operation(variables, int(right["job"]), int(right["operation"]))
        for i in left_indices:
            l_start = int(by_index[i]["start"])
            for j in right_indices:
                r_start = int(by_index[j]["start"])
                if _intervals_overlap(l_start, int(left["duration"]), r_start, int(right["duration"])):
                    builder.add_quadratic(i, j, penalty)
                    constraint_counts["machine_overlap_invalid_pairs"] += 1

    # 所有工序都必须在选中的 Cmax 前完成；对最终工序而言这就是 makespan 约束。
    for var in variables:
        if var["kind"] != "start":
            continue
        finish = int(var["finish"])
        for cmax_var in variables:
            if cmax_var["kind"] != "cmax":
                continue
            if finish > int(cmax_var["makespan"]):
                builder.add_quadratic(int(var["index"]), int(cmax_var["index"]), penalty)
                constraint_counts["finish_after_cmax_invalid_pairs"] += 1

    linear_terms = [
        {
            "index": int(i),
            "name": variables[int(i)]["name"],
            "coefficient": float(coeff),
        }
        for i, coeff in sorted(builder.linear.items())
        if abs(coeff) > 1e-12
    ]
    quadratic_terms = [
        {
            "i": int(i),
            "j": int(j),
            "name_i": variables[int(i)]["name"],
            "name_j": variables[int(j)]["name"],
            "coefficient": float(coeff),
        }
        for (i, j), coeff in sorted(builder.quadratic.items())
        if abs(coeff) > 1e-12
    ]

    return {
        "description": "JSSP 时间索引 QUBO；低能量代表更短且更少违规的排程",
        "penalty": float(penalty),
        "horizon": int(instance["horizon"]),
        "lower_bound_makespan": int(cmax_values[0]),
        "candidate_makespans": cmax_values,
        "num_variables": len(variables),
        "variables": variables,
        "offset": float(builder.offset),
        "linear": linear_terms,
        "quadratic": quadratic_terms,
        "constraint_counts": dict(sorted(constraint_counts.items())),
    }


def _qubo_to_ising(qubo: dict[str, Any]):
    from qiskit.quantum_info import SparsePauliOp

    n = int(qubo["num_variables"])
    offset = float(qubo["offset"])
    z_terms: defaultdict[int, float] = defaultdict(float)
    zz_terms: defaultdict[tuple[int, int], float] = defaultdict(float)

    for term in qubo["linear"]:
        i = int(term["index"])
        coeff = float(term["coefficient"])
        offset += coeff / 2.0
        z_terms[i] += -coeff / 2.0

    for term in qubo["quadratic"]:
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coefficient"])
        offset += coeff / 4.0
        z_terms[i] += -coeff / 4.0
        z_terms[j] += -coeff / 4.0
        if i > j:
            i, j = j, i
        zz_terms[(i, j)] += coeff / 4.0

    pauli_terms: list[tuple[str, complex]] = []
    identity = "I" * n
    pauli_terms.append((identity, complex(offset)))
    for i, coeff in sorted(z_terms.items()):
        if abs(coeff) <= 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        pauli_terms.append(("".join(label), complex(coeff)))
    for (i, j), coeff in sorted(zz_terms.items()):
        if abs(coeff) <= 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        label[n - 1 - j] = "Z"
        pauli_terms.append(("".join(label), complex(coeff)))

    return SparsePauliOp.from_list(pauli_terms).simplify(atol=1e-12)


def _plot_qubo_matrix(qubo: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = int(qubo["num_variables"])
    matrix = np.zeros((n, n), dtype=float)
    for term in qubo["linear"]:
        i = int(term["index"])
        matrix[i, i] += float(term["coefficient"])
    for term in qubo["quadratic"]:
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coefficient"])
        matrix[i, j] += coeff
        matrix[j, i] += coeff

    out = ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    vmax = max(1.0, float(np.max(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    labels = [v["name"].replace("_", "\n") for v in qubo["variables"]]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("QUBO coefficient matrix: diagonal = linear, off-diagonal = quadratic")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 02 / 构造 QUBO 与 Ising Hamiltonian")
    instance = read_json(args.instance)
    print(f"  penalty = {args.penalty}")
    print(f"  horizon = {instance['horizon']}")

    with Timer("手写时间索引 QUBO"):
        qubo = _build_qubo(instance, args.penalty)
    print(f"  QUBO 变量数 = {qubo['num_variables']}")
    print(f"  线性项数 = {len(qubo['linear'])}, 二次项数 = {len(qubo['quadratic'])}")
    print(f"  Cmax 候选 = {qubo['candidate_makespans']}")

    with Timer("QUBO 映射到 Ising Hamiltonian"):
        qubit_operator = _qubo_to_ising(qubo)
    labels = qubit_operator.paulis.to_labels()
    coeffs = qubit_operator.coeffs
    order = np.argsort(-np.abs(coeffs))
    pauli_terms = [
        {
            "pauli": labels[i],
            "coefficient": complex_to_record(complex(coeffs[i])),
            "abs_coefficient": float(abs(coeffs[i])),
        }
        for i in order
    ]
    metadata = {
        "mapping": "x=(1-Z)/2",
        "num_qubits": int(qubit_operator.num_qubits),
        "num_pauli_terms": int(len(qubit_operator)),
        "pauli_terms": pauli_terms,
        "qubo": {
            "num_variables": qubo["num_variables"],
            "offset": qubo["offset"],
            "penalty": qubo["penalty"],
            "lower_bound_makespan": qubo["lower_bound_makespan"],
            "candidate_makespans": qubo["candidate_makespans"],
            "variables": qubo["variables"],
        },
    }
    print(f"  qubit 数 = {metadata['num_qubits']}, Pauli 项数 = {metadata['num_pauli_terms']}")

    qubo_out = write_json(args.qubo_output, qubo)
    print(f"  -> 写入 {qubo_out}")

    hamiltonian_out = ensure_parent_dir(args.hamiltonian_output)
    with hamiltonian_out.open("wb") as f:
        dill.dump({"operator": qubit_operator, "qubo": qubo, "metadata": metadata}, f)
    print(f"  -> 写入 {hamiltonian_out}")

    metadata_out = write_json(args.metadata_output, metadata)
    print(f"  -> 写入 {metadata_out}")

    with Timer("绘制 QUBO 矩阵热力图"):
        _plot_qubo_matrix(qubo, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
