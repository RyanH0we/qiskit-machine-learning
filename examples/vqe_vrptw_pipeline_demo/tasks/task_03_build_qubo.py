"""[CLASSICAL] Task 03 -- 构造路线选择 QUBO 并映射为 Ising Hamiltonian。

每个二进制变量 x_r 表示“是否选择第 r 条完整路线”。one-hot 约束
``(sum_r x_r - 1)^2`` 保证最终只选择一条路线。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    ensure_parent_dir,
    pauli_records,
    print_banner,
    qubo_to_sparse_pauli,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", type=Path, required=True, help="task_02 输出的 routes.json")
    p.add_argument("--qubo-output", type=Path, required=True, help="输出 qubo.json")
    p.add_argument("--hamiltonian-output", type=Path, required=True, help="输出 hamiltonian.json")
    p.add_argument("--hamiltonian-dill", type=Path, required=True, help="输出 hamiltonian.dill")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 QUBO/Hamiltonian 可视化 PNG")
    p.add_argument("--penalty", type=float, default=0.0, help="one-hot 惩罚；<=0 表示自动设置")
    return p.parse_args()


def _build_route_selection_qubo(routes_payload: dict, penalty: float) -> dict:
    routes = routes_payload["routes"]
    route_costs = [float(route["cost"]) for route in routes]
    cost_scale = max(route_costs)
    normalized_costs = [cost / cost_scale for cost in route_costs]
    if penalty <= 0:
        penalty = 2.0

    num_variables = len(routes)
    constant = float(penalty)
    linear = [float(cost - penalty) for cost in normalized_costs]
    quadratic = [[0.0 for _ in range(num_variables)] for _ in range(num_variables)]
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            quadratic[i][j] = 2.0 * penalty

    return {
        "description": "route-selection QUBO for a one-vehicle VRPTW toy instance",
        "objective": "min sum(normalized_route_cost[r] * x_r) + penalty * (sum(x_r) - 1)^2",
        "num_variables": num_variables,
        "cost_scale": float(cost_scale),
        "cost_units": "normalized route cost; multiply one-hot route energy by cost_scale to recover route-cost units",
        "variables": [
            {
                "index": i,
                "name": f"x_{route['route_id']}",
                "route_id": route["route_id"],
                "route_label": route["label"],
                "route_cost": float(route["cost"]),
                "route_cost_normalized": float(normalized_costs[i]),
            }
            for i, route in enumerate(routes)
        ],
        "penalty": float(penalty),
        "constant": constant,
        "linear": linear,
        "quadratic": quadratic,
    }


def _plot_qubo_and_hamiltonian(qubo: dict, hamiltonian: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    num_variables = int(qubo["num_variables"])
    matrix = np.zeros((num_variables, num_variables))
    for i, value in enumerate(qubo["linear"]):
        matrix[i, i] = value
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            matrix[i, j] = qubo["quadratic"][i][j]
            matrix[j, i] = qubo["quadratic"][i][j]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_matrix, ax_terms = axes

    im = ax_matrix.imshow(matrix, cmap="coolwarm")
    labels = [var["route_id"] for var in qubo["variables"]]
    ax_matrix.set_xticks(np.arange(num_variables))
    ax_matrix.set_yticks(np.arange(num_variables))
    ax_matrix.set_xticklabels(labels, rotation=45, ha="right")
    ax_matrix.set_yticklabels(labels)
    for i in range(num_variables):
        for j in range(num_variables):
            ax_matrix.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax_matrix.set_title("QUBO coefficients")
    fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

    top_terms = hamiltonian["pauli_terms"][: min(12, len(hamiltonian["pauli_terms"]))]
    paulis = [term["pauli"] for term in top_terms]
    coeffs = [term["coefficient"]["real"] for term in top_terms]
    y = np.arange(len(top_terms))
    ax_terms.barh(y, coeffs, color=["#2ca25f" if c >= 0 else "#de2d26" for c in coeffs], edgecolor="black", linewidth=0.5)
    ax_terms.axvline(0, color="black", linewidth=0.8)
    ax_terms.set_yticks(y)
    ax_terms.set_yticklabels(paulis, fontsize=8)
    ax_terms.invert_yaxis()
    ax_terms.set_xlabel("coefficient")
    ax_terms.set_title("Largest Ising Pauli terms")
    ax_terms.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Task 03: QUBO to Ising Hamiltonian")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 03 / 构造 QUBO 和 Ising Hamiltonian")
    routes_payload = read_json(args.routes)

    with Timer("构造 route-selection QUBO"):
        qubo = _build_route_selection_qubo(routes_payload, args.penalty)
    print(f"  变量数 = {qubo['num_variables']}")
    print(f"  one-hot penalty = {qubo['penalty']:.3f}")

    with Timer("QUBO -> Pauli-Z Hamiltonian"):
        operator = qubo_to_sparse_pauli(qubo["constant"], qubo["linear"], qubo["quadratic"])
        hamiltonian = {
            "description": "Ising Hamiltonian equivalent to the route-selection QUBO",
            "num_qubits": qubo["num_variables"],
            "num_pauli_terms": int(len(operator)),
            "pauli_terms": pauli_records(operator),
        }
    print(f"  qubit 数 = {hamiltonian['num_qubits']}")
    print(f"  Pauli 项数 = {hamiltonian['num_pauli_terms']}")

    qubo_out = write_json(args.qubo_output, qubo)
    print(f"  -> 写入 {qubo_out}")

    ham_out = write_json(args.hamiltonian_output, hamiltonian)
    print(f"  -> 写入 {ham_out}")

    dill_out = ensure_parent_dir(args.hamiltonian_dill)
    with dill_out.open("wb") as f:
        dill.dump({"operator": operator, "qubo": qubo, "hamiltonian": hamiltonian}, f)
    print(f"  -> 写入 {dill_out}")

    with Timer("绘制 QUBO 矩阵和 Pauli 项"):
        _plot_qubo_and_hamiltonian(qubo, hamiltonian, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
