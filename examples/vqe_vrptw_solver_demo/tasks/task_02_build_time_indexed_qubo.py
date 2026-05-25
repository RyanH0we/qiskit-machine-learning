"""[CLASSICAL] Task 02 -- 构造时间索引 VRPTW-QUBO 并映射为 Ising Hamiltonian。

本任务直接建模 VRPTW 的车辆、访问序位、客户和服务开始时间，不枚举完整
路线池。所有变量和约束都会写入 JSON，方便新手逐项检查。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    QuboBuilder,
    TaskKind,
    Timer,
    ensure_parent_dir,
    matrix_value,
    pauli_records,
    print_banner,
    qubo_to_ising,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--penalty", type=float, default=0.0, help="约束惩罚；<=0 表示自动设置")
    p.add_argument("--qubo-output", type=Path, required=True, help="输出 qubo.json")
    p.add_argument("--hamiltonian-output", type=Path, required=True, help="输出 hamiltonian.json")
    p.add_argument("--hamiltonian-dill", type=Path, required=True, help="输出 hamiltonian.dill")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 QUBO/Hamiltonian 可视化 PNG")
    return p.parse_args()


def _start_candidates(customer: dict[str, Any], granularity: float) -> list[float]:
    ready, due = [float(x) for x in customer["time_window"]]
    first = math.ceil(ready / granularity - 1e-12)
    last = math.floor(due / granularity + 1e-12)
    return [round(k * granularity, 10) for k in range(first, last + 1)]


def _make_variables(instance: dict[str, Any]) -> list[dict[str, Any]]:
    variables: list[dict[str, Any]] = []
    granularity = float(instance["solver_config"]["time_granularity"])
    max_positions = int(instance["solver_config"]["max_stops_per_vehicle"])

    for vehicle_index, vehicle in enumerate(instance["vehicles"]):
        for position in range(max_positions):
            for customer in instance["customers"]:
                starts = _start_candidates(customer, granularity)
                if not starts:
                    raise ValueError(f"客户 {customer['id']} 在当前时间粒度下没有可选服务开始时间")
                for start_time in starts:
                    idx = len(variables)
                    variables.append(
                        {
                            "index": idx,
                            "name": f"x_v{vehicle_index}_p{position}_{customer['id']}_t{start_time:g}",
                            "kind": "x",
                            "vehicle": vehicle_index,
                            "vehicle_id": vehicle["id"],
                            "position": position,
                            "customer_id": customer["id"],
                            "start_time": float(start_time),
                            "demand": float(customer.get("demand", 0.0)),
                            "service_time": float(customer["service_time"]),
                        }
                    )
            idx = len(variables)
            variables.append(
                {
                    "index": idx,
                    "name": f"y_v{vehicle_index}_p{position}",
                    "kind": "y",
                    "vehicle": vehicle_index,
                    "vehicle_id": vehicle["id"],
                    "position": position,
                }
            )

    if max_positions > 2:
        for vehicle_index, vehicle in enumerate(instance["vehicles"]):
            capacity = int(vehicle["capacity"])
            num_bits = max(1, math.ceil(math.log2(capacity + 1)))
            for bit in range(num_bits):
                idx = len(variables)
                variables.append(
                    {
                        "index": idx,
                        "name": f"cap_slack_v{vehicle_index}_b{bit}",
                        "kind": "capacity_slack",
                        "vehicle": vehicle_index,
                        "vehicle_id": vehicle["id"],
                        "weight": 2**bit,
                    }
                )
    return variables


def _group_variables(variables: list[dict[str, Any]]) -> dict[str, Any]:
    x_vars = [v for v in variables if v["kind"] == "x"]
    y_vars = [v for v in variables if v["kind"] == "y"]
    slack_vars = [v for v in variables if v["kind"] == "capacity_slack"]
    by_customer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slot: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_vehicle: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    y_by_slot: dict[tuple[int, int], dict[str, Any]] = {}
    slack_by_vehicle: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for var in x_vars:
        by_customer[str(var["customer_id"])].append(var)
        by_slot[(int(var["vehicle"]), int(var["position"]))].append(var)
        by_vehicle[int(var["vehicle"])].append(var)
    for var in y_vars:
        y_by_slot[(int(var["vehicle"]), int(var["position"]))] = var
    for var in slack_vars:
        slack_by_vehicle[int(var["vehicle"])].append(var)
    return {
        "x": x_vars,
        "y": y_vars,
        "slack": slack_vars,
        "by_customer": by_customer,
        "by_slot": by_slot,
        "by_vehicle": by_vehicle,
        "y_by_slot": y_by_slot,
        "slack_by_vehicle": slack_by_vehicle,
    }


def _distance_scale(instance: dict[str, Any]) -> float:
    nodes = [instance["depot"]["id"], *[customer["id"] for customer in instance["customers"]]]
    distances = [
        matrix_value(instance, "distance_matrix", src, dst)
        for src in nodes
        for dst in nodes
        if src != dst
    ]
    return max(1.0, max(distances))


def _build_qubo(instance: dict[str, Any], penalty_arg: float) -> dict[str, Any]:
    variables = _make_variables(instance)
    groups = _group_variables(variables)
    builder = QuboBuilder()
    counts: defaultdict[str, int] = defaultdict(int)
    depot_id = instance["depot"]["id"]
    max_positions = int(instance["solver_config"]["max_stops_per_vehicle"])
    distance_scale = _distance_scale(instance)
    penalty = float(penalty_arg) if penalty_arg > 0 else 10.0

    # 目标：真实路径距离 / distance_scale。返回 depot 的成本用 y 变量判断谁是最后一个 stop。
    for var in groups["x"]:
        vehicle = int(var["vehicle"])
        position = int(var["position"])
        cid = str(var["customer_id"])
        idx = int(var["index"])
        if position == 0:
            builder.add_linear(idx, matrix_value(instance, "distance_matrix", depot_id, cid) / distance_scale)
            counts["objective_depot_to_first_terms"] += 1
        if position == max_positions - 1:
            builder.add_linear(idx, matrix_value(instance, "distance_matrix", cid, depot_id) / distance_scale)
            counts["objective_return_terms"] += 1
        else:
            next_y = groups["y_by_slot"][(vehicle, position + 1)]
            return_cost = matrix_value(instance, "distance_matrix", cid, depot_id) / distance_scale
            builder.add_linear(idx, return_cost)
            builder.add_quadratic(idx, int(next_y["index"]), -return_cost)
            counts["objective_conditional_return_terms"] += 1

    for vehicle_index in range(len(instance["vehicles"])):
        for position in range(max_positions - 1):
            left_vars = groups["by_slot"][(vehicle_index, position)]
            right_vars = groups["by_slot"][(vehicle_index, position + 1)]
            for left in left_vars:
                for right in right_vars:
                    coeff = matrix_value(
                        instance,
                        "distance_matrix",
                        str(left["customer_id"]),
                        str(right["customer_id"]),
                    ) / distance_scale
                    builder.add_quadratic(int(left["index"]), int(right["index"]), coeff)
                    counts["objective_between_customer_terms"] += 1

    # 每个客户必须且只能服务一次。
    for customer in instance["customers"]:
        expr = {int(var["index"]): 1.0 for var in groups["by_customer"][customer["id"]]}
        builder.add_square_constraint(expr, rhs=1.0, weight=penalty)
        counts["customer_served_once_constraints"] += 1

    # 每个 slot 的 x 选择数量必须等于 y。
    for slot, x_vars in groups["by_slot"].items():
        y_var = groups["y_by_slot"][slot]
        expr = {int(var["index"]): 1.0 for var in x_vars}
        expr[int(y_var["index"])] = -1.0
        builder.add_square_constraint(expr, rhs=0.0, weight=penalty)
        counts["slot_occupancy_link_constraints"] += 1

    # 路线不能有空洞：后一个 slot 被占用时，前一个 slot 也必须被占用。
    for vehicle_index in range(len(instance["vehicles"])):
        for position in range(1, max_positions):
            y_now = groups["y_by_slot"][(vehicle_index, position)]
            y_prev = groups["y_by_slot"][(vehicle_index, position - 1)]
            builder.add_implication_not(int(y_now["index"]), int(y_prev["index"]), penalty)
            counts["no_hole_constraints"] += 1

    # 起点 depot 可达，服务后能返回 depot。
    depot_due = float(instance["depot"]["time_window"][1])
    for var in groups["x"]:
        vehicle = instance["vehicles"][int(var["vehicle"])]
        cid = str(var["customer_id"])
        start = float(var["start_time"])
        service = float(var["service_time"])
        from_depot = float(vehicle.get("start_time", 0.0)) + matrix_value(instance, "travel_time_matrix", depot_id, cid)
        if start + 1e-9 < from_depot:
            builder.add_linear(int(var["index"]), penalty)
            counts["depot_departure_invalid_terms"] += 1
        if start + service + matrix_value(instance, "travel_time_matrix", cid, depot_id) > depot_due + 1e-9:
            builder.add_linear(int(var["index"]), penalty)
            counts["depot_return_invalid_terms"] += 1

    # 同车相邻访问必须满足服务时间和 travel time。
    for vehicle_index in range(len(instance["vehicles"])):
        for position in range(max_positions - 1):
            for left in groups["by_slot"][(vehicle_index, position)]:
                for right in groups["by_slot"][(vehicle_index, position + 1)]:
                    earliest_next = (
                        float(left["start_time"])
                        + float(left["service_time"])
                        + matrix_value(instance, "travel_time_matrix", str(left["customer_id"]), str(right["customer_id"]))
                    )
                    if float(right["start_time"]) + 1e-9 < earliest_next:
                        builder.add_quadratic(int(left["index"]), int(right["index"]), penalty)
                        counts["time_precedence_invalid_pairs"] += 1

    # 容量约束：默认 max_stops=2 时用精确非法二次对；更大时自动加入 slack bits。
    if max_positions <= 2:
        for vehicle_index, vehicle in enumerate(instance["vehicles"]):
            cap = float(vehicle["capacity"])
            vehicle_x = groups["by_vehicle"][vehicle_index]
            for var in vehicle_x:
                if float(var["demand"]) > cap + 1e-9:
                    builder.add_linear(int(var["index"]), penalty)
                    counts["capacity_single_invalid_terms"] += 1
            for left, right in combinations(vehicle_x, 2):
                if int(left["position"]) == int(right["position"]):
                    continue
                if float(left["demand"]) + float(right["demand"]) > cap + 1e-9:
                    builder.add_quadratic(int(left["index"]), int(right["index"]), penalty)
                    counts["capacity_pair_invalid_terms"] += 1
    else:
        for vehicle_index, vehicle in enumerate(instance["vehicles"]):
            expr = {int(var["index"]): float(var["demand"]) for var in groups["by_vehicle"][vehicle_index]}
            for slack in groups["slack_by_vehicle"][vehicle_index]:
                expr[int(slack["index"])] = float(slack["weight"])
            builder.add_square_constraint(expr, rhs=float(vehicle["capacity"]), weight=penalty)
            counts["capacity_slack_constraints"] += 1

    linear_terms = [
        {"index": int(i), "name": variables[int(i)]["name"], "coefficient": float(coeff)}
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
        "description": "time-indexed VRPTW QUBO; x[v,p,c,t] chooses service, y[v,p] marks occupied route slots",
        "objective": "minimize normalized total route distance with penalties for VRPTW constraint violations",
        "penalty": float(penalty),
        "penalty_mode": "auto" if penalty_arg <= 0 else "manual",
        "distance_scale": float(distance_scale),
        "num_variables": len(variables),
        "num_x_variables": len(groups["x"]),
        "num_y_variables": len(groups["y"]),
        "num_capacity_slack_variables": len(groups["slack"]),
        "variables": variables,
        "offset": float(builder.offset),
        "linear": linear_terms,
        "quadratic": quadratic_terms,
        "constraint_counts": dict(sorted(counts.items())),
    }


def _plot_qubo(qubo: dict[str, Any], hamiltonian: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = int(qubo["num_variables"])
    matrix = np.zeros((n, n), dtype=float)
    for term in qubo["linear"]:
        matrix[int(term["index"]), int(term["index"])] += float(term["coefficient"])
    for term in qubo["quadratic"]:
        i = int(term["i"])
        j = int(term["j"])
        matrix[i, j] += float(term["coefficient"])
        matrix[j, i] += float(term["coefficient"])

    out = ensure_parent_dir(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    ax_matrix, ax_terms = axes

    vmax = max(1.0, float(np.max(np.abs(matrix))))
    im = ax_matrix.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    labels = [var["name"].replace("_", "\n") for var in qubo["variables"]]
    ax_matrix.set_xticks(np.arange(n))
    ax_matrix.set_yticks(np.arange(n))
    ax_matrix.set_xticklabels(labels, rotation=90, fontsize=5)
    ax_matrix.set_yticklabels(labels, fontsize=5)
    ax_matrix.set_title("QUBO coefficient matrix")
    fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

    top_terms = hamiltonian["pauli_terms"][: min(12, len(hamiltonian["pauli_terms"]))]
    y = np.arange(len(top_terms))
    coeffs = [term["coefficient"]["real"] for term in top_terms]
    ax_terms.barh(
        y,
        coeffs,
        color=["#2ca25f" if coeff >= 0 else "#de2d26" for coeff in coeffs],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_terms.axvline(0, color="black", linewidth=0.8)
    ax_terms.set_yticks(y)
    ax_terms.set_yticklabels([term["pauli"] for term in top_terms], fontsize=7)
    ax_terms.invert_yaxis()
    ax_terms.set_xlabel("coefficient")
    ax_terms.set_title("Largest Pauli-Z terms")
    ax_terms.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        f"Task 02: VRPTW QUBO -> Ising / qubits={qubo['num_variables']} / penalty={qubo['penalty']:.1f}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 02 / 构造时间索引 VRPTW-QUBO")
    instance = read_json(args.instance)

    with Timer("构造 VRPTW 时间索引 QUBO"):
        qubo = _build_qubo(instance, args.penalty)
    print(f"  变量数 = {qubo['num_variables']} (x={qubo['num_x_variables']}, y={qubo['num_y_variables']})")
    print(f"  penalty = {qubo['penalty']:.3f}")
    print(f"  线性项数 = {len(qubo['linear'])}, 二次项数 = {len(qubo['quadratic'])}")

    with Timer("QUBO -> Pauli-Z Hamiltonian"):
        operator = qubo_to_ising(qubo)
        hamiltonian = {
            "description": "Ising Hamiltonian equivalent to the time-indexed VRPTW QUBO",
            "num_qubits": int(qubo["num_variables"]),
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
        _plot_qubo(qubo, hamiltonian, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
