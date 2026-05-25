"""[CLASSICAL] Task 08 -- 生成最终 dashboard、最佳路线图和 metrics.json。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="instance.json")
    p.add_argument("--routes", type=Path, required=True, help="routes.json")
    p.add_argument("--qubo", type=Path, required=True, help="qubo.json")
    p.add_argument("--hamiltonian", type=Path, required=True, help="hamiltonian.json")
    p.add_argument("--reference", type=Path, required=True, help="reference.json")
    p.add_argument("--ansatz", type=Path, required=True, help="ansatz.json")
    p.add_argument("--initial", type=Path, required=True, help="initial_quantum.json")
    p.add_argument("--vqe-result", type=Path, required=True, help="vqe_result.json")
    p.add_argument("--trace", type=Path, required=True, help="vqe_trace.csv")
    p.add_argument("--metrics-output", type=Path, required=True, help="输出 metrics.json")
    p.add_argument("--dashboard-output", type=Path, required=True, help="输出 dashboard PNG")
    p.add_argument("--route-output", type=Path, required=True, help="输出最佳路线地图 PNG")
    return p.parse_args()


def _read_trace(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"eval_count": float(row["eval_count"]), "energy": float(row["energy"])})
    return rows


def _build_metrics(reference: dict, initial: dict, vqe: dict) -> dict:
    exact_route = reference["best"]["selected_route_id"]
    best_prob_route = vqe["probability_summary"]["best_probability_route"]
    vqe_route = None if best_prob_route is None else best_prob_route["route_id"]
    vqe_route_cost = None if best_prob_route is None else float(best_prob_route["route_cost"])
    initial_energy = float(initial["initial_energy"])
    vqe_energy = float(vqe["vqe_energy"])
    exact_energy = float(reference["best"]["energy"])
    exact_route_cost = float(reference["best"]["selected_route_cost"])
    invalid_probability = float(vqe["probability_summary"]["invalid_probability"])
    return {
        "exact_best_route_id": exact_route,
        "exact_best_route_label": reference["best"]["selected_route_label"],
        "exact_best_route_cost": exact_route_cost,
        "exact_route_cost": exact_route_cost,
        "exact_best_energy": exact_energy,
        "vqe_energy": vqe_energy,
        "initial_energy": initial_energy,
        "vqe_energy_gap_to_exact": vqe_energy - exact_energy,
        "improvement_from_initial": initial_energy - vqe_energy,
        "vqe_best_probability_route_id": vqe_route,
        "vqe_best_probability_route_label": None if best_prob_route is None else best_prob_route["route_label"],
        "vqe_best_probability_route_cost": vqe_route_cost,
        "vqe_route_cost": vqe_route_cost,
        "route_cost_gap": None if vqe_route_cost is None else vqe_route_cost - exact_route_cost,
        "vqe_best_route_probability": 0.0 if best_prob_route is None else best_prob_route["probability"],
        "vqe_matches_exact_best_route": vqe_route == exact_route,
        "vqe_energy_lower_than_initial": vqe_energy < initial_energy,
        "one_hot_success_probability": 1.0 - invalid_probability,
        "invalid_probability": invalid_probability,
        "top_probability_states": vqe["probability_summary"]["top_states"][:6],
        "route_probabilities": vqe["probability_summary"]["route_probabilities"],
    }


def _route_by_id(routes: list[dict], route_id: str) -> dict:
    return next(route for route in routes if route["route_id"] == route_id)


def _plot_route_map(instance: dict, routes: list[dict], exact_route_id: str, vqe_route_id: str | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    exact_route = _route_by_id(routes, exact_route_id)
    vqe_route = _route_by_id(routes, vqe_route_id) if vqe_route_id is not None else None

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for route in routes:
        points = [depot] + [customers[cid] for cid in route["order"]] + [depot]
        ax.plot([p["x"] for p in points], [p["y"] for p in points], color="#d9d9d9", linewidth=1.0, alpha=0.5)

    exact_points = [depot] + [customers[cid] for cid in exact_route["order"]] + [depot]
    ax.plot(
        [p["x"] for p in exact_points],
        [p["y"] for p in exact_points],
        color="#31a354",
        linewidth=2.8,
        marker="o",
        label=f"Exact: {exact_route['route_id']} {exact_route['label']}",
    )
    if vqe_route is not None and vqe_route["route_id"] != exact_route["route_id"]:
        vqe_points = [depot] + [customers[cid] for cid in vqe_route["order"]] + [depot]
        ax.plot(
            [p["x"] for p in vqe_points],
            [p["y"] for p in vqe_points],
            color="#de2d26",
            linewidth=2.0,
            marker="x",
            linestyle="--",
            label=f"VQE prob.: {vqe_route['route_id']} {vqe_route['label']}",
        )
    for node_id, node in {"D": depot, **customers}.items():
        marker = "s" if node_id == "D" else "o"
        color = "#525252" if node_id == "D" else "#9ecae1"
        ax.scatter([node["x"]], [node["y"]], s=150, marker=marker, color=color, edgecolor="black", zorder=4)
        ax.text(node["x"] + 0.08, node["y"] + 0.08, node_id, fontsize=11, weight="bold")
    ax.set_title("Best VRPTW route")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_dashboard(
    instance: dict,
    routes: list[dict],
    qubo: dict,
    hamiltonian: dict,
    reference: dict,
    ansatz: dict,
    initial: dict,
    vqe: dict,
    metrics: dict,
    trace: list[dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    ax_problem, ax_costs, ax_conv, ax_probs = axes.ravel()

    customers = {c["id"]: c for c in instance["customers"]}
    depot = instance["depot"]
    best_route = _route_by_id(routes, reference["best"]["selected_route_id"])
    best_points = [depot] + [customers[cid] for cid in best_route["order"]] + [depot]
    ax_problem.plot([p["x"] for p in best_points], [p["y"] for p in best_points], color="#31a354", linewidth=2.4, marker="o")
    for node_id, node in {"D": depot, **customers}.items():
        marker = "s" if node_id == "D" else "o"
        ax_problem.scatter([node["x"]], [node["y"]], s=120, marker=marker, color="#9ecae1", edgecolor="black", zorder=4)
        ax_problem.text(node["x"] + 0.08, node["y"] + 0.08, node_id, fontsize=10, weight="bold")
    summary = (
        f"routes: {len(routes)}\n"
        f"qubits: {hamiltonian['num_qubits']}\n"
        f"Pauli terms: {hamiltonian['num_pauli_terms']}\n"
        f"ansatz params: {ansatz['num_parameters']}\n"
        f"penalty: {qubo['penalty']:.1f}"
    )
    ax_problem.text(0.03, 0.05, summary, transform=ax_problem.transAxes, fontsize=9, va="bottom")
    ax_problem.set_title("Problem -> candidate routes")
    ax_problem.set_xlabel("x")
    ax_problem.set_ylabel("y")
    ax_problem.grid(True, alpha=0.25)
    ax_problem.set_aspect("equal")

    sorted_routes = sorted(routes, key=lambda route: route["cost"])
    labels = [route["route_id"] + "\n" + route["label"] for route in sorted_routes]
    costs = [route["cost"] for route in sorted_routes]
    colors = ["#31a354" if route["route_id"] == reference["best"]["selected_route_id"] else "#9ecae1" for route in sorted_routes]
    x = np.arange(len(sorted_routes))
    ax_costs.bar(x, costs, color=colors, edgecolor="black", linewidth=0.6)
    ax_costs.set_xticks(x)
    ax_costs.set_xticklabels(labels, fontsize=8)
    ax_costs.set_ylabel("route cost")
    ax_costs.set_title("Classical route costs")
    ax_costs.grid(True, axis="y", alpha=0.25)

    if trace:
        ax_conv.plot([row["eval_count"] for row in trace], [row["energy"] for row in trace], color="#756bb1", marker="o", markersize=3, linewidth=1.2)
    ax_conv.axhline(reference["best"]["energy"], color="#31a354", linestyle="--", linewidth=1.2, label="Exact best")
    ax_conv.scatter([0], [initial["initial_energy"]], color="#fd8d3c", s=70, label="Initial")
    ax_conv.set_xlabel("function evaluation")
    ax_conv.set_ylabel("energy")
    ax_conv.set_title("VQE hybrid loop")
    ax_conv.grid(True, alpha=0.25)
    ax_conv.legend(fontsize=9)

    route_probs = vqe["probability_summary"]["route_probabilities"]
    prob_labels = [row["route_id"] + "\n" + row["route_label"] for row in route_probs]
    probs = [row["probability"] for row in route_probs]
    prob_colors = ["#31a354" if row["route_id"] == reference["best"]["selected_route_id"] else "#9ecae1" for row in route_probs]
    x = np.arange(len(route_probs))
    ax_probs.bar(x, probs, color=prob_colors, edgecolor="black", linewidth=0.6)
    ax_probs.set_xticks(x)
    ax_probs.set_xticklabels(prob_labels, fontsize=8)
    ax_probs.set_ylim(0, max(probs + [0.1]) * 1.25)
    ax_probs.set_ylabel("probability")
    ax_probs.set_title("Final route probabilities")
    ax_probs.grid(True, axis="y", alpha=0.25)
    status = "route match" if metrics["vqe_matches_exact_best_route"] else "route mismatch"
    ax_probs.text(
        0.98,
        0.92,
        f"{status}\nenergy gap={metrics['vqe_energy_gap_to_exact']:.3f}",
        transform=ax_probs.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    fig.suptitle("VQE for VRPTW route selection", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 08 / 最终可视化和指标汇总")
    instance = read_json(args.instance)
    routes = read_json(args.routes)["routes"]
    qubo = read_json(args.qubo)
    hamiltonian = read_json(args.hamiltonian)
    reference = read_json(args.reference)
    ansatz = read_json(args.ansatz)
    initial = read_json(args.initial)
    vqe = read_json(args.vqe_result)
    trace = _read_trace(args.trace)

    metrics = _build_metrics(reference, initial, vqe)
    out = write_json(args.metrics_output, metrics)
    print(f"  -> 写入 {out}")
    print(f"  VQE 是否选中精确最优路线: {metrics['vqe_matches_exact_best_route']}")
    print(f"  VQE 是否低于初始能量: {metrics['vqe_energy_lower_than_initial']}")

    with Timer("绘制最终 dashboard"):
        _plot_dashboard(instance, routes, qubo, hamiltonian, reference, ansatz, initial, vqe, metrics, trace, args.dashboard_output)
    print(f"  -> 写入 {args.dashboard_output}")

    with Timer("绘制最佳路线地图"):
        _plot_route_map(
            instance,
            routes,
            metrics["exact_best_route_id"],
            metrics["vqe_best_probability_route_id"],
            args.route_output,
        )
    print(f"  -> 写入 {args.route_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
