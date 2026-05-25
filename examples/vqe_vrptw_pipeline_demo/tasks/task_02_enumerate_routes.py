"""[CLASSICAL] Task 02 -- 枚举候选路线并计算时间窗代价。

对 3 个客户共有 3! = 6 条访问顺序。本任务逐条模拟车辆行驶、等待、服务和
迟到，把每条完整路线压缩成一个候选列，供后续 QUBO 选择。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--output-json", type=Path, required=True, help="输出 routes.json")
    p.add_argument("--output-csv", type=Path, required=True, help="输出 routes.csv")
    p.add_argument("--figure-output", type=Path, required=True, help="输出路线可视化 PNG")
    p.add_argument("--wait-weight", type=float, default=0.2, help="等待时间权重")
    p.add_argument("--late-weight", type=float, default=8.0, help="迟到时间权重")
    return p.parse_args()


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _simulate_route(instance: dict, order: tuple[dict, ...], wait_weight: float, late_weight: float, route_id: str) -> dict:
    depot = instance["depot"]
    time = float(instance["vehicle"]["start_time"])
    previous = depot
    distance = 0.0
    total_wait = 0.0
    total_late = 0.0
    events: list[dict] = []
    legs: list[dict] = []

    for customer in order:
        travel = _distance(previous, customer)
        distance += travel
        arrival = time + travel
        ready, due = [float(x) for x in customer["time_window"]]
        start_service = max(arrival, ready)
        wait = max(0.0, ready - arrival)
        late = max(0.0, start_service - due)
        depart = start_service + float(customer["service_time"])

        total_wait += wait
        total_late += late
        legs.append(
            {
                "from": previous["id"],
                "to": customer["id"],
                "travel_time": travel,
                "arrival_time": arrival,
            }
        )
        events.append(
            {
                "customer_id": customer["id"],
                "arrival_time": arrival,
                "service_start_time": start_service,
                "departure_time": depart,
                "waiting_time": wait,
                "late_time": late,
                "time_window": [ready, due],
            }
        )
        time = depart
        previous = customer

    travel_home = _distance(previous, depot)
    distance += travel_home
    completion_time = time + travel_home
    legs.append(
        {
            "from": previous["id"],
            "to": depot["id"],
            "travel_time": travel_home,
            "arrival_time": completion_time,
        }
    )

    cost = distance + wait_weight * total_wait + late_weight * total_late
    return {
        "route_id": route_id,
        "label": "-".join(customer["id"] for customer in order),
        "order": [customer["id"] for customer in order],
        "distance": distance,
        "total_waiting_time": total_wait,
        "total_late_time": total_late,
        "completion_time": completion_time,
        "cost": cost,
        "is_time_window_feasible": total_late <= 1e-9,
        "events": events,
        "legs": legs,
    }


def _write_csv(routes: list[dict], out_path: Path) -> None:
    out = ensure_parent_dir(out_path)
    fields = [
        "route_id",
        "label",
        "distance",
        "total_waiting_time",
        "total_late_time",
        "completion_time",
        "cost",
        "is_time_window_feasible",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for route in routes:
            writer.writerow({field: route[field] for field in fields})


def _plot_routes(instance: dict, routes: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    depot = instance["depot"]
    customers = {c["id"]: c for c in instance["customers"]}
    sorted_routes = sorted(routes, key=lambda route: route["cost"])
    best = sorted_routes[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_map, ax_bar = axes

    all_nodes = {"D": depot, **customers}
    for route in routes:
        points = [depot] + [customers[cid] for cid in route["order"]] + [depot]
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        ax_map.plot(xs, ys, color="#bdbdbd", linewidth=1.0, alpha=0.35)

    best_points = [depot] + [customers[cid] for cid in best["order"]] + [depot]
    ax_map.plot(
        [p["x"] for p in best_points],
        [p["y"] for p in best_points],
        color="#de2d26",
        linewidth=2.3,
        marker="o",
        label=f"Best route: {best['label']}",
    )
    for node_id, node in all_nodes.items():
        marker = "s" if node_id == "D" else "o"
        color = "#525252" if node_id == "D" else "#6baed6"
        ax_map.scatter([node["x"]], [node["y"]], s=130, marker=marker, color=color, edgecolor="black", zorder=3)
        ax_map.text(node["x"] + 0.08, node["y"] + 0.08, node_id, fontsize=11, weight="bold")
    ax_map.set_title("Candidate routes")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="lower right", fontsize=9)

    labels = [route["route_id"] + "\n" + route["label"] for route in sorted_routes]
    costs = [route["cost"] for route in sorted_routes]
    colors = ["#31a354" if route["is_time_window_feasible"] else "#fd8d3c" for route in sorted_routes]
    bars = ax_bar.bar(np.arange(len(routes)), costs, color=colors, edgecolor="black", linewidth=0.6)
    for bar, route in zip(bars, sorted_routes):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.015,
            f"late={route['total_late_time']:.1f}",
            ha="center",
            fontsize=8,
        )
    ax_bar.set_xticks(np.arange(len(routes)))
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel("route cost")
    ax_bar.set_title("Distance + waiting + late penalty")
    ax_bar.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Task 02: route enumeration")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 02 / 枚举候选路线")
    instance = read_json(args.instance)
    customers = instance["customers"]

    with Timer("枚举 3! 条路线并模拟时间窗"):
        routes = [
            _simulate_route(instance, order, args.wait_weight, args.late_weight, f"R{i + 1:02d}")
            for i, order in enumerate(itertools.permutations(customers))
        ]
        routes.sort(key=lambda route: route["route_id"])

    best = min(routes, key=lambda route: route["cost"])
    print(f"  候选路线数 = {len(routes)}")
    print(f"  当前最低代价路线 = {best['route_id']} / {best['label']} / cost={best['cost']:.3f}")

    payload = {
        "source_instance": str(args.instance),
        "wait_weight": args.wait_weight,
        "late_weight": args.late_weight,
        "routes": routes,
    }
    out_json = write_json(args.output_json, payload)
    print(f"  -> 写入 {out_json}")

    _write_csv(routes, args.output_csv)
    print(f"  -> 写入 {args.output_csv}")

    with Timer("绘制候选路线和代价"):
        _plot_routes(instance, routes, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
