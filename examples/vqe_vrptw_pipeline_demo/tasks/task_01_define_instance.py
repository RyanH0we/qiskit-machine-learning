"""[CLASSICAL] Task 01 -- 定义一个小型 VRPTW 实例。

本任务只做经典数据准备：定义 depot、3 个客户、坐标、服务时间和时间窗。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="输出 instance.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出实例可视化 PNG")
    return p.parse_args()


def _default_instance() -> dict:
    return {
        "name": "toy_vrptw_1_vehicle_3_customers",
        "description": "1 辆车服务 3 个客户的 VRPTW 教学实例",
        "vehicle": {
            "count": 1,
            "capacity": None,
            "speed": 1.0,
            "start_time": 0.0,
        },
        "depot": {
            "id": "D",
            "name": "Depot",
            "x": 0.0,
            "y": 0.0,
            "service_time": 0.0,
            "time_window": [0.0, 40.0],
        },
        "customers": [
            {
                "id": "A",
                "name": "Customer A",
                "x": 1.0,
                "y": 3.0,
                "service_time": 0.4,
                "time_window": [2.5, 6.5],
            },
            {
                "id": "B",
                "name": "Customer B",
                "x": 4.0,
                "y": 2.0,
                "service_time": 0.4,
                "time_window": [5.0, 9.0],
            },
            {
                "id": "C",
                "name": "Customer C",
                "x": 5.0,
                "y": 5.0,
                "service_time": 0.4,
                "time_window": [7.0, 12.0],
            },
        ],
        "units": {
            "distance": "arbitrary coordinate unit",
            "time": "same unit as distance because speed=1",
        },
    }


def _plot_instance(instance: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    depot = instance["depot"]
    customers = instance["customers"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax_map, ax_tw = axes

    ax_map.scatter([depot["x"]], [depot["y"]], s=180, marker="s", color="#525252", label="Depot")
    ax_map.text(depot["x"] + 0.08, depot["y"] + 0.08, depot["id"], fontsize=11, weight="bold")
    colors = ["#3182bd", "#31a354", "#de2d26"]
    for customer, color in zip(customers, colors):
        ax_map.scatter([customer["x"]], [customer["y"]], s=140, color=color, edgecolor="black", linewidth=0.8)
        ax_map.text(customer["x"] + 0.08, customer["y"] + 0.08, customer["id"], fontsize=11, weight="bold")
    ax_map.set_title("VRPTW coordinates")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="lower right", fontsize=9)

    y = np.arange(len(customers))
    for i, (customer, color) in enumerate(zip(customers, colors)):
        ready, due = customer["time_window"]
        ax_tw.barh(i, due - ready, left=ready, height=0.5, color=color, alpha=0.8, edgecolor="black")
        ax_tw.text(due + 0.2, i, f"{customer['id']}  [{ready:.1f}, {due:.1f}]", va="center", fontsize=9)
    ax_tw.set_yticks(y)
    ax_tw.set_yticklabels([c["id"] for c in customers])
    ax_tw.set_xlabel("time")
    ax_tw.set_title("Customer time windows")
    ax_tw.grid(True, axis="x", alpha=0.25)
    ax_tw.set_xlim(0, max(c["time_window"][1] for c in customers) + 3.0)

    fig.suptitle("Task 01: problem instance")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义 VRPTW 实例")

    instance = _default_instance()
    print("  车辆数 = 1")
    print("  客户数 = 3")
    for customer in instance["customers"]:
        print(f"  客户 {customer['id']}: 坐标=({customer['x']}, {customer['y']}), 时间窗={customer['time_window']}")

    out = write_json(args.output, instance)
    print(f"  -> 写入 {out}")

    with Timer("绘制坐标和时间窗"):
        _plot_instance(instance, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
