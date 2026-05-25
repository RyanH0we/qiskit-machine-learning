"""[CLASSICAL] Task 01 -- 定义或读取一个小型 VRPTW 实例。

本任务只做经典数据准备：车辆、客户、容量、服务时间、时间窗、距离矩阵和
离散 travel time 矩阵。后续 task 会直接基于这些数据构造 QUBO。
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, euclidean, print_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance-input", type=Path, default=None, help="可选：自定义 VRPTW JSON")
    p.add_argument("--output", type=Path, required=True, help="输出 instance.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出实例可视化 PNG")
    p.add_argument("--time-granularity", type=float, default=1.0, help="离散时间粒度")
    p.add_argument("--max-stops-per-vehicle", type=int, default=2, help="每辆车最多服务客户数")
    return p.parse_args()


def _default_instance() -> dict[str, Any]:
    return {
        "name": "vrptw_2_vehicle_4_customer_default",
        "description": "2 辆车、4 个客户、容量和硬时间窗的最小通用 VRPTW 示例",
        "depot": {
            "id": "Depot",
            "name": "Depot",
            "x": 0.0,
            "y": 0.0,
            "service_time": 0.0,
            "time_window": [0.0, 8.0],
        },
        "vehicles": [
            {"id": "V0", "capacity": 2, "start_time": 0.0},
            {"id": "V1", "capacity": 2, "start_time": 0.0},
        ],
        "customers": [
            {"id": "A", "x": 1.0, "y": 0.0, "demand": 1, "service_time": 1.0, "time_window": [1.0, 1.0]},
            {"id": "B", "x": 2.0, "y": 0.0, "demand": 1, "service_time": 1.0, "time_window": [3.0, 3.0]},
            {"id": "C", "x": 0.0, "y": 1.0, "demand": 1, "service_time": 1.0, "time_window": [1.0, 1.0]},
            {"id": "D", "x": 0.0, "y": 2.0, "demand": 1, "service_time": 1.0, "time_window": [3.0, 3.0]},
        ],
        "distance_matrix": None,
        "travel_time_matrix": None,
        "units": {
            "distance": "Euclidean coordinate unit",
            "time": "discrete time unit after applying time_granularity",
        },
    }


def _validate(instance: dict[str, Any]) -> None:
    depot_id = instance["depot"]["id"]
    customer_ids = [customer["id"] for customer in instance["customers"]]
    vehicle_ids = [vehicle["id"] for vehicle in instance["vehicles"]]
    if depot_id in customer_ids:
        raise ValueError("depot id 不能和 customer id 重复")
    if len(customer_ids) != len(set(customer_ids)):
        raise ValueError("customer id 不能重复")
    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError("vehicle id 不能重复")
    for customer in instance["customers"]:
        ready, due = customer["time_window"]
        if float(ready) > float(due):
            raise ValueError(f"客户 {customer['id']} 的时间窗不合法")
    if not instance["vehicles"]:
        raise ValueError("至少需要一辆车")
    if not instance["customers"]:
        raise ValueError("至少需要一个客户")


def _complete_matrices(instance: dict[str, Any], granularity: float) -> dict[str, Any]:
    if granularity <= 0:
        raise ValueError("--time-granularity 必须为正数")
    nodes = [instance["depot"], *instance["customers"]]
    ids = [node["id"] for node in nodes]

    distance_matrix = instance.get("distance_matrix")
    if distance_matrix is None:
        distance_matrix = {
            src["id"]: {dst["id"]: float(euclidean(src, dst)) for dst in nodes}
            for src in nodes
        }

    travel_time_matrix = instance.get("travel_time_matrix")
    if travel_time_matrix is None:
        travel_time_matrix = {
            src: {
                dst: int(math.ceil(float(distance_matrix[src][dst]) / granularity - 1e-12))
                for dst in ids
            }
            for src in ids
        }

    out = dict(instance)
    out["distance_matrix"] = distance_matrix
    out["travel_time_matrix"] = travel_time_matrix
    out["solver_config"] = {
        "time_granularity": float(granularity),
        "max_stops_per_vehicle": int(instance["solver_config"]["max_stops_per_vehicle"]),
    }
    return out


def _plot_instance(instance: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    depot = instance["depot"]
    customers = instance["customers"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax_map, ax_tw = axes

    ax_map.scatter([depot["x"]], [depot["y"]], s=180, marker="s", color="#525252", edgecolor="black", label="Depot")
    ax_map.text(depot["x"] + 0.05, depot["y"] + 0.05, depot["id"], fontsize=10, weight="bold")
    colors = ["#3182bd", "#31a354", "#de2d26", "#756bb1"]
    for customer, color in zip(customers, colors):
        ax_map.scatter([customer["x"]], [customer["y"]], s=130, color=color, edgecolor="black")
        ax_map.text(customer["x"] + 0.05, customer["y"] + 0.05, customer["id"], fontsize=10, weight="bold")
    ax_map.set_title("VRPTW coordinates")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(fontsize=9)

    y = np.arange(len(customers))
    for i, (customer, color) in enumerate(zip(customers, colors)):
        ready, due = customer["time_window"]
        width = max(0.08, float(due) - float(ready))
        ax_tw.barh(i, width, left=float(ready), height=0.48, color=color, edgecolor="black", alpha=0.85)
        ax_tw.text(float(due) + 0.12, i, f"{customer['id']} demand={customer['demand']}", va="center", fontsize=9)
    ax_tw.set_yticks(y)
    ax_tw.set_yticklabels([c["id"] for c in customers])
    ax_tw.set_xlabel("time")
    ax_tw.set_title("Customer hard time windows")
    ax_tw.grid(True, axis="x", alpha=0.25)
    ax_tw.set_xlim(0, float(instance["depot"]["time_window"][1]) + 0.8)

    fig.suptitle(
        f"Task 01: {len(instance['vehicles'])} vehicles x {len(customers)} customers / "
        f"max stops={instance['solver_config']['max_stops_per_vehicle']}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义或读取 VRPTW 实例")
    if args.instance_input is None:
        instance = _default_instance()
        print("  使用默认 2车4客户 VRPTW 实例")
    else:
        instance = read_json(args.instance_input)
        print(f"  读取自定义实例: {args.instance_input}")

    instance = dict(instance)
    instance["solver_config"] = {
        **dict(instance.get("solver_config", {})),
        "max_stops_per_vehicle": int(args.max_stops_per_vehicle),
    }
    _validate(instance)
    instance = _complete_matrices(instance, args.time_granularity)
    print(f"  车辆数 = {len(instance['vehicles'])}")
    print(f"  客户数 = {len(instance['customers'])}")
    print(f"  时间粒度 = {instance['solver_config']['time_granularity']}")
    for vehicle in instance["vehicles"]:
        print(f"  车辆 {vehicle['id']}: capacity={vehicle['capacity']}, start={vehicle['start_time']}")
    for customer in instance["customers"]:
        print(f"  客户 {customer['id']}: demand={customer['demand']}, time_window={customer['time_window']}")

    out = write_json(args.output, instance)
    print(f"  -> 写入 {out}")

    with Timer("绘制坐标和时间窗"):
        _plot_instance(instance, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")

    if args.instance_input is not None and args.instance_input.resolve() != args.output.resolve():
        copied = args.output.with_suffix(".source.json")
        shutil.copyfile(args.instance_input, copied)
        print(f"  -> 备份原始输入 {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
