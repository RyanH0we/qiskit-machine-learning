"""[CLASSICAL] Task 01 -- 定义一个小型 JSSP 实例。

这一步只是在经典计算机上写下调度问题本身：有哪些作业、每个作业有哪些
工序、每道工序需要哪台机器和多长时间。它还不会运行量子电路。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=5, help="离散时间上界，默认 5")
    p.add_argument("--output", type=Path, required=True, help="输出 instance.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出实例结构图 PNG")
    return p.parse_args()


def _build_default_instance(horizon: int) -> dict:
    if horizon < 5:
        raise ValueError("默认 2x2 JSSP 实例需要 horizon >= 5")

    jobs = [
        {
            "job": 0,
            "operations": [
                {"operation": 0, "machine": 0, "duration": 2},
                {"operation": 1, "machine": 1, "duration": 1},
            ],
        },
        {
            "job": 1,
            "operations": [
                {"operation": 0, "machine": 1, "duration": 1},
                {"operation": 1, "machine": 0, "duration": 2},
            ],
        },
    ]
    operations = []
    for job in jobs:
        for op in job["operations"]:
            operations.append(
                {
                    "job": int(job["job"]),
                    "operation": int(op["operation"]),
                    "machine": int(op["machine"]),
                    "duration": int(op["duration"]),
                    "label": f"J{job['job']}O{op['operation']}",
                }
            )

    return {
        "name": "jssp_2x2_teaching",
        "description": "面向新手的 2 作业 x 2 机器 JSSP VQE 示例；经典最优 makespan 为 4",
        "time_unit": "离散时间片",
        "horizon": int(horizon),
        "machines": [0, 1],
        "jobs": jobs,
        "operations": operations,
        "known_optimal_makespan": 4,
    }


def _draw_instance(instance: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = {0: "#4e79a7", 1: "#f28e2b"}
    y_positions = {0: 1, 1: 0}

    for job in instance["jobs"]:
        y = y_positions[int(job["job"])]
        cursor = 0
        for op in job["operations"]:
            duration = int(op["duration"])
            machine = int(op["machine"])
            ax.barh(
                y,
                duration,
                left=cursor,
                height=0.5,
                color=colors[int(job["job"])],
                edgecolor="black",
                linewidth=0.8,
            )
            ax.text(
                cursor + duration / 2,
                y,
                f"O{op['operation']} / M{machine} / p={duration}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                weight="bold",
            )
            if op is not job["operations"][-1]:
                ax.annotate(
                    "",
                    xy=(cursor + duration + 0.18, y),
                    xytext=(cursor + duration, y),
                    arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
                )
            cursor += duration + 0.35

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Job 1", "Job 0"])
    ax.set_xlim(0, 4.2)
    ax.set_xlabel("Operation order inside each job; not the final schedule time")
    ax.set_title("JSSP input instance: operations must run from left to right")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义 JSSP 实例")
    instance = _build_default_instance(args.horizon)
    print("  实例 = 2 个作业, 2 台机器, 每个作业 2 道工序")
    print(f"  horizon = {instance['horizon']}, 已知最优 makespan = {instance['known_optimal_makespan']}")

    out = write_json(args.output, instance)
    print(f"  -> 写入 {out}")

    with Timer("绘制实例结构图"):
        _draw_instance(instance, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
