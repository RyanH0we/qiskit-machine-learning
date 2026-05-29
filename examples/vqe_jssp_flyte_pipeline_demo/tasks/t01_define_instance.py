"""[CLASSICAL] Task 01 -- 定义一个小型 JSSP 实例。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, print_banner, task_workdir, vqe_jssp_image


class InstanceOut(NamedTuple):
    instance_json: FlyteFile
    instance_png: FlyteFile


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="300m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=5),
)
def t01_define_instance(horizon: int = 5) -> InstanceOut:
    """[CLASSICAL] 定义 2x2 JSSP 输入问题并绘制结构图。"""

    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义 JSSP 实例")
    instance = _build_default_instance(horizon)
    print("  实例 = 2 个作业, 2 台机器, 每个作业 2 道工序", flush=True)
    print(f"  horizon = {instance['horizon']}, 已知最优 makespan = 4", flush=True)

    work = task_workdir("t01")
    json_path = work / "instance.json"
    json_path.write_text(
        json.dumps(instance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "01_instance.png"
    _draw_instance(instance, png_path)

    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)
    return InstanceOut(instance_json=FlyteFile(str(json_path)), instance_png=FlyteFile(str(png_path)))
