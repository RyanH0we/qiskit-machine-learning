"""VQE-JSSP + Flyte 示例的共用工具与 ImageSpec 定义。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from flytekit import ImageSpec
from flytekit.types.file import FlyteFile


COMMON_PACKAGES: tuple[str, ...] = (
    "flytekit==1.16.21",
    "qiskit>=2,<3",
    "qiskit-algorithms==0.4.0",
    "qiskit-aer>=0.17",
    "numpy>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "dill>=0.3.4",
    "pylatexenc>=2.10",
)

CACHE_VERSION = "jssp-v1"


def _resolve_registry() -> str:
    """返回 ImageSpec 镜像仓地址。

    本机 flytectl demo sandbox 自带 registry，地址是 ``localhost:30000``。
    迁移到正式 Flyte 集群时，通过 ``FLYTE_IMAGE_REGISTRY`` 覆盖。
    """

    return os.environ.get("FLYTE_IMAGE_REGISTRY", "localhost:30000")


vqe_jssp_image = ImageSpec(
    name="vqe-jssp-flyte",
    python_version="3.12",
    packages=list(COMMON_PACKAGES),
    registry=_resolve_registry(),
)


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
    """在终端或 Flyte pod 日志中打印任务类型 banner。"""

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}", flush=True)
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}", flush=True)
    print(f"{color}{bold}{bar}{reset}", flush=True)


def task_workdir(prefix: str) -> Path:
    """为单个 Flyte task 创建绝对路径临时目录。"""

    return Path(tempfile.mkdtemp(prefix=f"vqejssp-{prefix}-"))


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


def create_sampler(shots: int = 4096, seed: int | None = None):
    """创建本地量子采样器。

    默认使用 Qiskit ``StatevectorSampler``，只在本地模拟量子线路，不连接真机。
    """

    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler(default_shots=shots, seed=seed)


def complex_to_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def qubo_energy(qubo: dict[str, Any], bits: list[int]) -> float:
    energy = float(qubo.get("offset", 0.0))
    for term in qubo.get("linear", []):
        energy += float(term["coefficient"]) * bits[int(term["index"])]
    for term in qubo.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        energy += float(term["coefficient"]) * bits[i] * bits[j]
    return float(energy)


def bits_to_bitstring(bits: list[int]) -> str:
    """把变量顺序 bits 转为 Qiskit 风格 bitstring。"""

    return "".join(str(int(b)) for b in reversed(bits))


def bitstring_to_bits(bitstring: str, num_variables: int | None = None) -> list[int]:
    bits = [int(ch) for ch in reversed(bitstring.strip())]
    if num_variables is not None and len(bits) != num_variables:
        raise ValueError(f"bitstring 长度 {len(bits)} 与变量数 {num_variables} 不一致")
    return bits


def int_to_bits(value: int, num_variables: int) -> list[int]:
    return [(value >> i) & 1 for i in range(num_variables)]


def decode_schedule(instance: dict[str, Any], variables: list[dict[str, Any]], bits: list[int]) -> dict[str, Any]:
    """把 QUBO bitstring 解码为 JSSP 排程，并检查约束。"""

    operations = instance["operations"]
    horizon = int(instance["horizon"])
    selected_starts: dict[tuple[int, int], list[dict[str, Any]]] = {}
    selected_cmax: list[int] = []

    for var in variables:
        idx = int(var["index"])
        if idx >= len(bits) or bits[idx] == 0:
            continue
        if var["kind"] == "start":
            key = (int(var["job"]), int(var["operation"]))
            selected_starts.setdefault(key, []).append(var)
        elif var["kind"] == "cmax":
            selected_cmax.append(int(var["makespan"]))

    violations: list[str] = []
    schedule: list[dict[str, Any]] = []
    op_map = {(int(op["job"]), int(op["operation"])): op for op in operations}

    for op in operations:
        key = (int(op["job"]), int(op["operation"]))
        starts = selected_starts.get(key, [])
        if len(starts) != 1:
            violations.append(f"工序 J{key[0]}-O{key[1]} 选择了 {len(starts)} 个开始时间")
            continue
        start = int(starts[0]["start"])
        duration = int(op["duration"])
        finish = start + duration
        schedule.append(
            {
                "job": key[0],
                "operation": key[1],
                "machine": int(op["machine"]),
                "duration": duration,
                "start": start,
                "finish": finish,
                "label": f"J{key[0]}O{key[1]}",
            }
        )
        if finish > horizon:
            violations.append(f"工序 J{key[0]}-O{key[1]} 完成时间 {finish} 超过 horizon {horizon}")

    if len(selected_cmax) != 1:
        violations.append(f"Cmax 选择了 {len(selected_cmax)} 个候选值")
        chosen_cmax = None
    else:
        chosen_cmax = selected_cmax[0]

    schedule_by_key = {(row["job"], row["operation"]): row for row in schedule}
    for job in instance["jobs"]:
        ops = job["operations"]
        for prev, nxt in zip(ops, ops[1:]):
            prev_key = (int(job["job"]), int(prev["operation"]))
            next_key = (int(job["job"]), int(nxt["operation"]))
            if prev_key not in schedule_by_key or next_key not in schedule_by_key:
                continue
            if schedule_by_key[next_key]["start"] < schedule_by_key[prev_key]["finish"]:
                violations.append(
                    f"作业 J{job['job']} 的 O{prev['operation']} 与 O{nxt['operation']} 顺序冲突"
                )

    for machine in instance["machines"]:
        rows = sorted(
            [row for row in schedule if row["machine"] == int(machine)],
            key=lambda row: (row["start"], row["finish"], row["job"], row["operation"]),
        )
        for left, right in zip(rows, rows[1:]):
            if right["start"] < left["finish"]:
                violations.append(f"机器 M{machine} 上 {left['label']} 与 {right['label']} 时间重叠")

    actual_makespan = max((row["finish"] for row in schedule), default=None)
    if chosen_cmax is not None and actual_makespan is not None and actual_makespan > chosen_cmax:
        violations.append(f"实际 makespan {actual_makespan} 超过选择的 Cmax {chosen_cmax}")

    for op in operations:
        key = (int(op["job"]), int(op["operation"]))
        if key not in schedule_by_key:
            continue
        if chosen_cmax is not None and schedule_by_key[key]["finish"] > chosen_cmax:
            violations.append(
                f"工序 J{key[0]}-O{key[1]} 完成时间 {schedule_by_key[key]['finish']} 超过 Cmax {chosen_cmax}"
            )

    schedule.sort(key=lambda row: (row["machine"], row["start"], row["job"], row["operation"]))
    return {
        "feasible": len(violations) == 0,
        "selected_cmax": chosen_cmax,
        "actual_makespan": actual_makespan,
        "schedule": schedule,
        "violations": violations,
        "num_scheduled_operations": len(schedule),
        "num_operations": len(op_map),
    }


def draw_gantt(
    instance: dict[str, Any],
    decoded: dict[str, Any],
    out_path: str | os.PathLike,
    title: str,
) -> None:
    """绘制 JSSP 甘特图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    machines = [int(m) for m in instance["machines"]]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]
    job_colors = {int(job["job"]): colors[i % len(colors)] for i, job in enumerate(instance["jobs"])}

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    schedule = decoded.get("schedule", [])
    if not schedule:
        ax.text(0.5, 0.5, "No complete schedule to draw", ha="center", va="center", transform=ax.transAxes)
    for row in schedule:
        y = machines.index(int(row["machine"]))
        ax.barh(
            y,
            int(row["duration"]),
            left=int(row["start"]),
            height=0.55,
            color=job_colors[int(row["job"])],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            int(row["start"]) + int(row["duration"]) / 2,
            y,
            row["label"],
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    chosen = decoded.get("selected_cmax")
    actual = decoded.get("actual_makespan")
    if chosen is not None:
        ax.axvline(chosen, color="#d62728", linestyle="--", linewidth=1.2, label=f"Cmax={chosen}")
    elif actual is not None:
        ax.axvline(actual, color="#d62728", linestyle="--", linewidth=1.2, label=f"makespan={actual}")

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlim(0, int(instance["horizon"]) + 0.5)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    if chosen is not None or actual is not None:
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_trace_csv(trace: list[dict], out_path: Path, num_parameters: int) -> None:
    out = ensure_parent_dir(out_path)
    fields = ["eval_count", "mean_qubo_energy"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            flat = {"eval_count": row["eval_count"], "mean_qubo_energy": row["mean_qubo_energy"]}
            for i, value in enumerate(row["parameters"]):
                flat[f"theta_{i}"] = value
            writer.writerow(flat)


def archive_named_outputs(outputs: Any, destination: Path) -> list[Path]:
    """把本地 Flyte 执行返回的 FlyteFile 输出复制到归档目录。"""

    destination.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    for field in getattr(outputs, "_fields", []):
        value = getattr(outputs, field)
        if not isinstance(value, FlyteFile):
            continue
        source = Path(value.path)
        if not source.exists():
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        archived.append(target)
    return archived


def join_packages(extra: Iterable[str] = ()) -> list[str]:
    return list(COMMON_PACKAGES) + list(extra)
