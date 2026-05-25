"""VQE-JSSP 示例的共用工具。

这里集中放轻量级基础设施：路径常量、任务类型、JSON/CSV 工具、QUBO
能量计算、bitstring 解码、甘特图绘制，以及本地量子 sampler 创建函数。
后续如果要把本地模拟器换成真实量子后端，优先从 ``create_sampler()`` 改起。
"""

from __future__ import annotations

import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEMO_ROOT / "artifacts"
INSTANCE_DIR = ARTIFACTS_DIR / "instance"
QUBO_DIR = ARTIFACTS_DIR / "qubo"
ANSATZ_DIR = ARTIFACTS_DIR / "ansatz"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

INSTANCE_JSON = INSTANCE_DIR / "instance.json"
QUBO_JSON = QUBO_DIR / "qubo.json"
HAMILTONIAN_DILL = QUBO_DIR / "hamiltonian.dill"
HAMILTONIAN_JSON = QUBO_DIR / "hamiltonian.json"
REFERENCE_JSON = RESULTS_DIR / "reference.json"
ANSATZ_DILL = ANSATZ_DIR / "ansatz.dill"
ANSATZ_JSON = ANSATZ_DIR / "ansatz.json"
INITIAL_POINT_NPY = ANSATZ_DIR / "initial_point.npy"
INITIAL_ENERGY_JSON = RESULTS_DIR / "initial_energy.json"
VQE_RESULT_JSON = RESULTS_DIR / "vqe_result.json"
VQE_RESULT_DILL = RESULTS_DIR / "vqe_result.dill"
VQE_TRACE_CSV = RESULTS_DIR / "vqe_trace.csv"
DECODED_SOLUTION_JSON = RESULTS_DIR / "decoded_solution.json"
SAMPLES_CSV = RESULTS_DIR / "samples.csv"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"


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
    """打印任务类型 banner，帮助新手区分当前步骤属于哪类计算。"""

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}")
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}")
    print(f"{color}{bold}{bar}{reset}")


class Timer:
    """简单计时器。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        print(f"  开始: {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._t0
        status = "完成" if exc_type is None else "失败"
        print(f"  [{status}: {self.label}, 用时 {self.elapsed:.2f}s]")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    """创建本地采样器。

    默认使用 ``StatevectorSampler``，它在本机精确模拟量子态，再按 ``shots``
    采样测量结果。未来接真实量子硬件时，可在这里替换成 IBM Runtime
    ``SamplerV2``，并把 backend/session/shots 等参数透传进来。
    """

    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler(default_shots=shots, seed=seed)


def complex_to_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def qubo_energy(qubo: dict[str, Any], bits: list[int]) -> float:
    """计算 QUBO 目标值 ``offset + linear + quadratic``。"""

    energy = float(qubo.get("offset", 0.0))
    for term in qubo.get("linear", []):
        energy += float(term["coefficient"]) * bits[int(term["index"])]
    for term in qubo.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        energy += float(term["coefficient"]) * bits[i] * bits[j]
    return float(energy)


def bits_to_bitstring(bits: list[int]) -> str:
    """把变量顺序 bits 转为 Qiskit 风格 bitstring。

    变量 ``i`` 映射到 qubit ``i``。Qiskit 打印 bitstring 时左侧是高编号
    qubit，所以这里需要反转。
    """

    return "".join(str(int(b)) for b in reversed(bits))


def bitstring_to_bits(bitstring: str, num_variables: int | None = None) -> list[int]:
    """把 Qiskit 风格 bitstring 转回变量顺序 bits。"""

    bits = [int(ch) for ch in reversed(bitstring.strip())]
    if num_variables is not None and len(bits) != num_variables:
        raise ValueError(f"bitstring 长度 {len(bits)} 与变量数 {num_variables} 不一致")
    return bits


def int_to_bits(value: int, num_variables: int) -> list[int]:
    return [(value >> i) & 1 for i in range(num_variables)]


def variable_lookup(variables: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(v["name"]): v for v in variables}


def decode_schedule(instance: dict[str, Any], variables: list[dict[str, Any]], bits: list[int]) -> dict[str, Any]:
    """把二进制变量解码为 JSSP 排程，并检查约束是否满足。"""

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
                violations.append(
                    f"机器 M{machine} 上 {left['label']} 与 {right['label']} 时间重叠"
                )

    actual_makespan = max((row["finish"] for row in schedule), default=None)
    if chosen_cmax is not None and actual_makespan is not None and actual_makespan > chosen_cmax:
        violations.append(f"实际 makespan {actual_makespan} 超过选择的 Cmax {chosen_cmax}")

    for op in operations:
        key = (int(op["job"]), int(op["operation"]))
        if key not in schedule_by_key:
            continue
        if chosen_cmax is not None and schedule_by_key[key]["finish"] > chosen_cmax:
            # 非最终工序通常会被作业顺序约束带住；这里仍显式检查，便于教学。
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
