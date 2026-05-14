"""VQE H2 示例的共用工具。

本文件只放轻量级基础设施：路径常量、任务类型、彩色 banner、计时器、
JSON 读写和 estimator 创建函数。把 estimator 创建集中在这里，是为了以后
把本地模拟器替换成真实量子后端时，只改一个地方。
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
MOLECULE_DIR = ARTIFACTS_DIR / "molecule"
HAMILTONIAN_DIR = ARTIFACTS_DIR / "hamiltonian"
ANSATZ_DIR = ARTIFACTS_DIR / "ansatz"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

MOLECULE_JSON = MOLECULE_DIR / "molecule.json"
PROBLEM_DILL = HAMILTONIAN_DIR / "problem.dill"
HAMILTONIAN_JSON = HAMILTONIAN_DIR / "hamiltonian.json"
REFERENCE_JSON = RESULTS_DIR / "reference.json"
ANSATZ_DILL = ANSATZ_DIR / "ansatz.dill"
ANSATZ_JSON = ANSATZ_DIR / "ansatz.json"
INITIAL_POINT_NPY = ANSATZ_DIR / "initial_point.npy"
INITIAL_ENERGY_JSON = RESULTS_DIR / "initial_energy.json"
VQE_RESULT_JSON = RESULTS_DIR / "vqe_result.json"
VQE_RESULT_DILL = RESULTS_DIR / "vqe_result.dill"
VQE_TRACE_CSV = RESULTS_DIR / "vqe_trace.csv"
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


def complex_to_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def create_estimator(seed: int | None = None):
    """创建量子期望值估计器。

    默认使用本地 ``StatevectorEstimator``，这是精确、无噪声的状态向量模拟。
    后续要切到真实量子计算机时，可把这里替换为 IBM Runtime 的 EstimatorV2，
    同时在任务参数里传入 backend/session 信息。
    """

    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator(seed=seed)


def hartree_to_ev(value: float) -> float:
    """Hartree 转电子伏特。"""

    return value * 27.211386245988
