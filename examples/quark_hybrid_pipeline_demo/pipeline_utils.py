"""共用工具：任务类型枚举、彩色 banner、artifact 路径常量。

只在 task 脚本与 main.py 中被 import；不依赖 qiskit / sklearn，保持极轻。
"""

from __future__ import annotations

import os
import sys
import time
from enum import Enum
from pathlib import Path


# demo 根目录 = examples/quark_hybrid_pipeline_demo/
DEMO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEMO_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
KERNEL_DIR = ARTIFACTS_DIR / "kernel"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


class TaskKind(str, Enum):
    QUANTUM = "QUANTUM"
    CLASSICAL = "CLASSICAL"
    HYBRID = "HYBRID"


# ANSI 颜色（terminal 可识别；不可识别时只是多几个不可见字符，不影响功能）
_COLORS = {
    TaskKind.QUANTUM: "\033[95m",   # magenta
    TaskKind.CLASSICAL: "\033[94m", # blue
    TaskKind.HYBRID: "\033[93m",    # yellow
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def print_banner(kind: TaskKind, name: str) -> None:
    """在 task 开头打印类型与名称，方便区分量子 / 经典 / 混合段落。"""
    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 70
    print(f"\n{color}{bold}{bar}{reset}")
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}")
    print(f"{color}{bold}{bar}{reset}")


class Timer:
    """简易计时上下文。退出时打印一行 [done in X.Xs]。"""

    def __init__(self, label: str = "task") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._t0
        status = "done" if exc_type is None else "FAILED"
        print(f"  [{status} in {self.elapsed:.2f}s]")


def ensure_parent_dir(path: str | os.PathLike) -> Path:
    """保证 path 所在目录存在，返回 Path 对象。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | os.PathLike) -> Path:
    """保证目录本身存在，返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
