"""Hybrid QSVC + Flyte 示例的共用工具与 ImageSpec 定义。"""

from __future__ import annotations

import os
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from flytekit import ImageSpec


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1] if len(DEMO_ROOT.parents) > 1 else DEMO_ROOT

CACHE_VERSION = "hybrid-qsvc-v1"

COMMON_PACKAGES: tuple[str, ...] = (
    "flytekit==1.16.21",
    "qiskit>=2,<3",
    "qiskit-aer>=0.17",
    "numpy>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "dill>=0.3.4",
    "pylatexenc>=2.10",
    "setuptools>=40.1",
)


def _resolve_registry() -> str:
    """返回 ImageSpec 推送的镜像仓。

    本机 flytectl demo 默认自带 ``localhost:30000`` registry。迁移到生产
    Flyte 集群时，通过 ``FLYTE_IMAGE_REGISTRY`` 覆盖即可。
    """

    return os.environ.get("FLYTE_IMAGE_REGISTRY", "localhost:30000")


hybrid_qsvc_image = ImageSpec(
    name="hybrid-qsvc-flyte",
    python_version="3.12",
    packages=list(COMMON_PACKAGES),
    registry=_resolve_registry(),
    source_root=str(REPO_ROOT),
    copy=[
        "qiskit_machine_learning",
        "README.md",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "MANIFEST.in",
    ],
    commands=["pip install --no-deps -e /root"],
)


class TaskKind(str, Enum):
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
    """打印任务类型 banner，方便在 Flyte pod 日志中定位阶段。"""

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}", flush=True)
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}", flush=True)
    print(f"{color}{bold}{bar}{reset}", flush=True)


def task_workdir(prefix: str) -> Path:
    """为单次 task 调用创建绝对路径工作目录。"""

    return Path(tempfile.mkdtemp(prefix=f"hybrid-qsvc-{prefix}-"))


def parse_gamma(value: str) -> float | str:
    """把 CLI / workflow 传入的 gamma 转成 sklearn 可接受的值。"""

    if value in {"scale", "auto"}:
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("rbf_gamma 只能是 'scale'、'auto' 或数字字符串") from exc


def json_ready(value: Any) -> Any:
    """把 numpy 标量等对象转成 JSON 友好的 Python 原生类型。"""

    if hasattr(value, "item"):
        return value.item()
    return value
