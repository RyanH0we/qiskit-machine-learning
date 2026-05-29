"""VQE H2 + Flyte 示例的共用工具与 ImageSpec 定义。

把易变与稳定两部分集中管理：

* :data:`vqe_image` —— 所有 ``@task`` 共用的 ``ImageSpec``。改一处镜像配置
  即可影响整条流水线；切到生产集群时，只需要把 ``registry`` 改成你的镜像仓。
* :data:`COMMON_PACKAGES` —— 与 ``requirements-flyte.txt`` 和
  ``environment.yml`` 中 pip 列表保持完全一致。这是本示例可重复运行的关键。
* :func:`create_estimator` —— 默认返回本地 ``StatevectorEstimator``。
  切换到 IBM 真实硬件时只需要改这一处。
* :class:`TaskKind` / :func:`print_banner` —— 与原 ``vqe_h2_pipeline_demo``
  风格一致的 banner，方便新手区分每个 task 是 CLASSICAL / QUANTUM / HYBRID。
"""

from __future__ import annotations

import os
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Iterable

from flytekit import ImageSpec


COMMON_PACKAGES: tuple[str, ...] = (
    "qiskit>=2,<3",
    "qiskit-algorithms==0.4.0",
    "qiskit-nature[pyscf]==0.7.2",
    "qiskit-aer>=0.17",
    "numpy>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "dill>=0.3.4",
    "pylatexenc>=2.10",
)


def _resolve_registry() -> str | None:
    """决定 ImageSpec 推到哪个镜像仓。

    默认推到 flytectl demo sandbox 自带的本地 registry ``localhost:30000``，
    生产部署时通过 ``FLYTE_IMAGE_REGISTRY`` 环境变量覆盖，例如：

        export FLYTE_IMAGE_REGISTRY=ghcr.io/your-org

    使用环境变量而不是硬编码，可以让同一份代码在本地 sandbox 和生产集群上
    完全不改动。
    """

    return os.environ.get("FLYTE_IMAGE_REGISTRY", "localhost:30000")


vqe_image = ImageSpec(
    name="vqe-h2-flyte",
    python_version="3.12",
    packages=list(COMMON_PACKAGES),
    registry=_resolve_registry(),
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
    """打印任务类型 banner。

    在 Flyte pod 日志里看不到 TTY 也没关系，函数会自动退化为纯文本。
    """

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}", flush=True)
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}", flush=True)
    print(f"{color}{bold}{bar}{reset}", flush=True)


def create_estimator(seed: int | None = None):
    """创建量子期望值估计器。

    默认使用本地 ``StatevectorEstimator``（精确、无噪声、无硬件），与
    ``examples/vqe_h2_pipeline_demo`` 完全一致。后续要接 IBM Runtime 时
    只需要替换这一处。
    """

    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator(seed=seed)


def task_workdir(prefix: str) -> Path:
    """为单次 task 调用分配一个绝对路径的工作目录。

    返回 ``Path``，task 内部所有产物文件都应该写在它下面。返回绝对路径有两个
    好处：

    1. ``FlyteFile`` 引用绝对路径，跨容器/跨进程都能稳定找到文件。
    2. flytekit 的本地缓存（``cache=True``）命中时只复用上次返回的 URI；
       如果用相对路径，下次跑时 cwd 不同就会出现 "not a file" 错误。

    远端容器执行时，每个 task pod 都有独立的临时目录；本地执行时也是
    ``tempfile`` 系统级临时目录，互不干扰。
    """

    return Path(tempfile.mkdtemp(prefix=f"vqeh2-{prefix}-"))


def hartree_to_ev(value: float) -> float:
    return value * 27.211386245988


def complex_to_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def join_packages(extra: Iterable[str] = ()) -> list[str]:
    """供 task 级 ``ImageSpec`` 覆盖时合并依赖（暂未使用，给后续扩展留口）。"""

    return list(COMMON_PACKAGES) + list(extra)
