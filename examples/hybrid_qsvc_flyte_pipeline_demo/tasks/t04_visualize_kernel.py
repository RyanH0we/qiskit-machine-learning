"""[CLASSICAL] Task 04 -- 可视化量子核 vs 经典 RBF 核。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class KernelFigureOut(NamedTuple):
    kernels_png: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t04_visualize_kernel(
    kernel_train_npz: FlyteFile,
    rbf_gamma_for_plot: float = 1.0,
) -> KernelFigureOut:
    """[CLASSICAL] 生成 ``02_kernels.png`` 热力图对比。"""

    print_banner(TaskKind.CLASSICAL, "Task 04 / visualize quantum vs RBF kernel")
    print(f"  kernel_train = {kernel_train_npz.path}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import rbf_kernel

    npz = np.load(kernel_train_npz.download(), allow_pickle=True)
    k_quantum = npz["kernel"]
    x_train = npz["x"]
    y_train = npz["y"]

    order = np.argsort(y_train, kind="stable")
    k_quantum_sorted = k_quantum[np.ix_(order, order)]
    x_train_sorted = x_train[order]
    y_train_sorted = y_train[order]
    k_rbf_sorted = rbf_kernel(x_train_sorted, gamma=rbf_gamma_for_plot)
    n0 = int((y_train_sorted == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title in (
        (axes[0], k_quantum_sorted, "Quantum kernel (Fidelity, ZZFeatureMap)"),
        (axes[1], k_rbf_sorted, f"Classical kernel (RBF, gamma={rbf_gamma_for_plot})"),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.axhline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
        ax.axvline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("sample index (sorted by label)")
        ax.set_ylabel("sample index (sorted by label)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Training kernel matrices (top-left {n0}x{n0} block = class 0)",
        fontsize=12,
    )
    fig.tight_layout()

    work = task_workdir("t04")
    out = work / "02_kernels.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    same_class_q = (
        k_quantum_sorted[:n0, :n0].sum() + k_quantum_sorted[n0:, n0:].sum()
    ) / (n0 * n0 + (len(y_train) - n0) ** 2)
    diff_class_q = k_quantum_sorted[:n0, n0:].mean()
    print(
        "  Quantum same-class mean = "
        f"{same_class_q:.4f}, diff-class mean = {diff_class_q:.4f}, "
        f"gap = {same_class_q - diff_class_q:+.4f}",
        flush=True,
    )
    print(f"  -> wrote {out}", flush=True)
    return KernelFigureOut(kernels_png=FlyteFile(str(out)))
