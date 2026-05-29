"""[CLASSICAL] Task 02 -- 把 ad_hoc 数据画成 2D 散点图。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class DataFigureOut(NamedTuple):
    data_png: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t02_visualize_data(data_npz: FlyteFile) -> DataFigureOut:
    """[CLASSICAL] 绘制训练集 / 测试集散点图。"""

    print_banner(TaskKind.CLASSICAL, "Task 02 / visualize data scatter")
    print(f"  data = {data_npz.path}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz = np.load(data_npz.download(), allow_pickle=True)
    x_tr, y_tr, x_te, y_te = npz["x_train"], npz["y_train"], npz["x_test"], npz["y_test"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (x, y, title) in zip(
        axes,
        (
            (x_tr, y_tr, f"Training set (n={len(y_tr)})"),
            (x_te, y_te, f"Test set (n={len(y_te)})"),
        ),
    ):
        for cls, marker, color in ((0, "o", "tab:blue"), (1, "s", "tab:red")):
            mask = y == cls
            ax.scatter(
                x[mask, 0],
                x[mask, 1],
                marker=marker,
                c=color,
                s=70,
                edgecolors="black",
                linewidth=0.6,
                label=f"class {cls}",
            )
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title(title)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("ad_hoc_data (n=2): non-trivially separable in $[0, 2\\pi]^2$")
    fig.tight_layout()

    work = task_workdir("t02")
    out = work / "01_data.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out}", flush=True)
    return DataFigureOut(data_png=FlyteFile(str(out)))
