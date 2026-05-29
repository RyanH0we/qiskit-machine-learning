"""[CLASSICAL] Task 08 -- 输出准确率柱状图和 2D 决策边界。"""

from __future__ import annotations

import json
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class FinalFiguresOut(NamedTuple):
    results_png: FlyteFile
    decision_png: FlyteFile


def _plot_bars(metrics: dict, out_path) -> None:
    import matplotlib.pyplot as plt

    qsvc = metrics["models"]["qsvc"]
    csvc = metrics["models"]["classical_svc"]
    names = [qsvc["name"], csvc["name"]]
    test_accs = [qsvc["test_acc"], csvc["test_acc"]]
    train_accs = [qsvc["train_acc"], csvc["train_acc"]]
    advantage = metrics["quantum_advantage_test_acc"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    x = np.arange(len(names))
    bars_tr = ax.bar(
        x - width / 2,
        train_accs,
        width,
        color=["#d2b4de", "#aed6f1"],
        edgecolor="black",
        linewidth=0.6,
        label="train acc",
    )
    bars_te = ax.bar(
        x + width / 2,
        test_accs,
        width,
        color=["#9b59b6", "#3498db"],
        edgecolor="black",
        linewidth=0.6,
        label="test acc",
    )
    for bars, values in ((bars_tr, train_accs), (bars_te, test_accs)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{name}\n[{metrics['models'][key]['kind']}]"
            for name, key in zip(names, ("qsvc", "classical_svc"))
        ]
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Test accuracy: quantum advantage = {advantage:+.2%}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_decision_boundaries(
    data_npz: FlyteFile,
    qsvc_dill: FlyteFile,
    classical_svc_dill: FlyteFile,
    feature_map_dill: FlyteFile,
    grid: int,
    out_path,
) -> None:
    import matplotlib.pyplot as plt
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    if grid < 2:
        raise ValueError("grid 必须 >= 2")

    npz = np.load(data_npz.download(), allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]

    with open(feature_map_dill.download(), "rb") as f:
        fm_pkg = dill.load(f)
    with open(qsvc_dill.download(), "rb") as f:
        qsvc_bundle = dill.load(f)
    with open(classical_svc_dill.download(), "rb") as f:
        csvc_bundle = dill.load(f)

    xs = np.linspace(0, 2 * np.pi, grid)
    ys = np.linspace(0, 2 * np.pi, grid)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    print(f"  decision grid: {grid}x{grid} = {grid * grid} points", flush=True)
    quantum_kernel = FidelityQuantumKernel(feature_map=fm_pkg["feature_map"])
    k_grid = quantum_kernel.evaluate(x_vec=grid_points, y_vec=x_train)

    z_q = qsvc_bundle["model"].predict(k_grid).reshape(grid, grid)
    z_c = csvc_bundle["model"].predict(grid_points).reshape(grid, grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, z_values, title, kind in (
        (axes[0], z_q, qsvc_bundle["name"], qsvc_bundle["kind"]),
        (axes[1], z_c, csvc_bundle["name"], csvc_bundle["kind"]),
    ):
        ax.contourf(
            xx,
            yy,
            z_values,
            levels=[-0.5, 0.5, 1.5],
            colors=["#cce5ff", "#ffd2cf"],
            alpha=0.7,
        )
        for cls, marker, color, label in (
            (0, "o", "tab:blue", "class 0"),
            (1, "s", "tab:red", "class 1"),
        ):
            mtr = y_train == cls
            mte = y_test == cls
            ax.scatter(
                x_train[mtr, 0],
                x_train[mtr, 1],
                c=color,
                marker=marker,
                s=55,
                edgecolors="black",
                linewidth=0.6,
                label=f"{label} (train)",
            )
            ax.scatter(
                x_test[mte, 0],
                x_test[mte, 1],
                c=color,
                marker=marker,
                s=120,
                edgecolors="black",
                linewidth=1.6,
                alpha=0.95,
                label=f"{label} (test)",
            )
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title(f"{title}  [{kind}]")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    seen: set[str] = set()
    kept_handles = []
    kept_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            kept_handles.append(handle)
            kept_labels.append(label)
    fig.legend(kept_handles, kept_labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Decision boundaries on $[0, 2\\pi]^2$", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t08_visualize_results(
    data_npz: FlyteFile,
    metrics_json: FlyteFile,
    qsvc_dill: FlyteFile,
    classical_svc_dill: FlyteFile,
    feature_map_dill: FlyteFile,
    grid: int = 30,
) -> FinalFiguresOut:
    """[CLASSICAL] 生成 ``03_results.png`` 与 ``04_decision.png``。"""

    print_banner(TaskKind.CLASSICAL, "Task 08 / final visualization")

    import matplotlib

    matplotlib.use("Agg")

    metrics = json.loads(open(metrics_json.download(), encoding="utf-8").read())
    work = task_workdir("t08")
    results_png = work / "03_results.png"
    decision_png = work / "04_decision.png"

    _plot_bars(metrics, results_png)
    _plot_decision_boundaries(
        data_npz=data_npz,
        qsvc_dill=qsvc_dill,
        classical_svc_dill=classical_svc_dill,
        feature_map_dill=feature_map_dill,
        grid=grid,
        out_path=decision_png,
    )

    print(f"  -> wrote {results_png}", flush=True)
    print(f"  -> wrote {decision_png}", flush=True)
    return FinalFiguresOut(
        results_png=FlyteFile(str(results_png)),
        decision_png=FlyteFile(str(decision_png)),
    )
