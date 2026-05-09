"""可视化工具集合。

所有函数都接受 ``out_path`` 参数；为 None 时只显示不保存。
所有保存路径都会使用 dpi=120 的高清 PNG。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _save_or_show(fig, out_path: str | Path | None) -> None:
    if out_path is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_samples(
    images: np.ndarray,
    labels: np.ndarray,
    n: int = 8,
    out_path: str | Path | None = None,
) -> None:
    """显示 n 张原始 8x8 数字图（带标签）。"""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"label={int(labels[i])}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sample digits (raw 8x8)", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_circuit(circuit, out_path: str | Path | None = None, fold: int = -1) -> None:
    """画出 QCNN 完整电路（feature map + ansatz）。

    用 matplotlib 后端，避免 latex 依赖。
    """
    fig = circuit.draw(output="mpl", fold=fold, style="iqp")
    if out_path is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    loss_history: Sequence[float],
    out_path: str | Path | None = None,
) -> None:
    """绘制 COBYLA 训练损失下降曲线。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="tab:blue", linewidth=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value (squared error)")
    ax.set_title("QCNN training curve (COBYLA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str | int] = (0, 1),
    out_path: str | Path | None = None,
    title: str = "QCNN confusion matrix",
) -> None:
    """绘制混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_misclassified(
    raw_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_n: int = 6,
    out_path: str | Path | None = None,
) -> bool:
    """显示错分样本（最多 max_n 张）。返回是否真的画了图。"""
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return False
    show_idx = wrong_idx[:max_n]
    n = len(show_idx)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.0), squeeze=False)
    for ax, idx in zip(axes[0], show_idx):
        ax.imshow(raw_images[idx], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"true={int(y_true[idx])}\npred={int(y_pred[idx])}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Misclassified samples ({len(wrong_idx)} total)", fontsize=11)
    fig.tight_layout()
    _save_or_show(fig, out_path)
    return True


def plot_comparison(
    results: dict[str, dict[str, float]],
    out_path: str | Path | None = None,
) -> None:
    """对比柱状图：左 test accuracy，右 训练耗时（秒，log 尺度）。

    Args:
        results: 形如 {"QCNN": {"test_acc": 0.97, "train_time": 700.0}, ...}
    """
    names = list(results.keys())
    accs = [results[n]["test_acc"] for n in names]
    times = [results[n]["train_time"] for n in names]

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(10, 4))

    bars = ax_acc.bar(names, accs, color=["tab:purple", "tab:orange", "tab:green"][: len(names)])
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_title("Test accuracy comparison")
    ax_acc.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            fontsize=10,
        )

    bars2 = ax_time.bar(
        names, times, color=["tab:purple", "tab:orange", "tab:green"][: len(names)]
    )
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Training time (s, log scale)")
    ax_time.set_title("Training time comparison")
    ax_time.grid(True, axis="y", alpha=0.3, which="both")
    for bar, t in zip(bars2, times):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{t:.1f}s",
            ha="center",
            fontsize=10,
        )

    fig.suptitle("QCNN vs Classical baselines", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, out_path)
