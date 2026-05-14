"""[CLASSICAL] Task 08 -- 出最终准确率柱状图。

- 03_results.png: 准确率柱状图(QSVC vs Classical SVC),并标注量子优势。

GJQ Yudu 真机版默认不画 2D 决策边界，因为那需要为 grid^2 * n_train
个点重新提交硬件量子核任务。这里保持 task 08 为纯经典可视化，避免任何本地量子模拟器。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True, help="task_07 输出的 metrics.json")
    p.add_argument("--bar-output", type=Path, required=True, help="柱状图 PNG 路径")
    return p.parse_args()


def _plot_bars(metrics: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    qsvc = metrics["models"]["qsvc"]
    csvc = metrics["models"]["classical_svc"]
    names = [qsvc["name"], csvc["name"]]
    test_accs = [qsvc["test_acc"], csvc["test_acc"]]
    train_accs = [qsvc["train_acc"], csvc["train_acc"]]
    advantage = metrics["quantum_advantage_test_acc"]
    colors_test = ["#9b59b6", "#3498db"]
    colors_train = ["#d2b4de", "#aed6f1"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    x = np.arange(len(names))
    bars_tr = ax.bar(x - width / 2, train_accs, width, color=colors_train,
                     edgecolor="black", linewidth=0.6, label="train acc")
    bars_te = ax.bar(x + width / 2, test_accs, width, color=colors_test,
                     edgecolor="black", linewidth=0.6, label="test acc")
    for bars, vals in ((bars_tr, train_accs), (bars_te, test_accs)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n[{metrics['models'][k]['kind']}]"
                        for n, k in zip(names, ["qsvc", "classical_svc"])])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(
        f"Test accuracy: quantum advantage = {advantage:+.2%}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 08 / final visualization (accuracy bars only)")
    print(f"  metrics    = {args.metrics}")
    print(f"  bar-output = {args.bar_output}")

    metrics = json.loads(args.metrics.read_text())

    bar_out = ensure_parent_dir(args.bar_output)
    with Timer("plot bars"):
        _plot_bars(metrics, bar_out)
    print(f"  -> wrote {bar_out}")
    print("  decision boundary skipped: GJQ Yudu 版本避免提交网格量子任务")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
