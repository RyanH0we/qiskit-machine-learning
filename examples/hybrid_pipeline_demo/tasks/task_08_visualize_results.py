"""[CLASSICAL] Task 08 -- 出最终的两张对比图: 准确率柱状图 + 2D 决策边界。

- 03_results.png: 准确率柱状图(QSVC vs Classical SVC),并标注量子优势。
- 04_decision.png: 在 [0, 2π]^2 网格上对两个模型分别画决策边界 contour,
                  叠加训练/测试样本散点。这能直观看到"量子核学到了 ad_hoc 数据
                  集的真实棋盘式分布,经典核没学到"。

为画 QSVC 的决策边界,需要在新网格点上重算量子核 K_grid_vs_train ——
这是流水线里第二次(也是最后一次)调用量子电路。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="task_01 输出的 .npz")
    p.add_argument("--metrics", type=Path, required=True, help="task_07 输出的 metrics.json")
    p.add_argument("--qsvc", type=Path, required=True, help="task_05 输出的 .dill")
    p.add_argument("--classical", type=Path, required=True, help="task_06 输出的 .dill")
    p.add_argument("--feature-map", type=Path, required=True,
                   help="task_03 输出的 feature_map.dill")
    p.add_argument("--bar-output", type=Path, required=True, help="柱状图 PNG 路径")
    p.add_argument("--decision-output", type=Path, required=False, default=None,
                   help="决策边界 PNG 路径; 设为 None 或加 --no-decision 时跳过")
    p.add_argument("--no-decision", action="store_true",
                   help="跳过决策边界绘制(省 ~30s); --decision-output 也会被忽略")
    p.add_argument("--grid", type=int, default=40,
                   help="决策边界网格分辨率(每边); grid=40 => 1600 点 -> 1600 次量子核求值, 约 30s")
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


def _plot_decision_boundaries(
    args: argparse.Namespace,
    qsvc_bundle: dict,
    csvc_bundle: dict,
    out_path: Path,
) -> None:
    """画两张并排 contour 图。注意 QSVC 这边需要重算量子核(grid_vs_train)。"""
    import matplotlib.pyplot as plt
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    npz = np.load(args.data, allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]

    with args.feature_map.open("rb") as f:
        fm_pkg = dill.load(f)
    feature_map = fm_pkg["feature_map"]

    g = args.grid
    xs = np.linspace(0, 2 * np.pi, g)
    ys = np.linspace(0, 2 * np.pi, g)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]  # (g*g, 2)

    print(f"  decision grid: {g}x{g} = {g * g} points")

    print(f"  recompute quantum kernel grid_vs_train  ({g * g} x {len(x_train)})")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    with Timer("FidelityQuantumKernel grid eval"):
        k_grid = quantum_kernel.evaluate(x_vec=grid_pts, y_vec=x_train)

    qsvc_model = qsvc_bundle["model"]
    csvc_model = csvc_bundle["model"]

    z_q = qsvc_model.predict(k_grid).reshape(g, g)
    z_c = csvc_model.predict(grid_pts).reshape(g, g)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, z, title, kind in [
        (axes[0], z_q, qsvc_bundle["name"], qsvc_bundle["kind"]),
        (axes[1], z_c, csvc_bundle["name"], csvc_bundle["kind"]),
    ]:
        ax.contourf(XX, YY, z, levels=[-0.5, 0.5, 1.5], colors=["#cce5ff", "#ffd2cf"], alpha=0.7)
        for cls, marker, color, label in [
            (0, "o", "tab:blue", "class 0"),
            (1, "s", "tab:red", "class 1"),
        ]:
            mtr = y_train == cls
            mte = y_test == cls
            ax.scatter(x_train[mtr, 0], x_train[mtr, 1],
                       c=color, marker=marker, s=55, edgecolors="black",
                       linewidth=0.6, label=f"{label} (train)")
            ax.scatter(x_test[mte, 0], x_test[mte, 1],
                       c=color, marker=marker, s=120, edgecolors="black",
                       linewidth=1.6, alpha=0.95, label=f"{label} (test)")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
        ax.set_title(f"{title}  [{kind}]")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    # 去重 legend
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Decision boundaries on $[0, 2\\pi]^2$", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 08 / final visualization (bars + decision boundary)")
    print(f"  data            = {args.data}")
    print(f"  metrics         = {args.metrics}")
    print(f"  qsvc            = {args.qsvc}")
    print(f"  classical       = {args.classical}")
    print(f"  feature_map     = {args.feature_map}")
    print(f"  bar-output      = {args.bar_output}")
    print(f"  decision-output = {args.decision_output}")

    metrics = json.loads(args.metrics.read_text())
    with args.qsvc.open("rb") as f:
        qsvc_bundle = dill.load(f)
    with args.classical.open("rb") as f:
        csvc_bundle = dill.load(f)

    bar_out = ensure_parent_dir(args.bar_output)
    with Timer("plot bars"):
        _plot_bars(metrics, bar_out)
    print(f"  -> wrote {bar_out}")

    if args.no_decision or args.decision_output is None:
        print("  [skipped] decision boundary (--no-decision or --decision-output not set)")
        return 0

    dec_out = ensure_parent_dir(args.decision_output)
    _plot_decision_boundaries(args, qsvc_bundle, csvc_bundle, dec_out)
    print(f"  -> wrote {dec_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
