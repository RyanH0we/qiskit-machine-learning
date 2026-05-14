"""[CLASSICAL] Task 04 -- 可视化量子核 vs 经典 RBF 核的训练核矩阵对比。

读 task_03 输出的量子核矩阵，再用 sklearn 算同一份训练数据的经典 RBF 核矩阵,
画两张并排热力图。预期效果：
  - 量子核矩阵呈现明显的"块状"结构(同类样本更亲近),反映 ZZFeatureMap 的特征表达
  - 经典 RBF 核矩阵几乎是个对角主导的"局部高斯"结构,缺乏全局判别力
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quantum-kernel", type=Path, required=True, help="task_03 输出的 kernel_train.npz")
    p.add_argument("--output", type=Path, required=True, help="输出 PNG 路径")
    p.add_argument("--rbf-gamma", type=float, default=1.0, help="经典 RBF 核 gamma")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 04 / visualize quantum kernel vs classical RBF kernel")
    print(f"  quantum-kernel = {args.quantum_kernel}")
    print(f"  output         = {args.output}")
    print(f"  rbf gamma      = {args.rbf_gamma}")

    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import rbf_kernel

    npz = np.load(args.quantum_kernel, allow_pickle=True)
    k_quantum = npz["kernel"]
    x_train = npz["x"]
    y_train = npz["y"]

    # 按标签排序,同类聚在一起,块结构更明显
    order = np.argsort(y_train, kind="stable")
    k_quantum_sorted = k_quantum[np.ix_(order, order)]
    x_train_sorted = x_train[order]
    y_train_sorted = y_train[order]

    with Timer("compute RBF kernel"):
        k_rbf_sorted = rbf_kernel(x_train_sorted, gamma=args.rbf_gamma)

    n0 = int((y_train_sorted == 0).sum())

    with Timer("plot heatmaps"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, mat, title in [
            (axes[0], k_quantum_sorted, "Quantum kernel (Fidelity, ZZFeatureMap)"),
            (axes[1], k_rbf_sorted, f"Classical kernel (RBF, gamma={args.rbf_gamma})"),
        ]:
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            # 在类边界上画线,方便看块结构
            ax.axhline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
            ax.axvline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("sample index (sorted by label)")
            ax.set_ylabel("sample index (sorted by label)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            "Training kernel matrices "
            f"(top-left {n0}x{n0} block = class 0; bottom-right = class 1)",
            fontsize=12,
        )
        fig.tight_layout()

    out = ensure_parent_dir(args.output)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out}")

    # 顺便打印一个简单的"块对比"指标:同类块 vs 异类块的均值差。
    # 量子核应有显著差异;经典 RBF 在 ad_hoc 上几乎没有差异。
    same_class_q = (k_quantum_sorted[:n0, :n0].sum() + k_quantum_sorted[n0:, n0:].sum()) / (n0 * n0 + (len(y_train) - n0) ** 2)
    diff_class_q = k_quantum_sorted[:n0, n0:].mean()
    same_class_c = (k_rbf_sorted[:n0, :n0].sum() + k_rbf_sorted[n0:, n0:].sum()) / (n0 * n0 + (len(y_train) - n0) ** 2)
    diff_class_c = k_rbf_sorted[:n0, n0:].mean()
    print(f"  Quantum  : same-class mean = {same_class_q:.4f}, diff-class mean = {diff_class_q:.4f}, gap = {same_class_q - diff_class_q:+.4f}")
    print(f"  Classical: same-class mean = {same_class_c:.4f}, diff-class mean = {diff_class_c:.4f}, gap = {same_class_c - diff_class_c:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
