"""[CLASSICAL] Task 02 -- 把 ad_hoc 数据画成 2D 散点图。

只用 matplotlib，不调用任何量子电路。
输出 PNG: 上下两个子图，左训练集，右测试集，按真实标签上色。
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
    p.add_argument("--input", type=Path, required=True, help="task_01 输出的 .npz")
    p.add_argument("--output", type=Path, required=True, help="输出 PNG 路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 02 / visualize data scatter")
    print(f"  input  = {args.input}")
    print(f"  output = {args.output}")

    import matplotlib.pyplot as plt

    npz = np.load(args.input, allow_pickle=True)
    x_tr, y_tr, x_te, y_te = npz["x_train"], npz["y_train"], npz["x_test"], npz["y_test"]

    with Timer("plot scatter"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, (x, y, title) in zip(
            axes,
            [(x_tr, y_tr, f"Training set (n={len(y_tr)})"),
             (x_te, y_te, f"Test set (n={len(y_te)})")],
        ):
            for cls, marker, color in [(0, "o", "tab:blue"), (1, "s", "tab:red")]:
                mask = y == cls
                ax.scatter(
                    x[mask, 0], x[mask, 1],
                    marker=marker, c=color, s=70, edgecolors="black",
                    linewidth=0.6, label=f"class {cls}",
                )
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$x_1$")
            ax.set_title(title)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
        fig.suptitle("ad_hoc_data (n=2): non-trivially separable in $[0, 2\\pi]^2$", fontsize=12)
        fig.tight_layout()

    out = ensure_parent_dir(args.output)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
