"""[CLASSICAL] Task 06 -- 经典 RBF 核 SVC 基线。

只用 sklearn,完全不碰量子。在与 QSVC 完全相同的训练数据上跑 SVC(kernel='rbf'),
作为对照组。预期结果:测试准确率 显著低于 QSVC,直接量化"经典核学不到 ad_hoc
数据集的隐含结构"这一现象。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="task_01 输出的 .npz")
    p.add_argument("--output", type=Path, required=True, help="输出 .dill 文件")
    p.add_argument("--c", dest="c", type=float, default=1.0, help="SVC 的 C 参数")
    p.add_argument("--gamma", default="scale", help="RBF 核 gamma; sklearn 默认 'scale'")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 06 / train classical RBF SVC baseline")
    print(f"  input  = {args.input}")
    print(f"  output = {args.output}")
    print(f"  C={args.c}  gamma={args.gamma}")

    from sklearn.svm import SVC

    npz = np.load(args.input, allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]

    # 处理 gamma: 数字字符串就转 float
    try:
        gamma_val: float | str = float(args.gamma)
    except ValueError:
        gamma_val = args.gamma

    model = SVC(kernel="rbf", C=args.c, gamma=gamma_val, random_state=args.seed)
    with Timer("SVC.fit (rbf)"):
        model.fit(x_train, y_train)
    with Timer("SVC.predict"):
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

    train_acc = float((y_train_pred == y_train).mean())
    test_acc = float((y_test_pred == y_test).mean())
    print(f"  train_acc = {train_acc:.4f}")
    print(f"  test_acc  = {test_acc:.4f}")

    out = ensure_parent_dir(args.output)
    with out.open("wb") as f:
        dill.dump(
            {
                "name": "Classical SVC (RBF)",
                "kind": "classical",
                "model": model,
                "y_train": y_train, "y_train_pred": y_train_pred,
                "y_test": y_test, "y_test_pred": y_test_pred,
                "train_acc": train_acc, "test_acc": test_acc,
                "C": args.c, "gamma": gamma_val,
            },
            f,
        )
    print(f"  -> wrote {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
