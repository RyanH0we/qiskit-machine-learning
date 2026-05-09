"""[CLASSICAL] Task 01 -- 生成 ad_hoc 合成数据集。

用 ``qiskit_machine_learning.datasets.ad_hoc_data`` 生成一个二分类合成数据集。
该数据集的标签函数本身就是用 ZZFeatureMap 在希尔伯特空间里定义的，所以量子核
SVC 能近乎完美分类，而经典 RBF 核 SVC 会显著低于量子核 —— 这是 Havlíček et al.
(Nature 2019) 论文里展示量子核优势的标准 benchmark。

输出 npz 包含: x_train, y_train, x_test, y_test 四个 ndarray。
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
    p.add_argument("--n", type=int, default=2, help="特征维度 / 量子比特数 (推荐 2 以便后续 2D 决策边界可视化)")
    p.add_argument("--train", type=int, default=20, help="每类的训练样本数 (实际训练集大小 = 2 * train)")
    p.add_argument("--test", type=int, default=10, help="每类的测试样本数")
    p.add_argument("--gap", type=float, default=0.3, help="ad_hoc 标签的分离 gap, 越大越易分类")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--output", type=Path, required=True, help="输出 .npz 文件路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 01 / generate ad_hoc data")
    print(f"  n={args.n}  train(per class)={args.train}  test(per class)={args.test}  gap={args.gap}  seed={args.seed}")

    # 这里只用经典 numpy/scipy 生成几何数据,不调用任何量子电路 —— 故标记 CLASSICAL。
    # ad_hoc_data 内部会用一个 V·Z⊗n·V† 算符的期望来给标签,但这是在 numpy 层面的
    # 矩阵运算,等价于经典预计算,与 QSVC 训练时的 statevector 模拟是不同的事。
    from qiskit_machine_learning.datasets import ad_hoc_data
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = args.seed
    np.random.seed(args.seed)

    with Timer("ad_hoc_data"):
        x_train, y_train, x_test, y_test = ad_hoc_data(
            training_size=args.train,
            test_size=args.test,
            n=args.n,
            gap=args.gap,
            one_hot=False,
        )

    print(f"  x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")
    print(f"  x_test.shape  = {x_test.shape},  y_test.shape  = {y_test.shape}")
    print(f"  y_train counts: 0 -> {int((y_train == 0).sum())}, 1 -> {int((y_train == 1).sum())}")

    out = ensure_parent_dir(args.output)
    np.savez_compressed(
        out,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        meta=np.array(
            {"n": args.n, "train_per_class": args.train, "test_per_class": args.test,
             "gap": args.gap, "seed": args.seed},
            dtype=object,
        ),
    )
    print(f"  -> wrote {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
