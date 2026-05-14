"""[HYBRID] Task 05 -- 用预先算好的量子核训练 SVC（QSVC 的拆解形式）。

这是真正的"量子-经典混合"步骤:
  - 输入: task_03 产出的 K_train (来自量子电路求 fidelity)
  - 处理: sklearn 的 SVC(kernel="precomputed").fit(K_train, y_train)
          —— 这里完全是经典凸优化器(SMO)在解 SVM 的 dual 问题
  - 输出: 训练好的 sklearn SVC 模型 + 训练集预测结果(用于 task_07 的 metrics)

注意: 我们故意不用 qiskit_machine_learning.algorithms.QSVC ——
     QSVC 内部就是这两步的封装,但封装后看不到清晰的"量子在哪里、经典在哪里"。
     这里手动拆开,流水线意图更显式,也方便后续替换底层 kernel / classifier。
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
    p.add_argument("--kernel-train", type=Path, required=True, help="task_03 输出的 kernel_train.npz")
    p.add_argument("--kernel-test", type=Path, required=True, help="task_03 输出的 kernel_test.npz")
    p.add_argument("--output", type=Path, required=True, help="输出 .dill 文件 (含模型+预测)")
    p.add_argument("--c", dest="c", type=float, default=1.0, help="SVC 的 C 参数 (正则)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.HYBRID, "Task 05 / train QSVC = quantum kernel + classical SVM")
    print(f"  kernel-train = {args.kernel_train}")
    print(f"  kernel-test  = {args.kernel_test}")
    print(f"  output       = {args.output}")
    print(f"  C = {args.c}")

    from sklearn.svm import SVC

    npz_tr = np.load(args.kernel_train, allow_pickle=True)
    npz_te = np.load(args.kernel_test, allow_pickle=True)
    k_train = npz_tr["kernel"]; y_train = npz_tr["y"]
    k_test = npz_te["kernel"]; y_test = npz_te["y_test"]

    # ----- 经典侧: SVM dual 优化 -----
    model = SVC(kernel="precomputed", C=args.c, random_state=args.seed)
    with Timer("SVC.fit on quantum K"):
        model.fit(k_train, y_train)

    with Timer("SVC.predict"):
        y_train_pred = model.predict(k_train)
        y_test_pred = model.predict(k_test)

    train_acc = float((y_train_pred == y_train).mean())
    test_acc = float((y_test_pred == y_test).mean())
    n_sv = int(model.support_.size)
    print(f"  train_acc = {train_acc:.4f}")
    print(f"  test_acc  = {test_acc:.4f}")
    print(f"  #support_vectors = {n_sv} / {len(y_train)}")

    out = ensure_parent_dir(args.output)
    with out.open("wb") as f:
        dill.dump(
            {
                "name": "QSVC (quantum kernel + SVC)",
                "kind": "hybrid",
                "model": model,
                "y_train": y_train, "y_train_pred": y_train_pred,
                "y_test": y_test, "y_test_pred": y_test_pred,
                "train_acc": train_acc, "test_acc": test_acc,
                "n_support_vectors": n_sv,
                "C": args.c,
            },
            f,
        )
    print(f"  -> wrote {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
