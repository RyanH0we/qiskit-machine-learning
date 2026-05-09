"""QCNN 手写数字二分类 demo —— 一键运行脚本。

运行方式：
    cd examples/qcnn_digits_demo
    python qcnn_demo.py

完成后会在 outputs/ 下生成：
    samples.png             - 原始数字图像样例
    circuit.png             - QCNN 完整电路图
    training_curve.png      - 训练损失下降曲线
    confusion_matrix.png    - 混淆矩阵
    misclassified.png       - 错分样本（如果有）
    comparison.png          - QCNN vs 经典基线对比
    qcnn_model.dill         - 训练好的模型（可重新加载推理）

用 ``--help`` 查看可调参数。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.classical_baselines import train_mlp, train_svm  # noqa: E402
from src.data import load_and_prepare_data  # noqa: E402
from src.model import build_qcnn  # noqa: E402
from src import viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer",
        choices=["SPSA", "COBYLA", "L_BFGS_B"],
        default="SPSA",
        help=(
            "优化器: SPSA (默认, 推荐, 每步只 2 次电路求值, 高维友好); "
            "COBYLA (无梯度, 简单); "
            "L_BFGS_B (基于 ParamShift 梯度, 每步昂贵但精确)"
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=120,
        help="优化器最大迭代次数 (SPSA 推荐 100-200, COBYLA 推荐 300+, L_BFGS_B 推荐 30-60)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--classes",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("A", "B"),
        help="参与二分类的两个数字标签 (默认 0 1)",
    )
    parser.add_argument(
        "--data-mode",
        choices=["spatial", "pca"],
        default="spatial",
        help=(
            "数据预处理模式: spatial (默认, 8x8 -> 2x4 平均池化, 保留空间结构, "
            "QCNN 卷积/池化层能正常学习); pca (PCA 64 -> 8, 抽象但 QCNN 难以收敛, 仅作对照)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "outputs",
        help="输出目录 (默认 ./outputs)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="跳过经典 SVM/MLP 基线（默认运行）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设种子（必须在导入 algorithm_globals 之前不行，所以这里重新设）
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = args.seed
    np.random.seed(args.seed)

    print("=" * 60)
    print(" QCNN 手写数字二分类 demo")
    print("=" * 60)
    print(f"  类别        : {args.classes[0]} vs {args.classes[1]}")
    print(f"  最大迭代次数: {args.maxiter}")
    print(f"  随机种子    : {args.seed}")
    print(f"  输出目录    : {out_dir}")
    print()

    # ------------------------------------------------------------------
    # 1) 数据加载与预处理
    # ------------------------------------------------------------------
    print("[1/6] 加载并预处理数据 ...")
    data = load_and_prepare_data(
        classes=tuple(args.classes),
        test_size=0.2,
        n_components=8,
        seed=args.seed,
        mode=args.data_mode,
    )
    print(
        f"      模式 {data.mode!r}, 训练集 {data.x_train.shape[0]} 样本，"
        f"测试集 {data.x_test.shape[0]} 样本，特征维度 {data.x_train.shape[1]}"
    )

    print("      画样本图 -> samples.png")
    viz.plot_samples(
        data.raw_train_images, data.y_train, n=8, out_path=out_dir / "samples.png"
    )

    # ------------------------------------------------------------------
    # 2) 构建 QCNN
    # ------------------------------------------------------------------
    print("[2/6] 构建 8 比特 QCNN ...")
    bundle = build_qcnn(num_qubits=8)
    n_weights = len(bundle.ansatz.parameters)
    print(f"      ansatz 可训练参数数 = {n_weights}")
    print("      画电路图 -> circuit.png")
    viz.plot_circuit(bundle.full_circuit.decompose(), out_path=out_dir / "circuit.png")

    # ------------------------------------------------------------------
    # 3) 训练
    # ------------------------------------------------------------------
    print(f"[3/6] 训练 QCNN ({args.optimizer} maxiter={args.maxiter}) ...")
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA

    loss_history: list[float] = []

    def callback(*args):
        # 兼容两种签名：
        #   - SciPy 优化器 (COBYLA, L_BFGS_B): (weights, loss)
        #   - SPSA: (nfev, params, loss, stepsize, accepted)
        if len(args) == 2:
            loss = args[1]
        elif len(args) >= 3:
            loss = args[2]
        else:
            return
        loss_history.append(float(loss))
        if len(loss_history) % 5 == 0 or len(loss_history) == 1:
            print(f"      iter {len(loss_history):3d} | loss = {float(loss):.4f}")

    # 小幅初始化（区间 [-0.5, 0.5]）有助于避开 barren plateau
    initial_point = (algorithm_globals.random.random(n_weights) - 0.5) * 1.0

    if args.optimizer == "L_BFGS_B":
        optimizer = L_BFGS_B(maxiter=args.maxiter)
    elif args.optimizer == "SPSA":
        # SPSA 的 callback 签名不同：(nfev, params, loss, stepsize, accepted)
        # NeuralNetworkClassifier 不会直接把 callback 传给 SPSA，所以这里我们
        # 让 NeuralNetworkClassifier 在每次目标函数求值时调用 callback。
        optimizer = SPSA(maxiter=args.maxiter)
    else:
        optimizer = COBYLA(maxiter=args.maxiter)

    classifier = NeuralNetworkClassifier(
        neural_network=bundle.qnn,
        loss="squared_error",
        optimizer=optimizer,
        callback=callback,
        initial_point=initial_point,
    )

    # 把 {0, 1} 映射到 {-1, +1}，与 Z 期望值的 [-1, 1] 输出对齐
    y_train_pm = 2 * data.y_train - 1
    y_test_pm = 2 * data.y_test - 1

    t_start = time.perf_counter()
    classifier.fit(data.x_train, y_train_pm)
    qcnn_time = time.perf_counter() - t_start
    print(f"      训练完成，用时 {qcnn_time:.1f} s，迭代次数 {len(loss_history)}")

    print("      画训练曲线 -> training_curve.png")
    viz.plot_training_curve(loss_history, out_path=out_dir / "training_curve.png")

    # ------------------------------------------------------------------
    # 4) 评估
    # ------------------------------------------------------------------
    print("[4/6] 在测试集上评估 ...")
    qcnn_train_acc = float(classifier.score(data.x_train, y_train_pm))
    qcnn_test_acc = float(classifier.score(data.x_test, y_test_pm))
    y_pred_pm = classifier.predict(data.x_test)
    # 预测回到 {0,1}
    y_pred = ((np.asarray(y_pred_pm).ravel() + 1) // 2).astype(np.int64)
    print(f"      train acc = {qcnn_train_acc:.4f}")
    print(f"      test  acc = {qcnn_test_acc:.4f}")

    viz.plot_confusion_matrix(
        data.y_test,
        y_pred,
        classes=tuple(str(c) for c in args.classes),
        out_path=out_dir / "confusion_matrix.png",
    )
    has_misclf = viz.plot_misclassified(
        data.raw_test_images,
        data.y_test,
        y_pred,
        max_n=6,
        out_path=out_dir / "misclassified.png",
    )
    if not has_misclf:
        print("      测试集上没有错分样本！(misclassified.png 未生成)")

    # ------------------------------------------------------------------
    # 5) 模型保存与加载验证
    # ------------------------------------------------------------------
    print("[5/6] 保存与重新加载模型 ...")
    model_path = out_dir / "qcnn_model.dill"
    classifier.to_dill(str(model_path))
    loaded = NeuralNetworkClassifier.from_dill(str(model_path))
    reloaded_acc = float(loaded.score(data.x_test, y_test_pm))
    print(f"      原模型 acc = {qcnn_test_acc:.4f}, 加载后 acc = {reloaded_acc:.4f}")
    assert abs(qcnn_test_acc - reloaded_acc) < 1e-9, "保存/加载后准确率不一致！"

    # ------------------------------------------------------------------
    # 6) 经典基线对比
    # ------------------------------------------------------------------
    results = {
        "QCNN": {"test_acc": qcnn_test_acc, "train_time": qcnn_time},
    }

    if not args.skip_baselines:
        print("[6/6] 训练经典基线 (SVM / MLP) ...")
        svm_res = train_svm(
            data.x_train, data.y_train, data.x_test, data.y_test, seed=args.seed
        )
        mlp_res = train_mlp(
            data.x_train, data.y_train, data.x_test, data.y_test, seed=args.seed
        )
        for r in (svm_res, mlp_res):
            print(
                f"      {r.name:14s} train_acc={r.train_acc:.4f} "
                f"test_acc={r.test_acc:.4f} time={r.train_time:.3f}s"
            )
            results[r.name] = {"test_acc": r.test_acc, "train_time": r.train_time}

        viz.plot_comparison(results, out_path=out_dir / "comparison.png")
        print("      已生成 comparison.png")
    else:
        print("[6/6] 跳过经典基线 (--skip-baselines)")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(" 全部完成！输出文件位于:", out_dir)
    print("=" * 60)
    for name, info in results.items():
        print(f"  {name:14s}: test_acc={info['test_acc']:.4f}, train_time={info['train_time']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
