"""[QUANTUM] Task 03 -- 量子核矩阵计算（本流水线唯一一个真正运行量子电路的步骤）。

对每对 (x_i, x_j) 计算 fidelity:

    K[i, j] = |<phi(x_i) | phi(x_j)>|^2

其中 phi 用 ZZFeatureMap(n=2, reps=2) 编码。每个矩阵元素 = 一次量子电路求值
(本地 statevector 模拟器, 不接真实硬件)。

输出:
  - kernel_train.npz : K_train (n_train x n_train), 用于 SVC.fit
  - kernel_test.npz  : K_test  (n_test x n_train),  用于 SVC.predict
  - circuit.png      : ZZFeatureMap 的电路图
  - feature_map.dill : feature map 实例,后续 task_08 画决策边界时复用,无需重新构建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_dir, ensure_parent_dir, print_banner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="task_01 输出的 .npz")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--reps", type=int, default=2, help="ZZFeatureMap 的重复次数")
    p.add_argument("--entanglement", default="linear",
                   choices=["linear", "circular", "full"], help="ZZFeatureMap 纠缠拓扑")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 03 / compute quantum kernel matrix")
    print(f"  input      = {args.input}")
    print(f"  output-dir = {args.output_dir}")
    print(f"  ZZFeatureMap(reps={args.reps}, entanglement={args.entanglement!r})")

    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = args.seed

    npz = np.load(args.input, allow_pickle=True)
    x_train, x_test = npz["x_train"], npz["x_test"]
    y_train, y_test = npz["y_train"], npz["y_test"]
    n_features = x_train.shape[1]
    print(f"  n_features = {n_features}, n_train = {len(x_train)}, n_test = {len(x_test)}")

    feature_map = zz_feature_map(n_features, reps=args.reps, entanglement=args.entanglement)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    out_dir = ensure_dir(args.output_dir)

    with Timer("evaluate K_train"):
        # K_train[i,j] = |<phi(x_train_i) | phi(x_train_j)>|^2
        # FidelityQuantumKernel 内部会构造 U(x_i)·U†(x_j) 电路,然后测量 |0..0> 概率。
        # 默认后端是 ComputeUncompute fidelity + 本地 Sampler primitive = 状态向量精确模拟。
        k_train = quantum_kernel.evaluate(x_vec=x_train)

    with Timer("evaluate K_test"):
        # K_test[i,j] = fidelity(x_test_i, x_train_j) -- 注意形状是 (n_test, n_train)
        # 这是 sklearn SVC(kernel='precomputed').predict 期望的形状。
        k_test = quantum_kernel.evaluate(x_vec=x_test, y_vec=x_train)

    print(f"  K_train.shape = {k_train.shape}, diag mean = {k_train.diagonal().mean():.4f}")
    print(f"  K_train range = [{k_train.min():.4f}, {k_train.max():.4f}]")
    print(f"  K_test.shape  = {k_test.shape}, range = [{k_test.min():.4f}, {k_test.max():.4f}]")

    np.savez_compressed(
        out_dir / "kernel_train.npz",
        kernel=k_train, x=x_train, y=y_train,
        kind="quantum",
    )
    np.savez_compressed(
        out_dir / "kernel_test.npz",
        kernel=k_test, x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train,
        kind="quantum",
    )
    print(f"  -> wrote {out_dir / 'kernel_train.npz'}")
    print(f"  -> wrote {out_dir / 'kernel_test.npz'}")

    # 画电路图
    circuit_png = out_dir / "circuit.png"
    try:
        with Timer("draw circuit"):
            fig = feature_map.decompose().draw(output="mpl", style="iqp", fold=-1)
            fig.savefig(circuit_png, dpi=120, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
        print(f"  -> wrote {circuit_png}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not draw circuit: {e}")

    # 持久化 feature_map,供 task_08 重建量子核以画决策边界用
    fm_pkl = out_dir / "feature_map.dill"
    with ensure_parent_dir(fm_pkl).open("wb") as f:
        dill.dump(
            {"feature_map": feature_map, "reps": args.reps,
             "entanglement": args.entanglement, "n_features": n_features},
            f,
        )
    print(f"  -> wrote {fm_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
