"""GJQ Yudu 真机混合量子-经典流水线入口编排器。

按顺序通过 ``subprocess`` 调用 tasks/ 下 8 个独立 CLI 脚本，完全靠文件
artifact 在任务之间传递数据。这与 flyte/airflow 的 task + container 执行模型
保持同构：每个 task 是独立进程，失败边界清晰，也便于单独调试。

用法:
  conda activate qml-gjq
  python main.py --clean  # 小规模 Yudu 真机全流程
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 必须早于任何 print，否则 subprocess 写 stdout 会与 Python buffer 出现顺序错乱。
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from pipeline_utils import (
    ARTIFACTS_DIR,
    DATA_DIR,
    FIGURES_DIR,
    KERNEL_DIR,
    METRICS_PATH,
    MODELS_DIR,
    TaskKind,
    ensure_dir,
    print_banner,
)

DEMO_ROOT = Path(__file__).resolve().parent
TASKS_DIR = DEMO_ROOT / "tasks"

DATA_NPZ = DATA_DIR / "data.npz"
KERNEL_TRAIN = KERNEL_DIR / "kernel_train.npz"
KERNEL_TEST = KERNEL_DIR / "kernel_test.npz"
QSVC_DILL = MODELS_DIR / "qsvc.dill"
CSVC_DILL = MODELS_DIR / "classical_svc.dill"

FIG_DATA = FIGURES_DIR / "01_data.png"
FIG_KERNELS = FIGURES_DIR / "02_kernels.png"
FIG_BARS = FIGURES_DIR / "03_results.png"


def parse_optimization_level(value: str) -> str:
    text = value.strip().lower()
    if text == "none":
        return "none"
    try:
        level = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--optimization-level 必须是 none、0、1、2 或 3") from exc
    if level not in {0, 1, 2, 3}:
        raise argparse.ArgumentTypeError("--optimization-level 必须是 none、0、1、2 或 3")
    return str(level)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2, help="ad_hoc 特征维度 / 量子比特数")
    p.add_argument("--train", type=int, default=2, help="每类训练样本数")
    p.add_argument("--test", type=int, default=1, help="每类测试样本数")
    p.add_argument("--gap", type=float, default=0.3, help="ad_hoc 标签 gap")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--reps", type=int, default=2, help="ZZFeatureMap reps")
    p.add_argument("--c", dest="c", type=float, default=1.0, help="SVC C")
    p.add_argument("--rbf-gamma", default="scale", help="经典 SVC gamma")
    p.add_argument("--shots", type=int, default=1024, help="Yudu 采样次数 shots")
    p.add_argument(
        "--optimization-level",
        type=parse_optimization_level,
        default="2",
        help="GJQ 转译优化等级：none、0、1、2 或 3，默认 2",
    )
    p.add_argument("--clean", action="store_true", help="开始前先 rm -rf artifacts/")
    return p.parse_args()


def run_task(label: str, kind: TaskKind, script: str, cli_args: list[str]) -> float:  # noqa: ARG001
    """subprocess 跑一个 task，返回 wall-clock 用时（秒）。"""
    cmd = [sys.executable, str(TASKS_DIR / script), *cli_args]
    print(f"\n  $ {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def main() -> int:
    args = parse_args()

    if args.clean and ARTIFACTS_DIR.exists():
        print(f"--clean: removing {ARTIFACTS_DIR}")
        shutil.rmtree(ARTIFACTS_DIR)
    for d in (DATA_DIR, KERNEL_DIR, MODELS_DIR, FIGURES_DIR):
        ensure_dir(d)

    print_banner(TaskKind.CLASSICAL, "Pipeline launch (orchestrator)")
    print(f"  artifacts dir      = {ARTIFACTS_DIR}")
    print(f"  python             = {sys.executable}")
    print(
        f"  config             : n={args.n}, train={args.train}, test={args.test}, "
        f"gap={args.gap}, reps={args.reps}, C={args.c}, gamma={args.rbf_gamma}, "
        f"seed={args.seed}, backend=Yudu, shots={args.shots}, "
        f"optimization_level={args.optimization_level}"
    )
    print("  真机任务默认规模    : K_train 上三角非对角 + K_test；不绘制决策边界")

    timings: dict[str, float] = {}

    timings["01"] = run_task(
        "Task 01 / generate ad_hoc data",
        TaskKind.CLASSICAL,
        "task_01_generate_data.py",
        [
            "--n",
            str(args.n),
            "--train",
            str(args.train),
            "--test",
            str(args.test),
            "--gap",
            str(args.gap),
            "--seed",
            str(args.seed),
            "--output",
            str(DATA_NPZ),
        ],
    )

    timings["02"] = run_task(
        "Task 02 / visualize data scatter",
        TaskKind.CLASSICAL,
        "task_02_visualize_data.py",
        ["--input", str(DATA_NPZ), "--output", str(FIG_DATA)],
    )

    timings["03"] = run_task(
        "Task 03 / compute quantum kernel matrix on GJQ Yudu",
        TaskKind.QUANTUM,
        "task_03_quantum_kernel.py",
        [
            "--input",
            str(DATA_NPZ),
            "--output-dir",
            str(KERNEL_DIR),
            "--reps",
            str(args.reps),
            "--seed",
            str(args.seed),
            "--shots",
            str(args.shots),
            "--optimization-level",
            args.optimization_level,
        ],
    )

    timings["04"] = run_task(
        "Task 04 / visualize kernel matrices",
        TaskKind.CLASSICAL,
        "task_04_visualize_kernel.py",
        ["--quantum-kernel", str(KERNEL_TRAIN), "--output", str(FIG_KERNELS)],
    )

    timings["05"] = run_task(
        "Task 05 / train QSVC (quantum kernel + classical SVM)",
        TaskKind.HYBRID,
        "task_05_train_qsvc.py",
        [
            "--kernel-train",
            str(KERNEL_TRAIN),
            "--kernel-test",
            str(KERNEL_TEST),
            "--output",
            str(QSVC_DILL),
            "--c",
            str(args.c),
            "--seed",
            str(args.seed),
        ],
    )

    timings["06"] = run_task(
        "Task 06 / train classical RBF SVC baseline",
        TaskKind.CLASSICAL,
        "task_06_train_classical_svc.py",
        [
            "--input",
            str(DATA_NPZ),
            "--output",
            str(CSVC_DILL),
            "--c",
            str(args.c),
            "--gamma",
            str(args.rbf_gamma),
            "--seed",
            str(args.seed),
        ],
    )

    timings["07"] = run_task(
        "Task 07 / aggregate metrics",
        TaskKind.CLASSICAL,
        "task_07_evaluate.py",
        ["--qsvc", str(QSVC_DILL), "--classical", str(CSVC_DILL), "--output", str(METRICS_PATH)],
    )

    timings["08"] = run_task(
        "Task 08 / final visualization (accuracy bars only)",
        TaskKind.CLASSICAL,
        "task_08_visualize_results.py",
        ["--metrics", str(METRICS_PATH), "--bar-output", str(FIG_BARS)],
    )

    print_banner(TaskKind.CLASSICAL, "Pipeline complete")
    total = sum(timings.values())
    print("  per-task wall-clock:")
    for k in sorted(timings):
        print(f"    task_{k}  {timings[k]:>6.2f}s")
    print(f"  total      {total:>6.2f}s")
    print()
    print(f"  metrics  -> {METRICS_PATH}")
    print(f"  figures  -> {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"             {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
