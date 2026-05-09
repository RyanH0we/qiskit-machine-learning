"""Hybrid Quantum-Classical Pipeline 入口编排器。

按顺序通过 ``subprocess`` 调用 tasks/ 下 8 个独立 CLI 脚本,
完全靠文件 artifact 在任务之间传递数据 —— 这就是 flyte/airflow 的
最朴素同构形式: orchestrator + container task + persistent artifact。

为什么用 subprocess 而不是 import + 函数调用?
  - 每个 task 是独立 OS 进程, 资源边界清晰, 失败不会污染编排器
  - 与 flyte container task / dagster job step 的执行模型一致, 后续迁移零摩擦
  - 你可以单独 `python tasks/task_03_quantum_kernel.py ...` 跑任何一步, 调试或替换

用法:
  conda activate qml      # 复用本机已有环境
  python main.py          # 一键全流程

或:
  python main.py --skip-decision-grid    # 跳过 task_08 的网格量子核重算 (省 30s)
  python main.py --grid 50               # 决策边界用更细网格 (会变慢)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 必须早于任何 print —— 否则 subprocess 写 stdout 会与 Python buffer 出现顺序错乱
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from pipeline_utils import (
    ARTIFACTS_DIR, DATA_DIR, FIGURES_DIR, KERNEL_DIR, METRICS_PATH, MODELS_DIR,
    TaskKind, ensure_dir, print_banner,
)

DEMO_ROOT = Path(__file__).resolve().parent
TASKS_DIR = DEMO_ROOT / "tasks"

DATA_NPZ = DATA_DIR / "data.npz"
KERNEL_TRAIN = KERNEL_DIR / "kernel_train.npz"
KERNEL_TEST = KERNEL_DIR / "kernel_test.npz"
FEATURE_MAP_DILL = KERNEL_DIR / "feature_map.dill"
QSVC_DILL = MODELS_DIR / "qsvc.dill"
CSVC_DILL = MODELS_DIR / "classical_svc.dill"

FIG_DATA = FIGURES_DIR / "01_data.png"
FIG_KERNELS = FIGURES_DIR / "02_kernels.png"
FIG_BARS = FIGURES_DIR / "03_results.png"
FIG_DECISION = FIGURES_DIR / "04_decision.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2, help="ad_hoc 特征维度 / 量子比特数")
    p.add_argument("--train", type=int, default=20, help="每类训练样本数")
    p.add_argument("--test", type=int, default=10, help="每类测试样本数")
    p.add_argument("--gap", type=float, default=0.3, help="ad_hoc 标签 gap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=2, help="ZZFeatureMap reps")
    p.add_argument("--c", dest="c", type=float, default=1.0, help="SVC C")
    p.add_argument("--rbf-gamma", default="scale", help="经典 SVC gamma")
    p.add_argument("--grid", type=int, default=30,
                   help="决策边界网格分辨率 (默认 30, 1900 grid pts -> ~30s)")
    p.add_argument("--skip-decision-grid", action="store_true",
                   help="跳过 task_08 的决策边界网格重算 (其他都跑)")
    p.add_argument("--clean", action="store_true",
                   help="开始前先 rm -rf artifacts/")
    return p.parse_args()


def run_task(label: str, kind: TaskKind, script: str, cli_args: list[str]) -> float:  # noqa: ARG001
    """subprocess 跑一个 task,返回 wall-clock 用时(秒)。

    label / kind 形参保留,是为了调用点对每个 task 的"声明式"描述 —— 它们的
    banner 会由 task 进程内部自己打印,这里只额外 trace 一行启动 cmd。
    """
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
    print(f"  artifacts dir = {ARTIFACTS_DIR}")
    print(f"  python        = {sys.executable}")
    print(f"  config        : n={args.n}, train={args.train}, test={args.test}, gap={args.gap}, "
          f"reps={args.reps}, C={args.c}, gamma={args.rbf_gamma}, grid={args.grid}, "
          f"skip_decision_grid={args.skip_decision_grid}, seed={args.seed}")

    timings: dict[str, float] = {}

    timings["01"] = run_task(
        "Task 01 / generate ad_hoc data", TaskKind.CLASSICAL,
        "task_01_generate_data.py",
        ["--n", str(args.n), "--train", str(args.train), "--test", str(args.test),
         "--gap", str(args.gap), "--seed", str(args.seed),
         "--output", str(DATA_NPZ)],
    )

    timings["02"] = run_task(
        "Task 02 / visualize data scatter", TaskKind.CLASSICAL,
        "task_02_visualize_data.py",
        ["--input", str(DATA_NPZ), "--output", str(FIG_DATA)],
    )

    timings["03"] = run_task(
        "Task 03 / compute quantum kernel matrix", TaskKind.QUANTUM,
        "task_03_quantum_kernel.py",
        ["--input", str(DATA_NPZ), "--output-dir", str(KERNEL_DIR),
         "--reps", str(args.reps), "--seed", str(args.seed)],
    )

    timings["04"] = run_task(
        "Task 04 / visualize kernel matrices", TaskKind.CLASSICAL,
        "task_04_visualize_kernel.py",
        ["--quantum-kernel", str(KERNEL_TRAIN), "--output", str(FIG_KERNELS)],
    )

    timings["05"] = run_task(
        "Task 05 / train QSVC (quantum kernel + classical SVM)", TaskKind.HYBRID,
        "task_05_train_qsvc.py",
        ["--kernel-train", str(KERNEL_TRAIN), "--kernel-test", str(KERNEL_TEST),
         "--output", str(QSVC_DILL), "--c", str(args.c), "--seed", str(args.seed)],
    )

    timings["06"] = run_task(
        "Task 06 / train classical RBF SVC baseline", TaskKind.CLASSICAL,
        "task_06_train_classical_svc.py",
        ["--input", str(DATA_NPZ), "--output", str(CSVC_DILL),
         "--c", str(args.c), "--gamma", str(args.rbf_gamma), "--seed", str(args.seed)],
    )

    timings["07"] = run_task(
        "Task 07 / aggregate metrics", TaskKind.CLASSICAL,
        "task_07_evaluate.py",
        ["--qsvc", str(QSVC_DILL), "--classical", str(CSVC_DILL),
         "--output", str(METRICS_PATH)],
    )

    task_08_args = [
        "--data", str(DATA_NPZ), "--metrics", str(METRICS_PATH),
        "--qsvc", str(QSVC_DILL), "--classical", str(CSVC_DILL),
        "--feature-map", str(FEATURE_MAP_DILL),
        "--bar-output", str(FIG_BARS),
    ]
    if args.skip_decision_grid:
        task_08_args.append("--no-decision")
        task_08_label = "Task 08 / final visualization (bars only, decision skipped)"
    else:
        task_08_args += ["--decision-output", str(FIG_DECISION), "--grid", str(args.grid)]
        task_08_label = "Task 08 / final visualization (bars + decision boundary)"
    timings["08"] = run_task(task_08_label, TaskKind.CLASSICAL,
                             "task_08_visualize_results.py", task_08_args)

    # ---- 总结 ----
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
