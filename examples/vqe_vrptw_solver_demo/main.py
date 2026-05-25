"""VQE-VRPTW solver 流水线入口编排器。

本脚本按顺序通过 subprocess 调用 tasks/ 下的独立 CLI。任务之间只通过
artifacts/ 里的文件传递数据，便于后续迁移到 Flyte、Airflow 或 Dagster。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from pipeline_utils import (
    ANSATZ_DILL,
    ANSATZ_DIR,
    ANSATZ_JSON,
    ARTIFACTS_DIR,
    DECODED_SOLUTION_JSON,
    FIGURES_DIR,
    HAMILTONIAN_DILL,
    HAMILTONIAN_JSON,
    INITIAL_ENERGY_JSON,
    INITIAL_POINT_NPY,
    INSTANCE_DIR,
    INSTANCE_JSON,
    METRICS_JSON,
    QUBO_DIR,
    QUBO_JSON,
    REFERENCE_JSON,
    RESULTS_DIR,
    SAMPLES_CSV,
    SOLUTION_ROUTES_CSV,
    TaskKind,
    VQE_RESULT_DILL,
    VQE_RESULT_JSON,
    VQE_TRACE_CSV,
    ensure_dir,
    print_banner,
)


DEMO_ROOT = Path(__file__).resolve().parent
TASKS_DIR = DEMO_ROOT / "tasks"

FIG_INSTANCE = FIGURES_DIR / "01_instance.png"
FIG_QUBO = FIGURES_DIR / "02_qubo_hamiltonian.png"
FIG_REFERENCE = FIGURES_DIR / "03_reference_routes.png"
FIG_ANSATZ = FIGURES_DIR / "04_ansatz_circuit.png"
FIG_INITIAL = FIGURES_DIR / "05_initial_energy.png"
FIG_CONVERGENCE = FIGURES_DIR / "06_vqe_convergence.png"
FIG_PROBABILITIES = FIGURES_DIR / "07_solution_probabilities.png"
FIG_VQE_ROUTES = FIGURES_DIR / "08_vqe_routes.png"
FIG_SUMMARY = FIGURES_DIR / "09_summary_dashboard.png"
ENV_LOCK = ARTIFACTS_DIR / "environment.lock.yml"
ENV_NAME = "qml-vrptw-solver"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance-input", type=Path, default=None, help="可选：自定义 VRPTW JSON")
    p.add_argument("--time-granularity", type=float, default=1.0, help="离散时间粒度")
    p.add_argument("--max-stops-per-vehicle", type=int, default=2, help="每辆车最多服务客户数")
    p.add_argument("--penalty", type=float, default=0.0, help="约束惩罚；0 表示自动设置")
    p.add_argument("--ansatz-reps", type=int, default=2, help="RY product ansatz 重复层数")
    p.add_argument("--optimizer", default="COBYLA", choices=["COBYLA", "SLSQP", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=120, help="VQE 经典优化器最大迭代次数")
    p.add_argument("--shots", type=int, default=4096, help="Sampler 每次评估的采样次数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exact-max-qubits", type=int, default=22, help="经典参考枚举变量数上限")
    p.add_argument("--clean", action="store_true", help="运行前删除 artifacts/")
    p.add_argument("--no-env-lock", action="store_true", help="跳过 conda 环境导出")
    return p.parse_args()


def run_task(script: str, cli_args: list[str]) -> float:
    cmd = [sys.executable, str(TASKS_DIR / script), *cli_args]
    print(f"\n  $ {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def export_environment_lock() -> None:
    """导出 conda 环境，去掉机器相关 prefix。"""

    conda = shutil.which("conda")
    if conda is None:
        print("  [warn] 未找到 conda，跳过 environment.lock.yml 导出")
        return

    commands = [
        [conda, "env", "export", "-n", ENV_NAME, "--no-builds"],
        [conda, "env", "export", "--no-builds"],
    ]
    result = None
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 环境导出命令失败，尝试下一个: {exc}")
    if result is None:
        print("  [warn] conda env export 失败，跳过环境锁定")
        return

    lines = [line for line in result.stdout.splitlines() if not line.startswith("prefix:")]
    ENV_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> 当前环境已导出到 {ENV_LOCK}")


def main() -> int:
    args = parse_args()
    if args.clean and ARTIFACTS_DIR.exists():
        print(f"--clean: 删除 {ARTIFACTS_DIR}")
        shutil.rmtree(ARTIFACTS_DIR)

    for d in (INSTANCE_DIR, QUBO_DIR, ANSATZ_DIR, RESULTS_DIR, FIGURES_DIR):
        ensure_dir(d)

    print_banner(TaskKind.CLASSICAL, "VQE-VRPTW solver pipeline 启动")
    print(f"  artifacts = {ARTIFACTS_DIR}")
    print(f"  python    = {sys.executable}")
    print(
        "  config    = "
        f"time_granularity={args.time_granularity}, max_stops_per_vehicle={args.max_stops_per_vehicle}, "
        f"penalty={'auto' if args.penalty <= 0 else args.penalty}, ansatz_reps={args.ansatz_reps}, "
        f"optimizer={args.optimizer}, maxiter={args.maxiter}, shots={args.shots}, seed={args.seed}"
    )

    timings: dict[str, float] = {}

    task01_args = [
        "--output",
        str(INSTANCE_JSON),
        "--figure-output",
        str(FIG_INSTANCE),
        "--time-granularity",
        str(args.time_granularity),
        "--max-stops-per-vehicle",
        str(args.max_stops_per_vehicle),
    ]
    if args.instance_input is not None:
        task01_args.extend(["--instance-input", str(args.instance_input)])
    timings["01"] = run_task("task_01_define_or_load_instance.py", task01_args)

    timings["02"] = run_task(
        "task_02_build_time_indexed_qubo.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--penalty",
            str(args.penalty),
            "--qubo-output",
            str(QUBO_JSON),
            "--hamiltonian-output",
            str(HAMILTONIAN_JSON),
            "--hamiltonian-dill",
            str(HAMILTONIAN_DILL),
            "--figure-output",
            str(FIG_QUBO),
        ],
    )

    timings["03"] = run_task(
        "task_03_exact_reference.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--qubo",
            str(QUBO_JSON),
            "--output",
            str(REFERENCE_JSON),
            "--figure-output",
            str(FIG_REFERENCE),
            "--exact-max-qubits",
            str(args.exact_max_qubits),
        ],
    )

    timings["04"] = run_task(
        "task_04_build_ansatz.py",
        [
            "--hamiltonian",
            str(HAMILTONIAN_JSON),
            "--qubo",
            str(QUBO_JSON),
            "--output",
            str(ANSATZ_DILL),
            "--metadata-output",
            str(ANSATZ_JSON),
            "--initial-point-output",
            str(INITIAL_POINT_NPY),
            "--figure-output",
            str(FIG_ANSATZ),
            "--ansatz-reps",
            str(args.ansatz_reps),
            "--seed",
            str(args.seed),
        ],
    )

    timings["05"] = run_task(
        "task_05_initial_energy.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--hamiltonian",
            str(HAMILTONIAN_DILL),
            "--ansatz",
            str(ANSATZ_DILL),
            "--initial-point",
            str(INITIAL_POINT_NPY),
            "--output",
            str(INITIAL_ENERGY_JSON),
            "--figure-output",
            str(FIG_INITIAL),
            "--shots",
            str(args.shots),
            "--seed",
            str(args.seed),
        ],
    )

    timings["06"] = run_task(
        "task_06_run_vqe.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--hamiltonian",
            str(HAMILTONIAN_DILL),
            "--ansatz",
            str(ANSATZ_DILL),
            "--initial-point",
            str(INITIAL_POINT_NPY),
            "--reference",
            str(REFERENCE_JSON),
            "--output-json",
            str(VQE_RESULT_JSON),
            "--output-dill",
            str(VQE_RESULT_DILL),
            "--trace-output",
            str(VQE_TRACE_CSV),
            "--figure-output",
            str(FIG_CONVERGENCE),
            "--optimizer",
            args.optimizer,
            "--maxiter",
            str(args.maxiter),
            "--shots",
            str(args.shots),
            "--seed",
            str(args.seed),
        ],
    )

    timings["07"] = run_task(
        "task_07_decode_solution.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--qubo",
            str(QUBO_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--output",
            str(DECODED_SOLUTION_JSON),
            "--samples-output",
            str(SAMPLES_CSV),
            "--routes-output",
            str(SOLUTION_ROUTES_CSV),
            "--probability-figure-output",
            str(FIG_PROBABILITIES),
            "--route-figure-output",
            str(FIG_VQE_ROUTES),
        ],
    )

    timings["08"] = run_task(
        "task_08_evaluate.py",
        [
            "--reference",
            str(REFERENCE_JSON),
            "--initial-energy",
            str(INITIAL_ENERGY_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--decoded-solution",
            str(DECODED_SOLUTION_JSON),
            "--output",
            str(METRICS_JSON),
        ],
    )

    timings["09"] = run_task(
        "task_09_visualize_summary.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--qubo",
            str(QUBO_JSON),
            "--hamiltonian",
            str(HAMILTONIAN_JSON),
            "--ansatz",
            str(ANSATZ_JSON),
            "--reference",
            str(REFERENCE_JSON),
            "--initial-energy",
            str(INITIAL_ENERGY_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--decoded-solution",
            str(DECODED_SOLUTION_JSON),
            "--metrics",
            str(METRICS_JSON),
            "--trace",
            str(VQE_TRACE_CSV),
            "--samples",
            str(SAMPLES_CSV),
            "--output",
            str(FIG_SUMMARY),
        ],
    )

    print_banner(TaskKind.CLASSICAL, "VQE-VRPTW solver pipeline 完成")
    total = sum(timings.values())
    for key in sorted(timings):
        print(f"  task_{key}: {timings[key]:6.2f}s")
    print(f"  total  : {total:6.2f}s")

    if not args.no_env_lock:
        export_environment_lock()

    print()
    print(f"  metrics -> {METRICS_JSON}")
    print(f"  figures -> {FIGURES_DIR}/")
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        print(f"             {figure.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
