"""VQE-VRPTW 流水线入口编排器。

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
    ANSATZ_DIR,
    ANSATZ_DILL,
    ANSATZ_JSON,
    ARTIFACTS_DIR,
    FIGURES_DIR,
    HAMILTONIAN_DILL,
    HAMILTONIAN_JSON,
    INITIAL_POINT_NPY,
    INITIAL_RESULT_JSON,
    INSTANCE_DIR,
    INSTANCE_JSON,
    METRICS_JSON,
    QUBO_DIR,
    QUBO_JSON,
    REFERENCE_JSON,
    RESULTS_DIR,
    ROUTES_CSV,
    ROUTES_DIR,
    ROUTES_JSON,
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
FIG_ROUTES = FIGURES_DIR / "02_routes.png"
FIG_QUBO = FIGURES_DIR / "03_qubo_hamiltonian.png"
FIG_REFERENCE = FIGURES_DIR / "04_exact_reference.png"
FIG_ANSATZ = FIGURES_DIR / "05_ansatz_circuit.png"
FIG_INITIAL = FIGURES_DIR / "06_initial_quantum.png"
FIG_VQE = FIGURES_DIR / "07_vqe_convergence.png"
FIG_DASHBOARD = FIGURES_DIR / "08_summary_dashboard.png"
FIG_BEST_ROUTE = FIGURES_DIR / "09_best_route_map.png"
ENV_LOCK = ARTIFACTS_DIR / "environment.lock.yml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", default="COBYLA", choices=["COBYLA", "SLSQP", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=120, help="VQE 经典优化器最大迭代次数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ansatz-reps", type=int, default=2, help="RealAmplitudes 层数")
    p.add_argument("--penalty", type=float, default=0.0, help="one-hot 约束惩罚；0 表示自动设置")
    p.add_argument("--wait-weight", type=float, default=0.2, help="等待时间在路线代价中的权重")
    p.add_argument("--late-weight", type=float, default=8.0, help="迟到时间在路线代价中的权重")
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
    """导出当前 conda 环境，去掉机器相关 prefix。"""

    conda = shutil.which("conda")
    if conda is None:
        print("  [warn] 未找到 conda，跳过 environment.lock.yml 导出")
        return
    try:
        result = subprocess.run(
            [conda, "env", "export", "--no-builds"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] conda env export 失败，跳过环境锁定: {exc}")
        return

    lines = [line for line in result.stdout.splitlines() if not line.startswith("prefix:")]
    ENV_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> 当前环境已导出到 {ENV_LOCK}")


def main() -> int:
    args = parse_args()
    if args.clean and ARTIFACTS_DIR.exists():
        print(f"--clean: 删除 {ARTIFACTS_DIR}")
        shutil.rmtree(ARTIFACTS_DIR)

    for d in (INSTANCE_DIR, ROUTES_DIR, QUBO_DIR, ANSATZ_DIR, RESULTS_DIR, FIGURES_DIR):
        ensure_dir(d)

    print_banner(TaskKind.CLASSICAL, "VQE-VRPTW pipeline 启动")
    print(f"  artifacts = {ARTIFACTS_DIR}")
    print(f"  python    = {sys.executable}")
    print(
        "  config    = "
        f"optimizer={args.optimizer}, maxiter={args.maxiter}, seed={args.seed}, "
        f"ansatz_reps={args.ansatz_reps}, wait_weight={args.wait_weight}, "
        f"late_weight={args.late_weight}, penalty={'auto' if args.penalty <= 0 else args.penalty}"
    )

    timings: dict[str, float] = {}

    timings["01"] = run_task(
        "task_01_define_instance.py",
        [
            "--output",
            str(INSTANCE_JSON),
            "--figure-output",
            str(FIG_INSTANCE),
        ],
    )

    timings["02"] = run_task(
        "task_02_enumerate_routes.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--output-json",
            str(ROUTES_JSON),
            "--output-csv",
            str(ROUTES_CSV),
            "--figure-output",
            str(FIG_ROUTES),
            "--wait-weight",
            str(args.wait_weight),
            "--late-weight",
            str(args.late_weight),
        ],
    )

    timings["03"] = run_task(
        "task_03_build_qubo.py",
        [
            "--routes",
            str(ROUTES_JSON),
            "--qubo-output",
            str(QUBO_JSON),
            "--hamiltonian-output",
            str(HAMILTONIAN_JSON),
            "--hamiltonian-dill",
            str(HAMILTONIAN_DILL),
            "--figure-output",
            str(FIG_QUBO),
            "--penalty",
            str(args.penalty),
        ],
    )

    timings["04"] = run_task(
        "task_04_exact_reference.py",
        [
            "--qubo",
            str(QUBO_JSON),
            "--routes",
            str(ROUTES_JSON),
            "--output",
            str(REFERENCE_JSON),
            "--figure-output",
            str(FIG_REFERENCE),
        ],
    )

    timings["05"] = run_task(
        "task_05_build_ansatz.py",
        [
            "--hamiltonian",
            str(HAMILTONIAN_JSON),
            "--output",
            str(ANSATZ_DILL),
            "--metadata-output",
            str(ANSATZ_JSON),
            "--initial-point-output",
            str(INITIAL_POINT_NPY),
            "--figure-output",
            str(FIG_ANSATZ),
            "--reps",
            str(args.ansatz_reps),
        ],
    )

    timings["06"] = run_task(
        "task_06_initial_quantum.py",
        [
            "--hamiltonian-dill",
            str(HAMILTONIAN_DILL),
            "--qubo",
            str(QUBO_JSON),
            "--routes",
            str(ROUTES_JSON),
            "--ansatz",
            str(ANSATZ_DILL),
            "--initial-point",
            str(INITIAL_POINT_NPY),
            "--output",
            str(INITIAL_RESULT_JSON),
            "--figure-output",
            str(FIG_INITIAL),
            "--seed",
            str(args.seed),
        ],
    )

    timings["07"] = run_task(
        "task_07_run_vqe.py",
        [
            "--hamiltonian-dill",
            str(HAMILTONIAN_DILL),
            "--qubo",
            str(QUBO_JSON),
            "--routes",
            str(ROUTES_JSON),
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
            str(FIG_VQE),
            "--optimizer",
            args.optimizer,
            "--maxiter",
            str(args.maxiter),
            "--seed",
            str(args.seed),
        ],
    )

    timings["08"] = run_task(
        "task_08_visualize_summary.py",
        [
            "--instance",
            str(INSTANCE_JSON),
            "--routes",
            str(ROUTES_JSON),
            "--qubo",
            str(QUBO_JSON),
            "--hamiltonian",
            str(HAMILTONIAN_JSON),
            "--reference",
            str(REFERENCE_JSON),
            "--ansatz",
            str(ANSATZ_JSON),
            "--initial",
            str(INITIAL_RESULT_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--trace",
            str(VQE_TRACE_CSV),
            "--metrics-output",
            str(METRICS_JSON),
            "--dashboard-output",
            str(FIG_DASHBOARD),
            "--route-output",
            str(FIG_BEST_ROUTE),
        ],
    )

    print_banner(TaskKind.CLASSICAL, "VQE-VRPTW pipeline 完成")
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
