"""VQE H2 流水线入口编排器。

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
    FIGURES_DIR,
    HAMILTONIAN_DIR,
    HAMILTONIAN_JSON,
    INITIAL_ENERGY_JSON,
    INITIAL_POINT_NPY,
    METRICS_JSON,
    MOLECULE_DIR,
    MOLECULE_JSON,
    PROBLEM_DILL,
    REFERENCE_JSON,
    RESULTS_DIR,
    TaskKind,
    VQE_RESULT_DILL,
    VQE_RESULT_JSON,
    VQE_TRACE_CSV,
    ensure_dir,
    print_banner,
)


DEMO_ROOT = Path(__file__).resolve().parent
TASKS_DIR = DEMO_ROOT / "tasks"

FIG_MOLECULE = FIGURES_DIR / "01_molecule.png"
FIG_HAMILTONIAN = FIGURES_DIR / "02_hamiltonian_terms.png"
FIG_REFERENCE = FIGURES_DIR / "03_reference_energy.png"
FIG_ANSATZ = FIGURES_DIR / "04_ansatz_circuit.png"
FIG_INITIAL = FIGURES_DIR / "05_initial_energy.png"
FIG_CONVERGENCE = FIGURES_DIR / "06_vqe_convergence.png"
FIG_SUMMARY = FIGURES_DIR / "07_summary_dashboard.png"
ENV_LOCK = ARTIFACTS_DIR / "environment.lock.yml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bond-length", type=float, default=0.735, help="H-H 键长，单位 Angstrom")
    p.add_argument("--basis", default="sto3g", help="量子化学基组")
    p.add_argument("--charge", type=int, default=0, help="分子总电荷")
    p.add_argument("--spin", type=int, default=0, help="2S；H2 基态默认 0")
    p.add_argument("--optimizer", default="SLSQP", choices=["SLSQP", "COBYLA", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=100, help="VQE 经典优化器最大迭代次数")
    p.add_argument("--seed", type=int, default=42)
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

    for d in (MOLECULE_DIR, HAMILTONIAN_DIR, ANSATZ_DIR, RESULTS_DIR, FIGURES_DIR):
        ensure_dir(d)

    print_banner(TaskKind.CLASSICAL, "VQE H2 pipeline 启动")
    print(f"  artifacts = {ARTIFACTS_DIR}")
    print(f"  python    = {sys.executable}")
    print(
        "  config    = "
        f"bond_length={args.bond_length}, basis={args.basis}, charge={args.charge}, "
        f"spin={args.spin}, optimizer={args.optimizer}, maxiter={args.maxiter}, seed={args.seed}"
    )

    timings: dict[str, float] = {}

    timings["01"] = run_task(
        "task_01_define_molecule.py",
        [
            "--bond-length",
            str(args.bond_length),
            "--basis",
            args.basis,
            "--charge",
            str(args.charge),
            "--spin",
            str(args.spin),
            "--output",
            str(MOLECULE_JSON),
            "--figure-output",
            str(FIG_MOLECULE),
        ],
    )

    timings["02"] = run_task(
        "task_02_build_hamiltonian.py",
        [
            "--molecule",
            str(MOLECULE_JSON),
            "--problem-output",
            str(PROBLEM_DILL),
            "--hamiltonian-output",
            str(HAMILTONIAN_JSON),
            "--figure-output",
            str(FIG_HAMILTONIAN),
        ],
    )

    timings["03"] = run_task(
        "task_03_exact_reference.py",
        [
            "--problem",
            str(PROBLEM_DILL),
            "--output",
            str(REFERENCE_JSON),
            "--figure-output",
            str(FIG_REFERENCE),
        ],
    )

    timings["04"] = run_task(
        "task_04_build_ansatz.py",
        [
            "--problem",
            str(PROBLEM_DILL),
            "--output",
            str(ANSATZ_DILL),
            "--metadata-output",
            str(ANSATZ_JSON),
            "--initial-point-output",
            str(INITIAL_POINT_NPY),
            "--figure-output",
            str(FIG_ANSATZ),
        ],
    )

    timings["05"] = run_task(
        "task_05_initial_energy.py",
        [
            "--problem",
            str(PROBLEM_DILL),
            "--ansatz",
            str(ANSATZ_DILL),
            "--initial-point",
            str(INITIAL_POINT_NPY),
            "--reference",
            str(REFERENCE_JSON),
            "--output",
            str(INITIAL_ENERGY_JSON),
            "--figure-output",
            str(FIG_INITIAL),
            "--seed",
            str(args.seed),
        ],
    )

    timings["06"] = run_task(
        "task_06_run_vqe.py",
        [
            "--problem",
            str(PROBLEM_DILL),
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
            "--seed",
            str(args.seed),
        ],
    )

    timings["07"] = run_task(
        "task_07_evaluate.py",
        [
            "--reference",
            str(REFERENCE_JSON),
            "--initial-energy",
            str(INITIAL_ENERGY_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--output",
            str(METRICS_JSON),
        ],
    )

    timings["08"] = run_task(
        "task_08_visualize_summary.py",
        [
            "--molecule",
            str(MOLECULE_JSON),
            "--hamiltonian",
            str(HAMILTONIAN_JSON),
            "--reference",
            str(REFERENCE_JSON),
            "--ansatz",
            str(ANSATZ_JSON),
            "--initial-energy",
            str(INITIAL_ENERGY_JSON),
            "--vqe-result",
            str(VQE_RESULT_JSON),
            "--metrics",
            str(METRICS_JSON),
            "--trace",
            str(VQE_TRACE_CSV),
            "--output",
            str(FIG_SUMMARY),
        ],
    )

    print_banner(TaskKind.CLASSICAL, "VQE H2 pipeline 完成")
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
