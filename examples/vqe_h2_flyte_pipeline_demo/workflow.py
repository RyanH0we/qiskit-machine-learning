"""VQE H2 + Flyte 编排示例的 @workflow 入口。

把 8 个 ``@task`` 按数据依赖拼成 DAG。flyte 引擎会根据依赖自动并行无关分支：

* t03 / t04 都只依赖 t02，会并行。
* t05 同时需要 t02 (problem) 与 t04 (ansatz)，等这两个都好才能跑。
* t06 与 t05 互不依赖，会并行（但二者都依赖 t02/t04/t03）。
* t07 等 t05 / t06 都返回后聚合。
* t08 收尾，等前面所有产物。

业务逻辑 100% 复用 ``tasks/`` 下的 8 个独立模块；本文件只做编排，不写
任何 VQE / 量子化学相关代码。

入口有两种：

* ``python workflow.py``  —— flyte local execution，本机进程内顺序跑全部
  task，不需要 K8s。用于 sanity check / 调试。
* ``pyflyte run --remote workflow.py vqe_h2_workflow`` —— 提交到 flyte demo
  sandbox 或生产集群，每个 task 跑在独立 K8s pod 中，UI 可观测。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 `python workflow.py` 与 `pyflyte run workflow.py` 都能直接 import
# 同级目录下的 tasks/ 与 pipeline_lib.py。`pyflyte register` 也会沿用
# 这个 sys.path 解析逻辑，所以容器内一样工作。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import NamedTuple

from flytekit import workflow
from flytekit.types.file import FlyteFile

from tasks import (
    t01_define_molecule,
    t02_build_hamiltonian,
    t03_exact_reference,
    t04_build_ansatz,
    t05_initial_energy,
    t06_run_vqe,
    t07_evaluate,
    t08_visualize_summary,
)


class VQEWorkflowOutputs(NamedTuple):
    """``vqe_h2_workflow`` 的所有最终交付物。

    使用 ``NamedTuple`` 而不是裸 tuple，是为了让 Flyte Console UI 和
    ``pyflyte run`` 命令行都能按可读名字展示输出。
    """

    metrics_json: FlyteFile
    summary_png: FlyteFile

    molecule_png: FlyteFile
    hamiltonian_png: FlyteFile
    reference_png: FlyteFile
    ansatz_png: FlyteFile
    initial_energy_png: FlyteFile
    vqe_convergence_png: FlyteFile

    vqe_result_json: FlyteFile
    vqe_trace_csv: FlyteFile


@workflow
def vqe_h2_workflow(
    bond_length: float = 0.735,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    optimizer: str = "SLSQP",
    maxiter: int = 100,
    seed: int = 42,
) -> VQEWorkflowOutputs:
    """端到端 VQE-H2 流水线，由 flyte 调度。

    与 ``examples/vqe_h2_pipeline_demo/main.py`` 编排同一组 task，但这里由
    flyte 根据数据依赖自动并行。预期总能量 ~= -1.137306 Hartree。
    """

    molecule = t01_define_molecule(
        bond_length=bond_length,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    hamiltonian = t02_build_hamiltonian(molecule_json=molecule.molecule_json)

    reference = t03_exact_reference(problem_dill=hamiltonian.problem_dill)
    ansatz = t04_build_ansatz(problem_dill=hamiltonian.problem_dill)

    initial = t05_initial_energy(
        problem_dill=hamiltonian.problem_dill,
        ansatz_dill=ansatz.ansatz_dill,
        initial_point_npy=ansatz.initial_point_npy,
        reference_json=reference.reference_json,
        seed=seed,
    )

    vqe = t06_run_vqe(
        problem_dill=hamiltonian.problem_dill,
        ansatz_dill=ansatz.ansatz_dill,
        initial_point_npy=ansatz.initial_point_npy,
        reference_json=reference.reference_json,
        optimizer=optimizer,
        maxiter=maxiter,
        seed=seed,
    )

    metrics = t07_evaluate(
        reference_json=reference.reference_json,
        initial_energy_json=initial.initial_energy_json,
        vqe_result_json=vqe.vqe_result_json,
    )

    summary = t08_visualize_summary(
        molecule_json=molecule.molecule_json,
        hamiltonian_json=hamiltonian.hamiltonian_json,
        reference_json=reference.reference_json,
        ansatz_json=ansatz.ansatz_json,
        initial_energy_json=initial.initial_energy_json,
        vqe_result_json=vqe.vqe_result_json,
        metrics_json=metrics,
        vqe_trace_csv=vqe.vqe_trace_csv,
    )

    return VQEWorkflowOutputs(
        metrics_json=metrics,
        summary_png=summary,
        molecule_png=molecule.molecule_png,
        hamiltonian_png=hamiltonian.hamiltonian_png,
        reference_png=reference.reference_png,
        ansatz_png=ansatz.ansatz_png,
        initial_energy_png=initial.initial_energy_png,
        vqe_convergence_png=vqe.vqe_convergence_png,
        vqe_result_json=vqe.vqe_result_json,
        vqe_trace_csv=vqe.vqe_trace_csv,
    )


if __name__ == "__main__":
    print("Running VQE H2 workflow in Flyte local execution mode (no K8s)...", flush=True)
    out = vqe_h2_workflow()
    print("done.", flush=True)
    print(f"  metrics_json        = {out.metrics_json.path}", flush=True)
    print(f"  summary_png         = {out.summary_png.path}", flush=True)
    print(f"  vqe_result_json     = {out.vqe_result_json.path}", flush=True)
    print(f"  vqe_trace_csv       = {out.vqe_trace_csv.path}", flush=True)
    print(f"  molecule_png        = {out.molecule_png.path}", flush=True)
    print(f"  hamiltonian_png     = {out.hamiltonian_png.path}", flush=True)
    print(f"  reference_png       = {out.reference_png.path}", flush=True)
    print(f"  ansatz_png          = {out.ansatz_png.path}", flush=True)
    print(f"  initial_energy_png  = {out.initial_energy_png.path}", flush=True)
    print(f"  vqe_convergence_png = {out.vqe_convergence_png.path}", flush=True)
