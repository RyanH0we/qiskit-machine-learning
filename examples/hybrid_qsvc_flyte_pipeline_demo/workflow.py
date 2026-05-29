"""Hybrid QSVC + Flyte 编排示例入口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flytekit import workflow
from flytekit.types.file import FlyteFile

from tasks import (
    t01_generate_data,
    t02_visualize_data,
    t03_quantum_kernel,
    t04_visualize_kernel,
    t05_train_qsvc,
    t06_train_classical_svc,
    t07_evaluate,
    t08_visualize_results,
)


class HybridQSVCWorkflowOutputs(NamedTuple):
    """``hybrid_qsvc_workflow`` 的最终交付物。"""

    metrics_json: FlyteFile
    data_npz: FlyteFile
    kernel_train_npz: FlyteFile
    kernel_test_npz: FlyteFile
    feature_map_dill: FlyteFile
    qsvc_dill: FlyteFile
    classical_svc_dill: FlyteFile
    data_png: FlyteFile
    kernels_png: FlyteFile
    results_png: FlyteFile
    decision_png: FlyteFile
    circuit_png: FlyteFile


@workflow
def hybrid_qsvc_workflow(
    n: int = 2,
    train: int = 20,
    test: int = 10,
    gap: float = 0.3,
    reps: int = 2,
    entanglement: str = "linear",
    c: float = 1.0,
    rbf_gamma: str = "scale",
    seed: int = 42,
    grid: int = 30,
) -> HybridQSVCWorkflowOutputs:
    """端到端 Hybrid QSVC Flyte workflow。

    数据依赖让 Flyte 自动并行：
    ``t02``/``t03``/``t06`` 都只依赖 ``t01``，``t04``/``t05`` 依赖量子核，
    ``t07`` 等两个模型，``t08`` 收尾产出最终图。
    """

    data = t01_generate_data(n=n, train=train, test=test, gap=gap, seed=seed)
    data_fig = t02_visualize_data(data_npz=data.data_npz)

    kernel = t03_quantum_kernel(
        data_npz=data.data_npz,
        reps=reps,
        entanglement=entanglement,
        seed=seed,
    )
    kernel_fig = t04_visualize_kernel(kernel_train_npz=kernel.kernel_train_npz)

    qsvc = t05_train_qsvc(
        kernel_train_npz=kernel.kernel_train_npz,
        kernel_test_npz=kernel.kernel_test_npz,
        c=c,
        seed=seed,
    )
    classical = t06_train_classical_svc(
        data_npz=data.data_npz,
        c=c,
        rbf_gamma=rbf_gamma,
        seed=seed,
    )
    metrics = t07_evaluate(
        qsvc_dill=qsvc.qsvc_dill,
        classical_svc_dill=classical.classical_svc_dill,
    )
    final_figs = t08_visualize_results(
        data_npz=data.data_npz,
        metrics_json=metrics.metrics_json,
        qsvc_dill=qsvc.qsvc_dill,
        classical_svc_dill=classical.classical_svc_dill,
        feature_map_dill=kernel.feature_map_dill,
        grid=grid,
    )

    return HybridQSVCWorkflowOutputs(
        metrics_json=metrics.metrics_json,
        data_npz=data.data_npz,
        kernel_train_npz=kernel.kernel_train_npz,
        kernel_test_npz=kernel.kernel_test_npz,
        feature_map_dill=kernel.feature_map_dill,
        qsvc_dill=qsvc.qsvc_dill,
        classical_svc_dill=classical.classical_svc_dill,
        data_png=data_fig.data_png,
        kernels_png=kernel_fig.kernels_png,
        results_png=final_figs.results_png,
        decision_png=final_figs.decision_png,
        circuit_png=kernel.circuit_png,
    )


if __name__ == "__main__":
    print("Running Hybrid QSVC workflow in Flyte local execution mode (no K8s)...", flush=True)
    out = hybrid_qsvc_workflow()
    print("done.", flush=True)
    print(f"  metrics_json         = {out.metrics_json.path}", flush=True)
    print(f"  data_npz             = {out.data_npz.path}", flush=True)
    print(f"  kernel_train_npz     = {out.kernel_train_npz.path}", flush=True)
    print(f"  kernel_test_npz      = {out.kernel_test_npz.path}", flush=True)
    print(f"  feature_map_dill     = {out.feature_map_dill.path}", flush=True)
    print(f"  qsvc_dill            = {out.qsvc_dill.path}", flush=True)
    print(f"  classical_svc_dill   = {out.classical_svc_dill.path}", flush=True)
    print(f"  data_png             = {out.data_png.path}", flush=True)
    print(f"  kernels_png          = {out.kernels_png.path}", flush=True)
    print(f"  results_png          = {out.results_png.path}", flush=True)
    print(f"  decision_png         = {out.decision_png.path}", flush=True)
    print(f"  circuit_png          = {out.circuit_png.path}", flush=True)

    metrics_path = Path(out.metrics_json.path)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        qsvc_acc = metrics["models"]["qsvc"]["test_acc"]
        advantage = metrics["quantum_advantage_test_acc"]
        print(f"  qsvc.test_acc        = {qsvc_acc:.4f}", flush=True)
        print(f"  quantum_advantage    = {advantage:+.4f}", flush=True)
