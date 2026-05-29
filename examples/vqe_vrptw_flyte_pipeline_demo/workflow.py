"""VQE-VRPTW + Flyte 编排示例的 @workflow 入口。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flytekit import workflow
from flytekit.types.file import FlyteFile

from pipeline_lib import archive_named_outputs
from tasks import (
    t01_define_or_load_instance,
    t02_build_time_indexed_qubo,
    t03_exact_reference,
    t04_build_ansatz,
    t05_initial_energy,
    t06_run_vqe,
    t07_decode_solution,
    t08_evaluate,
    t09_visualize_summary,
)


class VRPTWWorkflowOutputs(NamedTuple):
    metrics_json: FlyteFile
    summary_png: FlyteFile

    instance_json: FlyteFile
    qubo_json: FlyteFile
    hamiltonian_json: FlyteFile
    hamiltonian_dill: FlyteFile
    reference_json: FlyteFile
    ansatz_json: FlyteFile
    ansatz_dill: FlyteFile
    initial_point_npy: FlyteFile
    initial_energy_json: FlyteFile
    vqe_result_json: FlyteFile
    vqe_result_dill: FlyteFile
    decoded_solution_json: FlyteFile
    vqe_trace_csv: FlyteFile
    samples_csv: FlyteFile
    solution_routes_csv: FlyteFile

    instance_png: FlyteFile
    qubo_hamiltonian_png: FlyteFile
    reference_routes_png: FlyteFile
    ansatz_png: FlyteFile
    initial_energy_png: FlyteFile
    vqe_convergence_png: FlyteFile
    solution_probabilities_png: FlyteFile
    vqe_routes_png: FlyteFile


@workflow
def vqe_vrptw_workflow(
    instance_input: FlyteFile | None = None,
    time_granularity: float = 1.0,
    max_stops_per_vehicle: int = 2,
    penalty: float = 0.0,
    ansatz_reps: int = 2,
    optimizer: str = "COBYLA",
    maxiter: int = 120,
    shots: int = 4096,
    seed: int = 42,
    exact_max_qubits: int = 22,
) -> VRPTWWorkflowOutputs:
    """端到端 VQE-VRPTW 流水线，由 Flyte 调度。"""

    instance = t01_define_or_load_instance(
        instance_input=instance_input,
        time_granularity=time_granularity,
        max_stops_per_vehicle=max_stops_per_vehicle,
    )
    qubo = t02_build_time_indexed_qubo(instance_json=instance.instance_json, penalty=penalty)
    reference = t03_exact_reference(
        instance_json=instance.instance_json,
        qubo_json=qubo.qubo_json,
        exact_max_qubits=exact_max_qubits,
    )
    ansatz = t04_build_ansatz(
        hamiltonian_json=qubo.hamiltonian_json,
        qubo_json=qubo.qubo_json,
        ansatz_reps=ansatz_reps,
        seed=seed,
    )
    initial = t05_initial_energy(
        instance_json=instance.instance_json,
        hamiltonian_dill=qubo.hamiltonian_dill,
        ansatz_dill=ansatz.ansatz_dill,
        initial_point_npy=ansatz.initial_point_npy,
        shots=shots,
        seed=seed,
    )
    vqe = t06_run_vqe(
        instance_json=instance.instance_json,
        hamiltonian_dill=qubo.hamiltonian_dill,
        ansatz_dill=ansatz.ansatz_dill,
        initial_point_npy=ansatz.initial_point_npy,
        reference_json=reference.reference_json,
        optimizer=optimizer,
        maxiter=maxiter,
        shots=shots,
        seed=seed,
    )
    decoded = t07_decode_solution(
        instance_json=instance.instance_json,
        qubo_json=qubo.qubo_json,
        vqe_result_json=vqe.vqe_result_json,
    )
    metrics = t08_evaluate(
        reference_json=reference.reference_json,
        initial_energy_json=initial.initial_energy_json,
        vqe_result_json=vqe.vqe_result_json,
        decoded_solution_json=decoded.decoded_solution_json,
    )
    summary = t09_visualize_summary(
        instance_json=instance.instance_json,
        qubo_json=qubo.qubo_json,
        hamiltonian_json=qubo.hamiltonian_json,
        ansatz_json=ansatz.ansatz_json,
        reference_json=reference.reference_json,
        initial_energy_json=initial.initial_energy_json,
        vqe_result_json=vqe.vqe_result_json,
        decoded_solution_json=decoded.decoded_solution_json,
        metrics_json=metrics,
        vqe_trace_csv=vqe.vqe_trace_csv,
        samples_csv=decoded.samples_csv,
    )

    return VRPTWWorkflowOutputs(
        metrics_json=metrics,
        summary_png=summary,
        instance_json=instance.instance_json,
        qubo_json=qubo.qubo_json,
        hamiltonian_json=qubo.hamiltonian_json,
        hamiltonian_dill=qubo.hamiltonian_dill,
        reference_json=reference.reference_json,
        ansatz_json=ansatz.ansatz_json,
        ansatz_dill=ansatz.ansatz_dill,
        initial_point_npy=ansatz.initial_point_npy,
        initial_energy_json=initial.initial_energy_json,
        vqe_result_json=vqe.vqe_result_json,
        vqe_result_dill=vqe.vqe_result_dill,
        decoded_solution_json=decoded.decoded_solution_json,
        vqe_trace_csv=vqe.vqe_trace_csv,
        samples_csv=decoded.samples_csv,
        solution_routes_csv=decoded.solution_routes_csv,
        instance_png=instance.instance_png,
        qubo_hamiltonian_png=qubo.qubo_hamiltonian_png,
        reference_routes_png=reference.reference_routes_png,
        ansatz_png=ansatz.ansatz_png,
        initial_energy_png=initial.initial_energy_png,
        vqe_convergence_png=vqe.vqe_convergence_png,
        solution_probabilities_png=decoded.solution_probabilities_png,
        vqe_routes_png=decoded.vqe_routes_png,
    )


if __name__ == "__main__":
    print("Running VQE-VRPTW workflow in Flyte local execution mode (no K8s)...", flush=True)
    out = vqe_vrptw_workflow()
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / "local_run"
    archived = archive_named_outputs(out, artifact_dir)
    print("done.", flush=True)
    print(f"  archived artifacts = {artifact_dir}", flush=True)
    for path in sorted(archived):
        print(f"    - {path.name}", flush=True)
