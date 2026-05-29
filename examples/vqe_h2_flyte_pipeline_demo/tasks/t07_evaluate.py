"""[CLASSICAL] Task 07 -- 汇总 VQE 误差与验收指标到 metrics.json。"""

from __future__ import annotations

import json
from pathlib import Path

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, hartree_to_ev, print_banner, task_workdir, vqe_image


CHEMICAL_ACCURACY_HARTREE = 0.0016


@task(
    container_image=vqe_image,
    requests=Resources(cpu="200m", mem="256Mi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t07_evaluate(
    reference_json: FlyteFile,
    initial_energy_json: FlyteFile,
    vqe_result_json: FlyteFile,
) -> FlyteFile:
    """[CLASSICAL] 汇总误差、chemical accuracy 判定。"""

    print_banner(TaskKind.CLASSICAL, "Task 07 / 汇总指标")
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    initial = json.loads(Path(initial_energy_json.download()).read_text(encoding="utf-8"))
    vqe = json.loads(Path(vqe_result_json.download()).read_text(encoding="utf-8"))

    exact = reference["exact_total_energy_hartree"]
    initial_energy = initial["initial_total_energy_hartree"]
    vqe_energy = vqe["vqe_total_energy_hartree"]
    initial_error = abs(initial_energy - exact)
    vqe_error = abs(vqe_energy - exact)

    payload = {
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "chemical_accuracy_ev": hartree_to_ev(CHEMICAL_ACCURACY_HARTREE),
        "energies_hartree": {
            "hartree_fock": reference["hartree_fock_total_energy_hartree"],
            "initial_ansatz": initial_energy,
            "vqe": vqe_energy,
            "exact": exact,
        },
        "errors_hartree": {
            "initial_ansatz_abs_error": initial_error,
            "vqe_abs_error": vqe_error,
        },
        "errors_ev": {
            "initial_ansatz_abs_error": hartree_to_ev(initial_error),
            "vqe_abs_error": hartree_to_ev(vqe_error),
        },
        "vqe_within_chemical_accuracy": vqe_error <= CHEMICAL_ACCURACY_HARTREE,
        "vqe_passes_1e_minus_3_hartree": vqe_error <= 1e-3,
        "improvement_from_initial_hartree": initial_error - vqe_error,
    }

    print(f"  exact = {exact:.12f} Hartree", flush=True)
    print(f"  VQE   = {vqe_energy:.12f} Hartree", flush=True)
    print(f"  abs error = {vqe_error:.6e} Hartree", flush=True)
    print(f"  chemical accuracy = {CHEMICAL_ACCURACY_HARTREE:.4f} Hartree", flush=True)
    print(f"  是否达到 chemical accuracy: {payload['vqe_within_chemical_accuracy']}", flush=True)

    out_path = task_workdir("t07") / "metrics.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  -> 写入 {out_path}", flush=True)
    return FlyteFile(str(out_path))
