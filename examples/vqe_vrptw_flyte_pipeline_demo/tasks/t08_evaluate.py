"""[CLASSICAL] Task 08 -- 汇总 VQE-VRPTW solver 验收指标。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, print_banner, task_workdir, vqe_vrptw_image


ENERGY_TOLERANCE = 1e-9


def _has_violation(decoded: dict | None, keywords: tuple[str, ...]) -> bool:
    if decoded is None:
        return True
    return any(any(keyword in violation for keyword in keywords) for violation in decoded.get("violations", []))


@task(
    container_image=vqe_vrptw_image,
    requests=Resources(cpu="300m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=5),
)
def t08_evaluate(
    reference_json: FlyteFile,
    initial_energy_json: FlyteFile,
    vqe_result_json: FlyteFile,
    decoded_solution_json: FlyteFile,
) -> FlyteFile:
    """[CLASSICAL] 汇总可行性、能量、距离和验收指标。"""

    print_banner(TaskKind.CLASSICAL, "Task 08 / 汇总指标")
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    initial = json.loads(Path(initial_energy_json.download()).read_text(encoding="utf-8"))
    vqe = json.loads(Path(vqe_result_json.download()).read_text(encoding="utf-8"))
    decoded = json.loads(Path(decoded_solution_json.download()).read_text(encoding="utf-8"))

    best_feasible = decoded["best_feasible_sample"]
    most_likely = decoded["most_likely_sample"]
    best_measurement = decoded.get("best_measurement_decoded")
    best_feasible_found = best_feasible is not None
    reference_distance = float(reference["best_feasible_total_distance"])
    reference_energy = float(reference["best_feasible_qubo_energy"])

    best_distance = None if best_feasible is None else float(best_feasible["total_distance"])
    best_energy = None if best_feasible is None else float(best_feasible["qubo_energy"])
    best_distance_is_reference = bool(best_feasible_found and abs(best_distance - reference_distance) <= 1e-9)
    best_qubo_energy_is_reference = bool(best_feasible_found and abs(best_energy - reference_energy) <= ENERGY_TOLERANCE)
    best_measurement_feasible = bool(best_measurement is not None and best_measurement["feasible"])
    most_likely_feasible = bool(most_likely is not None and most_likely["feasible"])
    top_contains_feasible = any(row["feasible"] for row in decoded["top_candidates"])

    payload = {
        "reference": {
            "best_feasible_total_distance": reference_distance,
            "best_feasible_qubo_energy": reference_energy,
            "best_feasible_bitstring": reference["best_feasible_bitstring"],
            "num_feasible_states": reference["num_feasible_states"],
        },
        "initial": {
            "mean_qubo_energy": initial["initial_mean_qubo_energy"],
            "best_sample_by_energy": initial["initial_best_sample_by_energy"],
            "best_feasible_sample": initial["initial_best_feasible_sample"],
        },
        "vqe": {
            "mean_qubo_energy": vqe["vqe_mean_qubo_energy"],
            "energy_gap_to_reference_feasible": vqe["vqe_mean_qubo_energy"] - reference_energy,
            "best_measurement": vqe["best_measurement"],
            "num_evaluations_recorded": vqe["num_evaluations_recorded"],
        },
        "decoded": {
            "num_sampled_bitstrings": decoded["num_sampled_bitstrings"],
            "num_feasible_sampled_bitstrings": decoded["num_feasible_sampled_bitstrings"],
            "most_likely_feasible": most_likely_feasible,
            "best_feasible_found": best_feasible_found,
            "best_feasible_bitstring": None if best_feasible is None else best_feasible["bitstring"],
            "best_feasible_total_distance": best_distance,
            "best_feasible_qubo_energy": best_energy,
            "best_feasible_distance_is_reference": best_distance_is_reference,
            "best_feasible_qubo_energy_is_reference": best_qubo_energy_is_reference,
            "best_measurement_feasible": best_measurement_feasible,
            "best_measurement_total_distance": None if best_measurement is None else best_measurement["total_distance"],
            "best_measurement_qubo_energy": None if best_measurement is None else best_measurement["qubo_energy"],
            "top_candidates_contain_feasible_solution": top_contains_feasible,
        },
        "acceptance": {
            "best_feasible_found": best_feasible_found,
            "all_customers_served_once": bool(best_feasible and best_feasible["all_customers_served_once"]),
            "capacity_feasible": bool(best_feasible and best_feasible["capacity_feasible"]),
            "time_window_feasible": bool(best_feasible and not _has_violation(best_feasible, ("时间窗",))),
            "route_continuity_feasible": bool(best_feasible and not _has_violation(best_feasible, ("前一位置为空", "到客户"))),
            "returns_to_depot": bool(best_feasible and not _has_violation(best_feasible, ("回到 depot",))),
            "vqe_sampled_reference_distance": best_distance_is_reference,
            "vqe_sampled_reference_qubo_energy": best_qubo_energy_is_reference,
            "top_candidates_contain_feasible_solution": top_contains_feasible,
        },
    }

    work = task_workdir("t08")
    json_path = work / "metrics.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"  参考最优总距离 = {reference_distance:.6f}", flush=True)
    print(f"  VQE 最佳可行总距离 = {best_distance}", flush=True)
    print(f"  VQE 样本中找到可行解: {best_feasible_found}", flush=True)
    print(f"  每个客户恰好一次: {payload['acceptance']['all_customers_served_once']}", flush=True)
    print(f"  容量约束满足: {payload['acceptance']['capacity_feasible']}", flush=True)
    print(f"  时间窗约束满足: {payload['acceptance']['time_window_feasible']}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    return FlyteFile(str(json_path))
