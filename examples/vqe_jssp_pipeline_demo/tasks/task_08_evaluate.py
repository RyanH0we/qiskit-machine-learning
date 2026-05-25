"""[CLASSICAL] Task 08 -- 汇总 VQE-JSSP 验收指标。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, print_banner, read_json, write_json

ENERGY_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="task_03 输出的 reference.json")
    p.add_argument("--initial-energy", type=Path, required=True, help="task_05 输出的 initial_energy.json")
    p.add_argument("--vqe-result", type=Path, required=True, help="task_06 输出的 vqe_result.json")
    p.add_argument("--decoded-solution", type=Path, required=True, help="task_07 输出的 decoded_solution.json")
    p.add_argument("--output", type=Path, required=True, help="输出 metrics.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 08 / 汇总指标")
    reference = read_json(args.reference)
    initial = read_json(args.initial_energy)
    vqe = read_json(args.vqe_result)
    decoded = read_json(args.decoded_solution)

    optimal_makespan = reference["optimal_makespan"]
    optimal_energy = reference["optimal_qubo_energy"]
    best_feasible = decoded["best_feasible_sample"]
    most_likely = decoded["most_likely_sample"]
    best_measurement = decoded.get("best_measurement_decoded")

    best_feasible_found = best_feasible is not None
    best_makespan = best_feasible["makespan"] if best_feasible_found else None
    best_energy = best_feasible["qubo_energy"] if best_feasible_found else None
    best_feasible_makespan_is_optimal = bool(best_feasible_found and best_makespan == optimal_makespan)
    best_feasible_qubo_energy_is_optimal = bool(
        best_feasible_found and abs(float(best_energy) - float(optimal_energy)) <= ENERGY_TOLERANCE
    )
    best_measurement_qubo_energy_is_optimal = bool(
        best_measurement is not None
        and abs(float(best_measurement["qubo_energy"]) - float(optimal_energy)) <= ENERGY_TOLERANCE
    )
    most_likely_sample_is_optimal = bool(
        most_likely
        and most_likely["feasible"]
        and most_likely["makespan"] == optimal_makespan
        and abs(float(most_likely["qubo_energy"]) - float(optimal_energy)) <= ENERGY_TOLERANCE
    )
    top_contains_optimal = any(
        row["feasible"] and row["makespan"] == optimal_makespan
        for row in decoded["top_candidates"]
    )

    payload = {
        "reference": {
            "optimal_makespan": optimal_makespan,
            "optimal_qubo_energy": optimal_energy,
            "optimal_bitstring": reference["optimal_bitstring"],
        },
        "initial": {
            "mean_qubo_energy": initial["initial_mean_qubo_energy"],
            "best_sample_qubo_energy": initial["initial_best_sample"]["qubo_energy"],
            "best_sample_feasible": initial["initial_best_sample"]["feasible"],
        },
        "vqe": {
            "mean_qubo_energy": vqe["vqe_mean_qubo_energy"],
            "energy_gap_to_classical_optimum": vqe["vqe_mean_qubo_energy"] - optimal_energy,
            "best_measurement": vqe["best_measurement"],
            "num_evaluations_recorded": vqe["num_evaluations_recorded"],
        },
        "decoded": {
            "num_sampled_bitstrings": decoded["num_sampled_bitstrings"],
            "num_feasible_sampled_bitstrings": decoded["num_feasible_sampled_bitstrings"],
            "most_likely_feasible": most_likely["feasible"] if most_likely else False,
            "best_feasible_found": best_feasible_found,
            "best_feasible_bitstring": best_feasible["bitstring"] if best_feasible_found else None,
            "best_feasible_makespan": best_makespan,
            "best_feasible_qubo_energy": best_energy,
            "best_feasible_makespan_is_optimal": best_feasible_makespan_is_optimal,
            "best_feasible_qubo_energy_is_optimal": best_feasible_qubo_energy_is_optimal,
            "best_measurement_bitstring": best_measurement["bitstring"] if best_measurement else None,
            "best_measurement_feasible": best_measurement["feasible"] if best_measurement else False,
            "best_measurement_makespan": best_measurement["makespan"] if best_measurement else None,
            "best_measurement_qubo_energy": best_measurement["qubo_energy"] if best_measurement else None,
            "best_measurement_qubo_energy_is_optimal": best_measurement_qubo_energy_is_optimal,
            "most_likely_sample_is_optimal": most_likely_sample_is_optimal,
            "top_candidates_contain_optimal_makespan": top_contains_optimal,
        },
        "acceptance": {
            "constraint_satisfied": best_feasible_found,
            "default_instance_optimal_makespan_is_4": optimal_makespan == 4,
            "vqe_top_candidates_include_makespan_4": top_contains_optimal and optimal_makespan == 4,
            "vqe_sampled_optimal_makespan": best_feasible_makespan_is_optimal,
            "vqe_sampled_optimal_qubo_energy": best_feasible_qubo_energy_is_optimal,
            "vqe_best_measurement_reaches_optimal_qubo_energy": best_measurement_qubo_energy_is_optimal,
            "most_likely_sample_is_optimal": most_likely_sample_is_optimal,
        },
    }

    print(f"  经典最优 makespan = {optimal_makespan}")
    print(f"  VQE 最佳可行 makespan = {best_makespan}")
    print(f"  VQE top candidates 包含最优 makespan: {top_contains_optimal}")
    print(f"  最佳可行排程 makespan 是否最优: {best_feasible_makespan_is_optimal}")
    print(f"  最佳可行 bitstring QUBO energy 是否最优: {best_feasible_qubo_energy_is_optimal}")
    print(f"  best_measurement QUBO energy 是否最优: {best_measurement_qubo_energy_is_optimal}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
