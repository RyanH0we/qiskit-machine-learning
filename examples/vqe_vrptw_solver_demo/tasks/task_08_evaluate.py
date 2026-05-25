"""[CLASSICAL] Task 08 -- 汇总 VQE-VRPTW solver 验收指标。"""

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


def _has_violation(decoded: dict | None, keywords: tuple[str, ...]) -> bool:
    if decoded is None:
        return True
    return any(any(keyword in violation for keyword in keywords) for violation in decoded.get("violations", []))


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 08 / 汇总指标")
    reference = read_json(args.reference)
    initial = read_json(args.initial_energy)
    vqe = read_json(args.vqe_result)
    decoded = read_json(args.decoded_solution)

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
            "route_continuity_feasible": bool(
                best_feasible and not _has_violation(best_feasible, ("前一位置为空", "到客户"))
            ),
            "returns_to_depot": bool(best_feasible and not _has_violation(best_feasible, ("回到 depot",))),
            "vqe_sampled_reference_distance": best_distance_is_reference,
            "vqe_sampled_reference_qubo_energy": best_qubo_energy_is_reference,
            "top_candidates_contain_feasible_solution": top_contains_feasible,
        },
    }

    print(f"  参考最优总距离 = {reference_distance:.6f}")
    print(f"  VQE 最佳可行总距离 = {best_distance}")
    print(f"  VQE 样本中找到可行解: {best_feasible_found}")
    print(f"  每个客户恰好一次: {payload['acceptance']['all_customers_served_once']}")
    print(f"  容量约束满足: {payload['acceptance']['capacity_feasible']}")
    print(f"  时间窗约束满足: {payload['acceptance']['time_window_feasible']}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
