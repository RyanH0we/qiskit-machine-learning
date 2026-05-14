"""[CLASSICAL] Task 07 -- 汇总 VQE 误差和验收指标。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, hartree_to_ev, print_banner, read_json, write_json


CHEMICAL_ACCURACY_HARTREE = 0.0016


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="task_03 输出的 reference.json")
    p.add_argument("--initial-energy", type=Path, required=True, help="task_05 输出的 initial_energy.json")
    p.add_argument("--vqe-result", type=Path, required=True, help="task_06 输出的 vqe_result.json")
    p.add_argument("--output", type=Path, required=True, help="输出 metrics.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 07 / 汇总指标")
    reference = read_json(args.reference)
    initial = read_json(args.initial_energy)
    vqe = read_json(args.vqe_result)

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

    print(f"  exact = {exact:.12f} Hartree")
    print(f"  VQE   = {vqe_energy:.12f} Hartree")
    print(f"  abs error = {vqe_error:.6e} Hartree")
    print(f"  chemical accuracy = {CHEMICAL_ACCURACY_HARTREE:.4f} Hartree")
    print(f"  是否达到 chemical accuracy: {payload['vqe_within_chemical_accuracy']}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
