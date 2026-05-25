"""[CLASSICAL] Task 03 -- 暴力枚举得到 JSSP 的经典参考最优解。

默认实例只有 14 个二进制变量，所以可以直接枚举所有 bitstring。这个参考
答案用于验证 VQE 是否找到了可行且最优的排程。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    bits_to_bitstring,
    decode_schedule,
    draw_gantt,
    int_to_bits,
    print_banner,
    qubo_energy,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--qubo", type=Path, required=True, help="task_02 输出的 qubo.json")
    p.add_argument("--output", type=Path, required=True, help="输出 reference.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出经典最优甘特图 PNG")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 03 / 经典暴力枚举参考答案")
    instance = read_json(args.instance)
    qubo = read_json(args.qubo)
    n = int(qubo["num_variables"])
    total_states = 2**n
    print(f"  变量数 = {n}, 枚举 bitstring 数 = {total_states}")

    best_energy = float("inf")
    best_energy_bits: list[int] | None = None
    best_feasible: dict | None = None
    best_feasible_bits: list[int] | None = None
    num_feasible = 0
    top_feasible: list[dict] = []

    with Timer("枚举所有 QUBO bitstring"):
        for state in range(total_states):
            bits = int_to_bits(state, n)
            energy = qubo_energy(qubo, bits)
            if energy < best_energy:
                best_energy = energy
                best_energy_bits = bits

            decoded = decode_schedule(instance, qubo["variables"], bits)
            if not decoded["feasible"]:
                continue
            num_feasible += 1
            record = {
                "bitstring": bits_to_bitstring(bits),
                "qubo_energy": energy,
                "makespan": decoded["actual_makespan"],
                "selected_cmax": decoded["selected_cmax"],
                "schedule": decoded["schedule"],
            }
            top_feasible.append(record)
            if best_feasible is None or (
                decoded["actual_makespan"],
                energy,
                record["bitstring"],
            ) < (
                best_feasible["actual_makespan"],
                best_feasible["qubo_energy"],
                best_feasible["bitstring"],
            ):
                best_feasible = {
                    "bitstring": record["bitstring"],
                    "qubo_energy": energy,
                    **decoded,
                }
                best_feasible_bits = bits

    if best_energy_bits is None or best_feasible is None or best_feasible_bits is None:
        raise RuntimeError("没有找到任何可行排程，请检查 QUBO 建模")

    top_feasible.sort(key=lambda row: (row["makespan"], row["qubo_energy"], row["bitstring"]))
    payload = {
        "method": "brute_force_enumeration",
        "num_variables": n,
        "num_enumerated_states": total_states,
        "num_feasible_states": num_feasible,
        "exact_min_qubo_energy": best_energy,
        "exact_min_energy_bitstring": bits_to_bitstring(best_energy_bits),
        "optimal_makespan": best_feasible["actual_makespan"],
        "optimal_bitstring": best_feasible["bitstring"],
        "optimal_qubo_energy": best_feasible["qubo_energy"],
        "optimal_selected_cmax": best_feasible["selected_cmax"],
        "optimal_schedule": best_feasible["schedule"],
        "known_optimal_makespan_from_instance": instance.get("known_optimal_makespan"),
        "top_feasible": top_feasible[:10],
    }
    print(f"  可行 bitstring 数 = {num_feasible}")
    print(f"  QUBO 最小能量 = {best_energy:.6f}")
    print(f"  经典最优 makespan = {payload['optimal_makespan']}")
    print(f"  最优 bitstring = {payload['optimal_bitstring']}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制经典最优甘特图"):
        draw_gantt(instance, best_feasible, args.figure_output, "Classical optimum by brute force")
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
