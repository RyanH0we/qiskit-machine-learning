"""[CLASSICAL] Task 03 -- 暴力枚举得到小规模 VRPTW-QUBO 的经典参考解。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    bits_to_bitstring,
    decode_vrptw_solution,
    draw_vrptw_routes,
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
    p.add_argument("--figure-output", type=Path, required=True, help="输出经典参考路线 PNG")
    p.add_argument("--exact-max-qubits", type=int, default=22, help="超过该变量数则拒绝暴力枚举")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 03 / 经典暴力枚举参考解")
    instance = read_json(args.instance)
    qubo = read_json(args.qubo)
    n = int(qubo["num_variables"])
    if n > args.exact_max_qubits:
        raise ValueError(f"变量数 {n} 超过 --exact-max-qubits={args.exact_max_qubits}，请缩小实例或提高上限")

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

            decoded = decode_vrptw_solution(instance, qubo["variables"], bits)
            if not decoded["feasible"]:
                continue
            num_feasible += 1
            record = {
                "bitstring": bits_to_bitstring(bits),
                "qubo_energy": float(energy),
                "total_distance": float(decoded["total_distance"]),
                "num_vehicles_used": int(decoded["num_vehicles_used"]),
                "routes": decoded["routes"],
            }
            top_feasible.append(record)
            key = (record["total_distance"], record["qubo_energy"], record["bitstring"])
            if best_feasible is None or key < (
                best_feasible["total_distance"],
                best_feasible["qubo_energy"],
                best_feasible["bitstring"],
            ):
                best_feasible = {"bitstring": record["bitstring"], "qubo_energy": energy, **decoded}
                best_feasible_bits = bits

    if best_energy_bits is None or best_feasible is None or best_feasible_bits is None:
        raise RuntimeError("没有找到任何可行 VRPTW 解，请检查实例或 QUBO 建模")

    top_feasible.sort(key=lambda row: (row["total_distance"], row["qubo_energy"], row["bitstring"]))
    payload = {
        "method": "brute_force_enumeration",
        "num_variables": n,
        "num_enumerated_states": total_states,
        "num_feasible_states": num_feasible,
        "exact_min_qubo_energy": float(best_energy),
        "exact_min_energy_bitstring": bits_to_bitstring(best_energy_bits),
        "best_feasible_bitstring": best_feasible["bitstring"],
        "best_feasible_qubo_energy": float(best_feasible["qubo_energy"]),
        "best_feasible_total_distance": float(best_feasible["total_distance"]),
        "best_feasible_num_vehicles_used": int(best_feasible["num_vehicles_used"]),
        "best_feasible_solution": best_feasible,
        "top_feasible": top_feasible[:20],
    }
    print(f"  可行 bitstring 数 = {num_feasible}")
    print(f"  QUBO 最小能量 = {best_energy:.6f}")
    print(f"  参考最优总距离 = {payload['best_feasible_total_distance']:.6f}")
    print(f"  参考最优 bitstring = {payload['best_feasible_bitstring']}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制经典参考车辆路径"):
        draw_vrptw_routes(instance, best_feasible, args.figure_output, "Classical reference solution")
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
