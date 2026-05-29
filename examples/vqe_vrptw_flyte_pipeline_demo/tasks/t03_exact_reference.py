"""[CLASSICAL] Task 03 -- 暴力枚举得到小规模 VRPTW-QUBO 经典参考解。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import (
    CACHE_VERSION,
    TaskKind,
    bits_to_bitstring,
    decode_vrptw_solution,
    draw_vrptw_routes,
    int_to_bits,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_vrptw_image,
)


class ReferenceOut(NamedTuple):
    reference_json: FlyteFile
    reference_routes_png: FlyteFile


@task(
    container_image=vqe_vrptw_image,
    requests=Resources(cpu="2", mem="3Gi"),
    limits=Resources(cpu="4", mem="5Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=30),
)
def t03_exact_reference(
    instance_json: FlyteFile,
    qubo_json: FlyteFile,
    exact_max_qubits: int = 22,
) -> ReferenceOut:
    """[CLASSICAL] 枚举 bitstring，得到默认小规模实例的参考最优解。"""

    print_banner(TaskKind.CLASSICAL, "Task 03 / 经典暴力枚举参考解")
    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    qubo = json.loads(Path(qubo_json.download()).read_text(encoding="utf-8"))
    n = int(qubo["num_variables"])
    if n > exact_max_qubits:
        raise ValueError(f"变量数 {n} 超过 exact_max_qubits={exact_max_qubits}，请缩小实例或提高上限")

    total_states = 2**n
    print(f"  变量数 = {n}, 枚举 bitstring 数 = {total_states}", flush=True)
    best_energy = float("inf")
    best_energy_bits: list[int] | None = None
    best_feasible: dict | None = None
    best_feasible_bits: list[int] | None = None
    num_feasible = 0
    top_feasible: list[dict] = []

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

    work = task_workdir("t03")
    json_path = work / "reference.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    png_path = work / "03_reference_routes.png"
    draw_vrptw_routes(instance, best_feasible, png_path, "Classical reference solution")

    print(f"  可行 bitstring 数 = {num_feasible}", flush=True)
    print(f"  QUBO 最小能量 = {best_energy:.6f}", flush=True)
    print(f"  参考最优总距离 = {payload['best_feasible_total_distance']:.6f}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)
    return ReferenceOut(reference_json=FlyteFile(str(json_path)), reference_routes_png=FlyteFile(str(png_path)))
