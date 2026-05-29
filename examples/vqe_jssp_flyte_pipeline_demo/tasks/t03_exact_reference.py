"""[CLASSICAL] Task 03 -- 暴力枚举得到 JSSP 经典参考最优解。"""

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
    decode_schedule,
    draw_gantt,
    int_to_bits,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_jssp_image,
)


class ReferenceOut(NamedTuple):
    reference_json: FlyteFile
    reference_gantt_png: FlyteFile


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=10),
)
def t03_exact_reference(instance_json: FlyteFile, qubo_json: FlyteFile) -> ReferenceOut:
    """[CLASSICAL] 枚举全部 bitstring，得到验证 VQE 的参考答案。"""

    print_banner(TaskKind.CLASSICAL, "Task 03 / 经典暴力枚举参考答案")
    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    qubo = json.loads(Path(qubo_json.download()).read_text(encoding="utf-8"))
    n = int(qubo["num_variables"])
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
            best_feasible = {"bitstring": record["bitstring"], "qubo_energy": energy, **decoded}
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

    work = task_workdir("t03")
    json_path = work / "reference.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "03_reference_gantt.png"
    draw_gantt(instance, best_feasible, png_path, "Classical optimum by brute force")

    print(f"  可行 bitstring 数 = {num_feasible}", flush=True)
    print(f"  QUBO 最小能量 = {best_energy:.6f}", flush=True)
    print(f"  经典最优 makespan = {payload['optimal_makespan']}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)
    return ReferenceOut(reference_json=FlyteFile(str(json_path)), reference_gantt_png=FlyteFile(str(png_path)))
