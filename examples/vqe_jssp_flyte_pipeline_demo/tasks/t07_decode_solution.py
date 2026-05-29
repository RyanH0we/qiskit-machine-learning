"""[CLASSICAL] Task 07 -- 把 VQE 采样 bitstring 解码成 JSSP 排程。"""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import (
    CACHE_VERSION,
    TaskKind,
    bitstring_to_bits,
    decode_schedule,
    draw_gantt,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_jssp_image,
)


class DecodeOut(NamedTuple):
    decoded_solution_json: FlyteFile
    samples_csv: FlyteFile
    solution_probabilities_png: FlyteFile
    vqe_gantt_png: FlyteFile


def _decode_bitstring(instance: dict, qubo: dict, bitstring: str, probability: float | None = None) -> dict:
    bits = bitstring_to_bits(bitstring, int(qubo["num_variables"]))
    energy = qubo_energy(qubo, bits)
    decoded = decode_schedule(instance, qubo["variables"], bits)
    return {
        "bitstring": bitstring,
        "probability": probability,
        "qubo_energy": float(energy),
        "feasible": bool(decoded["feasible"]),
        "makespan": decoded["actual_makespan"],
        "selected_cmax": decoded["selected_cmax"],
        "violations": decoded["violations"],
        "schedule": decoded["schedule"],
    }


def _decode_samples(instance: dict, qubo: dict, eigenstate: dict[str, float]) -> list[dict]:
    samples = [_decode_bitstring(instance, qubo, bitstring, float(probability)) for bitstring, probability in eigenstate.items()]
    samples.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return samples


def _write_samples_csv(samples: list[dict], out_path: Path) -> None:
    fields = [
        "rank_by_probability",
        "bitstring",
        "probability",
        "qubo_energy",
        "feasible",
        "makespan",
        "selected_cmax",
        "num_violations",
        "violations",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(samples, start=1):
            writer.writerow(
                {
                    "rank_by_probability": rank,
                    "bitstring": row["bitstring"],
                    "probability": row["probability"],
                    "qubo_energy": row["qubo_energy"],
                    "feasible": row["feasible"],
                    "makespan": row["makespan"],
                    "selected_cmax": row["selected_cmax"],
                    "num_violations": len(row["violations"]),
                    "violations": " | ".join(row["violations"][:4]),
                }
            )


def _plot_probabilities(samples: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = samples[: min(12, len(samples))]
    labels = [row["bitstring"] for row in top]
    probs = [row["probability"] for row in top]
    colors = ["#31a354" if row["feasible"] else "#756bb1" for row in top]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(range(len(top)), probs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, top):
        tag = f"C={row['makespan']}" if row["feasible"] else "invalid"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            tag,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Sample probability")
    ax.set_title("VQE candidate bitstrings (green = feasible schedule)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _candidate_union(samples: list[dict], feasible_samples: list[dict]) -> list[dict]:
    by_probability = samples[:15]
    by_energy = sorted(samples, key=lambda row: (row["qubo_energy"], -row["probability"], row["bitstring"]))[:15]
    by_feasible = sorted(
        feasible_samples,
        key=lambda row: (row["makespan"], row["qubo_energy"], -row["probability"], row["bitstring"]),
    )[:5]
    merged: dict[str, dict] = {}
    for row in [*by_probability, *by_energy, *by_feasible]:
        merged[row["bitstring"]] = row
    return sorted(merged.values(), key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=10),
)
def t07_decode_solution(
    instance_json: FlyteFile,
    qubo_json: FlyteFile,
    vqe_result_json: FlyteFile,
) -> DecodeOut:
    """[CLASSICAL] 解码 VQE 输出的 bitstring 概率分布。"""

    print_banner(TaskKind.CLASSICAL, "Task 07 / 解码 VQE bitstring")
    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    qubo = json.loads(Path(qubo_json.download()).read_text(encoding="utf-8"))
    vqe = json.loads(Path(vqe_result_json.download()).read_text(encoding="utf-8"))
    samples = _decode_samples(instance, qubo, vqe["eigenstate"])

    best_measurement = vqe.get("best_measurement")
    best_measurement_decoded = None
    if best_measurement is not None and best_measurement.get("bitstring") is not None:
        best_measurement_decoded = _decode_bitstring(
            instance,
            qubo,
            str(best_measurement["bitstring"]),
            float(best_measurement.get("probability", 0.0)),
        )
        best_measurement_decoded["measurement_value"] = best_measurement.get("value")
        best_measurement_decoded["state"] = best_measurement.get("state")

    feasible_samples = [row for row in samples if row["feasible"]]
    best_feasible = None
    if feasible_samples:
        best_feasible = min(
            feasible_samples,
            key=lambda row: (row["makespan"], row["qubo_energy"], -row["probability"], row["bitstring"]),
        )
    most_likely = samples[0] if samples else None

    payload = {
        "num_sampled_bitstrings": len(samples),
        "num_feasible_sampled_bitstrings": len(feasible_samples),
        "most_likely_sample": most_likely,
        "best_feasible_sample": best_feasible,
        "best_measurement_decoded": best_measurement_decoded,
        "top_candidates": _candidate_union(samples, feasible_samples),
    }

    work = task_workdir("t07")
    json_path = work / "decoded_solution.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = work / "samples.csv"
    _write_samples_csv(samples, csv_path)
    probability_png = work / "07_solution_probabilities.png"
    _plot_probabilities(samples, probability_png)
    gantt_png = work / "08_vqe_gantt.png"
    if best_feasible is None:
        decoded_for_plot = {"schedule": [], "selected_cmax": None, "actual_makespan": None}
    else:
        decoded_for_plot = best_feasible
    draw_gantt(instance, decoded_for_plot, gantt_png, "Best feasible schedule sampled by VQE")

    if best_feasible is not None:
        print(f"  最佳可行 bitstring = {best_feasible['bitstring']}", flush=True)
        print(f"  最佳可行 makespan = {best_feasible['makespan']}", flush=True)
    else:
        print("  [warn] VQE 样本中没有找到可行排程", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {csv_path}", flush=True)
    print(f"  -> 写入 {probability_png}", flush=True)
    print(f"  -> 写入 {gantt_png}", flush=True)
    return DecodeOut(
        decoded_solution_json=FlyteFile(str(json_path)),
        samples_csv=FlyteFile(str(csv_path)),
        solution_probabilities_png=FlyteFile(str(probability_png)),
        vqe_gantt_png=FlyteFile(str(gantt_png)),
    )
