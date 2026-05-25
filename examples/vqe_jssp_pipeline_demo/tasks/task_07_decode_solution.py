"""[CLASSICAL] Task 07 -- 把 VQE 采样 bitstring 解码成 JSSP 排程。

VQE 输出的是 bitstring 概率分布；这一步把它翻译回“哪道工序何时开始”的
甘特图语义，并检查约束是否满足。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    bitstring_to_bits,
    decode_schedule,
    draw_gantt,
    ensure_parent_dir,
    print_banner,
    qubo_energy,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--qubo", type=Path, required=True, help="task_02 输出的 qubo.json")
    p.add_argument("--vqe-result", type=Path, required=True, help="task_06 输出的 vqe_result.json")
    p.add_argument("--output", type=Path, required=True, help="输出 decoded_solution.json")
    p.add_argument("--samples-output", type=Path, required=True, help="输出 samples.csv")
    p.add_argument("--probability-figure-output", type=Path, required=True, help="输出候选 bitstring 概率图")
    p.add_argument("--gantt-figure-output", type=Path, required=True, help="输出 VQE 最佳排程甘特图")
    return p.parse_args()


def _decode_samples(instance: dict, qubo: dict, eigenstate: dict[str, float]) -> list[dict]:
    samples: list[dict] = []
    for bitstring, probability in eigenstate.items():
        samples.append(_decode_bitstring(instance, qubo, bitstring, float(probability)))
    samples.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return samples


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


def _write_samples_csv(samples: list[dict], out_path: Path) -> None:
    out = ensure_parent_dir(out_path)
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
    with out.open("w", newline="", encoding="utf-8") as f:
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
    import matplotlib.pyplot as plt

    top = samples[: min(12, len(samples))]
    labels = [row["bitstring"] for row in top]
    probs = [row["probability"] for row in top]
    colors = ["#31a354" if row["feasible"] else "#756bb1" for row in top]

    out = ensure_parent_dir(out_path)
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
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _candidate_union(samples: list[dict], feasible_samples: list[dict]) -> list[dict]:
    """保留高概率候选，也保留低能量候选，避免把教学上最重要的可行最优解截掉。"""

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


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 07 / 解码 VQE bitstring")
    instance = read_json(args.instance)
    qubo = read_json(args.qubo)
    vqe = read_json(args.vqe_result)
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
            key=lambda row: (
                row["makespan"],
                row["qubo_energy"],
                -row["probability"],
                row["bitstring"],
            ),
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
    if best_feasible is not None:
        print(f"  最佳可行 bitstring = {best_feasible['bitstring']}")
        print(f"  最佳可行 makespan = {best_feasible['makespan']}")
        print(f"  采样概率 = {best_feasible['probability']:.6f}")
    else:
        print("  [warn] VQE 样本中没有找到可行排程")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    _write_samples_csv(samples, args.samples_output)
    print(f"  -> 写入 {args.samples_output}")

    with Timer("绘制候选 bitstring 概率图"):
        _plot_probabilities(samples, args.probability_figure_output)
    print(f"  -> 写入 {args.probability_figure_output}")

    with Timer("绘制 VQE 最佳可行排程甘特图"):
        if best_feasible is None:
            decoded_for_plot = {"schedule": [], "selected_cmax": None, "actual_makespan": None}
        else:
            decoded_for_plot = best_feasible
        draw_gantt(instance, decoded_for_plot, args.gantt_figure_output, "Best feasible schedule sampled by VQE")
    print(f"  -> 写入 {args.gantt_figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
