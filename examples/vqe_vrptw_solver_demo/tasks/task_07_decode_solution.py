"""[CLASSICAL] Task 07 -- 把 VQE bitstring 概率分布解码成 VRPTW 车辆路径。"""

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
    decode_vrptw_solution,
    draw_vrptw_routes,
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
    p.add_argument("--routes-output", type=Path, required=True, help="输出 solution_routes.csv")
    p.add_argument("--probability-figure-output", type=Path, required=True, help="输出候选 bitstring 概率图")
    p.add_argument("--route-figure-output", type=Path, required=True, help="输出最佳路线图")
    return p.parse_args()


def _decode_bitstring(instance: dict, qubo: dict, bitstring: str, probability: float | None = None) -> dict:
    bits = bitstring_to_bits(bitstring, int(qubo["num_variables"]))
    decoded = decode_vrptw_solution(instance, qubo["variables"], bits)
    return {
        "bitstring": bitstring,
        "probability": probability,
        "qubo_energy": float(qubo_energy(qubo, bits)),
        **decoded,
    }


def _decode_samples(instance: dict, qubo: dict, eigenstate: dict[str, float]) -> list[dict]:
    samples = [
        _decode_bitstring(instance, qubo, str(bitstring), float(probability))
        for bitstring, probability in eigenstate.items()
    ]
    samples.sort(key=lambda row: (-float(row["probability"]), row["qubo_energy"], row["bitstring"]))
    return samples


def _candidate_union(samples: list[dict], feasible_samples: list[dict]) -> list[dict]:
    by_probability = samples[:20]
    by_energy = sorted(samples, key=lambda row: (row["qubo_energy"], -row["probability"], row["bitstring"]))[:20]
    by_feasible = sorted(
        feasible_samples,
        key=lambda row: (row["total_distance"], row["qubo_energy"], -row["probability"], row["bitstring"]),
    )[:10]
    merged: dict[str, dict] = {}
    for row in [*by_probability, *by_energy, *by_feasible]:
        merged[row["bitstring"]] = row
    return sorted(merged.values(), key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))


def _write_samples_csv(samples: list[dict], out_path: Path) -> None:
    out = ensure_parent_dir(out_path)
    fields = [
        "rank_by_probability",
        "bitstring",
        "probability",
        "qubo_energy",
        "feasible",
        "total_distance",
        "num_customers_served",
        "num_vehicles_used",
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
                    "total_distance": row["total_distance"],
                    "num_customers_served": row["num_customers_served"],
                    "num_vehicles_used": row["num_vehicles_used"],
                    "num_violations": len(row["violations"]),
                    "violations": " | ".join(row["violations"][:4]),
                }
            )


def _write_routes_csv(decoded: dict | None, out_path: Path) -> None:
    out = ensure_parent_dir(out_path)
    fields = [
        "vehicle_id",
        "position",
        "customer_id",
        "leg_from",
        "arrival_time",
        "service_start_time",
        "departure_time",
        "leg_distance",
        "leg_travel_time",
        "demand",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if decoded is None:
            return
        for route in decoded["routes"]:
            for event in route["events"]:
                writer.writerow({field: event.get(field) for field in fields})


def _plot_probabilities(samples: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    top = samples[: min(12, len(samples))]
    labels = [row["bitstring"] for row in top]
    probs = [row["probability"] for row in top]
    colors = ["#31a354" if row["feasible"] else "#756bb1" for row in top]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    bars = ax.bar(range(len(top)), probs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, top):
        tag = f"d={row['total_distance']:.1f}" if row["feasible"] else "invalid"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, tag, ha="center", fontsize=8)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("sample probability")
    ax.set_title("VQE candidate bitstrings (green = feasible VRPTW route)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 07 / 解码 VQE bitstring")
    instance = read_json(args.instance)
    qubo = read_json(args.qubo)
    vqe = read_json(args.vqe_result)

    samples = _decode_samples(instance, qubo, vqe["eigenstate"])
    feasible_samples = [row for row in samples if row["feasible"]]
    most_likely = samples[0] if samples else None
    best_feasible = None
    if feasible_samples:
        best_feasible = min(
            feasible_samples,
            key=lambda row: (row["total_distance"], row["qubo_energy"], -row["probability"], row["bitstring"]),
        )

    best_measurement = vqe.get("best_measurement")
    best_measurement_decoded = None
    if best_measurement is not None and best_measurement.get("bitstring"):
        best_measurement_decoded = _decode_bitstring(
            instance,
            qubo,
            str(best_measurement["bitstring"]),
            float(best_measurement.get("probability", 0.0)),
        )
        best_measurement_decoded["measurement_value"] = best_measurement.get("value")
        best_measurement_decoded["state"] = best_measurement.get("state")

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
        print(f"  最佳可行总距离 = {best_feasible['total_distance']:.6f}")
        print(f"  采样概率 = {best_feasible['probability']:.6f}")
    else:
        print("  [warn] VQE 样本中没有找到可行 VRPTW 解")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    _write_samples_csv(samples, args.samples_output)
    print(f"  -> 写入 {args.samples_output}")

    _write_routes_csv(best_feasible, args.routes_output)
    print(f"  -> 写入 {args.routes_output}")

    with Timer("绘制候选 bitstring 概率图"):
        _plot_probabilities(samples, args.probability_figure_output)
    print(f"  -> 写入 {args.probability_figure_output}")

    with Timer("绘制 VQE 最佳可行路线"):
        decoded_for_plot = best_feasible or {"feasible": False, "routes": [], "total_distance": 0.0}
        draw_vrptw_routes(instance, decoded_for_plot, args.route_figure_output, "Best feasible VRPTW solution sampled by VQE")
    print(f"  -> 写入 {args.route_figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
