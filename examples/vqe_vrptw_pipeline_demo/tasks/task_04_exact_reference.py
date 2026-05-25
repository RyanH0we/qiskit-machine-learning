"""[CLASSICAL] Task 04 -- 穷举 bitstring 得到精确参考答案。

这个 toy problem 只有 6 个变量，所以可以枚举 2^6=64 个 bitstring。参考答案
用于检查 VQE 是否找到了同一条最低代价路线。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, bits_label, bits_to_qiskit_bitstring, ensure_parent_dir, print_banner, qubo_energy, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubo", type=Path, required=True, help="task_03 输出的 qubo.json")
    p.add_argument("--routes", type=Path, required=True, help="task_02 输出的 routes.json")
    p.add_argument("--output", type=Path, required=True, help="输出 reference.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出能量地形图 PNG")
    return p.parse_args()


def _enumerate_states(qubo: dict) -> list[dict]:
    n = int(qubo["num_variables"])
    rows: list[dict] = []
    for mask in range(2**n):
        bits = [(mask >> i) & 1 for i in range(n)]
        selected = [i for i, bit in enumerate(bits) if bit == 1]
        energy = qubo_energy(bits, qubo["constant"], qubo["linear"], qubo["quadratic"])
        rows.append(
            {
                "bits": bits,
                "bitstring": bits_label(bits),
                "qiskit_bitstring": bits_to_qiskit_bitstring(bits),
                "energy": energy,
                "num_selected_routes": len(selected),
                "is_one_hot": len(selected) == 1,
                "selected_route_indices": selected,
            }
        )
    rows.sort(key=lambda row: row["energy"])
    return rows


def _attach_route_info(row: dict, routes: list[dict]) -> dict:
    out = dict(row)
    if row["is_one_hot"]:
        route = routes[row["selected_route_indices"][0]]
        out["selected_route_id"] = route["route_id"]
        out["selected_route_label"] = route["label"]
        out["selected_route_cost"] = route["cost"]
        out["selected_route_is_time_window_feasible"] = route["is_time_window_feasible"]
    else:
        out["selected_route_id"] = None
        out["selected_route_label"] = None
    return out


def _plot_energy_landscape(states: list[dict], best: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    xs = np.arange(len(states))
    energies = [state["energy"] for state in states]
    colors = ["#31a354" if state["is_one_hot"] else "#bdbdbd" for state in states]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(xs, energies, c=colors, s=28, edgecolor="black", linewidth=0.25)
    ax.scatter([0], [best["energy"]], s=120, color="#de2d26", marker="*", label="Best bitstring", zorder=4)
    ax.set_xlabel("bitstrings sorted by QUBO energy")
    ax.set_ylabel("QUBO / Hamiltonian energy")
    ax.set_title("Task 04: exact enumeration over 64 bitstrings")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    text = (
        f"best bits = {best['bitstring']}\n"
        f"route = {best['selected_route_id']} / {best['selected_route_label']}\n"
        f"energy = {best['energy']:.3f}"
    )
    ax.text(
        0.98,
        0.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 04 / 经典穷举精确参考")
    qubo = read_json(args.qubo)
    routes = read_json(args.routes)["routes"]

    with Timer("枚举 2^6 个 bitstring"):
        states = _enumerate_states(qubo)
        states = [_attach_route_info(state, routes) for state in states]

    best = states[0]
    print(f"  精确最优 bitstring = {best['bitstring']}")
    print(f"  精确最优路线 = {best['selected_route_id']} / {best['selected_route_label']}")
    print(f"  精确最优能量 = {best['energy']:.6f}")

    payload = {
        "method": "brute_force_enumeration",
        "num_states": len(states),
        "best": best,
        "top_states": states[:12],
        "all_states": states,
    }
    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制能量地形图"):
        _plot_energy_landscape(states, best, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
