"""[CLASSICAL] Task 01 -- 定义 H2 分子结构。

这一步只是在经典计算机上写下问题本身：两个氢原子相距多少、使用什么基组、
电荷和自旋是多少。它还不会运行量子电路。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bond-length", type=float, default=0.735, help="H-H 键长，单位 Angstrom")
    p.add_argument("--basis", default="sto3g", help="量子化学基组，默认 sto3g")
    p.add_argument("--charge", type=int, default=0, help="分子总电荷")
    p.add_argument("--spin", type=int, default=0, help="2S，自旋多重度相关量；H2 基态为 0")
    p.add_argument("--output", type=Path, required=True, help="输出 molecule.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出分子结构示意图 PNG")
    return p.parse_args()


def _draw_molecule(bond_length: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    xs = [0.0, bond_length]
    ys = [0.0, 0.0]
    ax.plot(xs, ys, color="#4d4d4d", linewidth=3, zorder=1)
    ax.scatter(xs, ys, s=1500, color=["#6baed6", "#6baed6"], edgecolor="black", linewidth=1.5, zorder=2)
    for x, label in zip(xs, ["H", "H"]):
        ax.text(x, 0.0, label, ha="center", va="center", fontsize=20, weight="bold")
    ax.annotate(
        f"{bond_length:.3f} Angstrom",
        xy=(bond_length / 2, 0.0),
        xytext=(bond_length / 2, 0.32),
        ha="center",
        arrowprops={"arrowstyle": "<->", "color": "#333333"},
        fontsize=11,
    )
    pad = max(0.4, bond_length * 0.35)
    ax.set_xlim(-pad, bond_length + pad)
    ax.set_ylim(-0.55, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("H2 molecule geometry", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义 H2 分子")
    print(f"  键长 = {args.bond_length:.6f} Angstrom")
    print(f"  基组 = {args.basis}, 电荷 = {args.charge}, spin = {args.spin}")

    atom = f"H 0 0 0; H 0 0 {args.bond_length:.12f}"
    payload = {
        "name": "H2",
        "description": "氢分子基态能量 VQE 示例输入",
        "atom": atom,
        "geometry": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": args.bond_length},
        ],
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "charge": args.charge,
        "spin": args.spin,
        "unit": "Angstrom",
        "driver": "PySCFDriver",
    }
    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制分子结构示意图"):
        _draw_molecule(args.bond_length, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
