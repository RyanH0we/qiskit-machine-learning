"""[CLASSICAL] Task 01 -- 定义 H2 分子几何与计算参数。

与 ``examples/vqe_h2_pipeline_demo/tasks/task_01_define_molecule.py`` 业务逻辑
一比一对应，区别只是把 ``argparse`` + ``write_json`` 改成 Flyte 的
``@task`` + ``FlyteFile`` 返回。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, print_banner, task_workdir, vqe_image


class MoleculeOut(NamedTuple):
    """``t01`` 多返回值。

    使用 ``NamedTuple`` 而不是裸 tuple，是为了让 Flyte Console UI 中
    每个输出文件有可读的名字，下游 task 也能按字段名引用。
    """

    molecule_json: FlyteFile
    molecule_png: FlyteFile


def _draw_molecule(bond_length: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="300m", mem="512Mi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t01_define_molecule(
    bond_length: float = 0.735,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
) -> MoleculeOut:
    """[CLASSICAL] 定义 H2 几何与基组等参数。

    输出 ``molecule.json`` 是后续所有任务的"问题书"，仅含纯文本元数据，
    没有任何量子电路。
    """

    print_banner(TaskKind.CLASSICAL, "Task 01 / 定义 H2 分子")
    print(f"  键长 = {bond_length:.6f} Angstrom", flush=True)
    print(f"  基组 = {basis}, 电荷 = {charge}, spin = {spin}", flush=True)

    atom = f"H 0 0 0; H 0 0 {bond_length:.12f}"
    payload = {
        "name": "H2",
        "description": "氢分子基态能量 VQE 示例输入",
        "atom": atom,
        "geometry": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": bond_length},
        ],
        "bond_length_angstrom": bond_length,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "unit": "Angstrom",
        "driver": "PySCFDriver",
    }

    work = task_workdir("t01")
    json_path = work / "molecule.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "01_molecule.png"
    _draw_molecule(bond_length, png_path)

    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return MoleculeOut(
        molecule_json=FlyteFile(str(json_path)),
        molecule_png=FlyteFile(str(png_path)),
    )
