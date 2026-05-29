"""VQE H2 + Flyte 示例的 task 模块。

每个文件对应一个 ``@task``，与 ``examples/vqe_h2_pipeline_demo/tasks/`` 中的
原始 CLI 任务一一对应；区别只在编排器：这里返回 ``FlyteFile`` 并由 Flyte
按数据依赖自动调度，原版用 ``argparse`` + ``subprocess`` 顺序串。
"""

from tasks.t01_define_molecule import t01_define_molecule
from tasks.t02_build_hamiltonian import t02_build_hamiltonian
from tasks.t03_exact_reference import t03_exact_reference
from tasks.t04_build_ansatz import t04_build_ansatz
from tasks.t05_initial_energy import t05_initial_energy
from tasks.t06_run_vqe import t06_run_vqe
from tasks.t07_evaluate import t07_evaluate
from tasks.t08_visualize_summary import t08_visualize_summary

__all__ = [
    "t01_define_molecule",
    "t02_build_hamiltonian",
    "t03_exact_reference",
    "t04_build_ansatz",
    "t05_initial_energy",
    "t06_run_vqe",
    "t07_evaluate",
    "t08_visualize_summary",
]
