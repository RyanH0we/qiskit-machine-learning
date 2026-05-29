"""Hybrid QSVC Flyte task 集合。"""

from .t01_generate_data import t01_generate_data
from .t02_visualize_data import t02_visualize_data
from .t03_quantum_kernel import t03_quantum_kernel
from .t04_visualize_kernel import t04_visualize_kernel
from .t05_train_qsvc import t05_train_qsvc
from .t06_train_classical_svc import t06_train_classical_svc
from .t07_evaluate import t07_evaluate
from .t08_visualize_results import t08_visualize_results

__all__ = [
    "t01_generate_data",
    "t02_visualize_data",
    "t03_quantum_kernel",
    "t04_visualize_kernel",
    "t05_train_qsvc",
    "t06_train_classical_svc",
    "t07_evaluate",
    "t08_visualize_results",
]
