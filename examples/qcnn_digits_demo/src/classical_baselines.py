"""经典基线：在与 QCNN 完全相同的 8 维 PCA 输入上训练 SVM 与 MLP。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


@dataclass
class BaselineResult:
    name: str
    train_acc: float
    test_acc: float
    train_time: float


def _evaluate(model, x_train, y_train, x_test, y_test) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0
    train_acc = float(model.score(x_train, y_train))
    test_acc = float(model.score(x_test, y_test))
    return train_acc, test_acc, train_time


def train_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> BaselineResult:
    model = SVC(kernel="rbf", random_state=seed)
    train_acc, test_acc, train_time = _evaluate(model, x_train, y_train, x_test, y_test)
    return BaselineResult("SVM (RBF)", train_acc, test_acc, train_time)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> BaselineResult:
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=400,
        random_state=seed,
    )
    train_acc, test_acc, train_time = _evaluate(model, x_train, y_train, x_test, y_test)
    return BaselineResult("MLP (16-8)", train_acc, test_acc, train_time)
