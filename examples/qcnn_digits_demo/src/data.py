"""数据加载与预处理。

支持两种把 8x8 灰度数字图压成 8 维向量（用于 8 比特 QCNN）的方式：

* ``mode="spatial"`` (默认，**强烈推荐**)：
    8x8 → 2x4 average pooling → flatten → 缩到 [-π, π]。
    保留了图像的空间局部结构，与 QCNN 卷积层"邻居比特相关"的假设契合。

* ``mode="pca"``：
    StandardScaler → PCA(8) → MinMaxScaler(-π, π)。
    抽象但通用；缺点是 PCA 主成分之间相互独立、没有空间局部性，
    会让 QCNN 的卷积/池化层无从利用先验，训练困难（loss 下降微弱）。
    保留这个模式作为对照实验用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class DigitsData:
    """打包预处理结果与可视化所需的原始图像。"""

    x_train: np.ndarray  # shape (n_train, 8), values in [-pi, pi]
    x_test: np.ndarray   # shape (n_test, 8)
    y_train: np.ndarray  # shape (n_train,), values in {0, 1}
    y_test: np.ndarray
    raw_train_images: np.ndarray  # shape (n_train, 8, 8), 原始 8x8 灰度图
    raw_test_images: np.ndarray
    pipeline: object | None  # PCA 模式下是 sklearn Pipeline；spatial 模式下为 None
    mode: str


def _spatial_pool_8x8_to_8(images: np.ndarray) -> np.ndarray:
    """把 (n, 8, 8) 灰度图通过 4x2 average pooling 压成 (n, 8) 向量。

    8x8 -> 2x4 (块大小 4x2) -> flatten -> 8。
    保留了上半部 / 下半部 × 4 列 的空间区域信息。
    """
    if images.ndim != 3 or images.shape[1:] != (8, 8):
        raise ValueError(f"expected (n, 8, 8) images, got shape {images.shape}")
    pooled = images.reshape(-1, 2, 4, 4, 2).mean(axis=(2, 4))  # (n, 2, 4)
    return pooled.reshape(-1, 8)


def load_and_prepare_data(
    classes: tuple[int, int] = (0, 1),
    test_size: float = 0.2,
    n_components: int = 8,
    seed: int = 42,
    mode: Literal["spatial", "pca"] = "spatial",
) -> DigitsData:
    """加载 digits 数据并完成全部预处理。

    Args:
        classes: 二分类用的两个数字标签。
        test_size: 测试集占比。
        n_components: PCA 模式下的降维维度（spatial 模式下忽略）。
        seed: 随机种子。
        mode: ``"spatial"`` (推荐) 或 ``"pca"``。详见模块 docstring。
    """
    digits = load_digits()
    images_full = digits.images           # (1797, 8, 8)
    features_full = digits.data           # (1797, 64)
    labels_full = digits.target           # (1797,)

    mask = np.isin(labels_full, classes)
    images = images_full[mask]
    features = features_full[mask]
    labels = labels_full[mask].copy()

    # 把 classes -> {0, 1}
    label_map = {classes[0]: 0, classes[1]: 1}
    labels = np.array([label_map[v] for v in labels], dtype=np.int64)

    x_train_raw, x_test_raw, y_train, y_test, img_train, img_test = train_test_split(
        features,
        labels,
        images,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    if mode == "spatial":
        # 用 raw 8x8 图像做空间下采样
        x_train = _spatial_pool_8x8_to_8(img_train)
        x_test = _spatial_pool_8x8_to_8(img_test)
        # 原始像素 0..16 → 缩到 [-π, π]
        x_train = (x_train - 8.0) / 8.0 * np.pi
        x_test = (x_test - 8.0) / 8.0 * np.pi
        pipeline = None
    elif mode == "pca":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=seed)),
                ("rescale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
            ]
        )
        x_train = pipeline.fit_transform(x_train_raw)
        x_test = pipeline.transform(x_test_raw)
    else:
        raise ValueError(f"未知 mode: {mode!r}")

    return DigitsData(
        x_train=x_train.astype(np.float64),
        x_test=x_test.astype(np.float64),
        y_train=y_train,
        y_test=y_test,
        raw_train_images=img_train,
        raw_test_images=img_test,
        pipeline=pipeline,
        mode=mode,
    )
