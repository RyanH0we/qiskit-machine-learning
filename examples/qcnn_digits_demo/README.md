# QCNN 手写数字二分类 demo

一个**端到端可运行**的量子机器学习演示项目：在 sklearn `load_digits` 真实手写数字数据集上做 0/1 二分类，使用 8 比特量子卷积神经网络（QCNN），并与经典 SVM/MLP 做横向对比。

## 1. 目录结构

```
examples/qcnn_digits_demo/
├── README.md
├── qcnn_demo.ipynb        # Notebook 形式（推荐边看边跑）
├── qcnn_demo.py           # 等价脚本，一键运行
├── src/
│   ├── data.py            # 数据加载与预处理流水线（spatial / pca 两种模式）
│   ├── circuits.py        # 量子电路构件：conv_circuit / pool_circuit ...
│   ├── model.py           # build_qcnn()：组装 8 比特 QCNN
│   ├── classical_baselines.py  # SVM / MLP 经典基线
│   └── viz.py             # 各类可视化函数
└── outputs/               # 运行后自动生成的图与模型
```

## 2. 安装

进入仓库根目录（不是 demo 目录）：

```bash
cd /home/ryan/guojiqe-qcdemo/qiskit-machine-learning

# 推荐先建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装本仓库（editable，便于改源码即时生效）
pip install -e .

# 安装可视化与 notebook 依赖
pip install matplotlib seaborn jupyter pylatexenc
```

## 3. 运行

### 方式一：脚本（一键）

```bash
cd examples/qcnn_digits_demo
python qcnn_demo.py
```

可选参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--optimizer` | `SPSA` | 优化器: `SPSA`（推荐）/ `COBYLA` / `L_BFGS_B` |
| `--maxiter` | 150 | 优化器最大迭代数（SPSA 推荐 100-200） |
| `--data-mode` | `spatial` | `spatial`（8x8 → 2x4 池化, 推荐）/ `pca`（PCA 64 → 8, 仅作对照） |
| `--classes A B` | `0 1` | 用于二分类的两个数字 |
| `--seed` | 42 | 随机种子 |
| `--out-dir PATH` | `./outputs` | 输出目录 |
| `--skip-baselines` | off | 跳过经典基线 |

例如：跑得更快（仅 80 次迭代）：

```bash
python qcnn_demo.py --maxiter 80
```

跑对照实验（PCA 模式，看 QCNN 收敛失败）：

```bash
python qcnn_demo.py --data-mode pca --maxiter 150 --skip-baselines
```

### 方式二：Notebook（推荐学习）

```bash
cd examples/qcnn_digits_demo
jupyter notebook qcnn_demo.ipynb
```

按顺序执行 8 个 section，每一段都有中文说明。

## 4. 输出文件

成功运行后，`outputs/` 下会生成：

| 文件 | 说明 |
|---|---|
| `samples.png` | 8 张原始 8x8 数字图 |
| `circuit.png` | QCNN 完整电路图（feature_map + ansatz 解构后） |
| `training_curve.png` | 训练损失下降曲线 |
| `confusion_matrix.png` | 测试集混淆矩阵 |
| `misclassified.png` | 错分样本（无错分则不生成） |
| `comparison.png` | QCNN vs SVM vs MLP 准确率/训练时间对比 |
| `qcnn_model.dill` | 训练好的 classifier，可重新加载推理 |

## 5. 原理速览

### 5.1 数据流（默认 `spatial` 模式）

```
load_digits (8x8 灰度)
   |  筛选 label in {0, 1}
   v
train_test_split (80/20, stratified)
   |
   v
8x8 -> 2x4 平均池化 (4x2 块)  ← 保留空间局部结构！
   |  flatten 到 8 维
   v
线性映射 (x - 8) / 8 * π → [-π, π]
   |
   v
8 比特 z_feature_map（角度编码）
   |
   v
QCNN ansatz: conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> pool3
                 8       8->4    4       4->2    2       2->1
   |
   v
末位比特上 Z 期望 ∈ [-1, 1]
   |
   v
NeuralNetworkClassifier + SPSA + squared_error
```

### 5.2 几个关键设计选择（"为什么这么做"）

**为什么 spatial 池化（默认），而不是 PCA？**
QCNN 的卷积层假设"**邻居比特之间存在相关性**"。8x8 图像 → 2x4 池化保留了"上半部 / 下半部 + 4 列"这种空间结构，相邻量子比特对应相邻区域，QCNN 能利用这种先验。
PCA 主成分相互独立、**没有空间局部性**，QCNN 卷积层无从利用先验，loss 几乎不下降。
你可以用 `--data-mode pca` 复现这种"训练失败"的对照实验。

**为什么 8 比特？**
QCNN 输入维度 = 量子比特数。8 比特 statevector 模拟刚好 CPU 可承受（$2^8=256$ 维态向量），且能做 8→4→2→1 的三层池化。
更高比特数会让训练时间指数增长。

**为什么标签从 {0, 1} 映射到 {-1, +1}？**
QNN 的输出是 Pauli-Z 期望值 $\langle Z \rangle \in [-1, 1]$，与 $\pm 1$ 标签直接做平方误差最自然。

**为什么测末位比特？**
QCNN 经过 3 次池化（8→4→2→1）后，"信息汇聚"到最后一个 sink 比特上。这是 Cong 等人原始 QCNN 论文的标准设计。

**为什么默认 SPSA 而不是 COBYLA？**
- COBYLA 是 trust-region 单纯形法，对 63 维参数空间需要至少 65 次电路求值（一次单纯形顶点采样），且对随机初始化收敛极慢
- L-BFGS-B 基于 ParamShift 梯度，每步需要 $2 \times 63 = 126$ 次电路求值
- SPSA 每步只需 2 次电路求值（与维度无关），高维 + noisy 友好，是 NISQ 时代的常用选择

## 6. 期望结果

CPU 单线程运行（参考机型：i5/i7 + 16G 内存）：

| 指标 | 期望值 |
|---|---|
| QCNN 测试准确率 | ~85–95% |
| QCNN 训练时间 | ~5–8 分钟（150 次 SPSA 迭代） |
| SVM 测试准确率 | ~100% |
| SVM 训练时间 | < 0.1 秒 |
| MLP 测试准确率 | ~90–100% |
| MLP 训练时间 | < 0.5 秒 |

**对 0/1 这种线性可分性极强的问题，经典模型几乎没难度。** 这个 demo 的价值在于：

1. 让初学者把 QCNN 完整流程跑通一遍
2. 直观看到经典 vs 量子在简单问题上的"差距"——这正是 NISQ 时代的现实
3. 给出可被替换、可被扩展的**模块化骨架**
4. 演示一个常被忽视的关键点：**QCNN 需要数据具有空间局部性**（PCA 模式的对照实验明确说明了这点）

## 7. 常见报错

| 报错 | 解决 |
|---|---|
| `ModuleNotFoundError: qiskit_machine_learning` | 没装包，回到仓库根目录跑 `pip install -e .` |
| `ModuleNotFoundError: matplotlib / seaborn` | `pip install matplotlib seaborn` |
| `MissingOptionalLibraryError: 'pylatexenc' library is required` | `pip install pylatexenc`（画电路图所需） |
| `ModuleNotFoundError: src` | 确认你在 `examples/qcnn_digits_demo/` 目录下运行；脚本里已自动 `sys.path.insert` |
| Notebook kernel 找不到 | `python -m ipykernel install --user --name=qml-demo` 然后在 Jupyter 里切换 |
| 训练特别慢 | 减小 `--maxiter`（80 也够看到 loss 下降趋势）；或换更小问题 `--classes 0 7` |
| QCNN acc 一直在 50% 附近 | 检查是否用了 `--data-mode pca`，建议改回默认 `spatial` |

## 8. 进阶玩法

- **三分类**：`--classes 0 1 2` 不行，需要改 classifier 用 `one_hot=True` 并把输出改成多个 observable
- **更高精度**：`--maxiter 300` 通常能再提升 5-10 个百分点（代价：训练时间翻倍）
- **换 ansatz**：把 `src/model.py` 里的 ansatz 替换成 `qiskit.circuit.library.real_amplitudes(8, reps=3)`，比较收敛性
- **换优化器**：`--optimizer COBYLA` 或 `--optimizer L_BFGS_B`，看曲线
- **更难任务**：`--classes 4 9` 或 `--classes 3 8`，看 QCNN 在不易分的数字对上表现
- **加噪声**：用 `qiskit_aer.AerSimulator` 构造 `EstimatorV2`，传给 `EstimatorQNN(estimator=...)`
- **GPU 加速**：`pip install 'qiskit-aer-gpu'` + 设置 `QISKIT_GPU=true`
- **混合模型**：把 `bundle.qnn` 用 `TorchConnector` 包成 `nn.Module`，前面接经典 CNN 提取特征
- **真实硬件**：使用 `qiskit-ibm-runtime` 的 `EstimatorV2` 替换默认 estimator，跑在 IBM 量子计算机上

## 9. 引用

- Cong, I., Choi, S., & Lukin, M. D. (2019). *Quantum convolutional neural networks*. Nature Physics, 15(12), 1273–1278.
- Qiskit Machine Learning 官方教程 11：[`docs/tutorials/11_quantum_convolutional_neural_networks.ipynb`](../../docs/tutorials/11_quantum_convolutional_neural_networks.ipynb)
- Spall, J. C. (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation*. IEEE Transactions on Automatic Control, 37(3), 332–341.
