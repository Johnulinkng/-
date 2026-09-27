# WiDAN：可审计的跨传感器、跨工况故障诊断

[English README](README.md)

本项目研究工业旋转机械振动信号在测点、工况、通道数量和噪声发生变化时的故障诊断。实验采用严格无监督域适应协议：训练阶段可以使用源域故障标签，但目标训练域故障标签保持隐藏。

> **当前状态：**完整性审计通过，CRA-v2A 注册性能门失败。神经候选用于研究评估和负迁移分析，不作为推荐部署模型。

## 覆盖范围

- 3→3 同工况跨测点迁移
- 3→3 跨工况和双重分布偏移
- 3→1 逐目标通道独立训练、独立评估和独立保存权重
- clean 与 AWGN 20/10/0 dB 评估
- 线性 Ridge、RBF Kernel Ridge 强基线
- CRA-v2A 通道鲁棒神经候选及其训练、推理、评估和独立审计工具

## 主要结果与边界

源验证选参的线性 Ridge 在 clean 开发矩阵上的 Accuracy / Macro-F1 均值为：

| 数据集 | 主开发矩阵 | 跨工况/双跨 3→1 |
| --- | ---: | ---: |
| 水泵 | 68.400 / 67.003% | 57.622 / 56.918% |
| 实验室齿轮箱 | 76.500 / 75.818% | 73.500 / 72.345% |
| SEU 齿轮箱 | 63.917 / 62.465% | 32.861 / 31.592% |

CRA-v2A 与共同锚点控制的16个配对中，只有7个同时提高 Accuracy 与 Macro-F1；平均变化分别为 -0.34 和 -0.37 个百分点，双跨 Macro-F1 下降1.98个百分点，并新增3个零召回事件。因此没有注册推荐神经模型。

校正后的44条件固定模型估计为：线性 Ridge 59.675 / 58.338%，RBF-KRR 55.893 / 53.666%。由于早期无效评估曾暴露 final 元数据，这些结果只能称为 **post-exposure fixed-model estimate**，不能称为全新盲测结果。

## 目录结构

```text
.
|-- cra_v2/                 # 冻结的 CRA-v2A 代码、权重、报告和验包工具
|-- baselines/              # Ridge/RBF 推理入口与使用说明
|-- docs/                   # 验收矩阵、证据和论文表述边界
|-- requirements-cpu.txt
|-- requirements-gpu.txt
|-- README.md
`-- README_CN.md
```

体积较大的 Ridge/RBF 完整研究快照通过 GitHub Release 提供，不写入普通 Git 历史。

## 安装与核验

CPU 环境：

```bash
python -m venv .venv
pip install -r requirements-cpu.txt
```

核验 CRA-v2A 冻结包：

```bash
python cra_v2/tools/verify_channel_robust_v2_delivery.py cra_v2
```

该命令检查文件清单、SHA-256、模型注册表、实验收据及配对结构，不需要原始数据集。

## 推理

根据 `cra_v2/model_registry.json` 选择任务、目标通道、模型角色、采样率和 SHA-256 完全匹配的 checkpoint：

```bash
python cra_v2/inference/predict_channel_robust_v2.py \
  --checkpoint cra_v2/models/candidates/<arm-id>/best_model.pth \
  --checkpoint-sha256 <registry中的sha256> \
  --task-id PG-S1 \
  --variant candidate \
  --target-slots 1 \
  --sampling-rate-hz 24000 \
  --input /path/to/raw_windows.npy \
  --output /path/to/predictions.npz
```

输入必须是有限 `float32 [N,C,2048]` NPY。程序不会自动判断传感器、调整通道顺序或重采样。

## 数据与结论边界

- 仓库不包含原始信号、标签、窗口数据、私有评估根或目标开发/最终标签。
- 严格 UDA 下，目标训练标签必须保持隐藏。
- 3→1 的三个目标通道分别训练和评估，不是一份权重通用于任意通道。
- 审计通过只证明工件和流程完整，不代表模型性能有效。
- 当前结果适合系统评估、强基线复现和负迁移研究，不能宣传为统一高性能模型。

## 权利说明

当前项目没有获批的项目级开源许可证。仓库公开可见不代表授权复制、修改、再分发或再许可。详见 [RIGHTS_AND_LICENSE.md](RIGHTS_AND_LICENSE.md) 和 [第三方与分发边界](cra_v2/docs/V2_THIRD_PARTY_AND_RIGHTS.md)。

