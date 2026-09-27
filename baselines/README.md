# WIDAN 可运行强基线研究快照 v3（2026-09-25）

本包交付当前证据最完整的线性 Ridge 与 RBF Kernel Ridge 基线、推理入口、模型工件、预测结果及独立审计。它用于复现严格无标签跨传感器/跨工况系统评估，不是已经证明统一优越的新 WIDAN 方法。

## 已包含

- 20 条主矩阵：3→3 同工况、只跨工况、双跨，以及同工况 3→1。
- 24 条跨工况/双跨 3→1 扩展：每个目标槽 0、1、2 分别训练、分别评估，结构与代码共用。
- 水泵、实验室齿轮箱、东南大学齿轮箱上的线性 Ridge 与 RBF-KRR 工件。
- clean、20/10/0 dB AWGN 结果、校准失配实验、逐类召回和混淆矩阵。
- `predict_kernel_readout.py`：对 `[N,C,128]` 特征做无标签目标统计校准和推理；不读取目标标签、不重新调参。
- 44 条件、88 个固定模型的校正后 target-final 估计及完整撤回记录。

## 主要结果边界

开发矩阵中，源验证选参线性 Ridge 的 clean Acc/Macro-F1 按数据集等权平均为：水泵 68.400/67.003%，实验室齿轮箱 76.500/75.818%，东南大学齿轮箱 63.917/62.465%。跨工况/双跨 3→1 扩展的 clean 均值分别为 57.622/56.918%、73.500/72.345%、32.861/31.592%。

校正后的固定模型 target-final 全 44 条件等权估计为：线性 Ridge 59.675/58.338%，RBF-KRR 55.893/53.666%。由于早期无效 v1 已解析过 final 元数据，这些数值只能写成 **post-exposure fixed-model estimate**，不能写成全新盲测。

AWGN 结果证明中等噪声下仍有可用性，也显示 0 dB 下明显退化；噪声种子不是训练种子或独立采集重复。当前每类/工况主要来自一条长记录切窗，不能声称已证明跨独立采集泛化。

## 使用入口

1. 先阅读 `KERNEL_INFERENCE.md` 与 `SHARED_RIDGE_INFERENCE.md`。
2. 使用 `predict_kernel_readout.py` 加载冻结模型与已计算的 `[N,C,128]` 特征。
3. 论文数字以 `evidence/reconcile_publication_tables_20260919.md`、`evidence/kernel_3to1_expansion_findings_20260918.md` 和 `evidence/kernel_final_corrected_v2_summary_20260919.md` 为准。
4. 运行 `verify_snapshot.py` 校验整个目录；原始波形数据不随包复制，仍从项目数据目录读取。

本快照从 `20260919_research_snapshot_v2` 完整复制，只修正了旧 README/MANIFEST 对 target-final 状态的过时描述，并新增独立目录校验器。模型、实验结果和原审计文件未修改。
