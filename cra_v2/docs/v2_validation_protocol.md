# CRA-v2 最小诚实验证协议

日期：2026-09-25  
状态：**只完成协议设计，未启动训练、WSL 或 GPU**。机器可读协议为 `D:\3800\audit\channel_robust_v2_validation_protocol_20260925.json`，SHA-256：`5ebc2070eff4ed3aafe1dea1e6c3c5e9d86e184ef811315b1b3f48d000a0c208`。

## 结论先行

CRA-v1 的四任务是 `WP-S0 / WP-D1 / PG-S1 / PG-D1`。v2 不重复用这四项做“新验证”，而按与历史分数无关的固定规则，在每个数据集的 S/C/D 家族中选择**第二个 source-first 任务**：

| 数据集 | 同工况跨测点 3→3 | 同测点跨工况 3→3 | 测点与工况双跨 3→3 | 逐槽 3→1 |
|---|---|---|---|---|
| 水泵 | `WP-S1`，C2/P1→C2/P2 | `WP-C2`，C2/P1→C3/P1 | `WP-D2`，C3/P1→C4/P2 | `WP-S1` 的 slot 0/1/2 |
| 实验室齿轮箱 | `PG-S2`，C2/G2→C2/G3 | `PG-C2`，C2/G2→C4/G2 | `PG-D2`，C3/G2→C2/G3 | `PG-S2` 的 slot 0/1/2 |
| SEU | `SEU-S2`，20_0 平行→行星 | `SEU-C2`，30_2→20_0 行星 | `SEU-D2`，30_2 平行→20_0 行星 | `SEU-S2` 的 x/y/z 三槽 |

实验室先运行；只有实验室冻结门槛通过，才运行 SEU。总上限为 **216 个训练臂**：实验室 144，SEU 72。

## 证据边界

这不是新盲测。安全的历史结果索引确认：

- `WP-S1 / PG-S2` 的 all3、`WP/PG-C2/D2` 的 all3 已出现在 2026-09-12 的 target-dev 校准结果中；
- `SEU-S2/C2/D2` 已出现在 `shared_ridge_seu_v1` 的 target-dev 结果中，其中 `SEU-S2` 的 all3 与三个单槽均已评分；
- 实验室 `WP-S1 / PG-S2` 的三个单槽没有在本次有界历史指标索引中找到精确的独立训练 3→1 task-mode 记录，但它们复用的 target-dev 分区已被 all3 实验解码，所以只能称为“新的任务—通道组合”，不能称为未见数据。

因此，v2 结果只能写成：**在历史已暴露开发分区上、预先冻结的新任务方向验证**。禁止写“全新盲测”“独立确认集”“独立采集泛化”。本协议没有读取 target-final，也永久不允许构造 target-final loader。

## 数据合同

正式 v2 只能使用以下 schema-3 public 根：

- 水泵：`D:\3800\data_versions\antipair_v1_waterpump_source_first`
- 实验室齿轮箱：`D:\3800\data_versions\antipair_v1_gearbox_source_first`
- SEU：`D:\3800\data_versions\antipair_v1_seu_gearbox_source_first`

旧 `prepared_transfer_tasks*` 直接包含 `target_label.npy`，v2 runner 必须拒绝这些路径。schema-3 public 目标域只提供波形和 provenance，目标标签位于 sibling private-audit 树，训练入口必须把 target-train 标签视为 `-1`。

三套 schema 都标记 `record_independent=false`：窗口来自长时记录或时间块，不是独立重复采集。 每套数据的 `*_private_audit/reference/raw_records.json` 仅作为原始文件名、工况、通道和类别的 lineage 收据，其 SHA 已写入 JSON；不得沿该路径打开任何 target-label 数组。实验室水泵与齿轮箱 manifest 没有权威采样率，因此本轮可以冻结 v1 的 bin-index 表示做配对比较，但不能把 `low3k_pool2` 解读成已经核实的 0–3 kHz 物理频带。SEU 的 5120 Hz 有 manifest 约束。

## 四个固定配置

1. `matched_control`：均值融合，无 mask、MMD 和投影；它保留共享校准合同，不能偷换成严格 Source Only。
2. `cra_v1_frozen`：原 CRA-v1 完整配方，保留 3→1/3→3 样本数不等的有偏 channel-MMD。
3. `cra_v2_no_channel_mmd`：保留融合、mask、fused-MMD 和投影，但 channel-MMD 权重严格为 0。
4. `cra_v2_balanced_channel_mmd`：唯一变化是 channel-MMD 两侧都抽取 `[B,D]`，按物理槽分层轮转；损失权重和其他算法不变。

种子固定 `42/43/44`；mask seed 固定为 `424242 + seed`，channel-MMD 独立 seed 固定为 `525252 + seed`。四套配置、三个种子、所有任务和目标槽必须完整报告。

## 训练矩阵

每个 task-mode 有 4 配置 × 3 seeds = 12 个训练臂。UDA checkpoint 依赖 target-train 的目标槽，因此不能把 all3 checkpoint 直接拿来冒充 slot checkpoint。

- 实验室：6 个 all3 task + `WP-S1/PG-S2` 各 3 个单槽，共 12 个 condition cells，**144 臂**。
- SEU：3 个 all3 task + `SEU-S2` 三个单槽，共 6 个 condition cells，**72 臂**。
- 最大总计：**216 臂**。

训练参数与 v1 配对冻结：batch 32，60 epochs，epoch 21 起按 source-val 准确率、较低交叉熵、较早 epoch 依次选模，lr 0.001，weight decay 0.0005，adaptation weight 0.1，gradient norm 5，encoder width 64，`low3k_pool2`，单臂 1800 秒硬上限。target-dev 不参与训练、epoch 选择、方向选择或超参数选择。

## 执行顺序

1. **P0 实现门**：先通过等样本数、500-seed 同分布、域偏移单调、1000-batch 槽平衡、RNG 隔离、梯度投影、物理槽置换和 checkpoint fail-closed 测试。失败则不训练。
2. **P1 实验室密封训练**：144 臂全部完成并写入 checkpoint/code/data/RNG 哈希；此时不得读取 target-dev。
3. **P2 一次性开发评分**：先固定所有预测，再一次解码完整实验室 target-dev，不能边看边改。
4. **P3 SEU 密封训练**：只有 `G_lab` 原样通过才启动 72 臂。
5. **P4 SEU 一次性评分**：72 checkpoint 全部密封后统一评分。

## 冻结门槛

`G_lab` 同时要求：

- v2 相对 v1 的实验室 pooled Accuracy 与 Macro-F1 都大于 0；
- all3 的 S/C/D 三个 family 的 Macro-F1 差均不负，D 必须严格为正；
- 水泵、齿轮箱各自 all3 pooled Macro-F1 不负；
- 3→1 pooled Accuracy/F1 都为正，6 个 task-slot 三种子均值中至少 4 个 Accuracy/F1 同时改善，且两个数据集的 3→1 pooled F1 都不负；
- v2 相对 no-channel-MMD、matched control 的 pooled Accuracy/F1 均为正；
- 相对每个控制，新零召回类别总数不增加。

任一条件失败：停止 SEU、AWGN 和更大矩阵；保留负结果，不改权重，不删任务，不追加“有利”种子。

`G_external` 要求 v2 相对 v1 的 SEU pooled Accuracy/F1 为正，S/C/D all3 的 F1 均不负，三个 `SEU-S2` 单槽至少两个共同改善，同时超过 no-channel-MMD 和 matched control，且不增加零召回类别。失败时只能把 SEU 写成外部限制，不能删除不利方向。

## 资源预算

v4 的 32 臂累计训练 991.26 秒，中位单臂 20.31 秒，最大 84.90 秒。按该实测并加约 50% 余量：

- 实验室训练上限 2.0 GPU-hours；
- SEU 训练上限 1.0 GPU-hours；
- 两次完整评分上限 0.5 GPU-hours；
- 总上限 **3.5 GPU-hours**。

启动前必须重新核验 GPU compute process/显存、RAM/commit、WSL/PID/锁、磁盘以及 plan/code/data hash。超预算只能在臂边界暂停，不能减少种子、任务、epoch 或控制组。

## 可以支持和不能支持的结论

通过本协议可以支持：“balanced channel-MMD 在已暴露开发分区的新任务方向上，相对 v1 和机制消融稳定改善。”它仍不能单独支持 SOTA、高性能、独立记录泛化、抗噪或优于原 WIDAN/Ridge/严格 Source Only。后四项必须在本门槛通过后另行冻结强基线与 AWGN 协议。
