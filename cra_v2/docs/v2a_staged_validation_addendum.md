# CRA-v2A 分阶段验证补充说明

日期：2026-09-25  
状态：**仅冻结执行顺序与停止规则；未启动训练、WSL 或 GPU，未读取 target-final。**

本文件不改写 `channel_robust_v2_design_decision_20260925.md`、`channel_robust_v1_failure_diagnosis_20260925.md` 或 `channel_robust_v2_validation_protocol_20260925.{md,json}` 的证据和结论。原 216 臂协议保留为后续完整扩展注册表；当前只采用下面的分阶段执行路径，避免在主方案尚未通过开发门槛前直接投入 216 臂。

## 1. 第一阶段：v2A 的 32 工件开发筛查

固定条件为：

`WP-S0 / WP-D1 / PG-S1 / PG-D1 × all3 / slot0 / slot1 / slot2 × common_anchor_20ep_control / v2A-U-fixedK × seed42`。

这是 16 个 task-mode 条件、每条件 2 个第二阶段分支，共 **32 个第二阶段工件**。此外每个 task/seed 只生成一个共同 source anchor，共 4 个共享前置锚点；它们不重复计作 32 个独立比较臂。所有工件完成 checkpoint、代码、数据、配置和 RNG 哈希及 predev seal 后，才允许一次性读取 target-dev 评分。

v2A 是完整的两阶段 recipe，不能写成“只改变 MMD”：

1. 两支先共用固定 60 epoch 的 source-only 预训练，并只按 source-val Accuracy、较低 CE、较早 epoch选择共同锚点；
2. `common_anchor_20ep_control` 与 `v2A-U-fixedK` 从完全相同的 model、optimizer、scheduler 和 RNG state 分叉，再各跑固定 20 epoch，使用相同 source update、batch/mask 序列并只取末轮；
3. 两支的允许差异仅为 v2A 的 fixed-kernel、equal-B、signed unbiased U-statistic fused/channel MMD、固定 `lambda=0.02` 和冲突时关闭 alignment gradient；
4. target-train 标签始终为 `-1`，target-dev 不参与训练、锚点选择、epoch 选择或方法选择。

冻结 v4 的强 `matched_control` 只作为**第二、历史基线**复用，不新增训练臂。复用前必须核对 v4 plan/summary 以及逐臂 checkpoint、代码、数据和配置哈希；条件不完全匹配或 seal 缺失时标记为不可比较，禁止补跑后冒充历史基线。由于 v4 `matched_control` 没有本轮共同 60 epoch 锚点加配对 20 epoch 合同，它不能替代 `common_anchor_20ep_control`，也不能用于归因 v2A 的因果增益。它只回答 v2A 是否超过现有强基线；若未超过，结果仍可用于机制判断，但不得声称“当前最佳模型”。

第一阶段继续门槛保持原样：

1. 16 条件中至少 10 条 Accuracy 与 Macro-F1 同时提高；
2. 16 条件等权平均 Accuracy 至少 `+1 pp`、Macro-F1 至少 `+2 pp`；
3. 双跨 8 条件等权 Macro-F1 至少 `+2 pp`；
4. 每个任务的三个单目标槽中至少两个槽的 Accuracy 与 Macro-F1 同时提高；
5. control 中最弱的 3→1 条件 Macro-F1 提高；
6. 不新增零召回类别。

任一项失败即停止：不补 seeds 43/44，不进入第二阶段，不启动 SEU/AWGN，也不按已见 target-dev 结果改任务、权重或门槛。

## 2. 第二阶段：六个固定新方向

只有第一阶段原门槛全部通过，才固定以下实验室方向；选择规则与历史结果无关，且水泵用 `S1`、齿轮箱用 `S2`：

| 家族 | 水泵 | 实验室齿轮箱 |
|---|---|---|
| 同工况跨测点 3→3 | `WP-S1`：C2/P1→C2/P2 | `PG-S2`：C2/G2→C2/G3 |
| 同测点纯跨工况 3→3 | `WP-C2`：C2/P1→C3/P1 | `PG-C2`：C2/G2→C4/G2 |
| 测点与工况双跨 3→3 | `WP-D2`：C3/P1→C4/P2 | `PG-D2`：C3/G2→C2/G3 |

第二阶段先只跑 `all3`，仍比较 `common_anchor_20ep_control` 与 `v2A-U-fixedK`，训练合同与第一阶段完全相同。最小启动批为 `6 directions × 2 branches × seed42 = 12` 个第二阶段工件。它只承担方向资格筛查，不能作为三种子稳定性或论文主结论。

12 工件全部密封后才统一评分；继续 seeds 43/44 必须同时满足：

1. 6 个方向至少 4 个 Accuracy 与 Macro-F1 同时提高；
2. 6 个方向等权平均 Accuracy 至少 `+1 pp`、Macro-F1 至少 `+2 pp`；
3. 两个 `D2` 方向等权平均 Macro-F1 至少 `+2 pp`；
4. 水泵和齿轮箱各自至少 2/3 方向双指标提高；
5. 不新增零召回类别。

任一项失败即停在 12 工件并完整报告负结果，不补有利种子、单槽、SEU 或噪声实验。全部通过后再增加 `6 × 2 × seeds{43,44} = 24` 个工件；第二阶段总上限因此为 **36 个第二阶段工件**，而不是 216 个训练臂。三种子最终报告按每个方向的三种子均值计算上述同一组门槛，同时报告每种子的原始 Accuracy、Macro-F1、逐类召回、混淆矩阵和均值/标准差；不得删除失败方向或失败种子。

第二阶段这些方向的 target-dev 在历史实验中已有不同方法的暴露记录，因此它们属于“冻结新方向的开发确认”，不是新盲测、独立确认集或独立采集泛化。`target-final` 永不构造 loader、永不读取；公开数据与 AWGN 只有在本补充说明规定的实验室门槛通过后，才能按各自预注册协议启动。

## 3. 证据身份

- v2A 设计决定：`D:\3800\audit\channel_robust_v2_design_decision_20260925.md`，SHA-256 `90AC6ED390C3FBF8C9038D7F029BAEC3740548F007345C164AD33D97DA789EDF`；
- v1 失败诊断与原门槛：`D:\3800\audit\channel_robust_v1_failure_diagnosis_20260925.md`，SHA-256 `A6B2DC6D1FBAC6CDB24EE57C26DA9C186ED0D41C5126BCC3AF6205A712633C97`；
- 完整 v2 扩展协议 JSON：`D:\3800\audit\channel_robust_v2_validation_protocol_20260925.json`，SHA-256 `5EBC2070EFF4ED3AAFE1DEA1E6C3C5E9D86E184EF811315B1B3F48D000A0C208`；
- v4 历史基线 plan：`D:\3800\experiments\20260925_channel_robust_screen_v4\plan.json`，SHA-256 `5FB4A31DF51AC095A14C656936708C5132FC4A2DFB01FABD958F67FAE6730645`；
- v4 历史基线 summary：`D:\3800\experiments\20260925_channel_robust_screen_v4\summary.json`，SHA-256 `5782D567FEE6B4817F733E22CBC6107D0975F98DB7672CE659154BA73AD91617`。
