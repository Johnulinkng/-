# CRA-v2A screen_v5 结果与机制审计

日期：2026-09-25  
对象：`experiments/20260925_channel_robust_v2a_screen_v5`  
范围：只读复核已封存的 4 个共同锚点、32 个分支、32 个 target-dev 结果、逐 epoch 日志、kernel/calibration/alignment 统计；未训练新模型，未读取 target-final。

## 结论

`v2A-U-fixedK` **没有通过扩展门，不能补 seed43/44，也不能按数据集或槽位挑选有利结果继续使用**。相对配对的 common-anchor control：

- 16 个条件仅 7 个 Accuracy 与 Macro-F1 同时提高；
- 等权平均 Accuracy `-0.34 pp`，Macro-F1 `-0.37 pp`；
- 双跨 8 条件 Macro-F1 `-1.98 pp`；
- 新增 3 个零召回：`PG-D1/slot1/class0`，`WP-D1/slot2/class2,class4`；
- 最差为 `WP-D1/slot0`：Accuracy `-11.47 pp`、Macro-F1 `-8.44 pp`；`WP-D1/slot2` 为 `-4.00/-4.02 pp`。

失败不是冻结计划、分叉公平性、目标统计不一致或 signed U-statistic 数值错误造成的。16 对分支的起点、source rows、mask、source calibration 与 paired target calibration 均一致；全部 source-val 在第二阶段保持 `100%`；MMD 有限且绝大多数为正。现有证据支持的解释是：**无条件边际对齐与双跨后的类别条件结构不兼容；全局梯度符号门没有控制对齐梯度的相对范数、累计剂量或逐类损害。**

不建议立刻启动另一套真实目标 32-arm GPU screen。先完成下述 v2B 的源侧伪目标预验证；若预验证不过，双跨应回退 common-anchor control，停止继续增加 target-gradient 模块。

## 1. 分层结果

|切片|条件数|Accuracy 差 pp|Macro-F1 差 pp|平均 alignment 接受率|加权 alignment 标量 / source 标量|
|---|---:|---:|---:|---:|---:|
|PG|8|+0.75|+0.43|9.22%|10.2×|
|WP|8|-1.43|-1.17|5.23%|11.7×|
|同工况跨测点 S|8|+1.24|+1.23|11.19%|6.1×|
|测点+工况双跨 D|8|-1.93|-1.98|3.26%|15.8×|
|all3|4|+0.33|+0.22|4.88%|13.1×|
|三个 singleton 合计|12|-0.57|-0.57|8.00%|10.2×|

细分后，`PG-S1` 三个 singleton 全部改善（Accuracy `+2～+5 pp`，Macro-F1 `+3.20～+3.54 pp`），但 `PG-D1` 没有一个 singleton 同时提高。`WP-S0` 三个 singleton 一正一负一近中性；`WP-D1` 的 all3 与 slot1 小幅提高，slot0 与 slot2 明显下降。因此不能把差异归因于某一个数据集或“单通道天然更难”；决定性相互作用是 `数据集 × S/D × 目标槽位`。

逐类证据比总体均值更清楚：

- `PG-D1/slot1` 的 class0 召回从 `8%` 降至 `0`；
- `WP-D1/slot2` 的 class2 从 `0.67%` 降至 `0`，class4 从 `19.33%` 降至 `0`；
- `WP-D1/slot0` 虽未新增零召回，class4 从 `82.67%` 降到 `36.00%`，造成最大总体退化；
- `WP-D1/all3` 在 control 与 v2A 中 class0、class2 本来都为零；小幅总体提高不能称为解决双跨。

## 2. 为什么 WP、PG 分化

### 2.1 累计更新剂量不同

相同的“20 epoch、lambda=0.02”并不代表相同干预剂量。WP 每个分支有 `920` 个第二阶段 optimizer steps，PG 只有 `160` 个，比例为 `5.75×`。最终接受的 alignment steps 为：

- PG：每臂 `5～34`；
- WP：每臂 `18～91`。

候选相对 control 的 encoder 参数相对 L2 位移，WP 八臂平均约 `13%`，PG 约 `2%`。因此 hard gate 虽把约 78.8%～98.0% batch 的 alignment 关闭，WP 仍累积了更大的表示位移。按 epoch 固定预算不能跨数据量公平，应改成按梯度范数和累计参数冲量定义预算。

### 2.2 source 已饱和，低 lambda 仍不是小扰动

全部分支的 source-val Accuracy 都是 `100%`，source CE 已很小。`0.02 × alignment_scalar / source_total_scalar` 的逐臂均值范围约 `3.8×～20.4×`；D 族平均 `15.8×`，S 族 `6.1×`。标量比不等于梯度范数比，但足以否定“lambda=0.02 所以干预必然很弱”的解释。现有实现只检查全局 `dot(g_source,g_align)>=0`，没有：

- 对齐梯度相对 source 梯度的范数上限；
- 每个类的 source 梯度保护；
- 跨 epoch 的累计 alignment impulse 上限；
- Adam 一、二阶矩中 target gradient 的累计占比上限。

一个全局正内积更新仍可损害某个源类的判别方向，并在目标域表现为类别召回归零。

### 2.3 WP 的 train-only 边际偏移更大

从已封存 checkpoint 的 calibration buffers 计算，source/target center RMS：

- PG 各臂 `0.601～1.132`；
- WP 各臂 `1.281～2.212`。

log-scale RMS：PG `0.205～0.312`，WP `0.330～0.418`。target z-score 会消除逐频率一、二阶边际差异，却不能保证故障类别条件分布保持同一映射。WP 的剩余条件错位更强，解释了为何更大的 encoder 位移没有转成统一收益。偏移大小本身也不能作选择器：`WP-D1/slot1` 的 center RMS 最大（2.212）却小幅改善。

## 3. 为什么 S 与 D、all3 与 singleton 分化

### 3.1 双跨不是单纯更大的同一种 covariate shift

D 族平均梯度 cosine 约 `-0.21`，S 族约 `-0.13`；D 族 alignment 接受率更低，同时加权标量比更高。D 族 MMD 在训练中显著下降，但 Macro-F1 反而下降，尤其 WP-D1 slot0/slot2。这说明优化器确实在匹配域边际，问题是匹配方向没有保持故障类别对应关系。速度/负载改变会移动与重排类别频谱；对混合分布做无条件 MMD 可以把一个目标故障簇拉向错误源类。

本轮 16 臂中，Macro-F1 差与 `alignment/source scalar ratio` 的描述性 Pearson 相关约 `-0.52`、Spearman 约 `-0.58`；与 channel-MMD 均值分别约 `-0.49/-0.51`。样本仅 16 且分组混杂，这些数值不能用来拟合真实目标阈值，但支持“更强边际压力不是更安全”的机制诊断。

### 3.2 all3 的传感器平均提供了决策缓冲

all3 四条件平均 F1 `+0.22 pp`，singleton 合计 `-0.57 pp`。PG-D1 all3/slot2 的候选与 control 预测完全相同，即使分别接受 5/6 个 alignment step；更新没有跨过决策边界。单槽缺少其它传感器平均后的冗余，较小 encoder 位移也更容易改变类别边界。WP singleton 的 encoder 位移约 `11.5%～17.3%`，明显高于 PG singleton 的 `1.3%～3.5%`。

attention/fusion 不是本轮分化的直接来源：候选相对 control 的 `fusion_score` 参数位移为零；差异来自共享 encoder，分类头只因后续 source update 与 encoder 轨迹不同而间接分化。

## 4. v2B 建议：类条件、稳定分配、预算化且可回退

建议把下一机制命名为 **v2B-BCPA（Budgeted Class-Prototype Alignment）**，避免与旧文档中的“名义阶次 v2B”混淆。它保留共同 source anchor、独立 3→1 训练和 target-train z-score，删除 fused/channel 无条件 MMD。

### 4.1 目标损失

1. 用有标签 source-train 的 anchor features 计算每类冻结原型与 source-val 类内半径；温度、距离归一化和支持阈值只由 source 或源侧伪目标确定。
2. 对无标签 target-adapt 计算到源原型的 soft assignment。只有在两种确定性视图/增强下类别一致、margin 达标且落在 source 支持范围内的样本才进入对齐。
3. 每类单独计算 soft target centroid 到对应冻结 source prototype 的距离；合格类别等权平均，不按目标预测数量加权。任一类的有效样本量不足时，该类不产生梯度；若覆盖门整体失败，整臂回退 common-anchor control。
4. 不把 target entropy、confidence 或预测均匀度单独当安全门。CRA-v1 的现有回顾已证明简单 entropy/confidence 规则会在 WP-D1 和 PG-D1 选错。

该机制仍依赖 pseudo assignment 正确，不能理论保证无负迁移。特别是 `WP-D1/slot2` control 低于随机水平附近，可能得到“稳定但错误”的 assignment。因此必须配套源侧伪目标预验证和 fail-closed 回退，不能直接跑真实目标后按 dev 结果调阈值。

### 4.2 source 保护与数据量无关的剂量合同

每个 target gradient 同时满足：

- **逐类保护**：对每个当前 source batch 中出现的类，投影后 target gradient 与该类 source gradient 的内积不小于零；只检查聚合 source gradient 不够；
- **瞬时范数上限**：`||g_target_safe|| <= rho * ||g_source||`；`rho` 由源侧伪目标 episode 预注册，不用真实 target-dev 调；
- **累计冲量上限**：记录 `sum ||delta_theta_target|| / sum ||delta_theta_source||`，到达固定预算后本臂余下 step 只做 source update；
- **固定 step 而非固定 epoch**：WP/PG 使用相同的 source examples/step 合同，或至少报告并限制每千 source 样本的 target impulse；
- classifier、fusion 和 source/target calibration buffers 不接收 target gradient。

### 4.3 只用无标签目标数据可计算的门

训练前将 target-train 按时间顺序锁成 `target-adapt / target-val-u`，中间留 gap；以下全部不读 target-dev/final 标签：

|门|计算量|失败动作|
|---|---|---|
|逐类 soft mass 与 ESS|`m_k=sum q_ik`，`ESS_k=(sum q_ik)^2/sum q_ik^2`|任一类低于预注册阈值则不对齐该类；覆盖不足则整臂回退|
|跨视图 assignment 一致性|同一窗口两视图的 hard agreement、Jensen-Shannon divergence|低一致样本不参与；总体不足回退|
|source-support 距离|目标到最近源原型距离相对 source-val 类内半径|超支持样本不参与；超支持比例过高回退|
|prototype margin|第一、第二近原型距离差/soft margin|margin 不足不生成 pseudo assignment|
|逐类梯度 cosine|每类 source gradient 与 target prototype gradient|任一冲突方向先投影，无法满足所有约束则该 step 纯 source|
|瞬时与累计 impulse|梯度范数比、参数更新分解、Adam moment 中 target 占比|超过预算即停止 target update|
|control/candidate 预测漂移|target-val-u 上 label-switch 率、soft KL、类别质量变化|只作审计与回退辅证，不能单独决定采用候选|

若目标数据构建协议明确承诺各类等配额，可把“均匀先验”作为公开先验写入合同；否则不得用均匀 Sinkhorn 强制每类相等，因为那会把未知 label shift 当成已知。

原始 DEV 在本批不可用：source-val 0/1 error 全为零，两个模型风险同为零。CE-DEV/IW-GAE 只能在独立源侧伪目标 episode 上验证 ESS、域分类器和 selector regret 后作为附加 validator；任一 validator 无效或相互冲突时回退 control。

## 5. 是否启动与最小探索矩阵

### 决策

**现在不启动真实目标 v2B 主屏。**先做无 GPU/小 CPU 的实现与 source-side pseudo-target 门；只有门通过才启动 seed42。旧“名义阶次 v2B”也不建议直接启动：现有 WP 探针把 WP-C1 target-zscore 从 `100%` 降到 `45.2%`，WP-D1 虽升到 `63.6%`，正常类召回仅 `8%`；2048 点低速窗口不足一转，普通频轴缩放不能恢复缺失分辨率。

### A. 必须先过的无标签/源侧门

1. 合成单元测试：类别覆盖、ESS、稳定 assignment、逐类梯度投影、范数和累计预算、回退、RNG replay。
2. 至少 4 个与真实 dev 分离的源侧伪目标 episode，覆盖 `WP/PG × S/D`；每个 episode 隐藏伪目标标签，固定机制和门后才解封评分。
3. 每个 episode 保留 `all3/slot0/slot1/slot2`，比较 common-anchor control 与 v2B，seed42：`4 × 4 × 2 = 32` 个第二阶段分支，4 个共同锚点。v2A 已证明数据集、S/D 与槽位有强交互，再缩成 8 臂会漏掉已经出现的失败类型。
4. 伪目标扩展门沿用当前门：至少 10/16 双指标提高、均值 Acc `>=+1 pp`、F1 `>=+2 pp`、D 族 F1 `>=+2 pp`、每任务至少两个 singleton 提高、最弱 3→1 改善、无新增零召回；另加“自动回退所选模型相对 control 最差退化不超过 2 pp”。

门失败则不读取真实 target-dev 来修阈值，停止 v2B target-gradient 路线。

### B. 通过后最小真实开发屏

`WP-S0/WP-D1/PG-S1/PG-D1 × all3/slot0/slot1/slot2 × common-anchor control/v2B × seed42`，仍为 32 个第二阶段分支、4 个锚点。所有 threshold、rho、累计预算和回退规则必须来自 A 并在计划中冻结；完成全批和 SHA seal 后才一次性读取 target-dev。只有同一门通过后才增加 seeds43/44。

为保持可归因性，首个 v2B screen 只比较 control 与完整 v2B。若通过，再在 `WP-D1/PG-D1 × 四目标 × seed42/43/44` 做三项单因素消融：去掉类条件、去掉逐类保护、去掉累计预算。未通过时不再为完整 recipe 追加消融以寻找有利子集。

## 6. 不建议的下一步

- 不按 PG 保留 v2A、WP 回退 control；这是看完 target-dev 后的数据集特定策略。
- 不仅把 lambda 再降一档后重复 32 臂；它不能解决类别条件错配，也没有数据量无关的累计剂量定义。
- 不把 fused-only EQ-Lite 直接当主方案；删除 channel-MMD 会降低复杂度，但 fused marginal alignment 仍有同一类错配风险。
- 不用 target entropy、confidence、预测均匀度或 MMD 降幅单独选 checkpoint；现有项目证据已否证这些简单规则。
- 不依据本轮 16 个 target-dev 点拟合阈值。本文给出的相关性只作失败解释，不是 v2B selector 训练数据。

## 证据入口

- 完整门与 16 对结果：`experiments/20260925_channel_robust_v2a_screen_v5/summary.json`
- 32 个混淆矩阵与逐类召回：`experiments/20260925_channel_robust_v2a_screen_v5/dev_results.json`
- 分支逐 epoch MMD、cosine、接受/抑制步：`experiments/20260925_channel_robust_v2a_screen_v5/runs/*/epoch_metrics.jsonl`
- 锚点 kernel 与最终 alignment 计数：对应 `run_state.json`、`run_metadata.json`
- 已有工件级参数位移诊断：`audit/channel_robust_v2a_mechanism_diagnostic_20260925.json`
- 简单无标签选择器的既有否证：`audit/channel_robust_v1_unsupervised_selection_diagnosis_20260925.md`
- 名义阶次负面边界：`order_probe/20260912_wp_nominal_order_v1/REPORT.md`

本审计只支持“v2A 在固定开发矩阵失败，以及下一机制应如何被证伪”。它不支持 v2B 会提高真实目标性能，也不改变 target-final 继续封存、每类每工况单记录不能证明独立采集泛化的边界。
