# CRA-v2A screen v5：负迁移机制诊断与下一步设计

日期：2026-09-25。对象：`experiments/20260925_channel_robust_v2a_screen_v5`。
本诊断读取已完成的 4 anchor、32 checkpoint、epoch 日志、既有 dev 混淆矩阵，未训练、未加载数据波形、未打开目标标签文件或 final，也未改变主代码。可复核计算见 `audit/diagnose_channel_robust_v2a_20260925.py` 和同名机制 JSON。

## 结论

v2A 的失败不是“校准缓冲跑乱”或“3→1 的源/目标样本数仍不相等”所致。共同锚点、成对源批次和 mask、目标校准缓冲的身份检查均通过。**对齐仅在 5.82% 的批次被允许，仍出现类别边界改变与双跨退化；减少域间边缘距离、保留源域准确率并不能保证目标类别语义保持。** 当前没有证据认定某一个 MMD 层单独负责负迁移，因此下一轮不宜直接把 channel-MMD 删除后宣称原因已解决。

原门槛继续有效：16 对仅 7 对双指标提升，平均 Accuracy −0.342 pp、Macro-F1 −0.374 pp；双跨 Macro-F1 −1.98 pp。3 个新零召回事件。v2A 不应扩 seed43/44 或按数据集挑选保留。

## 1. 定量轨迹

|任务/目标槽|Accuracy Δ pp|Macro-F1 Δ pp|对齐实际注入率 %|平均 encoder cosine|fused/channel MMD 均值|
|---|---:|---:|---:|---:|---:|
|WP-S0/all3|−0.267|−0.363|4.46|−0.219|0.569/0.268|
|WP-S0/0|+1.333|+1.202|8.48|−0.168|0.293/0.260|
|WP-S0/1|−3.733|−2.278|7.28|−0.156|0.304/0.273|
|WP-S0/2|+1.600|+1.070|9.89|−0.155|0.257/0.230|
|WP-D1/all3|+1.600|+1.255|1.96|−0.310|0.703/0.316|
|WP-D1/0|−11.467|−8.444|3.70|−0.228|0.382/0.348|
|WP-D1/1|+3.467|+2.179|3.48|−0.234|0.386/0.354|
|WP-D1/2|−4.000|−4.015|2.61|−0.224|0.489/0.448|
|PG-S1/all3|0|0|10.00|−0.120|0.499/0.205|
|PG-S1/0|+4.000|+3.476|21.25|−0.076|0.348/0.251|
|PG-S1/1|+2.000|+3.198|14.37|−0.084|0.555/0.348|
|PG-S1/2|+5.000|+3.543|13.75|−0.083|0.355/0.218|
|PG-D1/all3|0|0|3.13|−0.202|0.710/0.463|
|PG-D1/0|−3.000|−4.007|3.13|−0.190|0.790/0.686|
|PG-D1/1|−2.000|−2.802|4.38|−0.159|0.694/0.608|
|PG-D1/2|0|0|3.75|−0.166|0.427/0.371|

所有 32 支在全部第二阶段 epoch 的 source-val Accuracy 都是 100%。平均 raw encoder cosine 为 −0.173；总体 94.18% batch 关闭对齐。每条件注入率为 1.96–21.25%。不能用这些饱和的 source-val Accuracy 选择 target 表现更好的分支。

fused/channel 的 signed MMD 大多数为正；负值仅见 WP-S0 的个别单槽，最高负值比例不足 1%。因此该次失败不能主要归因于 U-statistic 出现负值。WP-D1/0 的 fused MMD 从前五轮 0.51 降到末五轮 0.29，同时 dev Accuracy 反而下降 11.47 pp；“loss 下降等于适应有效”在此明确不成立。注意这是同一 candidate 的时序 loss 与末端配对 dev 差的描述，不是对未评分中间 checkpoint 的性能推断。

固定权重 0.02 也不等于弱相对干预：`0.02 × mean(alignment) / mean(source_total)` 范围约 3.83–20.41。但这是**标量比值，不是梯度范数或实际 update 比值**。本次日志没有逐层范数、clip coefficient、Adam 预条件后的 update geometry，所以不能量化究竟哪个层的更新主导。

## 2. 已排除的混杂与未覆盖的保护

- 16 对 start model/optimizer/scheduler/RNG 身份、source row digest、source mask digest 完全一致；采用同一 source anchor 与固定 20-epoch 末端，因果比较成立在此配对合同下。
- 16 对 target_center/target_scale/target_calibrated 完全相同；每支 source_center/source_scale 都与 anchor 相同。目标缓冲没有训练漂移或 pair 之间不一致。此检查不证明共享的 z-score 假设本身最佳。
- channel U-MMD 的源/目标都是 B 行，物理槽平衡；修复了旧 3B-vs-B 问题。
- **fused 路径仍有聚合宽度不匹配**：3→1 时源端是三传感器融合表示，目标端是单传感器表示。即使每侧 B 行，融合方差/支持集/类别几何仍可能不同；equal-B 没有解决这个因素。现有两层联合对齐不允许把它与 channel loss 的贡献分开。
- gate 只在 `encoder.*` 参数上计算未预条件 raw gradients 的 dot，但通过后还给 `fusion_score` 注入 alignment。故不能称为“全部接收 alignment 的参数都不冲突”。classifier 没有直接 alignment gradient，但共享 encoder 改变后 source 梯度和 Adam 轨迹会间接改变 classifier；checkpoint 实际存在这种差异。
- Adam 将一阶/二阶矩、weight decay 与 clip 后梯度转成实际更新；raw encoder dot≥0 不能保证实际单步 source loss 不增加，更不能保证 target risk。以前通过的 alignment 会留在 optimizer 状态中，当前 batch gate 关闭并不恢复 pure-source 轨迹。这是数学/实现层面的保护范围限制，不是对本次退化的已证实唯一归因。[Adam 原论文](https://arxiv.org/abs/1412.6980)

另有非因果解释性细节：协议文字写 `KL(masked||stopgrad(fused))`，代码 `kl_div(log_softmax(masked), softmax(fused.detach()))` 实际为 `KL(stopgrad(fused)||masked)`。两支共享同一实现，不能解释成对负迁移，但论文公式必须与已运行代码一致；本诊断未修改冻结代码。

## 3. 混淆矩阵显示类别交换，而非简单全局塌缩

- WP-D1/0：class 4 recall 从 82.67% 降到 36.00%，class 3 反而从 95.33% 升到 97.33%。该条件预测为 class 0 的比例从 7.07% 增至 14.80%，但是 class 0 真召回从 8.67% 降到 6.67%。预测分布改变不等于类别对齐改善。
- WP-D1/2：class 2 从 0.67% 到 0，class 4 从 19.33% 到 0，形成两个新零召回；但依然分别有 14.93% 和 26.40% 的样本被预测为这两类。**不是头部不再输出这些类别，而是输出落在错误样本上**，与条件类别语义错配一致。
- PG-D1/1：class 0 从 8% 到 0，其他类召回不变。源域始终 100% 无法发现此问题。
- WP-D1 两个弱槽 candidate/control encoder 相对 L2 差约 17%，仅几个百分点 batch 注入不能推出参数差很小；PG-D1 encoder 相对差约 1–2% 仍可跨过弱类决策边界。

这些证据支持“边缘匹配可以改变错误的类别关系”的解释，但不能从 dev 混淆反向决定训练时应把目标样本映射到哪一源类。理论上源误差小和分布不变也不足以保证目标风险小；本实验不假设该论文的 label-marginal-shift 条件一定在这里成立。[Zhao 等，ICML 2019](https://proceedings.mlr.press/v97/zhao19a.html)

## 4. 下一版：先检验聚合宽度假说，再决定是否启动新方法筛查

建议主线是**源端对齐视图与目标可见通道数一致的 fused 对齐**，而不是放宽 conflict gate、用目标熵挑 epoch、加伪标签或继续降低 lambda。动机是当前 equal-B 只匹配样本数，未匹配融合度。该建议是开发历史驱动的下一候选，不应宣称事前未看过 dev。

### 4.1 首先做来源严格的 source-side 机制预检

在现有 source-train 内预先锁定互斥时间宏块和传感器伪域：pseudo-source 使用指定源槽，pseudo-target 使用其余源槽/固定信号变换。pseudo-target fault labels 在训练 API 中仍屏蔽；其 label 只在独立 source-side 选择阶段使用，并明确仍是**源域来源的监督开发**，不是目标无标签性能证明。不使用已经看过的真实 target-dev 选视图、lambda 或预算。若每类宏块不足，停止而非随机打散相邻窗口伪造独立性。

对固定源 anchor，仅比较预定的两种 alignment 表示：三源融合 vs 与 pseudo-target 同可见通道数的源子集融合。源子集按物理槽连续平衡轮换，不按预测类别或目标置信度选槽，不构造跨域同步对。日志增加 fused/channel 各自的梯度范数、各自 source dot、fusion_score dot、合成梯度 norm、clip coefficient、Adam 实际更新与源梯度 dot、源每类 CE/margin、校准缓冲 SHA。全部都可在 source-side/pre-update 信息中计算。

先检验不含适配标签的合成反例：相同单传感器分布经三通道独立平均后方差变小，equal-B MMD 应识别这只是聚合度差；同宽融合应消除该差。这个检查只能验证实现/假说，不能代替真实性能。

### 4.2 若预检成立，冻结一个单一、可归因的真实候选

保持 v5 的共享 encoder、attention、mask/source loss、anchor、kernel source pool、20 epoch、lambda、冲突阈值、源选模、目标统计不变。只把 **fused MMD 的源表示**替换为和目标可见通道数一致的源子集融合：all3 不变；singleton 用平衡源槽单视图。channel MMD 暂时原样保留，以便把唯一因子归为“聚合宽度”，不要同时改 Adam/gate 或把两层删成一层。源端监督仍使用全部三通道及既有 masked objective，保留多源监督信息；每个真实目标槽分别训练评估，结构代码统一。

为了证明 channel 层的净贡献，**另做事前定义的 removal 对照**，在同一 source-side 伪域协议里只把 channel 系数置 0；该对照不作为看完真实 target-dev 后临时挑选的新主模型。若源预检没有支持聚合宽度假说，则不启动真实候选，保留失败诊断，转去已有证据支持的物理表示/采样协议问题。

真实筛查仅一个统一 candidate（16 条件），复用四个 SHA 锁定 source anchor；共同 anchor 控制必须逐轨迹复现，已有 control 可作为数值 replay 对照但不是新增训练样本。候选全部 fit/seal 后统一评分 dev，仍用既有扩展门槛，失败不补种子、不按槽挑方法。target-final 继续封存。这能检验具体机制，不保证提升，也不把当前 7/16 的局部提升变成论文方法成功。

gradient protection 如需改善，另立 optimizer-aware 研究：在相同 optimizer state 上计算 pure-source 与 combined 两个候选 update，并用独立 source-train guard microbatch 检查 source objective/class margin；拒绝时恢复全部 optimizer state 和参数，仅提交 pure-source update。不能仅检查 raw dot 后称作 source-risk safety。其严格保证也最多条件于 guard batch，不是目标风险保证。[PCGrad 原论文](https://arxiv.org/abs/2001.06782)讨论的是梯度干扰处理，不能提供本项目 target fault semantics 的监督信息。
