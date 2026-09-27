# CRA-v2 独立方法决策审查

日期：2026-09-25  
输入：`channel_robust_v1_failure_diagnosis_20260925.md`、`channel_robust_v2_literature_design_20260925.md` 及其中引用的一手论文。  
状态：**只读方法决策；未修改 frozen v4/训练代码，未启动训练，未读取 target-final。** 已有 target-dev 属于开发历史，不能把下一批称为全新盲测。

## 1. 决策

本轮最小可运行主方案定为 **v2A-U-fixedK**：

1. 共同完成 60 epoch source-only 预训练，只按 source-val Accuracy、CE、较早 epoch 选择一个共同锚点；
2. matched control 与 v2A 从**完全相同的模型、optimizer、scheduler 和 RNG state** 分叉，各自再运行固定 20 epoch；二者 source minibatch 顺序与 source loss 相同；
3. v2A 使用 fused 与 channel 两层 equal-B、去对角、signed U-statistic MK-MMD；RBF bandwidth 在共同 source-only 锚点上冻结，第二阶段不随当前 source/target batch 重估；
4. `lambda=0.02` 在 20 个适配 epoch 内固定，不做 sweep；若 source/alignment encoder 梯度点积 `<0`，该 minibatch 的 alignment 梯度整体置零；否则使用 `g_s + 0.02 g_a`；
5. 第二阶段只取第 20 epoch 的最终 checkpoint，不做 target 指标选模，也不在 20 个 epoch 内重新按 source-val 挑时点；
6. seed42 先完成原 32 条件臂筛查，沿用 v1 的同一冻结门槛。只有通过后才补 seed43/44 和估计器/模块消融。

保守备选是 **EQ-Lite-2stage**：使用同一 60+20 共同锚点合同、相同 `lambda=0.02` 和冲突关闭，但只保留 fused-level B-vs-B biased MK-MMD，完全删除 channel-MMD。它用于 v2A 的 fixed-kernel/U-statistic 预检失败时，不与 v2A 在看到 target-dev 后按数据集切换。

**EQ-Guard 不属于本轮必需实现。** DEV/IW-GAE 可以成为 v2A 通过后的无标签部署选择研究，但现在加入会同时改变 target-train 分割、checkpoint selection、validator 超参和回退逻辑，无法判断 v2 的变化来自训练机制还是选择器。更关键的是，当前 source-val 0/1 error 全为零，原始 DEV 对现有候选的 risk 均为零，不能排序；CE-DEV 属于修改版方法，必须先在 source-side pseudo-target episodes 上独立验证。IW-GAE 同样需要先冻结并验证 group/temperature/importance-weight 合同。

## 2. 三方案比较

|方案|解决的核心问题|本轮可归因性|主要风险|决策|
|---|---|---|---|---|
|v2A-U-fixedK|共同锚点消除训练时点失配；equal-B 修复 3→1 样本数不对称；低权重与冲突关闭降低长期反向干预|较高；仍是四项联合 recipe，只能先判断 recipe 是否值得扩展|signed U-statistic 有负值和 batch 方差；固定 kernel 可能对后期表示不敏感；冲突关闭可能使约 90% batch 不适配|**本轮主方案**|
|EQ-Lite-2stage|保留共同锚点和较弱 fused 对齐，移除已知不对称的 channel-MMD|最高，代码/统计失效模式最少|不能验证逐传感器对齐；若 fused marginal alignment 本身造成类别错配仍会失败|**预检失败时的统一备选**|
|EQ-Guard|equal-B channel alignment 加 Source Only/DEV/IW-GAE fail-closed 选择|适合作为后续“无标签部署选择”研究|原始 DEV 当前退化；IW-GAE 与 CE-DEV 未经本项目伪目标验证；新增 split 和选择器使本轮因果问题膨胀|**本轮不实现**|

文献上，Gretton 等给出了固定 kernel 下的无偏 MMD U-statistic；其有限样本实现允许负值。[JMLR 2012](https://www.jmlr.org/papers/v13/gretton12a.html)。DAN 说明深层多核 MMD 和线性无偏估计已有先例。[ICML 2015](https://proceedings.mlr.press/v37/long15.html)。DEV 与 IW-GAE 支持无目标标签选模，但都依赖可验证的估计条件，且 *Better Practices* 已显示 validator 即使把 Source Only 放入候选池也可能选择破坏性适配。[DEV](https://proceedings.mlr.press/v97/you19a.html)、[Better Practices](https://proceedings.mlr.press/v224/ericsson23a.html)、[IW-GAE](https://proceedings.mlr.press/v235/joo24a.html)。

## 3. 共同锚点是否公平

**共同 60 epoch 预训练是必要条件，但单独还不够。** 公平比较要求第二阶段也匹配训练机会：

\[
\theta_0=\operatorname{SelectSourceVal}\{\theta^{pre}_1,\ldots,\theta^{pre}_{60}\}.
\]

对每个 task/seed，`theta0` 只能有一个 SHA。若目标 all3/slot0/slot1/slot2 的 source 数据、模型和 source normalization 合同完全相同，可复用同一锚点；目标 train-only normalization 作为目标路径 metadata 在分叉后附加，不得导致四个不同的“共同源锚点”。

分叉时必须复制：

- model state；
- 被选中 epoch 当时的 optimizer state 与 scheduler state；
- 下一 source DataLoader shuffle state、mask RNG state；
- source calibration、class mapping、physical-slot mapping 和代码 SHA。

之后 control 也必须完成同样 20 epoch source update：

\[
g_c=g_s,\qquad
g_{v2A}=g_s+\mathbf 1[\langle g_s,g_a\rangle\ge0]\,0.02g_a.
\]

若 control 固定在 60-epoch 锚点而 v2A 再训练 20 epoch，则“对齐效果”与“额外 20 epoch source 更新”混杂，不公平。若 control 虽训练 20 epoch但重新按 source-val 选最佳、v2A 固定取末轮，也不公平。两支都固定取第二阶段末轮，才能把差异限定为 alignment 梯度。

每一对 arm 必须核对：第二阶段 step 0 的 model/optimizer hash 相同；20 epoch 内 source batch ids、mask 序列和 source-loss forward 输入相同；差异只允许出现在 target batch、MMD 和由其导致的后续模型状态。

## 4. equal-B signed U-statistic 的严格合同

对 fused features，源和目标本来各为 B 行。对 channel features，每个 source/target 样本按独立固定 RNG 从其可见物理槽中选一个：

\[
X=\{h^s_{i,c_i}\}_{i=1}^{B},\qquad
Y=\{h^t_{i,d_i}\}_{i=1}^{B}.
\]

`c_i,d_i` 按物理槽循环平衡；1000 batches 内同域可用槽暴露计数最大差不超过 1。相同 seed 必须 bitwise replay，且该 RNG 不改变初始化、loader 或 mask RNG。

对冻结的 multi-kernel (k=\sum_r\beta_rk_r)，使用：

\[
\widehat{\mathrm{MMD}}_u^2(X,Y)=
\frac{1}{B(B-1)}\sum_{i\ne j}k(x_i,x_j)
+\frac{1}{B(B-1)}\sum_{i\ne j}k(y_i,y_j)
-\frac{2}{B^2}\sum_{i,j}k(x_i,y_j).
\]

### 4.1 负值不是 bug

该有限样本 U-statistic **允许小于零**。实现必须保留 signed scalar 和 signed gradient：

- 不得 `relu`、`clamp_min(0)`、`abs`、平方或开方；
- checkpoint/log 同时记录 fused/channel 的负值比例、均值、标准差、分位数；
- 若 loss 为负但 `dot(g_s,g_a)>=0`，按预注册规则仍允许 alignment update；不得事后增加“负值关闭”规则；
- 非有限值立即 fail closed，不能替换成 0 后继续。

负值表示估计噪声下的 signed statistic，不表示“分布距离为负”。因此论文应写“unbiased U-statistic estimator”，不能把每个 batch scalar称为非负距离。

### 4.2 何时才可诚实称 unbiased

Gretton 的无偏结论以 kernel 固定为前提。若 RBF bandwidth 用当前同一 source/target batch 的成对距离重估，则 kernel 本身依赖这批样本，严格的 U-statistic 无偏性表述不再直接成立。故本轮必须二选一：

1. **推荐**：在共同 source-only 锚点上，用固定 seed 抽取固定 source-train feature pool，冻结五个 bandwidth、kernel weights、pool indices 和 SHA；所有 target mode/control/candidate 共用。此时可称 fixed-kernel unbiased U-statistic。
2. 若继续使用当前 batch-adaptive bandwidth，则文档名称降级为 **equal-B diagonal-deleted signed MK-MMD estimator**，不得声称严格 unbiased。

不得用 target-train 或 target-dev 表现挑 bandwidth。若 source-only bandwidth pool 的中位距离为零、非有限或五核退化，v2A 预检失败，整批切换 EQ-Lite-2stage，而不是按 task/slot 混用 estimator。

## 5. v2A-U-fixedK 的完整最小合同

### 第一阶段：共同 source-only 预训练

\[
L_s=CE(p_f,y)+0.5CE(p_m,y)+0.1KL(p_m\|\operatorname{sg}(p_f))+0.01L_{bal}.
\]

- 保持 v1 的 2048 点 Hann、128 维 log-power、共享 128→64→64 encoder、均值锚定 attention、七种非空 source mask；
- Adam lr `0.001`、weight decay `0.0005`、batch `32`、clip norm `5`、seed `42`；
- 固定跑满 60 epoch；只按 source-val Accuracy、CE、较早 epoch选共同锚点；保存该 epoch 的完整 optimizer/RNG state；
- target 波形、统计和标签均不参与第一阶段。

### 第二阶段：固定 20 epoch 配对分叉

\[
L_a=\widehat{\mathrm{MMD}}_{u,fused}^2
+0.5\widehat{\mathrm{MMD}}_{u,channel}^2.
\]

- `lambda=0.02` 为常数；它来自 v1 `0.1` 上限的事前五倍缩小，不按 task/slot 调；
- 用 encoder 参数上的全局 dot 判断一次冲突；冲突时整个 alignment gradient 置零，非冲突才注入；classifier/attention balance/source losses 不接收 target label；
- matched control 和 v2A 使用相同 20 epoch source update 与最终 checkpoint；
- target-train 标签继续为 `-1`，只允许 train-only waveform/statistics 和 MMD；
- 不做 checkpoint selection，不读 target-dev/final；先完成全部 arm、hash 和 seal，再由独立评分器一次性读取 target-dev。

`lambda=0.02` 与“冲突即关闭”是基于 v1 约 90% 冲突和对齐标量约为同期 source CE 15–20 倍的风险缩减，不是理论最优值。若大多数 batch 被关闭，结论是对齐目标与源判别目标不兼容，不得降低冲突阈值以追求更多 update。

## 6. 训练前硬门

以下任一失败，不启动 32-arm screen；直接修测试或统一降级 EQ-Lite-2stage：

1. fixed-kernel identity：五个 bandwidth/weights、source feature pool indices、anchor SHA 可重放；任何 target 输入都不能改变它们；
2. equal-B identity：all3/三个 slot 的 fused/channel MMD 两侧 shape 均为 `[32,D]`；记录每行样本和 physical slot；
3. null test：同分布至少 500 seeds 时 signed U-statistic 均值的 95% 区间覆盖 0，3→3/3→1 均值差不超过 pooled SE 的 2 倍；负值必须实际出现以证明未 clamp；
4. shift test：固定 mean shift `delta={0,.25,.5,1}` 时重复均值单调不减；
5. gradient test：冲突 batch 的实际 optimizer input 与 pure-source gradient bitwise 相同；非冲突 batch 等于 `g_s+0.02g_a`；classifier 无 alignment gradient；
6. common-anchor replay：每对 arm 的 step-0 model/optimizer/RNG hash 相同，source batch/mask 序列相同；
7. information boundary：target label sentinel 为 `-1`，trainer 不接受 target-dev/final path；错误 slot/split/hash fail closed。

## 7. 本轮最小矩阵与同一冻结门槛

最小 screen 沿用失败诊断中已经事前提出的 32 个第二阶段 arm：

`WP-S0, WP-D1, PG-S1, PG-D1 × all3, slot0, slot1, slot2 × matched_control, v2A-U-fixedK × seed42`。

共同 source-only anchor 理论上每个 task/seed 只需训练一次，共 4 个；四个 target mode 从同一 task anchor 分叉。若现有代码为简化而重复预训练，四份锚点必须 bitwise/hash 相同，否则 campaign fail closed。32 是第二阶段分支数，不把共同 anchor 重复虚报为独立模型。

继续门槛原样保留，不因 v1 失败或 v2 更保守而降低：

1. 16 条件中至少 10 条 Accuracy 与 Macro-F1 同时提高；
2. 16 条件等权 Accuracy 至少 `+1 pp`，Macro-F1 至少 `+2 pp`；
3. 双跨 8 条件等权 Macro-F1 至少 `+2 pp`；
4. 每个任务的三个单目标槽中至少两个槽双指标提高；
5. control 中最弱的 3→1 条件 Macro-F1 提高；
6. 不新增零召回类别。

任一失败则 v2A 不升级，不补 seed43/44，不切换成“只在 PG 使用”。保留完整失败表，并将主张降级为 estimator/负迁移边界研究。通过后才运行 seed43/44、equal-B biased 对照、channel-MMD removal、WIDAN-reference、严格 Source Only 和外部数据。

## 8. 本轮禁止混入

- DEV、CE-DEV、IW-GAE、RankMe、target entropy 或任何 target-based checkpoint/model selection；
- order tracking、名义阶次 v2B、新频谱网格、wavelet 分支或新 backbone；
- pseudo-label、class-conditional MMD、LMMD/JMMD、adversarial discriminator；
- attention 结构、mask 概率、七子集分布、一致性权重、target z-score 范围的改变；
- `lambda` sweep、按数据集/slot 设置不同 lambda、冲突阈值或 kernel；
- unbiased/biased estimator 在同一主 arm 中自动切换，或负 MMD 时临时 clamp/关闭；
- target-dev early stopping、看分数后补训、只保留有利 task/slot；
- 任何 target-final 访问。

这些都可能是后续合理实验，但与 v2A 同批加入会破坏“共同锚点+等样本+低强度冲突关闭是否足以修复 v1”的问题。

## 9. DEV/IW-GAE 是否必须

**本轮不必须，而且不应实现为主方案依赖。** 理由不是它们无价值，而是当前尚不具备可靠接入条件：

- 原始 DEV 使用逐样本 0/1 error；当前所有选中 checkpoint 的 source-val accuracy 都为 100%，风险恒为 0；
- CE-DEV 改变了原论文的 loss，需要 source-side pseudo-target episode 预验证、ESS/权重/域分类器可靠性门；
- IW-GAE 需要新的 unlabeled target validation split、group 数、temperature 和 importance-weight optimization 合同；
- *Better Practices* 已表明无监督 validator 可能无法选回 Source Only，不能把“装了 validator”写成安全保证。

若 v2A 通过训练门，后续可把 strict Source Only、control、v2A checkpoints 全部先封存，再独立研究 CE-DEV/IW-GAE 是否能在不读 target labels 时做正确回退。该研究必须报告 selector regret、回退率和失败案例，与 v2A 训练贡献分开。

## 10. 论文创新边界

本轮若成功，可写的窄主张是：

> 面向有标签三传感器源域到无标签三/单传感器目标域，以及测点与工况双重偏移，构建共享 source anchor 的两阶段严格 UDA 协议；通过物理槽平衡的 equal-B fixed-kernel U-statistic 和 source-conflict hard gate，减少通道宽度引入的估计不对称，并在冻结任务矩阵中评估负迁移边界。

这项贡献是**问题约束、估计合同、配对实验和审计协议的组合**。不能声称首次提出 MMD、unbiased MMD、传感器 dropout、attention、冲突梯度处理、多传感器 UDA、3→1 或 order tracking。PCGrad 已提出冲突梯度投影；DAN 已使用无偏 MK-MMD；ModDrop 已研究整模态丢弃；OCGN 已在旋转机械速度变化中使用 tacholess order tracking；2025–2026 已有多传感器/跨测点域适应论文。

即使过门，也只能先写“在固定开发矩阵中相对 matched control 改善并值得多种子扩展”。seed42 screen 不支持稳定性、SOTA、普适负迁移安全或独立泛化主张。若 fixed-kernel U-statistic、共同锚点或冲突关闭任一消融不支持收益，应删除对应机制主张，而不是用完整 recipe 的均值替代因果证据。

