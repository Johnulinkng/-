# v2B 针对性一手文献调研与启动决策

日期：2026-09-25  
范围：严格无监督域适应（源域有标签、目标训练域无标签）；不使用目标域标签或窗口对应关系选模。本文只引用原论文、作者代码或官方论文页。

## 1. 当前证据与问题定义

`screen_v5` 不是“对齐不足”的正结果：16 个 joint 对比中仅 7 个提升，平均 Accuracy 为 -0.34 个百分点、平均 Macro-F1 为 -0.37 个百分点，双跨 Macro-F1 为 -1.98 个百分点；PG-D1/slot1 与 WP-D1/slot2 还出现新的零召回类别。已有路线已覆盖或明显失败的机制包括边缘 MMD（含 fixed equal-B U-stat）、DANN、WDGRL、随机通道屏蔽、受限注意力和源梯度保护。下一版不应继续更换另一种全局距离或叠加更多相似正则。

最贴合现状的解释是：跨工况和弱通道下，目标样本的类别语义与源域决策区没有被可靠保持；全局对齐可以减小域距离，同时把某些类别推入错误决策区。v2B 应先处理类别混淆/决策边界，并把无标签选模和 source-only 回退设为强制门控。

## 2. 推荐方法

### 2.1 第一优先：Minimum Class Confusion（MCC）

目标批次 logits 为 \(Z\in\mathbb{R}^{B\times K}\)，温度缩放概率为

\[
\hat Y_{ij}=\frac{\exp(Z_{ij}/T)}{\sum_{j'}\exp(Z_{ij'}/T)}.
\]

令 \(H_i=-\sum_j\hat Y_{ij}\log\hat Y_{ij}\)，样本权重

\[
W_{ii}=\frac{B[1+\exp(-H_i)]}{\sum_{i'}[1+\exp(-H_{i'})]}.
\]

类别相关矩阵 \(C=\hat Y^\top W\hat Y\)，按行归一化得到 \(\tilde C\)，损失为

\[
L_{MCC}=\frac1K\sum_j\sum_{j'\ne j}\tilde C_{jj'};
\qquad L=L_{CE}^{src}+\mu L_{MCC}^{tgt}.
\]

**为什么适合：** 它直接针对 screen_v5 暴露的类别混淆与零召回，不显式对齐源/目标边缘分布，不需要硬伪标签，也不需要域判别器；实现只需在目标 logits 上增加一个约二十行的损失函数，因此最适合作为低风险首个诊断分支。

**最小实现：** 保持当前编码器、融合器、分类器、数据和训练预算不变，仅增加 `mcc_loss(target_logits, T)`；先固定论文常用温度并只比较一个预注册的 \(\mu\)，不同时调多个超参数。

**主要风险：** 如果目标预测已经稳定地错到另一个类别，MCC 仍可能把错误预测变得更尖锐；小批次中的类别相关矩阵也可能不稳定。因此必须同时监控预测类别占用率、每类源验证召回和 source-only 锚点，不能因无标签损失下降就宣称有效。

来源：[ECCV 2020 原论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)；[作者代码](https://github.com/thuml/Versatile-Domain-Adaptation)

### 2.2 第二优先、独立分支：Maximum Classifier Discrepancy（MCD）

两个分类头的目标概率差异为

\[
d(p_1,p_2)=\frac1K\sum_{k=1}^K|p_{1k}-p_{2k}|.
\]

每轮分三步：

1. \(\min_{G,F_1,F_2}L_s\)，保证两个头都能正确分类源域；
2. 固定 \(G\)，\(\min_{F_1,F_2}L_s-\lambda\mathbb E_t[d(p_1,p_2)]\)，让两个头暴露源支持集之外的目标样本；
3. 固定两个头，\(\min_G\mathbb E_t[d(p_1,p_2)]\)，把目标特征拉回源类别支持区域。

**为什么适合：** MCD 使用任务分类边界，不是 DANN/WDGRL 的域判别器，也不是 MMD 的边缘匹配。当前失败表现恰好是“域距离可能下降但类别被推错”，所以决策边界机制比再换一个全局距离更有针对性。

**最小实现：** 在现有共享特征后复制一个轻量分类头；严格采用 A/B/C 三阶段交替更新，第一轮只用论文的 L1 discrepancy 和固定重复次数，不与 MCC 同时叠加。

**主要风险：** 优化复杂度和方差明显高于 MCC；若源域两个头本身不能形成互补边界，差异信号会很弱；若某弱通道存在真正的类别条件错配，MCD 也可能把目标拉向错误的源支持集。因此它应是 MCC 的平行对照，不是首轮组合模块。

来源：[CVPR 2018 官方论文页](https://openaccess.thecvf.com/content_cvpr_2018/html/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.html)；[作者代码](https://github.com/mil-tokyo/MCD_DA)

### 2.3 第三优先：SENTRY 式委员会一致性选择性熵

对目标样本 \(x\) 生成 \(m\) 个已验证为“故障语义保持”的增强 \(a_i(x)\)。若增强预测与原样本预测的多数一致，则最小化该样本熵；若多数不一致，则最大化熵，避免对不可靠预测做自训练。可写成最小化

\[
L_{sel}(x)=\mathbf 1_{cons}H[p(a_c(x))]-\mathbf 1_{incons}H[p(a_u(x))].
\]

**为什么适合：** 置信度高不等于正确，尤其是当前弱通道已经出现零召回。SENTRY 先用多视图一致性判断预测是否可靠，再决定让预测更确定还是更保守，能减少直接熵最小化或硬伪标签的误差累积。

**最小实现：** 只实现“选择性熵”消融，不先启用论文中的伪类别平衡；委员会大小固定为 3。振动信号第一轮只允许经人工确认的圆周/时间平移、轻微幅值缩放和低强度 AWGN。若未完整复现论文的类别平衡，应在论文中称为 `SENTRY-inspired selective entropy`，不能写成完整 SENTRY。

**主要风险：** 增强若改变故障语义，一致性判断本身会失真；所有视图可能一致地预测错误；噪声增强强度也不能借目标标签调参。因此该方法只在 MCC/MCD 至少一个通过小筛选后再加，不应首轮同时启动。

来源：[ICCV 2021 原论文](https://arxiv.org/abs/2012.11460)；[作者代码](https://github.com/virajprabhu/SENTRY)。时序增强的可用先例来自 [CLUDA 原论文](https://arxiv.org/abs/2206.06243) 与 [作者代码](https://github.com/oezyurty/CLUDA)；这里只借其时间序列增强证据，不建议照搬 CLUDA 的完整 DANN/多损失结构。

### 2.4 强制基础设施：DEV + SND 无标签选模与 source-only 回退

**DEV 风险估计。** 在候选模型特征空间训练源/目标二分类器 \(M\)（源为 1、目标为 0），源验证样本的密度比估计为

\[
w_f(x)=\frac{n_s}{n_t}\frac{1-M(f(x))}{M(f(x))}.
\]

令源验证损失为 \(\ell_i\)，则

\[
R_{DEV}=\operatorname{mean}(w_i\ell_i)+\eta[\operatorname{mean}(w_i)-1],
\quad
\eta=-\frac{\operatorname{Cov}(w_i\ell_i,w_i)}{\operatorname{Var}(w_i)}.
\]

**SND 结构评分。** 对归一化目标表示 \(z_i\)，计算 \(s_{ij}=z_i^\top z_j\ (i\ne j)\)，

\[
P_{ij}=\frac{\exp(s_{ij}/\tau)}{\sum_{j'\ne i}\exp(s_{ij'}/\tau)},
\qquad
SND=-\frac1N\sum_i\sum_{j\ne i}P_{ij}\log P_{ij}.
\]

论文用更高的 SND 表示更稠密的目标软邻域。

**建议的保守回退规则：** 每个任务先保存 source-only 锚点。适配模型只有同时满足以下预注册条件才可替代锚点：

1. 源验证 Macro-F1 非劣，且没有新增源域零召回；
2. DEV 风险低于锚点，并用样本 bootstrap 给出差值区间；
3. SND 不恶化，且目标预测类别占用没有塌缩或近空类别；
4. 不读取目标验证/测试标签做超参和 epoch 选择。

任一条件不满足即回退 source-only。DEV 和 SND 是互补代理，不是目标准确率的数学保证：DEV 依赖特征空间协变量漂移/支持覆盖，条件漂移时可能失真；SND 可能偏爱类别置换或整体塌缩。因此交付中应称为“无标签保守门控”，不能称为“保证无负迁移”。

来源：[DEV，ICML 2019 原论文](https://proceedings.mlr.press/v97/you19a.html)；[作者代码](https://github.com/thuml/Deep-Embedded-Validation)；[SND，ICCV 2021 原论文](https://arxiv.org/abs/2108.10860)；[作者代码](https://github.com/VisionLearningGroup/SND)

### 2.5 3→1 后续分支：EmbraceNet 可用通道融合

每个通道经共享编码器和 docking 层得到 \(d^{(k)}\in\mathbb R^c\)。每个特征坐标独立采样

\[
r_i\sim\operatorname{Multinomial}(1,p),
\qquad e_i=\sum_k r_i^{(k)}d_i^{(k)}.
\]

给定通道可用向量 \(u_k\in\{0,1\}\)，使用

\[
\hat p_k=\frac{u_kp_k}{\sum_j u_jp_j}.
\]

当目标只有一个通道时，该通道的 \(\hat p=1\)，融合退化为该通道表示本身，不需要补全缺失通道。

**为什么适合：** 它直接支持同一套结构处理 3 通道源域和任一单通道目标域，也不同于 screen_v5 中接近均匀的确定性注意力。可用通道掩码是结构输入，不需要目标标签。

**最小实现：** 保持共享通道编码器；在分类头前加入 docking + embracement。第一轮仅比较 `mean fusion` 与 `EmbraceNet fusion`，不再叠加额外随机通道 dropout；训练用固定 \(p\)，推理固定随机种子或多次采样求平均。

**主要风险：** 坐标级随机选择增加方差并可能损失跨通道联合关系；完整 EmbraceNet 的 modality dropout 与本项目已经失败的随机通道屏蔽存在部分重叠。因此它不是当前负迁移的第一修复项，只在类别机制通过后用于 3→1 专项消融。

来源：[EmbraceNet 原论文](https://arxiv.org/abs/1904.09078)；[作者代码](https://github.com/idearibosome/embracenet)

## 3. 已筛除为本轮首选的路线

- **CDAN：** 虽是类别条件对齐，但仍依赖目标预测参与域对抗；当前目标语义已出现系统性错误，可能强化错配，且 DANN 已负向，不作为 v2B 第一轮。
- **完整 CLUDA：** 时间序列针对性强，但包含 DANN、通道 dropout、Gaussian augmentation、队列对比学习和多项权重，和已失败机制重叠且一次改变过多因素；仅借其时序增强设计证据。
- **RAINCOAT：** 时间/频率表示有价值，但核心仍含全局 Sinkhorn 对齐和后处理修正；在 marginal alignment 已连续负向后，优先级低于 MCC/MCD。其时频编码器可在以后作为“仅表示变化”的独立消融。
- **继续调 MMD/WDGRL 或直接硬伪标签：** 与当前负证据重复，且不能解释新零召回，不建议。

## 4. 是否在交付前启动 v2B

**建议启动，但仅启动一个边界清晰、可随时回退的小型 MCC 诊断；不建议在交付前直接启动完整 v2B 大矩阵。**

建议首轮固定为：

- 任务：只选最难且已出现类级退化的 WP-D1 与 PG-D1；
- 视图：每个任务的 `all3` 加该任务出现新零召回的弱单通道（WP-D1/slot2、PG-D1/slot1）；
- 方法：source-only 锚点与 MCC-only；不叠加 MMD、DANN、attention、mask 或 SENTRY；
- 重复：3 个预注册种子，共 12 个 MCC 训练（锚点若已有同协议 checkpoint 可冻结复用）；
- 选模：只用源验证 + DEV/SND/类别占用门控；目标标签在超参和 checkpoint 冻结前保持封存；
- 通过条件：四个任务-视图组合中至少 3 个达到 source-only 非劣，两个零召回视图均不得再产生目标预测近空类别；打开目标标签后的 Macro-F1 汇总才用于一次性审计，不返调。

如果 MCC 小筛选失败，停止 v2B 并保留 source-only/现有稳定模型作为交付锚点；失败本身应写入论文边界。若通过，再扩到完整任务矩阵；MCD 作为独立第二分支，SENTRY 和 EmbraceNet 分别留给鲁棒性与 3→1 后续消融。

这一定义能在交付前获得一个真正回答“类别混淆是否是负迁移主因”的证据，同时不破坏现有交付冻结状态。它仍不是论文效果提升的证明；只有完整任务、3–5 个种子、Macro-F1/每类召回/置信区间和公开数据确认通过后，才能形成论文主结果。
