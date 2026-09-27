# CRA-v2 交付层实现与验证回执

日期：2026-09-25（Asia/Shanghai）  
范围：`D:\3800` 的 CRA-v2 推理、AWGN 开发集评估、完整 campaign 审计、交付构建和离线验包。  
边界：未修改 v2 trainer、runner 或 target-dev evaluator；未启动训练；未索引、解码或评分 target-final 标签。

## 结论

前置审计中“v2 没有专用预测器、AWGN 评估器、完整 campaign 审计器和打包/验包工具”的交付阻塞已由下列实现关闭。工具接受 `channel_robust_v2a_u_fixedk_campaign_v1` 完整结果，实际 campaign 路径由参数传入，不硬编码结果目录。

当前权威 campaign 是 `experiments/20260925_channel_robust_v2a_screen_v5`，plan SHA-256 为 `8bf146d8412b4aaef104e2ae39769a42494597880090d1d36b072e1b85f6fac9`。本回执生成时训练状态为 4/4 anchors、7/32 branches，target-dev 尚未开放，target-final=false。因此交付层已经可用，但最终 ZIP 必须等该 campaign 完成、封存、开发集评分和报告后生成；本回执不把进行中的训练称为模型交付完成。

## 实现

|能力|文件|关键约束|
|---|---|---|
|v2 无标签预测|`delivery_tools/predict_channel_robust_v2.py`|同一模型结构；三通道和各单通道使用独立 checkpoint；校验任务、通道、采样率、模型代码、固定 epoch、校准和 SHA；不打开数据集标签|
|v2 AWGN|`protocol_next/evaluate_channel_robust_v2_noise.py`|仅在因果 screen gate 通过后接受 candidate arm；clean + 20/10/0 dB × 3 seeds；先固定全部预测，再解码 target-dev；不打开 private origin metadata，不索引 target-final|
|完整 campaign 审计|`audit/verify_completed_channel_robust_v2_campaign.py`|重建 4 anchors、32 branches、16 对 replay、32 个 dev 结果、因果/历史比较和 gate；检查模型配置、校准、优化/对齐步数、代码/数据/参考哈希|
|交付构建|`delivery_tools/build_channel_robust_v2_delivery.py`|独立审计通过后才复制；Windows/WSL 路径占位；可选纳入注册 AWGN 结果；生成精确 manifest、SHA256SUMS、ZIP CRC 和 ZIP sidecar|
|离线验包|`delivery_tools/verify_channel_robust_v2_delivery.py`|不需要训练数据或私有标签；校验精确成员、全部 SHA、36 个模型产物、32 个独立目标视图 checkpoint、开发分数、v4 参考和可选 AWGN 收据|
|文档/权利边界|`delivery_tools/V2_DELIVERY_README.md`、`delivery_tools/V2_THIRD_PARTY_AND_RIGHTS.md`|运行命令、证据边界、无通用权重声明、数据不入包、v2 不误用旧 CRA-v1 DAN notice|

交付注册表明确写入：`shared_model_structure=true`，`one common architecture; separate training and checkpoint per target view`，`universal_weight_model_delivered=false`。这符合“先共用一套模型结构和代码，针对每个目标通道分别训练、分别评估；通用权重留作后续扩展”的决定。

## target-final 隔离

预测器不接受标签路径。campaign 审计器只读取冻结 checkpoint、训练日志、target-dev 结果 JSON 和公开元数据哈希。

AWGN 评估器不再调用会解析全部 private origin rows 的旧 `_preflight`。它通过公开 manifest 推导目标标签银行，只把整个标签文件作为不透明字节做 SHA 校验；NPY 数据访问只索引 `target_test`，并用公开的 `target_final_test` 索引集合检查不相交。输出收据固定包含：

- `private_origin_metadata_opened=false`
- `target_final_indices_accessed=false`
- `final_signal_or_label_rows_decoded=false`
- `target_final_evaluated=false`

构建器和离线验包器都会再次拒绝缺少上述收据的 AWGN 结果。

## 验证

最终联合命令覆盖：

- v2 predictor 合成 checkpoint、错误 SHA/任务/通道/采样率、包内冻结代码布局；
- AWGN 确定性、物理通道映射、SNR、标准化、candidate/gate、dev-only label preflight；
- 合成完整 campaign 的独立审计、篡改拒绝、构建、离线验包、可选 AWGN 注册、ZIP CRC/SHA、包内 noise campaign 布局；
- 既有 v2 模型和 runner 静态回归。

结果：`39 passed in 8.10s`。Python 编译检查通过；新 predictor 对 screen_v5 已完成的真实 `WP-S0/all3/common_anchor_control` checkpoint 通过严格载入，checkpoint SHA-256 为 `b73289b46127e5152850fbb4df0c502abfdd7afa7230eea0968cbfa50fea101c`。该检查没有运行 target-dev/final 评分。

## 核心文件 SHA-256

```text
a698863c0f4262566cbf585487d32592534c8cb7faeea837adcf1a4051e3d2c6  delivery_tools/predict_channel_robust_v2.py
78c43395b8970f33cdf8a5432aeb4430e0cf330a5c9a390ee20bf70eb0dd7914  protocol_next/evaluate_channel_robust_v2_noise.py
843dbf49a25e87fe7f9ee085310e39a77aba8f0858ba8cb9da965257ac474f80  audit/verify_completed_channel_robust_v2_campaign.py
4e6d02732dd8e71f0ded6c691b5dc54a10901c748ad20939a83acf8155ffb825  delivery_tools/build_channel_robust_v2_delivery.py
bee17c4e02f2e27edc41bb0671c23e71a7ee169d89e273cab87e6f3b834bb7f7  delivery_tools/verify_channel_robust_v2_delivery.py
9be80c85adde602e2be03ef14d7426544f7ed6442be0e8b09d8a1cd11a7687fa  delivery_tools/V2_DELIVERY_README.md
5b4effe5c0fd8cf81acc005814fe212d52aa9378e0840e4fa89382b0bd73ae6e  delivery_tools/V2_THIRD_PARTY_AND_RIGHTS.md
```

测试辅助文件同样进入工作区测试，不进入默认交付 ZIP；ZIP 内只复制工具、冻结 code、运行产物、审计收据、文档和配置。

## 最终构建门

最终构建必须满足：campaign `fit_state/dev_state/pipeline_state=completed`；4/32/32 数量一致；pre-dev fit audit 和 seal 完整；因果 gate 从实际 target-dev 结果独立重建；所有 checkpoint/score/code/data/reference SHA 一致。若 gate 未通过，工具仍能交付完整系统评估和负结果，但 `promotion_status` 固定为 `not_promoted_beyond_completed_system_evaluation`，不得包装成模型优越性结论。AWGN 只在 gate 通过后执行和入包。
