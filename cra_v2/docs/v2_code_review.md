# CRA-v2A 代码独立严格审查

日期：2026-09-25  
审查对象：CRA-v2A-U-fixedK 当前冻结前实现与已 prepare 的 campaign snapshot。  
结论：**BLOCKER = 0，HIGH = 1，MEDIUM = 3，LOW = 2。** 当前实现满足启动 4 个共同 anchor 与 32 个配对分支的核心方法合同；本报告不构成训练完成、target-dev 过门或论文有效性证明。

## 审查边界

- 仅做 Windows CPU 静态检查、合成张量测试及冻结 JSON/代码哈希检查。
- 未启动 WSL、GPU 或训练进程；未打开 target-dev/final 信号或标签数组。
- 未修改 frozen v4，也未修改 v2 主模型、损失、trainer、runner 或 evaluator；新增测试仅位于 `audit/tests/test_channel_robust_v2_code_review.py`。
- 方法判据以 `audit/channel_robust_v2_design_decision_20260925.md` 为主，并核对 `audit/channel_robust_alignment_v2_predev_20260925.md` 的执行细化。

## Blocker

**无。** 审查过程中曾发现并即时上报四项 blocker，当前冻结 snapshot 均已修复：

1. evaluator 原先在读取 target-dev 后才索引部分 provenance；现于 `protocol_next/evaluate_channel_robust_alignment_v2_dev.py:63-134` 在任何私有 evaluator 调用前验证 payload、pair identity、kernel、step 及 code identity。
2. runner 原先把扩展门施加于 frozen v4 control；现于 `run_channel_robust_v2_campaign.py:734-751` 用本轮 `common_anchor_control` 的 16 对结果计算 causal expansion gate，v4 只作不决定扩展的历史诊断。
3. frozen runner 原先会从 snapshot `ROOT` 错误重建 v4 路径；现于 `run_channel_robust_v2_campaign.py:698-726` 使用 Windows/WSL 双路径并验证相对项目根、各证据 SHA 及独立 v4 audit PASS。
4. report 原先直接信任 `dev_results.json`；现于 `run_channel_robust_v2_campaign.py:587-613,730-734` 从每臂 evaluator JSON 重建 registry，逐字段核对 arm、checkpoint SHA、result SHA、指标及 target-final=false 后才计算 gate。

## High

### H1. predev fit audit 尚未限制 run/checkpoint 路径，也未全面拒绝非有限日志值

证据：

- `run_channel_robust_v2_campaign.py:348-365` 和 `422-475` 直接使用 fit registry 中的绝对 `run`/`checkpoint` 路径，没有要求它们位于当前 campaign 的 `runs` 目录且名称等于冻结 run id。
- `read_jsonl`（`400-407`）使用标准 `json.loads`；Python 会接受 `NaN`/`Infinity`。`selected_anchor_record`（`410-419`）只检查 Accuracy/Loss 是 `int/float`，未检查 `math.isfinite`；branch 的其他记录字段也未递归检查有限性。
- 审计已经正确重算 60 轮 source-val 最优、验证 20 轮 endpoint、checkpoint 与 pair replay，但路径替换或对非关键日志字段的自洽改写仍可能进入 PASS receipt。

影响：正常由冻结 runner 生成的工件不受影响；该缺口影响的是抗本地状态替换的审计强度。若 fit registry 与外部同配置工件被共同替换，seal 不能证明工件来自本 campaign 目录。

建议：在 seal 前把每个路径 `resolve(strict=True)` 后约束在 `<campaign>/runs`，要求精确 run name；递归拒绝所有 JSONL 非有限数值，并核对 checkpoint/run_state/fit row 的 code identity、source channels、seed、task、dataset 和 optimizer/alignment step 总数。该加固应在首次 seal 前完成；无需改变算法或重训已不存在的工件。

## Medium

### M1. trainer CLI 可接受非预注册超参，固定合同主要由 runner 保证

`train_channel_robust_alignment_v2.py:97-151` 允许 batch size、lr、weight decay、encoder width、grid、max grad norm 和 strict CUDA 等参数变化；checkpoint metadata（`348-384`）没有完整记录全部 runtime/optimizer 参数。正式 runner 在 `run_channel_robust_v2_campaign.py:318-345` 固定传入设计值，plan 和 snapshot 也锁定它们，因此正式 campaign 成立；单独调用 trainer 产生的 checkpoint 不应被称为本冻结方案。建议 evaluator/auditor 从 plan 传入预期训练合同并逐字段核对。

### M2. 分支 CLI seed metadata 可与实际恢复的 generator state 不同

`train_channel_robust_alignment_v2.py:400-407` 先按 CLI seed 建 generators，`223-238,725` 随后用 anchor 的完整状态覆盖它们；`348-384` 仍记录 CLI seed。正式 runner 对 anchor/control/adapt 传相同固定 seed，因此当前 campaign 没有偏差；独立调用时 metadata 可能不代表实际随机流。建议 `load_anchor` 要求 CLI mask/source-draw/target-draw seed 等于 anchor metadata，或把字段命名为 requested seed，并以 state hash 为真实身份。

### M3. fit/dev 仍存在进程中断后的不可恢复提交窗口

runner 先写 active state，再由 trainer/evaluator 排他创建 run/output；若子过程完成后、registry 落盘前中断，重试会因既有 run 或 dev JSON 而失败（`run_channel_robust_v2_campaign.py:464-515,616-643`）。这不会静默改变结果，但需要书面恢复流程。建议验证既有工件全部身份后幂等补写 registry，禁止删除后无收据重跑。

## Low

### L1. `formal_gate_baseline` 命名与实际历史诊断角色不一致

`run_channel_robust_v2_campaign.py:198` 在 v4 reference 中仍写 `formal_gate_baseline=true`，而 plan 的 screen gate 和 report 已明确 v4 不决定扩展。建议改为 `historical_threshold_baseline`，避免后续工具误读；当前 gate 计算路径不使用该布尔值。

### L2. anchor 阶段会构造 target memmap/header，但不读取 target waveform

trainer anchor 调用通用 `load_public_task(..., role="train")`（`train_channel_robust_alignment_v2.py:387-397,504-513`）；public loader 会打开 target `.npy` 的 memmap/header并构造 `target_train` view，但 anchor 不迭代该 view，也不计算 target 统计。它满足设计决定中的“target 波形、统计、标签不参与第一阶段”，但 predev 文档“anchor 不构造 target 数据”措辞更强。建议文档改为“不访问 target 样本/统计”，或增加 source-only loader role。

## 已确认成立的核心合同

- **共同 anchor 与 20 轮配对**：正式矩阵是每 task 一个 60-epoch anchor、每 task-mode 两个 20-epoch 分支。`run_channel_robust_v2_campaign.py:382-397,422-475` 将每个 branch 的顶层 anchor SHA、step-0 identity 内 anchor SHA及 task 唯一实际 anchor SHA 三方绑定，并要求 pair 的 anchor/start/source-index/source-mask digest 完全相同。
- **60 轮选择与固定 endpoint**：`train_channel_robust_alignment_v2.py:504-623` 跑满 60 轮，按 source-val Accuracy、低 CE、早 epoch选 anchor并保存该时点 model/optimizer/scheduler/RNG；branch 于 `694-925` 固定跑 20 轮并只保存 epoch 19。predev audit 从日志独立重算两项事实。
- **完整 RNG 与 optimizer state**：`capture_rng/restore_rng`（`204-238`）覆盖 Python、NumPy、Torch CPU、全部 CUDA 及 source-loader/target-loader/mask/source-draw/target-draw generators；anchor 保存 optimizer/scheduler state 和哈希，分支加载后记录 step-0 identity。pair gate 比较完整 identity。
- **fixed kernel**：`freeze_source_anchor_kernels`（`411-455`）只用固定 seed 抽取 source-train anchor features，冻结五个 bandwidth、weights、indices SHA和feature SHA；`validate_anchor_payload`（`458-501`）拒绝 schema、bandwidth 或 replay hash 篡改。target 输入不参与 kernel 选择。
- **signed equal-B U-statistic**：`balanced_unbiased_mmd.py:91-170` 要求 source/target shape 完全相同，删除 XX/YY 对角，cross term 使用全部 B² 项，不含 clamp、abs、square 或 batch bandwidth 重估，负值和梯度保留。
- **物理槽与独立 RNG**：`BalancedPhysicalSlotCycle`（`balanced_unbiased_mmd.py:16-79`）每样本取一个槽，连续循环保证任意累计前缀的槽计数差不超过 1；singleton 保留真实 physical slot。source/target 使用独立 generator。
- **梯度门**：`agreement_backward`（`train_channel_robust_alignment_v2.py:659-691`）只在 encoder 全局 dot 非负时向所有非-classifier 参数注入 `0.02*g_alignment`；冲突时 optimizer input 等于 source gradient，classifier 始终只有 source gradient。
- **目标标签边界**：`HiddenTargetDataset` 强制 target-train label 为 `-1`，每个 adaptation batch再次检查（`798-799`）；trainer CLI 没有 dev/final 参数。evaluator 独立读取 target-dev，并在读取前完成 checkpoint/provenance验证；没有 target-final入口。
- **代码和评测先后**：checkpoint `code_identity` 覆盖 trainer、loader、v2/v1 model、MMD、spectral实现及指标实现；plan 另锁 runner、evaluator、launcher、测试和设计文档。全部 4+32 fit 工件及 predev audit PASS 被 SHA seal 后，dev 才能从 blocked 状态转为 prepared。

## 已 prepare campaign 核对

路径：`experiments/20260925_channel_robust_v2a_screen_v1`。

- `plan.json` 声明与实际 SHA-256 均为 `44309a5c0d676815c84cdbe421c841ee752c4569b9bd9a7f76078ad92ae2370e`。
- 矩阵：4 anchors、16 `common_anchor_control`、16 `channel_robust_alignment_v2a`；共 36 optimizer runs。
- causal control 为 `common_anchor_control`，`decides_expansion=true`；v4 historical comparison 不决定扩展。
- v4 独立完成态审计为 `channel_robust_completed_campaign_audit_v1 / PASS`，receipt SHA 为 `1961ad2e3a082ddca6fc27285f88b8aad9bccb9163c38876eda2f1aaaca56fb1`。
- 冻结 snapshot 中 18 个 code/doc/test 文件全部与 plan 声明和 live 文件哈希一致。
- 当前状态为 fit `prepared 0/4 anchors, 0/32 arms`；dev `blocked_until_seal 0/32`；不存在 predev seal；pipeline 明确 target-dev=false、target-final=false。

## 测试

- 最新 live 代码：正式 core/static 与独立审查测试合计 `24 passed`，Windows CPU，4.24 秒。
- 正式 campaign 的 frozen snapshot：`17 passed`，Windows CPU，4.00 秒。
- Python 五个核心文件 AST 解析通过；PowerShell launcher AST 解析通过。
- 独立测试覆盖 literal signed U-stat、负值不截断、equal-row拒绝、物理槽映射/平衡/replay、全局 RNG 隔离、冲突/非冲突梯度精确式、classifier隔离、评分前 provenance 拒绝和 malformed kernel anchor拒绝。

## 论文与运行边界

当前结论只支持“实现与冻结协议通过训练前代码门”。训练结果、target-dev gate、稳定性或性能结论均尚不存在。即使 seed42 causal gate 通过，也只支持在历史已暴露开发分区上的固定配方开发证据；在 seeds 43/44、预注册消融、外部任务和独立记录验证完成前，不能声称稳定改进、普适负迁移安全、SOTA 或独立泛化。
