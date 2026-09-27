# CRA-v2A-U-fixedK 预开发协议

日期：2026-09-25  
状态：实现与预注册；尚未启动训练，尚未读取本批 target-dev，target-final 禁止访问。

## 1. 固定问题与矩阵

本批只检验一个统一方法 `v2A-U-fixedK`，不按数据集、任务或目标槽切换策略。任务为
`WP-S0 / WP-D1 / PG-S1 / PG-D1 × all3 / slot0 / slot1 / slot2 × common_anchor_control / v2A × seed42`，
共 32 个第二阶段 arm。每个 task 只训练一个 60 epoch source-only common anchor，共 4 个 anchor；
两支各从同一 anchor 完整复制 model、optimizer、scheduler 与全部 RNG state，再各训练固定 20 epoch。
因此共有 4 个 anchor optimizer run 和 32 个第二阶段 optimizer run，合计 36 个，不把 anchor 复用虚报成独立模型。

`common_anchor_control` 与 candidate 构成严格配对的因果比较，并单独决定是否扩展 seed43/44。frozen v4 中 16 个
`matched_control` 是历史强基线，只报告是否超过既有系统，不能替代配对因果门。prepare 只读取 v4
plan/fit/predev seal，并对已存在的 dev/summary/report 做字节 SHA 锁定，不解析分数。只有新 4+32 工件完成并封印、
且新 32 arm 评分完成后，report 才能读取已锁定的 v4 dev 结果。

## 2. 训练合同

第一阶段 source loss 固定为：

`CE(fused)+0.5 CE(masked)+0.1 KL(masked||stopgrad(fused))+0.01 attention_balance`。

60 epoch 跑满，只按 source-val Accuracy 更高、CE 更低、epoch 更早的顺序选择一个 anchor。该时点的
model/optimizer/scheduler、Python/NumPy/Torch/CUDA 与 source-loader/target-loader/mask/source-draw/target-draw
显式 generator 全部保存。anchor 使用独立 `source_only` loader role，只返回 source-train/source-val；不构造
target-train dataset、不迭代 target waveform sample。公共协议仍核对 target 文件的只读 header/hash metadata。

第二阶段两支都继续 20 epoch 相同 source update，并固定取 epoch 19。candidate 的唯一额外项为：

`L_a = MMD_u_fixedK(fused_s,fused_t) + 0.5 MMD_u_fixedK(channel_s,channel_t)`。

`lambda=0.02` 固定。五个 RBF bandwidth 为 common anchor 的固定 source-only feature pool 中位平方距离乘
`[0.25,0.5,1,2,4]`；target 不参与 kernel 决定。MMD 两侧严格 equal-B，删除同域对角，保留 signed
U-statistic，不使用 clamp/abs/ReLU 或 batch-adaptive bandwidth。channel 两域分别用独立 RNG 初始化一个物理槽
连续 cycle；每行取一个槽，任意累计前缀内可见槽计数差不超过 1，singleton slot2 仍记录 physical slot 2。

在 encoder 参数上计算全局 `dot(g_source,g_alignment)`。若小于 0，整个 alignment gradient 置零；否则所有
非 classifier alignment 参数接收 `g_source + 0.02 g_alignment`。classifier 永远只接收 source gradient。
每 epoch 记录 fused/channel signed MMD 的负值率、mean/std/p05/p50/p95，以及冲突关闭率。

## 3. 信息边界与因果核对

- anchor 不构造 target-train dataset、不迭代 target waveform sample；branch 只允许 source labels 与 target-train waveform，target 标签必须逐批为 `-1`；
- trainer CLI 不接受 target-dev/final 路径；目标开发评分由独立 evaluator 完成；
- control/candidate 每对必须有相同 anchor SHA、step-0 model/optimizer/scheduler/all-RNG hash、source row digest 和 mask digest；
- 任一 checkpoint、数据 manifest、代码 snapshot、kernel schema、replay hash、pair replay 或 label sentinel 不一致即失败；
- seal 前重新读取每个 epoch JSONL、run_state 与 checkpoint：验证递归数值有限、run/checkpoint 位于当前 campaign/runs
  且 run_id 精确一致、四个 seed 和代码身份一致、60/20 epoch 与 optimizer/alignment step 总数一致；
- 若进程在“run 已完成、fit_results 尚未登记”的提交窗口中断，只允许在既有工件通过同一完整审计后幂等补登记；
  missing/partial/身份不符的 orphan 一律停止，禁止删除或静默重跑；
- 第二阶段不按 source-val、target entropy、DEV、CE-DEV、IW-GAE 或 target 指标选时点；
- 本批不加入 order normalization、pseudo-label、conditional MMD、新 backbone 或按任务调参。

## 4. 训练前硬门

1. fixed-kernel payload 和 replay hash 可重放；篡改 kernel 或 RNG hash 必须被拒绝；
2. channel sampler 连续 1000 minibatch 的累计槽计数差不超过 1，相同 seed bitwise replay，slot2 记录正确；
3. equal-B 不等行数直接报错；同分布 500 seeds 的 95% CI 覆盖 0，all3 与 singleton 均值差不超过 2 pooled SE，并实际出现负值；
4. mean shift `0,.25,.5,1` 的重复 MMD 均值单调；
5. 冲突时 optimizer input 与 pure-source gradient bitwise 相同；非冲突时精确等于 `g_s+0.02g_a`，classifier 无 alignment gradient；
6. evaluator 在任何 target-dev 预测或私有标签解码前验证完整 provenance；
7. runner matrix 必须恰为 4 anchors、16 control、16 candidate，且 target-final 始终为 false。

## 5. 冻结门槛

因果扩展门比较 `v2A` 相对本轮配对 common-anchor control：16 条件至少 10 条 Accuracy 与 Macro-F1 同时提高；16 条件
等权 Accuracy 至少 `+1 pp`、Macro-F1 至少 `+2 pp`；双跨 8 条件 Macro-F1 至少 `+2 pp`；每个任务三个
single-slot 中至少两个双指标提高；strong control 中最弱 single-slot Macro-F1 必须提高；不新增零召回类别。
任一失败则 v2A 不升级、不补 seed43/44、不按数据集保留有利策略。相对 frozen v4 strong matched control 另算同一组
阈值作为历史诊断，但其结果不决定是否扩展，也不能解释 alignment 的净因果效果。

## 6. 执行边界

prepare 本身不训练。正式启动必须使用 frozen snapshot 的 runner。PowerShell 启动器在每个 optimizer run 边界记录
Windows RAM、GPU utilization/free memory、完整 compute-process 列表、允许 PID、锁文件与 PowerShell PID；未知 GPU
计算进程会暂停，脚本不终止其他进程。stale lock 仅在 payload/output 匹配、owner PID 已不存在且锁龄至少 300 秒时
自动解除并写审计事件。默认训练完成后只写 predev seal 并停止；必须显式加
`-ScoreDevelopment` 才读取 target-dev。target-final 没有入口。

计划创建命令（不训练）：

```powershell
& 'C:\Users\Administrator.SC-202505251914\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  D:\3800\protocol_next\run_channel_robust_v2_campaign.py prepare `
  --output D:\3800\experiments\20260925_channel_robust_v2a_screen_v2
```

正式训练命令（由用户确认资源与允许的既有 GPU PID 后运行）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\3800\protocol_next\start_channel_robust_v2_campaign.ps1 `
  -Output D:\3800\experiments\20260925_channel_robust_v2a_screen_v2 `
  -AllowedGpuPid <explicit-existing-pid-list>
```

训练与 seal 完成后，显式评分：

```powershell
powershell -ExecutionPolicy Bypass -File D:\3800\protocol_next\start_channel_robust_v2_campaign.ps1 `
  -Output D:\3800\experiments\20260925_channel_robust_v2a_screen_v2 `
  -ScoreDevelopment
```
