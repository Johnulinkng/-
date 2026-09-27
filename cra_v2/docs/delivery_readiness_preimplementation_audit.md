# 通道鲁棒迁移项目独立交付就绪性审计

日期：2026-09-25  
审计范围：`D:\3800` 当前工作区；重点检查 CRA-v2A-U-fixedK 的训练后交付链，以及现有 CRA-v1 交付工具能否复用。  
信息边界：本次没有启动训练，没有读取、索引或评分任何 target-final 标签；只检查代码、公开元数据、既有开发结果状态和合成测试。没有修改 v2 trainer、runner 或 evaluator。

## 1. 结论

**当前状态为“训练前实现基本可运行，但 v2 最终交付链尚未建立”，不能把现有目录或 CRA-v1 包当作 v2 最终交付。**

已具备的部分：

- v2 模型、fixed-kernel equal-B signed U-statistic MMD、训练器、开发评分器、冻结 runner 与 PowerShell 资源门禁已经有代码入口；
- 当前合成/静态 v2 测试为 `12 passed`；
- 旧 CRA-v1 的打包、完整性验证、推理和噪声工具自身测试通过：交付/推理测试 `22 passed, 10 subtests passed`，AWGN 测试 `4 passed`；
- 水泵、实验室齿轮箱的 schema-3 public 数据目录和本轮四个任务 `WP-S0/WP-D1/PG-S1/PG-D1` 均存在，任务窗口为 2048，分别为 5 类和 4 类；
- 上游来源、数据分发边界和 WIDAN 未发现许可证的事实已有审计材料。

未具备的关键部分：

- 尚无完成并封存、评分、独立审计通过的 v2 campaign；
- 现有 builder/verifier/auditor、推理 CLI、AWGN evaluator 全部绑定 CRA-v1 schema，不能接受 v2 checkpoint/campaign；
- 方法章节和交付清单仍描述 CRA-v1、60 epoch 单阶段选模与“5/32 暂停”，与当前 v2 两阶段机制不一致；
- 当前训练数据元数据和启动器绑定本机绝对路径，无法从移动后的干净目录直接复现训练；
- 缺少 v2 专用的环境锁、从零预检、模型注册表、最终用户说明和一个经过解压换目录验证的 ZIP。

因此当前可以继续执行 v2 冻结筛查，但还不能进入“最终可交付”状态。训练结果出来后，必须先完成下列 P0 项，再打包。

## 2. 已检查的入口与证据

### 2.1 v2 研究入口

|用途|当前入口|现状|
|---|---|---|
|计划、逐臂拟合、封存、逐臂开发评分、报告|`protocol_next/run_channel_robust_v2_campaign.py`|CLI 可加载；冻结矩阵为 4 anchors + 32 branches|
|共享主机资源门禁和串行启动|`protocol_next/start_channel_robust_v2_campaign.ps1`|具备 lock、Windows RAM、GPU utilization/free memory、compute-process allowlist；默认训练后停在 seal|
|单臂训练|`protocol_next/train_channel_robust_alignment_v2.py`|anchor 60 epoch；control/adapt 从同一锚点各固定 20 epoch|
|开发评分|`protocol_next/evaluate_channel_robust_alignment_v2_dev.py`|只接受 v2 固定端点 checkpoint；无 final 入口|
|模型|`workbench/models/channel_robust_alignment_v2.py`|沿用共享编码器/有界融合；拆分 source-anchor 与 target-train 校准生命周期|
|对齐估计器|`workbench/loss/balanced_unbiased_mmd.py`|equal-row、去对角、signed、固定多核|

只读 `--help` 检查全部退出码为 0。v2 合成/静态测试命令：

```powershell
& 'C:\Users\Administrator.SC-202505251914\AppData\Local\Programs\Python\Python313\python.exe' -B -m pytest -q `
  protocol_next\tests\test_channel_robust_alignment_v2.py `
  protocol_next\tests\test_channel_robust_v2_campaign_static.py `
  --basetemp D:\3800\.pytest_tmp_v2_delivery_readiness
```

结果：`12 passed in 4.59s`。这只证明训练前代码级合同，不是实际数据性能或最终交付证明。

### 2.2 现有 v1 交付链

|用途|当前入口|审计结果|
|---|---|---|
|打包|`delivery_tools/build_channel_robust_delivery.py`|v1 可用；v2 不兼容|
|离线验包|`delivery_tools/verify_channel_robust_delivery.py`|v1 可用；v2 不兼容|
|完整 campaign 独立审计|`audit/verify_completed_channel_robust_campaign.py`|只认 `channel_robust_alignment_campaign_v1` 和每臂 60 epoch|
|原始 NPY 推理|`delivery_tools/predict_channel_robust_alignment.py`|只认 `matched_control/full` 与 v1 模型代码身份|
|AWGN|`protocol_next/evaluate_channel_robust_alignment_noise.py`|只认 v1 campaign kind、v1 trainer/evaluator/model|

合成测试：

```powershell
& 'C:\Users\Administrator.SC-202505251914\AppData\Local\Programs\Python\Python313\python.exe' -B -m pytest -q `
  delivery_tools\tests\test_channel_robust_delivery.py `
  delivery_tools\tests\test_channel_robust_alignment_inference.py `
  --basetemp D:\3800\.pytest_tmp_delivery_readiness

& 'C:\Users\Administrator.SC-202505251914\AppData\Local\Programs\Python\Python313\python.exe' -B -m pytest -q `
  protocol_next\tests\test_channel_robust_noise_evaluator.py `
  --basetemp D:\3800\.pytest_tmp_noise_delivery_readiness
```

结果分别为 `22 passed, 10 subtests passed` 与 `4 passed`。结论是旧链路没有因本轮开发而损坏；它的通过不能外推为 v2 可交付。

## 3. P0：最终打包前必须解决

### P0-1 完成 v2 campaign 并产生封存证据

最终包至少需要：4/4 anchor、32/32 branch、每对 control/candidate 的共同锚点和 replay identity、`predev_seal.json`、32/32 target-dev 结果、`summary.json`、`REPORT.md`、`pipeline_state.status=completed`、`target_final_evaluated=false`。当前没有对应完成目录，故无法生成真实模型注册表或交付 ZIP。

v2 的筛查结果必须原样落盘。若因果扩展门失败，交付层级应明确为“统一变通道模型的系统评估与负迁移边界”，不能写成高性能方法；若通过，也只能按预注册分阶段规则继续多方向、多种子、公开集与噪声验证，不能把 seed42 开发筛查直接写成论文终证据。

### P0-2 新建 v2 独立完成审计器

旧审计器硬编码：

- campaign kind 为 `channel_robust_alignment_campaign_v1`；
- variants 为 `matched_control/channel_robust_alignment_v1`；
- `fit_results.json` 是 arm list；
- 每臂 60 epoch；
- v1 summary/gate 重建规则。

v2 审计器应独立重建并验证：

1. 4 个 source anchor 与 32 个第二阶段 branch 的数量、顺序和 checkpoint SHA；
2. anchor 的 60 epoch source-val 选模，以及 branch 固定 epoch 19，不在 branch 中二次选模；
3. 每对 branch 的 `anchor_checkpoint_sha256`、step-0 model/optimizer/scheduler/all-RNG identity、source row digest 和 mask digest相同；
4. fixed-kernel payload、source-only kernel pool identity、equal-B physical-slot cycle、signed MMD 日志、冲突抑制计数；
5. target-train 标签 sentinel `-1`、无同步配对、开发评分发生在 seal 之后、target-final 始终 false；
6. 32 个开发 JSON 的指标、混淆矩阵、逐类召回、summary 和两种比较均可独立重算；
7. `common_anchor_control` 决定因果扩展门，frozen v4 `matched_control` 只作哈希锁定的历史强基线；
8. v4 reference 的 plan/fit/seal/dev/summary/report/独立审计 receipt 与准备时 SHA 一致。

### P0-3 新建或泛化 v2 builder/verifier

现有 builder 在 `validate_campaign` 中把 `fit_results.json` 当 list，并要求 `len(fits)==len(plan.arms)`；v2 实际为 `{anchors: [...], arms: [...]}`。现有 verifier 又要求 `epochs_per_arm==60`、v1 registry/status schema 和 v1 审计 receipt，因此即使 v2 完成也会拒绝。

v2 builder/verifier 至少要：

- 分别登记 anchors、common-anchor controls、v2 candidates；
- 明确部署默认模型只指向冻结 candidate，controls/anchors 标为复现与审计用途；
- 对每个 task 的 all3/slot0/slot1/slot2 分别登记 checkpoint，满足“一套结构与代码、每个目标通道分别训练评估”；
- 打包 campaign frozen code，而不是重新复制可能已变化的 workspace 训练代码；
- 包含 v2 独立审计 receipt、两种 gate、完整指标和限制；
- 生成相对路径 manifest、文件数、每文件 SHA、ZIP SHA，并在解压到新目录后重新运行 verifier；
- fail closed：缺 checkpoint、hash 不一致、额外文件、缓存文件、错误 schema、target-final 标记异常均失败。

### P0-4 提供 v2 推理 CLI 与逐样本等价测试

现有 predictor 的公开 variant 仅有 `matched_control/full`，内部身份仅有 `channel_robust_alignment_v1`，模型文件与 checkpoint metadata 检查均绑定 v1。v2 checkpoint 必然被正确拒绝，不能作为 v2 使用说明中的入口。

v2 推理器须：

- 加载 `ChannelRobustAlignmentV2`、v2 base model 与精确冻结依赖；
- 验证 `adaptation_endpoint=true`、selected epoch 19、variant、task、dataset、physical target slots、模型代码 SHA、checkpoint SHA、target-train 校准 buffers 和 provenance；
- 输入固定为 finite float32 `[N,C,2048]`；all3 的通道顺序固定为 `[0,1,2]`，singleton 必须显式指定物理 slot；
- 先执行与 public loader 相同的逐窗口逐通道 population z-score，再使用 checkpoint 内 target-train spectral calibration；
- 输出类别 index、经证据约束的显示名、logits、全部身份 metadata，拒绝覆盖；
- 对 all3、slot0、slot1、slot2 至少各做一次 evaluator-vs-CLI 样本级回放，类别 bitwise 相同，logits 使用事前容差；
- 解压换目录后单独运行，且不读取数据集、标签、target-dev/final。

### P0-5 提供 v2 AWGN 适配器，或在 gate 失败时明确不运行

现有噪声 evaluator 写死 v1 campaign kind 和 v1 checkpoint 依赖。v2 版本需要保留已冻结协议：raw waveform 上加噪后再做逐窗口标准化；clean target-train 校准不重算；20/10/0 dB × 1701/1702/1703；仅扰动指定可见物理槽；先固定所有预测，再解码 target-dev 标签；永不索引 target-final。

按预注册 staged protocol，第一阶段 gate 任一失败就不启动 v2 AWGN/SEU 扩展。此时最终包应保留既有强基线噪声证据并写清“v2 未取得运行资格”，不能为补齐交付表而事后越过停止规则。

### P0-6 重写最终文档，使公式和实际 v2 一致

下列文件目前过时，不能原样进入最终包：

- `项目当前状态.md` 首段仍写 CRA-v1 与 5/32 暂停；
- `deliverables/20260925_channel_robust_method_draft/README.md` 仍声明 v1 未完成；
- `方法与论文第三章草案.md` 仍写 v1 的 60 epoch 单阶段、biased/batch MMD 语义、冲突梯度投影、v1 消融与待跑表；
- `最终交付清单草案.md` 仍以 v1 入口、5/32 和旧恢复命令为事实；
- `deliverables/README.md` 仍把 2026-09-16 研究快照列为当前可运行阶段成果。

最终第三章必须改写为实际 v2：共享 60-epoch source anchor；相同状态分叉；两支固定 20 epoch；candidate 唯一多出 fixed-kernel equal-B signed U-statistic MMD 与冲突时整项关闭；因果门与历史强基线分开；结果表用真实 summary 填写。若 gate 失败，要把失败表、机制诊断和主张降级一起写入，不得只更新模型名称。

### P0-7 修正交付数据清单的绝对路径泄漏与不可搬移问题

schema-3 的 `version_manifest.json` 与 task manifest 内记录 `/mnt/d/3800/...` 绝对路径；`public_loader._inside` 会解析该路径并要求等于 `<public_root>/domains/...`。因此把数据目录移动到另一台机器或另一路径后，loader 会 fail closed。现有 builder 又会复制 task manifest 到包内，第三方通知却要求公开 manifest 使用相对或占位路径，两者存在冲突。

建议分成两类工件：

1. 交付包中的 `data_identity/` 使用经过明确 schema 标记的脱敏相对路径清单，仅用于说明 task、shape、class、slot、SHA 和重建来源，不假装可直接训练；
2. 训练复现使用 `paths.local.example.json` + 数据重建/重定位工具，在新位置重新生成自洽 manifest/hash。不要直接手改受 SHA 保护的旧 JSON。

若本次只要求在原 `D:\3800` 内部交接，可把这一项降为外部复现限制，但仍要在 README 明示。它不影响打包后纯 checkpoint 推理的可搬移性。

### P0-8 明确内部交付与公开发布的许可边界

WIDAN 仓库未发现许可证；CRA 项目自有代码也没有已批准的 outbound license。内部合同交付可以按合同权属处理，但公开 GitHub、论文附件或向第三方分发源码前必须由权利人补充许可。v2 最小运行依赖并不需要把 WIDAN 上游源码或旧 `DAN.py` 一并塞入部署包；按实际包成员生成第三方通知，避免继续沿用“适用 CRA-v1 且包含 DAN.py”的旧文本。

## 4. P1：可直接修复的可用性问题

1. **启动器参数化。** `start_channel_robust_v2_campaign.ps1` 当前写死 `D:\3800`、`VMUNet-Ubuntu22` 与 `/home/vmunet/venvs/widan/bin/python`。增加 `-Project/-Distro/-Python/-Output`，并把映射后的 WSL 路径写入 launch audit；冻结 campaign 仍继续执行 snapshot runner。
2. **拆分环境。** 当前 Windows Python 3.13 环境为 torch 2.7.0+cu118，CPU venv 为 Python 3.12/torch 2.14.0+cpu 且没有 pytest，历史 WSL GPU lock 为 Python 3.10/torch 2.1.2+cu118。最终包应提供 `requirements-training-gpu.lock.txt`、`requirements-inference-cpu.lock.txt`、`requirements-test.lock.txt` 和实际 `environment-training.json`；builder 不能只把自己的环境版本写成训练环境。
3. **从零 preflight。** 增加只读命令，检查 Python/Torch/CUDA、模型代码 hash、数据 manifest、任务和 checkpoint registry，不加载标签、不写模型；退出码区分缺依赖、数据缺失、hash 失败和 CUDA 不可用。
4. **统一入口说明。** 顶层中文 README 给出四条最短命令：验包、CPU 推理、训练复现、开发/噪声重算；每条都使用相对 package root 或显式配置文件。
5. **模型选择说明。** registry 为每个 task/mode 指明默认 checkpoint、类别顺序、采样率证据、输入通道、物理槽、clean 指标、gate 状态和是否只供 control/audit。
6. **包体配置。** 不包含原始实验室/SEU 数据、private audit、target-dev/final 标签或绝对私有路径；需要复算开发指标时由有权限的独立 evaluator 在包外挂载数据。
7. **最终 smoke。** 在全新解压目录运行 verifier；随机生成合法 `[N,C,2048]` float32，跑 all3 和三个 singleton；再分别验证错误 dtype、窗口、通道顺序、slot、checkpoint SHA、已有输出都会失败。

## 5. 建议的最终交付树

```text
WIDAN_CRA_DELIVERY_<version>/
├─ README_CN.md
├─ STATUS.json
├─ MANIFEST.json
├─ SHA256SUMS.txt
├─ environment/
│  ├─ environment-training.json
│  ├─ requirements-training-gpu.lock.txt
│  ├─ requirements-inference-cpu.lock.txt
│  └─ requirements-test.lock.txt
├─ config/
│  ├─ paths.local.example.json
│  ├─ task_registry.json
│  └─ class_mapping.json
├─ code/
│  ├─ protocol_next/                 # frozen v2 runner/trainer/evaluator/loader
│  └─ workbench/                     # frozen v2/base model, spectrum, loss, metrics
├─ models/
│  ├─ anchors/<task>/
│  ├─ controls/<task>/<all3|slot0|slot1|slot2>/
│  └─ candidates/<task>/<all3|slot0|slot1|slot2>/
├─ registry/
│  └─ model_registry.json
├─ experiment/
│  ├─ plan.json
│  ├─ predev_seal.json
│  ├─ fit_results.json
│  ├─ dev_results.json
│  ├─ summary.json
│  └─ REPORT.md
├─ reference/
│  ├─ v4_strong_control/             # 被 v2 plan 锁定的比较证据
│  └─ kernel_baselines/              # 论文强基线及 post-exposure 边界
├─ robustness/
│  ├─ protocol.json
│  ├─ results/                       # 仅在预注册 gate 允许后存在
│  └─ evaluate_awgn_v2.py
├─ inference/
│  ├─ predict_channel_robust_v2.py
│  └─ INFERENCE_CN.md
├─ data_identity/
│  ├─ README.md
│  └─ relative_manifests/            # 无原始数据、私有标签或绝对内部路径
├─ docs/
│  ├─ 方法与论文第三章.md
│  ├─ 实验结果与提升率.md
│  ├─ 使用说明与故障排查.md
│  ├─ 已知限制.md
│  └─ THIRD_PARTY_NOTICES.md
├─ audit/
│  ├─ completed_v2_campaign_audit.json
│  ├─ code_review.md
│  ├─ delivery_verification.json
│  └─ test_report.txt
├─ tools/
│  ├─ preflight.py
│  ├─ verify_delivery.py
│  └─ reproduce_screen.ps1
└─ tests/
   └─ synthetic_smoke/
```

若第一阶段 gate 失败，目录树仍可使用，但 `STATUS.json.delivery_class` 必须是 `system_evaluation_negative_transfer_boundary`，`models/candidates` 标为被评估但未晋级的候选，`robustness/results` 不伪造，README 直接给出失败项。若后续门槛全部通过，再把状态提升为方法候选交付，并把多种子/公开集/噪声证据补齐。

## 6. 最终签收门槛

必须全部满足才标为可交付：

1. 完成的 v2 campaign 与独立 v2 审计均为 PASS；
2. v2 predictor 对 all3/slot0/slot1/slot2 的锁定 checkpoint 回放通过；
3. builder 和 verifier 在干净输出目录通过，ZIP CRC 与 SHA 通过，解压换目录复验通过；
4. 文档、模型 registry、summary、论文表格与同一组真实结果一致；
5. target-final 为 false，历史 final 暴露边界没有被删除；
6. 数据与许可边界明确，包内无原始数据、私有标签、无意泄露的绝对路径或无权再分发的上游代码；
7. 若声称方法提升，必须有预注册 gate、多种子、任务方向、逐槽和噪声/公开集证据；否则按系统评估型成果交付。

## 7. 本次审计快照与哈希

本次重点检查文件在审计时的 SHA-256：

|文件|SHA-256|
|---|---|
|`protocol_next/run_channel_robust_v2_campaign.py`|`8830bc3f1dd6f3966aa838f9bc77419fb366035cd9927edfe738e3e215495533`|
|`protocol_next/start_channel_robust_v2_campaign.ps1`|`da800c75afc1c0227f8cf5498d9606f761d06cbe599972580cef7a581df735e5`|
|`protocol_next/train_channel_robust_alignment_v2.py`|`81a4089749d651140568a55fa6b02d63fcd34b03187c88d71f0bbeb9735acdae`|
|`protocol_next/evaluate_channel_robust_alignment_v2_dev.py`|`4cc0eed877a23d62b91ddfb0616e2d8c6836e509a037384f88a3cf246ff3171e`|
|`workbench/models/channel_robust_alignment_v2.py`|`3555cd9a935db606efd432458eb2c1fc5dc4d97f3998c6d683e6426c0a8fe772`|
|`workbench/loss/balanced_unbiased_mmd.py`|`2fb4e6778f81cefc56dfb872730702fab4671f5785d646a1e67b787a33314d46`|
|`delivery_tools/build_channel_robust_delivery.py`|`aa03c2175d74f6f48ddfe1be15cc36698caa670d2ce05d8a6076948f10c681aa`|
|`delivery_tools/verify_channel_robust_delivery.py`|`0a88fb90b1da2e092d6c0fa431ee2e825767a715c5fa839710885f6d6ca65b8d`|
|`delivery_tools/predict_channel_robust_alignment.py`|`2ccc60c1deb5b30b1d7d2768d6d25eac27a1f6a2f039d31a3ad72f1fa5c3cd26`|
|`protocol_next/evaluate_channel_robust_alignment_noise.py`|`55b495b0c8956b73cbb6091d84edc8e446a441dbbd035be104930753a1d3eac2`|

v2 文件在本次审查期间仍处于开发收束阶段；冻结 campaign 的 plan/code snapshot 才是后续训练与交付的权威代码身份，不能用本节哈希替代 campaign seal。
