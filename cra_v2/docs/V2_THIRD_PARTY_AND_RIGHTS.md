# CRA-v2 第三方与分发边界

生成日期：2026-09-25。适用对象：`channel_robust_v2a_u_fixedk_campaign_v1` 的合同内代码、模型和结果交付。

本文件记录当前 v2 冻结代码的可见边界，不替代合同、学校/实验室数据协议或权利人的许可决定。

## 冻结代码边界

v2 campaign 的 `plan.json/code_sha256` 是唯一代码成员清单。核心模型、训练和协议文件为：

- `workbench/models/channel_robust_alignment_v2.py`
- `workbench/models/channel_robust_alignment.py`
- `workbench/models/spectral_grid_pilot.py`
- `workbench/models/spectral_shared.py`
- `workbench/loss/balanced_unbiased_mmd.py`
- `workbench/experiment_protocol.py`
- `protocol_next/train_channel_robust_alignment_v2.py`
- `protocol_next/evaluate_channel_robust_alignment_v2_dev.py`
- `protocol_next/public_loader.py`
- `protocol_next/private_evaluator.py`
- `protocol_next/antipair_builder.py`
- campaign runner、启动脚本和对应测试/设计收据。

这些路径在已审计的 WIDAN 固定树 `liguge/WIDAN@3ce417098bdeec5c8e006540969eb075e961edd4` 中不存在。v2 冻结清单不包含 WIDAN 的 `TICNN.py`、小波初始化、原训练脚本或 UDTL 派生的 `loss/DAN.py`；v2 使用的是独立的 `balanced_unbiased_mmd.py`。因此旧 CRA-v1 包针对 `DAN.py` 的 MIT notice 不能被误写成整个 v2 项目的许可证，也不是本 v2 冻结包的必需成员。

WIDAN 仓库在审计时未提供可见的 `LICENSE`、`COPYING` 或 `NOTICE`。本交付只把 WIDAN 作为研究与对照来源引用，不据此授予 WIDAN 特有代码的再分发权。

## 项目代码与模型权重

当前工作区没有经权利人批准的项目级开源许可证。v2 源码、交付工具和模型权重可作为合同内交付物；这不自动授权第三方公开发布、再许可或把项目标为 MIT/Apache。若要对外开源，应由合同权利人确认版权主体、选择许可证，并按最终 ZIP 的实际文件重新审计。

## 运行依赖

交付包只记录依赖版本，不捆绑 PyTorch、NumPy、SciPy、scikit-learn、pytest、CUDA 或 Python 运行时。使用者从各项目官方发行渠道安装依赖并遵守对应许可证。若后续打包 wheel、conda 包或 CUDA 组件，应加入精确版本的随附许可证，不能沿用本说明代替。

## 数据边界

交付包不包含：

- 实验室水泵或齿轮箱的原始记录、窗口数组、标签或私有评估根；
- 东南大学齿轮箱的 MAT/CSV/NPY 信号或标签；
- target-dev/target-final 私有标签文件；
- 训练数据的绝对本机路径配置。

包内 `config/task_registry.json` 只保存任务语义和三个公开元数据文件的 SHA-256；`paths.local.example.json` 只有 Windows/WSL 占位符。使用者在包外维护本机数据路径。实验室数据和东南大学数据的对外再分发须分别取得数据权利人的授权。

## 最终核验

`MANIFEST.json`、`SHA256SUMS.txt` 和 ZIP sidecar 是文件完整性证明，不是许可证。每次改变代码、模型、数据、依赖或对外发布范围，都应重新生成包并复核本文件。
