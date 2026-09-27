# CRA-v2 交付层说明

本目录的 v2 工具只接受 `channel_robust_v2a_u_fixedk_campaign_v1` 完整实验。构建器会先独立重建 4 个 source anchor、32 个二阶段分支、32 份 target-dev 结果、16 组因果配对和屏幕门，再开始复制文件。它不打开波形数组、私有标签或 target-final 分区。

## 工具

- `predict_channel_robust_v2.py`：读取一个明确的目标通道 checkpoint，对原始 `float32 [N,C,2048]` NPY 做无标签推理。三通道和三个单通道使用同一模型结构，但必须选择各自独立训练的 checkpoint。
- `protocol_next/evaluate_channel_robust_v2_noise.py`：屏幕门通过后，对 candidate arm 的 target-dev 执行 clean、20/10/0 dB × 3 seeds AWGN 评估。噪声先加在原始窗口指定物理通道，再执行训练一致的逐窗口标准化。目标最终分区不评分。
- `audit/verify_completed_channel_robust_v2_campaign.py`：对原始完整 campaign 做独立、fail-closed 审计。
- `build_channel_robust_v2_delivery.py`：生成目录、ZIP、外部 ZIP SHA256 sidecar。
- `verify_channel_robust_v2_delivery.py`：离线验证交付包的精确成员、SHA、模型注册表、实验收据和 v4 冻结参考；不需要数据集。

## 构建

在 Windows PowerShell 中：

```powershell
python D:\3800\delivery_tools\build_channel_robust_v2_delivery.py `
  --campaign D:\3800\experiments\20260925_channel_robust_v2a_screen_v5 `
  --output D:\3800\deliverables\20260925_channel_robust_v2 `
  --archive D:\3800\deliverables\20260925_channel_robust_v2.zip `
  --execute
```

在 WSL 中可以把三个路径替换为 `/mnt/d/...`。campaign 必须已完成 fit、pre-dev seal、全部 dev scoring 和 report；构建器不会继续训练或补评分。

若已按注册协议完成 AWGN target-dev 后续评估，可重复加入 `--noise-result D:\path\to\awgn_result.json`。构建器会核对 arm、candidate checkpoint、10 个固定条件和“未解码 target-final”收据；因果 screen gate 未通过时拒绝纳入 AWGN 结果。

## 离线校验

```powershell
python D:\3800\delivery_tools\verify_channel_robust_v2_delivery.py `
  D:\3800\deliverables\20260925_channel_robust_v2
```

交付目录内也能运行：

```powershell
python tools\verify_channel_robust_v2_delivery.py .
```

`MANIFEST.json` 覆盖除自身外的每个文件；`SHA256SUMS.txt` 覆盖生成它之前的每个文件；ZIP 的 SHA256 写入同名 `.zip.sha256` sidecar。任何增删或修改都会使校验失败。

## 推理

从 `model_registry.json` 选择与任务、目标通道和角色完全一致的记录。论文方法使用 `v2a_candidate`；`common_anchor_control` 只用于因果对照。示例：

```powershell
python inference\predict_channel_robust_v2.py `
  --checkpoint models\candidates\<arm-id>\best_model.pth `
  --checkpoint-sha256 <registry-sha256> `
  --task-id PG-S1 `
  --variant candidate `
  --target-slots 1 `
  --sampling-rate-hz 24000 `
  --input D:\path\raw_windows.npy `
  --output D:\path\predictions.npz
```

程序不重采样，不自动判断传感器，且拒绝 checkpoint SHA、任务、通道、采样率、模型代码或 v2 信息边界不一致的调用。单通道输入的 `C=1`，三通道输入的 `C=3`，通道次序必须与训练协议一致。

## AWGN 开发集验证

复制 `config/paths.local.example.json` 到交付目录外并填写本机 public/private audit roots。对某个已注册 candidate checkpoint 运行：

```powershell
python robustness\evaluate_channel_robust_v2_noise.py `
  --campaign experiment\20260925_channel_robust_v2a_screen_v5 `
  --code-root code `
  --arm-id <candidate-arm-id> `
  --checkpoint models\candidates\<candidate-arm-id>\best_model.pth `
  --public-root D:\path\to\public_root `
  --private-root D:\path\to\private_audit_root `
  --affected-slots all `
  --output-json D:\path\to\awgn_result.json `
  --execute
```

输出必须放在 campaign、public data 和 private audit data 之外。该工具只有在因果 screen gate 为真时执行，先固定全部 clean/noisy 预测，再首次解码 target-dev 标签。结果仍属于开发集后续验证，不能称作 target-final 或独立测试结果。

## 结果表述边界

`STATUS.json` 是交付状态入口。若 `screen_gate_passed=false`，交付仍可作为完整的系统评估和负结果包，但 candidate 不得宣称优于 common-anchor control。即使该门通过，`target_final_evaluated` 仍为 `false`；最终论文表述需要遵守实验报告中的数据划分、单长记录和开发集复用限制。
