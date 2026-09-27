# 加载已有共享谱分类器做预测

入口：[predict_shared_ridge.py](predict_shared_ridge.py)。它读取已保存的共享Ridge/线性谱NPZ模型，输出每个窗口的类别编号与线性分数；不训练、不重算目标统计、不需要标签。当前已用实验室水泵、齿轮箱和SEU的四个真实模型、32个源训练窗口核对，类别与分数均与原评分器完全一致；另有5项输入/格式/CLI测试。[核验](../audit/shared_ridge_inference_integrity_20260915.json)。这是推理接口核验，不是新的目标准确率。

## 输入约定

- `.npy`数组形状必须为`[窗口数, 目标通道数, 2048]`，保存未做逐窗口标准化的原始实数波形。程序按原数据加载器先逐窗、逐通道标准化，再计算固定对数谱和应用模型中保存的统计量。
- 仅输入这个模型训练时可用的目标槽，顺序通过`--target-slots`声明。例如槽2模型的数组是`[N,1,2048]`，仍应填`--target-slots 2`，不能把槽2重新标为槽0。三通道模型填`--target-slots 0 1 2`。
- 采样率、测点和轴向应与模型的数据协议一致。本入口不重采样，也不能从数组自动判断物理传感器身份；SEU5120Hz与实验室24kHz的同样2048点代表不同时间长度。类别编号按对应训练数据协议解释。
- 输入文件只应包含本次确实要预测的窗口；不要把包含train/dev/final的完整角色库直接作为输入。日常推理接口不替代论文的来源隔离评分器。

## 可复现的接口示例

在PowerShell运行下面命令。示例使用随核验保存的8个源窗口，便于检查安装和预测流程；它不是目标测试集。输出文件必须不存在，下面用时间戳生成新名称。

```powershell
$ridgeOutput = '/mnt/d/3800/predictions/demo_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.npz'
wsl.exe -d VMUNet-Ubuntu22 -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/vmunet/venvs/widan/bin/python /mnt/d/3800/protocol_next/predict_shared_ridge.py --model /mnt/d/3800/experiments/20260915_shared_ridge_baseline_v1/fit/WP-S0_source3_target2_zscore.npz --input /mnt/d/3800/audit/shared_ridge_inference_smoke_20260915/source_windows.npy --target-slots 2 --output $ridgeOutput
```

对新数据预测时，把`--input`替换成遵守上述约定的NPY；模型和槽号也应对应同一任务。程序使用CPU单线程、按32窗分批处理，可用`--batch-size`调整。

输出NPZ包含：

- `predicted_class`：`[N]`整数类别编号。
- `scores`：`[N,K]`线性分数，最大分数对应预测类别，**不是已校准概率或可靠度**。
- `metadata_json`：模型及输入SHA256、通道顺序、训练任务身份、预处理和推理源码摘要；明确没有读取标签或做测试时重新校准。

此入口只支持当前共享线性谱模型格式；神经网络`.pth`权重尚不通过该入口加载。项目最终主模型、跨工况高性能和完整论文交付仍需后续工作，不能把本接口可运行视为全部目标完成。
