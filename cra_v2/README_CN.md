# CRA-v2A-U-fixedK 可验证交付包

本包保存一套共用模型结构，以及每个任务、每种目标通道视图分别训练的 checkpoint。`model_registry.json` 是模型入口；不存在声称可直接覆盖任意目标通道的通用权重。

先运行 `python tools/verify_channel_robust_v2_delivery.py .`。推理入口为 `inference/predict_channel_robust_v2.py`，输入必须是 float32、形状 `[N,C,2048]` 的原始窗口。实验室水泵/齿轮箱使用 24 kHz 合同；程序不会自动重采样或识别传感器。

完整性审计：`PASS`。性能门：`FAILED`。本次注册开发门为 `False`，推荐模型 ID 为 `null`，用途限定为 `research_evaluation_only`。性能门失败意味着候选方法未达到预注册提升标准，不能作为高性能或已推广模型交付。目标最终分区从未评估。AWGN 是开发集后续验证，不得写成独立最终测试结果。完整状态见 `STATUS.json`，结果与限制以 `experiment/REPORT.md` 和 `docs/` 为准。
