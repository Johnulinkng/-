# Kernel readout inference

`predict_kernel_readout.py` is a deployment-only entry point for the frozen source-selected linear Ridge and RBF-KRR models. It does not fit parameters, read labels, select a model, or recalibrate target statistics.

Inputs are explicit `.npy` arrays:

- `--input-type features`: finite `[N, C, 128]` log-power features, where `C` is 1 or 3.
- `--input-type raw`: finite `[N, C, 2048]` windows. Each channel is normalized independently per window, then the fixed low3k_pool2 transform is applied (symmetric Hann, remove DC, 256 positive bins grouped in pairs, normalized log power).

The target slot list must exactly match the saved model metadata: `[0]`, `[1]`, `[2]`, or `[0,1,2]`. Batch size only controls memory use and cannot change predictions. Output is a new `.npz` containing `predicted_class`, `scores`, and a provenance `metadata_json`; labels are never accepted or read.

Example:

```text
python delivery_tools/predict_kernel_readout.py --model MODEL.npz --input windows.npy --input-type raw --target-slots 0 --batch-size 64 --output predictions.npz
```

The test suite checks all 88 saved models against their frozen CPU kernel core, both readout methods, all three datasets, and one- and three-slot models. It also checks raw preprocessing against the frozen Torch spectral grid. Run it with the project Python environment and one CPU thread.
