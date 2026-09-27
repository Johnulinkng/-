# Channel-robust alignment campaign

Study: `screen`. All 32 checkpoints were selected by source validation and SHA-sealed before target-development scoring.
The target-development partitions were used in earlier research, so these are matched development results rather than untouched confirmation evidence.
Target final was not evaluated.

|Task|Target|Seed|Variant|Accuracy|Macro-F1|Recall by class|
|---|---|---:|---|---:|---:|---|
|WP-S0|all3|42|matched_control|68.93%|67.13%|[0.6133333333333333, 0.8, 0.22, 0.8133333333333334, 1.0]|
|WP-S0|all3|42|channel_robust_alignment_v1|62.27%|60.15%|[0.4533333333333333, 0.7666666666666667, 0.16, 0.7333333333333333, 1.0]|
|WP-S0|slot0|42|matched_control|65.07%|63.78%|[0.38, 0.6133333333333333, 0.46, 0.8133333333333334, 0.9866666666666667]|
|WP-S0|slot0|42|channel_robust_alignment_v1|61.87%|59.57%|[0.26666666666666666, 0.6466666666666666, 0.38666666666666666, 0.7933333333333333, 1.0]|
|WP-S0|slot1|42|matched_control|58.67%|57.04%|[0.7733333333333333, 0.7466666666666667, 0.08, 0.3466666666666667, 0.9866666666666667]|
|WP-S0|slot1|42|channel_robust_alignment_v1|48.93%|48.09%|[0.52, 0.6866666666666666, 0.013333333333333334, 0.24666666666666667, 0.98]|
|WP-S0|slot2|42|matched_control|62.40%|60.40%|[0.6933333333333334, 0.46, 0.10666666666666667, 0.8666666666666667, 0.9933333333333333]|
|WP-S0|slot2|42|channel_robust_alignment_v1|64.13%|61.30%|[0.6533333333333333, 0.6533333333333333, 0.1, 0.8, 1.0]|
|WP-D1|all3|42|matched_control|38.67%|34.29%|[0.04666666666666667, 0.2, 0.0, 0.7666666666666667, 0.92]|
|WP-D1|all3|42|channel_robust_alignment_v1|30.53%|28.70%|[0.0, 0.31333333333333335, 0.0, 0.9133333333333333, 0.3]|
|WP-D1|slot0|42|matched_control|44.40%|39.14%|[0.22, 0.1, 0.02666666666666667, 0.9533333333333334, 0.92]|
|WP-D1|slot0|42|channel_robust_alignment_v1|42.40%|35.49%|[0.06666666666666667, 0.16, 0.013333333333333334, 0.96, 0.92]|
|WP-D1|slot1|42|matched_control|26.80%|29.55%|[0.06666666666666667, 0.38666666666666666, 0.0, 0.4, 0.4866666666666667]|
|WP-D1|slot1|42|channel_robust_alignment_v1|22.13%|23.62%|[0.006666666666666667, 0.43333333333333335, 0.0, 0.62, 0.04666666666666667]|
|WP-D1|slot2|42|matched_control|27.33%|27.53%|[0.3, 0.11333333333333333, 0.013333333333333334, 0.35333333333333333, 0.5866666666666667]|
|WP-D1|slot2|42|channel_robust_alignment_v1|13.07%|14.57%|[0.03333333333333333, 0.14, 0.0, 0.4533333333333333, 0.02666666666666667]|
|PG-S1|all3|42|matched_control|42.00%|35.71%|[0.0, 0.88, 0.8, 0.0]|
|PG-S1|all3|42|channel_robust_alignment_v1|77.00%|76.82%|[0.56, 0.92, 0.64, 0.96]|
|PG-S1|slot0|42|matched_control|44.00%|36.22%|[0.0, 0.96, 0.8, 0.0]|
|PG-S1|slot0|42|channel_robust_alignment_v1|68.00%|66.53%|[0.24, 0.96, 0.8, 0.72]|
|PG-S1|slot1|42|matched_control|36.00%|30.55%|[0.0, 1.0, 0.44, 0.0]|
|PG-S1|slot1|42|channel_robust_alignment_v1|48.00%|46.32%|[0.32, 1.0, 0.4, 0.2]|
|PG-S1|slot2|42|matched_control|18.00%|15.70%|[0.0, 0.12, 0.08, 0.52]|
|PG-S1|slot2|42|channel_robust_alignment_v1|38.00%|36.94%|[0.04, 0.48, 0.16, 0.84]|
|PG-D1|all3|42|matched_control|66.00%|59.84%|[0.0, 0.84, 0.8, 1.0]|
|PG-D1|all3|42|channel_robust_alignment_v1|59.00%|56.12%|[0.0, 0.92, 0.44, 1.0]|
|PG-D1|slot0|42|matched_control|39.00%|36.80%|[0.0, 0.56, 0.12, 0.88]|
|PG-D1|slot0|42|channel_robust_alignment_v1|50.00%|49.51%|[0.0, 1.0, 0.0, 1.0]|
|PG-D1|slot1|42|matched_control|62.00%|56.28%|[0.0, 0.76, 0.8, 0.92]|
|PG-D1|slot1|42|channel_robust_alignment_v1|53.00%|52.83%|[0.0, 1.0, 0.12, 1.0]|
|PG-D1|slot2|42|matched_control|68.00%|62.96%|[0.08, 0.76, 0.88, 1.0]|
|PG-D1|slot2|42|channel_robust_alignment_v1|71.00%|70.42%|[0.32, 1.0, 0.52, 1.0]|

## Predeclared screen gate

Passed: **False**.
Jointly improved conditions: 7/16.
Mean Acc delta: +2.63 pp; mean Macro-F1 delta: +4.63 pp.
Double-cross Macro-F1 delta: -1.89 pp.
