# CRA-v2A-U-fixedK development system-evaluation report

> **NOT PROMOTED — the registered development performance gate failed.**
> Integrity checks passed, but this report does not support a superiority or high-performance claim.

## Evidence boundary

|Source|Tier|Role|Split|Clean rows|Noise rows|
|---|---|---|---|---|---|
|v2a_seed42_development|development|laboratory|target_dev_only|32|0|
|v4_frozen_reference_development|development|laboratory|target_dev_only|32|0|

- Development rows are model-development evidence and must not be described as untouched confirmation.
- External rows document performance on a separately named public/external dataset; they do not replace an internal untouched final set.
- No target-final evidence was read; final superiority or deployment claims are unsupported by this report.
- A blank SD means that only one independent model seed was available; it is not reported as zero variability.

## Coverage

|Tier|Required cell|Covered|
|---|---|---|
|development|3-to-3 same-condition|True|
|development|3-to-3 double-cross|True|
|development|3-to-1 slot0|True|
|development|3-to-1 slot1|True|
|development|3-to-1 slot2|True|
|development|noise SNR clean|False|
|development|noise SNR 20|False|
|development|noise SNR 10|False|
|development|noise SNR 0|False|
|external|3-to-3 same-condition|False|
|external|3-to-3 double-cross|False|
|external|3-to-1 slot0|False|
|external|3-to-1 slot1|False|
|external|3-to-1 slot2|False|
|external|noise SNR clean|False|
|external|noise SNR 20|False|
|external|noise SNR 10|False|
|external|noise SNR 0|False|
|final|3-to-3 same-condition|False|
|final|3-to-3 double-cross|False|
|final|3-to-1 slot0|False|
|final|3-to-1 slot1|False|
|final|3-to-1 slot2|False|
|final|noise SNR clean|False|
|final|noise SNR 20|False|
|final|noise SNR 10|False|
|final|noise SNR 0|False|

## Clean Accuracy and Macro-F1

|Tier|Dataset|Task|Scenario|View|Variant|Accuracy %, mean ± SD|Macro-F1 %, mean ± SD|Zero-recall runs|
|---|---|---|---|---|---|---|---|---|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot0|channel_robust_alignment_v1|50.00 (n=1; SD NA)|49.51 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot0|channel_robust_alignment_v2a|39.00 (n=1; SD NA)|35.17 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot0|common_anchor_control|42.00 (n=1; SD NA)|39.18 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot0|matched_control|39.00 (n=1; SD NA)|36.80 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot1|channel_robust_alignment_v1|53.00 (n=1; SD NA)|52.83 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot1|channel_robust_alignment_v2a|57.00 (n=1; SD NA)|55.25 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot1|common_anchor_control|59.00 (n=1; SD NA)|58.05 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot1|matched_control|62.00 (n=1; SD NA)|56.28 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot2|channel_robust_alignment_v1|71.00 (n=1; SD NA)|70.42 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot2|channel_robust_alignment_v2a|72.00 (n=1; SD NA)|71.31 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot2|common_anchor_control|72.00 (n=1; SD NA)|71.31 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-1 / double-cross|slot2|matched_control|68.00 (n=1; SD NA)|62.96 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-3 / double-cross|all3|channel_robust_alignment_v1|59.00 (n=1; SD NA)|56.12 (n=1; SD NA)|1/1|
|development|gearbox|PG-D1|3-to-3 / double-cross|all3|channel_robust_alignment_v2a|62.00 (n=1; SD NA)|58.86 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-3 / double-cross|all3|common_anchor_control|62.00 (n=1; SD NA)|58.86 (n=1; SD NA)|0/1|
|development|gearbox|PG-D1|3-to-3 / double-cross|all3|matched_control|66.00 (n=1; SD NA)|59.84 (n=1; SD NA)|1/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot0|channel_robust_alignment_v1|68.00 (n=1; SD NA)|66.53 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot0|channel_robust_alignment_v2a|80.00 (n=1; SD NA)|79.65 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot0|common_anchor_control|76.00 (n=1; SD NA)|76.18 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot0|matched_control|44.00 (n=1; SD NA)|36.22 (n=1; SD NA)|1/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot1|channel_robust_alignment_v1|48.00 (n=1; SD NA)|46.32 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot1|channel_robust_alignment_v2a|52.00 (n=1; SD NA)|50.70 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot1|common_anchor_control|50.00 (n=1; SD NA)|47.51 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot1|matched_control|36.00 (n=1; SD NA)|30.55 (n=1; SD NA)|1/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot2|channel_robust_alignment_v1|38.00 (n=1; SD NA)|36.94 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot2|channel_robust_alignment_v2a|33.00 (n=1; SD NA)|33.50 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot2|common_anchor_control|28.00 (n=1; SD NA)|29.96 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-1 / same-condition cross-sensor|slot2|matched_control|18.00 (n=1; SD NA)|15.70 (n=1; SD NA)|1/1|
|development|gearbox|PG-S1|3-to-3 / same-condition cross-sensor|all3|channel_robust_alignment_v1|77.00 (n=1; SD NA)|76.82 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-3 / same-condition cross-sensor|all3|channel_robust_alignment_v2a|71.00 (n=1; SD NA)|72.07 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-3 / same-condition cross-sensor|all3|common_anchor_control|71.00 (n=1; SD NA)|72.07 (n=1; SD NA)|0/1|
|development|gearbox|PG-S1|3-to-3 / same-condition cross-sensor|all3|matched_control|42.00 (n=1; SD NA)|35.71 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot0|channel_robust_alignment_v1|42.40 (n=1; SD NA)|35.49 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot0|channel_robust_alignment_v2a|33.07 (n=1; SD NA)|30.18 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot0|common_anchor_control|44.53 (n=1; SD NA)|38.62 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot0|matched_control|44.40 (n=1; SD NA)|39.14 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot1|channel_robust_alignment_v1|22.13 (n=1; SD NA)|23.62 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot1|channel_robust_alignment_v2a|24.80 (n=1; SD NA)|25.85 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot1|common_anchor_control|21.33 (n=1; SD NA)|23.67 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot1|matched_control|26.80 (n=1; SD NA)|29.55 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot2|channel_robust_alignment_v1|13.07 (n=1; SD NA)|14.57 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot2|channel_robust_alignment_v2a|14.67 (n=1; SD NA)|15.49 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot2|common_anchor_control|18.67 (n=1; SD NA)|19.50 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-1 / double-cross|slot2|matched_control|27.33 (n=1; SD NA)|27.53 (n=1; SD NA)|0/1|
|development|waterpump|WP-D1|3-to-3 / double-cross|all3|channel_robust_alignment_v1|30.53 (n=1; SD NA)|28.70 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-3 / double-cross|all3|channel_robust_alignment_v2a|33.20 (n=1; SD NA)|30.97 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-3 / double-cross|all3|common_anchor_control|31.60 (n=1; SD NA)|29.71 (n=1; SD NA)|1/1|
|development|waterpump|WP-D1|3-to-3 / double-cross|all3|matched_control|38.67 (n=1; SD NA)|34.29 (n=1; SD NA)|1/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot0|channel_robust_alignment_v1|61.87 (n=1; SD NA)|59.57 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot0|channel_robust_alignment_v2a|66.80 (n=1; SD NA)|64.78 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot0|common_anchor_control|65.47 (n=1; SD NA)|63.58 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot0|matched_control|65.07 (n=1; SD NA)|63.78 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot1|channel_robust_alignment_v1|48.93 (n=1; SD NA)|48.09 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot1|channel_robust_alignment_v2a|49.20 (n=1; SD NA)|49.71 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot1|common_anchor_control|52.93 (n=1; SD NA)|51.98 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot1|matched_control|58.67 (n=1; SD NA)|57.04 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot2|channel_robust_alignment_v1|64.13 (n=1; SD NA)|61.30 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot2|channel_robust_alignment_v2a|66.53 (n=1; SD NA)|62.97 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot2|common_anchor_control|64.93 (n=1; SD NA)|61.90 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-1 / same-condition cross-sensor|slot2|matched_control|62.40 (n=1; SD NA)|60.40 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-3 / same-condition cross-sensor|all3|channel_robust_alignment_v1|62.27 (n=1; SD NA)|60.15 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-3 / same-condition cross-sensor|all3|channel_robust_alignment_v2a|63.20 (n=1; SD NA)|61.02 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-3 / same-condition cross-sensor|all3|common_anchor_control|63.47 (n=1; SD NA)|61.38 (n=1; SD NA)|0/1|
|development|waterpump|WP-S0|3-to-3 / same-condition cross-sensor|all3|matched_control|68.93 (n=1; SD NA)|67.13 (n=1; SD NA)|0/1|

## Paired candidate-control changes

|Tier|Comparison|Channel|Family|Pairs|Accuracy delta pp|Macro-F1 delta pp|Joint gains|New zero-recall|
|---|---|---|---|---|---|---|---|---|
|development|frozen_v4_reference|3-to-1|double-cross|6|-4.50|-3.17|1|1|
|development|frozen_v4_reference|3-to-1|same-condition cross-sensor|6|+10.57|+12.94|5|0|
|development|frozen_v4_reference|3-to-3|double-cross|2|-4.73|-2.15|0|0|
|development|frozen_v4_reference|3-to-3|same-condition cross-sensor|2|+11.63|+15.13|1|0|
|development|paired_causal|3-to-1|double-cross|6|-2.83|-2.85|1|2|
|development|paired_causal|3-to-1|same-condition cross-sensor|6|+1.70|+1.70|5|0|
|development|paired_causal|3-to-3|double-cross|2|+0.80|+0.63|1|0|
|development|paired_causal|3-to-3|same-condition cross-sensor|2|-0.13|-0.18|0|0|

## AWGN robustness

Noise results are pending; the noise figure is a placeholder.

## Frozen campaign gates

|Source|Gate|Passed|Joint|Mean Acc delta pp|Mean F1 delta pp|Double-cross F1 delta pp|
|---|---|---|---|---|---|---|
|v2a_seed42_development|causal_comparison.expansion_gate|False|7|-0.34|-0.37|-1.98|
|v2a_seed42_development|formal_comparison.historical_threshold_diagnostic|False|7|+3.14|+5.29|-2.91|
|v4_frozen_reference_development|screen_gate|False|7|+2.63|+4.63|-1.89|

## Figure files

- `figure_clean_macro_f1.png`
- `figure_paired_macro_f1_delta.png`
- `figure_noise_robustness.png`

All numeric source rows, group summaries, pairwise deltas, coverage checks, and hashes are available in the adjacent CSV and receipt files.
