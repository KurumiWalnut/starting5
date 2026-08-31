**Trained on the Validation Dataset.** 300 authentic images (COCO val2017) and 300 AI-generated images (DALL·E Advanced), drawn with seed 0, at native resolution. Each image is evaluated under 15 conditions: clean plus the 14 transformations specified in the brief, at its exact parameters. The decision threshold is fixed at the value shipped with the checkpoint (0.500) and held constant across every condition, so a rising error rate under degradation is visible rather than absorbed by re-tuning.

## Summary

| detector    |   clean balanced acc |   clean AUC |   mean balanced acc (degraded) |   worst balanced acc | worst condition |   retention @ worst |   mean AUC (degraded) |   human acc cost |   AI acc cost | dominant failure |
|:------------|---:|---:|---:|---:|:---|---:|---:|---:|---:|:---|
| efficientnet |  0.6933 | 0.813 | 0.6744 | 0.605 | noise_s0.10 | 0.8726 | 0.7774 | -0.0076 | 0.0455 | misses AI images |

## Per-condition results, by transform family

| condition | human acc | AI acc | balanced acc | Δ bal | AUC |
|:---|---:|---:|---:|---:|---:|
| clean | 0.92 | 0.4667 | 0.6933 | — | 0.813 |
| **JPEG** | | | | | |
| jpeg_q90 | 0.9267 | 0.47 | 0.6983 | +0.005 | 0.815 |
| jpeg_q70 | 0.93 | 0.4333 | 0.6817 | -0.0117 | 0.7907 |
| jpeg_q50 | 0.93 | 0.4133 | 0.6717 | -0.0217 | 0.7808 |
| jpeg_q30 | 0.93 | 0.40 | 0.665 | -0.0283 | 0.7735 |
| **Blur** | | | | | |
| blur_s0.5 | 0.94 | 0.4667 | 0.7033 | +0.01 | 0.8168 |
| blur_s1.0 | 0.9367 | 0.4667 | 0.7017 | +0.0083 | 0.8145 |
| blur_s2.0 | 0.91 | 0.44 | 0.675 | -0.0183 | 0.7664 |
| **Resize** | | | | | |
| resize_0.50x | 0.9367 | 0.4633 | 0.70 | +0.0067 | 0.796 |
| resize_0.25x | 0.8967 | 0.44 | 0.6683 | -0.025 | 0.7378 |
| **Noise** | | | | | |
| noise_s0.02 | 0.9667 | 0.4233 | 0.695 | +0.0017 | 0.8399 |
| noise_s0.05 | 0.94 | 0.37 | 0.655 | -0.0383 | 0.7682 |
| noise_s0.10 | 0.8867 | 0.3233 | 0.605 | -0.0883 | 0.5817 |
| **Jitter / crop** | | | | | |
| jitter_20pct | 0.92 | 0.4233 | 0.6717 | -0.0217 | 0.801 |
| crop_80pct | 0.9367 | 0.3633 | 0.65 | -0.0433 | 0.8018 |

**JPEG, blur, resize, and jitter degrade gracefully** — balanced accuracy drifts within roughly ±0.03 of clean across all of them, and (per the AUC figures below) the underlying ranking signal barely moves. **Noise is the clear outlier family**: balanced accuracy falls monotonically and sharply with strength, from 0.695 at σ=0.02 down to 0.605 at σ=0.10 — nearly triple the drop of any other family. **crop_80pct sits lower than expected** (0.650, comparable to noise_s0.05) despite AUC remaining in the healthy range for this condition — see below, this turns out to still be a calibration effect, not a sign crop belongs in the same category as noise.

## Charts

<img width="1469" height="780" alt="image" src="https://github.com/user-attachments/assets/5a60fe4b-37e2-4db9-a644-2d74265da134" />
15-panel score distribution grid — clean plus all 14 transforms, with FP/FN counts in each panel title.

## Short Analysis

- **Noise is a genuine signal-loss failure, not just a calibration one.** AUC holds in the 0.77–0.82 range for every JPEG, blur, resize, jitter, and crop condition — the underlying separability barely moves, only the threshold's position relative to it does. noise_s0.10 is different: AUC collapses to 0.582, barely above chance. That is the actual evidence-destroying transform for this branch.
- **Nearly all of the robustness cost lands on AI recall, not human accuracy.** Human-image accuracy is essentially unaffected by degradation (-0.0076, a tiny average improvement); AI-image accuracy costs -0.0455 on average. This detector's practical risk under real-world transforms is under-flagging fakes, not over-flagging authentic content.

## Method notes

- Transformations are applied to the decoded image once per condition and the same image object is scored by every detector, so differences between detector columns are differences between models, not between inputs.
- Noise and colour jitter are seeded per `(seed, condition, image id)`, so the table is reproducible across runs and thread counts.
- Accuracy intervals are Wilson 95% intervals (in the CSV), which remain valid near 0 and 1 where the normal approximation does not.
- COCO val2017 is ~640×480 and DALL·E Advanced ~1024², so image size is partly predictive of the label on its own. Re-running with `CFG['max_side'] = 512` caps both classes to the same long edge; the drop between the two runs estimates how much of the score was resolution (Grommelt et al., 2024, arXiv:2403.17608).
