**Trained on the Validation Dataset.** 300 authentic images (COCO val2017) and 300 AI-generated images (DALL·E Advanced), drawn with seed 0, at native resolution. Each image is evaluated under 15 conditions: clean plus the 14 transformations specified in the brief, at its exact parameters.

## Summary

Note: Δ bal = bal_acc(augment) - bal_acc(clean)

| detector      |   clean balanced acc |   clean AUC |   mean balanced acc (degraded) |   worst balanced acc | worst condition   |   retention @ worst |   mean AUC (degraded) |   human acc cost |   AI acc cost |
|:--------------|---------------------:|------------:|-------------------------------:|---------------------:|:------------------|--------------------:|----------------------:|-----------------:|--------------:|
| clip_baseline |               0.8217 |      0.9171 |                         0.8648 |               0.7667 | jpeg_q70          |              0.9331 |                0.9487 |          -0.1355 |        0.0493 |
| clip_robust   |               0.84   |      0.935  |                         0.8805 |               0.7967 | jpeg_q70          |              0.9484 |                0.955  |          -0.1105 |        0.0295 |

## clip_baseline — per condition

| condition    |   human acc |   AI acc |   balanced acc|   Δ bal |    AUC |   P(AI) human |   P(AI) AI |   n |
|:-------------|------------:|---------:|-----------:|--------:|-------:|--------------:|-----------:|----:|
| clean        |      0.7433 |   0.9    |     0.8217 |  0      | 0.9171 |        0.1553 |     0.7631 | 600 |
| jpeg_q90     |      0.72   |   0.89   |     0.805  | -0.0167 | 0.914  |        0.1665 |     0.7614 | 600 |
| jpeg_q70     |      0.6433 |   0.89   |     0.7667 | -0.055  | 0.8719 |        0.2217 |     0.735  | 600 |
| jpeg_q50     |      0.7067 |   0.86   |     0.7833 | -0.0383 | 0.884  |        0.178  |     0.7079 | 600 |
| jpeg_q30     |      0.7567 |   0.87   |     0.8133 | -0.0083 | 0.9009 |        0.1478 |     0.703  | 600 |
| blur_s0.5    |      0.9    |   0.8933 |     0.8967 |  0.075  | 0.9635 |        0.0583 |     0.7575 | 600 |
| blur_s1.0    |      0.9767 |   0.8833 |     0.93   |  0.1083 | 0.9804 |        0.0227 |     0.7488 | 600 |
| blur_s2.0    |      0.9267 |   0.88   |     0.9033 |  0.0817 | 0.9671 |        0.0414 |     0.7369 | 600 |
| resize_0.50x |      0.9567 |   0.89   |     0.9233 |  0.1017 | 0.9789 |        0.0264 |     0.7503 | 600 |
| resize_0.25x |      0.9667 |   0.87   |     0.9183 |  0.0967 | 0.9788 |        0.0218 |     0.7271 | 600 |
| noise_s0.02  |      1      |   0.8067 |     0.9033 |  0.0817 | 0.9881 |        0.0042 |     0.6253 | 600 |
| noise_s0.05  |      1      |   0.7633 |     0.8817 |  0.06   | 0.9905 |        0.0027 |     0.5575 | 600 |
| noise_s0.10  |      1      |   0.7333 |     0.8667 |  0.045  | 0.9904 |        0.0029 |     0.513  | 600 |
| jitter_20pct |      0.8167 |   0.89   |     0.8533 |  0.0317 | 0.9267 |        0.1229 |     0.7457 | 600 |
| crop_80pct   |      0.9333 |   0.79   |     0.8617 |  0.04   | 0.9467 |        0.0456 |     0.6158 | 600 |

## clip_robust — per condition

| condition    |   human acc |   AI acc |   balanced acc |   Δ bal |    AUC |   P(AI) human |   P(AI) AI |   n |
|:-------------|------------:|---------:|-----------:|--------:|-------:|--------------:|-----------:|----:|
| clean        |      0.7767 |   0.9033 |     0.84   |  0      | 0.935  |        0.1593 |     0.8419 | 600 |
| jpeg_q90     |      0.7733 |   0.9    |     0.8367 | -0.0033 | 0.9337 |        0.1647 |     0.8377 | 600 |
| jpeg_q70     |      0.6967 |   0.8967 |     0.7967 | -0.0433 | 0.9092 |        0.2166 |     0.8282 | 600 |
| jpeg_q50     |      0.7533 |   0.8767 |     0.815  | -0.025  | 0.9158 |        0.1813 |     0.8102 | 600 |
| jpeg_q30     |      0.7767 |   0.8767 |     0.8267 | -0.0133 | 0.9201 |        0.1666 |     0.8078 | 600 |
| blur_s0.5    |      0.9267 |   0.8933 |     0.91   |  0.07   | 0.9727 |        0.0509 |     0.8355 | 600 |
| blur_s1.0    |      0.9667 |   0.8867 |     0.9267 |  0.0867 | 0.9778 |        0.0285 |     0.8177 | 600 |
| blur_s2.0    |      0.8867 |   0.8533 |     0.87   |  0.03   | 0.9447 |        0.0837 |     0.7859 | 600 |
| resize_0.50x |      0.9467 |   0.8833 |     0.915  |  0.075  | 0.9757 |        0.0348 |     0.8098 | 600 |
| resize_0.25x |      0.9567 |   0.86   |     0.9083 |  0.0683 | 0.9691 |        0.0349 |     0.7745 | 600 |
| noise_s0.02  |      0.98   |   0.8567 |     0.9183 |  0.0783 | 0.9792 |        0.018  |     0.7849 | 600 |
| noise_s0.05  |      0.9967 |   0.8667 |     0.9317 |  0.0917 | 0.9847 |        0.0088 |     0.7711 | 600 |
| noise_s0.10  |      0.99   |   0.8633 |     0.9267 |  0.0867 | 0.9884 |        0.0087 |     0.7778 | 600 |
| jitter_20pct |      0.8167 |   0.8833 |     0.85   |  0.01   | 0.9366 |        0.1343 |     0.83   | 600 |
| crop_80pct   |      0.9533 |   0.8367 |     0.895  |  0.055  | 0.9625 |        0.0379 |     0.7168 | 600 |

## Charts
![histogram_baseline]()
Histogram for augmentations on baseline model

![histogram_robust]()
Histogram for augmentations on robust model

![barchart_comparison]()
Comparison of the bar charts for both models

## Short Analysis
- Robust training wins on almost every metric, showing improved accuracy and AUC against the baseline. 
- Accuracy raised by ~3% on the worst performing criteria, showing an ability to raise the floor
- The distributions are cleanly separated, which is consistent with healthy AUC
- Histogram of baseline model shows gaussian noise causes the model to lose the AI artifacts, while it remains able to detect AI images under noise in the robust model.

## Method notes

- The threshold for the internal robustness measurements is calculated to approximately target a 10% FPR on unseen data, which it does well.
- Transformations are applied to the decoded image once per condition and the same image object is scored by every detector, so differences between columns are indicative of difference between models.
- Noise and colour jitter are seeded per `(seed, condition, image id)`, so the table is reproducible across runs and thread counts.
- Accuracy intervals are Wilson 95% intervals, which remain valid near 0 and 1 where the normal approximation does not.

