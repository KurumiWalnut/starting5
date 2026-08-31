# TextureCrop for CLIP-VIT — a small aside for the robustness writeup

## Reasoning behind implementation test
The idea behind *TextureCrop* is to preserve the fine high-frequency generator fingerprint so it survives rescaling. 
The CLIP branch preprocesses each image with *TextureCrop*, which research shows can improve AUC by 6.1% ([Konstantinidou et al., WACV 2025](https://arxiv.org/abs/2407.15500)): instead of resizing the whole image to 224, it keeps the `top-k` highest-entropy 224×224 windows at native resolution, encodes each with CLIP and averages the features to feed the linear probe.

In tests against in-distribution data, texture-cropping performed similarly to center-cropping, with AUC of > 0.97 and accuracy > 0.95. Hence, we decided to test it against the validation dataset to check if it holds up in the real world data. Below holds the collected data.


## Summary

| detector      |   clean balanced acc |   clean AUC |   mean balanced acc (degraded) |   worst balanced acc | worst condition   |   retention @ worst |   mean AUC (degraded) |   human acc cost |   AI acc cost | dominant failure            |
|:--------------|---------------------:|------------:|-------------------------------:|---------------------:|:------------------|--------------------:|----------------------:|-----------------:|--------------:|:----------------------------|
| clip_baseline |               0.695  |      0.8644 |                         0.6677 |               0.4817 | blur_s2.0         |              0.693  |                0.8478 |            0.021 |        0.0336 | misses AI images            |
| clip_robust   |               0.7367 |      0.8822 |                         0.7364 |               0.6267 | blur_s2.0         |              0.8507 |                0.8671 |            0.029 |       -0.0286 | over-flags authentic images |

## clip_baseline — per condition

| condition    |   human acc |   AI acc |   balanced |   Δ bal |    AUC |   P(AI) human |   P(AI) AI |   n |
|:-------------|------------:|---------:|-----------:|--------:|-------:|--------------:|-----------:|----:|
| clean        |      0.95   |   0.44   |     0.695  |  0      | 0.8644 |        0.0841 |     0.4201 | 600 |
| jpeg_q90     |      0.9433 |   0.5833 |     0.7633 |  0.0683 | 0.9045 |        0.0823 |     0.5352 | 600 |
| jpeg_q70     |      0.9733 |   0.43   |     0.7017 |  0.0067 | 0.9097 |        0.0512 |     0.431  | 600 |
| jpeg_q50     |      0.9867 |   0.27   |     0.6283 | -0.0667 | 0.91   |        0.0305 |     0.3012 | 600 |
| jpeg_q30     |      0.9633 |   0.2167 |     0.59   | -0.105  | 0.8652 |        0.0484 |     0.2673 | 600 |
| blur_s0.5    |      0.9067 |   0.52   |     0.7133 |  0.0183 | 0.8477 |        0.1226 |     0.4774 | 600 |
| blur_s1.0    |      0.9233 |   0.45   |     0.6867 | -0.0083 | 0.8608 |        0.0953 |     0.4226 | 600 |
| blur_s2.0    |      0.7933 |   0.17   |     0.4817 | -0.2133 | 0.5947 |        0.24   |     0.2665 | 600 |
| resize_0.50x |      0.93   |   0.4433 |     0.6867 | -0.0083 | 0.8435 |        0.101  |     0.4011 | 600 |
| resize_0.25x |      0.9067 |   0.3233 |     0.615  | -0.08   | 0.7887 |        0.1309 |     0.3423 | 600 |
| noise_s0.02  |      0.9367 |   0.6733 |     0.805  |  0.11   | 0.9255 |        0.0833 |     0.6005 | 600 |
| noise_s0.05  |      0.94   |   0.4767 |     0.7083 |  0.0133 | 0.8907 |        0.0841 |     0.4555 | 600 |
| noise_s0.10  |      0.92   |   0.1933 |     0.5567 | -0.1383 | 0.7694 |        0.1011 |     0.2517 | 600 |
| jitter_20pct |      0.91   |   0.52   |     0.715  |  0.02   | 0.8683 |        0.11   |     0.4978 | 600 |
| crop_80pct   |      0.9733 |   0.42   |     0.6967 |  0.0017 | 0.8902 |        0.0564 |     0.418  | 600 |

## clip_robust — per condition

| condition    |   human acc |   AI acc |   balanced |   Δ bal |    AUC |   P(AI) human |   P(AI) AI |   n |
|:-------------|------------:|---------:|-----------:|--------:|-------:|--------------:|-----------:|----:|
| clean        |      0.95   |   0.5233 |     0.7367 |  0      | 0.8822 |        0.1222 |     0.6169 | 600 |
| jpeg_q90     |      0.9467 |   0.5733 |     0.76   |  0.0233 | 0.9054 |        0.1138 |     0.6654 | 600 |
| jpeg_q70     |      0.9467 |   0.5467 |     0.7467 |  0.01   | 0.9053 |        0.1025 |     0.6285 | 600 |
| jpeg_q50     |      0.9333 |   0.6367 |     0.785  |  0.0483 | 0.9063 |        0.1265 |     0.7066 | 600 |
| jpeg_q30     |      0.9167 |   0.6467 |     0.7817 |  0.045  | 0.8863 |        0.1515 |     0.7097 | 600 |
| blur_s0.5    |      0.93   |   0.52   |     0.725  | -0.0117 | 0.8669 |        0.1432 |     0.6211 | 600 |
| blur_s1.0    |      0.9233 |   0.4867 |     0.705  | -0.0317 | 0.8536 |        0.1472 |     0.602  | 600 |
| blur_s2.0    |      0.8167 |   0.4367 |     0.6267 | -0.11   | 0.6951 |        0.3231 |     0.5558 | 600 |
| resize_0.50x |      0.9033 |   0.57   |     0.7367 |  0      | 0.8471 |        0.1886 |     0.6571 | 600 |
| resize_0.25x |      0.88   |   0.54   |     0.71   | -0.0267 | 0.8015 |        0.2538 |     0.6343 | 600 |
| noise_s0.02  |      0.9267 |   0.7133 |     0.82   |  0.0833 | 0.9324 |        0.1322 |     0.7773 | 600 |
| noise_s0.05  |      0.9467 |   0.6233 |     0.785  |  0.0483 | 0.9161 |        0.1196 |     0.705  | 600 |
| noise_s0.10  |      0.93   |   0.4167 |     0.6733 | -0.0633 | 0.8473 |        0.1605 |     0.5727 | 600 |
| jitter_20pct |      0.9167 |   0.5433 |     0.73   | -0.0067 | 0.8756 |        0.1428 |     0.6463 | 600 |
| crop_80pct   |      0.9767 |   0.4733 |     0.725  | -0.0117 | 0.9008 |        0.08   |     0.5785 | 600 |

## Similar in-domain, poor here — a possible analysis as to why

**High AI Recall.** The model only catches 44% of AI for the baseline model and 52% for clean. This suggests that the AI model cannot tell apart real and fake - akin to coin flipping. 

**Texture might not be the best solution for Dall-e in particular.**Dall-E images tend to be high resolution and glossy, hence it misses a lot of the high frequency and compression artifacts from the original training set. 

**Augments might destroy texture.** The worst performers for both models are high blur and high noise. This suggests an issue with texturecropping itself - as it selects the highest entropy windows, gaussian blur would remove the high-frequency content, leaving little to read. 

### Testing texture-crop on test set against 

| detector      | condition   | domain      | human  | AI     | bal    | AUC    |
|---------------|-------------|-------------|--------|--------|--------|--------|
| clip_baseline | clean       | SID (ID)    | 0.9867 | 0.9933 | 0.9900 | 0.9992 |
|               |             | COCO/DALLE  | 0.9500 | 0.4400 | 0.6950 | 0.8644 |
| clip_baseline | blur_s2.0   | SID (ID)    | 0.9933 | 0.8067 | 0.9000 | 0.9912 |
|               |             | COCO/DALLE  | 0.7933 | 0.1700 | 0.4817 | 0.5947 |
| clip_baseline | noise_s0.10 | SID (ID)    | 0.9967 | 0.6533 | 0.8250 | 0.9847 |
|               |             | COCO/DALLE  | 0.9200 | 0.1933 | 0.5567 | 0.7694 |
| clip_robust   | clean       | SID (ID)    | 0.9833 | 0.9967 | 0.9900 | 0.9991 |
|               |             | COCO/DALLE  | 0.9500 | 0.5233 | 0.7367 | 0.8822 |
| clip_robust   | blur_s2.0   | SID (ID)    | 0.9433 | 0.9833 | 0.9633 | 0.9942 |
|               |             | COCO/DALLE  | 0.8167 | 0.4367 | 0.6267 | 0.6951 |
| clip_robust   | noise_s0.10 | SID (ID)    | 0.9133 | 0.9900 | 0.9517 | 0.9937 |
|               |             | COCO/DALLE  | 0.9300 | 0.4167 | 0.6733 | 0.8473 |


## Conclusion
While the model still looked strong on data from the test set, it collapses on real-world data, particularly dall-e images. With texture cropping being heavily impacted by some of the augmentations, it might need to be reconsidered before future implementation.