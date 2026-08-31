# Brief Error Analysis Note for CLIP-VIT

![image](https://github.com/KurumiWalnut/starting5/blob/main/images/fig_error_analysis__clip_robust__clean-2.png)

From the model provided, we analysed the validation dataset and extracted the 3 which the the AI was most confidently wrong in both directions (FP and FN), and we found the results are consistent with our hypothesis.

## False Positives
The three worst false positives are a silhouetted figure on a beach and two clocks. What makes the images likely detected to be "AI" is the high dynamic range, strong symmetry, ornate repeating detail, or dramatic lighting. These are precisely the qualities diffusion models overproduce, so the detector has learned to associate them with "AI.". In addition, clocks may be more easily recognised as AI due to the numerous fine details. For a model trained on semantic content, potential inconsistencies on objects like clock hands caused by issues such as compression can make the AI confidently incorrect in its opinion the image is AI.

## False Negatives
All 3 images with the largest false negative are images in an out-of-distribution style: Anime images. The detector's concept of "AI" is built almost entirely from photorealistic generator outputs. Anime / illustration / cartoon renders live in a different region of CLIP space entirely, and nothing in training taught the probe that this region is also "AI." The training set must span generator style, not just generator identity such as anime/illustration diffusion outputs, or meme-format AI images to reduce such false negatives.


# Brief Error Analysis Note for EfficientNet

<img width="1044" height="790" alt="image" src="https://github.com/user-attachments/assets/03fa61f9-5433-48f1-a7c6-3d3f5777b5b8" />
Representative false positives / false negatives, clean condition.

<img width="1046" height="787" alt="image" src="https://github.com/user-attachments/assets/4b66a4fc-08df-496d-af73-3602287314cf" />
Representative false positives / false negatives, noise_s0.10 condition.

## Insight
**The false negatives are systematically stylized/illustrated content, not photorealistic near-misses.** The same three images — a costumed-animal parade photo and two Simpsons-style Halloween illustrations — appear as confident false negatives (P(AI)≈0.000) in *both* the clean and noise_s0.10 error grids. Missed even with zero corruption, this is a blind spot in what the model learned to recognize as "AI-generated," not a robustness gap.
