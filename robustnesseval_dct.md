# Test Set up
To validate the DCT branch beyond its training distribution, we evaluated the trained (pre-augmentation) checkpoint on a held-out, balanced sample from the official WildFake validation benchmark: 300 real images (val2017) and 300 AI-generated images (DALL-E 3). The validation dataset is different from that of the training data. 

# Results 

=== dct_branch — per condition (shipped threshold) ===
              human acc  AI acc  balanced   Δ bal     AUC  P(AI) human  P(AI) AI    n
condition                                                                            
clean            0.9867  0.0733    0.5300  0.0000  0.5544       0.0176    0.0777  600
jpeg_q90         0.9867  0.0733    0.5300  0.0000  0.5549       0.0177    0.0775  600
jpeg_q70         0.9867  0.0767    0.5317  0.0017  0.5627       0.0164    0.0805  600
jpeg_q50         0.9900  0.0667    0.5283 -0.0017  0.5682       0.0166    0.0757  600
jpeg_q30         0.9900  0.0667    0.5283 -0.0017  0.5225       0.0136    0.0676  600
blur_s0.5        0.9867  0.0633    0.5250 -0.0050  0.4792       0.0184    0.0668  600
blur_s1.0        0.9767  0.0433    0.5100 -0.0200  0.2901       0.0337    0.0501  600
blur_s2.0        0.9467  0.0300    0.4883 -0.0417  0.1701       0.0869    0.0373  600
resize_0.50x     0.9733  0.0367    0.5050 -0.0250  0.2470       0.0451    0.0442  600
resize_0.25x     0.9433  0.0233    0.4833 -0.0467  0.1748       0.0976    0.0402  600
noise_s0.02      0.9900  0.0767    0.5333  0.0033  0.5791       0.0106    0.0780  600
noise_s0.05      1.0000  0.0700    0.5350  0.0050  0.6453       0.0038    0.0757  600
noise_s0.10      1.0000  0.0500    0.5250 -0.0050  0.7154       0.0016    0.0500  600
jitter_20pct     0.9833  0.0700    0.5267 -0.0033  0.5446       0.0196    0.0778  600
crop_80pct       0.9567  0.0300    0.4933 -0.0367  0.2949       0.0560    0.0312  600

