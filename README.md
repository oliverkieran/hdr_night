# Deep Learning for Smartphone Photo Processing and Enhancement at Night

<img src="webpage/huge-combi2.jpeg" alt="DPED_results" width="100%"/>

### 1. Overview 

This repository provides the code belonging to the paper "Deep Learning for Smartphone Photo Processing and Enhancement at Night" by Oliver Lehmann and Samuel Käser.

---
### 2. Prerequisites

- Python: numpy, scipy, imageio and pillow packages
- [TensorFlow 1.15.0](https://www.tensorflow.org/install/) + [CUDA cuDNN](https://developer.nvidia.com/cudnn)
- GPU for training (e.g., Nvidia GeForce GTX 1080)

[[back]](#contents)
<br/>

---
### 3. Folder Structure
>```DPED/```                &nbsp; - &nbsp; directory with the DPED code used for training on the daytime dataset<br/>
>```DPED2/```               &nbsp; - &nbsp; directory with the attempted Tensorflow 2 translation of the DPED code<br/>
>```DPED_night/```          &nbsp; - &nbsp; directory with the DPED code used for training on the SKOL dataset<br/>
>```learned_isp/```         &nbsp; - &nbsp; directory with the PUNET code used for training on the daytime dataset<br/>
>```learned_isp_night/```   &nbsp; - &nbsp; directory with the DPED code used for training on the SKOL dataset<br/>
>```raw_images/```          &nbsp; - &nbsp; directory where we stored the daytime dataset<br/>
>```../crops/```            &nbsp; - &nbsp; directory where we stored the crops of the SKOL dataset<br/>
>```../dataset_raw/```      &nbsp; - &nbsp; directory where we stored the full-resolution images of the SKOL dataset<br/>
>```webpage/```             &nbsp; - &nbsp; directory where we store images used on this webpage<br/>

>```calculate_lpips.py```   &nbsp; - &nbsp; Python file to calculate LPIPS of already generated images<br/>
>```calculate_scores.py```  &nbsp; - &nbsp; Python file to calculate PSNR and MS-SSIM of already generated images<br/>
>```create_baseline.py```   &nbsp; - &nbsp; De-Bayer python file to create RGB photos out of Bayer Raw photos<br/> 
>```visualize_crops.py```   &nbsp; - &nbsp; Python file that de-bayers a Bayer Raw photo and converts it to a RGB photo.<br/>

### 4. Acknowledgments


We want to thank our advisors, Andrey Ignatov and Radu Timofte, who gave us the chance to take a deep dive into the world of deep learning for smartphone image enhancement. 
They provided the whole camera equipment as well as the two deep neural networks that we have used in this thesis. 
Thanks again for all the knowledge and support which enabled this project.

---
### 5. Contact

Please contact [**Oliver Lehmann**](mailto:ollehman@ethz.ch) or [**Samuel Käser**](mailto:skaeser@ethz.ch) for more information. <br/>
