## DPED Network with SKOL Dataset

#### 1. Overview 

The provided code implements the modified DPED model used in the paper "Deep Learning for Smartphone Photo Processing and Enhancement at Night".
This is the model that was used to train with the SKOL dataset, provided by Andrey Ignatov and Radu Timofte.


#### 2. Prerequisites

- Python + Pillow, scipy, numpy, imageio packages
- [TensorFlow 1.x / 2.x](https://www.tensorflow.org/install/) + [CUDA CuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU


#### 3. First steps

- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing)</sup> and put it into `vgg_pretrained/` folder
- Please contact either [**Oliver Lehmann**](mailto:ollehman@ethz.ch) or [**Samuel Käser**](mailto:skaeser@ethz.ch) to download the dataset used for training and testing thsi model.

<br/>

#### 4. Train the model

##### Train the generator:
```bash
python train_generative_part.py
```

##### Train the full model (generator + discriminator):
```bash
python train_model.py
```

Optional parameters and their default values:

>```batch_size```: **```32```** &nbsp; - &nbsp; batch size [smaller values can lead to unstable training] <br/>
>```train_size```: **```5000```** &nbsp; - &nbsp; the number of training patches randomly loaded each ```eval_step``` iterations <br/>
>```learning_rate```: **```5e-5```** &nbsp; - &nbsp; learning rate <br/>
>```num_train_iters```: **```20000```** &nbsp; - &nbsp; the number of training iterations <br/>
> ```restore_iter```: **```0```** &nbsp; - &nbsp; iteration to restore <br/>
> ```loss```: **```dped```** &nbsp; - &nbsp; which loss the model should be trained on: [dped, pynet] <br/>
>```dped_dir```: **```../../crops/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; path to the pre-trained VGG-19 network <br/>
>```run_id```: **```default_run```** &nbsp; - &nbsp; the name of the folder where parameters and results get saved <br/>
> ```from_model```: **```None```** &nbsp; - &nbsp; this has the same purpose as run_id and is only used, if you want to restore a previous model and save the future checkpoints in a different directory (run_id). <br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; each ```eval_step``` iterations the model is saved and the training data is reloaded <br/>
> ```input_images```: **```1```** &nbsp; - &nbsp; whether to take 1 or 3 images as input <br/>

Example:

```bash
python train_model.py num_train_iters=30000 loss=pynet run_id=pynet_loss input_images=1
```

<br/>

#### 5. Test the obtained models

```bash
python test_model.py
```

Optional parameters:

>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>
>```test_dir```: **```full_res_test/```**  &nbsp; - &nbsp; path to the folder with test dataset <br/>
>```iteration```: **```20000```**  &nbsp; - &nbsp; restore model at this iteration <br/>
>```resolution```: **```full_res```**,**```phone```** &nbsp; - &nbsp; the resolution of the test images <br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```run_id```: **```default_run```** &nbsp; - &nbsp; the name of the folder where parameters and results get saved <br/>
> ```input_images```: **```1```** &nbsp; - &nbsp; whether to take 1 or 3 images as input <br/>

Example:

```bash
python test_model.py  dped_dir=../../dataset_raw/ test_dir=MTK_RAW/ iteration=33000 run_id=pynet_loss input_images=3
```

<br/>

#### 6. Folder structure

>```logs_single/```             &nbsp; - &nbsp; logs that are saved during the training process for single image input and can be used for tensorboard visualization<br/>
>```logs_multi/```              &nbsp; - &nbsp; logs that are saved during the training process for multiple images input and can be used for tensorboard visualization<br/>
>```models_single/```           &nbsp; - &nbsp; models that are saved during the training process for single image input <br/>
>```models_multi/```            &nbsp; - &nbsp; models that are saved during the training process for multiple images input <br/>
>```vgg-pretrained/```          &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>
>```visual_results_single/```   &nbsp; - &nbsp; processed [enhanced] test/full-resolution images with single image input <br/>
>```visual_results_multi/```    &nbsp; - &nbsp; processed [enhanced] test/full-resolution images with multiple images input <br/>

>```evaluate_scores.py```       &nbsp; - &nbsp; python script that calculates PSNR and MS-SSIM scores <br/>
>```load_dataset.py```          &nbsp; - &nbsp; python script that loads training data <br/>
>```models.py```                &nbsp; - &nbsp; architecture of the image enhancement [resnet] and adversarial networks <br/>
>```models_lastlayer.py```      &nbsp; - &nbsp; same as models.py, but with the upscaling layer in the last layer <br/>
>```train_model.py```           &nbsp; - &nbsp; implementation of the training procedure <br/>
>```train_generative_part.py``` &nbsp; - &nbsp; implementation of the training procedure of the generator only <br/>
>```test_model.py```            &nbsp; - &nbsp; applying the pre-trained models to test images <br/>
>```utils.py```                 &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```                   &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>

<br/>

#### 7. Acknowledgments

This code is heavily inspired through the original work of Andrey Ignatov (see citation).

#### 8. Citation

```
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3277--3285},
  year={2017}
}
```


#### 9. Any further questions?

```
Please contact Oliver Lehmann (ollehman@ethz.ch) or Andrey Ignatov (andrey.ignatoff@gmail.com) for more information
```
