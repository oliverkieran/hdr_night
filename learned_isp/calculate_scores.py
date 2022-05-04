import tensorflow as tf
import numpy as np
import os
import imageio

fake_dir = "results/full-resolution/"
dslr_dir = "raw_images/learned_isp_test_dataset/fujifilm/"

NUM_TEST_IMAGES = 3145

fake_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))
dslr_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))

for i in range(0, NUM_TEST_IMAGES):
	fake_imgs[i, :] = np.asarray(imageio.imread(fake_dir + str(i) + '-punet_iteration_99000.png'))
	dslr_imgs[i, :] = np.asarray(imageio.imread(dslr_dir + str(i) + '.png'))

ms_ssim = tf.image.ssim_multiscale(tf.clip_by_value(fake_imgs, 0, 255), tf.clip_by_value(dslr_imgs, 0, 255), max_val=255)
psnr = tf.image.psnr(tf.clip_by_value(fake_imgs, 0, 255), tf.clip_by_value(dslr_imgs, 0, 255), max_val=255)
print("MS-SSIM = ", np.average(ms_ssim))
print("PSNR = ", np.average(psnr))