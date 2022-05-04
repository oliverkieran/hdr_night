import tensorflow as tf
import numpy as np
import os
import imageio

baseline_dir = "../crops/test/enhanced/"
dslr_dir = "../crops/test/fujifilm/"

baseline_photos = [f for f in os.listdir(baseline_dir) if os.path.isfile(baseline_dir + f)]
NUM_TEST_IMAGES = len(baseline_photos)
format_baseline = str.split(os.listdir(baseline_dir)[0],'.')[-1]

baseline_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))
dslr_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))

psnr = 0.0
ms_ssim = 0.0

for i, photo in enumerate(baseline_photos):
	print("processing image nr", i)
	img_id = photo[:-(len(format_baseline)+1)]
	try:
		#baseline_imgs[i, :] = np.asarray(imageio.imread(baseline_dir + photo))
		#dslr_imgs[i, :] = np.asarray(imageio.imread(dslr_dir + photo))
		baseline_img = np.float32(imageio.imread(baseline_dir + photo))
		dslr_img = np.float32(imageio.imread(dslr_dir + img_id + '.png'))
	except ValueError as e:
		print(e)
		print("ERROR:", photo)
		continue

	psnr += tf.image.psnr(tf.clip_by_value(baseline_img, 0, 255), tf.clip_by_value(dslr_img, 0, 255), max_val=255)
	ms_ssim += tf.image.ssim_multiscale(tf.clip_by_value(baseline_img, 0, 255), tf.clip_by_value(dslr_img, 0, 255), max_val=255)

#ms_ssim = tf.image.ssim_multiscale(tf.clip_by_value(baseline_imgs, 0, 255), tf.clip_by_value(dslr_imgs, 0, 255), max_val=255)
#psnr = tf.image.psnr(tf.clip_by_value(baseline_imgs, 0, 255), tf.clip_by_value(dslr_imgs, 0, 255), max_val=255)
print("MS-SSIM = ", ms_ssim / NUM_TEST_IMAGES)
print("PSNR = ", psnr / NUM_TEST_IMAGES)