import tensorflow as tf
import numpy as np
import sys
import lpips_tf

from model import PUNET
import utils

from load_dataset import load_test_data

PATCH_WIDTH, PATCH_HEIGHT = 128, 128
TARGET_SIZE = (PATCH_WIDTH * 2) * (PATCH_HEIGHT * 2) * 3
config = None
model_dir, num_input_images = utils.process_test_scores_args(sys.argv)

# Path to the datasets:
dataset_dir = "../raw_images/"
dslr_dir = 'fujifilm/'
phone_dir = "mediatek_raw"
resolution = "full_res"
use_gpu = "true"

RI = int(model_dir.split("RI_")[1].split("_NI")[0])
if RI == 0:
    restore_iter = 100000
if RI == 100000:
    restore_iter = 150000
if model_dir =="learning_R_1e-05_RI_0_NI_1/":
    restore_iter = 95000
print(model_dir)
print(restore_iter)
print(RI)

# Disable gpu (if needed):
# config = tf.ConfigProto(device_count={'GPU': 0})

with tf.compat.v1.Session(config=config) as sess:

    # Create placeholders for input and target images

    phone_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_HEIGHT, 4*num_input_images])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [1, int(PATCH_HEIGHT * 2), int(PATCH_WIDTH * 2), 3])

    # Process raw images with your model:
    # enhanced = your_model(phone_), e.g.:
    enhanced = PUNET(phone_)

    # Compute losses

    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2)) / TARGET_SIZE
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    image0_ph = tf.compat.v1.placeholder(tf.float32)
    image1_ph = tf.compat.v1.placeholder(tf.float32)
    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

    # load previously trained model
    saver = tf.compat.v1.train.Saver()
    name_model_restore_full = model_dir + "punet_iteration_" + str(restore_iter)
    saver.restore(sess, "models/punet/" + name_model_restore_full + ".ckpt")

    print("Loading test data...")
    test_data, test_answ = load_test_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, num_input_images)
    print("Test data was loaded\n")
    print(np.shape(test_data))

    loss_ssim_ = 0.0
    loss_psnr_ = 0.0
    loss_lpips_ = 0.0

    test_size = test_data.shape[0]
    for j in range(test_size):

        if j % 100 == 0:
            print(j)
        phone_images = np.reshape(test_data[j], [1, PATCH_HEIGHT, PATCH_WIDTH, 4*num_input_images])
        dslr_images = np.reshape(test_answ[j], [1, int(PATCH_HEIGHT * 2), int(PATCH_WIDTH * 2), 3])
        enhanced_images = sess.run(enhanced, feed_dict={phone_: phone_images})

        losses = sess.run([loss_psnr, loss_ms_ssim], feed_dict={phone_: phone_images, dslr_: dslr_images})
        distance = sess.run(distance_t, feed_dict={image0_ph: enhanced_images, image1_ph: dslr_images})

        loss_psnr_ += losses[0]
        loss_ssim_ += losses[1]
        loss_lpips_ += distance

    loss_psnr_ = float(loss_psnr_) / test_size
    loss_ssim_ = float(loss_ssim_) / test_size
    loss_lpips_ = float(loss_lpips_) / test_size



    print(model_dir)
    output_logs = "PSNR: %.4g, MS-SSIM: %.4g, LPIPS: %.4g\n" % (loss_psnr_, loss_ssim_, loss_lpips_)
    print(output_logs)

