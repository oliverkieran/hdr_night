import tensorflow as tf
import numpy as np
import sys

import lpips_tf
from models import resnet
import utils
from load_dataset import load_val_data

PATCH_WIDTH, PATCH_HEIGHT = 128, 128
TARGET_SIZE = (PATCH_WIDTH * 2) * (PATCH_HEIGHT * 2) * 3
config = None

# process command arguments
dped_dir, phone_dir, restore_iter, resolution, use_gpu, run_id, num_input_images = utils.process_test_model_args(sys.argv)

restore_iter = "20000"
run_id = "dped_loss"
num_input_images = 1

dslr_dir = 'fujifilm/'
if num_input_images == 1:
    model_dir = "models_single/"
elif num_input_images == 3:
    model_dir = "models_multi/"
else:
    print("Please choose 1 or 3 input images.")
    sys.exit()


# Disable gpu (if needed):
# config = tf.ConfigProto(device_count={'GPU': 0})

with tf.compat.v1.Session(config=config) as sess:

    # Create placeholders for input and target images

    phone_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_HEIGHT, int(4*num_input_images)])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [1, int(PATCH_HEIGHT * 2), int(PATCH_WIDTH * 2), 3])

    # Process raw images with your model:
    enhanced = resnet(phone_, num_input_images)

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
    saver.restore(sess, model_dir + run_id + "/" + "iteration_" + restore_iter + ".ckpt")

    print("Loading test data...")
    test_data, test_answ = load_val_data(dped_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, num_input_images, "test/")
    print("Test data was loaded\n")

    loss_ssim_ = 0.0
    loss_psnr_ = 0.0
    loss_lpips_ = 0.0


    test_size = test_data.shape[0]
    for j in range(test_size):

        if j % 100 == 0:
            print(j)

        phone_images = np.reshape(test_data[j], [1, PATCH_HEIGHT, PATCH_WIDTH, int(4*num_input_images)])
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


    output_logs = "PSNR: %.4g, MS-SSIM: %.4g, LPIPS: %.4g\n" % (loss_psnr_, loss_ssim_, loss_lpips_)
    print(output_logs)

