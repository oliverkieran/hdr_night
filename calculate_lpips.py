import tensorflow as tf
import numpy as np
import os
import imageio

from DPED_night import lpips_tf

baseline_dir = "../crops/test/visualized_raw_crops/"
dslr_dir = "../crops/test/fujifilm/"

baseline_photos = [f for f in os.listdir(baseline_dir) if os.path.isfile(baseline_dir + f)]
NUM_TEST_IMAGES = len(baseline_photos)
format_baseline = str.split(os.listdir(baseline_dir)[0], '.')[-1]

baseline_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))
dslr_imgs = np.zeros((NUM_TEST_IMAGES, 256, 256, 3))

loss_lpips_ = 0.0

# Load photos
for i, photo in enumerate(baseline_photos):
    print("processing image nr", i)
    img_id = photo[:-(len(format_baseline) + 1)]
    try:
        baseline_imgs[i, :] = np.asarray(imageio.imread(baseline_dir + photo))
        dslr_imgs[i, :] = np.asarray(imageio.imread(dslr_dir + img_id + '.png'))
    except ValueError as e:
        print(e)
        print("ERROR:", photo)
        continue

print("Photos were loaded.")

with tf.compat.v1.Session(config=None) as sess:
    baseline_ = tf.compat.v1.placeholder(tf.float32, [1, 256, 256, 3])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [1, 256, 256, 3])

    image0_ph = tf.compat.v1.placeholder(tf.float32)
    image1_ph = tf.compat.v1.placeholder(tf.float32)
    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

    for j in range(NUM_TEST_IMAGES):

        if j % 100 == 0:
            print(j)

        baseline_images = np.reshape(baseline_imgs[j], [1, 256, 256, 3])
        dslr_images = np.reshape(dslr_imgs[j], [1, 256, 256, 3])

        lpips_distance = sess.run(distance_t, feed_dict={image0_ph: baseline_images, image1_ph: dslr_images})

        loss_lpips_ += lpips_distance

    loss_lpips_ = float(loss_lpips_) / NUM_TEST_IMAGES

    output_logs = "LPIPS: %.4g" % loss_lpips_
    print(output_logs)
