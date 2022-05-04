import numpy as np
import tensorflow as tf
import imageio
import sys
import os

from load_dataset_raw import extract_bayer_channels
from models import resnet
import utils

tf.compat.v1.disable_v2_behavior()

# process command arguments
dped_dir, raw_dir, restore_iter, resolution, use_gpu, run_id = utils.process_test_model_args(sys.argv)

# define image dimensions
if resolution == "phone":
    image_height, image_width = 128, 128
elif resolution == "full_res":
    image_height, image_width = 3000//2, 4000//2
else:
    print("Please choose 'phone' or 'full_res' as resolution.")
    sys.exit()

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

with tf.compat.v1.Session(config=config) as sess:

    # Placeholders for test data
    phone_ = tf.compat.v1.placeholder(tf.float32, [1, image_height, image_width, 4])

    # generate enhanced image
    enhanced = resnet(phone_)

    # load previously trained model
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "models/" + run_id + "/iteration_{}.ckpt".format(restore_iter))

    # Processing test/full-res RAW images
    test_dir = dped_dir + 'test/' + raw_dir
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    for photo in test_photos:
        print("Processing image " + photo)

        I = np.asarray(imageio.imread((test_dir + photo)))
        I = extract_bayer_channels(I)

        I = I[0:image_height, 0:image_width, :]
        I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

        # Run inference
        enhanced_tensor = sess.run(enhanced, feed_dict={phone_: I})
        enhanced_image = np.reshape(enhanced_tensor, [int(I.shape[1] * 2.0), int(I.shape[2] * 2.0), 3])

        # Save the results as .png images
        photo_name = photo.rsplit(".", 1)[0]
        save_img_dir = "results/full-resolution/" + run_id + "/iteration_" + str(restore_iter)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        imageio.imwrite(save_img_dir + "/" + photo_name + ".png", enhanced_image)
