import numpy as np
import tensorflow as tf
import imageio
import sys
import os

from load_dataset import extract_bayer_channels
from models import resnet
import utils

tf.compat.v1.disable_v2_behavior()

# process command arguments
dped_dir, raw_dir, restore_iter, resolution, use_gpu, run_id, num_input_images = utils.process_test_model_args(sys.argv)

# define image dimensions
if resolution == "phone":
    image_height, image_width = 128, 128
    save_dir = "crops/"
elif resolution == "full_res":
    image_height, image_width = 3000//2, 4000//2
    save_dir = "full-resolution/"
else:
    print("Please choose 'phone' or 'full_res' as resolution.")
    sys.exit()

# define directories
test_dir = dped_dir + 'test/' + raw_dir
test_dirs = [test_dir + '/']
if num_input_images == 1:
    model_dir = "models_single/"
    result_dir = "results_single/"
    visual_results_dir = "visual_results_single/"
elif num_input_images == 3:
    model_dir = "models_multi/"
    result_dir = "results_multi/"
    visual_results_dir = "visual_results_multi/"
    test_dirs.append(test_dir + '_over/')
    test_dirs.append(test_dir + '_under/')
else:
    print("Please choose 1 or 3 input images.")
    sys.exit()

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

with tf.compat.v1.Session(config=config) as sess:

    # Placeholders for test data
    phone_ = tf.compat.v1.placeholder(tf.float32, [1, image_height, image_width, 4*num_input_images])

    # generate enhanced image
    enhanced = resnet(phone_, num_input_images)

    # load previously trained model
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, model_dir + run_id + "/iteration_{}.ckpt".format(restore_iter))

    # Processing test/full-res RAW images
    test_photos = [f for f in os.listdir(test_dirs[0]) if os.path.isfile(test_dirs[0] + f)]
    for photo in test_photos:
        print("Processing image " + photo)

        image = np.zeros((image_height, image_width, int(4*num_input_images)))
        for j in range(num_input_images):
            photo_dir = test_dirs[j]
            I = np.asarray(imageio.imread((photo_dir + photo)))
            I = extract_bayer_channels(I)
            image[:, :, 4*j:4*(j+1)] = I

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 4*num_input_images])

        # Run inference
        enhanced_tensor = sess.run(enhanced, feed_dict={phone_: image})
        enhanced_image = np.reshape(enhanced_tensor, [int(image.shape[1] * 2.0), int(image.shape[2] * 2.0), 3])

        # Save the results as .png images
        photo_name = photo.rsplit(".", 1)[0]
        save_img_dir = visual_results_dir + save_dir + run_id + "/iteration_" + str(restore_iter)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        imageio.imwrite(save_img_dir + "/" + photo_name + ".png", enhanced_image)
