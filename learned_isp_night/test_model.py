###########################################
# Run the trained model on testing images #
###########################################

import numpy as np
import tensorflow as tf
import imageio
import sys
import os

from model import PUNET
import utils

from datetime import datetime
from load_dataset import extract_bayer_channels

dataset_dir, test_dir, model_dir, result_dir, arch, LEVEL, inst_norm, num_maps_base,\
    orig_model, rand_param, restore_iter, IMAGE_HEIGHT, IMAGE_WIDTH, use_gpu, save_model, test_image, num_input_images = \
        utils.process_test_model_args(sys.argv)
DSLR_SCALE = float(1) / (2 ** (max(LEVEL, 0) - 1))

# Disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None

with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # determine model name
    if arch == "punet":
        name_model = "punet"

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 4*num_input_images])

    # generate enhanced image
    if arch == "punet":
        enhanced = PUNET(x_, instance_norm=inst_norm, instance_norm_level_1=False, num_maps_base=num_maps_base)
    
    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    if orig_model: # load official pre-trained weights
        model_dir = "models/original/"
        name_model_full = name_model + '_pretrained'
        saver.restore(sess, model_dir + name_model_full + ".ckpt")
    else:
        if rand_param: # use random weights
            name_model_full = name_model
            global_vars = [v for v in tf.compat.v1.global_variables()]
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(var_list=global_vars)
        else: # load previous/restored pre-trained weights
            name_model_full = name_model + "_iteration_" + str(restore_iter)
            print("restoring model: ", model_dir + name_model_full + ".ckpt")
            saver.restore(sess, model_dir + name_model_full + ".ckpt")

    # Processing test images
    if test_image:
        test_dir_full = [dataset_dir + test_dir + 'mediatek_raw/']
        if num_input_images == 3:
            test_dir_full.append(dataset_dir + test_dir + 'mediatek_raw_over/')
            test_dir_full.append(dataset_dir + test_dir + 'mediatek_raw_under/')
        test_photos = [f for f in os.listdir(test_dir_full[0]) if os.path.isfile(test_dir_full[0] + f)]
        print("amount of photos found:", len(test_photos))
        image_data = np.zeros((IMAGE_HEIGHT//2, IMAGE_WIDTH//2, int(4 * num_input_images)))

        for photo in test_photos:

            print("Processing image " + photo)

            for j in range(num_input_images):
                photo_dir = test_dir_full[j]
                I = np.asarray(imageio.imread((photo_dir + photo)))
                I = extract_bayer_channels(I)
                I = I[0:((IMAGE_HEIGHT // 32)*16), 0:(IMAGE_WIDTH // 32)*16,:]
                image_data[:, :, 4 * j:4 * (j + 1)] = I

            image = image_data[0:IMAGE_HEIGHT//2, 0:IMAGE_WIDTH//2, :]
            image = np.reshape(image, [1, image.shape[0], image.shape[1], 4*num_input_images])

            # Run inference
            enhanced_tensor = sess.run(enhanced, feed_dict={x_: image})
            enhanced_image = np.reshape(enhanced_tensor, [int(image.shape[1] * DSLR_SCALE), int(image.shape[2] *
                                                                                                DSLR_SCALE), 3])

            # Save the results as .png images
            photo_name = photo.rsplit(".", 1)[0]
            path_1 = result_dir + model_dir.split("/")[-2]
            path_2 = path_1 + "/" + photo_name + "/"
            try:
                os.mkdir(path_1)
            except OSError:
                print("Creation of the directory %s failed" % path_1)
            else:
                print("Successfully created the directory %s " % path_1)
            try:
                os.mkdir(path_2)
            except OSError:
                print("Creation of the directory %s failed" % path_2)
            else:
                print("Successfully created the directory %s " % path_2)
            imageio.imwrite(path_2 + str(restore_iter) + "_" + "_NI_" + str(num_input_images) + "_" + photo_name + ".png", enhanced_image)

    print('total test time:', datetime.now() - time_start)

    # save model again (optional, but useful for MAI challenge)
    if save_model:

        saver.save(sess, model_dir + name_model_full + ".ckpt") # pre-trained weight + meta graph
        utils.export_pb(sess, 'output_l0', model_dir, name_model_full + ".pb") # frozen graph (i.e. protobuf)
        tf.compat.v1.summary.FileWriter(model_dir + name_model_full, sess.graph) # tensorboard
        print('model saved!')