from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys

from load_dataset import load_train_patch, load_val_data
import models
import utils
import vgg

tf.compat.v1.disable_v2_behavior()

# defining size of the training image patches

PATCH_WIDTH = 256//2
PATCH_HEIGHT = 256//2
TARGET_WIDTH = PATCH_WIDTH * 2
TARGET_HEIGHT = PATCH_HEIGHT * 2
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 4
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * 3

# processing command arguments

batch_size, train_size, learning_rate, num_train_iters, restore_iter, \
loss, dped_dir, vgg_dir, run_id, from_model, eval_step, num_input_images = utils.process_command_args(sys.argv)

np.random.seed(0)

# define directories
if num_input_images == 1:
    model_dir = "models_single/"
    log_dir = "logs_single/"
    result_dir = "results_single/"
    visual_results_dir = "visual_results_single/"
elif num_input_images == 3:
    model_dir = "models_multi/"
    log_dir = "logs_multi/"
    result_dir = "results_multi/"
    visual_results_dir = "visual_results_multi/"
else:
    print("Please choose 1 or 3 input images.")
    sys.exit()

# defining system architecture

with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()
    
    # placeholders for training data

    phone_image = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, int(4*num_input_images)])
    dslr_image = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, 3])

    adv_ = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # get processed enhanced image
    enhanced = models.resnet(phone_image, num_input_images)

    # transform both dslr and enhanced images to grayscale
    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, TARGET_WIDTH * TARGET_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, TARGET_WIDTH * TARGET_HEIGHT])
    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_image, [-1, TARGET_SIZE])


    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator
    
    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    #adversarial_ = tf.multiply(enhanced_flat, 1 - adv_) + tf.multiply(dslr_flat, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, TARGET_HEIGHT, TARGET_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)

    loss_discrim = -tf.reduce_sum(discrim_target * tf.compat.v1.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    loss_texture = -loss_discrim
    loss_texture_summ = tf.compat.v1.summary.scalar("texture_loss", loss_texture)

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discrim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    discrim_accuracy_summ = tf.compat.v1.summary.scalar("discrim_acc", discrim_accuracy)

    # 2) content loss

    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
    loss_content_summ = tf.compat.v1.summary.scalar("content_loss", loss_content)

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)
    loss_color_summ = tf.compat.v1.summary.scalar("color_loss", loss_color)

    # 4) total variation loss

    batch_shape = (batch_size, TARGET_WIDTH, TARGET_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size
    loss_tv_summ = tf.compat.v1.summary.scalar("tv_loss", loss_tv)

    # 5.) SSIM loss
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_image, 1.0))

    # MS-SSIM loss
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_image, 1.0))
    loss_ms_ssim_summ = tf.compat.v1.summary.scalar("ms_ssim_loss", loss_ms_ssim)

    # MSE loss
    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2)) / (TARGET_SIZE * batch_size)

    # final loss
    if loss == "pynet":
        loss_generator = 20 * loss_content + 2 * loss_texture + 15 * loss_mse + 1 * (1-loss_ssim)
    elif loss == "dped_night":
        loss_generator = 10 * loss_content + 1 * loss_texture + 20 * loss_mse + 15 * (1 - loss_ssim)
    else:
        loss_generator = 20 * loss_content + 4 * loss_texture + 1 * loss_color + 15 * (1-loss_ssim)

    loss_generator_summ = tf.compat.v1.summary.scalar('total_loss', loss_generator)

    # psnr loss
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))
    loss_psnr_summ = tf.compat.v1.summary.scalar("psnr_loss", loss_psnr)

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    discriminator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("discriminator")]

    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
    train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    if restore_iter > 0:  # restore the variables/weights
        name_model_restore_full = from_model + "/iteration_" + str(restore_iter)
        print("Restoring Variables from:", name_model_restore_full)
        saver.restore(sess, model_dir + name_model_restore_full + ".ckpt")

    # merge all tensorboard summaries
    merged = tf.compat.v1.summary.merge(
        [loss_generator_summ,
         loss_content_summ,
         loss_color_summ,
         loss_tv_summ,
         loss_psnr_summ,
         loss_ms_ssim_summ,
         loss_texture_summ,
         discrim_accuracy_summ]
    )

    train_writer = tf.compat.v1.summary.FileWriter(log_dir + run_id + '/train')
    val_writer = tf.compat.v1.summary.FileWriter(log_dir + run_id + '/val')

    print('Initializing variables')
    sess.run(tf.compat.v1.global_variables_initializer())

    # loading training and test data

    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir=dped_dir,
                                       dslr_dir="fujifilm/",
                                       phone_dir="mediatek_raw",
                                       PATCH_WIDTH=PATCH_WIDTH,
                                       PATCH_HEIGHT=PATCH_HEIGHT,
                                       input_images=num_input_images)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir=dped_dir,
                                              dslr_dir="fujifilm/",
                                              phone_dir="mediatek_raw",
                                              TRAIN_SIZE=train_size,
                                              PATCH_WIDTH=PATCH_WIDTH,
                                              PATCH_HEIGHT=PATCH_HEIGHT,
                                              input_images=num_input_images)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    print('Training network')

    start_iter = restore_iter + 1 if restore_iter > 0 else 0

    train_loss_gen = 0.0
    train_acc_discrim = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

    # Create folder target_directory if it doesn't exist yet
    model_log_dir = model_dir + run_id
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)
    logs = open(model_log_dir + '/' + run_id + '.txt', "w+")
    logs.close()

    for i in range(start_iter, num_train_iters+1):

        # get train data

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        # Data augmentation: random flips and rotations

        for k in range(batch_size):

            random_rotate = np.random.randint(1, 100) % 4
            phone_images[k] = np.rot90(phone_images[k], random_rotate)
            dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2

            if random_flip == 1:
                phone_images[k] = np.flipud(phone_images[k])
                dslr_images[k] = np.flipud(dslr_images[k])

        # train generator

        if i % eval_step != 0:
            [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                         feed_dict={phone_image: phone_images,
                                                    dslr_image: dslr_images, adv_: all_zeros})
            train_loss_gen += loss_temp / eval_step

            # train discriminator
            idx_train = np.random.randint(0, train_size, batch_size)
    
            # generate image swaps (dslr or enhanced) for discriminator
            swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])
    
            phone_images = train_data[idx_train]
            dslr_images = train_answ[idx_train]
    
            [accuracy_temp, temp] = sess.run([discrim_accuracy, train_step_disc],
                                             feed_dict={phone_image: phone_images, dslr_image: dslr_images, adv_: swaps})
            train_acc_discrim += accuracy_temp / eval_step

        # evaluate on validation data

        else:
            [loss_temp, temp, train_summary] = sess.run([loss_generator, train_step_gen, merged],
                                                        feed_dict={phone_image: phone_images,
                                                                   dslr_image: dslr_images, adv_: all_zeros})
            train_writer.add_summary(train_summary, i)
            train_loss_gen += loss_temp / eval_step

            # test generator and discriminator CNNs

            val_losses_gen = np.zeros((1, 7))
            val_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(num_val_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                [enhanced_crops, accuracy_disc, losses, val_summary] = sess.run([enhanced, discrim_accuracy,
                                                                                 [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr, loss_ms_ssim], merged],
                                                                                feed_dict={phone_image: phone_images, dslr_image: dslr_images, adv_: swaps})

                val_losses_gen += np.asarray(losses) / num_val_batches
                val_accuracy_disc += accuracy_disc / num_val_batches

            # record summaries and write to disk
            val_writer.add_summary(val_summary, i)

            logs_disc = "step %d | discriminator accuracy | train: %.4g, test: %.4g" %\
                        (i, train_acc_discrim, val_accuracy_disc)

            logs_gen = "step %d generator losses | train: %.4g, val: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ms-ssim: %.4g\n" % \
                       (i, train_loss_gen, val_losses_gen[0][0], val_losses_gen[0][1], val_losses_gen[0][2],
                        val_losses_gen[0][3], val_losses_gen[0][4], val_losses_gen[0][5], val_losses_gen[0][6])

            print(logs_disc)
            print(logs_gen)

            print('total train/eval time:', datetime.now() - time_start)

            # save the results to log file
            logs = open(model_dir + run_id + '/' + run_id + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration
            saver.save(sess, model_dir + str(run_id) + '/iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir=dped_dir, dslr_dir="fujifilm/",
                                                      phone_dir="mediatek_raw", TRAIN_SIZE=train_size,
                                                      PATCH_WIDTH=PATCH_WIDTH, PATCH_HEIGHT=PATCH_HEIGHT,
                                                      input_images=num_input_images)

    print('total train/eval time:', datetime.now() - time_start)
