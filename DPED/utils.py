import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys

from functools import reduce


def log10(x):
  numerator = tf.compat.v1.log(x)
  denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')


def process_command_args(arguments):

    # specifying default parameters

    batch_size = 32
    train_size = 5000
    learning_rate = 5e-5
    num_train_iters = 20000
    restore_iter = 0

    loss = "dped"

    dped_dir = '../raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000

    run_id = 'default_run'

    for args in arguments:

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("loss"):
            loss = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

        if args.startswith("run_id"):
            run_id = args.split("=")[1]

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Batch size:", batch_size)
    print("Train size:", train_size)
    print("Learning rate:", learning_rate)
    print("Training iterations:", str(num_train_iters))
    print("Restore iteration at:", str(restore_iter))
    print()
    print("Loss:", loss)
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print("Run Id:", run_id)
    print()
    return batch_size, train_size, learning_rate, num_train_iters, restore_iter, \
            loss, dped_dir, vgg_dir, eval_step, run_id


def process_test_model_args(arguments):

    dped_dir = "../raw_images/"
    test_dir = "full_res_test/"
    iteration = "20000"
    resolution = "full_res"  # [phone, full_res]
    use_gpu = "true"
    run_id = "default"

    for args in arguments:

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_dir"):
            test_dir = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("run_id"):
            run_id = args.split("=")[1]

    return dped_dir, test_dir, iteration, resolution, use_gpu, run_id
