###########################################
# Dataloader for training/validation data #
###########################################

from __future__ import print_function
import imageio
import numpy as np
import os
from PIL import Image


def extract_bayer_channels(raw):

    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE=2, input_images=1):

    val_directory_dslr = dataset_dir + 'val/' + dslr_dir
    val_directory_phone = [dataset_dir + 'val/' + phone_dir + '/']
    if input_images == 3:
        val_directory_phone.append(dataset_dir + 'val/' + phone_dir + '_over/')
        val_directory_phone.append(dataset_dir + 'val/' + phone_dir + '_under/')

    # get the image format (e.g. 'png')
    format_dslr = str.split(os.listdir(val_directory_dslr)[0],'.')[-1]

    # determine validation image numbers by listing all files in the folder
    NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone[0])
                           if os.path.isfile(os.path.join(val_directory_phone[0], name))])

    val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, int(4*input_images)))
    val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    for i in range(0, NUM_VAL_IMAGES):

        for j in range(int(input_images)):
            photo_dir = val_directory_phone[j]
            I = np.asarray(imageio.imread((photo_dir + str(i) + '.png')))
            I = extract_bayer_channels(I)
            val_data[i, :, :, 4*j:4*(j+1)] = I

        I = Image.open(val_directory_dslr + str(i) + '.' + format_dslr)
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        val_answ[i, :] = I

    return val_data, val_answ


def load_train_patch(dataset_dir, dslr_dir, phone_dir, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE=2, input_images=1):

    train_directory_dslr = dataset_dir + 'train/' + dslr_dir
    train_directory_phone = [dataset_dir + 'train/' + phone_dir + '/']
    if input_images == 3:
        train_directory_phone.append(dataset_dir + 'train/' + phone_dir + '_over/')
        train_directory_phone.append(dataset_dir + 'train/' + phone_dir + '_under/')

    # get the image format (e.g. 'png')
    format_dslr = str.split(os.listdir(train_directory_dslr)[0],'.')[-1]

    # determine training image numbers by listing all files in the folder
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone[0])
                               if os.path.isfile(os.path.join(train_directory_phone[0], name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, int(4*input_images)))
    train_answ = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3))

    i = 0
    for img in TRAIN_IMAGES:

        for j in range(int(input_images)):
            photo_dir = train_directory_phone[j]
            I = np.asarray(imageio.imread((photo_dir + str(img) + '.png')))
            I = extract_bayer_channels(I)
            train_data[i, :, :, 4*j:4*(j+1)] = I

        I = Image.open(train_directory_dslr + str(img) + '.' + format_dslr)
        I = np.array(I.resize((int(I.size[0] * DSLR_SCALE / 2), int(I.size[1] * DSLR_SCALE / 2)), resample=Image.BICUBIC))
        I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3])) / 255
        train_answ[i, :] = I

        i += 1

    return train_data, train_answ

def load_test_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, input_images=1):

    val_directory_dslr = dataset_dir + 'test/' + dslr_dir
    val_directory_phone = [dataset_dir + 'test/' + phone_dir + '/']
    if input_images == 3:
        val_directory_phone.append(dataset_dir + 'test/' + phone_dir + '_over/')
        val_directory_phone.append(dataset_dir + 'test/' + phone_dir + '_under/')

    # get the image format (e.g. 'png')
    format_dslr = str.split(os.listdir(val_directory_dslr)[0],'.')[-1]

    # determine validation image numbers by listing all files in the folder
    NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone[0])
                           if os.path.isfile(os.path.join(val_directory_phone[0], name))])

    val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, int(4*input_images)))
    val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH * 2), int(PATCH_HEIGHT * 2), 3))

    for i in range(0, NUM_VAL_IMAGES):

        for j in range(int(input_images)):
            photo_dir = val_directory_phone[j]
            I = np.asarray(imageio.imread((photo_dir + str(i) + '.png')))
            I = extract_bayer_channels(I)
            val_data[i, :, :, 4*j:4*(j+1)] = I
        I = Image.open(val_directory_dslr + str(i) + '.' + format_dslr)
        I = np.array(I.resize((int(I.size[0] * 2 / 2), int(I.size[1] * 2 / 2)), resample=Image.BICUBIC))
        I = np.float16(np.reshape(I, [1, int(PATCH_WIDTH * 2), int(PATCH_HEIGHT * 2), 3])) / 255
        val_answ[i, :] = I

    return val_data, val_answ

