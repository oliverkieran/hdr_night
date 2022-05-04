# Imports
from imutils import paths
import shutil
import os

import random
random.seed(42)

"""
This script splits the directory raw_images2 into train and 
validation sets according to a certain threshold (default = 0.925)
"""


def move_images(image_paths, target_directory):

    # Create folder target_directory if it doesn't exist yet
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Loop through all the image paths
    for path in image_paths:
        if os.path.isfile(path):
            filename = path.split("/")[-1]
            target_path = os.path.join(target_directory, filename)
            shutil.copy2(path, target_path)
    total_images = list(paths.list_images(target_directory))
    print("Moved over {} images".format(len(total_images)))


def rename_images(path_raw, path_rgb):

    for path in [path_raw, path_rgb]:
        for i, img in enumerate(sorted(os.listdir(path))):
            os.rename(path + img, path + str(i).zfill(5) + ".png")

    for path in [path_raw, path_rgb]:
        for i, img in enumerate(sorted(os.listdir(path))):
            os.rename(path + img, path + str(i) + ".png")
        print("Renamed all images in " + path)


# raw_images are the inputs and rgb_images the outputs
rgb_images = sorted(list(paths.list_images("raw_images2/total/fujifilm")))
raw_images = sorted(list(paths.list_images("raw_images2/total/mediatek_raw")))

# Split data into train and val sets
train_ratio = 0.925
idx = int(len(rgb_images) * train_ratio)

train_rgb_paths = rgb_images[:idx]
train_raw_paths = raw_images[:idx]
valid_rgb_paths = rgb_images[idx:]
valid_raw_paths = raw_images[idx:]

print("number of training RGB examples = " + str(len(train_rgb_paths)))
print("number of training RAW examples = " + str(len(train_raw_paths)))
print("number of validation RGB examples = " + str(len(valid_rgb_paths)))
print("number of validation RAW examples = " + str(len(valid_raw_paths)))
print("Train: {}%, Valitation: {}%".format(train_ratio*100, (1-train_ratio)*100))

move_images(train_rgb_paths, "raw_images2/train/fujifilm")
move_images(train_raw_paths, "raw_images2/train/mediatek_raw")
move_images(valid_rgb_paths, "raw_images2/val/fujifilm")
move_images(valid_raw_paths, "raw_images2/val/mediatek_raw")

rename_images("../raw_images2/train/mediatek_raw/", "../raw_images2/train/fujifilm/")
rename_images("raw_images2/val/mediatek_raw/", "raw_images2/val/fujifilm/")
