# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

# python dng_to_png.py path_to_my_dng_file.dng

import numpy as np
import imageio
import rawpy
import sys
import os


def extract_bayer_channels(raw):

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    return ch_R, ch_Gr, ch_B, ch_Gb


if __name__ == "__main__":

    dng_dir = sys.argv[1]
    png_dir = sys.argv[2]

    dng_photos = [f for f in os.listdir(dng_dir) if os.path.isfile(dng_dir + f)]

    for photo in dng_photos:
        raw_file = dng_dir + photo
        print("Converting file " + raw_file)

        if not os.path.isfile(raw_file):
            print("The file doesn't exist!")
            sys.exit()

        raw = rawpy.imread(raw_file)
        raw_image = raw.raw_image
        del raw

        # Use the following code to rotate the image (if needed)
        # raw_image = np.rot90(raw_image, k=2)

        raw_image = raw_image.astype(np.float32)
        ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw_image)

        png_image = raw_image.astype(np.uint16)
        new_name = photo.replace(".dng", ".png")
        imageio.imwrite(png_dir + new_name, png_image)
