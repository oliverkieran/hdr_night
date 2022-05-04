from PIL import Image
import numpy as np
import imageio
import os

from visualize_crops import process_for_visualizing, extract_bayer_channels

test_dir = "../crops/test/mediatek_raw/"
test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

for photo in test_photos:
    print("processing photo:", photo)
    raw = imageio.imread(test_dir + photo)
    raw = raw.astype(np.float32)

    ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw)
    ch_G = (ch_Gr + ch_Gb) * 0.5

    img = np.dstack((ch_R, ch_G, ch_B))

    img_processed = process_for_visualizing(img) * 255
    img_PIL_processed = Image.fromarray(img_processed.astype(np.uint8))
    final_image = img_PIL_processed.resize((256,256), Image.BICUBIC)
    final_image.save("../crops/test/visualized_raw_crops/" + photo)

