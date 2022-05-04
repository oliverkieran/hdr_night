from PIL import Image
import numpy as np
import imageio


def process_for_visualizing(image):

    image = image.astype(np.float32)
    img_black_leveled = image - 64
    img_black_leveled[img_black_leveled < 0] = 0

    wb_R = np.mean(img_black_leveled[:, :, 0])
    wb_G = np.mean(img_black_leveled[:, :, 1])
    wb_B = np.mean(img_black_leveled[:, :, 2])

    gain_R = wb_G / wb_R
    gain_B = wb_G / wb_B

    if gain_R > 1.0 and gain_B > 1.0:
        img_black_leveled[:, :, 0] *= gain_R
        img_black_leveled[:, :, 2] *= gain_B
        pass

    im_max = np.max(img_black_leveled)
    img_scaled = img_black_leveled / im_max

    img_gammaed = ((img_scaled) ** (1 / 2.2))
    return img_gammaed


def extract_bayer_channels(raw):

    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_B  = raw[1::2, 1::2]

    return ch_R, ch_Gr, ch_B, ch_Gb


if __name__ == "__main__":

    img_nr = "1017"
    filename = "test/mediatek_raw/" + img_nr + ".png"

    raw = imageio.imread(filename)
    raw = raw.astype(np.float32)

    ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw)
    ch_G = (ch_Gr + ch_Gb) * 0.5

    img = np.dstack((ch_R, ch_G, ch_B))

    img_processed = process_for_visualizing(img) * 255
    img_PIL_processed = Image.fromarray(img_processed.astype(np.uint8))
    img_PIL_processed.save("test/visualized_raw/" + img_nr + "_raw.png")
    #img_PIL_processed.save("crop_visualized.png")
