from machinevisiontoolbox import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

size = 5
radius = 1.5  # roughly fills a 5x5 grid

y, x = np.ogrid[:size, :size]
center = (size - 1) / 2

mask = (x - center)**2 + (y - center)**2 <= radius**2
mask = mask.astype(np.int8)


def get_roi(im: np.ndarray):
    height, width, _ = im.shape
    roi = im[140:height]
    min_val = roi.min()
    roi = (roi - min_val) / (roi.max() - min_val)
    return roi


def detect_red_chunck(im: np.ndarray, red_thresh=0.7, green_thresh=0.45, blue_thresh=0.45):

    # TODO: Adaptively threshold>?

    red = im[..., 0]
    green = im[..., 1]
    blue = im[..., 2]

    r_blobs = red > red_thresh
    g_blobs = green < green_thresh
    b_blobs = blue < blue_thresh
    # r_blobs = red.threshold(red_thresh)
    # g_blobs = green.threshold(green_thresh, opt='binary_inv')
    # b_blobs = blue.threshold(blue_thresh, opt='binary_inv')
    blobs = np.logical_and(r_blobs, g_blobs)
    blobs = np.logical_and(blobs, b_blobs)

    opened_blobs = ndimage.binary_opening(blobs, mask)

    return opened_blobs


def detect_white_chunck(img: np.ndarray):

    thresh = 0.7
    thresholded = img > thresh
    thresholded = thresholded.astype(float)
    all_chans = np.sum(thresholded, axis=-1)
    all_chans = all_chans >= 3.0

    # print(blobs.mono())
    opened_blobs = ndimage.binary_opening(all_chans, mask)
    # plt.figure()
    # plt.imshow(all_chans)
    # plt.figure()
    # plt.imshow(opened_blobs)
    # plt.show()

    return opened_blobs


def detect_stop_sign(im: np.ndarray, within_stop_sign=10, n_pixels=20):
    roi = get_roi(im)
    red_blobs = detect_red_chunck(roi)
    white_blobs = detect_white_chunck(roi)
    dist_red = ndimage.distance_transform_edt(~red_blobs)
    dist_white = ndimage.distance_transform_edt(~white_blobs)
    close_white = np.logical_and(white_blobs, dist_red < within_stop_sign)
    close_red = np.logical_and(red_blobs, dist_white < within_stop_sign)
    together_blobs = np.logical_or(close_white, close_red)
    local_blob_pixel_count = ndimage.uniform_filter(
        together_blobs.astype(np.float64), 10, mode='constant') * 100

    if np.max(local_blob_pixel_count) > 20:
        return True
    else:
        return False


if __name__ == "__main__":
    # im_path = "data/track3/000195-0.10.jpg"
    # im_path = "data/track3/001034-0.50.jpg"
    im_path = "data/track3/001038-0.40.jpg"
    # im_path = "data/track3/000526-0.30.jpg"
    # im_path = "pics/sample_stop_sign.png"

    # orange road
    # im_path = 'data/track3/000809-0.30.jpg'
    # im_path = "data/track3/001065-0.10.jpg"

    image = Image.open(im_path)
    im = np.asarray(image)
    detect_stop_sign(im)
