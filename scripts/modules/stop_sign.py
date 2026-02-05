from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv
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
    # min_val = roi.min()
    # roi = (roi - min_val) / (roi.max() - min_val)
    return roi


def normalise(im: np.ndarray):
    min_val = im.min()
    im = (im - min_val) / (im.max() - min_val)
    return im


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
    roi = normalise(roi)

    red_blobs = detect_red_chunck(roi)
    white_blobs = detect_white_chunck(roi)
    dist_red = ndimage.distance_transform_edt(~red_blobs)
    dist_white = ndimage.distance_transform_edt(~white_blobs)
    close_white = np.logical_and(white_blobs, dist_red < within_stop_sign)
    close_red = np.logical_and(red_blobs, dist_white < within_stop_sign)
    together_blobs = np.logical_or(close_white, close_red)
    local_blob_pixel_count = ndimage.uniform_filter(
        together_blobs.astype(np.float64), 10, mode='constant') * 100

    plt.figure()
    plt.imshow(roi)
    plt.figure()
    plt.imshow(local_blob_pixel_count)
    plt.show()

    if np.max(local_blob_pixel_count) > 20:
        return True
    else:
        return False


def detect_red_hsv(hsv: np.ndarray, near_red=0.02, saturation=0.3, value=0.5):
    hue = hsv[..., 0]
    near_red_up = hue > 1 - near_red
    near_red_down = hue < near_red
    blobs = np.logical_or(near_red_up, near_red_down)

    sat = hsv[..., 1] > saturation
    val = hsv[..., 2] > value
    blobs = np.logical_and(blobs, sat)
    blobs = np.logical_and(blobs, val)

    opened_blobs = ndimage.binary_opening(blobs, mask)

    return opened_blobs


def detect_white_hsv(hsv, saturation=0.2, value=0.7):
    sat = hsv[..., 1] < saturation
    val = hsv[..., 2] > value
    blobs = np.logical_and(sat, val)

    opened_blobs = ndimage.binary_opening(blobs, mask)

    return opened_blobs


def detect_stop_sign_hsv(im: np.ndarray, within_stop_sign=10, n_pixels=20):
    roi = get_roi(im)
    print(roi.dtype)
    if im.dtype == np.uint8:
        roi = roi.astype(np.float64) / 255

    hsv_roi = rgb_to_hsv(roi)
    red_blobs = detect_red_hsv(hsv_roi)
    # white_blobs = detect_white_hsv(hsv_roi)

    # dist_red = ndimage.distance_transform_edt(~red_blobs)
    # dist_white = ndimage.distance_transform_edt(~white_blobs)
    # close_white = np.logical_and(white_blobs, dist_red < within_stop_sign)
    # close_red = np.logical_and(red_blobs, dist_white < within_stop_sign)
    # together_blobs = np.logical_or(close_white, close_red)
    local_blob_pixel_count = ndimage.uniform_filter(
        red_blobs.astype(np.float64), 10, mode='constant') * 100

    plt.figure()
    plt.imshow(hsv_roi)
    plt.figure()
    plt.imshow(roi)
    plt.figure()
    plt.imshow(local_blob_pixel_count)
    plt.show()

    if np.max(local_blob_pixel_count) > 15:
        return True
    else:
        return False


if __name__ == "__main__":
    # im_path = 'data/000371-0.50.jpg'
    # positive
    # im_path = 'data/000372-0.50.jpg'
    # im_path = 'data/000564-0.50.jpg'
    # im_path = 'data/000579-0.25.jpg'
    # im_path = 'data/0002120.00.jpg'
    im_path = 'data/0003760.00.jpg'

    # negative
    # im_path = 'data/0000500.50.jpg'
    # im_path = 'data/000626-0.50.jpg'
    # im_path = 'data/0002650.00.jpg'
    # im_path = 'data/000837-0.50.jpg'

    image = Image.open(im_path)
    im = np.asarray(image)
    detect_stop_sign_hsv(im)
