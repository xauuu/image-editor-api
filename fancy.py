import cv2
from utils import readb64
import numpy as np
from scipy.interpolate import UnivariateSpline


def _create_LUT_BUC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def _create_loopup_tables():
    incr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    return incr_ch_lut, decr_ch_lut

def _warming(path):
    image = readb64(path)
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, incr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output


def _cooling(path):
    image = readb64(path)
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, decr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output

def _cartoon(path):
    image = readb64(path)
    img = np.copy(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    img_bilateral = cv2.edgePreservingFilter(
        img, flags=2, sigma_s=50, sigma_r=0.4)
    output = np.zeros(img_gray.shape)
    output = cv2.bitwise_and(img_bilateral, img_bilateral, mask=edge_mask)
    return output

def _sketch_pencil_using_blending(path):
    image = readb64(path)
    sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray

def _adjust_saturation(orig, saturation_scale=1.0):
    img = np.copy(orig)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.float32(hsv_img)
    H, S, V = cv2.split(hsv_img)
    S = np.clip(S * saturation_scale, 0, 255)
    hsv_img = np.uint8(cv2.merge([H, S, V]))
    im_sat = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return im_sat

def _moon(path):
    image = readb64(path)
    origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
    _curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])
    full_range = np.arange(0, 256)

    _LUT = np.interp(full_range, origin, _curve)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_img[:, :, 0] = cv2.LUT(lab_img[:, :, 0], _LUT)
    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    img = _adjust_saturation(img, 0.01)
    return img

def gotham(path):
    image = readb64(path)
    image = image/255
    b, g, r = cv2.split(image)
    image = np.stack([r, g, b], axis=2)
    r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    bluer_blacks = np.stack([r_boost_lower, g, np.clip(b + 0.03, 0, 1.0)], axis=2)
    sharper = sharpen(bluer_blacks, 1.3, 0.3)
    r, g, b = cv2.split(sharper)
    b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1]) 
    gotham = np.stack([r, g, b_adjusted], axis=2)
    return gotham


def xau(path):
    return