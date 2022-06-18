import cv2
from utils import readb64
import numpy as np
from scipy.interpolate import UnivariateSpline

def gaussian_blur(path, x):
    image = readb64(path)
    dst = cv2.GaussianBlur(image, (x, x), cv2.BORDER_DEFAULT)
    return dst


def _create_LUT_BUC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def _create_loopup_tables():
    incr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    return incr_ch_lut, decr_ch_lut

def warming(path):
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


def cooling(path):
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

def cartoon(path):
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

def sketch_pencil_using_blending(path):
    image = readb64(path)
    sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray

def pencil_sketch_col(path):
    image = readb64(path)
    sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_color

def _adjust_saturation(orig, saturation_scale=1.0):
    img = np.copy(orig)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.float32(hsv_img)
    H, S, V = cv2.split(hsv_img)
    S = np.clip(S * saturation_scale, 0, 255)
    hsv_img = np.uint8(cv2.merge([H, S, V]))
    im_sat = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return im_sat

def moon(path):
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

def sepia(path):
    image = readb64(path)
    img_sepia = np.array(image, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia

def HDR(path):
    image = readb64(path)
    hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
    return  hdr

def greyscale(path):
    image = readb64(path)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return greyscale

def vintage(path):
    image = readb64(path)
    rows, cols = image.shape[:2]
    # Create a Gaussian filter
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    filter = 255 * kernel / np.linalg.norm(kernel)
    vintage_im = np.copy(image)
    # for each channel in the input image, we will apply the above filter
    for i in range(3):
        vintage_im[:, :, i] = vintage_im[:, :, i] * filter
    return vintage_im
