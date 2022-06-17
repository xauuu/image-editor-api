import cv2
import numpy as np
from utils import readb64

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

filter = np.array([(1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)]) * (1/25)
def matrix(igray):
    row = igray.shape[0] + filter.shape[0] - 1
    col = igray.shape[1] + filter.shape[1] - 1
    img_1 = np.zeros((row, col))
    for i in range(igray.shape[0]):
        for j in range(igray.shape[1]):
            img_1[i + int((filter.shape[0] - 1) // 2), j + int((filter.shape[1] - 1) // 2)] = igray[i, j]
    return img_1

def tb_so_hoc(path):
    img = readb64(path)
    igray = convert_to_gray(img)
    img_1 = matrix(igray)
    img_1_res = np.array(img)
    for i in range(igray.shape[0]):
        for j in range(igray.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.sum(temp * filter)
            img_1_res[i, j] = res
    return img_1_res

def tb_hinh_hoc(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.prod(temp ** filter)
            img_1_res[i, j] = res
    return img_1_res


def tb_harmonic(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = (filter.shape[0] * filter.shape[1]) // np.sum(np.reciprocal(temp))
            img_1_res[i, j] = res
    return img_1_res

def tb_contraharmonic(path):
    Q = -0.5
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = (np.sum(temp ** (Q + 1))) // (np.sum(temp ** Q))
            img_1_res[i, j] = res
    return img_1_res

def loc_trung_vi(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.median(temp)
            img_1_res[i, j] = res
    return img_1_res

def loc_max(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.max(temp)
            img_1_res[i, j] = res
    return img_1_res

def loc_min(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.min(temp)
            img_1_res[i, j] = res
    return img_1_res

def loc_midpoint(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            res = np.amax(temp * (1/2)) + np.amin(temp * (1/2))
            img_1_res[i, j] = res
    return img_1_res

def loc_alpha(path):
    d = 2
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)
    img_1_res = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]].copy()
            # max
            indices = temp.argpartition(temp.size - (d//2), axis = None)[-(d//2):]
            x, y = np.unravel_index(indices, temp.shape)
            temp[x, y] = 0
            
            # min
            indices = temp.argpartition(temp.size - (d//2), axis = None)[:(d//2)]
            x, y = np.unravel_index(indices, temp.shape)
            temp[x, y] = 0
            
            res = np.sum(temp * filter)
            img_1_res[i, j] = res
    return img_1_res

def loc_tuong_thich(path):
    i = readb64(path)
    img = convert_to_gray(i)
    img_1 = matrix(img)

    local_var = np.zeros((img.shape[0], img.shape[1]))
    local_mean = np.zeros((img.shape[0], img.shape[1]))


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = img_1[i:i+filter.shape[0], j:j+filter.shape[1]]
            local_mean[i, j] = np.mean(temp)
            local_var[i, j] = np.mean(temp ** 2) - (local_mean[i, j] ** 2)  
    noise_var = np.sum(local_var) // len(local_var)
    local_var = np.maximum(noise_var, local_var)
    img = img - np.multiply((noise_var / local_var), (img - local_mean))
    img = img.astype(np.uint8)
    return img

def loc_thong_thap_ly_tuong(path, d):
    img = readb64(path)
    kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
    sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)
    return sharp_img