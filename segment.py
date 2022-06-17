import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from utils import readb64


def kmean(path, k=3):
    image = readb64(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image


def graph_cut(path):
    img = readb64(path)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
    return img


def mean_shift(path):
    originImg = readb64(path)
    originShape = originImg.shape
    flatImg = np.reshape(originImg, [-1, 3])
    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Performing meanshift on flatImg
    ms.fit(flatImg)
    labels = ms.labels_

    # Remaining colors after meanshift
    cluster_centers = ms.cluster_centers_

    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])
    return segmentedImg
