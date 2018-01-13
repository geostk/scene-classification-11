import numpy as np
import cv2

def kmeans(data, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print(len(data))
    print(ret)
    print(label)
    print(len(center))
    return ret, label, center
