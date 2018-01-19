import os
import numpy as np
import extractor
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans

DIR_DATA = "data"
DIR_BOW = os.path.join(DIR_DATA, "bow", "FaceNot")
DIR_FEATURES = os.path.join(DIR_DATA, "features", 'FaceNot')

def find_closest_word(each_sift,centeroids):
    centeroids = np.asarray(centeroids)
    compared = centeroids - each_sift
    distance = np.einsum('ij,ij->i', compared, compared)
    return np.argmin(distance)

def centeroid(centers_file):
    centers = np.load(centers_file)
    return centers

def build_vector_10k(dir_class,dir_img,centeroids):
    feature_img = os.path.join(DIR_FEATURES, dir_class, dir_img + ".npy")
    bow_img = os.path.join(DIR_BOW, dir_class, dir_img)
    vector_10k = np.zeros(10000)
    sift_features = np.load(feature_img)
    for each_sift in sift_features:
        pos = find_closest_word(each_sift,centeroids)
        vector_10k[pos] += 1
    np.save(bow_img, vector_10k)



