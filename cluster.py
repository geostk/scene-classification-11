import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

DIR_DATA = "data"
NAME_CENTERS = "centers_Multiple.npy"
VALUE_CLUSTER = 10000

def kmeans(data):
    data = np.load(data)
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=VALUE_CLUSTER, batch_size=9000, n_init=10, max_no_improvement=10, verbose=0)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    np.save(os.path.join(DIR_DATA,NAME_CENTERS), centers)
    return centers

