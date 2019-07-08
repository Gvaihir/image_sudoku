from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


'''

   function to cluster centroids of polygons based on proximity

    Arguments:
   -----------
       points: 1D array 
           x,y coordinates of the pixel (centroid)

        eps: float 
            distance to the nearest point
            
        min_samples: int
            minimum number of points to call a cluster

   Returns:
   -----------
   list - cluster labels for each centroid

   '''
def dbscan_alg(points, eps=15, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    return labels


'''

   function to find medoid of a cluster

    Arguments:
   -----------
       cluster: int 
           cluster number currently analyzed

        points: list of arrays 
            coordinated of StarDist predicted centroids of polygons

        labels: list of int
            list of clusters for each centroids, predicted by DBSCAN

   Returns:
   -----------
   xy coordinates of polygon centroid which is medoid of a cluster 

   '''



def find_medoid(cluster, points, labels, area):
    rel_points = [points[x] for x in range(len(points)) if labels[x] == cluster]
    rel_area = [area[x] for x in range(len(area)) if labels[x] == cluster]
    pwdist = metrics.pairwise_distances(rel_points)
    point_out = rel_points[np.argmin(pwdist.sum(axis=1))]
    area_out = rel_area[np.argmin(pwdist.sum(axis=1))]
    return [point_out, area_out]



