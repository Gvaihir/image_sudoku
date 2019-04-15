from __future__ import print_function, unicode_literals, absolute_import, division



# pip
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression

'''

   function to detect centroids of polygons

    Arguments:
   -----------
       x: lists of 2D or 3D np.ndarray's
           ndarrays - images

       model: class StarDist model

       prob_thresh: float
           probability for stardist coordinate prediction
           
        nms_thresh: float
            threshold for stardist non maximum suppression

   Returns:
   -----------
   Array of centroid coordinates.

   '''


def stardist_predict(x, model, size, prob_thresh=0.5, coef_dist=0.5, nms_thresh=0.5):

    # Normalize images and fill small label holes
    axis_norm = (0, 1)  # normalize channels independently
    # axis_norm = (0, 1, 2)  # normalize channels jointly

    # normalized image
    img = normalize(x, 1, 99.8, axis=axis_norm)

    # Predict object probabilities and star-convex polygon distances
    prob, dist = model.predict(img)

    # Convert distances to polygon vertex coordinates
    coord = dist_to_coord(dist)

    # Perform non-maximum suppression for polygons above object probability threshold
    points = non_maximum_suppression(coord, prob, prob_thresh=prob_thresh, b=int(size*coef_dist), nms_thresh=nms_thresh)

    return coord, points


'''

   function to detect centroids of polygons, uses coord and point from StarDist prediction

    Arguments:
   -----------
       point: 1D array 
           x,y coordinates of the pixel (centroid)
        
        coord 4D array 
            xy coordinates for each of 32 vertexes associated with each point

   Returns:
   -----------
   float - surface area of the polygon.

   '''



def PolyArea(point, coord):
    x, y = coord[point[0], point[1], 1], coord[point[0], point[1], 0]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



