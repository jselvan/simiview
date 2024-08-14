from matplotlib.path import Path
import numpy as np

def points_in_polygon(points, poly):
    path = Path(poly)
    inside = path.contains_points(points[:, :2])
    return np.where(inside)[0]
