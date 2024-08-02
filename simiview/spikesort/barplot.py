import numpy as np
from vispy.scene.visuals import Polygon

class BarPlot(Polygon):
    def __init__(self, x, y, bottom=-.1, **kwargs):
        bottom = self._get_bottoms(x, bottom)
        pos = self.get_vertices(x, y, bottom)
        # print(pos)
        # kwargs['triangulate'] = False
        Polygon.__init__(self, pos=pos, **kwargs)

    def _get_bottoms(self, x, bottom):
        if np.isscalar(bottom):
            bottom = np.repeat(bottom, x.size)
        return bottom

    def set_data(self, **kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y')
        bottom = self._get_bottoms(x, kwargs.pop('bottom', -.1))
        self.pos = self.get_vertices(x, y, bottom)
        self._update()

    def get_vertices(self, x, y, bottom):
        width = np.diff(x)
        width = np.insert(width, -1, width[-1])
        vertices = [(x[0]-width[0]/2, bottom[0])]
        for xi, yi, wi in zip(x, y, width):
            vertices.append((xi-wi/2, yi))
            vertices.append((xi+wi/2, yi))
        vertices.append((x[-1]+width[-1]/2, bottom[-1]))
        return np.array(vertices)