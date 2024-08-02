import numpy as np
from vispy.scene.visuals import Line

class LineCollection(Line):
    def __init__(self, lines, **kwargs):
        if 'pos' in kwargs:
            raise ValueError
        if 'connect' in kwargs:
            raise ValueError
        self.lines = lines
        self.n_lines, self.n_points = n_lines, n_points = lines.shape
        pos = self.get_pos(lines)

        a = np.arange(n_points-1)
        b = a + 1
        connect = np.tile(np.stack([a, b]), n_lines)
        connect = connect + np.expand_dims(np.repeat(np.arange(n_lines)*n_points, n_points-1), 0)

        Line.__init__(self, 
            pos=pos,
            connect=connect.T,
            **kwargs
        )

    def get_pos(self, lines, zorder=None):
        # if zorder is None:
        #     zorder = np.zeros(self.n_lines)
        x = np.broadcast_to(np.arange(self.n_points), lines.shape)
        pos = np.stack([x, lines], axis=-1).reshape(-1, 2)
        # x = np.broadcast_to(np.arange(self.n_points), lines.shape)
        # z = np.broadcast_to(zorder, lines.shape, )
        # x, lines, z = np.broadcast_arrays(np.arange(self.n_points), lines, zorder[:, None])
        # pos = np.stack([x, lines, z], axis=-1).reshape(-1, 3)
        return pos

    def set_data(self, **kwargs):
        zorder = kwargs.pop('zorder')
        if zorder is None:
            zorder = np.zeros(self.n_lines)
        idx = np.argsort(-zorder)
        lines = self.lines[idx]
        kwargs['pos'] = self.get_pos(lines)
        if 'color' in kwargs:
            kwargs['color'] = np.repeat(kwargs['color'][idx], self.n_points, axis=0)
            # kwargs['color'] = np.repeat(kwargs['color'], self.n_points, axis=0)

        return super().set_data(**kwargs)
