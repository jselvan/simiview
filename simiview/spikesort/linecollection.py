import numpy as np
from vispy.scene.visuals import Line

class LineCollection(Line):
    def __init__(self, lines, **kwargs):
        if 'pos' in kwargs:
            raise ValueError
        if 'connect' in kwargs:
            raise ValueError
        self.lines = lines
        kwargs['connect'] = self.get_connect()
        self.offset = kwargs.pop('offset', 0)
        Line.__init__(self)
        self.set_data(**kwargs)

    def get_pos(self, lines: np.ndarray, offset: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Generate (x, y) positions for points in multiple lines with an offset applied.

        Parameters
        ----------
        lines (np.ndarray):  
            A 2D array of shape (n_lines, n_points) representing line data.  
        offset (Union[np.ndarray, float, int]):  
            An offset array of shape (n_lines,) or a scalar to be applied to each line.  
        idx (np.ndarray):
            An array of indices to sort the lines by.

        Returns
        -------
        np.ndarray: A 2D array of shape (n_lines * n_points, 2) containing (x, y) positions.
        """
        # Ensure lines is a 2D array
        if lines.ndim != 2:
            raise ValueError("Lines must be a 2D array")

        # Handle scalar offset
        if np.isscalar(offset):
            offset = np.arange(self.n_lines) * offset
        elif offset.shape != (self.n_lines,):
            raise ValueError("Offset must be a scalar or an array with shape (n_lines,)")

        # Apply the offset to each line
        lines_with_offset = lines + offset[:, np.newaxis]

        # Sort lines by index
        lines_with_offset = lines_with_offset[idx]

        # Generate x-coordinates
        x_coords = np.broadcast_to(np.arange(self.n_points), lines.shape)

        # Stack x and y coordinates
        positions = np.stack([x_coords, lines_with_offset], axis=-1).reshape(-1, 2)

        return positions

    def get_connect(self):
        a = np.arange(self.n_points-1)
        b = a + 1
        connect = np.tile(np.stack([a, b]), self.n_lines)
        connect = connect + np.expand_dims(np.repeat(np.arange(self.n_lines)*self.n_points, self.n_points-1), 0)
        return connect.T

    @property
    def n_lines(self):
        return self.lines.shape[0]

    @property
    def n_points(self):
        return self.lines.shape[1]

    def set_data(self, **kwargs):
        zorder = kwargs.pop('zorder', None)
        offset = kwargs.pop('offset', self.offset)

        if 'lines' in kwargs:
            self.lines = kwargs.pop('lines')
            kwargs['connect'] = self.get_connect()

        # parse zorder
        if zorder is None:
            zorder = np.zeros(self.n_lines)
        idx = np.argsort(-zorder)

        # sort lines and convert to vertex position array
        kwargs['pos'] = self.get_pos(self.lines, offset, idx)

        # define and sort color array if provided
        if 'color' in kwargs:
            kwargs['color'] = np.repeat(kwargs['color'][idx], self.n_points, axis=0)

        return super().set_data(**kwargs)
