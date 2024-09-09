import numpy as np
import numpy.typing as npt
from vispy.scene.visuals import Line

class PathCollection(Line):
    def __init__(self, **kwargs):
        if 'pos' in kwargs:
            raise ValueError
        if 'connect' in kwargs:
            raise ValueError
        self.paths = None
        Line.__init__(self)
        if 'paths' in kwargs:
            self.set_data(**kwargs)

    @property
    def n_lines(self):
        return self.paths.shape[0]

    @property
    def n_points(self):
        return self.paths.shape[1]

    def get_pos(self, idx: np.ndarray | None = None) -> np.ndarray:
        """
        Generate (x, y) positions for points in multiple paths.

        Parameters
        ----------
        idx (np.ndarray):
            An array of indices to sort the paths by.

        Returns
        -------
        np.ndarray: A 2D array of shape (n_paths * n_points, 2) containing (x, y) positions.
        """
        # Sort paths by index
        paths = self.paths if idx is None else self.paths[idx]
        # Stack x and y coordinates
        positions = paths.reshape(-1, 2)

        return positions
    
    def get_connect(self):
        a = np.arange(self.n_points-1)
        b = a + 1
        connect = np.tile(np.stack([a, b]), self.n_lines)
        connect = connect + np.expand_dims(np.repeat(np.arange(self.n_lines)*self.n_points, self.n_points-1), 0)
        return connect.T


    def set_data(self, **kwargs):
        zorder = kwargs.pop('zorder', None)

        if 'paths' in kwargs:
            self.paths = kwargs.pop('paths')
            kwargs['connect'] = self.get_connect()

        # parse zorder
        if zorder is None:
            zorder = np.zeros(self.n_lines)
        idx = np.argsort(-zorder)

        # sort lines and convert to vertex position array
        kwargs['pos'] = self.get_pos(idx)

        # define and sort color array if provided either per-line or per-vertex
        if 'vertex_colors' in kwargs and 'color' in kwargs:
            raise ValueError("Cannot specify both 'vertex_colors' and 'color")
        if 'vertex_colors' in kwargs:
            vert_colors = kwargs.pop('vertex_colors')
            n_color_dims = vert_colors.shape[-1]
            vert_colors = vert_colors.reshape(self.n_lines, self.n_points, n_color_dims)
            vert_colors = vert_colors[idx].reshape(-1, n_color_dims)
            kwargs['color'] = vert_colors
        if 'color' in kwargs:
            kwargs['color'] = np.repeat(kwargs['color'][idx], self.n_points, axis=0)
        # optionally apply alpha values to color array if provided
        if 'alpha' in kwargs:
            alpha = kwargs.pop('alpha')
            alpha = np.repeat(alpha[idx], self.n_points)
            kwargs['color'][:, 3] = alpha

        return super().set_data(**kwargs)

class LineCollection(Line):
    def __init__(self, **kwargs):
        if 'pos' in kwargs:
            raise ValueError
        if 'connect' in kwargs:
            raise ValueError
        self.lines = None
        self.offset = kwargs.pop('offset', 0)
        Line.__init__(self)
        if 'lines' in kwargs:
            self.set_data(**kwargs)

    def get_pos(self, offset: npt.NDArray | None = None, idx: npt.NDArray | None = None) -> np.ndarray:
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
        lines = self.lines
        if offset is None:
            offset = self.offset

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
        if idx is not None:
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
        kwargs['pos'] = self.get_pos(offset, idx)

        # define and sort color array if provided either per-line or per-vertex
        if 'vertex_colors' in kwargs and 'color' in kwargs:
            raise ValueError("Cannot specify both 'vertex_colors' and 'color")
        if 'vertex_colors' in kwargs:
            vert_colors = kwargs.pop('vertex_colors')
            n_color_dims = vert_colors.shape[-1]
            vert_colors = vert_colors.reshape(self.n_lines, self.n_points, n_color_dims)
            vert_colors = vert_colors[idx].reshape(-1, n_color_dims)
            kwargs['color'] = vert_colors
        if 'color' in kwargs:
            kwargs['color'] = np.repeat(kwargs['color'][idx], self.n_points, axis=0)
        # optionally apply alpha values to color array if provided
        if 'alpha' in kwargs:
            alpha = kwargs.pop('alpha')
            alpha = np.repeat(alpha[idx], self.n_points)
            kwargs['color'][:, 3] = alpha

        return super().set_data(**kwargs)
    
    def get_closest_line(self, position : np.ndarray) -> int:
        """
        Get the index of the line closest to a given position.

        Parameters
        ----------
        position (np.ndarray):  
            A 1D array containing the (x, y) position to search for.

        Returns
        -------
        int: The index of the closest line.
        """
        # Calculate the distance between the position and each line
        pos = self.get_pos()
        distances = np.linalg.norm(pos - position, axis=1)
        idx = int(np.argmin(distances) // self.n_points)

        # Return the index of the line with the smallest distance
        return idx
    
    def get_closest_line_from_mouse_event(self, event) -> int:
        """
        Get the index of the line closest to a given mouse event.

        Parameters
        ----------
        event (MouseEvent):  
            A MouseEvent object containing the mouse position.

        Returns
        -------
        int: The index of the closest line.
        """
        # Get the position of the mouse event 
        # and transform to visual coordinates
        pos = self.get_transform('canvas', 'visual').map(event.pos)[:2]

        # Find the index of the line closest to the mouse position
        return self.get_closest_line(pos)