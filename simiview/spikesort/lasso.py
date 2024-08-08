import numpy as np
from vispy.scene.visuals import Line

class LassoSelector:
    MIN_MOVE_UPDATE_THRESHOLD = 5
    def __init__(self, scatter, points, container, callback=None):
        self.lasso_visual = Line(color='yellow', width=2, method='gl', connect='strip')
        self.lasso_view = container.add_view()
        self.lasso_view.interactive = False  # Make the lasso view non-interactive
        self.lasso_view.add(self.lasso_visual)

        self.scatter_data = points
        self.scatter = scatter
        self.selected_points = []
        self.lasso_points = []
        self.dragging = False
        self.last_position = None
        self.callback = callback
        self.active = False

    def on_mouse_press(self, event):
        if self.active and event.button == 1:
            self.lasso_points = [event.pos]
            self.dragging = True
            self.last_position = None
            self.update_lasso()

    def register_events(self, parent):
        parent.events.mouse_press.connect(self.on_mouse_press)
        parent.events.mouse_release.connect(self.on_mouse_release)
        parent.events.mouse_move.connect(self.on_mouse_move)

    def on_mouse_release(self, event):
        if self.dragging:
            self.dragging = False
            poly = self._get_lasso_poly(event.trail(), True)
            self.select_points(poly)
            self.update_lasso()
            self.active = False

    def on_mouse_move(self, event):
        if self.dragging and self._moved_from_last_position(event):
            self.last_position = event.pos
            self.update_lasso(event.trail())

    def _moved_from_last_position(self, event):
        if self.last_position is None:
            return True
        xf, yf = event.pos
        xi, yi = self.last_position
        return (abs(xf-xi) > self.MIN_MOVE_UPDATE_THRESHOLD or abs(yf-yi) > self.MIN_MOVE_UPDATE_THRESHOLD)

    def _get_lasso_poly(self, trail, closed):
        if closed:
            trail = np.insert(trail, len(trail), trail[0], axis=0)
        return trail

    def update_lasso(self, trail=None, closed=True):
        if trail is None:
            self.lasso_visual.set_data(np.empty((0, 2)))
        else:
            trail = self._get_lasso_poly(trail, closed)
            self.lasso_visual.set_data(trail, color='yellow', width=2)

    def select_points(self, poly):
        if poly.shape[0] < 3:
            return

        points = self.scatter.get_transform('visual', 'canvas').map(self.scatter_data)
        points /= points[:,3:]
        indices = self.points_in_polygon(points, poly)
        self.selected_points = indices
        if self.callback is not None:
            self.callback(indices)

    def points_in_polygon(self, points, poly):
        from matplotlib.path import Path
        path = Path(poly)
        inside = path.contains_points(points[:, :2])
        return np.where(inside)[0]
