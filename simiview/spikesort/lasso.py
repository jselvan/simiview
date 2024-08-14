import numpy as np
from vispy.scene.visuals import Line

from simiview.spikesort.points_in_poly import points_in_polygon
class LassoSelector:
    MIN_MOVE_UPDATE_THRESHOLD = 5
    def __init__(self, scatter_manager, callback=None, get_active_color=None):
        if get_active_color is None:
            get_active_color = lambda: 'yellow'
        self.get_active_color = get_active_color
        self.lasso_visual = Line(color=self.get_active_color(), width=2, method='gl', connect='strip')
        self.lasso_view = scatter_manager.widget.add_view()
        self.lasso_view.interactive = False  # Make the lasso view non-interactive
        self.lasso_view.add(self.lasso_visual)

        self.scatter_manager = scatter_manager
        self.selected_points = []
        self.lasso_points = []
        self.dragging = False
        self.last_position = None
        self.callback = callback
        self.active = False

    @property
    def points(self):
        points = self.scatter_manager.scatter.get_transform('visual', 'canvas').map(self.scatter_manager.points)
        points /= points[:,3:]
        return points

    def register_events(self, parent):
        parent.events.mouse_press.connect(self.on_mouse_press)
        parent.events.mouse_release.connect(self.on_mouse_release)
        parent.events.mouse_move.connect(self.on_mouse_move)

    def on_mouse_press(self, event):
        if self.active and event.button == 1:
            self.lasso_points = [event.pos]
            self.dragging = True
            self.last_position = None
            self.update_lasso()

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
            self.lasso_visual.set_data(trail, color=self.get_active_color(), width=2)

    def select_points(self, poly):
        if poly.shape[0] < 3:
            return

        indices = points_in_polygon(self.points, poly)
        self.selected_points = indices
        if self.callback is not None:
            self.callback(indices)