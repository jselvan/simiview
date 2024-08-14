import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, 
    QComboBox, QWidget, QVBoxLayout, QLabel
)

from vispy.scene.visuals import XYZAxis, Markers
from vispy.scene import Widget
from vispy.scene.cameras import ArcballCamera

from simiview.spikesort.colours import COLOURS

class Toolbar(QWidget):
    def __init__(self, parent=None, dimensions=None, dimension_changed_callback=None):
        super().__init__(parent)
        
        # Create layout for the custom widget
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create and configure the toolbar
        self.toolbar = QToolBar("My Toolbar")
        layout.addWidget(self.toolbar)
        
        # Create three dropdowns
        if dimensions is None:
            dimensions = ["PCA 1", "PCA 2", "PCA 3"]
        self.dimensions = dimensions
        self.dimension_changed_callback = dimension_changed_callback
        self.combo_boxes = {}
        for idx, dim in enumerate("XYZ"):
            label = QLabel(f"{dim}:")
            label.setStyleSheet("font-weight: bold; color: white;")
            self.toolbar.addWidget(label)
            combo_box = QComboBox()
            combo_box.addItems(self.dimensions)
            combo_box.setCurrentIndex(idx)
            combo_box.currentIndexChanged.connect(self.on_combobox_changed)
            self.combo_boxes[dim] = combo_box
            self.toolbar.addWidget(combo_box)

    def on_combobox_changed(self):
        dimensions = [box.currentText() for box in self.combo_boxes.values()]
        if self.dimension_changed_callback is not None:
            self.dimension_changed_callback(dimensions)

class PointCloudManager:
    DIMENSIONS = [
        "PCA 1", "PCA 2", "PCA 3", "Timestamp", "Peak Amplitude", 
        "Peak Time", "Valley Amplitude", "Valley Time"
    ]
    def __init__(self, parent, widget, callback=None):
        self.active_dimensions = self.DIMENSIONS[:3]
        self.points = None
        # Store reference to parent SpikeSortApp
        self.parent = parent
        self.widget = widget

        # Initialize view for 3D point cloud
        # make the widget a grid with one small row and one big one
        self.grid = self.widget.add_grid()
        self.toolbar = self.grid.add_widget(row=0, col=0)
        self.toolbar_widget = Toolbar(
            self.toolbar.canvas.native,
            dimensions=self.DIMENSIONS,
            dimension_changed_callback=self.update_active_dimensions
        )
        self.view = self.grid.add_view(row=1, col=0, row_span=15)
        self.view.camera = ArcballCamera(fov=0, scale_factor=200)
        self.view.interactive = True
        self.view.border_color = 'red'

        # Add XYZ axis to the view
        axis = XYZAxis(parent=self.view.scene)
        self.view.add(axis)

        # Initialize scatter plot
        self.scatter = Markers()
        self.view.add(self.scatter)

        # # Initialize lasso selector for selecting points in the scatter plot
        # self.lasso = LassoSelector(self.parent, container, callback=callback)
        # self.lasso.register_events(self.parent)

        # Store home position of camera
        self.view.camera.set_default_state()
        self.view.events.mouse_move.connect(self.on_mouse_move)

    def update_active_dimensions(self, dimensions):
        self.active_dimensions = dimensions
        self.update_points()

    def update_points(self):
        """Update the visuals with the current data."""
        self.points = self.parent.get_points(self.active_dimensions)
        if self.points is not None:
            self.update_colors()

    def update_colors(self):
        """Update colors of the scatter plot based on clusters."""
        colors = self.parent.get_colors()
        self.scatter.set_data(self.points, face_color=colors, edge_color=None)

    def reset_camera(self):
        """Reset camera to its default position."""
        self.view.camera.reset()

    def close(self):
        """Clean up before closing the application."""
        self.lasso.unregister_events(self.parent)

    def on_mouse_move(self, event):
        """Handle mouse movement for highlighting points."""
        if 'Alt' in event.mouse_event.modifiers and self.points is not None:
            # Find the nearest point in the scatter plot
            pos = event.mouse_event.pos
            points = self.scatter.get_transform('visual', 'canvas').map(self.points)
            points = points[:, :2] / points[:, 3:]

            distances = np.linalg.norm(points - pos, axis=1)
            active_point = np.argmin(distances)
            self.parent.set_active_point(active_point)
        else:
            if self.parent.active_point is not None:
                self.parent.set_active_point(None)
