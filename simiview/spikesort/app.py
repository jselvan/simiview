from pathlib import Path
import numpy as np
from vispy import scene
from vispy.scene.visuals import XYZAxis, Markers
from vispy.scene.cameras import ArcballCamera

import simianpy as simi
from simiview.spikesort.lasso import LassoSelector
from simiview.spikesort.linecollection import LineCollection
from simiview.spikesort.ccg_view_manager import CCGViewManager
from simiview.spikesort.single_channel_viewer import SingleChannelViewer
from simiview.spikesort.unit_view_manager import UnitViewManager
from simiview.util import scale_time
from simiview.spikesort.colours import COLOURS

class SpikeSortApp(scene.SceneCanvas):
    @simi.misc.add_logging
    def __init__(self, logger=None):
        # Initialize SceneCanvas with no data
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 600))
        self.unfreeze()
        self.logger = logger
        self.threads = {}

        self.data_directory = None
        self.save_path = None

        # Initialize data-related attributes
        self.points = None
        self.waveforms = None
        self.timestamps = None
        self.timestamps_ms = None
        self.clusters = None
        self.active_cluster = 0
        self.active_point = None

        # Set up the main grid layout
        self.grid = grid = self.central_widget.add_grid()
        self.pointcloud_container = grid.add_widget(row=0, col=0, row_span=5, col_span=2)
        self.pointcloud_container.interactive = True

        # Initialize view for 3D point cloud
        self.view = self.pointcloud_container.add_view()
        self.view.camera = ArcballCamera(fov=0, scale_factor=200)
        self.view.interactive = True
        self.view.border_color = 'red'
        axis = XYZAxis(parent=self.view.scene)
        self.view.add(axis)

        # Initialize view for waveform graph
        self.graph_widget = grid.add_widget(row=0, col=2, row_span=2)
        self.graph_view = self.graph_widget.add_view()
        self.graph_view.camera = 'panzoom'

        # Initialize unit manager and CCG manager
        self.units_widget = grid.add_widget(row=2, col=2)
        self.unit_manager = UnitViewManager(self, self.units_widget)

        self.ccg_widget = grid.add_widget(row=3, col=2, row_span=2)
        self.ccg_manager = CCGViewManager(self, self.ccg_widget)

        widget = grid.add_widget(row=5, col=0, col_span=3)
        view = widget.add_view()
        self.continuous_viewer = SingleChannelViewer(view, self.load_data, logger=self.logger)
        self.continuous_viewer.register_events(self)

        # Store home position of cameras
        self.view.camera.set_default_state()
        self.graph_view.camera.set_default_state()

        # Initialize scatter plot and lines for waveforms
        self.scatter = Markers()
        self.view.add(self.scatter)
        self.lines = LineCollection()
        self.graph_view.add(self.lines)

        # Lasso selector for selecting points in the scatter plot
        self.lasso = LassoSelector(self, self.pointcloud_container, callback=self.update_cluster)
        self.lasso.register_events(self)

        self.show()
    
    def set_parent_directory(self, path):
        # Create a save path directory if it does not exist
        self.data_directory = Path(path)
        if not self.data_directory.exists():
            self.data_directory.mkdir(parents=True)
        self.save_path = None
    
    def load_session(self, path, sig=None, channels=None):
        self.set_parent_directory(path)
        if sig is not None:
            self.continuous_viewer.sig = sig
            self.continuous_viewer.all_channels = channels

    def load_channel(self, name, index):
        if self.data_directory is None:
            raise ValueError("Parent directory not set")
        self.save_path = self.data_directory / name
        self.save_path.mkdir(parents=True, exist_ok=True)

        if self.continuous_viewer.sig is not None:
            self.continuous_viewer.channel_idx = index
            self.continuous_viewer.update_plot()

        if (self.save_path / 'waveforms.npy').exists() and (self.save_path / 'timestamps.npy').exists():
            waveforms = np.load(self.save_path / 'waveforms.npy')
            timestamps = np.load(self.save_path / 'timestamps.npy')
            if (self.save_path / 'clusters.npy').exists():
                clusters = np.load(self.save_path / 'clusters.npy')
            else:
                clusters = None
            if (self.save_path / 'points.npy').exists():
                points = np.load(self.save_path / 'points.npy')
            else:
                points = None
            self.load_data(waveforms, timestamps, clusters=clusters, points=points, save_waveforms=False)

    def load_data(self, waveforms, timestamps, clusters=None, points=None, save_waveforms=True):
        """Load data into the SpikeSortApp and update the visualizations."""
        # self.save_path = self.data_directory / name
        # self.save_path.mkdir(parents=True, exist_ok=True)

        if points is None:
            # If points are not provided, compute them from waveforms using PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            points = pca.fit_transform(waveforms)
            np.save(self.save_path / 'points.npy', points)
        self.points = points
        self.waveforms = waveforms
        self.timestamps = timestamps
        if save_waveforms:
            np.save(self.save_path / 'waveforms.npy', waveforms)
            np.save(self.save_path / 'timestamps.npy', timestamps)
        self.timestamps_ms = scale_time(self.timestamps, 's', 'ms')

        if clusters is not None:
            self.clusters = clusters
        else:
            self.clusters = np.zeros(self.points.shape[0], dtype=np.int8)
            np.save(self.save_path / 'clusters.npy', self.clusters)

        # Update visual components
        self.update_visuals()
        self.update_colors()

        # Update the unit manager and CCG manager with new data
        self.unit_manager.update_units_view()
        self.ccg_manager.update_ccgs()
    
    def save_data(self):
        """Save the current data to the save path."""
        np.save(self.save_path / 'waveforms.npy', self.waveforms)
        np.save(self.save_path / 'timestamps.npy', self.timestamps)
        np.save(self.save_path / 'clusters.npy', self.clusters)
        np.save(self.save_path / 'points.npy', self.points)

    def update_visuals(self):
        """Update the visuals with the current data."""
        if self.points is None or self.waveforms is None:
            return

        # Update the scatter plot with new points
        self.scatter.set_data(self.points)

        # Update the waveform lines
        self.lines.set_data(lines=self.waveforms)

        # Update camera views
        self.view.camera.set_default_state()
        minval, maxval = self.waveforms.min(None), self.waveforms.max(None)
        self.graph_view.camera.rect = (0, minval), (40, maxval - minval)
        self.graph_view.camera.set_default_state()

    def update_cluster(self, indices):
        """Update clusters based on selected indices."""
        if self.state == 'add':
            self.clusters[indices] = self.active_cluster
        elif self.state == 'remove':
            self.clusters[indices] = 0
        elif self.state == 'replace':
            self.clusters[self.clusters == self.active_cluster] = 0
            self.clusters[indices] = self.active_cluster
        elif self.state == 'invalidate':
            self.clusters[indices] = -1
        np.save(self.save_path / 'clusters.npy', self.clusters)
        self.update_colors()
        self.ccg_manager.update_ccgs()
        self.unit_manager.update_units_view()

    def update_colors(self):
        """Update colors of the scatter plot and lines based on clusters."""
        if self.points is None:
            return

        colors = np.ones((self.points.shape[0], 4), dtype=np.float32)
        for cluster, color in COLOURS.items():
            indices = self.clusters == cluster
            colors[indices, :3] = color
        z_order = self.clusters.copy()
        if self.active_point is not None:
            colors[:, -1] = .1
            colors[self.active_point, -1] = 1.
            z_order[self.active_point] = 100
        else:
            colors[:, -1] = 1.

        # Update the colors of scatter plot and lines
        self.scatter.set_data(self.points, face_color=colors, edge_color=None)
        self.lines.set_data(color=colors, zorder=z_order)

    def reset_cameras(self):
        """Reset cameras to their default positions."""
        self.view.camera.reset()
        self.graph_view.camera.reset()

    def on_key_press(self, event):
        """Handle key press events for lasso and cluster operations."""
        handled = True
        if 'Control' in event.modifiers and event.key == "d":
            self.lasso.active = True
            self.state = 'remove'
        elif 'Control' in event.modifiers and event.key == "a":
            self.lasso.active = True
            self.state = 'add'
        elif 'Control' in event.modifiers and event.key == "n":
            self.lasso.active = True
            self.active_cluster = self.clusters.max() + 1
            self.state = 'add'
        elif event.key == 'h':
            self.reset_cameras()
        else:
            handled = False
            for key in '1234':
                if event.key == key:
                    self.active_cluster = int(key)
                    handled = True
        if not handled:
            return event

    def on_mouse_move(self, event):
        """Handle mouse movement for highlighting points."""
        if 'Alt' in event.modifiers and self.points is not None:
            # Find the nearest point in the scatter plot
            pos = event.pos
            points = self.scatter.get_transform('visual', 'canvas').map(self.points)
            points = points[:, :2] / points[:, 3:]

            distances = np.linalg.norm(points - pos, axis=1)
            self.active_point = np.argmin(distances)
            self.update_colors()
        else:
            if self.active_point is not None:
                self.active_point = None
                self.update_colors()

    def close(self):
        """Clean up before closing the application."""
        self.lasso.unregister_events(self)
        for thread in self.threads.values():
            thread.stop()
        super().close()
