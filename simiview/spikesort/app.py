from itertools import combinations
from pathlib import Path
import numpy as np
from vispy import scene
from vispy.scene import visuals

from simiview.spikesort.barplot import BarPlot
from simiview.spikesort.lasso import LassoSelector
from simiview.spikesort.linecollection import LineCollection
from simiview.spikesort.ccg_matrix import ccg_matrix
from simiview.util import scale_time

from threading import Thread


class SpikeSortApp(scene.SceneCanvas):
    COLOURS = {
        -1: [0.412, 0.412, 0.412],
         0: [1.000, 1.000, 1.000],
         1: [0.102, 0.522, 1.000],
         2: [0.831, 0.067, 0.349],
         3: [0.102, 1.000, 0.102],
         4: [0.365, 0.227, 0.608]
    }
    def __init__(self, points, waveforms, timestamps, save_path, clusters=None):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 600))
        self.unfreeze()
        self.threads = {}

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        self.points = points
        self.waveforms = waveforms
        self.timestamps = timestamps
        self.timestamps_ms = scale_time(self.timestamps, 's', 'ms')
        if clusters is not None:
            self.clusters = clusters
        else:
            self.clusters = np.zeros(self.points.shape[0], dtype=np.int8)
        self.active_cluster = 0
        self.active_point = None

        self.grid = grid = self.central_widget.add_grid()
        self.pointcloud_container = grid.add_widget(row=0, col=0, row_span=2, col_span=2)
        self.pointcloud_container.interactive = True


        self.view = self.pointcloud_container.add_view()
        # self.view.camera = 'turntable'
        self.view.camera = 'arcball'
        self.view.interactive = True
        self.view.border_color = 'red'
        axis = visuals.XYZAxis(parent=self.view.scene)
        self.view.add(axis)
        ###
        self.graph_widget = grid.add_widget(row=0, col=2)
        self.graph_view = self.graph_widget.add_view()
        self.graph_view.camera = 'panzoom'
        minval, maxval = self.waveforms.min(None), self.waveforms.max(None)
        self.graph_view.camera.rect = (0, minval), (40, maxval-minval)
        ###
        self.ccg_widget = grid.add_widget(row=1, col=2)
        self.ccg_grid = self.ccg_widget.add_grid()
        self.ccg_views = {}
        self.ccg_bars = {}
        self.update_ccg_grid()

        # store home position of cameras
        self.view.camera.set_default_state()
        self.graph_view.camera.set_default_state()

        ####
        self.update_ccgs()
        # Set up the scatter plot and lines
        self.scatter = visuals.Markers()
        self.scatter.set_data(self.points)
        self.view.add(self.scatter)
        self.lines = LineCollection(waveforms)
        self.graph_view.add(self.lines)
        self.update_colors()
        # Lasso visual to draw the lasso polygon
        self.lasso = LassoSelector(self.scatter, self.points, self.pointcloud_container, self, callback=self.lasso_callback)
        self.lasso.register_events(self)

        self.show()
    
    @classmethod
    def from_directory(cls, directory, save_path=None):
        directory = Path(directory)
        if save_path is None:
            save_path = directory
        else:
            save_path = Path(save_path)
        waveforms = np.load(directory/'waveforms.npy')
        timestamps = np.load(directory/'timestamps.npy')
        if (directory/'clusters.npy').exists():
            clusters = np.load(directory/'clusters.npy')
        else:
            clusters = np.zeros(waveforms.shape[0], dtype=np.int8)
        if not (directory/'pca.npy').exists():
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            points = pca.fit_transform(waveforms)
            np.save(directory/'pca.npy', points)
        else:
            points = np.load(directory/'pca.npy')
        
        return cls(points, waveforms, timestamps, save_path=save_path, clusters=clusters)

    def lasso_callback(self, indices):
        self.update_cluster(indices, self.state)
        self.update_colors()

    def update_cluster(self, indices, operation):
        if operation == 'add':
            self.clusters[indices] = self.active_cluster
        elif operation == 'remove':
            self.clusters[indices] = 0
        elif operation == 'replace':
            self.clusters[self.clusters == self.active_cluster] = 0
            self.clusters[indices] = self.active_cluster
        elif operation == 'invalidate':
            self.clusters[indices] = -1
        np.save(self.save_path/'clusters.npy', self.clusters)
        self.update_colors()
        self.update_ccgs()

    def _add_ccg_view(self, a, b):
        view = self.ccg_grid.add_view(row=a-1, col=b-1)
        view.camera = 'panzoom'
        view.camera.rect = (-10, 0, 20, 10)
        # view.camera.aspect = 1
        # view.camera.flip = (0, -1)
        # view.camera.interactive = False
        view.border_color = 'red'
        self.ccg_views[(a, b)] = view

    def update_ccg_grid(self):
        unique_clusters = self.get_sorted_cluster_ids()
        existing_views = list(self.ccg_views.keys())
        for a, b in existing_views:
            if a not in unique_clusters or b not in unique_clusters:
                widget = self.ccg_views.pop((a, b))
                if hasattr(widget.parent, 'remove'):
                    widget.parent.remove(widget)
        for a in unique_clusters:
            if (a, a) not in self.ccg_views:
                self._add_ccg_view(a, a)
        for (a, b) in combinations(unique_clusters, 2):
            if (a, b) not in self.ccg_views:
                self._add_ccg_view(a, b)
    
    def update_ccgs(self):
        self.threads["ccg"] = Thread(target=self._update_ccgs, name="update_ccgs", daemon=True)
        self.threads["ccg"].start()

    def get_sorted_cluster_ids(self):
        unique_clusters = np.unique(self.clusters)
        unique_clusters = unique_clusters[unique_clusters>0]
        unique_clusters = unique_clusters.tolist()
        return unique_clusters
    
    def _compute_ccgs(self):
        unique_clusters = self.get_sorted_cluster_ids()
        lags, ccg = ccg_matrix(
            self.timestamps_ms, 
            self.clusters, 
            bin_size=0.2, 
            max_lag=10, 
            unitids=unique_clusters
        )
        np.save(self.save_path/'lags.npy', lags)
        np.save(self.save_path/'ccg.npy', ccg)
        return lags, ccg
        

    def _update_ccgs(self):
        # get sorted unique clusters
        lags, ccg = self._compute_ccgs()
        self.update_ccg_grid()
        for (a, b), pair_ccg in ccg.items():
            if (a, b) not in self.ccg_bars:
                self.ccg_bars[a,b] = BarPlot(lags, pair_ccg, color=self.COLOURS[a], parent=self.ccg_views[a,b].scene)
            else:
                self.ccg_bars[a,b].set_data(x=lags, y=pair_ccg, color=self.COLOURS[a])
        self.update()

    def update_colors(self):
        colors = np.ones((self.points.shape[0], 4), dtype=np.float32)
        for cluster, color in self.COLOURS.items():
            indices = self.clusters == cluster
            colors[indices, :3] = color
        z_order = self.clusters.copy()
        if self.active_point is not None:
            colors[:, -1] = .1
            colors[self.active_point, -1] = 1.
            z_order[self.active_point] = 100
        else:
            colors[:, -1] = 1.

        # # update the colors of scatter plot and lines
        self.scatter.set_data(self.points, face_color=colors, edge_color=None)
        self.lines.set_data(color=colors, zorder=z_order)

    def reset_cameras(self):
        self.view.camera.reset()
        self.graph_view.camera.reset()

    def on_key_press(self, event):
        if 'Control' in event.modifiers and event.key == "d":
            self.lasso.active = True
            self.state = 'remove'
        elif 'Control' in event.modifiers and event.key == "a":
            self.lasso.active = True
            # self.active_cluster = 1
            self.state = 'add'
        elif 'Control' in event.modifiers and event.key == "n":
            self.lasso.active = True
            self.active_cluster = self.clusters.max() + 1
            self.state = 'add'
        elif event.key=='h':
            self.reset_cameras()
        for key in '1234':
            if event.key == key:
                self.active_cluster = int(key)

    def on_mouse_move(self, event):
        if 'Alt' in event.modifiers:
            # find nearest point in the scatter plot
            pos = event.pos
            points = self.scatter.get_transform('visual', 'canvas').map(self.points)
            points =  points[:, :2] / points[:,3:]

            distances = np.linalg.norm(points - pos, axis=1)
            self.active_point = np.argmin(distances)
            self.update_colors()
        else:
            if self.active_point is not None:
                self.active_point = None
                self.update_colors()
    
    def __exit__(self, type, value, traceback):
        self.threads["ccg"].join()
        return super().__exit__(type, value, traceback)