import numpy as np
from vispy.scene.visuals import InfiniteLine

from itertools import combinations

from simiview.spikesort.colours import COLOURS
from simiview.spikesort.barplot import BarPlot
from simiview.spikesort.ccg_matrix import ccg_matrix


class CCGViewManager:
    def __init__(self, parent, widget):
        self.parent = parent
        self.widget = widget
        self.ccg_views = {}
        self.ccg_bars = {}
        self.ccg_grid = self.widget.add_grid()

    @property
    def save_path(self):
        return self.parent.save_path

    @property
    def timestamps_ms(self):
        return self.parent.timestamps_ms

    @property
    def clusters(self):
        return self.parent.clusters

    def _add_ccg_view(self, a, b):
        view = self.ccg_grid.add_view(row=a - 1, col=b - 1)
        view.camera = 'panzoom'
        view.camera.rect = (-10, 0, 20, 1)
        view.border_color = 'red'
        InfiniteLine(-1, color=(1., 1., 1., 1.), parent=view.scene)
        InfiniteLine(1, color=(1., 1., 1., 1.), parent=view.scene)
        self.ccg_views[(a, b)] = view

    def update_ccg_grid(self):
        unique_clusters = self.get_sorted_cluster_ids()
        existing_views = list(self.ccg_views.keys())
        for a, b in existing_views:
            if a not in unique_clusters or b not in unique_clusters:
                widget = self.ccg_views.pop((a, b))
                try:
                    self.ccg_grid.remove_widget(widget)
                except Exception as e:
                    print(e)
        for a in unique_clusters:
            if (a, a) not in self.ccg_views:
                self._add_ccg_view(a, a)
        for (a, b) in combinations(unique_clusters, 2):
            if (a, b) not in self.ccg_views:
                self._add_ccg_view(a, b)

    def update_ccgs(self):
        self.update_ccg_grid()
        if len(self.get_sorted_cluster_ids()) == 0:
            return
        lags, ccg = self._compute_ccgs()
        for (a, b), pair_ccg in ccg.items():
            if (a, b) not in self.ccg_bars:
                self.ccg_bars[a, b] = BarPlot(lags, pair_ccg, color=COLOURS[a], parent=self.ccg_views[a, b].scene)
            else:
                self.ccg_bars[a, b].set_data(x=lags, y=pair_ccg, color=COLOURS[a])
        for view in self.ccg_views.values():
            view.bgcolor = 'black'

    def get_sorted_cluster_ids(self):
        unique_clusters = np.unique(self.clusters)
        unique_clusters = unique_clusters[unique_clusters > 0]
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
        # np.save(self.save_path / 'lags.npy', lags)
        # np.save(self.save_path / 'ccg.npy', ccg)
        return lags, ccg

