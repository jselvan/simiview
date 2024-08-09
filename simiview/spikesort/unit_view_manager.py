import numpy as np
from vispy.scene.visuals import Line, Text

from simiview.spikesort.colours import COLOURS

class UnitViewManager:
    def __init__(self, parent, widget):
        self.parent = parent
        self.widget = widget

        self.unit_views = {}
        self.unit_waveforms = {}
        self.units_grid = self.widget.add_grid()

        # select all clusters but the noise cluster by default
        if self.clusters is not None:
            self.selected = set(self.clusters[self.clusters != -1])
        else:
            self.selected = set()

    @property
    def waveform_xy(self):
        return 0, self.waveforms.min()
    
    @property
    def waveform_rect(self):
        return (0, self.waveforms.min()), (40, self.waveforms.max() - self.waveforms.min())

    @property
    def waveforms(self):
        return self.parent.waveforms

    @property
    def clusters(self):
        return self.parent.clusters

    def _add_units_view(self, cluster):
        view = self.units_grid.add_view(row=0, col=cluster)
        view.camera = 'panzoom'
        view.camera.rect = self.waveform_rect
        view.border_color = COLOURS[cluster]
        view.events.mouse_press.connect(self.mouse_press_handler_gen(cluster))
        self.unit_views[cluster] = view

    def set_selected(self, cluster):
        if cluster in self.selected:
            self.selected.remove(cluster)
        else:
            self.selected.add(cluster)
        for c, view in self.unit_views.items():
            if c in self.selected:
                view.bgcolor = 'black'
            else:
                view.bgcolor = 'grey'

    def set_active(self, cluster):
        for c, view in self.unit_views.items():
            if c == cluster:
                view.bgcolor = 'white'
            else:
                view.bgcolor = 'black'

    def mouse_press_handler_gen(self, cluster):
        def mouse_press_handler(event):
            if event.button == 1:
                self.set_active(cluster)
            elif event.button == 2:
                self.set_selected(cluster)
        return mouse_press_handler

    def update_units_grid(self):
        existing_views = list(self.unit_views.keys())
        clusters = np.unique(self.clusters)
        for unit in existing_views:
            if unit not in clusters:
                widget = self.unit_views.pop(unit)
                if hasattr(widget.parent, 'remove'):
                    widget.widget.remove(widget)
        for cluster in clusters:
            if cluster == -1:
                continue
            if cluster not in self.unit_views:
                self._add_units_view(cluster)

    def _compute_waveform_data(self):
        unique_clusters = np.unique(self.clusters)
        waveform_data = {}
        t = np.arange(self.waveforms.shape[1])
        for cluster in unique_clusters:
            if cluster == -1:
                continue
            mean_ = self.waveforms[self.clusters == cluster].mean(axis=0)
            mean_ = np.array([t, mean_]).T
            waveform_data[cluster] = {
                'mean': mean_,
                'count': (self.clusters == cluster).sum()
            }
        return waveform_data

    def update_units_view(self):
        #TODO: does not remove waveforms from empty clusters
        waveform_data = self._compute_waveform_data()
        self.update_units_grid()
        for cluster, wf_info in waveform_data.items():
            label_text = "N={count}".format(**wf_info)
            if cluster not in self.unit_waveforms:
                line = Line(
                    wf_info['mean'],
                    color=COLOURS[cluster],
                    parent=self.unit_views[cluster].scene
                )

                x, y = self.waveform_xy
                text_xy = x, y * 0.9
                label = Text(label_text,
                             color='w',
                             anchor_x='left',
                             parent=self.unit_views[cluster].scene,
                             pos=text_xy
                             )
                self.unit_waveforms[cluster] = {
                    'line': line,
                    'label': label
                }
            else:
                self.unit_waveforms[cluster]['line'].set_data(wf_info['mean'])
                self.unit_waveforms[cluster]['label'].text = label_text