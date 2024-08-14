import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.scene import PanZoomCamera
from PyQt5 import QtWidgets, QtCore, QtGui

from simiview.spikesort.colours import COLOURS

class UnitViewManager:
    def __init__(self, parent, widget):
        self.parent = parent
        self.widget = widget

        self.unit_views = {}
        self.unit_waveforms = {}
        self.units_grid = self.widget.add_grid()

        self.selected = set()
        self.active = None

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
        view.camera = PanZoomCamera(aspect=1)
        view.camera.interactive = False
        view.camera.rect = self.waveform_rect
        view.border_color = 'black'
        view.padding = 5
        view.events.mouse_press.connect(self.mouse_press_handler_gen(cluster))
        self.unit_views[cluster] = view

    def set_selected(self, cluster):
        if cluster in self.selected:
            self.selected.remove(cluster)
        else:
            self.selected.add(cluster)
        self.update_viewbox_states()
    def reset_selected(self):
        self.selected = set()
        self.update_viewbox_states()

    def update_viewbox_states(self):
        for c, view in self.unit_views.items():
            if c in self.selected:
                view.bgcolor = (0.2, 0.2, 0.2, 1)
            else:
                view.bgcolor = 'black'
            if c == self.active:
                view.border_color = COLOURS[c]
            else:
                view.border_color = 'black'

    def set_active(self, cluster):
        self.active = cluster
        self.update_viewbox_states()

    def mouse_press_handler_gen(self, cluster):
        def mouse_press_handler(event):
            if event.button == 1 and 'Shift' in event.mouse_event.modifiers:
                self.set_selected(cluster)
                self.set_active(cluster)
            elif event.button == 1:
                self.reset_selected()
                self.set_active(cluster)
            elif event.button == 2:
                # self.set_selected(cluster)
                # use pyqt5 to create a context menu
                if cluster not in self.selected:
                    self.reset_selected()
                    self.set_active(cluster)
                self.customContextMenu()
        return mouse_press_handler

    def customContextMenu(self):
        # widget = self.widget.canvas.native
        # widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # widget.customContextMenuRequested.connect(self.emptySpaceMenu)
        menu = QtWidgets.QMenu(self.widget.canvas.native)
        if self.selected:
            # allow merge
            menu.addAction("Merge selected", self.customAction)
        # check hidden state and set appropriate action
        # menu.addAction("Hide", self.customAction)
        # menu.addAction("Show", self.customAction)
        menu.addAction("Invalidate", self.customAction)
        menu.exec_(QtGui.QCursor.pos())

    def customAction(self):
        print("Custom action on", self.selected if self.selected else self.active)

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