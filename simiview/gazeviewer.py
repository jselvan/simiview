from simiview.util.linecollection import LineCollection, PathCollection
from vispy.scene import Widget

from simianpy.signal import sosFilter

filt = sosFilter('lowpass', 6, 50, 1000)

class GazeViewer(Widget):
    def __init__(self):
        super().__init__()
        self.unfreeze()
        self.grid = self.add_grid()

        self.views = {}
        self.traces = {}

        for idx, component in enumerate(['eyeh', 'eyev']):
            view = self.grid.add_view(row=idx, col=0)
            view.camera = 'panzoom'
            view.interactive = True
            view.camera.rect = (0, -15), (500, 30)
            trace = LineCollection()
            view.add(trace)
            self.views[component] = view
            self.traces[component] = trace
        
        self.views['gaze'] = gaze_view = self.grid.add_view(row=0, col=1, row_span=2, col_span=2)
        self.traces['gaze'] = gaze_trace = PathCollection()
        gaze_view.camera = 'panzoom'
        gaze_view.interactive = True
        gaze_view.camera.rect = (-15, -15), (30, 30)
        gaze_view.add(gaze_trace)


        self.gaze_data = None
        self.filter_enabled = False
        self.selected_lines = None
        self.freeze()

    def register_events(self, parent):
        # for component, view in self.views.items():
        #     view.events.mouse_press.connect(lambda e: self.on_mouse_press(e, component))
        self.views['eyeh'].events.mouse_press.connect(lambda e: self.on_mouse_press(e, 'eyeh'))
        self.views['eyev'].events.mouse_press.connect(lambda e: self.on_mouse_press(e, 'eyev'))
        parent.events.key_press.connect(self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'f':
            self.filter_enabled = not self.filter_enabled
            self.update_traces()

    def update_traces(self):
        if self.gaze_data is None:
            return
        colors = np.ones((self.gaze_data.shape[0], 4), dtype=np.float32)
        zorder = np.zeros(self.gaze_data.shape[0])
        if self.selected_lines is not None:
            colors[:, 3] = 0.1
            colors[self.selected_lines] = (1, 0, 0, 1)
            zorder[self.selected_lines] = 1
        for component in ['eyeh', 'eyev']:
            component_idx = 0 if component == 'eyeh' else 1
            traces = self.gaze_data[:, :, component_idx]
            if self.filter_enabled:
                traces = filt(traces, axis=-1)
            self.traces[component].set_data(lines=traces, color=colors, zorder=zorder)
        self.traces['gaze'].set_data(paths=self.gaze_data, color=colors, zorder=zorder)

    def on_mouse_press(self, event, component):
        if (event.button == 1 
            and 'Control' in event.mouse_event.modifiers 
            and self.gaze_data is not None):
            lineidx = self.traces[component].get_closest_line_from_mouse_event(event.mouse_event)
            self.selected_lines = lineidx
            self.update_traces()

    def load_data(self, gaze_data):
        self.selected_lines = None
        self.gaze_data = gaze_data
        # self.view.camera.rect = (0, -15), (gaze_data.shape[1], 30)
        self.update_traces()



from vispy import scene#, app
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QMenu, QAction, QActionGroup,
    QHBoxLayout, QTableView, QSizePolicy,
    QAbstractScrollArea
)
from PyQt5.QtCore import (Qt, QAbstractTableModel, QPoint)

class DataModel(QAbstractTableModel):
    index_col_map = {
        0: 'date',
        1: 'trialid',
        2: 'outcome'
    }
    def __init__(self, data):
        super(DataModel, self).__init__()
        self._data = data

    def setData(self, data):
        self._data = data
        self.layoutChanged.emit()

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
    def rowCount(self, index):
        return self._data.shape[0]
    def columnCount(self, index):
        return self._data.shape[1]
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Gaze Viewer")
        self.setGeometry(0, 0, 600, 600)
        # Set up menu bar
        self.menu_bar = self.menuBar()
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)
        
        load_action = QAction("&Load", self)
        load_action.triggered.connect(self.load_data)
        self.file_menu.addAction(load_action)

        help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(help_menu)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.table = QTableView()
        self.table_model = DataModel(pd.DataFrame({'date': [], 'trialid': [], 'outcome': []}))
        self.table.setModel(self.table_model)
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table.horizontalHeader().sectionClicked.connect(self.header_clicked)
        self.table.verticalHeader().sectionClicked.connect(self.index_clicked)

        self.layout.addWidget(self.table)

        self.gaze_viewer_widget = QWidget()
        self.gaze_viewer_widget.setMinimumSize(600, 600)
        # make gaze viewer widget resizable
        self.gaze_viewer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.gaze_viewer_widget)

        self.gaze_viewer_canvas = scene.SceneCanvas(keys='interactive', show=True, parent=self.gaze_viewer_widget)
        self.gaze_viewer = GazeViewer()
        self.gaze_viewer.register_events(self.gaze_viewer_canvas)
        self.gaze_viewer_canvas.central_widget.add_widget(self.gaze_viewer)

    def load_data(self):
        traces = np.load('eyedata.npy')
        data = pd.read_csv('eyedata.csv', index_col=0)
        self.table_model.setData(data)
        self.gaze_viewer.load_data(traces)
    
    def index_clicked(self, _):
        idx = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        self.gaze_viewer.selected_lines = idx
        self.gaze_viewer.update_traces()

    def header_clicked(self, logicalIndex):
        # Create a context menu
        menu = QMenu(self.table)

        # add filter submenu
        filter_submenu = menu.addMenu("Filter")
        filter_submenu.addAction("Filter by value")
        filter_submenu.addAction("Filter by range")

        # add sort submenu
        sort_submenu = menu.addMenu("Sort")
        action_group = QActionGroup(self)
        action1 = QAction("Sort ascending", self, checkable=True)
        action2 = QAction("Sort descending", self, checkable=True)

        action_group.addAction(action1)
        action_group.addAction(action2)
        action_group.setExclusive(True) 
        sort_submenu.addActions(action_group.actions())

        # Get the position of the clicked header
        root = self.table.horizontalHeader().pos()
        x_offset = self.table.horizontalHeader().sectionPosition(logicalIndex)
        y_offset = self.table.horizontalHeader().height()
        pos = self.table.mapToGlobal(root + QPoint(x_offset, y_offset))
        # Show the menu under the clicked header
        menu.exec_(pos)


    # def header_clicked(self, idx):
    #     """Called when a column header is clicked."""
    #     column_name = self.table_model.headerData(idx, Qt.Horizontal, Qt.DisplayRole)
    #     menu = QMenu(self)
    #     filter_action = menu.addAction("Filter")
    #     # draw menu below the column header
    #     pos = self.table.horizontalHeader().sectionViewportPosition(idx)
    #     pos = self.table.mapToGlobal(pos)
    #     menu.exec_(pos)
    #     print(f'Column header clicked: {column_name}')
    #     # filter_dialog = FilterDialog(column_name, self)

    #     # if filter_dialog.exec_() == QDialog.Accepted:
    #     #     filter_value = filter_dialog.get_filter_value()
    #     #     self.table_model.apply_filter(idx, filter_value)


app = QApplication([])
window = MainWindow()
window.show()
app.exec_()