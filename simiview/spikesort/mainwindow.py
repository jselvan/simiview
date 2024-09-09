from pathlib import Path
import warnings

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QHBoxLayout,
    QWidget, QAction, QFileDialog, QTableView, QHeaderView,
    QAbstractItemView, QTableWidgetItem, QTableWidget, QMenuBar, QMenu,
    QCheckBox, QAbstractItemView, QHeaderView, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from neo.io import SpikeGadgetsIO, Plexon2IO
import quantities as pq
import numpy as np

import simianpy as simi
from simiview.spikesort.app import SpikeSortApp
from simiview.spikesort.single_channel_viewer import SingleChannelViewer

class MainWindow(QMainWindow):
    @simi.misc.add_logging
    def __init__(self, logger=None):
        super(MainWindow, self).__init__()
        self.logger = logger
        self.setWindowTitle("Spike Sort Application")
        self.setGeometry(0, 0, 200, 1030)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Set up table view for channels
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Channels', 'Bad Channel'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.select_channel)
        self.layout.addWidget(self.table)

        # Container for VisPy app
        # self.vispy_widget = QWidget()
        # self.vispy_layout = QVBoxLayout(self.vispy_widget)
        # self.layout.addWidget(self.vispy_widget)

        # Set up menu bar
        self.menu_bar = self.menuBar()
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)
        
        load_action = QAction("&Load", self)
        load_action.triggered.connect(self.load_data)
        self.file_menu.addAction(load_action)

        save_action = QAction("&Save", self)
        save_action.triggered.connect(self.save_data)
        self.file_menu.addAction(save_action)

        help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(help_menu)

        version_action = QAction("&Version", self)
        version_action.triggered.connect(self.show_version)
        help_menu.addAction(version_action)

        doc_action = QAction("&Documentation", self)
        doc_action.triggered.connect(self.show_documentation)
        help_menu.addAction(doc_action)

        # Initial VisPy app setup
        self.spike_sort_app = SpikeSortApp(logger=self.logger)
        # self.continuous_viewer = SingleChannelViewer()
        self.current_file = None
        self.current_data = None
        self.data_path = None

    def get_channel_indices(self, channel_names):
        return self.current_file.channel_name_to_index(0, channel_names)

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, file_ext = QFileDialog.getOpenFileName(self, 
            "Open Data File", "", 
            "Spike Gadgets Rec File (*.rec);;Plexon Files (*.pl2);;All Files (*)", 
            options=options)
        ftype_handlers = {
            "Spike Gadgets Rec File (*.rec)": SpikeGadgetsIO,
            "Plexon Files (*.pl2)": Plexon2IO
        }
        if file_name:
            if file_ext in ftype_handlers:
                self.current_file = ftype_handlers[file_ext](file_name)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            # self.current_file = SpikeGadgetsIO(file_name)
            self.signal_data = self.current_file.read_block(lazy=True).segments[0].analogsignals[0]
            self.data_path = Path(file_name).with_suffix('.simiview') / 'spikesort'

            if file_ext == "Plexon Files (*.pl2)":
                channels = [channel for channel in self.current_file.header['signal_channels']['name'] if 'WB' in channel]
                warnings.warn("Plexon files are not fully supported. Do not use common median rejection.")
            else:
                channels = self.current_file.header['signal_channels']['name']

            self.spike_sort_app.load_session(self.data_path, self.signal_data, self.get_channel_indices(channels))
            if (self.data_path / 'bad_channels.txt').exists():
                with open(self.data_path / 'bad_channels.txt') as f:
                    bad_channels = f.read().splitlines()
            else:
                bad_channels = None
            self.populate_table(channels, bad_channels=bad_channels)

    def populate_table(self, channels, bad_channels=None):
        self.table.setRowCount(len(channels))
        self.channels = set(channels)
        if bad_channels is None:
            self.bad_channels = set()  # Track bad channels
        else:
            self.bad_channels = set(bad_channels)
            self.spike_sort_app.continuous_viewer.all_channels = self.get_channel_indices(list(self.channels - self.bad_channels))
        # for i, channel in enumerate(channels):
        #     item = QTableWidgetItem(channel)
        #     self.table.setItem(i, 0, item)

        for i, channel in enumerate(channels):
            item = QTableWidgetItem(channel)
            self.table.setItem(i, 0, item)

            check_box = QCheckBox()
            check_box.setChecked(channel in self.bad_channels)
            check_box.clicked.connect(self.bad_channel_handler(channel))
            self.table.setCellWidget(i, 1, check_box)

    def select_channel(self):
        selected_items = self.table.selectedItems()
        if not selected_items or self.current_file is None:
            return

        selected_channel = selected_items[0].text()
        selected_channel_index = self.get_channel_indices([selected_channel])
        self.spike_sort_app.load_channel(selected_channel, selected_channel_index)

    def bad_channel_handler(self, channel):
        def handler(state):
            if state:
                self.bad_channels.add(channel)
            else:
                self.bad_channels.discard(channel)
            with open(self.data_path / 'bad_channels.txt', 'w') as f:
                f.write('\n'.join(self.bad_channels))
            self.spike_sort_app.continuous_viewer.all_channels = self.get_channel_indices(list(self.channels - self.bad_channels))            
            self.spike_sort_app.continuous_viewer.update_plot()
        return handler

    def save_data(self):
        print("Save data")
        # if self.spike_sort_app:
        #     np.save('clusters.npy', self.spike_sort_app.clusters)
            # You can also add code to save other data as needed

    def show_version(self):
        # Display version information
        print("Version 1.0")

    def show_documentation(self):
        # Open a documentation file or webpage
        print("Documentation is available at: http://example.com")

if __name__ == '__main__':
    import sys
    loglevel = 'DEBUG' if  '-v' in sys.argv else 'WARN'
    app = QApplication([])
    window = MainWindow(logger_kwargs={'loggerName': 'SpikeSorter', 'fileName': 'spikesort.log', 'printLevel': loglevel})
    window.show()
    app.exec_()