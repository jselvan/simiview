import pandas as pd

from PyQt5.QtWidgets import QTableView, QMenu, QAction, QActionGroup, QAbstractScrollArea
from PyQt5.QtCore import Qt, QAbstractTableModel, QPoint

class ColumnMenu(QMenu):
    def __init__(self, parent, offset):
        super(ColumnMenu, self).__init__(parent)
        
        # add filter submenu
        filter_submenu = self.addMenu("Filter")
        filter_submenu.addAction("Filter by value")
        filter_submenu.addAction("Filter by range")

        # add sort submenu
        sort_submenu = self.addMenu("Sort")
        sort_asc = sort_submenu.addAction("Sort ascending")
        sort_desc = sort_submenu.addAction("Sort descending")

        sort_asc.triggered.connect(lambda: parent.model.sort_by_column(offset, ascending=True))
        sort_desc.triggered.connect(lambda: parent.model.sort_by_column(offset, ascending=False))
        # action_group = QActionGroup(self)
        # action1 = QAction("Sort ascending", self, checkable=True)
        # action1.triggered.connect(lambda: print(f'Sort column {offset} ascending'))
        # action1.triggered.connect(lambda: parent.model.sort_by_column(offset, ascending=True))
        # action2 = QAction("Sort descending", self, checkable=True)
        # action2.triggered.connect(lambda: print(f'Sort column {offset} descending'))

        # action_group.addAction(action1)
        # action_group.addAction(action2)
        # action_group.setExclusive(True) 
        # sort_submenu.addActions(action_group.actions())

        self.offset = offset

    def show(self):
        pos = self.parent()._column_menu_offset(self.offset)
        self.exec_(pos)


class pandasDataModel(QAbstractTableModel):
    def __init__(self, data):
        super(pandasDataModel, self).__init__()
        self._data = data
    @classmethod
    def empty(cls, columns):
        return cls(pd.DataFrame(columns=columns))
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
    def columns(self):
        return self._data.columns
    def sort_by_column(self, column, ascending=True):
        self._data.sort_values(by=self.columns()[column], ascending=ascending, inplace=True)
        self.layoutChanged.emit()


class SmartTable(QTableView):
    def __init__(self, parent=None, model=None):
        super(SmartTable, self).__init__(parent)
        if model is not None:
            self.setModel(model)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.horizontalHeader().sectionClicked.connect(self.header_clicked)
        self.verticalHeader().sectionClicked.connect(self.index_clicked)
    
    def setModel(self, model):
        super(SmartTable, self).setModel(model)
        self.model = model
        self.model.layoutChanged.connect(self.update_column_menus)

    def update_column_menus(self):
        self.column_menus = []
        for i, col in enumerate(self.model.columns()):
            menu = ColumnMenu(self, offset=i)
            self.column_menus.append(menu)

    def _column_menu_offset(self, index):
        root = self.horizontalHeader().pos()
        x_offset = self.horizontalHeader().sectionPosition(index)
        y_offset = self.horizontalHeader().height()
        return self.mapToGlobal(root + QPoint(x_offset, y_offset))

    def setData(self, data):
        if self.model is not None:
            self.model.setData(data)
        else:
            raise ValueError('Model is not set')
    
    def header_clicked(self, index):
        self.column_menus[index].show()
    
    def index_clicked(self, index):
        print(f'Index {index} clicked')

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    data = pd.read_csv('eyedata.csv', index_col=0)
    model = pandasDataModel.empty(['date', 'trialid', 'outcome'])

    app = QApplication([])
    table = SmartTable(model=model)
    table.setData(data)
    window = QMainWindow()
    window.setCentralWidget(table)
    window.show()
    app.exec_()