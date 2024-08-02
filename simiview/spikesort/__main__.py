from vispy import app
from simiview.spikesort.app import SpikeSortApp

if __name__ == '__main__':
    app.use_app('pyqt5')
    spikesort_app = SpikeSortApp.from_directory('simiview/data')
    app.run()
