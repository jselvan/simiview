import numpy as np
from vispy import scene

from simianpy.signal import sosFilter

filt = (
    sosFilter('bandstop', 6, [49.9, 50.1], 30000) 
    + sosFilter('bandstop', 6, [99.9, 100.1], 30000)
    + sosFilter('bandpass', 6, [300, 3000], 30000)
)

class SingleChannelViewer:
    def __init__(self, view, update_spikes_callback=None):
        self.view = view
        self.sig = None
        self.channel_idx = None
        self.all_channels = None
        self.is_filter_enabled = False
        self.is_cmr_enabled = False
        self.chunk_size=30000
        self.current_position = 0
        self.scroll_speed=3000
        self.scale_factor=1
        self.threshold = None
        self.update_spikes_callback = update_spikes_callback
    
        self._init_vispy()

    def _get_data_chunk(self):
        if self.channel_idx is None:
            return
        start = self.current_position / self.sig.sampling_rate
        end = start + self.chunk_size / self.sig.sampling_rate
        data_chunk = self.sig.load(time_slice=(start, end), channel_indexes=self.channel_idx)
        data_chunk = data_chunk.magnitude.squeeze()

        if self.is_cmr_enabled:
            all_channels = self.sig.load(time_slice=(start, end), channel_indexes=self.all_channels).magnitude
            data_chunk = data_chunk - np.median(all_channels, axis=1)
        data_chunk = data_chunk * self.scale_factor
        if self.is_filter_enabled:
            data_chunk = filt(data_chunk)
        time = np.arange(data_chunk.size)
        data_chunk = np.column_stack((time, data_chunk))
        return data_chunk

    def update_plot(self):
        """ Update the plot with the current position and channels """
        data_chunk = self._get_data_chunk()
        #TODO: implement colouring of detected waveforms
        self.line.set_data(pos=data_chunk)
        if self.threshold is not None:
            self.threshold_line.set_data(pos = self.threshold)

    def _init_vispy(self):
        # Set up the viewbox and line plots for the selected channels
        self.view.camera = 'panzoom'
        self.view.camera.interactive = False
        self.view.camera.rect = self.get_camera_rect()
        # Create LinePlot visuals for each channel
        self.line = scene.visuals.Line()
        self.view.add(self.line)
        self.threshold_line = scene.visuals.InfiniteLine(pos=0, color=(0,1,0,1), vertical=False, parent=self.view.scene)
        self.zero_line = scene.visuals.InfiniteLine(pos=0, color=(1,0,0,1), vertical=False, parent=self.view.scene)
        self.view.events.mouse_wheel.connect(self.on_scroll)
        self.view.events.mouse_press.connect(self.on_mouse_press)

    @property
    def n_samples(self):
        if self.sig is None:
            return 0
        else:
            return self.sig.shape[0]
    @property
    def current_position(self):
        return self._current_position
    @current_position.setter
    def current_position(self, value):
        max_pos = max(self.n_samples - self.chunk_size, 1)
        self._current_position = int(np.clip(value, 0, max_pos))
    @property
    def scale_factor(self):
        return self._scale_factor
    @scale_factor.setter
    def scale_factor(self, value):
        self._scale_factor = np.clip(value, 0.01, 10)
    @property
    def chunk_size(self):
        return self._chunk_size
    def get_camera_rect(self):
        bottom = -10
        height = 20
        return (0, bottom), (self.chunk_size, height)
    @chunk_size.setter
    def chunk_size(self, value):
        self._chunk_size = int(np.clip(value, 3_000, 300_000)) # 100 ms to 10 s
        if hasattr(self, 'view'):
            self.view.camera.rect = self.get_camera_rect()

    def on_scroll(self, event):
        """ Handle mouse wheel changes """
        if self.sig is None or self.channel_idx is None:
            return
        delta = int(event.delta[1])
        if "Shift" in event.mouse_event.modifiers:
            self.chunk_size -= delta * 1000
        elif "Control" in event.mouse_event.modifiers:
            self.scale_factor += delta * 0.1
        else:
            self.current_position += delta * self.scroll_speed
        self.update_plot()
    def register_events(self, parent):
        # parent.events.mouse_wheel.connect(self.on_scroll)
        parent.events.key_press.connect(self.on_key_press)
    def on_key_press(self, event):
        """ Handle key presses """
        if event.key == "f":
            self.is_filter_enabled = not self.is_filter_enabled
        elif event.key == "c":
            self.is_cmr_enabled = not self.is_cmr_enabled
        elif event.key == "t":
            self.detect_waveforms()
        self.update_plot()
    
    def detect_waveforms(self):
        if self.threshold is None:
            return
        self.detect_chunk_size = int(3e6)
        self.waveforms = []
        self.timestamps = []
        # iterate through the whole file in chunks
        for i in range(0, self.n_samples, self.detect_chunk_size):
            start = i / self.sig.sampling_rate
            stop = np.clip(i+self.detect_chunk_size, 0, self.n_samples) / self.sig.sampling_rate
            chunk = self.sig.load(time_slice=(start, stop), channel_indexes=self.channel_idx)
            chunk = chunk.magnitude.squeeze()
            if self.is_cmr_enabled:
                all_channels = self.sig.load(time_slice=(start, stop), channel_indexes=self.all_channels).magnitude
                chunk = chunk - np.median(all_channels, axis=1)
            chunk = chunk * self.scale_factor
            if self.is_filter_enabled:
                chunk = filt(chunk)
            # find the indices where the threshold is crossed
            crossings = np.where(chunk < self.threshold)[0]
            #remove crossings that are too close to each other
            crossings = crossings[np.diff(crossings, prepend=-40) > 32]
            # get indexes of the waveforms
            waveforms = np.array([chunk[crossing-8:crossing+32] for crossing in crossings])
            # if this is slow we can maybe do
            # waveforms = chunk[np.repeat(crossings, 40) + np.tile(np.arange(-8, 32), crossings.size)].reshape(-1, 40)
            self.waveforms.append(waveforms)
            timestamps = crossings + i
            self.timestamps.append(timestamps)

            # # iterate through the crossings and find the waveforms
            # for crossing in crossings:
            #     start = crossing - 8
            #     end = crossing + 32
            #     if start < 0 or end > chunk.size:
            #         continue
            #     waveform = chunk[start:end]
            #     # store the waveform
            #     self.waveforms.append(waveform)
            #     # store the timestamp
            #     self.timestamps.append(i + crossing)
        self.waveforms = np.concatenate(self.waveforms, axis=0)
        self.timestamps = np.concatenate(self.timestamps)
        # self.waveforms = np.array(self.waveforms)
        # self.timestamps = np.array(self.timestamps)
        if self.update_spikes_callback is not None:
            self.update_spikes_callback(self.waveforms, self.timestamps)        

    def on_mouse_press(self, event):
        """ Handle mouse presses """
        if event.button == 1:
            pos = self.threshold_line.get_transform('canvas', 'visual').map(event.mouse_event.pos)
            self.threshold = pos[1]
            self.update_plot()