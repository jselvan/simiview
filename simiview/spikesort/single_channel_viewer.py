import numpy as np
from vispy import scene

from simianpy.signal import sosFilter
import simianpy as simi

filt = (
    sosFilter('bandstop', 6, [49.9, 50.1], 30000) 
    + sosFilter('bandstop', 6, [99.9, 100.1], 30000)
    + sosFilter('bandpass', 6, [300, 3000], 30000)
)

class SingleChannelViewer:
    @simi.misc.add_logging
    def __init__(self, view, update_spikes_callback=None, logger=None):
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
        self.logger = logger

        self._median_trace = None
        self._init_vispy()

    def _get_data_chunk(self):
        if self.channel_idx is None:
            return
        start = self.current_position / self.sig.sampling_rate
        end = start + self.chunk_size / self.sig.sampling_rate
        data_chunk = self.sig.load(time_slice=(start, end), channel_indexes=self.channel_idx)
        data_chunk = data_chunk.magnitude.squeeze()

        if self.is_cmr_enabled:
            # all_channels = self.sig.load(time_slice=(start, end), channel_indexes=self.all_channels).magnitude
            # data_chunk = data_chunk - np.median(all_channels, axis=1)
            median = self.get_median_trace(time_slice=(start, end))
            data_chunk = data_chunk - median
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
    
    @property
    def sig(self):
        return self._sig
    @sig.setter
    def sig(self, value):
        self._sig = value
        self._median_trace = None
    @property
    def all_channels(self):
        return self._all_channels
    @all_channels.setter
    def all_channels(self, value):
        self._all_channels = value
        self._median_trace = None
    def get_median_trace(self, time_slice=None):
        """Get the median trace of all channels

        Parameters
        ----------
        time_slice : tuple, optional
            Tuple with the start and stop of the time slice in seconds, by default None

        Returns
        -------
        np.ndarray
            The median trace of all channels in the time slice 
            cast to np.float16
        """
        if time_slice is not None:
            self.logger.info(f"Getting median trace for time slice {time_slice}")
            start, stop = time_slice
            start_idx = int(round(start * self.sig.sampling_rate))
            stop_idx = int(round(stop * self.sig.sampling_rate))
            time_slice_samples = slice(start_idx, stop_idx)
        else:
            self.logger.info(f"Getting median trace for all time")
            time_slice_samples = slice(None)


        if self._median_trace is None:
            self.logger.info("No trace, allocating median trace empty array")
            self._median_trace = np.full(self.n_samples, np.nan, dtype=np.float16)
        if np.isnan(self._median_trace[time_slice_samples]).any():
            self.logger.info("Missing data, calculating median trace")
            self._median_trace[time_slice_samples] = np.median(
                self.sig.load(channel_indexes=self.all_channels, time_slice=time_slice).magnitude,
                axis=1
            )
        else:
            self.logger.info("Returning cached median trace")

        return self._median_trace[time_slice_samples]

    def detect_waveforms(self):
        if self.threshold is None:
            return
        self.detect_chunk_size = int(1e7)
        self.waveforms = []
        self.timestamps = []
        # iterate through the whole file in chunks
        self.logger.info(f"Detecting waveforms with threshold {self.threshold}")
        for i in range(0, self.n_samples, self.detect_chunk_size):
            start = i / self.sig.sampling_rate
            stop = np.clip(i+self.detect_chunk_size, 0, self.n_samples) / self.sig.sampling_rate
            self.logger.info(f"Processing chunk {i}: {start} - {stop}")
            chunk = self.sig.load(time_slice=(start, stop), channel_indexes=self.channel_idx)
            chunk = chunk.magnitude.squeeze()
            self.logger.info(f"Loaded chunk of size {chunk.size}")
            if self.is_cmr_enabled:
                median = self.get_median_trace(time_slice=(start, stop))
                chunk = chunk - median
                # all_channels = self.sig.load(time_slice=(start, stop), channel_indexes=self.all_channels).magnitude
                # chunk = chunk - np.median(all_channels, axis=1)
                self.logger.info(f"CMR applied")
            chunk = chunk * self.scale_factor
            if self.is_filter_enabled:
                chunk = filt(chunk)
                self.logger.info(f"Filter applied")
            # find the indices where the threshold is crossed
            crossings = np.where(chunk < self.threshold)[0]
            self.logger.info(f"Found {crossings.size} crossings")
            #remove crossings that are too close to each other
            crossings = crossings[np.diff(crossings, prepend=-40) > 2]
            self.logger.info(f"Removed crossings too close to each other. {crossings.size} crossings left")
            # get indexes of the waveforms
            # waveforms = np.array([chunk[crossing-8:crossing+32] for crossing in crossings])
            # if this is slow we can maybe do
            waveforms = chunk[np.repeat(crossings, 40) + np.tile(np.arange(-8, 32), crossings.size)].reshape(-1, 40)
            self.logger.info(f"Extracted {waveforms.shape[0]} waveforms")
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
        self.logger.info("Finished detecting waveforms, concatenating")
        self.waveforms = np.concatenate(self.waveforms, axis=0)
        self.timestamps = np.concatenate(self.timestamps)
        self.logger.info("Finished concatenating")
        # self.waveforms = np.array(self.waveforms)
        # self.timestamps = np.array(self.timestamps)
        if self.update_spikes_callback is not None:
            self.logger.info("Calling update_spikes_callback")
            self.update_spikes_callback(self.waveforms, self.timestamps)        

    def on_mouse_press(self, event):
        """ Handle mouse presses """
        if event.button == 1:
            pos = self.threshold_line.get_transform('canvas', 'visual').map(event.mouse_event.pos)
            self.threshold = pos[1]
            self.update_plot()