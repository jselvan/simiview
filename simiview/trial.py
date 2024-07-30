from collections import defaultdict
from numbers import Number
from typing import Any
import numpy as np

class Trial:

    def __init__(self, trialid, condition, markers, timestamps, attrspec=None, config=None):
        self.trialid = trialid
        self.condition = condition
        self.marker_codes = markers
        self.timestamps = np.array(timestamps)

        if attrspec is None:
            attrspec = {}
        if config is None:
            config = {}
        self.attrspec = attrspec
        self.config = config

        self.attributes = {}
        self._update_attributes()

        self.markers = np.array([self.config.get(marker, marker) for marker in self.markers])
        self.marker_dict = defaultdict(list)
        for marker, timestamp in zip(self.markers, self.timestamps):
            self.marker_dict[marker].append(timestamp)


    def _update_attributes(self):
        for attribute, attr_markers in self.attrspec.items():
            for marker in self.markers:
                if marker in attr_markers:
                    self.attributes[attribute] = self.config.get(marker, marker)
                    break
    
    def relative_to(self, timestamp):
        return {
            'markers': self.markers,
            'timestamps': self.timestamps - timestamp
        }


def get_end_from_start(markers, startidx):
    endidx = np.concatenate([startidx[1:]-1, markers[-1]])
    return endidx

class Trials:
    def __init__(self, trials: list[Trial]):
        self.trials = trials
    @classmethod
    def from_arrays(cls, 
            markers: list[Any] | np.ndarray, 
            timestamps: list[Number] | np.ndarray, 
            **kwargs
        ):
        method = kwargs.pop('method')
        start, end = kwargs.pop('start'), kwargs.pop('end')
        if method=='classic':
            startidx, = np.where(np.isin(markers, start))
            if end is None:
                endidx = get_end_from_start(markers, startidx)
            else:
                endidx, = np.where(np.isin(markers, end))
            conditions = markers[startidx]
        elif method=='boundary':
            condition_offset = kwargs.pop('condition_offset', 1)
            startidx, = np.where(markers==start)
            if end is None:
                endidx = get_end_from_start(markers, startidx)
            else:
                endidx, = np.where(markers==endidx)
            conditions = markers[startidx+condition_offset]
        trials = []
        for trialid, (condition, start, end) in enumerate(zip(conditions, startidx, endidx)):
            trialslice = slice(start, end)
            trials.append(
                Trial(trialid, 
                      condition, 
                      markers[trialslice], 
                      timestamps[trialslice])
            )
        return cls(trials)