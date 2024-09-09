from itertools import product

import numpy as np
import xarray as xr

from simianpy.misc import binary_digitize

def parse_query(x, query=None):
    if query==None:
        query={}
    min_ = query.get('min', -np.inf)
    max_ = query.get('max', np.inf)
    return (min_ < x) & (x < max_)

class GazeData:
    def __init__(self, time: np.ndarray, position: np.ndarray, dimensions: list[str]):
        self.data = xr.DataArray(
            position, 
            dims=("time", "dimension"), 
            coords=dict(time=time, dimension=dimensions)
        )
        self.blink_mask = np.ones(self.data.time.size, dtype=bool)
        self.inferred = {}

    def mask_blinks(self, threshold=30, pad=None):
        """Mask blinks in gaze data

        More specifically, this mask will hide all data points that are "off the screen" as defined by the threshold.

        Parameters
        ----------
        threshold : int, optional
            The absolute value threshold for gaze events off the screen, by default 30
        pad : int, optional
            The number of samples around "blink" events to mask, by default None
        """
        blink_start, blink_end = binary_digitize((np.abs(self.data) >= threshold).any('dimension'))
        blink_start = np.clip(blink_start-pad, 0, self.blink_mask.size, out=blink_start)
        blink_end = np.clip(blink_end+pad, 0, self.blink_mask.size, out=blink_end)
        idx = np.concatenate([np.arange(start, end) for start, end in zip(blink_start, blink_end)])
        self.blink_mask[idx] = False

    def differentiate(self, 
            diff_method="radial", 
            diff_dimensions: str | list[str] | None=None, 
            filter_method: callable=None
        ):
        """Differentiate gaze data

        Parameters
        ----------
        diff_method : str, optional
            radial distance is the only method supported
            by default "radial"
        diff_dimensions : str | list[str] | None, optional
            Select a single or a subset of dimensions
            If None, selects all dimensions, by default None
        filter_method : callable, optional
            If provided, applies this function to the data, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        time = np.diff(data.time)
        if diff_method=="radial":
            if diff_dimensions is None:
                data = self.data
            else:
                data = self.data.sel(dimension=diff_dimensions)
            difference = data.diff("time")
            if filter_method is not None:
                difference = filter_method(difference)
            difference = np.hypot(*difference.transpose(("dimension", "time")))
        else:
            raise ValueError
        diff = difference / time
        return diff

    def identify_velocity_events(self, velocity_params={}):
        velocity = self.differentiate(**velocity_params)
        velocity = np.abs(velocity)
        mask = parse_query(velocity, velocity_params)
        onsets, offsets = binary_digitize(mask)
        return {'onset': onsets, 'offset': offsets, 'velocity': velocity}

    def get_saccades(self, velocity_query, duration_query=None, peak_velocity_query=None, velocity_params={}):
        data = self.identify_velocity_events(velocity_query, velocity_params)
        dimensions = self.data.dimensions

        records = []
        for idx in range(data['onsets'].size):
            record = {}
            for field in ['onset', 'offset']:
                record[f'{field}.time'] = self.data.time[data[field][idx]]
            record['duration'] = duration = record['offset.time'] - record['onset.time']
            tslice = slice(record['onset.time'], record['offset.time'])
            record['peak_velocity'] = peak_velocity = data['velocity'].sel(time=tslice).max()
            if not parse_query(duration, duration_query): continue
            if not parse_query(peak_velocity, peak_velocity_query): continue

            for field, dim in product(['onset', 'offset'], dimensions):
                record[f'{field}.{dim}'] = self.data.isel(time=data[field][idx]).sel(dimension=dim)
            record['amplitude'] = np.hypot(*[record['offset.{dim}']-record['onset.{dim}'] for dim in dimensions])
            records.append(record)

        self.inferred['saccades'] = records

        return records

    def get_fixations(self, velocity_query, duration_query=None, velocity_params={}):
        data = self.identify_velocity_events(velocity_query, velocity_params)
        dimensions = self.data.dimensions

        records = []
        for idx in range(data['onsets'].size):
            record = {}
            for field in ['onset', 'offset']:
                record[f'{field}.time'] = self.data.time[data[field][idx]]
            record['duration'] = duration = record['offset.time'] - record['onset.time']
            if not parse_query(duration, duration_query): continue

            tslice = slice(record['onset.time'], record['offset.time'])
            position = self.data.sel(time=tslice).mean('time')
            for dim in dimensions:
                record[dim] = position.sel(dimension=dim)
            records.append(record)

        self.inferred['fixations'] = records

        return records

    def get_by_events(self, events, bounds):
        left, right = bounds
        result = []
        for event in events:
            data = {}
            l, r = event['timestamp']+left, event['timestamp']+right
            data['trace'] = trace = self.data.sel(time=slice(l, r)).copy()
            data['mask'] = mask = self.blink_mask[l:r]
            trace.time = trace.time - event['timestamp']

            for key, value in self.inferred.items():
                data[key] = []
                for record in value:
                    if r <= record['onset.time'] or record['offset.time'] <= l:
                        continue
                    relative_record = record.copy()
                    relative_record['latency'] = record['onset.time'] - event['timestamp']
                    data[key].append(record)
            result.append(data)
        return result

class GazeDataSet:
    def __init__(self, data: list[GazeData], attrs=None):
        self.data = data
    @classmethod
    def from_npy(cls, path):
        data = np.load(path)
        # data should be an ndarray of shape (n_trials, n_timepoints, n_dimensions)
        time = np.arange(data.shape[1])
        dimensions = [f'dim_{i}' for i in range(data.shape[2])]
        return cls([GazeData(time, trial, dimensions) for trial in data])