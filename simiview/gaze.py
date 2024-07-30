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
        self.inferred = {}

    def differentiate(self, diff_method="euclidean", diff_dimensions=None, filter_method=None):
        time = np.diff(data.time)
        if diff_method=="euclidean":
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

    def identify_velocity_events(self, velocity_query, velocity_params={}):
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
            trace.time = trace.time - event['timestamp']
            #todo: filter the inferred fields
            result.append(data)
        return result

