from itertools import combinations

import numpy as np

from simiview.util import scale_time

def ccg_matrix(spike_times, unit_ids, bin_size=0.1, max_lag=20, input_units='ms', sampling_rate=None, unitids=None):
    if input_units != 'ms':
        spike_times = scale_time(spike_times, input_units, 'ms', sampling_rate=sampling_rate)
        # spike_times = spike_times.astype(np.int32)

    if unitids is not None:
        unique_neurons = unitids
    else:
        unique_neurons = np.unique(unit_ids).tolist()
    # Bins for the histogram
    bins = np.arange(-max_lag, max_lag + bin_size*2, bin_size) - bin_size / 2
    lags = np.arange(-max_lag, max_lag + bin_size, bin_size)
    n_lags = int((bins.size - 1) / 2)

    corrs = {}
    # Compute crosscorrelograms
    for neuron in unique_neurons:
        neuron_spike_times = spike_times[unit_ids == neuron]
        diffs = np.subtract.outer(neuron_spike_times, neuron_spike_times)
        diffs = diffs[np.triu_indices_from(diffs, k=1)]  # Remove zero-lag and duplicate pairs
        acg = np.histogram(diffs, bins=bins)[0]
        acg[-n_lags:] = acg[:n_lags][::-1]
        corrs[neuron, neuron] = acg

    for neuron_i, neuron_j in combinations(unique_neurons, 2):
        neuron_i_spike_times = spike_times[unit_ids == neuron_i]
        neuron_j_spike_times = spike_times[unit_ids == neuron_j]
        diffs = np.subtract.outer(neuron_i_spike_times, neuron_j_spike_times)
        corrs[(neuron_i, neuron_j)] = np.histogram(diffs, bins=bins)[0]

    return lags, corrs