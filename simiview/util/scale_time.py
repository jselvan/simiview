scale = {
    'hr': 60**2,
    'min': 60,
    's': 1,
    'ms': 1e-3,
    'us': 1e-6
}
def scale_time(time, input_units, output_units, sampling_rate=None):
    input_scale = sampling_rate if input_units=='sampling_rate' else scale[input_units]
    output_scale = sampling_rate if output_units=='sampling_rate' else scale[output_units]
    return time * input_scale / output_scale