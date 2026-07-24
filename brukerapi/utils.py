import numpy as np


def index_to_slice(index, data_shape, dim_index):
    out = []

    if isinstance(index, range):
        index = slice(index.start, index.stop)

    for i in range(len(data_shape)):
        if i != dim_index:
            out.append(slice(0, data_shape[i]))
        else:
            out.append(index)

    return tuple(out)


def simple_measurement(dataset):
    if dataset.encoded_dim == 1:
        axes = (0,)
    elif dataset.encoded_dim == 2:
        axes = (0, 1)
    elif dataset.encoded_dim == 3:
        axes = (0, 1, 2)

    return np.fft.fftshift(np.fft.fft2(dataset.data, axes=axes), axes=axes)


def simple_reconstruction(dataset, **kwargs):
    """
    Simple Fourier reconstruction
    :return: image
    """
    if dataset.encoded_dim == 1:
        axes = (0,)
    elif dataset.encoded_dim == 2:
        axes = (0, 1)
    elif dataset.encoded_dim == 3:
        axes = (0, 1, 2)

    data = np.fft.fftshift(np.fft.ifft2(dataset.data, axes=axes), axes=axes)

    if kwargs.get("COMBINE_CHANNELS") is True:
        return combine_channels(dataset, data=data)
    return data


def combine_channels(dataset, data=None):
    if data is None:
        data = dataset.data

    if "channel" not in dataset.dim_type:
        return data

    channel_dim = dataset.dim_type.index("channel")
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=channel_dim, keepdims=True))
