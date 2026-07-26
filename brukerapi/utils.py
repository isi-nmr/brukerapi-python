import numpy as np

# Spec 8: DATTYPE is the legacy d3proc word type, written as the symbol rather
# than the ordinal. `#define ip_float ip_int` means it cannot express float
# data, so RECO_wordtype/VisuCoreWordType are always preferable.
DATTYPE_WORD_TYPES = {
    "ip_byte": "int8",
    "1": "int8",
    "ip_u_byte": "uint8",
    "2": "uint8",
    "ip_short": "int16",
    "3": "int16",
    "ip_u_short": "uint16",
    "4": "uint16",
    "ip_int": "int32",
    "5": "int32",
    "ip_u_int": "uint32",
    "6": "uint32",
}


def transposed_size(size, transposition):
    """`size` with the exchange recorded by a transposition parameter applied.

    Spec 6.9: the values are 1-based direction numbers, so `v` exchanges axes
    `v-1` and `v`, and `v == len(size)` exchanges the first and the last. For
    two dimensions both `1` and `2` therefore mean the same exchange.
    """
    size = tuple(int(length) for length in np.atleast_1d(size))
    value = int(np.atleast_1d(transposition).reshape(-1)[0])
    if not value or len(size) < 2:
        return size

    first, second = (0, len(size) - 1) if value >= len(size) else (value - 1, value)
    swapped = list(size)
    swapped[first], swapped[second] = swapped[second], swapped[first]
    return tuple(swapped)


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
