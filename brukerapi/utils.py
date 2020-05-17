import numpy as np

def index_to_slice(index, data_shape, dim_index):
    out = []

    for i in range(len(data_shape)):
        if i != dim_index:
            out.append(slice(0, data_shape[i]))
        else:
            out.append(index)

    return tuple(out)

def simple_measurement(self):
        if self._data is None:
            raise AttributeError('Data is not loaded')

        if self._fg_scheme.core_dim == 1:
            axes = (0,)
        elif self._fg_scheme.core_dim == 2:
            axes = (0,1)
        elif self._fg_scheme.core_dim == 3:
            axes = (0,1,2)

        return np.fft.fftshift(np.fft.fft2(self.data,axes=axes),axes=axes)

def simple_reconstruction(self, **kwargs):
        """
        Simple Fourier reconstruction
        :return: image
        """
        try:
            data = self.get_data()
        except AttributeError:
            print('Bruker object has no data, reconstruction is not possible')
            return


        if self.get_value('ACQ_dim') == 1:
            axes = (0,)
        elif self.get_value('ACQ_dim') == 2:
            axes = (0, 1)
        elif self.get_value('ACQ_dim') == 3:
            axes = (0, 1, 2)
        data = np.fft.fftshift(np.fft.ifft2(data, axes=axes),
                               axes=axes)

        if kwargs.get("COMBINE_CHANNELS") is True:
            return self.combine_channels(data=data)
        else:
            return data

def combine_channels(self, data=None):

        if self.acq_scheme is not None:
            channel_dim = self.acq_scheme.dim_desc.index('channel')
        else:
            raise NotImplemented

        if data is None:
            data = self.get_data()

        data = data ** 2
        data = np.expand_dims(np.sum(data, channel_dim), channel_dim)

        return np.sqrt(data)
