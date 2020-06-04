class DataRandomAccess():

    def __init__(self, dataset):
        self._dataset = dataset
        self._scheme = dataset._scheme

    def __getitem__(self, slice):
        return self._scheme.ra(slice)