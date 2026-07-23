class DataRandomAccess:
    def __init__(self, dataset):
        self._dataset = dataset
        self._schema = dataset._schema

    def __getitem__(self, key):
        return self._schema.ra(key)
