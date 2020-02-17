class Metric():

    def __init__(self):
        self.reset()

    def reset(self):
        self._data_count = 0
        self._batch_count = 0
        self._aggregated_value = 0

    @property
    def data_count(self):
        return self._data_count

    @property
    def batch_count(self):
        return self._batch_count

    def get_value(self):
        return self._aggregated_value / self._data_count

    def add_batch(self, y, preds):
        raise NotImplementedError
