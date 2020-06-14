import math

from data.datasets import BaseDataset


class BaseDataLoader:
    _dataset: BaseDataset

    @property
    def batch_size(self):
        return NotImplementedError

    @property
    def n_images(self):
        return self._dataset.n_images

    @property
    def n_batches(self):
        return math.ceil(len(self) / self.batch_size)

    def __len__(self):
        return NotImplementedError

    def get_generator(self):
        return NotImplementedError
