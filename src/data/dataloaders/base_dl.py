from data.datasets import BaseDataset


class BaseDataLoader:
    _dataset: BaseDataset

    @property
    def batch_size(self):
        return NotImplementedError

    @property
    def n_images(self):
        return self._dataset.n_images

    def __len__(self):
        return NotImplementedError

    def get_generator(self):
        return NotImplementedError
