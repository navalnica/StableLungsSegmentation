import numpy as np

import utils
from data.datasets import BaseDataset
from .base_dl import BaseDataLoader


class DataLoaderNoAugmentations(BaseDataLoader):
    """
    Data loader that yields images from Dataset as is.
    Use this loader if augmentations are performed in Dataset class itself
    to avoid performing augmentations twice.
    """

    def __init__(self, dataset: BaseDataset, batch_size: int, to_shuffle: bool):
        self._dataset = dataset
        self._batch_size = batch_size
        self._to_shuffle = to_shuffle

    def __str__(self):
        return (f'{utils.get_class_name(self)}('
                f'len: {len(self)}; '
                f'n_images: {self.n_images}; '
                f'batch_size: {self.batch_size}; '
                f'to_shuffle: {self._to_shuffle})'
                )

    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return len(self._dataset)

    def get_generator(self):
        orig_images_cnt = len(self)
        indices = np.arange(orig_images_cnt)

        if self._to_shuffle:
            np.random.shuffle(indices)

        batch_indices = [indices[a: a + self._batch_size]
                         for a in range(0, orig_images_cnt, self._batch_size)]

        for cur_indices in batch_indices:

            scans_batch, masks_batch, descriptions_batch = [], [], []
            for ix in cur_indices:
                sample = self._dataset[ix]
                scans_batch.append(sample['scan'])
                masks_batch.append(sample['mask'])
                descriptions_batch.append(sample['description'])

            yield scans_batch, masks_batch, descriptions_batch
