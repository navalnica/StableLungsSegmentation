import numpy as np

import utils
from data import augmentations
from data.datasets import BaseDataset
from .base_dl import BaseDataLoader


class DataLoaderWithAugmentations(BaseDataLoader):
    """
    Data loader that applies augmentations for scans and adds them to the same batch
    so less reads from files are performed. Do not know what speedup it introduces.
    Use this loader if no augmentations are performed in Dataset class while retrieving images.
    """

    def __init__(self, dataset: BaseDataset,
                 orig_img_per_batch,
                 aug_cnt,
                 to_shuffle):
        self._dataset = dataset
        self._orig_img_per_batch = orig_img_per_batch
        self._aug_cnt = aug_cnt
        self._to_shuffle = to_shuffle

    def __str__(self):
        return (f'{utils.get_class_name(self)}('
                f'len: {len(self)}; '
                f'n_images: {self.n_images}; '
                f'orig_img_per_batch: {self._orig_img_per_batch}; '
                f'aug_cnt: {self._aug_cnt}; '
                f'batch_size: {self.batch_size}; '
                f'to_shuffle: {self._to_shuffle})'
                )

    @property
    def batch_size(self):
        return self._orig_img_per_batch * (1 + self._aug_cnt)

    def __len__(self):
        return len(self._dataset) * (1 + self._aug_cnt)

    def get_generator(self):
        # TODO: consider replacing with __iter__ method

        orig_images_cnt = len(self._dataset)
        indices = np.arange(orig_images_cnt)

        if self._to_shuffle:
            np.random.shuffle(indices)

        batch_indices = [indices[a: a + self._orig_img_per_batch]
                         for a in range(0, orig_images_cnt, self._orig_img_per_batch)]

        for cur_indices in batch_indices:

            scans_batch, masks_batch, descriptions_batch = [], [], []
            for ix in cur_indices:
                sample = self._dataset[ix]
                scan_augs, mask_augs = augmentations.get_multiple_augmentations(
                    sample['scan'], sample['mask'], self._aug_cnt)
                scans_batch.extend(scan_augs)
                masks_batch.extend(mask_augs)
                descriptions_batch.extend([sample['description']] * (1 + self._aug_cnt))

            yield scans_batch, masks_batch, descriptions_batch
