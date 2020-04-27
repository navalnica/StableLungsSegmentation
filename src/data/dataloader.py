import numpy as np
from torch.utils.data import Dataset

from data import augmentations


class DataLoader:

    def __init__(self, dataset: Dataset,
                 orig_img_per_batch,
                 aug_cnt,
                 to_shuffle):
        self.dataset = dataset
        self.orig_img_per_batch = orig_img_per_batch
        self.aug_cnt = aug_cnt
        self.to_shuffle = to_shuffle

    def __str__(self):
        return (f'DataLoader('
                f'len: {len(self)}; '
                f'to_shuffle: {self.to_shuffle}; '
                f'batch_size: {self.batch_size}; '
                f'orig_img_per_batch: {self.orig_img_per_batch}; '
                f'aug_cnt: {self.aug_cnt})'
                )

    @property
    def batch_size(self):
        return self.orig_img_per_batch * (1 + self.aug_cnt)

    def __len__(self):
        return len(self.dataset) * (1 + self.aug_cnt)

    def get_generator(self):
        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)
        if self.to_shuffle:
            np.random.shuffle(indices)

        batch_indices = [indices[a: a + self.orig_img_per_batch]
                         for a in range(0, dataset_size, self.orig_img_per_batch)]

        for cur_indices in batch_indices:

            scans_batch, masks_batch, descriptions_batch = [], [], []
            for ix in cur_indices:
                sample = self.dataset[ix]
                scan_augs, mask_augs = augmentations.augment_slice(
                    sample['scan'], sample['mask'], self.aug_cnt)
                scans_batch.extend(scan_augs)
                masks_batch.extend(mask_augs)
                descriptions_batch.extend([sample['description']] * (1 + self.aug_cnt))

            yield scans_batch, masks_batch, descriptions_batch
