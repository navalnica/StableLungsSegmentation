from typing import List

from torch.utils.data import Dataset

import const
import utils


class BaseDataset(Dataset):
    """
    Base class for dataset abstractions that help to read images in different formats
    and yield results to Data Loaders in a unified way.

    You can specify different number of augmentations for two subsets of images with the help
    of `set_different_aug_cnt_for_two_subsets` method and use of `DataLoaderNoAugmentations`.
    """

    _slice_info = None

    def set_different_aug_cnt_for_two_subsets(
            self, augs_cnt: int, ids_heavy_augs: List[str], augs_cnt_heavy: int
    ):
        """
        Initialize `slice_info` dict with number of augmentations per images.
        Use this method to set different number of augmentations for two subsets of images.
        When images are retrieved with `__getitem__` method and `slice_info` dict has
        'augment' == True augmentations are applied before yielding scan and mask.

        :param augs_cnt: number of augmentations for all slices
        :param ids_heavy_augs: list with ids of images that need to be heavily augmented.
        pass None to use `augs_cnt` for slices of all the images.
        :param augs_cnt_heavy: number of augmentations for each slice
        of image with id in `ids_ids_heavy_augs`. overwrites `augs_cnt`.
        """

        print(const.SEPARATOR)
        print('BaseDataset.set_different_aug_cnt_for_two_subsets()')
        print('setting different number of augmentations for train subsets:')
        print(f'augs_cnt: {augs_cnt}')
        print(f'len(ids_heavy_augs): {len(ids_heavy_augs)}')
        print(f'augs_cnt_heavy: {augs_cnt_heavy}')

        utils.check_var_to_be_iterable_collection(ids_heavy_augs)

        new_slice_info = []
        for si in self._slice_info:
            new_slice_info.append({**si, 'augment': False})
            cur_augs = augs_cnt_heavy \
                if ids_heavy_augs is not None and si['id'] in ids_heavy_augs \
                else augs_cnt
            new_slice_info.extend([{**si, 'augment': True} for _ in range(cur_augs)])
        self._slice_info = new_slice_info

    def __len__(self):
        return len(self._slice_info)

    def __getitem__(self, item):
        raise NotImplementedError

    def _init_slice_info(self):
        raise NotImplementedError

    @property
    def n_images(self):
        raise NotImplementedError
