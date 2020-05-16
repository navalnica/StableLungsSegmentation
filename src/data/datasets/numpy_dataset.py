import os
import pickle
from typing import List

import tqdm

import const
import utils
from data import augmentations
from data.datasets import BaseDataset


class NumpyDataset(BaseDataset):
    def __init__(self, scans_dp: str, masks_dp: str, images_shapes_fp: str, img_ids: List[str] = None):
        utils.check_var_to_be_iterable_collection(img_ids)

        self._scans_dp = scans_dp
        self._masks_dp = masks_dp
        self._images_shapes_fp = images_shapes_fp
        self._img_ids = img_ids

        self._load_images_shapes()
        self._init_slice_info()

    def _load_images_shapes(self):
        if not os.path.isfile(self._images_shapes_fp):
            raise FileNotFoundError(f'{self._images_shapes_fp}')

        print(f'loading shapes dict from "{self._images_shapes_fp}"')
        with open(self._images_shapes_fp, 'rb') as fin:
            self._shapes = pickle.load(fin)

        # filter images
        if self._img_ids is not None:
            self._shapes = {k: v for (k, v) in self._shapes.items() if k in self._img_ids}

    def _init_slice_info(self):
        self._slice_info = [
            {
                'id': cur_id,
                'scan_fp': os.path.join(self._scans_dp, f'{cur_id}.npy'),
                'mask_fp': os.path.join(self._masks_dp, f'{cur_id}.npy'),
                'z_ix': z
            } for (cur_id, cur_shape) in self._shapes.items() for z in range(cur_shape[2])
        ]

    @property
    def n_images(self):
        return len(self._shapes)

    def __getitem__(self, ix):
        """
        Yield (scan, mask, description) tuple for single slice.

        If `slice_info` dict has 'augment' == True for specified slice
        than augmentations are applied before yielding results.
        """
        cur_info = self._slice_info[ix]
        cur_id = cur_info['id']
        z_ix = cur_info['z_ix']

        scan = utils.load_npy(cur_info['scan_fp'])[:, :, z_ix]
        mask = utils.load_npy(cur_info['mask_fp'])[:, :, z_ix]

        if 'augment' in cur_info and cur_info['augment'] is True:
            scan, mask = augmentations.get_single_augmentation(scan, mask)

        sample = {
            'scan': scan,
            'mask': mask,
            'description': f'{cur_id}_{z_ix}'
        }

        return sample

    @staticmethod
    def store_images_shapes(numpy_data_root_dp, out_fp: str = None):
        """
        Create .pickle file containing dictionary with .npy image shapes
        """
        numpy_data_paths = const.NumpyDataPaths(numpy_data_root_dp)
        scans_dp = numpy_data_paths.scans_dp
        masks_dp = numpy_data_paths.masks_dp
        print(f'will build shapes dict for .npy images under "{scans_dp}"')

        scans_fps = utils.get_npy_filepaths(scans_dp)
        masks_fps = utils.get_npy_filepaths(masks_dp)

        shapes = {}
        for cur_scan_fp, cur_mask_fp in tqdm.tqdm(zip(scans_fps, masks_fps)):
            cur_id = utils.parse_image_id_from_filepath(cur_scan_fp)
            scan = utils.load_npy(cur_scan_fp)
            mask = utils.load_npy(cur_mask_fp)

            assert scan.shape == mask.shape, (f'id: {cur_id}. '
                                              f'scan shape: {scan.shape}, '
                                              f'mask shape: {mask.shape}')
            shapes[cur_id] = scan.shape

        if out_fp is None:
            out_fp = numpy_data_paths.shapes_fp

        print(f'storing shapes dict to "{out_fp}"')
        with open(out_fp, 'wb') as fout:
            pickle.dump(shapes, fout)
