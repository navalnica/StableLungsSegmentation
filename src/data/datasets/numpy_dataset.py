import os
import pickle
from typing import List

import tqdm

import const
import utils
from data.datasets import BaseDataset


class NumpyDataset(BaseDataset):
    def __init__(self, scans_dp: str, masks_dp: str, images_shapes_fp: str, img_ids: List[str] = None):
        self.scans_dp = scans_dp
        self.masks_dp = masks_dp
        self.images_shapes_fp = images_shapes_fp
        self.img_ids = img_ids

        self._load_images_shapes()
        self._init_slice_info()

    def _load_images_shapes(self):
        if not os.path.isfile(self.images_shapes_fp):
            raise FileNotFoundError(f'{self.images_shapes_fp}')

        print(f'loading shapes dict from "{self.images_shapes_fp}"')
        with open(self.images_shapes_fp, 'rb') as fin:
            self.shapes = pickle.load(fin)

        # filter images
        self.shapes = {k: v for (k, v) in self.shapes.items() if k in self.img_ids}

    def _init_slice_info(self):
        self.slice_info = [
            {
                'id': cur_id,
                'scan_fp': os.path.join(self.scans_dp, f'{cur_id}.npy'),
                'mask_fp': os.path.join(self.masks_dp, f'{cur_id}.npy'),
                'z_ix': z
            } for (cur_id, cur_shape) in self.shapes.items() for z in range(cur_shape[2])
        ]

    @property
    def n_images(self):
        return len(self.shapes)

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, ix):
        cur_info = self.slice_info[ix]
        cur_id = cur_info['id']
        z_ix = cur_info['z_ix']

        sample = {
            'scan': utils.load_npy(cur_info['scan_fp'])[:, :, z_ix],
            'mask': utils.load_npy(cur_info['mask_fp'])[:, :, z_ix],
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
