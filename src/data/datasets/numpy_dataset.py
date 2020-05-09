import os
import pickle

import tqdm
from torch.utils.data import Dataset

import const
import utils


class NumpyDataset(Dataset):
    def __init__(self, scans_dp: str, masks_dp: str, images_shapes_fp: str):
        self.scans_dp = scans_dp
        self.masks_dp = masks_dp
        self.images_shapes_fp = images_shapes_fp

        self._load_images_shapes()
        self._init_slice_info()

    def _load_images_shapes(self):
        if not os.path.isfile(self.images_shapes_fp):
            raise FileNotFoundError(f'{self.images_shapes_fp}')
        print(f'loading shapes dict from "{self.images_shapes_fp}"')
        with open(self.images_shapes_fp, 'rb') as fin:
            self.shapes = pickle.load(fin)

    def _init_slice_info(self):
        self.slice_info = [
            {
                'id': cur_id,
                'scan_fp': os.path.join(self.scans_dp, f'{cur_id}.npy'),
                'mask_fp': os.path.join(self.masks_dp, f'{cur_id}.npy'),
                'z_ix': z
            } for (cur_id, cur_shape) in self.shapes.items() for z in range(cur_shape[2])
        ]

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
    def store_images_shapes(dataset_dp, out_fp: str = None):
        """
        Create .pickle file containing dictionary with .npy image shapes
        """
        scans_dp = const.get_numpy_scans_dp(dataset_dp)
        masks_dp = const.get_numpy_masks_dp(dataset_dp)
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
            out_fp = const.get_shapes_fp(dataset_dp)

        print(f'storing shapes dict to "{out_fp}"')
        with open(out_fp, 'wb') as fout:
            pickle.dump(shapes, fout)
