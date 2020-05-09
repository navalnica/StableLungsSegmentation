import os
import pickle
import shutil
from typing import List

import tqdm
from torch.utils.data import Dataset

import const
import utils
from data import preprocessing


class NiftiDataset(Dataset):
    def __init__(self, scans_dp: str, masks_dp: str, img_ids: List[str] = None):
        self.scans_dp = scans_dp
        self.masks_dp = masks_dp
        self.img_ids = img_ids

        self._init_info()
        self._init_slice_info()

    def _init_info(self):
        paths_dict = utils.get_files_dict(self.scans_dp, self.masks_dp, ids=self.img_ids)

        for cur_id, paths in paths_dict.items():
            if self.img_ids is None or cur_id in self.img_ids:
                img_scan, _ = utils.load_nifti(paths['scan_fp'], load_data=False)
                img_mask, _ = utils.load_nifti(paths['mask_fp'], load_data=False)
                assert img_scan.shape == img_mask.shape, (f'scan shape != mask shape. '
                                                          f'id: {cur_id}. '
                                                          f'scan shape: {img_scan.shape}, '
                                                          f'mask shape: {img_mask.shape}')
                paths['shape'] = img_scan.shape

        self.info = paths_dict

    def _init_slice_info(self):
        self.slice_info = [
            {'id': k, 'scan_fp': v['scan_fp'], 'mask_fp': v['mask_fp'], 'z_ix': x}
            for k, v in self.info.items()
            for x in range(v['shape'][2])
        ]

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, ix):
        cur_info = self.slice_info[ix]
        cur_id = cur_info['id']
        z_ix = cur_info['z_ix']

        scan = utils.load_nifti_slice(cur_info['scan_fp'], z_ix)
        mask = utils.load_nifti_slice(cur_info['mask_fp'], z_ix)

        # transforms
        scan = preprocessing.clip_intensities(scan)

        sample = {
            'scan': scan,
            'mask': mask,
            'description': f'{cur_id}_{z_ix}'
        }
        return sample

    def store_as_numpy_dataset(self, out_dp: str, zoom_factor: float = None):
        """
        Convert Nifti images to numpy nd.arrays and store them to .npy files
        to save time on probably time-expensive zoom.
        """
        print(const.SEPARATOR)
        print('NiftiDataset.store_as_numpy_dataset():')

        print(f'\nzoom factor: {zoom_factor}')

        if os.path.isdir(out_dp):
            print(f'\noutput dir "{out_dp}" already exists. \nwill remove and create a new one.')
            shutil.rmtree(out_dp)

        numpy_scans_dp = os.path.join(out_dp, 'numpy', 'scans')
        numpy_masks_dp = os.path.join(out_dp, 'numpy', 'masks')
        nifti_dp = os.path.join(out_dp, 'nifti')
        os.makedirs(numpy_scans_dp, exist_ok=True)
        os.makedirs(numpy_masks_dp, exist_ok=True)
        os.makedirs(nifti_dp, exist_ok=True)

        # store shapes dict for NumpyDataset
        shapes_dict = {k: v['shape'] for (k, v) in self.info.items()}
        shapes_dict_fp = os.path.join(out_dp, 'numpy', 'shapes.pickle')
        with open(shapes_dict_fp, 'wb')as fout:
            pickle.dump(shapes_dict, fout)

        # process and store scans with masks
        with tqdm.tqdm(total=len(self.info)) as pbar:
            for cur_id, cur_info in self.info.items():
                pbar.set_description(f'image: {cur_id}. shape: {cur_info["shape"]}')

                scan_img, scan_data = utils.load_nifti(cur_info['scan_fp'])
                mask_img, mask_data = utils.load_nifti(cur_info['mask_fp'])

                mask_is_ok, msg = utils.validate_binary_mask(mask_data)
                if not mask_is_ok:
                    raise ValueError(f'id: "{cur_id}". {msg}')

                # clip
                scan_data = preprocessing.clip_intensities(scan_data)

                # zoom
                scan_data = preprocessing.zoom_volume_along_x_y(scan_data, zoom_factor)
                mask_data = preprocessing.zoom_volume_along_x_y(mask_data, zoom_factor)

                # one more check
                mask_is_ok, msg = utils.validate_binary_mask(mask_data)
                if not mask_is_ok:
                    raise ValueError(f'id: "{cur_id}". {msg}')

                # store numpy arrays to files
                utils.store_npy(os.path.join(numpy_scans_dp, f'{cur_id}.npy'), scan_data)
                utils.store_npy(os.path.join(numpy_masks_dp, f'{cur_id}.npy'), mask_data)

                # also store processed images to Nifti
                scan_img_new = utils.change_nifti_data(
                    data_new=scan_data, nifti_original=scan_img, is_scan=True
                )
                mask_img_new = utils.change_nifti_data(
                    data_new=mask_data, nifti_original=mask_img, is_scan=False
                )
                utils.store_nifti_to_file(scan_img_new, os.path.join(nifti_dp, f'{cur_id}.nii.gz'))
                utils.store_nifti_to_file(mask_img_new, os.path.join(nifti_dp, f'{cur_id}_autolungs.nii.gz'))

                pbar.update()
