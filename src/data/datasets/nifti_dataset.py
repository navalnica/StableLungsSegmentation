from typing import List

from torch.utils.data import Dataset

import utils


class NiftiDataset(Dataset):
    def __init__(self, scans_dp: str, masks_dp: str, img_ids: List[str]):
        self.scans_dp = scans_dp
        self.masks_dp = masks_dp
        self.img_ids = img_ids

        self._init_info()
        self._init_slice_info()

    def _init_info(self):
        paths_dict = utils.get_files_dict(self.scans_dp, self.masks_dp)

        for cur_id, paths in paths_dict.items():
            if cur_id not in self.img_ids:
                continue

            img_scan, _ = utils.load_nifti(paths['scan_fp'], load_data=False)
            img_mask, _ = utils.load_nifti(paths['mask_fp'], load_data=False)
            assert img_scan.shape == img_mask.shape, (f'id: {cur_id}. '
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

        sample = {
            'scan': utils.load_nifti_slice(cur_info['scan_fp'], z_ix),
            'mask': utils.load_nifti_slice(cur_info['mask_fp'], z_ix),
            'description': f'{cur_id}_{z_ix}'
        }
        return sample
