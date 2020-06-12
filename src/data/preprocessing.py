import math
import os
import re
import shutil

import cv2
import nibabel
import numpy as np
import tqdm
from scipy.ndimage import morphology as morph, label

import const
import utils


def fix_resegm_masks(resegm_filenames, fixed_dp):
    if os.path.isdir(fixed_dp):
        print(f'removing directory with fixed scans: {fixed_dp}')
        shutil.rmtree(fixed_dp)
    os.makedirs(fixed_dp)

    pat = r'([^/]+)(.nii.gz)$'
    for fn in tqdm.tqdm(resegm_filenames.values()):
        img, img_data = utils.load_nifti(fn, load_data=False)
        new_affine = utils.diagonal_abs(img.affine)
        new_data = np.flip(img_data, axis=0)
        new_nii = nibabel.Nifti1Image(new_data, new_affine)
        new_fp = os.path.join(fixed_dp, '_fixed'.join(re.search(pat, fn).groups()))
        new_nii.to_filename(new_fp)


def zoom_slice(matrix, zoom_factor: float = None) -> np.ndarray:
    """
    zoom 2D matrix with using only pixel values
    that are present in the source matrix (nearest-neighbor interpolation).
    """
    if zoom_factor is None or zoom_factor == 1:
        return matrix

    dsize = tuple([math.floor(x * zoom_factor) for x in reversed(matrix.shape)])
    res = cv2.resize(matrix, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return res


def zoom_volume_along_x_y(volume: np.ndarray, zoom_factor: float = None) -> np.ndarray:
    """
    zoom 3D np.ndarray along X and Y axes
    """
    if zoom_factor is None or zoom_factor == 1:
        return volume

    res = []
    for z in range(volume.shape[2]):
        cur_slice = volume[:, :, z]
        cur_slice = zoom_slice(cur_slice, zoom_factor)
        res.append(cur_slice)
    res = np.stack(res, axis=2)

    return res


def clip_intensities(
        data: np.ndarray,
        thresh_lo: float = const.BODY_THRESH_LOW,
        thresh_hi: float = const.BODY_THRESH_HIGH
):
    res = np.clip(data, thresh_lo, thresh_hi)
    return res


def threshold_mask(source_mask: np.ndarray, thresh=const.MASK_BINARIZATION_THRESH):
    res = np.where(source_mask < thresh, 0, 1).astype(np.uint8)
    return res


def segment_body_from_scan(volume):
    """
    calculates body mask and removes table with background from the scan.

    written by Eduard Sniazko.

    :param volume: 3D matrix
    """

    # def calc_ct_body_mask(src_filename_, dst_filename_):
    #     if not os.path.isfile(src_filename_):
    #         print('File {} not exists'.format(src_filename_))
    #         return None
    #     img_ = nib.load(src_filename_)
    #     img_affine_ = img_.affine
    #     img_vol_ = img_.get_data()

    img_vol_ = volume.copy()
    res_img_vol_ = np.zeros(img_vol_.shape, np.uint8)
    res_img_vol_[img_vol_[:] > -700] = 1

    res_img_vol_ = morph.binary_erosion(res_img_vol_, iterations=3)
    # res_img_vol_[:, :, 0] = 1
    # res_img_vol_[:, :, res_img_vol_.shape[2] - 1] = 1

    for zz in range(res_img_vol_.shape[2]):
        res_img_vol_[:, :, zz] = morph.binary_fill_holes(res_img_vol_[:, :, zz])
    # res_img_vol_[:, :, 0] = 0
    # res_img_vol_[:, :, res_img_vol_.shape[2] - 1] = 0

    labeled_array, num_features = label(res_img_vol_)
    max_lbl_cnt = 0
    max_lbl_idx = -1
    if num_features > 0:
        for idx in range(num_features):
            cur_cnt = np.sum(labeled_array[:] == idx)
            if num_features == 1:
                max_lbl_idx = 1
                max_lbl_cnt = cur_cnt
            else:
                if cur_cnt > max_lbl_cnt and np.sum(res_img_vol_[labeled_array[:] == idx]) > 0:
                    max_lbl_cnt = cur_cnt
                    max_lbl_idx = idx
    res_img_vol_[labeled_array[:] != max_lbl_idx] = 0
    res_img_vol_[labeled_array[:] == max_lbl_idx] = 1

    res_img_vol_ = res_img_vol_.astype(np.uint8)
    img_vol_[res_img_vol_[:] == 0] = -4_000

    #     if img_affine_[0][0] < 0:
    #         img_vol_ = np.flip(img_vol_, axis=0)
    #         img_affine_[0][0] = -img_affine_[0][0]
    #     img_vol_ = img_vol_.astype(np.int16)
    #     dst_img_ = nib.Nifti1Image(img_vol_, img_affine_)
    #     nib.save(dst_img_, dst_filename_)

    return img_vol_
