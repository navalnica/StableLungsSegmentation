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
from data import augmentations


def fix_resegm_masks(resegm_filenames, fixed_dp):
    if os.path.isdir(fixed_dp):
        print(f'removing directory with fixed scans: {fixed_dp}')
        shutil.rmtree(fixed_dp)
    os.makedirs(fixed_dp)

    pat = r'([^/]+)(.nii.gz)$'
    for fn in tqdm.tqdm(resegm_filenames.values()):
        nii = nibabel.load(fn)
        new_affine = utils.diagonal_abs(nii.affine)
        new_data = np.flip(nii.get_data(), axis=0)
        new_nii = nibabel.Nifti1Image(new_data, new_affine)
        new_fp = os.path.join(fixed_dp, '_fixed'.join(re.search(pat, fn).groups()))
        new_nii.to_filename(new_fp)


def zoom_nearest(matrix, zoom_factor):
    """
    zoom 2D matrix with using only pixel values present in the source matrix.
    """
    dsize = tuple([math.floor(x * zoom_factor) for x in reversed(matrix.shape)])
    res = cv2.resize(matrix, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return res


def clip_intensities(scan, thresh_lo=const.BODY_THRESH_LOW, thresh_hi=const.BODY_THRESH_HIGH):
    res = np.clip(scan, thresh_lo, thresh_hi)
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


def process_scan_and_mask(scan, mask, aug_cnt=0, zoom_factor=None, to_log=False):
    """
    preprocess single CT-scan:
    segment body, clip values, add optional offline augmentations

    :param aug_cnt: number of augmentations per slice
    """
    assert scan.shape == mask.shape, f'different shapes for input arrays: {scan.shape}, {mask.shape}'

    scan = scan.astype(np.float32)
    if to_log:
        print('preprocess_scan():')
        utils.print_np_stats(scan, 'scan')
        utils.print_np_stats(mask, 'labels')
        print(f'zoom factor: {zoom_factor}')
        print(f'augmentations per slice: {aug_cnt}')

    scan = segment_body_from_scan(scan)
    if to_log:
        utils.print_np_stats(scan, 'scan segmented')

    scan = clip_intensities(scan)
    if to_log:
        utils.print_np_stats(scan, 'scan clipped')

    unwanted_ix = []
    res_body = []
    res_mask = []

    for z in range(scan.shape[2]):
        body_slice = scan[:, :, z]
        mask_slice = mask[:, :, z]

        # check that slices before zoom have enough relevant pixels
        if np.sum(mask_slice) < const.MASK_MIN_PIXELS_THRESH or \
                np.sum(body_slice > const.BODY_THRESH_LOW) < const.BODY_MIN_PIXELS_THRESH:
            # TODO: consider not removing such slices from dataset
            unwanted_ix.append(z)
            continue

        if zoom_factor is not None:
            body_slice = zoom_nearest(body_slice, zoom_factor)
            mask_slice = zoom_nearest(mask_slice, zoom_factor)

        if aug_cnt > 0:
            body_augs, mask_augs = augmentations.augment_slice(body_slice, mask_slice, aug_cnt)
            res_body.extend(body_augs)
            res_mask.extend(mask_augs)
        else:
            res_body.append(body_slice)
            res_mask.append(mask_slice)

    res_body = np.stack(res_body, axis=2)
    res_mask = np.stack(res_mask, axis=2)

    if to_log:
        print(f'{len(unwanted_ix)} unwanted indices: {unwanted_ix}')
        utils.print_np_stats(res_body, 'res_body')
        utils.print_np_stats(res_mask, 'res_mask')
        print('res_mask unique values:', np.unique(res_mask))

    return res_body, res_mask, unwanted_ix
