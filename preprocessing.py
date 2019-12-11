import math
import os
import re
import shutil

import cv2
import nibabel
import numpy as np
import tqdm
from scipy.ndimage import morphology as morph, label

import augmentations
import utils


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


def filter_scan(scan):
    lo_threshold = -1300
    hi_threshold = 1500
    res = np.where(scan < lo_threshold, lo_threshold, scan)
    res = np.where(res > hi_threshold, hi_threshold, res)
    return res


def get_binary_mask(source_mask):
    thresh = -500
    res = np.where(source_mask < thresh, 0, 1).astype(np.uint8)
    return res


def calc_ct_body_mask(volume):
    """
    segments body from the ct scan. removes table and background.
    written by Eduard Sniazko
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


def preprocess_scan(scan, labels, aug_cnt, zoom_factor=0.25, to_log=False):
    """
    preprocess CT-scans: segment body, filter and augment slices, binarize lung labels.
    :param aug_cnt: number of augmentations per slice
    """
    assert scan.shape == labels.shape, 'different shapes for the input arrays'

    labels = get_binary_mask(labels)
    scan = scan.astype(np.float32)
    segmented = calc_ct_body_mask(scan)
    filtered = filter_scan(segmented)

    if to_log:
        utils.print_np_stats(scan, 'scan')
        utils.print_np_stats(segmented, 'segmented')
        utils.print_np_stats(filtered, 'filtered')
        utils.print_np_stats(labels, 'labels')
        if zoom_factor is None:
            print('no zoom set')
        print(f'will perform {aug_cnt} augmenations per slice')

    body_pixels_threshold = 100
    lungs_pixels_threshold = 250
    unwanted_ix = []
    res_scan = []
    res_labels = []

    for z in range(filtered.shape[2]):
        filtered_s = filtered[:, :, z]
        labels_s = labels[:, :, z]

        # check number of 1 label pixels before zoom.
        # 1300 was set earlier as the low threshold for scan. it matches the background
        if np.sum(labels_s) < lungs_pixels_threshold or \
                np.sum(filtered_s > -1300) < body_pixels_threshold:
            unwanted_ix.append(z)
            continue

        if zoom_factor is not None:
            filtered_s = zoom_nearest(filtered_s, zoom_factor)
            labels_s = zoom_nearest(labels_s, zoom_factor)

        if aug_cnt > 0:
            slice_augs, labels_augs = augmentations.augment_slice(filtered_s, labels_s, aug_cnt)
            res_scan.extend(slice_augs)
            res_labels.extend(labels_augs)
        else:
            res_scan.append(filtered_s)
            res_labels.append(labels_s)

    res_scan = np.stack(res_scan, axis=2)
    res_labels = np.stack(res_labels, axis=2)

    if to_log:
        print(f'{len(unwanted_ix)} unwanted indices: {unwanted_ix}')
        utils.print_np_stats(res_scan, 'res_scan')
        utils.print_np_stats(res_labels, 'res_labels')
        print('res_labels unique values:', np.unique(res_labels))

    return res_scan, res_labels, unwanted_ix
