import datetime
import os
import re
import sys
import time
from collections import defaultdict
from glob import glob
from typing import List

import humanize
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
import tqdm
from tabulate import tabulate

import const


def get_class_name(cls):
    name = type(cls).__name__
    return name


def get_nii_gz_filepaths(dp: str):
    fps = glob(os.path.join(dp, '*.nii.gz'))
    return fps


def get_npy_filepaths(dp: str):
    fps = glob(os.path.join(dp, '*.npy'))
    return fps


def load_nifti(fp: str, load_data=True):
    """
    Read Nifti image and return Nifti1Image together with numpy data array.
    # TODO: use this function instead of nibabel.load(<filepath>).get_data()
        everywhere in the code due to nibabel's deprecation behavior.
    """
    image = nibabel.load(fp)
    data = None if not load_data else np.asanyarray(image.dataobj)
    return image, data


def load_nifti_slice(fp: str, ix: int):
    """
    Load single slice from Nifti image along z-axis
    """
    image = nibabel.load(fp)
    slice = image.dataobj[:, :, ix]
    return slice


def load_npy(fp: str):
    data = np.load(fp, allow_pickle=False)
    return data


def parse_image_id_from_filepath(fp: str, get_postfix=False):
    """Extract 'idXXXX' and optional postfix from .nii.gz or .npy filepath"""
    match = re.match(const.IMAGE_FP_RE_PATTERN, fp)
    if match is None:
        raise ValueError(f'could not match file "{fp}" against re')
    file_id = match.groups()[1]
    file_postfix = match.groups()[2]
    res = file_id if not get_postfix else (file_id, file_postfix)
    return res


def get_files_dict(scans_dp, masks_dp, ids: List[str] = None, mask_postfixes=('autolungs', 'mask')):
    """"
    Create dict of the following structure:
    `img_id: {'scan_fp': scan_filepath, 'mask_fp': mask_filepath}`
    and leave only those images that have both scans and masks.

    Function uses id and postfix (optionally) parsed from image filepath.
    If `scans_dp == masks_dp` then scan files only with '' postfix are selected
    and mask files only with postfix in `mask_postfixes` are selected.

    :param scans_dp: scans directory path
    :param masks_dp: masks directory path
    :param ids: list of ids to consider. if None parse all files
    :param mask_postfixes: tuple of valid postfixed for mask files
    """

    print(const.SEPARATOR)
    print('get_files_dict()')

    d = defaultdict(dict)
    same_dirs = (scans_dp == masks_dp)
    print(f'scans_dp == masks_dp: {same_dirs}')
    if same_dirs:
        print(f'will selected masks only with postfix in {mask_postfixes} and scans without any postfix')
    else:
        print('will read any .nii.gz files in both scans_dp and masks_dp folders')

    scans_fps = get_nii_gz_filepaths(scans_dp)
    for fp in scans_fps:
        img_id, img_postfix = parse_image_id_from_filepath(fp, get_postfix=True)
        if ids is None or img_id in ids:
            if not same_dirs or not img_postfix:
                d[img_id].update({'scan_fp': fp})

    masks_fps = get_nii_gz_filepaths(masks_dp)
    for fp in masks_fps:
        img_id, img_postfix = parse_image_id_from_filepath(fp, get_postfix=True)
        if ids is None or img_id in ids:
            if not same_dirs or img_postfix in mask_postfixes:
                d[img_id].update({'mask_fp': fp})

    d_intersection = {k: v for (k, v) in d.items() if 'scan_fp' in v and 'mask_fp' in v}
    scans_wo_masks = [k for (k, v) in d.items() if 'scan_fp' in v and 'mask_fp' not in v]
    masks_wo_scans = [k for (k, v) in d.items() if 'scan_fp' not in v and 'mask_fp' in v]

    print(f'\n# of scans found: {len(scans_fps)}')
    print(f'# of masks found: {len(masks_fps)}')
    print(f'# of images with scans and masks: {len(d_intersection)}')
    print(f'list of scans without masks: {scans_wo_masks}')
    print(f'list of masks without scans: {masks_wo_scans}')

    return d_intersection


def change_nifti_data(
        data_new: np.ndarray, nifti_original: nibabel.Nifti1Image, is_scan: bool
):
    """
    Create new Nifti1Image based on new data array and previous header.

    :param data_new: new data array
    :param nifti_original: previous scan or mask as Nifti1Image
    :param is_scan: True indicates that scan data is passed, False - mask data
    it helps to determine the right dtype to open scans or masks later with LesionLabeller
    """
    if is_scan:
        data_new = data_new.astype(np.int16)
    else:
        data_new = data_new.astype(np.uint8)

    header_new = nifti_original.header.copy()
    header_new.set_data_shape(data_new.shape)
    header_new.set_data_dtype(data_new.dtype)

    nifti_new = nibabel.Nifti1Image(data_new, affine=nifti_original.affine, header=header_new)
    return nifti_new


def store_nifti_to_file(image: nibabel.Nifti1Image, fp: str):
    image.to_filename(fp)


def create_nifti_image_from_mask_data(mask_data: np.ndarray, scan_nifti: nibabel.Nifti1Image):
    """
    create Nifti1Image for uint8 mask created for specified scan
    :param mask_data: np.ndarray with mask
    :param scan_nifti: nifti scan for which mask was created. used to extract affine matrix and slope & intercept
    """
    mask_nifti = nibabel.Nifti1Image(mask_data, scan_nifti.affine)
    assert mask_data.dtype == np.uint8
    assert mask_nifti.get_data_dtype() == np.uint8

    # set slope & intercept
    slope, inter = scan_nifti.dataobj.slope, scan_nifti.dataobj.inter
    assert (slope, inter) == (1, 0)
    mask_nifti.header.set_slope_inter(slope, inter)

    return mask_nifti


def get_elapsed_time_str(time_start_seconds: float):
    """
    :param time_start_seconds:
    time in seconds since the Epoch obtained with `time.time()` call
    """
    delta_seconds = time.time() - time_start_seconds
    trimmed = int(np.ceil(delta_seconds))  # trim microseconds
    res = str(datetime.timedelta(seconds=trimmed))
    return res


def show_slices(
        slices, titles=None,
        cols=4, width=5, height=5, to_show_axis=False
):
    rows = len(slices) // cols + (1 if len(slices) % cols > 0 else 0)
    fig, ax = plt.subplots(
        rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    ax_flatten = ax.flatten()

    if titles is not None:
        if not len(slices) == len(titles):
            raise ValueError('slices and title must have the same len')

    for i in range(len(slices)):
        ax_flatten[i].imshow(slices[i], origin='lower', cmap='gray')
        if not to_show_axis: ax_flatten[i].axis('off')
        if titles is not None: ax_flatten[i].set_title(titles[i])
    for i in range(len(slices), len(ax_flatten)):
        ax_flatten[i].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=(0.85 if titles is not None else 0.95))
    return fig, ax


def diagonal_abs(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('matrix must be square. shape: %s' % str(matrix.shape))
    new = matrix.copy()
    for i in range(len(matrix)):
        new[i, i] = np.abs(new[i, i])
    return new


def print_np_stats(obj, description=None, k=1):
    sz = sys.getsizeof(obj) / 1024 / 1024
    description = '%s: ' % description if description is not None else ''
    s = sorted(np.unique(obj))
    print(f'{description}shape: {obj.shape} dtype: {obj.dtype} min: {s[:k]} max: {s[-k:]} MB: {sz:.3g}')


def get_some_slices_from_img(data):
    z_max = data.shape[2]
    slices = [data[:, :, (z_max // 5 * x)] for x in range(1, 5)]
    return slices


def get_mid_slice(arr, matrix_axis=2):
    if matrix_axis not in [0, 1, 2]:
        raise ValueError('matrix_axis must be in [0, 1, 2]')
    data = arr[:, :, arr.shape[2] // 2] if matrix_axis == 2 \
        else arr[arr.shape[0] // 2, :, :] if matrix_axis == 0 \
        else arr[:, arr.shape[1] // 2, :]
    return data


def store_learning_curves(history: dict, out_dir: str = None):
    """
    :param history: dict of following structure:
    '<metric name>' : {'train': List[float], 'valid': Lists[float]}
    :param out_dir: directory path to save plots
    """
    loss_name = history['loss_name']
    metrics = history['metrics']
    n_epochs = len(list(metrics.values())[0]['train'])

    x = np.arange(1, n_epochs + 1)
    fig, ax = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
    fig.suptitle(f'loss: {loss_name}. epochs: {n_epochs}')

    for (m_name, m_dict), cur_ax in zip(metrics.items(), ax.flatten()):
        cur_ax.plot(x, metrics[m_name]['train'], marker='o', label='train')
        cur_ax.plot(x, metrics[m_name]['valid'], marker='o', label='valid')
        cur_ax.set_title(m_name)
        cur_ax.grid()
        cur_ax.legend()

    fig.savefig(f'{out_dir}/learning_curves_{loss_name}.png', dpi=200)


def squeeze_and_to_numpy(tz):
    return tz.squeeze().cpu().detach().numpy()


def get_single_image_slice_gen(data: np.ndarray, batch_size=4):
    z_indices = list(range(data.shape[2]))
    batch_indices = [z_indices[a:a + batch_size] for a in range(0, data.shape[2], batch_size)]

    for bi in batch_indices:
        cur_batch = data[:, :, bi]  # now it's an array of shape (H, W, N)
        cur_batch = np.transpose(cur_batch, [2, 0, 1])  # make it an array of shape (N, H, W)
        yield cur_batch


def print_cuda_memory_stats(device):
    print(const.SEPARATOR)
    print('cuda memory stats:')

    a = humanize.naturalsize(torch.cuda.memory_allocated(device=device))
    am = humanize.naturalsize(torch.cuda.max_memory_allocated(device=device))
    c = humanize.naturalsize(torch.cuda.memory_cached(device=device))
    cm = humanize.naturalsize(torch.cuda.max_memory_cached(device=device))
    print(f'allocated: {a} (max: {am}), cached: {c} (max: {cm})')


######### unused #########

def compare_pair(pair):
    initial = nibabel.load(pair['initial'])
    resegm = nibabel.load(pair['resegm'])
    transformed = np.flip(resegm.get_data(), axis=0)

    print(affine_tables_to_str([initial, resegm]))

    slices = [get_mid_slice(x, a) for a in [2, 1] for x in [initial.dataobj, resegm.dataobj, transformed]]
    titles = [x for x in ['initial', 'resegm', 'transformed']]
    titles.extend(titles)
    show_slices(slices, titles, cols=3)


def vec2str(vec):
    return ('{:+9.3f}' * len(vec)).format(*vec)


def affine_tables_to_str(nibabel_images):
    headers = [os.path.basename(x.get_filename()) for x in nibabel_images]
    table = [[vec2str(x.affine[i]) for x in nibabel_images] for i in range(4)]
    return tabulate(table, headers=headers, tablefmt='pipe', stralign='center')


def get_unique_signs_for_diag_elements(nibabel_fps):
    diag_initial = []
    for fn in tqdm.tqdm(nibabel_fps):
        nii = nibabel.load(fn)
        diag_initial.append(np.sign(np.diag(nii.affine)))
    unique = np.unique(diag_initial, axis=0)
    return unique


def get_lower_intensities(img_dicts_list, img_type):
    if not img_type in ['initial', 'fixed']:
        raise ValueError("img_type must be in ['initial', 'fixed']")
    lower_intensities = []
    with tqdm.tqdm(total=len(img_dicts_list)) as pbar:
        for k, v in img_dicts_list.items():
            img = nibabel.load(v[img_type])
            m = sorted(np.unique(img.get_data()))
            lower_intensities.append({k: m})
            pbar.update()
    return lower_intensities


def show_pixel_intensities_hist(img_dicts_list, img_ids, columns=4, width=5, height=4):
    rows = len(img_ids) // columns + (1 if len(img_ids) % columns > 0 else 0)
    fig, ax = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    ax_flatten = ax.flatten()
    with tqdm.tqdm(total=len(img_ids)) as pbar:
        for i in range(len(img_ids)):
            img = nibabel.load(img_dicts_list[img_ids[i]]['initial'])
            ax_flatten[i].hist(img.get_data().flatten())
            ax_flatten[i].set_title(img_ids[i])
            pbar.update()
    for i in range(len(img_ids), len(ax_flatten)):
        ax_flatten[i].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig, ax
