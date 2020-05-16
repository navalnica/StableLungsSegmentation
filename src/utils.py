import datetime
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from glob import glob
from typing import List

import humanize
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import torch
import yaml

import const


def get_class_name(cls):
    name = type(cls).__name__
    return name


def check_var_to_be_iterable_collection(var):
    check = var is None or isinstance(var, (list, tuple, np.ndarray))
    if not check:
        raise ValueError(f'expected iterable collection to be passed. got "{type(var)}"')


def get_nii_gz_filepaths(dp: str):
    fps = glob(os.path.join(dp, '*.nii.gz'))
    return fps


def get_npy_filepaths(dp: str):
    fps = glob(os.path.join(dp, '*.npy'))
    return fps


def load_nifti(fp: str, load_data=True):
    """
    Read Nifti image and return Nifti1Image together with numpy data array.
    Additionally check slope & intercept to be == (1, 0).
    """
    image = nibabel.load(fp)

    # # TODO: check slope & intercept
    # img_slope, img_inter = img.dataobj.slope, img.dataobj.inter
    # assert (img_slope, img_inter) == (1, 0)

    data = None if not load_data else np.asanyarray(image.dataobj)
    return image, data


def load_nifti_slice(fp: str, ix: int) -> np.ndarray:
    """
    Load single slice from Nifti image along z-axis
    """
    img, _ = load_nifti(fp, load_data=False)
    img_slice = img.dataobj[:, :, ix]
    return img_slice


def load_npy(fp: str):
    """
    Load np.ndarray from file
    """
    data = np.load(fp, allow_pickle=False)
    return data


def store_npy(fp: str, data: np.ndarray):
    """
    Store np.ndarray to file
    """
    np.save(fp, data, allow_pickle=False)


def load_split_from_yaml(split_fp: str):
    with open(split_fp) as in_stream:
        split = yaml.safe_load(in_stream)
    return split


def validate_binary_mask(mask: np.ndarray) -> (bool, str):
    """
    Check mask to have np.uint8 dtype and have values only from {0, 1}.
    """
    ok = True
    message = ''
    if mask.dtype != np.uint8:
        ok = False
        message += f'dtype np.uint8: False;'
    if not np.all((mask == 0) | (mask == 1)):
        ok = False
        message += f'is binary: False'
    return ok, message


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
    check_var_to_be_iterable_collection(ids)

    print(const.SEPARATOR)
    print('get_files_dict()')
    print(f'scans_dp: {scans_dp}')
    print(f'masks_dp: {masks_dp}')

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


def store_nifti_to_file(image: nibabel.Nifti1Image, fp: str):
    image.to_filename(fp)


def change_nifti_data(
        data_new: np.ndarray, nifti_original: nibabel.Nifti1Image, is_scan: bool
):
    """
    Create new Nifti1Image from `data_new` array and header extracted from `nifti_original`.

    :param data_new:        new data array
    :param nifti_original:  previous scan or mask as Nifti1Image
    :param is_scan: if True treat `data_new` as a scan.
                    if False - as a mask data. it's needed to check dtype
    """
    if is_scan:
        assert data_new.dtype == np.int16
    else:
        assert data_new.dtype == np.uint8

    header_new = nifti_original.header.copy()
    header_new.set_data_shape(data_new.shape)
    header_new.set_data_dtype(data_new.dtype)

    nifti_new = nibabel.Nifti1Image(data_new, affine=nifti_original.affine, header=header_new)
    return nifti_new


def get_elapsed_time_str(time_start_seconds: float):
    """
    :param time_start_seconds:
    time in seconds since the Epoch obtained with `time.time()` call
    """
    delta_seconds = time.time() - time_start_seconds
    trimmed = int(np.ceil(delta_seconds))  # trim microseconds
    res = str(datetime.timedelta(seconds=trimmed))
    return res


def yes_no_prompt(prompt_msg: str):
    while True:
        answer = input(f'{prompt_msg} [yes/no]: ').lower()
        if answer == 'yes':
            return True
        elif answer == 'no':
            return False
        else:
            print('please type "yes" or "no"')


def clear_dir_content(dp: str, remove_hidden_files: bool = False):
    print(f'\nclear_dir_content(). dp: "{dp}"')
    dir_content = [os.path.join(dp, x) for x in os.listdir(dp)]
    for c in dir_content:
        if os.path.isdir(c):
            shutil.rmtree(c)
        elif os.path.isfile(c):
            if os.path.basename(c).startswith('.') and not remove_hidden_files:
                continue
            os.unlink(c)


def prompt_to_clear_dir_content_if_nonempty(dp: str, remove_hidden_files: bool = False):
    print(const.SEPARATOR)
    print('prompt_to_clear_dir_content_if_nonempty()')

    if os.path.isdir(dp):
        dir_content = os.listdir(dp)
        dir_content_wo_hidden_files = [x for x in dir_content if not x.startswith('.')]
        print(f'len(content): {len(dir_content)}. len(non hidden): {len(dir_content_wo_hidden_files)}')

        # check if need to delete anything
        if remove_hidden_files and len(dir_content) == 0 or \
                not remove_hidden_files and len(dir_content_wo_hidden_files) == 0:
            return

        to_clear = yes_no_prompt(
            f'\ndirectory "{dp}" is not empty.\ndo you want to clear its content?'
        )
        if to_clear:
            print('removing...')
            clear_dir_content(dp, remove_hidden_files=remove_hidden_files)
        else:
            print('will leave its content as is. it will probably get overwritten.')
    else:
        print(f'"{dp}" is not a directory')


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


def build_learning_curves(metrics: dict, loss_name: str, out_dp: str = None):
    """
    :param metrics: dict of following structure:
    '<metric name>' : {'train': List[float], 'valid': Lists[float]}
    :param loss_name: name of the loss function
    :param out_dp: directory path to save plots. do not save anything if None passed
    """
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

    if out_dp is not None:
        out_fp = os.path.join(out_dp, f'learning_curves_{loss_name}.png')
        fig.savefig(out_fp, dpi=200)

    return fig, ax


def concat_histories_and_store_joined_learning_curves(
        history1: dict, history2: dict,
        warm_start_first_run_epoch_num: int, out_dp: str = None
):
    """
    :param history1: dict of following structure:
    '<metric name>' : {'train': List[float], 'valid': Lists[float]}
    :param history2: dict with the same structure as `history1`
    :param warm_start_first_run_epoch_num: epoch number (starting from 1) 
    with checkpoint that was used as warm start parameters for the second run 
    :param out_dp: directory path to save plots. do not save anything if None passed
    """
    metrics_names_common = sorted(history1['metrics'].keys() & history2['metrics'].keys())

    metrics_common = {}
    for c in metrics_names_common:
        metrics_common[c] = dict()
        for stage in ['train', 'valid']:
            values = history1['metrics'][c][stage][:warm_start_first_run_epoch_num] + \
                     history2['metrics'][c][stage]
            metrics_common[c][stage] = values

    loss_common = history1['loss_name'] \
        if history1['loss_name'] == history2['loss_name'] \
        else '<NO COMMON LOSS>'

    build_learning_curves(
        metrics=metrics_common, loss_name=loss_common, out_dp=out_dp
    )
    history_common = {'loss_name': loss_common, 'metrics': metrics_common}

    return history_common


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
    print('\nprint_cuda_memory_stats()')

    a = humanize.naturalsize(torch.cuda.memory_allocated(device=device))
    am = humanize.naturalsize(torch.cuda.max_memory_allocated(device=device))
    c = humanize.naturalsize(torch.cuda.memory_cached(device=device))
    cm = humanize.naturalsize(torch.cuda.max_memory_cached(device=device))
    print(f'allocated: {a} (max: {am}), cached: {c} (max: {cm})')


def create_heavy_augs_mapping_from_image_description_table(fp):
    """
    :param fp: path to image_mapping.xlsx
    """
    df = pd.read_excel(fp)
    print(f'df.shape: {df.shape}')

    # parse string to list of values
    df['image_ids'] = df['image_ids'].str.replace(r'[\'(),]', '').str.split()

    # explode list of values
    dfe = df.explode('image_ids')
    print(f'df shape after explosion: {df.shape}')

    # calc
    dfe['sum'] = dfe[['obscured', 'bottom', 'middle', 'top', 'shape']].sum(axis=1)
    dfe['hard_case'] = (dfe['severe'] == 1) | (dfe['sum'] > 1)

    print('\nheavy augs value counts:')
    print(dfe['hard_case'].value_counts())

    # prettify
    res = dfe[['image_ids', 'hard_case']].sort_values('image_ids')
    res = res.rename(columns={'image_ids': 'image_id'})

    # store to .csv
    res.to_csv('hard_cases_mapping.csv', index=False)


def get_image_ids_with_hard_cases_in_train_set(
        hard_cases_mapping_fp: str, split_fp: str
):
    print(const.SEPARATOR)
    print('get_image_ids_with_hard_cases_in_train_set()')

    df = pd.read_csv(hard_cases_mapping_fp)
    print(f'hard cases mapping shape: {df.shape}')

    ids_hard = df[df['hard_case']]['image_id'].unique()
    print(f'images with hard cases: {ids_hard.shape[0]}')

    split = load_split_from_yaml(split_fp)
    ids_train = np.unique(split['train'])
    print(f'images in train set: {ids_train.shape[0]}')

    ids_intersection = np.intersect1d(ids_hard, ids_train)
    print(f'images in intersection: {ids_intersection.shape[0]}')

    return ids_intersection
