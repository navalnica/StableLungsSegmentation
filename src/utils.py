import datetime
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
import tqdm
from tabulate import tabulate

import const
from data import augmentations


def get_nii_gz_files(dp: str):
    fps = glob(os.path.join(dp, '*.nii.gz'))
    return fps


def get_npy_files(dp: str):
    fps = glob(os.path.join(dp, '*.npy'))
    return fps


def load_nifti(fp: str):
    """
    Read .nii.gz image and return Nifti1Image together with numpy data array.
    # TODO: use this function instead of nibabel.load(<filepath>).get_data()
        everywhere in the code due to nibabel's deprecation behavior.
    """
    image = nibabel.load(fp)
    data = np.asanyarray(image.dataobj)
    return image, data


def get_nii_file_id(fp: str):
    """Extract idXXXX string from .nii.gz filepath"""
    match = re.match(const.NII_GZ_FP_RE_PATTERN, fp)
    if match is None:
        raise ValueError(f'could not match file "{fp}" against re')
    file_id = match.groups()[0]
    return file_id


def get_files_dict(scans_dp, masks_dp):
    """"
    create dict of the following structure:
        id: {'scan': scan_filepath, 'mask': mask_filepath}
    and leave only those ids that have both scans and masks
    """

    print(const.SEPARATOR)
    print('get_files_dict()')

    d = defaultdict(dict)

    scans_fps = get_nii_gz_files(scans_dp)
    print(f'# of scans found: {len(scans_fps)}')
    for fp in scans_fps:
        file_id = get_nii_file_id(fp)
        d[file_id].update({'scan': fp})

    masks_fps = get_nii_gz_files(masks_dp)
    print(f'# of masks found: {len(masks_fps)}')
    for fp in masks_fps:
        file_id = get_nii_file_id(fp)
        d[file_id].update({'mask': fp})

    d_filtered = {k: v for (k, v) in d.items() if 'scan' in v and 'mask' in v}
    print(f'# of (scan, mask) pairs found: {len(d_filtered)}')

    return d_filtered


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


def copy_checkpoints_to_gdrive(checkpoints_dp):
    date_str = datetime.date.today().strftime('%m.%d.%y')
    gdrive_dest = f'/gdrive/My Drive/datasets/kursavaja_7_sem/{date_str}'
    print(f'store checkpoints to {gdrive_dest}')

    if os.path.isdir(gdrive_dest):
        print('removed old dir')
        shutil.rmtree(gdrive_dest)

    shutil.copytree(checkpoints_dp, gdrive_dest)


def plot_learning_curves(epoch_metrics, dir='results'):
    n_epochs = len(list(epoch_metrics.values())[0]['train'])
    loss_name = epoch_metrics['loss_name']
    x = np.arange(1, n_epochs + 1)
    fig, ax = plt.subplots(1, len(epoch_metrics) - 1, figsize=(6 * len(epoch_metrics), 5), squeeze=False)
    fig.suptitle(f'loss: {loss_name}. epochs: {n_epochs}')

    for (m_name, m_dict), _ax in zip(epoch_metrics.items(), ax.flatten()):
        if type(m_dict) == type(dict()):
            _ax.plot(x, epoch_metrics[m_name]['train'], marker='o', label='train')
            _ax.plot(x, epoch_metrics[m_name]['valid'], marker='o', label='valid')
            _ax.set_title(m_name)
            _ax.grid()
            _ax.legend()

    fig.savefig(f'{dir}/{loss_name}_learning_curves.png', dpi=200)


def squeeze_and_to_numpy(tz):
    return tz.squeeze().cpu().detach().numpy()


def get_images_z_dimensions(images_fps, images_z_fp, restore_prev=True):
    """extract z-dimension for each scan and store to .pickle"""
    if restore_prev:
        if not os.path.isfile(images_z_fp):
            raise ValueError(f'no file found at {images_z_fp}')
        print(f'reading from {images_z_fp}')
        with open(images_z_fp, 'rb') as fin:
            images_z = pickle.load(fin)
    else:
        images_z = {}
        for fp in tqdm.tqdm(images_fps):
            scan = np.load(fp, allow_pickle=False)
            bn = os.path.basename(fp)
            images_z[bn] = scan.shape[2]
        print(f'storing images_z to {images_z_fp}')
        with open(images_z_fp, 'wb') as fout:
            pickle.dump(images_z, fout)

    return images_z


def get_scans_and_masks_batches(
        indices, scans_dp, masks_dp,
        source_slices_per_batch, aug_cnt, to_shuffle):
    """
    create generator that reads slices from numpy array and performs augmentations
    :param indices: tuple (filename, z_index)
    :param source_slices_per_batch: number of slices without augmentations. if None yield scans one by one
    :param aug_cnt: number of augmentations per scan. has no effect if batch_size is None
    :param to_shuffle: whether to shuffle passed indices
    :return:
    """

    if to_shuffle:
        np.random.shuffle(indices)

    foo = lambda indices, dp: (
        np.load(os.path.join(dp, x[0]), allow_pickle=False)[:, :, x[1]]
        for x in indices
    )
    scans_gen = foo(indices, scans_dp)
    masks_gen = foo(indices, masks_dp)

    if source_slices_per_batch is not None:
        # yield by batches
        batch_size = source_slices_per_batch * (1 + aug_cnt)
        scans_batch, masks_batch, ix_batch = [], [], []
        cnt = 0
        for s, m, ix in zip(scans_gen, masks_gen, indices):
            scan_augs, mask_augs = augmentations.augment_slice(s, m, aug_cnt)

            # fig, ax = show_slices(scan_augs);
            # fig.savefig('scan_augs.png')
            # fig, ax = show_slices(mask_augs);
            # fig.savefig('mask_augs.png')

            scans_batch.extend(scan_augs)
            masks_batch.extend(mask_augs)
            ix_batch.extend([ix] * (1 + aug_cnt))
            cnt += aug_cnt + 1

            if cnt % batch_size == 0:
                yield (scans_batch, masks_batch, ix_batch)
                scans_batch, masks_batch, ix_batch = [], [], []
                cnt = 0
        if len(scans_batch) > 0:
            yield (scans_batch, masks_batch, ix_batch)
    else:
        # yield by one item
        for s, m, ix in zip(scans_gen, masks_gen, indices):
            yield (s, m, ix)


def get_single_image_slice_gen(filepath, batch_size=4):
    scan = np.load(filepath, allow_pickle=False)

    z_indices = list(range(scan.shape[2]))
    batch_indices = [z_indices[a:a + batch_size] for a in range(0, scan.shape[2], batch_size)]

    for bi in batch_indices:
        cur_batch = scan[:, :, bi]  # array of shape (H, W, Z)
        cur_batch = np.transpose(cur_batch, [2, 0, 1])  # array of shape (Z, H, W)
        yield cur_batch


def print_cuda_memory_stats(device):
    a = torch.cuda.memory_allocated(device=device) / 1024 / 1024
    am = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024
    c = torch.cuda.memory_cached(device=device) / 1024 / 1024
    cm = torch.cuda.max_memory_cached(device=device) / 1024 / 1024
    print(f'allocated: {a : .2f} (max: {am : .2f}), cached: {c : .2f} (max: {cm : .2f})')


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
