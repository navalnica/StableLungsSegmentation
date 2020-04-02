import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

import const
import utils


def split_randomly(processed_dp, val_percent=0.15, random_state=17):
    """
    Split filename from scans_dp into train and valid sets.
    Assume that all images are present both in scans_dp and masks_dp.
    """
    print('data_train_test_split.split_randomly()')

    scans_dp = const.DataPaths.get_numpy_scans_dp(processed_dp)
    split_fp = const.DataPaths.get_train_valid_split_fp(processed_dp, is_random_split=True)

    scans_fns = sorted([os.path.basename(x) for x in utils.get_npy_files(scans_dp)])

    fns_train, fns_valid = train_test_split(
        scans_fns, random_state=random_state, test_size=val_percent)
    print(f'# of train, valid files: {len(fns_train)}, {len(fns_valid)}')

    fns_train = sorted(fns_train)
    fns_valid = sorted(fns_valid)

    print(f'storing train_valid split to {split_fp}')
    train_valid_dict = {'train': fns_train, 'valid': fns_valid}
    with open(split_fp, 'w') as out_stream:
        json.dump(train_valid_dict, out_stream, indent=4)


def load_split_from_json(processed_dp):
    split_fp = const.DataPaths.get_train_valid_split_fp(processed_dp)
    with open(split_fp) as in_stream:
        split = json.load(in_stream)
    return split


def check_consistency(processed_dp, check_with_split=False):
    print('data_train_test_split.check_consistency()')

    scans_dp = const.DataPaths.get_numpy_scans_dp(processed_dp)
    masks_dp = const.DataPaths.get_numpy_masks_dp(processed_dp)

    scans_fns = sorted([os.path.basename(x) for x in utils.get_npy_files(scans_dp)])
    masks_fns = sorted([os.path.basename(x) for x in utils.get_npy_files(masks_dp)])

    sd_scans_masks = set(scans_fns).symmetric_difference(set(masks_fns))
    if len(sd_scans_masks) != 0:
        raise ValueError(f'next files do not have either scan or mask pair: {sd_scans_masks}')
    del masks_fns

    if check_with_split:
        split = load_split_from_json(processed_dp)
        sd_scans_split = set(set(scans_fns)).symmetric_difference(
            np.concatenate([split['train'], split['valid']])
        )
        if len(sd_scans_split) != 0:
            raise ValueError(f'next files are not found either in split or in scans dir: {sd_scans_split}')


def store_z_dimensions_to_pickle(processed_dp):
    print('data_train_test_split.store_z_dimensions_to_pickle()')

    scans_dp = const.DataPaths.get_numpy_scans_dp(processed_dp)
    scans_fps = utils.get_npy_files(scans_dp)
    images_z_fp = const.DataPaths.get_images_z_dimensions_fp(processed_dp)
    _ = utils.get_images_z_dimensions(scans_fps, images_z_fp, restore_prev=False)


def get_train_valid_indices(processed_dp):
    """
    Get (filename, z-index) tuples from existing train-valid split and images z-dimension dict
    """
    print(const.SEPARATOR)
    print('train_valid_split.get_train_valid_indices()')

    # check_consistency(processed_dp, check_with_split=True)  # TODO: add 'ignore' list to split.json file

    scans_dp = const.DataPaths.get_numpy_scans_dp(processed_dp)
    scans_fps = utils.get_npy_files(scans_dp)
    images_z_fp = const.DataPaths.get_images_z_dimensions_fp(processed_dp)

    images_z = utils.get_images_z_dimensions(scans_fps, images_z_fp)
    split = load_split_from_json(processed_dp)

    n_slices_total = sum(images_z.values())
    print(f'# of images: {len(scans_fps)}')
    print(f'total # of slices: {n_slices_total}')
    print(f'example of images_z: {list(images_z.items())[:5]}')

    # create indices for each possible scan
    indices_train = [(fn, z) for fn in split['train'] for z in range(images_z[fn])]
    indices_valid = [(fn, z) for fn in split['valid'] for z in range(images_z[fn])]
    n_train = len(indices_train)
    n_valid = len(indices_valid)
    print(f'n_train, n_valid: {n_train, n_valid}')
    # assert n_train + n_valid == n_slices_total, 'wrong number of train/valid slices'  # TODO: improve check
    return indices_train, indices_valid


def main():
    # create images_z and train_valid_split manually
    processed_dp = '/media/rtn/storage/datasets/lungs/dataset/processed_z0.25_new'
    check_consistency(processed_dp, check_with_split=False)
    split_randomly(processed_dp)
    store_z_dimensions_to_pickle(processed_dp)


if __name__ == '__main__':
    main()
