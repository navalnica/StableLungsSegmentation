import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

import const
import utils


def split_randomly(dataset_dp, val_percent=0.15, random_state=17):
    """
    Split filename from scans_dp into train and valid sets.
    Assume that all images are present both in scans_dp and masks_dp.
    """
    print('data_train_test_split.split_randomly()')

    scans_dp = const.get_numpy_scans_dp(dataset_dp)
    split_fp = const.get_train_valid_split_fp(dataset_dp, is_random_split=True)

    scans_fns = sorted([os.path.basename(x) for x in utils.get_npy_filepaths(scans_dp)])

    fns_train, fns_valid = train_test_split(
        scans_fns, random_state=random_state, test_size=val_percent)
    print(f'# of train, valid files: {len(fns_train)}, {len(fns_valid)}')

    fns_train = sorted(fns_train)
    fns_valid = sorted(fns_valid)

    print(f'storing train_valid split to {split_fp}')
    train_valid_dict = {'train': fns_train, 'valid': fns_valid}
    with open(split_fp, 'w') as out_stream:
        json.dump(train_valid_dict, out_stream, indent=4)


def load_split_from_json(dataset_dp):
    split_fp = const.get_train_valid_split_fp(dataset_dp)
    with open(split_fp) as in_stream:
        split = json.load(in_stream)
    return split


def check_consistency(dataset_dp, check_with_split=False):
    """ TODO: improve checks """
    print('data_train_test_split.check_consistency()')

    scans_dp = const.get_numpy_scans_dp(dataset_dp)
    masks_dp = const.get_numpy_masks_dp(dataset_dp)

    scans_fns = sorted([os.path.basename(x) for x in utils.get_npy_filepaths(scans_dp)])
    masks_fns = sorted([os.path.basename(x) for x in utils.get_npy_filepaths(masks_dp)])

    sd_scans_masks = set(scans_fns).symmetric_difference(set(masks_fns))
    if len(sd_scans_masks) != 0:
        raise ValueError(f'next files do not have either scan or mask pair: {sd_scans_masks}')
    del masks_fns

    if check_with_split:
        split = load_split_from_json(dataset_dp)
        sd_scans_split = set(set(scans_fns)).symmetric_difference(
            np.concatenate([split['train'], split['valid']])
        )
        if len(sd_scans_split) != 0:
            raise ValueError(f'next files are not found either in split or in scans dir: {sd_scans_split}')


def main():
    # create train_valid_split manually
    dataset_dp = '/media/rtn/storage/datasets/lungs/dataset/processed_z0.25_new'
    check_consistency(dataset_dp, check_with_split=False)
    split_randomly(dataset_dp)


if __name__ == '__main__':
    main()
