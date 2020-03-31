import os
import shutil

import click
import nibabel
import numpy as np
import tqdm

import const
import preprocessing
import utils


def add_raw_masks(masks_raw_dp, masks_out_dp):
    """add raw (not thresholded) masks into dataset"""

    print(f'\nadd_raw_masks()')
    print(f'masks_raw_dp: {masks_raw_dp}')
    print(f'masks_out_dp: {masks_out_dp}')

    masks_raw_fps = utils.get_nii_gz_files(masks_raw_dp)
    print(f'# of raw masks to add: {len(masks_raw_fps)}')

    os.makedirs(masks_out_dp, exist_ok=True)

    with tqdm.tqdm(total=len(masks_raw_fps)) as pbar:
        for fp in masks_raw_fps:
            pbar.set_description(os.path.basename(fp))

            mask = nibabel.load(fp)
            data = np.asanyarray(mask.dataobj)
            data = preprocessing.threshold_mask(data)

            mask_id = utils.get_nii_file_id(fp)
            fp_new = os.path.join(masks_out_dp, f'{mask_id}_mask.nii.gz')

            mask_new = nibabel.Nifti1Image(data, header=mask.header, affine=mask.affine)
            mask_new.header.set_data_dtype(np.uint8)
            nibabel.save(mask_new, fp_new)

            pbar.update()


def add_binary_masks(masks_bin_dp, masks_out_dp, check_if_binary=False):
    print('\nadd_binary_masks()')
    print(f'masks_bin_dp: {masks_bin_dp}')
    print(f'masks_out_dp: {masks_out_dp}')

    masks_fps = utils.get_nii_gz_files(masks_bin_dp)
    print(f'# of bin masks to add: {len(masks_fps)}')

    os.makedirs(masks_out_dp, exist_ok=True)

    if check_if_binary:
        wrong_masks = check_masks_to_be_binary(masks_fps)

    # copy them into masks_out_dp
    for fp in masks_fps:
        mask_id = utils.get_nii_file_id(fp)
        fp_new = os.path.join(masks_out_dp, f'{mask_id}_mask.nii.gz')
        shutil.copyfile(fp, fp_new)


def check_masks_to_be_binary(masks_fps):
    wrong_masks = []

    # check that masks are truly binary
    with tqdm.tqdm(total=len(masks_fps)) as pbar:
        for fp in masks_fps:
            pbar.set_description(os.path.basename(fp))
            mask, mask_data = utils.load_nifti(fp)

            if mask_data.dtype != np.uint8:
                wrong_masks.append(fp)
                print(f'WARNING! "{os.path.basename(fp)}" has {mask_data.dtype} dtype')

            unique_values = np.unique(mask_data)
            if not np.array_equal(unique_values, [0, 1]):
                wrong_masks.append(fp)
                print(f'WARNING! "{os.path.basename(fp)}" is not binary. unique values: {unique_values}')

            pbar.update()

    return wrong_masks


@click.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
def main(launch):
    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    add_raw_masks(data_paths.masks_raw_dp, data_paths.masks_dp)
    add_binary_masks(data_paths.masks_bin_dp, data_paths.masks_dp, check_if_binary=False)


if __name__ == '__main__':
    main()
