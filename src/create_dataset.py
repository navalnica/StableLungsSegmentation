# TODO: пытанні
#   * што не так са 126 слайсам
#   * 123, 163, 206 - тэлеграм

import os
import shutil

import click
import numpy as np
import tqdm

import const
import preprocessing
import utils


def preprocessing_pipeline(files_dict, out_dp, zoom_factor, aug_cnt=0, store_nifti=True):
    print(const.SEPARATOR)
    print('preprocessing_pipeline():')

    if os.path.isdir(out_dp):
        print(f'will remove dir {out_dp}')
        shutil.rmtree(out_dp)

    numpy_scans_dp = os.path.join(out_dp, 'numpy', 'scans')
    numpy_masks_dp = os.path.join(out_dp, 'numpy', 'masks')
    os.makedirs(numpy_scans_dp, exist_ok=True)
    os.makedirs(numpy_masks_dp, exist_ok=True)

    if store_nifti:
        nifti_dp = os.path.join(out_dp, 'nifti')
        os.makedirs(nifti_dp, exist_ok=True)

    with tqdm.tqdm(total=len(files_dict)) as pbar:
        for ix, (k, v) in enumerate(files_dict.items()):
            pbar.set_description(k)

            scan, scan_data = utils.load_nifti(v['scan'])
            mask, mask_data = utils.load_nifti(v['mask'])
            res_scan_data, res_mask_data, unwanted_indices = preprocessing.preprocess_scan(
                scan_data, mask_data, aug_cnt=aug_cnt, zoom_factor=zoom_factor)

            # check scans to have continuous lung mask
            breaks_cnt = np.sum(np.diff(unwanted_indices) > 1)
            if breaks_cnt != 1:
                print(f'\nWARN: scan {k} has {breaks_cnt} breaks in unwanted indices: {unwanted_indices}\n')

            # store numpy arrays to files
            np.save(os.path.join(numpy_scans_dp, f'{k}.npy'), res_scan_data, allow_pickle=False)
            np.save(os.path.join(numpy_masks_dp, f'{k}.npy'), res_mask_data, allow_pickle=False)

            if store_nifti:
                scan_new = utils.change_nifti_data(
                    data_new=res_scan_data, nifti_original=scan, is_scan=True
                )
                mask_new = utils.change_nifti_data(
                    data_new=res_mask_data, nifti_original=mask, is_scan=False
                )
                utils.store_nifti_to_file(scan_new, os.path.join(nifti_dp, f'{k}.nii.gz'))
                utils.store_nifti_to_file(mask_new, os.path.join(nifti_dp, f'{k}_autolungs.nii.gz'))

            pbar.update()

            # if ix >= 10:
            #     break


def some_test():
    pass
    # k = '155'
    # print(const.SEPARATOR)
    # print(f'test: loading scan {k}')
    # scan = np.load(os.path.join(processed_dp, 'scans', f'{k}.npy'), allow_pickle=False)
    # labels = np.load(os.path.join(processed_dp, 'labels', f'{k}.npy'), allow_pickle=False)
    # utils.print_np_stats(scan, 'scan')
    # utils.print_np_stats(labels, 'labels')


@click.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
def main(launch):
    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    files_dict = utils.get_files_dict(data_paths.scans_dp, data_paths.masks_dp)

    zoom_factor = const.ZOOM_FACTOR
    processed_dp = data_paths.get_processed_dp(zoom_factor)
    preprocessing_pipeline(files_dict, processed_dp, zoom_factor=zoom_factor, aug_cnt=0, store_nifti=True)


if __name__ == '__main__':
    main()
