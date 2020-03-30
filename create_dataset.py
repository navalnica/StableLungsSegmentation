# пытанні
# * што не так са 126 слайсам
# * 123, 163, 206 - тэлеграм
# -----------------
# дарабіць
# * паспрабаваць зменшыць колькасць фільтраў пры згортванні
# * пашукаць альтэрнатывы для BatchNorm

import os
import shutil

import numpy as np
import tqdm

import const
import preprocessing
import utils


def preprocessing_pipeline(files_dict, out_dp, zoom_factor, aug_cnt):
    print(const.SEPARATOR)
    print('preprocessing_pipeline():')

    if os.path.isdir(out_dp):
        print(f'will remove dir {out_dp}')
        shutil.rmtree(out_dp)
    res_scans_dp = os.path.join(out_dp, 'scans')
    res_labels_dp = os.path.join(out_dp, 'masks')
    os.makedirs(res_scans_dp, exist_ok=True)
    os.makedirs(res_labels_dp, exist_ok=True)

    with tqdm.tqdm(total=len(files_dict)) as pbar:
        for ix, (k, v) in enumerate(files_dict.items()):
            pbar.set_description(k)

            scan = utils.get_numpy_arr_from_nii_gz(v['scan'])
            mask = utils.get_numpy_arr_from_nii_gz(v['mask'])
            res_scan, res_labels, unwanted_indices = preprocessing.preprocess_scan(
                scan, mask, aug_cnt=aug_cnt, zoom_factor=zoom_factor)

            # check scans to have continuous lung mask
            breaks_cnt = np.sum(np.diff(unwanted_indices) > 1)
            if breaks_cnt != 1:
                print(f'\nWARN: scan {k} has {breaks_cnt} breaks in unwanted indices: {unwanted_indices}\n')

            np.save(os.path.join(res_scans_dp, f'{k}.npy'), res_scan, allow_pickle=False)
            np.save(os.path.join(res_labels_dp, f'{k}.npy'), res_labels, allow_pickle=False)

            pbar.update()

            # if ix >= 10:
            #     break


def main():
    const.set_launch_type_env_var(True)
    data_paths = const.DataPaths()
    files_dict = utils.get_files_dict(data_paths.scans_dp, data_paths.masks_dp)

    zoom_factor = const.ZOOM_FACTOR
    processed_dp = data_paths.get_processed_dir(zoom_factor)
    preprocessing_pipeline(files_dict, processed_dp, zoom_factor=zoom_factor, aug_cnt=0)


def test():
    pass
    # k = '155'
    # print(const.SEPARATOR)
    # print(f'test: loading scan {k}')
    # scan = np.load(os.path.join(processed_dp, 'scans', f'{k}.npy'), allow_pickle=False)
    # labels = np.load(os.path.join(processed_dp, 'labels', f'{k}.npy'), allow_pickle=False)
    # utils.print_np_stats(scan, 'scan')
    # utils.print_np_stats(labels, 'labels')


if __name__ == '__main__':
    main()
