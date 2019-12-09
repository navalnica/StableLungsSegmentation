# пытанні
# * што не так са 126 слайсам
# * 123, 163, 206 - тэлеграм
# -----------------
# дарабіць
# * праверыць разрвывы ў unwanted indices
# * збалансаваць колькасць выпадкаў з захворваннем у train і valid
# * паспрабаваць focal loss, dice loss
# * паспрабаваць зменшыць колькасць фільтраў пры згортванні
# * пашукаць альтэрнатывы для BatchNorm
# * паспрабаваць розныя тыпы аўгментацыяў
import os
import re
import shutil

import nibabel
import numpy as np
import tqdm

import preprocessing
import utils


def preprocessing_pipeline(scans_dict, res_dp, zoom_factor=0.25, aug_cnt=2):
    if os.path.isdir(res_dp):
        print(f'will remove dir {res_dp}')
        shutil.rmtree(res_dp)
    res_scans_dp = os.path.join(res_dp, 'scans')
    res_labels_dp = os.path.join(res_dp, 'labels')
    os.makedirs(res_scans_dp, exist_ok=True)
    os.makedirs(res_labels_dp, exist_ok=True)

    with tqdm.tqdm(total=len(scans_dict)) as pbar:
        for ix, (k, v) in enumerate(scans_dict.items()):
            scan = nibabel.load(v['initial']).get_data()
            labels = nibabel.load(v['fixed']).get_data()
            res_scan, res_labels, unwanted_indices = preprocessing.preprocess_scan(
                scan, labels, zoom_factor=zoom_factor, aug_cnt=aug_cnt)

            # check scans to have continuous lung mask
            breaks_cnt = np.sum(np.diff(unwanted_indices) > 1)
            if breaks_cnt != 1:
                print(f'\nWARN: scan {k} has {breaks_cnt} breaks in unwanted indices: {unwanted_indices}\n')

            np.save(os.path.join(res_scans_dp, f'{k}.npy'), res_scan, allow_pickle=False)
            np.save(os.path.join(res_labels_dp, f'{k}.npy'), res_labels, allow_pickle=False)

            pbar.update()


            if ix >= 10:
                break


def main():
    datasets_dp = '/media/storage/datasets/kursavaja/7_sem/original_data'
    initial_dp = os.path.join(datasets_dp, 'initial_ct')
    resegm_dp = os.path.join(datasets_dp, 'resegm2_nii')

    pat = r'id([\d]{3})[\w]*.nii.gz$'
    initial_filenames = {
        re.search(pat, x).groups()[0]:
            os.path.join(initial_dp, x) for x in sorted(os.listdir(initial_dp))
    }
    resegm_filenames = {
        re.search(pat, x).groups()[0]:
            os.path.join(resegm_dp, x) for x in sorted(os.listdir(resegm_dp))
    }
    print('initial cnt: %d. resegm cnt: %d' % (len(initial_filenames), len(resegm_filenames)))
    print(sorted(initial_filenames.keys() - resegm_filenames.keys()))
    print(sorted(resegm_filenames.keys() - initial_filenames.keys()))

    fixed_dp = os.path.realpath(os.path.join(datasets_dp, '../test_labels_fixed'))

    # print(utils.separator)
    # print('fixing resegm2_nii labels')
    # preprocessing.fix_resegm_masks(resegm_filenames, fixed_dp)

    scans_dict = {
        k: {
            'initial': initial_filenames[k],
            'resegm': resegm_filenames[k],
            'fixed': os.path.join(fixed_dp, f'id{k}_resegm2_fixed.nii.gz')
        }
        for k in sorted(initial_filenames.keys() & resegm_filenames.keys())
    }

    print(utils.separator)
    print('preprocessing scans and labels')
    res_dp = os.path.realpath(os.path.join(datasets_dp, '../test_preprocessed'))
    preprocessing_pipeline(scans_dict, res_dp, aug_cnt=0)

    k = '155'
    print(utils.separator)
    print(f'test: loading scan {k}')
    scan = np.load(os.path.join(res_dp, 'scans', f'{k}.npy'), allow_pickle=False)
    labels = np.load(os.path.join(res_dp, 'labels', f'{k}.npy'), allow_pickle=False)
    utils.print_np_stats(scan, 'scan')
    utils.print_np_stats(labels, 'labels')


if __name__ == '__main__':
    main()
