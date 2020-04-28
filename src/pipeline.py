# TODO
#   * паспрабаваць зменшыць колькасць фільтраў пры згортванні
#   * пашукаць альтэрнатывы для BatchNorm
import os
import pickle
import time
from typing import List

import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import Dataset

import const
import model.utils as mu
import utils
from data.dataloader import DataLoader
from model import UNet
from model.losses import *

plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['figure.titlesize'] = 17
plt.rcParams['hist.bins'] = 100
plt.rcParams['image.cmap'] = 'gray'

METRICS_DICT = {
    'BCELoss': BCELoss(reduction='mean'),
    'NegDiceLoss': NegDiceLoss(),
    'FocalLoss': FocalLoss(gamma=2)
    # FocalLoss(alpha=0.75, gamma=2, reduction='mean'),
}


class Pipeline:
    net = None

    def __init__(self, device: torch.device):
        self.device = device
        self.results_dp = const.RESULTS_DP
        self.checkpoints_dp = const.MODEL_CHECKPOINTS_DP
        self.segmented_dp = const.SEGMENTED_DP

    def create_net(self):
        self.net = UNet(n_channels=1, n_classes=1)
        self.net.to(device=self.device)
        return self.net

    def train(
            self, train_dataset: Dataset, valid_dataset: Dataset,
            n_epochs: int, loss_func: nn.Module, metrics: List[nn.Module],
            train_orig_img_per_batch: int, train_aug_cnt: int, valid_batch_size: int,
            max_batches: int = None
    ):
        # check if `self.results_dp` is nonempty
        utils.prompt_to_clear_dir_content_if_nonempty(self.results_dp)
        # create directories if needed
        os.makedirs(self.results_dp, exist_ok=True)
        os.makedirs(self.checkpoints_dp, exist_ok=True)

        self.create_net()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_loader = DataLoader(
            train_dataset, orig_img_per_batch=train_orig_img_per_batch,
            aug_cnt=train_aug_cnt, to_shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, orig_img_per_batch=valid_batch_size,
            aug_cnt=0, to_shuffle=False
        )

        history = mu.train_valid(
            net=self.net, loss_func=loss_func, metrics=metrics,
            train_loader=train_loader, valid_loader=valid_loader,
            optimizer=optimizer, device=self.device, n_epochs=n_epochs,
            checkpoints_dp=self.checkpoints_dp, max_batches=max_batches
        )

        # store history dict to .pickle file
        print(const.SEPARATOR)
        history_out_fp = f'{self.results_dp}/train_history_{history["loss_name"]}.pickle'
        print(f'storing train history dict to "{history_out_fp}"')
        with open(history_out_fp, 'wb') as fout:
            pickle.dump(history, fout)

        # build and store learning curves plot
        utils.store_learning_curves(history, out_dir=self.results_dp)

        utils.print_cuda_memory_stats(self.device)

    def load_net_from_weights(self, checkpoint_fp: str):
        """load model parameters from checkpoint .pth file"""
        print(f'\nload_net_from_weights()')
        print(f'loading model parameters from "{checkpoint_fp}"')

        self.create_net()

        state_dict = torch.load(checkpoint_fp)
        self.net.load_state_dict(state_dict)

    # def evaluate_model(self):
    #     print(const.SEPARATOR)
    #     print('evaluate_model()')
    #
    #     if self.net is None:
    #         raise ValueError('must call train() or load_model() before evaluating')
    #
    #     # TODO: remove upper bound
    #     hd, hd_avg = mu.get_hd_for_valid_slices(
    #         self.net, self.device, loss_name, self.indices_valid[:], self.scans_dp, self.masks_dp
    #     )
    #
    #     hd_list = [x[1] for x in hd]
    #     mu.build_hd_boxplot(hd_list, False, loss_name)
    #     mu.visualize_worst_best(self.net, hd, False, self.scans_dp, self.masks_dp, self.device, loss_name)
    #
    #     hd_avg_list = [x[1] for x in hd_avg]
    #     mu.build_hd_boxplot(hd_avg_list, True, loss_name)
    #     mu.visualize_worst_best(self.net, hd_avg, True, self.scans_dp, self.masks_dp, self.device, loss_name)

    def segment_scans(
            self, checkpoint_fp: str, scans_dp: str,
            ids_list: List[str] = None, output_dp: str = None, postfix: str = None
    ):
        """
        :param checkpoint_fp:   path to .pth file with net's params dict
        :param scans_dp:    path directory with .nii.gz scans.
                            will check that scans do not have any postfixes in their filenames.
        :param ids_list:    list of image ids to consider. if None segment all scans under `scans_dp`
        :param output_dp:   path to directory to store results of segmentation
        :param postfix:     postfix of segmented filenames
        """
        print(const.SEPARATOR)
        print('Pipeline.segment_scans()')

        output_dp = output_dp or self.segmented_dp
        print(f'will store segmented masks under "{output_dp}"')
        os.makedirs(output_dp, exist_ok=True)

        postfix = postfix or 'autolungs'
        print(f'postfix: {postfix}')

        self.load_net_from_weights(checkpoint_fp)
        scans_fps = utils.get_nii_gz_filepaths(scans_dp)
        print(f'# of .nii.gz files under "{scans_dp}": {len(scans_fps)}')

        # filter filepaths to scans
        scans_fps_filtered = []
        for fp in scans_fps:
            img_id, img_postfix = utils.parse_image_id_from_filepath(fp, get_postfix=True)
            if img_postfix != '' or ids_list is not None and img_id not in ids_list:
                continue
            scans_fps_filtered.append(fp)
        print(f'# of scans left after filtering: {len(scans_fps_filtered)}')

        print('\nstarting segmentation...')
        time_start_segmentation = time.time()

        with tqdm.tqdm(total=len(scans_fps_filtered)) as pbar:
            for fp in scans_fps_filtered:
                cur_id = utils.parse_image_id_from_filepath(fp)
                pbar.set_description(cur_id)

                scan_nifti, scan_data = utils.load_nifti(fp)

                # TODO: add the same preprocessing as during training (`clip_intensities`)
                #  and compare results of segmentation

                segmented_data = mu.segment_single_scan(scan_data, self.net, self.device)
                segmented_nifti = utils.change_nifti_data(segmented_data, scan_nifti, is_scan=False)

                out_fp = os.path.join(output_dp, f'{cur_id}_{postfix}.nii.gz')
                utils.store_nifti_to_file(segmented_nifti, out_fp)

                pbar.update()

        print(f'\nsegmentation ended. elapsed time: {utils.get_elapsed_time_str(time_start_segmentation)}')
        utils.print_cuda_memory_stats(self.device)
