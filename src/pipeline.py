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
from torch.optim.optimizer import Optimizer

import const
import model.utils as mu
import utils
from data import preprocessing
from data.dataloaders import BaseDataLoader
from model import UNet, MobileNetV2_UNet
from model.losses import *
from model.lr_finder import LRFinder

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
    optimizer = None

    def __init__(self, model_architecture: str, device: torch.device):
        assert model_architecture in ['unet', 'mnet2']
        self.model_architecture = model_architecture
        self.device = device

    def create_net(self) -> nn.Module:
        if self.model_architecture == 'unet':
            self.net = UNet(n_channels=1, n_classes=1)
        elif self.model_architecture == 'mnet2':
            self.net = MobileNetV2_UNet()
        else:
            raise ValueError(f'model_architecture must be in ["unet", "mnet2"]. '
                             f'passed: {self.model_architecture}')

        self.net.to(device=self.device)

        return self.net

    def create_optimizer(self) -> Optimizer:
        """
        It is important to create optimizer only after moving model to appropriate device
        as model's parameters will be different after changing the device.
        """

        # optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        return self.optimizer

    def load_net_from_weights(self, checkpoint_fp: str):
        """load model parameters from checkpoint .pth file"""
        print(f'\nload_net_from_weights()')
        print(f'loading model parameters from "{checkpoint_fp}"')

        self.create_net()

        state_dict = torch.load(checkpoint_fp)
        self.net.load_state_dict(state_dict)

    def train(
            self, train_loader: BaseDataLoader, valid_loader: BaseDataLoader,
            n_epochs: int, loss_func: nn.Module, metrics: List[nn.Module],
            out_dp: str = None, max_batches: int = None, initial_checkpoint_fp: str = None
    ):
        """
        Train wrapper.

        :param max_batches: maximum number of batches for training and validation to perform sanity check
        :param initial_checkpoint_fp: path to .pth checkpoint for warm start
        """

        out_dp = out_dp or const.RESULTS_DN
        # check if dir is nonempty
        utils.prompt_to_clear_dir_content_if_nonempty(out_dp)
        os.makedirs(out_dp, exist_ok=True)

        print(const.SEPARATOR)
        if initial_checkpoint_fp is not None:
            print('training with WARM START')
            self.load_net_from_weights(initial_checkpoint_fp)
        else:
            print('training with COLD START')
            self.create_net()

        self.create_optimizer()

        # consider providing the same tolerance to ReduceLROnPlateau and Early Stopping
        tolerance = 1e-4

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2, min_lr=1e-6,
            threshold=tolerance, patience=2, threshold_mode='abs',
            cooldown=0, verbose=True
        )

        history = mu.train_valid(
            net=self.net, loss_func=loss_func, metrics=metrics,
            train_loader=train_loader, valid_loader=valid_loader,
            optimizer=self.optimizer, scheduler=scheduler, device=self.device,
            n_epochs=n_epochs, es_tolerance=tolerance, es_patience=15,
            out_dp=out_dp, max_batches=max_batches
        )

        # store history dict to .pickle file
        print(const.SEPARATOR)
        history_out_fp = f'{out_dp}/train_history_{history["loss_name"]}.pickle'
        print(f'storing train history dict to "{history_out_fp}"')
        with open(history_out_fp, 'wb') as fout:
            pickle.dump(history, fout)

        utils.print_cuda_memory_stats(self.device)

    # def evaluate_model(self):
    #     print(const.SEPARATOR)
    #     print('evaluate_model()')
    #
    #     if self.net is None:
    #         raise ValueError('must call train() or load_model() before evaluating')
    #
    #     hd, hd_avg = mu.get_hd_for_valid_slices(
    #         self.net, self.device, loss_name, self.indices_valid, self.scans_dp, self.masks_dp
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
            self, checkpoint_fp: str, scans_dp: str, postfix: str,
            ids: List[str] = None, output_dp: str = None
    ):
        """
        :param checkpoint_fp:   path to .pth file with net's params dict
        :param scans_dp:    path directory with .nii.gz scans.
                            will check that scans do not have any postfixes in their filenames.
        :param postfix:     postfix of segmented filenames
        :param ids:    list of image ids to consider. if None segment all scans under `scans_dp`
        :param output_dp:   path to directory to store results of segmentation
        """
        utils.check_var_to_be_iterable_collection(ids)

        print(const.SEPARATOR)
        print('Pipeline.segment_scans()')

        output_dp = output_dp or const.SEGMENTED_DN
        print(f'will store segmented masks under "{output_dp}"')
        os.makedirs(output_dp, exist_ok=True)

        print(f'postfix: {postfix}')

        self.load_net_from_weights(checkpoint_fp)
        scans_fps = utils.get_nii_gz_filepaths(scans_dp)
        print(f'# of .nii.gz files under "{scans_dp}": {len(scans_fps)}')

        # filter filepaths to scans
        scans_fps_filtered = []
        for fp in scans_fps:
            img_id, img_postfix = utils.parse_image_id_from_filepath(fp, get_postfix=True)
            if img_postfix != '' or ids is not None and img_id not in ids:
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

                # clip intensities as during training
                scan_data_clipped = preprocessing.clip_intensities(scan_data)

                segmented_data = mu.segment_single_scan(scan_data_clipped, self.net, self.device)
                segmented_nifti = utils.change_nifti_data(segmented_data, scan_nifti, is_scan=False)

                out_fp = os.path.join(output_dp, f'{cur_id}_{postfix}.nii.gz')
                utils.store_nifti_to_file(segmented_nifti, out_fp)

                pbar.update()

        print(f'\nsegmentation ended. elapsed time: {utils.get_elapsed_time_str(time_start_segmentation)}')
        utils.print_cuda_memory_stats(self.device)

    def lr_find_and_store(
            self, loss_func: nn.Module, train_loader: BaseDataLoader,
            out_dp: str = None
    ):
        """
        LRFinder wrapper.
        """
        out_dp = out_dp or os.path.join(const.RESULTS_DN, const.LR_FINDER_RESULTS_DN)
        os.makedirs(out_dp, exist_ok=True)
        utils.prompt_to_clear_dir_content_if_nonempty(out_dp)

        self.create_net()
        self.create_optimizer()

        lr_finder = LRFinder(
            net=self.net, loss_func=loss_func, optimizer=self.optimizer,
            train_loader=train_loader, device=self.device, out_dp=out_dp
        )
        lr_finder.lr_find()
        lr_finder.store_results()
