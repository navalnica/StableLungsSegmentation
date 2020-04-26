# TODO
#   * паспрабаваць зменшыць колькасць фільтраў пры згортванні
#   * пашукаць альтэрнатывы для BatchNorm
import os
import pickle
from typing import List

import click
import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import Dataset

import const
import model.utils as mu
import utils
from data.dataloader import DataLoader
from data.datasets import NumpyDataset, NiftiDataset
from data.train_valid_split import load_split_from_json
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

    def __init__(
            self, train_dataset: Dataset, valid_dataset: Dataset,
            loss_func: nn.Module, metrics: List[nn.Module],
            device: torch.device, n_epochs: int, sanity_check: bool = False):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss_func = loss_func
        self.metrics = metrics
        self.device = device
        self.n_epochs = n_epochs
        self.sanity_check = sanity_check

        self.results_dp = 'results'
        self.checkpoints_dp = os.path.join(self.results_dp, 'model_checkpoints')

    def create_net(self):
        self.net = UNet(n_channels=1, n_classes=1)
        self.net.to(device=self.device)
        return self.net

    def train(self):
        os.makedirs(self.results_dp, exist_ok=True)
        os.makedirs(self.checkpoints_dp, exist_ok=True)

        self.create_net()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        train_loader = DataLoader(self.train_dataset, orig_img_per_batch=4, aug_cnt=0, to_shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, orig_img_per_batch=4, aug_cnt=0, to_shuffle=False)

        history = mu.train_valid(
            net=self.net, loss_func=self.loss_func, metrics=self.metrics,
            train_loader=train_loader, valid_loader=valid_loader,
            optimizer=optimizer, device=self.device, n_epochs=self.n_epochs,
            checkpoints_dp=self.checkpoints_dp, sanity_check=self.sanity_check
        )

        # store history dict to .pickle file
        print(const.SEPARATOR)
        history_out_fp = f'{self.results_dp}/train_history_{history["loss_name"]}.pickle'
        print(f'storing train history dict to "{history_out_fp}"')
        with open(history_out_fp, 'wb') as fout:
            pickle.dump(history, fout)

        # build and store learning curves plot
        utils.store_learning_curves(history, out_dir=self.results_dp)

    def load_net_from_weights(self, checkpoint_fp: str):
        """load model parameters from checkpoint .pth file"""
        print(const.SEPARATOR)
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

    def segment_nifti_scans(self, checkpoint_fp: str, filepaths: List[str], out_dp):
        print(const.SEPARATOR)
        print('segment_scans()')

        self.load_net_from_weights(checkpoint_fp)

        with tqdm.tqdm(total=len(filepaths)) as pbar:
            for fp in filepaths:
                pbar.set_description(os.path.basename(fp))

                scan_nifti, scan_data = utils.load_nifti(fp)
                segmented_data = mu.segment_single_scan(scan_data, self.net, self.device)
                segmented_nifti = utils.create_nifti_image_from_mask_data(segmented_data, scan_nifti)

                scan_id = utils.parse_image_id_from_filepath(fp)
                out_fp = os.path.join(out_dp, f'{scan_id}_autolungs.nii.gz')
                utils.store_nifti_to_file(segmented_nifti, out_fp)

                pbar.update()

    def segment_valid_scans(self, checkpoint_fp, segmented_masks_dp):
        """
        # TODO: fix method
        """
        print(const.SEPARATOR)
        print('segment_valid_scans()')

        self.load_net_from_weights(checkpoint_fp)
        fns_valid = load_split_from_json(self.dataset_dp)['valid']

        scans_dp = const.get_numpy_scans_dp(self.dataset_dp)
        nifti_dp = const.get_nifti_dp(self.dataset_dp)

        with tqdm.tqdm(total=len(fns_valid)) as pbar:
            for fn in fns_valid:
                pbar.set_description(fn)

                scan = utils.load_npy(os.path.join(scans_dp, fn))
                segmented = mu.segment_single_scan(scan, self.net, self.device)

                # load corresponding nifti image to extract header and store `out_combined` data as nifti
                img_id = utils.parse_image_id_from_filepath(fn)
                nifti_fp = os.path.join(nifti_dp, f'{img_id}_autolungs.nii.gz')
                nifti, _ = utils.load_nifti(nifti_fp)
                segmented_nifti = utils.change_nifti_data(segmented, nifti, is_scan=False)

                out_fp = os.path.join(segmented_masks_dp, f'{img_id}_autolungs.nii.gz')
                utils.store_nifti_to_file(segmented_nifti, out_fp)

                pbar.update()


@click.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
def main(launch):
    print(const.SEPARATOR)
    print('train_pipeline()')

    loss_func = METRICS_DICT['NegDiceLoss']
    metrics = [
        METRICS_DICT['BCELoss'],
        METRICS_DICT['NegDiceLoss'],
        METRICS_DICT['FocalLoss']
    ]

    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()
    dataset_dp = data_paths.get_processed_dataset_dp(zoom_factor=0.25, mark_as_new=False)

    device = torch.device('cuda:0')
    split = load_split_from_json(data_paths.get_train_valid_split_fp())

    train_dataset = NumpyDataset(dataset_dp, split['train'])
    valid_dataset = NumpyDataset(dataset_dp, split['valid'])
    pipeline = Pipeline(
        train_dataset=train_dataset, valid_dataset=valid_dataset,
        loss_func=loss_func, metrics=metrics,
        device=device, n_epochs=8, sanity_check=False
    )

    # TODO: add as option
    to_train = True

    if to_train:
        pipeline.train()
    else:
        checkpoint_fp = f'results/model_checkpoints/cp_NegDiceLoss_best.pth'
        pipeline.load_net_from_weights(checkpoint_fp)

    # pipeline.evaluate_model()

    utils.print_cuda_memory_stats(device)


def sanity_check():
    loss_func = METRICS_DICT['NegDiceLoss']
    metrics = [
        METRICS_DICT['BCELoss'],
        METRICS_DICT['NegDiceLoss'],
        METRICS_DICT['FocalLoss']
    ]

    device = torch.device('cuda:0')

    scans_dp = 'data/scans/'
    masks_dp = 'data/masks/'
    train_dataset = NiftiDataset(scans_dp, masks_dp, ['id001', 'id002'])
    valid_dataset = NiftiDataset(scans_dp, masks_dp, ['id003'])

    pipeline = Pipeline(
        train_dataset=train_dataset, valid_dataset=valid_dataset,
        loss_func=loss_func, metrics=metrics,
        device=device, n_epochs=50, sanity_check=True
    )

    pipeline.train()
    utils.print_cuda_memory_stats(device)


if __name__ == '__main__':
    sanity_check()
    # main()
