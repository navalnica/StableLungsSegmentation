import sys

sys.path.append('./src')

import click

import const
import os
from data.datasets import NiftiDataset
from data.train_valid_split import load_split_from_json
from model.losses import *
from pipeline import Pipeline, METRICS_DICT


@click.group()
def cli():
    pass


@cli.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
@click.option('--device', help='device to use',
              type=click.Choice(['cpu', 'cuda:0', 'cuda:1']), default='cuda:0')
@click.option('--epochs', 'n_epochs', help='max number of epochs to train',
              type=click.INT, required=True)
@click.option('--max-batches', help='max number of batches to process. use as sanity check. '
                                    'if no value passed than will process the whole dataset.',
              type=click.INT, default=None)
def train(
        launch: str, device: str,
        n_epochs: int, max_batches: int
):
    loss_func = METRICS_DICT['NegDiceLoss']
    metrics = [
        METRICS_DICT['BCELoss'],
        METRICS_DICT['NegDiceLoss'],
        METRICS_DICT['FocalLoss']
    ]

    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    split = load_split_from_json(data_paths.train_valid_split_fp)
    scans_dp = data_paths.scans_dp
    masks_dp = data_paths.masks_dp
    train_dataset = NiftiDataset(scans_dp, masks_dp, split['train'])
    valid_dataset = NiftiDataset(scans_dp, masks_dp, split['valid'])

    device_t = torch.device(device)
    pipeline = Pipeline(device=device_t)

    pipeline.train(
        train_dataset=train_dataset, valid_dataset=valid_dataset,
        n_epochs=n_epochs, loss_func=loss_func, metrics=metrics,
        train_orig_img_per_batch=4, train_aug_cnt=0, valid_batch_size=4,
        max_batches=max_batches
    )


@cli.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
@click.option('--device', help='device to use',
              type=click.Choice(['cpu', 'cuda:0', 'cuda:1']), default='cuda:0')
@click.option('--checkpoint', 'checkpoint_fn',
              help='checkpoint .pth filename with model parameters. '
                   'the file is searched for under "results/model_checkpoints" dir',
              type=click.STRING, default=None)
@click.option('--scans', 'scans_dp', help='path to directory with nifti scans',
              type=click.STRING, default=None)
@click.option('--subset', help='what scans to segment under --scans dir: '
                               'either all, or the ones from "validation" dataset',
              type=click.Choice(['all', 'validation']), default='validation')
@click.option('--out', 'output_dp', help='path to output directory with segmented masks',
              type=click.STRING, default=None)
@click.option('--postfix', help='postfix to set for segmented masks',
              type=click.STRING, default=None)
def segment_scans(
        launch: str, device: str,
        checkpoint_fn: str, scans_dp: str, subset: str,
        output_dp: str, postfix: str
):
    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    device_t = torch.device(device)
    pipeline = Pipeline(device=device_t)

    checkpoint_fn = checkpoint_fn or 'cp_NegDiceLoss_best.pth'
    checkpoint_fp = os.path.join(const.MODEL_CHECKPOINTS_DP, checkpoint_fn)

    scans_dp = scans_dp or data_paths.scans_dp

    ids_list = None
    if subset == 'validation':
        split = load_split_from_json(data_paths.train_valid_split_fp)
        ids_list = split['valid']

    pipeline.segment_scans(
        checkpoint_fp=checkpoint_fp, scans_dp=scans_dp,
        ids_list=ids_list, output_dp=output_dp, postfix=postfix
    )


if __name__ == '__main__':
    cli()