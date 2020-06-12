import sys

sys.path.append('./src')

import click

import utils
import const
from data.datasets import *
from data.dataloaders import *
from model.losses import *
from pipeline import Pipeline, METRICS_DICT


@click.group()
def cli():
    pass


@cli.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
@click.option('--architecture', 'model_architecture', help='model architecture (unet, mobilenet)',
              type=click.Choice(['unet', 'mobilenet']), default='mobilenet')
@click.option('--device', help='device to use',
              type=click.Choice(['cpu', 'cuda:0', 'cuda:1']), default='cuda:0')
@click.option('--dataset', 'dataset_type', help='dataset type',
              type=click.Choice(['nifti', 'numpy']), default='numpy')
@click.option('--heavy-augs/--no-heavy-augs', 'apply_heavy_augs', default=True)
@click.option('--epochs', 'n_epochs', help='max number of epochs to train',
              type=click.INT, required=True)
@click.option('--out', 'out_dp', help='path to dir to store training artifacts',
              type=click.STRING, default=None)
@click.option('--max-batches', help='max number of batches to process. use as sanity check. '
                                    'if no value passed than will process the whole dataset.',
              type=click.INT, default=None)
@click.option('--checkpoint', 'initial_checkpoint_fp', help='path to initial .pth checkpoint for warm start',
              type=click.STRING, default=None)
def train(
        launch: str, model_architecture: str, device: str, dataset_type: str,
        apply_heavy_augs: bool, n_epochs: int, out_dp: str, max_batches: int,
        initial_checkpoint_fp: str
):
    loss_func = METRICS_DICT['NegDiceLoss']
    metrics = [
        METRICS_DICT['BCELoss'],
        METRICS_DICT['NegDiceLoss'],
        METRICS_DICT['FocalLoss']
    ]

    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    split = utils.load_split_from_yaml(const.TRAIN_VALID_SPLIT_FP)

    if dataset_type == 'nifti':
        train_dataset = NiftiDataset(data_paths.scans_dp, data_paths.masks_dp, split['train'])
        valid_dataset = NiftiDataset(data_paths.scans_dp, data_paths.masks_dp, split['valid'])
    elif dataset_type == 'numpy':
        ndp = const.NumpyDataPaths(data_paths.default_numpy_dataset_dp)
        train_dataset = NumpyDataset(ndp.scans_dp, ndp.masks_dp, ndp.shapes_fp, split['train'])
        valid_dataset = NumpyDataset(ndp.scans_dp, ndp.masks_dp, ndp.shapes_fp, split['valid'])
    else:
        raise ValueError(f"`dataset` should be in ['nifti', 'numpy']. passed '{dataset_type}'")

    # init train data loader
    if apply_heavy_augs:
        print('\nwill apply heavy augmentations')

        # set different augmentations for hard and general cases
        ids_hard_train = utils.get_image_ids_with_hard_cases_in_train_set(
            const.HARD_CASES_MAPPING, const.TRAIN_VALID_SPLIT_FP
        )
        train_dataset.set_different_aug_cnt_for_two_subsets(1, ids_hard_train, 3)
        # init loader
        train_loader = DataLoaderNoAugmentations(train_dataset, batch_size=4, to_shuffle=True)
    else:
        print('\nwill apply the same augmentations for all train images')
        train_loader = DataLoaderWithAugmentations(
            train_dataset, orig_img_per_batch=8, aug_cnt=1, to_shuffle=True
        )

    valid_loader = DataLoaderNoAugmentations(valid_dataset, batch_size=4, to_shuffle=False)

    device_t = torch.device(device)
    pipeline = Pipeline(model_architecture=model_architecture, device=device_t)

    pipeline.train(
        train_loader=train_loader, valid_loader=valid_loader,
        n_epochs=n_epochs, loss_func=loss_func, metrics=metrics,
        out_dp=out_dp, max_batches=max_batches, initial_checkpoint_fp=initial_checkpoint_fp
    )


@cli.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
@click.option('--architecture', 'model_architecture', help='model architecture (unet, mobilenet)',
              type=click.Choice(['unet', 'mobilenet']), default='unet')
@click.option('--device', help='device to use',
              type=click.Choice(['cpu', 'cuda:0', 'cuda:1']), default='cuda:0')
@click.option('--checkpoint', 'checkpoint_fp',
              help='path to checkpoint .pth file',
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
        launch: str, model_architecture: str, device: str,
        checkpoint_fp: str, scans_dp: str, subset: str,
        output_dp: str, postfix: str
):
    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    device_t = torch.device(device)
    pipeline = Pipeline(model_architecture=model_architecture, device=device_t)

    scans_dp = scans_dp or data_paths.scans_dp

    ids_list = None
    if subset == 'validation':
        split = utils.load_split_from_yaml(const.TRAIN_VALID_SPLIT_FP)
        ids_list = split['valid']

        # TODO
        # ids_list = ['id00502', 'id00521', 'id00527', 'id00668']

    pipeline.segment_scans(
        checkpoint_fp=checkpoint_fp, scans_dp=scans_dp,
        ids=ids_list, output_dp=output_dp, postfix=postfix
    )


@cli.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
@click.option('--scans', 'scans_dp', help='path to directory with nifti scans',
              type=click.STRING, default=None)
@click.option('--masks', 'masks_dp', help='path to directory with nifti binary masks',
              type=click.STRING, default=None)
@click.option('--zoom', 'zoom_factor', help='zoom factor for output images',
              type=click.FLOAT, default=0.25)
@click.option('--out', 'output_dp', help='path to output directory with numpy dataset',
              type=click.STRING, default=None)
def create_numpy_dataset(
        launch: str, scans_dp: str, masks_dp: str, zoom_factor: float, output_dp: str
):
    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()

    scans_dp = scans_dp or data_paths.scans_dp
    masks_dp = masks_dp or data_paths.masks_dp

    numpy_data_root_dp = data_paths.get_numpy_data_root_dp(zoom_factor=zoom_factor)
    output_dp = output_dp or numpy_data_root_dp

    ds = NiftiDataset(scans_dp, masks_dp)
    ds.store_as_numpy_dataset(output_dp, zoom_factor)


if __name__ == '__main__':
    cli()
