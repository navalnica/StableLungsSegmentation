import os
import shutil

import click

import const
from train_pipeline import TrainPipeline


@click.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
def main(launch):
    print(const.SEPARATOR)
    print('train_pipeline()')

    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()
    processed_dp = data_paths.get_processed_dp(zoom_factor=0.25, mark_as_new=False)

    device = 'cuda:0'
    # device = 'cuda:1'

    os.makedirs('autolungs', exist_ok=True)

    checkpoint_fp = 'results/model_checkpoints/cp_NegDiceLoss_best.pth'

    segmented_masks_dp = 'autolungs'
    if os.path.isdir(segmented_masks_dp):
        print(f'"{segmented_masks_dp}" dir exists. will remove and create new one')
        shutil.rmtree(segmented_masks_dp)
    os.makedirs(segmented_masks_dp, exist_ok=True)

    pipeline = TrainPipeline(dataset_dp=processed_dp, device=device, n_epochs=8)
    pipeline.segment_valid_scans(checkpoint_fp, segmented_masks_dp)


if __name__ == '__main__':
    main()
