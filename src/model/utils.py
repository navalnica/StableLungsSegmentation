import copy
import os
import pickle
import shutil
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
from sklearn.metrics import pairwise_distances
from torch import optim

import const
import utils
from data.dataloader import DataLoader
from model import UNet
from utils import get_single_image_slice_gen


def segment_single_scan(data: np.ndarray, net, device):
    gen = get_single_image_slice_gen(data)
    outs = []

    net.eval()
    with torch.no_grad():
        for scan_slices in gen:
            x = torch.tensor(scan_slices, dtype=torch.float, device=device).unsqueeze(1)
            out = net(x)
            out = utils.squeeze_and_to_numpy(out)
            out = (out > 0.5).astype(np.uint8)

            if len(out.shape) == 2:
                # `out` is an array of shape (H, W)
                outs.append(out)
            elif len(out.shape) == 3:
                # `out` is an array of shape (N, H, W)
                outs.extend(out)

    out_combined = np.stack(outs, axis=2)  # stack along axis 2 to get array of shape (H, W, N)
    return out_combined


# ----------- train functions ----------- #

def loss_batch(
        net: UNet, x_batch, y_batch,
        loss_func: nn.Module, metrics: List[nn.Module],
        device: torch.device, optimizer=None
) -> dict:
    batch_stats = {}
    x = torch.tensor(x_batch, dtype=torch.float, device=device).unsqueeze(1)
    y = torch.tensor(y_batch, dtype=torch.float, device=device).unsqueeze(1)

    out = net(x)
    loss = loss_func(out, y)

    loss_name = utils.get_class_name(loss_func)
    batch_stats[loss_name] = loss.item()

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # It's uncommon to set model to evaluation state with `net.eval()`
    # to calculate additional metrics during training because:
    # * it's resource expensive
    # * metrics calculated with `net.eval()` are similar to ones with `net.train()`
    # According to https://discuss.pytorch.org/t/model-eval-for-train-accuracy/61526
    # So we'll just disable gradient tracking.
    # `net.eval`() must be called outside this function when calculating metrics on validation set
    with torch.no_grad():
        for metric_func in metrics:
            metric_name = utils.get_class_name(metric_func)
            if metric_name != loss_name:
                metric_val = metric_func(out, y)
                batch_stats[metric_name] = metric_val.item()

    return batch_stats


def loss_epoch(
        net: UNet, dataloader: DataLoader,
        loss_func: nn.Module, metrics: List[nn.Module],
        device: torch.device, optimizer=None,
        tqdm_description: str = None, sanity_check: bool = False
) -> dict:
    n_samples = len(dataloader)
    epoch_stats = {utils.get_class_name(metric): 0 for metric in metrics}
    gen = dataloader.get_generator()

    with tqdm.tqdm(total=n_samples, desc=tqdm_description,
                   unit='slice', leave=True) as pbar_t:
        for batch_ix, (scans_batch, masks_batch, descriptions_batch) in enumerate(gen, start=1):
            batch_size = len(scans_batch)
            batch_stats = loss_batch(
                net=net, x_batch=scans_batch, y_batch=masks_batch,
                loss_func=loss_func, metrics=metrics, device=device, optimizer=optimizer
            )

            # TODO: what are the issues of using `reduction='sum'`?
            #   there was one as I remember

            # add values of batch stats to epoch stats.
            # multiply batch_stats by batch_size to handle reduction='mean'
            epoch_stats = {
                k: epoch_stats[k] + batch_stats[k] * batch_size
                for k in epoch_stats.keys() & batch_stats.keys()
            }

            pbar_t.update(batch_size)

            if sanity_check:
                break

    # average stats
    epoch_stats = {k: v / n_samples for k, v in epoch_stats.items()}

    return epoch_stats


def train_valid(
        net: UNet, loss_func: nn.Module, metrics: List[nn.Module],
        train_loader: DataLoader, valid_loader: DataLoader,
        optimizer: optim.SGD, device: torch.device, n_epochs: int,
        checkpoints_dp: str, sanity_check: bool = False
) -> dict:
    """
    :param sanity_check: whether to stop epoch after first batch
    Returns
    -------
    dict with training and validation history of following structure:
    '<metric name>' : {'train': List[float], 'valid': Lists[float]}
    """
    if os.path.isdir(checkpoints_dp):
        print(f'checkpoints dir "{checkpoints_dp}" exists. will remove and create new one')
        shutil.rmtree(checkpoints_dp)
    os.makedirs(checkpoints_dp, exist_ok=True)

    history = {utils.get_class_name(m): {'train': [], 'valid': []} for m in metrics}
    loss_name = utils.get_class_name(loss_func)

    # early stopping variables
    es_cnt = 0
    es_tolerance = 0.0001
    es_patience = 10

    best_loss_valid = float('inf')
    best_epoch_ix = -1
    best_net_params = copy.deepcopy(net.state_dict())

    print(const.SEPARATOR)
    print('start of the training')
    print(f'train parameters:\n\n'
          f'loss function: {loss_name}\n'
          f'optimizer: {optimizer}\n'
          f'number of epochs: {n_epochs}\n'
          f'device: {device}\n'
          f'checkpoints dir: {os.path.abspath(checkpoints_dp)}'
          )
    print(f'\ntrain_loader:\n{train_loader}')
    print(f'\nvalid_loader:\n{valid_loader}')

    # count global time of training
    train_time_start = time.time()

    for cur_epoch in range(1, n_epochs + 1):
        print(f'\n{"=" * 15} epoch {cur_epoch}/{n_epochs} {"=" * 15}')
        epoch_time_start = time.time()

        # ----------- train ----------- #
        net.train()
        tqdm_description = f'epoch {cur_epoch}. train'

        epoch_stats_train = loss_epoch(
            net, train_loader, loss_func, metrics,
            device=device, optimizer=optimizer,
            tqdm_description=tqdm_description, sanity_check=sanity_check
        )

        for m_name, val in epoch_stats_train.items():
            tqdm.tqdm.write(f'{m_name} train: {val : .4f}')

        # ----------- validation ----------- #
        net.eval()
        tqdm_description = f'epoch {cur_epoch}. validation'

        with torch.no_grad():
            epoch_stats_valid = loss_epoch(
                net, valid_loader, loss_func, metrics,
                device=device, optimizer=None,
                tqdm_description=tqdm_description, sanity_check=sanity_check
            )

        for m_name, val in epoch_stats_valid.items():
            tqdm.tqdm.write(f'{m_name} train: {val : .4f}')

        # append epoch stats to history
        for k in epoch_stats_train.keys() & epoch_stats_valid.keys():
            history[k]['train'].append(epoch_stats_train[k])
            history[k]['valid'].append(epoch_stats_valid[k])

        # ----------- print elapsed time for epoch ----------- #
        epoch_td = utils.seconds_to_str(time.time() - epoch_time_start)
        tqdm.tqdm.write(f'time elapsed for epoch: {epoch_td}')

        # ----------- early stopping ----------- #
        if history[loss_name]['valid'][-1] < best_loss_valid - es_tolerance:
            # store parameters
            torch.save(
                net.state_dict(),
                os.path.join(checkpoints_dp, f'cp_{loss_name}_epoch_{cur_epoch}.pth')
            )

            best_loss_valid = history[loss_name]['valid'][-1]
            tqdm.tqdm.write(f'epoch {cur_epoch}: new best loss valid: {best_loss_valid : .4f}')
            best_net_params = copy.deepcopy(net.state_dict())
            best_epoch_ix = cur_epoch
            es_cnt = 0
        else:
            es_cnt += 1

        if es_cnt >= es_patience:
            tqdm.tqdm.write(const.SEPARATOR)
            tqdm.tqdm.write(f'Early Stopping! no improvements for {es_patience} epochs for {loss_name} metric')
            break

    # modify history dict
    history = {'metrics': history}
    history['loss_name'] = loss_name
    history['best_loss_valid'] = best_loss_valid
    history['best_epoch_ix'] = best_epoch_ix

    # save best weights once again
    torch.save(
        best_net_params,
        os.path.join(checkpoints_dp, f'cp_{loss_name}_best.pth')
    )

    # load best model
    # TODO: check if weights of net outside this function are updated
    net.load_state_dict(best_net_params)

    # print summary
    print(const.SEPARATOR)
    train_td = utils.seconds_to_str(time.time() - train_time_start)
    print(f'time elapsed for training: {train_td}')
    print(f'best loss valid: {best_loss_valid : .4f}')
    print(f'best epoch: {best_epoch_ix}')

    return history


# ----------- model evaluation ----------- #

# def evaluate_net(
#         net, data_gen, metrics, total_samples_cnt,
#         device, tqdm_description=None
# ):
#     """
#     Evaluate the net
#
#     :param net: model
#     :param data_gen: data generator
#     :param metrics: dict(metric name: metric function)
#     :param total_samples_cnt: max_valid_samples or len(indices_valid)
#     :param device: device to perform evaluation
#     :param tqdm_description: string to print in tqdm progress bar
#
#     :return: dict: key - metric name, value - {'list': [tuple(slice_ix, value)], 'mean': float}
#     """
#     net.eval()
#     metrics_res = {m_name: {'list': [], 'mean': 0} for m_name, m_func in metrics.items()}
#
#     with torch.no_grad():
#         with tqdm.tqdm(total=total_samples_cnt, desc=tqdm_description,
#                        unit='scan', leave=True) as pbar:
#             for ix, (slice, labels, slice_ix) in enumerate(data_gen, start=1):
#                 x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
#                 y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
#                 out = net(x)
#
#                 for m_name, m_func in metrics.items():
#                     m_value = m_func(out, y).item()
#                     metrics_res[m_name]['mean'] += m_value
#                     metrics_res[m_name]['list'].append((slice_ix, m_value))
#
#                 pbar.update()
#
#     for m_name, m_dict in metrics_res.items():
#         m_dict['mean'] /= total_samples_cnt
#
#     return metrics_res


# def evaluate_segmentation(
#         net, loss_func, state_dict_fp, indices_valid, scans_dp,
#         labels_dp, max_top_losses_cnt=8, device='cuda', dir='results'
# ):
#     """
#     Find slices that have highest loss values for specified loss function.
#     Store indices of such slices to visualize results for the for networks with other weights.
#     Pass the same loss function that the model was trained with.
#     """
#     net.to(device=device)
#     state_dict = torch.load(state_dict_fp)
#     net.load_state_dict(state_dict)
#
#     loss_name = type(loss_func).__name__
#
#     gen = utils.get_scans_and_labels_batches(indices_valid, scans_dp, labels_dp, None, to_shuffle=False)
#     evaluation_res = evaluate_net(net, gen, {'loss': loss_func}, len(indices_valid), device,
#                                   f'evaluation for top losses')
#     top_losses = sorted(evaluation_res['loss']['list'], key=lambda x: x[1], reverse=True)
#     top_losses_indices = [x[0] for x in top_losses[:max_top_losses_cnt]]
#
#     # store top losses slice indices
#     top_losses_indices_fp = f'{dir}/top_losses_indices_{loss_name}.pickle'
#     print(f'storing top losses indices under "{top_losses_indices_fp}"')
#     with open(top_losses_indices_fp, 'wb') as fout:
#         pickle.dump(top_losses_indices, fout)
#
#     visualize_segmentation(net, top_losses_indices, scans_dp,
#                            labels_dp, dir=f'{dir}/top_{loss_name}_values_slices/{loss_name}/')
#
#     return top_losses_indices

# def visualize_worst_best(net, scan_ix_and_hd, average, scans_dp, labels_dp, device, loss_name, dir='results'):
#     # TODO: improve method
#     hd_sorted = sorted(scan_ix_and_hd, key=lambda x: x[1])
#
#     cnt = 8
#     worst = hd_sorted[:-cnt - 1:-1]
#     worst_ix = [x[0] for x in worst]
#     worst_values = [x[1] for x in worst]
#
#     best = hd_sorted[:cnt]
#     best_ix = [x[0] for x in best]
#     best_values = [x[1] for x in best]
#
#     for slice_indices, values, title in zip([worst_ix, best_ix], [worst_values, best_values], ['Worst', 'Best']):
#         gen = utils.get_scans_and_masks_batches(slice_indices, scans_dp, labels_dp, None, aug_cnt=0, to_shuffle=False)
#         fig, ax = plt.subplots(2, cnt // 2, figsize=(5 * cnt // 2, 5 * 2), squeeze=False)
#         net.eval()
#         with torch.no_grad():
#             for (slice, labels, scan_ix), v, _ax in zip(gen, values, ax.flatten()):
#                 x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
#                 out = net(x)
#                 out_bin = (out > 0.5).float()
#                 out_bin_np = utils.squeeze_and_to_numpy(out_bin).astype(np.int)
#
#                 slice_f = img_as_float((slice - np.min(slice)).astype(np.int))
#                 b_true = mark_boundaries(slice_f, labels)
#                 b_pred = mark_boundaries(slice_f, out_bin_np.astype(np.int), color=(1, 0, 0))
#                 b = np.max([b_true, b_pred], axis=0)
#                 _ax.imshow(slice_f, origin='lower')
#                 _ax.imshow(b, alpha=.4, origin='lower')
#                 _ax.set_title(f'{scan_ix}: {v : .3f}')
#
#         fig.tight_layout()
#         fig.subplots_adjust(top=0.85)
#         fig.suptitle(f'{loss_name}. {title} {"Average " if average else ""}Hausdorff distances')
#         fig.savefig(f'{dir}/{loss_name}_hd_{"avg_" if average else ""}{title}.png', dpi=200)


# ----------- Hausdorff distance functions ----------- #

def hausdorff_distance(input_bin, target, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    :param input_bin: HxW tensor
    :param target: HxW tensor
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    """

    # convert to numpy
    v1 = input_bin.cpu().detach().numpy()
    v2 = target.cpu().detach().numpy()

    # get coordinates of class 1 points
    p1 = np.argwhere(v1 == 1)
    p2 = np.argwhere(v2 == 1)

    if len(p1) == 0 or len(p2) == 0:
        return max_ahd

    d = pairwise_distances(p1, p2, metric='euclidean')
    hd1 = np.max(np.min(d, axis=0))
    hd2 = np.max(np.min(d, axis=1))
    res = max(hd1, hd2)

    return res


def average_hausdorff_distance(input_bin, target, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    :param input_bin: HxW tensor
    :param target: HxW tensor
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    """

    # convert to numpy
    v1 = input_bin.cpu().detach().numpy()
    v2 = target.cpu().detach().numpy()

    # get coordinates of class 1 points
    p1 = np.argwhere(v1 == 1)
    p2 = np.argwhere(v2 == 1)

    if len(p1) == 0 or len(p2) == 0:
        return max_ahd

    d = pairwise_distances(p1, p2, metric='euclidean')
    hd1 = np.mean(np.min(d, axis=0))
    hd2 = np.mean(np.min(d, axis=1))
    res = max(hd1, hd2)

    return res


def get_hd_for_valid_slices(net, device, loss_name, indices_valid, scans_dp, labels_dp, dir='results'):
    """
    :param checkpoints: dict(loss_name: checkpoint_path)
    """
    net.eval()
    n_valid = len(indices_valid)

    valid_gen = utils.get_scans_and_masks_batches(
        indices_valid, scans_dp, labels_dp, None, aug_cnt=0, to_shuffle=False)
    hd = []
    hd_avg = []
    with torch.no_grad():
        with tqdm.tqdm(total=n_valid, desc=f'{loss_name} model: hausdorff distance',
                       unit='scan', leave=True) as pbar:
            for ix, (slice, labels, slice_ix) in enumerate(valid_gen, start=1):
                x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                out = net(x)
                out_bin = (out > 0.5).float().squeeze(0).squeeze(0)
                y_squeezed = y.squeeze(0).squeeze(0)

                val = hausdorff_distance(out_bin, y_squeezed)
                hd.append((slice_ix, val))

                val_avg = average_hausdorff_distance(out_bin, y_squeezed)
                hd_avg.append((slice_ix, val_avg))

                pbar.update()

    # store hausdorff distance metric values to .pickle and .txt

    hd = sorted(hd, key=lambda x: x[1], reverse=True)
    with open(f'{dir}/{loss_name}_hd_valid.pickle', 'wb') as fout:
        pickle.dump(hd, fout)
    with open(f'{dir}/{loss_name}_hd_valid.txt', 'w') as fout:
        fout.writelines('\n'.join(map(str, hd)))

    hd_avg = sorted(hd_avg, key=lambda x: x[1], reverse=True)
    with open(f'{dir}/{loss_name}_hd_avg_valid.pickle', 'wb') as fout:
        pickle.dump(hd_avg, fout)
    with open(f'{dir}/{loss_name}_hd_avg_valid.txt', 'w') as fout:
        fout.writelines('\n'.join(map(str, hd_avg)))

    return hd, hd_avg


def build_hd_boxplot(hd_values, average, loss_name, dir='results', ax=None):
    """
    build Hausdorff distances box plot
    """
    store = True if ax is None else False
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(('average ' if average else '') + 'hausdorff values')
    else:
        fig = ax.get_figure()
    if not isinstance(hd_values, np.ndarray):
        hd_values = np.array(hd_values)
    ax.boxplot(hd_values, showfliers=False)
    hd_mean = np.mean(hd_values[np.isfinite(hd_values)])
    ax.set_title(f'{loss_name}. mean: {hd_mean : .3f}')
    if store:
        fig.savefig(f'{dir}/{loss_name}_hd_{"avg_" if average else ""}boxplot.png', dpi=200)

# ----------- not used methods ----------- #

# def build_multiple_hd_boxplots(metrics_hd, average, loss_names_list, dir='results'):
#     """
#     build Hausdorff distances box plot for multiple metrics on the same figure
#     """
#     n = len(metrics_hd)
#     fig, ax = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
#     for hd_tuples_list, loss_name, a in zip(metrics_hd, loss_names_list, ax.flatten()):
#         build_hd_boxplot([x[1] for x in hd_tuples_list], average, loss_name, dir=dir, ax=a)
#     fig.suptitle(f'total {"avg " if average else ""}Hausdorff distances boxplot')
#     fig.savefig(f'{dir}/total_hd_{"avg_" if average else ""}boxplot.png', dpi=200)
#
#
# def build_total_hd_boxplot():
#     # TODO
#     """Build Hausdorff distance boxplots for multiple models on single plot"""
#     for avg in [False, True]:
#         hd = []
#         for m_name in metrics:
#             with open(f'hd_to_plot/{m_name}_hd{"_avg" if avg else ""}_valid.pickle', 'rb') as fin:
#                 hd_cur = pickle.load(fin)
#                 hd.append((m_name, hd_cur))
#         mu.build_multiple_hd_boxplots([x[1] for x in hd], avg, [x[0] for x in hd])
