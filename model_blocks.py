import numpy as np
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F

import utils


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        # use non-learnable upsampling if bilinear=True
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is B*C*H*W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def evaluate_net(
        net, data_gen, metrics, total_samples_cnt,
        device, tqdm_description=None
):
    """
    Evaluate the net

    :param net: model
    :param data_gen: data generator
    :param metrics: dict(metric name: metric function)
    :param total_samples_cnt: max_valid_samples or len(indices_valid)
    :param device: device to perform evaluation
    :param tqdm_description: string to print in tqdm progress bar

    :return: dict: key - metric name, value - {'list': [tuple(slice_ix, value)], 'mean': float}
    """
    net.eval()
    metrics_res = {m_name: {'list': [], 'mean': 0} for m_name, m_func in metrics.items()}

    with torch.no_grad():
        with tqdm.tqdm(total=total_samples_cnt, desc=tqdm_description,
                       unit='scan', leave=True) as pbar:
            for ix, (slice, labels, slice_ix) in enumerate(data_gen, start=1):
                x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                out = net(x)

                for m_name, m_func in metrics.items():
                    m_value = m_func(out, y).item()
                    metrics_res[m_name]['mean'] += m_value
                    metrics_res[m_name]['list'].append((slice_ix, m_value))

                pbar.update()

    for m_name, m_dict in metrics_res.items():
        m_dict['mean'] /= total_samples_cnt

    return metrics_res


#
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


def mean_hausdorff_distance(input_bin, target, max_ahd=np.inf):
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


def build_hd_boxplots(net, device, loss_name, indices_valid, scans_dp, labels_dp, dir='results'):
    """
    :param checkpoints: dict(loss_name: checkpoint_path)
    """
    net.eval()
    n_valid = len(indices_valid)

    valid_gen = utils.get_scans_and_labels_batches(
        indices_valid, scans_dp, labels_dp, None, aug_cnt=0, to_shuffle=False)
    hd = []
    with torch.no_grad():
        with tqdm.tqdm(total=n_valid, desc=f'{loss_name} model: hausdorff distance',
                       unit='scan', leave=True) as pbar:
            for ix, (slice, labels, slice_ix) in enumerate(valid_gen, start=1):
                x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                out = net(x)
                out_bin = (out > 0.5).float().squeeze(0).squeeze(0)
                y_squeezed = y.squeeze(0).squeeze(0)
                val = mean_hausdorff_distance(out_bin, y_squeezed)
                hd.append((slice_ix, val))
                pbar.update()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    hd_values = np.array([x[1] for x in hd])
    ax.boxplot(hd_values, showfliers=False)
    hd_mean = np.mean(hd_values[np.isfinite(hd_values)])
    ax.set_title(f'{loss_name}')
    fig.suptitle(f'hausdorff values. mean: {hd_mean : .3f}')
    fig.savefig(f'{dir}/{loss_name}_hd_boxplot.png', dpi=200)
    return hd


def visualize_worst_best(net, hausdorff_list, scans_dp, labels_dp, device, loss_name, dir='results'):
    hd_sorted = sorted(hausdorff_list, key=lambda x: x[1])

    cnt = 6
    worst = hd_sorted[:-cnt - 1:-1]
    worst_ix = [x[0] for x in worst]
    worst_values = [x[1] for x in worst]

    best = hd_sorted[:cnt]
    best_ix = [x[0] for x in best]
    best_values = [x[1] for x in best]

    for slice_indices, values, title in zip([worst_ix, best_ix], [worst_values, best_values], ['worst', 'best']):
        gen = utils.get_scans_and_labels_batches(slice_indices, scans_dp, labels_dp, None, aug_cnt=0, to_shuffle=False)
        fig, ax = plt.subplots(2, cnt // 2, figsize=(5 * cnt // 2, 5 * 2), squeeze=False)
        net.eval()
        with torch.no_grad():
            for (slice, labels, scan_ix), v, _ax in zip(gen, values, ax.flatten()):
                x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                out = net(x)
                out_bin = (out > 0.5).float()
                out_bin_np = utils.squeeze_and_to_numpy(out_bin).astype(np.int)

                slice_f = img_as_float((slice - np.min(slice)).astype(np.int))
                b_true = mark_boundaries(slice_f, labels)
                b_pred = mark_boundaries(slice_f, out_bin_np.astype(np.int), color=(1, 0, 0))
                b = np.max([b_true, b_pred], axis=0)
                _ax.imshow(slice_f, origin='lower')
                _ax.imshow(b, alpha=.4, origin='lower')
                _ax.set_title(f'{scan_ix}: {v : .3f}')

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f'{loss_name}. {title} Hausdorff distances')
        fig.savefig(f'{dir}/{loss_name}_hd_{title}.png', dpi=200)
