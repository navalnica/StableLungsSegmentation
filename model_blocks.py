import os
import pickle

import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
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


def my_dice_score(true, pred):
    eps = 0.00001
    a = true.view(-1)
    b = pred.view(-1)
    intersection = torch.dot(a, b)
    union = torch.sum(a) + torch.sum(b)
    if union == 0:
        intersection = eps
    # print(f'intersection: {intersection}. union: {union}')
    dice = 2 * intersection / (union + 2 * eps)
    return dice


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
                    # m_value = m_func(y, out).item() # TODO
                    m_value = m_func(y, out).item()
                    metrics_res[m_name]['mean'] += m_value
                    metrics_res[m_name]['list'].append((slice_ix, m_value))

                pbar.update()

                if total_samples_cnt is not None:
                    if ix >= total_samples_cnt:
                        break

    for m_name, m_dict in metrics_res.items():
        m_dict['mean'] /= total_samples_cnt

    return metrics_res


def evaluate_segmentation(
        net, loss_func, state_dict_fp, indices_valid, scans_dp,
        labels_dp, max_top_losses_cnt=8, device='cuda', dir='results'
):
    """
    Find slices that have highest loss values for specified loss function.
    Store indices of such slices to visualize results for the for networks with other weights.
    Pass the same loss function that the model was trained with.
    """
    net.to(device=device)
    state_dict = torch.load(state_dict_fp)
    net.load_state_dict(state_dict)

    loss_name = type(loss_func).__name__

    gen = utils.get_scans_and_labels_batches(indices_valid, scans_dp, labels_dp, None, to_shuffle=False)
    evaluation_res = evaluate_net(net, gen, {'loss': loss_func}, len(indices_valid), device,
                                  f'evaluation for top losses')
    top_losses = sorted(evaluation_res['loss']['list'], key=lambda x: x[1], reverse=True)
    top_losses_indices = [x[0] for x in top_losses[:max_top_losses_cnt]]

    # store top losses slice indices
    top_losses_indices_fp = f'{dir}/top_losses_indices_{loss_name}.pickle'
    print(f'storing top losses indices under "{top_losses_indices_fp}"')
    with open(top_losses_indices_fp, 'wb') as fout:
        pickle.dump(top_losses_indices, fout)

    visualize_segmentation(net, top_losses_indices, scans_dp,
                           labels_dp, dir=f'{dir}/top_{loss_name}_values_slices/{loss_name}/')

    return top_losses_indices


def visualize_segmentation(
        net, slice_indices, scans_dp, labels_dp,
        dir='results', scans_on_img=4, device='cuda'
):
    os.makedirs(dir, exist_ok=True)
    net.eval()
    gen = utils.get_scans_and_labels_batches(slice_indices, scans_dp, labels_dp, None, to_shuffle=False)
    fig, ax = plt.subplots(scans_on_img, 3, figsize=(5 * 3, 5 * scans_on_img), squeeze=False)
    j = 1
    i = 0
    with torch.no_grad():
        for slice, labels, scan_ix in gen:
            ax[i][0].imshow(slice, origin='lower')
            ax[i][0].set_title(scan_ix)
            ax[i][1].imshow(labels, origin='lower')
            ax[i][1].set_title('true labels')
            x = torch.tensor(slice, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
            out = net(x)
            out_bin = (out > 0.5).float()
            out_bin_np = utils.squeeze_and_to_numpy(out_bin)
            ax[i][2].imshow(out_bin_np, origin='lower')
            ax[i][2].set_title(f'prediction')

            if (i + 1) % scans_on_img == 0 or ((i + 1) % scans_on_img != 0 and i + 1 == len(slice_indices)):
                fig.tight_layout()
                fig.savefig(f'{dir}/{j}.png', dpi=200)
                j += 1
                i = 0
                fig, ax = plt.subplots(scans_on_img, 3, figsize=(5 * 3, 5 * scans_on_img), squeeze=False)
            else:
                i += 1
