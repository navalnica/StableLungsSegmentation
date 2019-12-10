import torch
import torch.nn as nn
import tqdm
from torch.nn import functional as F


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
        device, max_samples=None, tqdm_description=None
):
    """
    Evaluate the net

    :param net: model
    :param data_gen: data generator
    :param metrics: dict(metric name: metric function)
    :param total_samples_cnt: total number of samples in data generator
    :param device: device to perform evaluation
    :param max_samples: stop if evaluated on more than max_samples objects.
    pass None to evaluate on the full dataset
    :param tqdm_description: string to print in tqdm progress bar

    :return:
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
                out_bin = (out > 0.5).float()

                for m_name, m_func in metrics.items():
                    # m_value = m_func(y, out).item() # TODO
                    m_value = m_func(y, out_bin).item()
                    metrics_res[m_name]['mean'] += m_value
                    metrics_res[m_name]['list'].append((slice_ix, m_value))

                pbar.update()

                if max_samples is not None:
                    if ix >= max_samples:
                        tqdm.tqdm.write(f'exceeded max_valid_batches: {max_samples}')
                        break

    for m_name, m_dict in metrics_res.items():
        m_dict['mean'] /= total_samples_cnt

    return metrics_res
