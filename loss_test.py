import numpy as np
import torch

from losses import FocalLoss


def my_loss(output, target):
    loss = torch.mean((output - target))
    return loss


if __name__ == '__main__':
    true = np.array([
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
    ])
    pred = np.array([
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1]
        ]
    ])

    eps = 1e-5
    pred = np.where(pred == 0, pred + eps, pred)
    pred = np.where(pred == 1, pred - eps, pred)

    true = torch.tensor(true, dtype=torch.float).unsqueeze(1)
    pred = torch.tensor(pred, dtype=torch.float, requires_grad=True).unsqueeze(1)

    # loss = NegDiceLoss()
    # k_in = true
    # k_out = pred
    # l = loss(k_in, k_out)
    # # l.backward()
    # print(l)

    # todo: check FocalLoss for stability
    loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    output = loss(pred, true)
    output.backward()
    print(output)

