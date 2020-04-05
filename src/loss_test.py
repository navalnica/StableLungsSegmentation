import numpy as np
import torch

from model.losses import FocalLoss


def my_loss(output, target):
    loss = torch.mean((output - target))
    return loss


if __name__ == '__main__':
    # x = torch.randn((4, 6), dtype=torch.float32, requires_grad=True)
    # w = torch.randn((6, 4), dtype=torch.float32)
    # true = torch.randn((4, 4), dtype=torch.float32, requires_grad=True)
    # c = torch.matmul(x, w)
    # cc = torch.clamp(c, 0.05, 0.15)
    # loss = MSELoss()
    # value = loss(cc, true)
    # value.backward()

    true = np.array([
        [
            [1, 1],
            [0, 0],
        ],
        [
            [1, 1],
            [0, 0],
        ]
    ])
    pred = np.array([
        [
            [0, 0],
            [1, 1],
        ],
        [
            [1, 0],
            [1, 0],
        ]
    ])

    eps = 1e-3
    # pred = np.where(pred == 0, pred + eps, pred)
    # pred = np.where(pred == 1, pred - eps, pred)

    # true = torch.tensor(true, dtype=torch.float).unsqueeze(1)
    # pred = torch.tensor(pred, dtype=torch.float, requires_grad=True).unsqueeze(1)

    true = torch.randint(0, 2, (2, 1, 2, 2)).float()
    pred = torch.rand((2, 1, 2, 2), requires_grad=True)
    pred_c = torch.clamp(pred, eps, 1 - eps)
    # pred = torch.clamp(torch.rand((2, 1, 2, 2), requires_grad=True), eps, 1 - eps)

    # loss = BCELoss(reduction='mean')
    loss = FocalLoss(alpha=0.75, gamma=2, reduction='mean')
    output = loss(pred_c, true)
    output.backward()
    print(output)
