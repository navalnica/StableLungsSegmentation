import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD

from losses import NegDiceLoss, FocalLoss


def my_loss(output, target):
    loss = torch.mean((output - target))
    return loss


def generate_data(size):
    t1 = torch.tensor([-1, -2, -3, 1, 2, 3], dtype=torch.float32)
    to = torch.tensor([1], dtype=torch.float32)
    tres = to - t1

    x = np.linspace(-10, 10, size)
    err = np.random.normal(0, 3, x.shape)
    y = x ** 3 - 3 * x ** 2 - 12 + err
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = torch.tensor(x_train, dtype=torch.float32, device='cuda').unsqueeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda').unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32, device='cuda').unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda').unsqueeze(1)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    y1 = np.array([
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
    y2 = np.array([
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
    ])

    eps = 0.1
    y2 = np.where(y2 == 0, y2 + eps, y2)
    y2 = np.where(y2 == 1, y2 - eps, y2)

    y1 = torch.tensor(y1, dtype=torch.float).unsqueeze(1)
    y2 = torch.tensor(y2, dtype=torch.float, requires_grad=True).unsqueeze(1)

    loss = NegDiceLoss()
    k_in = y1
    k_out = y2
    l = loss(k_in, k_out)
    # l.backward()
    print(l)

    ## fl example ##

    # # loss = binary_focal_loss([0, 1, 1], [0.1, 0.7, 0.9], gamma=2)
    # # [0.001 0.032 0.001]
    # # - a * (1 - p) ** gamma * log(p)
    # gamma = 2
    # target = np.array([0, 1, 1])
    # preds = np.array([0.1, 0.7, 0.9])
    # p = target * preds + (1 - target) * (1 - preds)
    # res = -(1 - p) ** gamma * np.log(p)

    ## end of fl example ##

    # todo: check FocalLoss for stability
    loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    output = loss(y2, y1)
    output.backward()
    print(output)

    ############################

    x_train, x_test, y_train, y_test = generate_data(2000)
    print(x_train.size(), y_train.size(), x_test.size(), y_test.size())

    model = nn.Sequential(
        nn.Linear(1, 7),
        nn.ReLU(),
        nn.Linear(7, 1)
    ).cuda()

    optimizer = SGD(model.parameters(), lr=0.00001, momentum=0.9)
    criterion = nn.MSELoss(reduction='mean')

    best_loss = 1e30
    best_net_wts = copy.deepcopy(model.state_dict())
    for i in range(500):
        output = model(x_train)
        # loss = criterion(output, y_train)
        loss = my_loss(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_net_wts = copy.deepcopy(model.state_dict())

        print(f'i: {i}. loss: {loss.item()}')

        for p in model.parameters():
            pass

    print(f'best loss: {best_loss}')
    model.load_state_dict(best_net_wts)

    with torch.no_grad():
        preds = model(x_test)
        preds_np = preds.cpu().detach().numpy()
        x_np = x_test.cpu().detach().numpy()
        y_np = y_test.cpu().detach().numpy()
        plt.scatter(x_np, y_np, label='true')
        plt.scatter(x_np, preds_np, label='preds')
        plt.grid()
        plt.savefig('res.png')
