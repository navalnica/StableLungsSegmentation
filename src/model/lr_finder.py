import os

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.optim.optimizer import Optimizer

import utils
from data.dataloaders import BaseDataLoader
from model import utils as mu


class LRFinder:
    """
    Class that allows to find optimal learning rates to use in 1-cycle policy learning.
    Inspired with implementation in fast.ai library.
    """

    def __init__(
            self, net: nn.Module,
            loss_func: nn.Module, optimizer: Optimizer, train_loader: BaseDataLoader,
            lr_min: float = 1e-8, lr_max: float = 1e1, beta=0.98,
            device: torch.device = torch.device('cuda:0'),
            out_dp: str = None
    ):
        self.net = net
        self.loss_func = loss_func
        self.loss_name = utils.get_class_name(loss_func)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.n_batches = train_loader.n_batches

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.q = (self.lr_max / self.lr_min) ** (1 / (self.n_batches - 1))

        self.beta = beta
        self.device = device
        self.out_dp = out_dp
        assert os.path.isdir(self.out_dp)

        self.lrs = None
        self.losses = None
        self.raw_losses = None

    def lr_find(self):
        print(f'LRFinder.lr_find:\n'
              f'lr_min: {self.lr_min : .3e}, lr_max: {self.lr_max : .3e},\n'
              f'q: {self.q : e}, batches: {self.n_batches}\n')

        self.lrs = []
        self.losses = []
        self.raw_losses = []
        gen = self.train_loader.get_generator()
        cur_lr = self.lr_min
        avg_loss = 0
        best_smoothed_loss = float('inf')

        with tqdm.tqdm(total=len(self.train_loader), desc='lr_finder',
                       unit='slice', leave=True) as pbar:
            for (ix, batch) in enumerate(gen, start=1):

                pbar.set_description(f'lr_finder. cur lr: {cur_lr : .3e}')

                xb, yb, descriptions_batch = batch
                self.optimizer.param_groups[0]['lr'] = cur_lr

                batch_stats = mu.loss_batch(
                    net=self.net, x_batch=xb, y_batch=yb, loss_func=self.loss_func,
                    metrics=[], device=self.device, optimizer=self.optimizer
                )
                loss_value = batch_stats[self.loss_name]

                avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_value
                smoothed_loss = avg_loss / (1 - self.beta ** ix)

                self.lrs.append(cur_lr)
                self.losses.append(smoothed_loss)
                self.raw_losses.append(loss_value)

                # use abs() to take care for negative valued losses such as Negative Dice Loss
                threshold = best_smoothed_loss + 3 * abs(best_smoothed_loss)
                if ix > 1 and (smoothed_loss > threshold or np.isnan(smoothed_loss)):
                    print(f'smoothed loss exploded:\n'
                          f'cur smoothed loss: {smoothed_loss : .3e}\n'
                          f'best smoothed loss: {best_smoothed_loss : .3e}\n'
                          f'threshold: {threshold : .3e}\n'
                          f'stopping...')
                    return self.lrs, self.losses

                if smoothed_loss < best_smoothed_loss:
                    best_smoothed_loss = smoothed_loss

                cur_lr *= self.q

                pbar.update(len(xb))

        print(f'stopping criteria was not met after the whole epoch.\n'
              f'probably there are to few batches in data loader.')
        return self.lrs, self.losses

    def store_results(self):
        assert self.lrs is not None
        assert self.losses is not None
        assert self.raw_losses is not None

        out_fp = os.path.join(self.out_dp, 'lr_finder_results.csv')
        print(f'LRFinder.store_results: storing results to "{out_fp}"')

        df = pd.DataFrame({
            'learning_rate': self.lrs, 'loss': self.losses, 'raw_losses': self.raw_losses
        })
        df.to_csv(out_fp, index=False)
