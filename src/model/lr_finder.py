import os

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.optimizer import Optimizer

import const
import utils
from data.dataloaders import BaseDataLoader
from model import utils as mu

sns.set(font_scale=1.3)
sns.set_style({'xtick.bottom': True})


class LRFinder:
    """
    Class that allows to find optimal learning rates to use in 1-cycle policy learning.

    Inspired with implementation in fast.ai library.
    """

    def __init__(
            self, net: nn.Module,
            loss_func: nn.Module, optimizer: Optimizer, train_loader: BaseDataLoader,
            lr_min: float = 1e-8, lr_max: float = 5e1, beta=0.97,
            device: torch.device = torch.device('cuda:0')
    ):
        """
        :param beta: smoothing factor for exponentially weighted window
        """
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

        self.lrs = None
        self.losses = None
        self.raw_losses = None

    def lr_find(self):
        print(const.SEPARATOR)
        print(
            f'LRFinder.lr_find:\n'
            f'model architecture: {utils.get_class_name(self.net)}\n'
            f'loss function: {self.loss_name}\n'
            f'lr_min: {self.lr_min : .3e}\n'
            f'lr_max: {self.lr_max : .3e}\n'
            f'batches: {self.n_batches}\n'
            f'q: {self.q : e}\n'
        )
        print(f'\ntrain_loader:\n{self.train_loader}\n')

        self.lrs = []
        self.losses = []
        self.raw_losses = []
        gen = self.train_loader.get_generator()
        cur_lr = self.lr_min
        avg_loss = 0
        best_smoothed_loss = float('inf')

        with tqdm.tqdm(total=len(self.train_loader), desc='lr_finder',
                       unit='slice', leave=True, bar_format=const.TQDM_BAR_FORMAT) as pbar:
            for (ix, batch) in enumerate(gen, start=1):

                pbar.set_description(f'lr_finder. cur lr: {cur_lr : .3e}')

                xb, yb, descriptions_batch = batch
                self.optimizer.param_groups[0]['lr'] = cur_lr

                batch_stats = mu.loss_batch(
                    net=self.net, x_batch=xb, y_batch=yb, loss_func=self.loss_func,
                    metrics=[], device=self.device, optimizer=self.optimizer
                )
                loss_value = batch_stats[self.loss_name]

                # calculate exponentially weighted average
                avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_value
                # perform bias correction
                smoothed_loss = avg_loss / (1 - self.beta ** ix)

                self.lrs.append(cur_lr)
                self.losses.append(smoothed_loss)
                self.raw_losses.append(loss_value)

                # TODO: stopping criteria was never met with NegDiceLoss as it has values from [-1; 0].
                #   So for now it doesn't work. It's not that important so I don't fix it.

                # use abs() to take care for negative valued losses such as Negative Dice Loss
                threshold = best_smoothed_loss + 3 * abs(best_smoothed_loss)
                if ix > 1 and (smoothed_loss > threshold or np.isnan(smoothed_loss)):
                    print(f'smoothed loss exploded:\n'
                          f'cur smoothed loss: {smoothed_loss : .3e}\n'
                          f'best smoothed loss: {best_smoothed_loss : .3e}\n'
                          f'threshold: {threshold : .3e}\n'
                          f'stopping...')
                    self._build_results_df()

                if smoothed_loss < best_smoothed_loss:
                    best_smoothed_loss = smoothed_loss

                cur_lr *= self.q

                pbar.update(len(xb))

        print(f'stopping criteria was not met after the whole epoch.\n'
              f'try to increase the `lr_max` value. or probably there are not enough batches - '
              f'either add augmentations or reset data loader in the end.')
        self._build_results_df()

    def _build_results_df(self):
        assert self.lrs is not None
        assert self.losses is not None
        assert self.raw_losses is not None

        self.results_df = pd.DataFrame({
            'learning_rate': self.lrs, 'loss': self.losses, 'raw_loss': self.raw_losses
        })

    def store_results(self, out_dp: str = None):
        assert os.path.isdir(out_dp)
        assert self.results_df is not None

        print(const.SEPARATOR)
        print(f'LRFinder.store_results: storing results under "{out_dp}"')
        self.results_df.to_csv(os.path.join(out_dp, 'lr_finder_results.csv'), index=False)
        self.plot_loss_values(self.results_df, os.path.join(out_dp, 'lr_finder_plots.png'))

    def plot_loss_values(self, df, out_fp: str = None):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=110)

        ax.plot(df['learning_rate'], df['raw_loss'], label='raw loss', alpha=.5)
        ax.plot(df['learning_rate'], df['loss'], linewidth=1.5, label='smoothed loss', alpha=0.8)

        # configure x axis
        ax.set_xscale('log')
        # configure major and minor ticks
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(1, 10), numticks=50))
        # set ticks style
        ax.tick_params(which='both', width=2)
        ax.tick_params(which='major', length=15)
        ax.tick_params(which='minor', length=8, color='r')

        ax.set_xlabel('learning rate (log scale)')
        ax.set_ylabel(self.loss_name)
        ax.set_title('LR finder results')
        ax.legend(loc='lower left')

        if out_fp is not None:
            fig.savefig(out_fp)
