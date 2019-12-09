"""
TODO
* build evaluation pipeline
* change loss function to dice or focal loss
------------------
* try to validate in batches
* unite train and validation pipeline in single construction
* compare offline augmentation with online
* compare small batches vs large
"""

import copy
import time

import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim

from model_blocks import my_dice_score
from unet import UNet
from utils import *

plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['figure.titlesize'] = 17
plt.rcParams['hist.bins'] = 100
plt.rcParams['image.cmap'] = 'gray'


def get_train_valid_indices(
        scans_dp, labels_dp, val_percent=0.15,
        restore_prev_z_dims=True, random_state=17,
        store_train_valid_filenames_to_file=True
):
    scans_fns = sorted(os.listdir(scans_dp))
    labels_fns = sorted(os.listdir(labels_dp))

    # check scans and labels filenames to match
    sd = set(scans_fns).symmetric_difference(set(labels_fns))
    if len(sd) != 0:
        raise ValueError(f'found not matching files for scans and labels: {sd}')
    del labels_fns
    print('total scans count: %d' % len(scans_fns))

    scans_z = get_scans_z_dimensions(scans_fns, scans_dp, restore_prev=restore_prev_z_dims)
    n_slices_total = sum(scans_z.values())
    print(f'total number of slices: {n_slices_total}')
    print(f'example of scans_z: {list(scans_z.items())[:5]}')

    # split filenames on tran/valid
    fns_train, fns_valid = train_test_split(
        scans_fns, random_state=random_state, test_size=val_percent)
    print('train, valid filenames cnt: %d, %d' % (len(fns_train), len(fns_valid)))

    if store_train_valid_filenames_to_file:
        print('storing train and valid filenames under "results" dir')
        with open('results/filenames_train.txt', 'w') as fout:
            fout.writelines('\n'.join(sorted(fns_train)))
        with open('results/filenames_valid.txt', 'w') as fout:
            fout.writelines('\n'.join(sorted(fns_valid)))

    # create indices for each possible scan
    indices_train = [(fn, z) for fn in fns_train for z in range(scans_z[fn])]
    indices_valid = [(fn, z) for fn in fns_valid for z in range(scans_z[fn])]
    n_train = len(indices_train)
    n_valid = len(indices_valid)
    print(f'n_train, n_valid: {n_train, n_valid}')
    assert n_train + n_valid == n_slices_total, 'wrong number of train/valid slices'
    return indices_train, indices_valid


def train(
        indices_train, indices_valid,
        scans_dp, labels_dp,
        net, criterion, optimizer, device,
        batch_size=4, n_epochs=10,
        max_train_samples=None, max_valid_samples=None,
        checkpoints_dp='./model_checkpoints'):

    actual_n_train = max_train_samples or len(indices_train)
    actual_n_valid = max_valid_samples or len(indices_valid)

    # early stopping variables
    es_cnt = 0
    es_tolerance = 0.001
    es_patience = 4

    # epoch metrics
    best_epoch = 0
    best_dice_score = 0.0
    best_net_wts = copy.deepcopy(net.state_dict())
    em = {m: [] for m in ['loss_train', 'loss_valid', 'dice_score']}

    net.to(device=device)

    print(separator)
    print('start of the training')
    print(f'parameters:\n'
          f'optimizer: {type(optimizer).__name__}\n'
          f'learning rate: {optimizer.defaults["lr"]}\n'
          f'momentum: {optimizer.defaults["momentum"]}\n'
          f'number of epochs: {n_epochs}\n'
          f'batch size: {batch_size}\n'
          f'actual train samples count: {actual_n_train}\n'
          f'actual valid samples count: {actual_n_valid}\n'
          f'device: {device}\n'
          f'checkpoints dir: {os.path.abspath(checkpoints_dp)}\n'
          )

    # count global time of training
    since = time.time()

    for cur_epoch in range(n_epochs):
        print(f'\n{"=" * 15} epoch {cur_epoch + 1}/{n_epochs} {"=" * 15}')
        epoch_time_start = time.time()

        ############# train #############
        net.train()
        train_gen = get_scans_and_labels_batches(
            indices_train, scans_dp, labels_dp, batch_size)
        running_loss_train = 0

        with tqdm.tqdm(total=actual_n_train, desc=f'epoch {cur_epoch + 1}. train',
                       unit='scan', leave=True) as pbar_t:
            for ix, (scans, labels, scans_ix) in enumerate(train_gen, start=1):
                x = torch.tensor(scans, dtype=torch.float, device=device).unsqueeze(1)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(1)

                out = net(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # multiply to find sum of losses for all the batch items.
                # len(scans) != batch_size for the last batch.
                running_loss_train += loss.item() * len(scans)
                pbar_t.update(len(scans))

                if max_train_samples is not None:
                    if ix * batch_size >= max_train_samples:
                        tqdm.tqdm.write(f'exceeded max_train_samples: {max_train_samples}')
                        break

        em['loss_train'].append(running_loss_train / actual_n_train)
        tqdm.tqdm.write(f'loss train: {em["loss_train"][-1]:.3f}')

        if checkpoints_dp is not None:
            os.makedirs(checkpoints_dp, exist_ok=True)
            torch.save(
                net.state_dict(),
                os.path.join(checkpoints_dp, f'cp_epoch_{cur_epoch}.pth')
            )

        ############# validate #############
        net.eval()
        valid_gen = get_scans_and_labels_batches(
            indices_valid, scans_dp, labels_dp, None, to_shuffle=False)
        running_loss_valid = 0
        running_dice_score = 0

        with torch.no_grad():
            with tqdm.tqdm(total=actual_n_valid, desc=f'epoch {cur_epoch + 1}. valid',
                           unit='scan', leave=True) as pbar_v:
                for ix, (s, l, scan_ix) in enumerate(valid_gen, start=1):
                    x = torch.tensor(s, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                    y = torch.tensor(l, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                    out = net(x)
                    out_bin = (out > 0.5).float()

                    loss = criterion(out_bin, y)
                    running_loss_valid += loss.item()
                    ds = my_dice_score(y, out_bin).item()
                    running_dice_score += ds

                    pbar_v.update()

                    if max_valid_samples is not None:
                        if ix >= max_valid_samples:
                            tqdm.tqdm.write(f'exceeded max_valid_batches: {max_valid_samples}')
                            break

        em['loss_valid'].append(running_loss_valid / actual_n_valid)
        em['dice_score'].append(running_dice_score / actual_n_valid)
        tqdm.tqdm.write(f'loss valid: {em["loss_valid"][-1]:.3f}')
        tqdm.tqdm.write(f'dice score valid: {em["dice_score"][-1]:.3f}')

        if em['dice_score'][-1] > best_dice_score:
            best_dice_score = em['dice_score'][-1]
            best_net_wts = copy.deepcopy(net.state_dict())
            best_epoch = cur_epoch
            es_cnt = 0
        elif em['dice_score'][-1] < best_dice_score - es_tolerance:
            es_cnt += 1
        if es_cnt >= es_patience:
            tqdm.tqdm.write(separator)
            tqdm.tqdm.write(f'Early Stopping! no imporvements for {es_patience} epochs')
            break

        epoch_time = time.time() - epoch_time_start
        tqdm.tqdm.write(f'epoch completed in {epoch_time // 60}m {epoch_time % 60 : .2f}s')

    # load the best model from all the epochs
    net.load_state_dict(best_net_wts)

    print(separator)
    train_time = time.time() - since
    print(f'training completed in {train_time // 60}m {train_time % 60 : .2f}s')
    print(f'best dice score: {best_dice_score:.3f}, best epoch: {best_epoch}')

    return em


def main():
    cur_dataset_dp = '/media/storage/datasets/kursavaja/7_sem/preprocessed_z0.25_a5'
    scans_dp = os.path.join(f'{cur_dataset_dp}/scans')
    labels_dp = os.path.join(f'{cur_dataset_dp}/labels')

    print(separator)
    print('create "results" dir if needed')
    os.makedirs('results', exist_ok=True)

    indices_train, indices_valid = get_train_valid_indices(
        scans_dp, labels_dp, restore_prev_z_dims=True)

    net = UNet(n_channels=1, n_classes=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    em = train(
        indices_train, indices_valid,
        scans_dp, labels_dp,
        net, criterion, optimizer, device,
        batch_size=16, n_epochs=10,
        max_train_samples=32, max_valid_samples=32)

    fig, ax = plot_train_stats(em['loss_train'], em['loss_valid'], em['dice_score'])
    fig.savefig('results/training results.png')

    print(separator)
    print('cuda memory stats:')
    print_cuda_memory_stats()

    print(separator)
    if 'DISPLAY' in os.environ:
        print(f'will show plots')
        plt.show()
    else:
        print('will not show plots. no DISPLAY in os.environ')


if __name__ == '__main__':
    main()




"""## visualize results"""
# 
# np.random.shuffle(indices_valid)
# valid_gen = get_scans_and_labels_batches(indices_valid, scans_dp, labels_dp, 6)
# (scans, labels, scans_ix) = next(valid_gen)
# x = torch.tensor(scans, dtype=torch.float, device=device).unsqueeze(1)
# with torch.no_grad():
#     out = net(x)
#     out_bin = (out > 0.5).float()
# 
# x = squeeze_and_to_numpy(x)
# labels = np.array(labels)
# out = squeeze_and_to_numpy(out)
# out_bin = squeeze_and_to_numpy(out_bin)
# items = (x, out, out_bin, labels)
# 
# for it in items:
#     show_slices(it, cols=6, width=4, height=4, titles=scans_ix);
# 
# fig, ax = plt.subplots(1, 4, figsize=(4 * 4, 4))
# for ix, z in enumerate(items):
#     print(z.shape)
#     ax[ix].hist(z.flatten(), bins=100);
