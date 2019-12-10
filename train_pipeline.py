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

from sklearn.model_selection import train_test_split

from model_blocks import *
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
        load_existing_train_valid_split=False
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

    if load_existing_train_valid_split:
        existing_fns_dir = 'results/existing_filenames'
        print(f'loading train, valid filenames from existing files under "{existing_fns_dir}"')
        with open(f'{existing_fns_dir}/filenames_train.txt') as fin:
            fns_train = [x.strip() for x in fin.readlines() if x.strip()]
        with open(f'{existing_fns_dir}/filenames_valid.txt') as fin:
            fns_valid = [x.strip() for x in fin.readlines() if x.strip()]
    else:
        print('performing train, test split')
        fns_train, fns_valid = train_test_split(
            scans_fns, random_state=random_state, test_size=val_percent)
        print('train, valid filenames cnt: %d, %d' % (len(fns_train), len(fns_valid)))

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
        net, optimizer, device,
        batch_size=4, n_epochs=10,
        max_train_samples=None, max_valid_samples=None,
        checkpoints_dp='./model_checkpoints'
):
    actual_n_train = max_train_samples or len(indices_train)
    actual_n_valid = max_valid_samples or len(indices_valid)

    # early stopping variables
    es_cnt = 0
    es_tolerance = 0.001
    es_patience = 4

    loss_func = nn.BCELoss(reduction='mean')

    # epoch metrics
    em = {m: [] for m in ['bce_train', 'bce_valid', 'dice_train', 'dice_valid']}
    loss_name = 'bce'
    best_loss_valid = 1e+100
    best_epoch_ix = 0
    best_net_wts = copy.deepcopy(net.state_dict())

    net.to(device=device)

    print(separator)
    print('start of the training')
    print(f'parameters:\n\n'
          f'loss function: {type(loss_func).__name__}\n'
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
    train_time_start = time.time()

    for cur_epoch in range(1, n_epochs + 1):
        print(f'\n{"=" * 15} epoch {cur_epoch}/{n_epochs} {"=" * 15}')
        epoch_time_start = time.time()

        # ----------- train ----------- #
        net.train()
        train_gen = get_scans_and_labels_batches(
            indices_train, scans_dp, labels_dp, batch_size)
        running_loss_train = 0

        with tqdm.tqdm(total=actual_n_train, desc=f'epoch {cur_epoch}. train',
                       unit='scan', leave=True) as pbar_t:
            for ix, (scans, labels, scans_ix) in enumerate(train_gen, start=1):
                x = torch.tensor(scans, dtype=torch.float, device=device).unsqueeze(1)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(1)

                out = net(x)
                loss = loss_func(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # multiply to find sum of losses for all the batch items.
                # len(scans) != batch_size for the last batch.
                running_loss_train += loss.item() * len(scans)
                pbar_t.update(len(scans))

                # todo: compute other metrics

                if max_train_samples is not None:
                    if ix * batch_size >= max_train_samples:
                        tqdm.tqdm.write(f'exceeded max_train_samples: {max_train_samples}')
                        break

        em['bce_train'].append(running_loss_train / actual_n_train)
        tqdm.tqdm.write(f'loss train: {em["bce_train"][-1]:.3f}')

        if checkpoints_dp is not None:
            os.makedirs(checkpoints_dp, exist_ok=True)
            torch.save(
                net.state_dict(),
                os.path.join(checkpoints_dp, f'cp_epoch_{cur_epoch}.pth')
            )

        # ----------- validation ----------- #
        valid_gen = get_scans_and_labels_batches(
            indices_valid, scans_dp, labels_dp, None, to_shuffle=False)
        evaluation_res = evaluate_net(
            net, valid_gen, {'bce': loss_func, 'dice': my_dice_score},
            actual_n_valid, device, f'epoch {cur_epoch}. valid')

        em['bce_valid'].append(evaluation_res['bce']['mean'])
        em['dice_valid'].append(evaluation_res['dice']['mean'])

        for x in sorted(em.keys()):
            if 'valid' in x:
                tqdm.tqdm.write(f'{x}: {em[x][-1] : .3f}')

        epoch_time = time.time() - epoch_time_start
        tqdm.tqdm.write(f'epoch completed in {epoch_time // 60}m {epoch_time % 60 : .2f}s')

        # ----------- early stopping ----------- #
        if em[f'{loss_name}_valid'][-1] < best_loss_valid - es_tolerance:
            best_loss_valid = em[f'{loss_name}_valid'][-1]
            best_net_wts = copy.deepcopy(net.state_dict())
            best_epoch_ix = cur_epoch
            es_cnt = 0
            tqdm.tqdm.write(f'epoch {cur_epoch}: new best loss valid: {best_loss_valid : .3f}')
        else:
            es_cnt += 1

        if es_cnt >= es_patience:
            tqdm.tqdm.write(separator)
            tqdm.tqdm.write(f'Early Stopping! no improvements for {es_patience} epochs for {loss_name} metric')
            break

    # ----------- load best model ----------- #
    net.load_state_dict(best_net_wts)

    print(separator)
    train_time = time.time() - train_time_start
    print(f'training completed in {train_time // 60}m {train_time % 60 : .2f}s')
    print(f'best loss valid: {best_loss_valid : .3f}, best epoch: {best_epoch_ix}')

    return em


def main():
    cur_dataset_dp = '/media/storage/datasets/kursavaja/7_sem/preprocessed_z0.25_a5'
    scans_dp = os.path.join(f'{cur_dataset_dp}/scans')
    labels_dp = os.path.join(f'{cur_dataset_dp}/labels')

    print(separator)
    print('create "results" dir if needed')
    os.makedirs('results', exist_ok=True)

    indices_train, indices_valid = get_train_valid_indices(
        scans_dp, labels_dp, restore_prev_z_dims=True, load_existing_train_valid_split=True)

    net = UNet(n_channels=1, n_classes=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # em = train(
    #     indices_train, indices_valid,
    #     scans_dp, labels_dp,
    #     net, optimizer, device,
    #     batch_size=4, n_epochs=4,
    #     max_train_samples=20, max_valid_samples=5)
    #
    # fig, ax = plot_learning_curves(em)
    # fig.savefig('results/learning_curves.png')

    loss_func = nn.BCELoss(reduction='mean')
    # todo: get top losses on the full valid dataset
    evaluate_segmentation(
        net, loss_func, 'results/existing_checkpoints/cp_epoch_9.pth',
        indices_valid[:100], scans_dp, labels_dp)

    print(separator)
    print('cuda memory stats:')
    print_cuda_memory_stats()



if __name__ == '__main__':
    main()
