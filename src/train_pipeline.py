# TODO
#   * паспрабаваць зменшыць колькасць фільтраў пры згортванні
#   * пашукаць альтэрнатывы для BatchNorm

import copy
import time

import click
from torch import optim
from torch.nn import BCELoss

import model.utils as mu
from losses import *
from model.unet import UNet
from model.utils import segment_scans
from train_valid_split import get_train_valid_indices
from train_valid_split import load_split_from_json
from utils import *

plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['figure.titlesize'] = 17
plt.rcParams['hist.bins'] = 100
plt.rcParams['image.cmap'] = 'gray'

metrics = {type(func).__name__: func
           for func in
           [
               BCELoss(reduction='mean'),
               NegDiceLoss(),
               # FocalLoss(alpha=0.75, gamma=2, reduction='mean'),
               FocalLoss(gamma=2)
           ]}
loss_name, loss_func = list(metrics.items())[1]


def train_net(
        indices_train, indices_valid, scans_dp, labels_dp,
        net, optimizer, source_slices_per_batch, aug_cnt, device, n_epochs,
        checkpoints_dp='results/model_checkpoints'
):
    if os.path.isdir(checkpoints_dp):
        print(f'checkpoints dir "{checkpoints_dp}" exists. will remove and create new one')
        shutil.rmtree(checkpoints_dp)
    os.makedirs(checkpoints_dp, exist_ok=True)

    n_train = len(indices_train) * (1 + aug_cnt)
    n_valid = len(indices_valid)
    batch_size = source_slices_per_batch * (1 + aug_cnt)

    # early stopping variables
    es_cnt = 0
    es_tolerance = 0.0001
    es_patience = 10

    em = {m: {'train': [], 'valid': []} for m in metrics.keys()}  # epoch metrics
    em['loss_name'] = loss_name

    best_loss_valid = 1e+30
    best_epoch_ix = 0
    best_net_wts = copy.deepcopy(net.state_dict())

    print(const.SEPARATOR)
    print('start of the training')
    print(f'parameters:\n\n'
          f'loss function: {loss_name}\n'
          f'optimizer: {type(optimizer).__name__}\n'
          f'learning rate: {optimizer.defaults["lr"]}\n'
          f'momentum: {optimizer.defaults["momentum"]}\n'
          f'number of epochs: {n_epochs}\n'
          f'device: {device}\n'
          f'source slices per batch: {source_slices_per_batch}\n'
          f'augmentations per source slice: {aug_cnt}\n'
          f'resultant batch size: {batch_size}\n'
          f'actual train samples count: {n_train}\n'
          f'actual valid samples count: {n_valid}\n'
          f'checkpoints dir: {os.path.abspath(checkpoints_dp)}\n'
          )

    # count global time of training
    train_time_start = time.time()

    for cur_epoch in range(1, n_epochs + 1):
        print(f'\n{"=" * 15} epoch {cur_epoch}/{n_epochs} {"=" * 15}')
        epoch_time_start = time.time()

        # ----------- train ----------- #
        net.train()
        train_gen = get_scans_and_masks_batches(
            indices_train, scans_dp, labels_dp, source_slices_per_batch, aug_cnt, to_shuffle=True)

        for m_name, m_dict in em.items():
            if isinstance(m_dict, dict):
                m_dict['train'].append(0)

        with tqdm.tqdm(total=n_train, desc=f'epoch {cur_epoch}. train',
                       unit='scan', leave=True) as pbar_t:
            for ix, (scans, labels, scans_ix) in enumerate(train_gen, start=1):
                x = torch.tensor(scans, dtype=torch.float, device=device).unsqueeze(1)
                y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(1)

                out = net(x)

                min, max = torch.min(out).item(), torch.max(out).item()
                if min < 0 or max > 1:
                    tqdm.tqdm.write(f'\n*** WARN ***\nbatch scans: {scans_ix}\nmin: {min}\tmax: {max}')
                    continue

                loss = metrics[loss_name](out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # multiply to find sum of losses for all the batch items.
                # use len(scans) instead of batch_size because they are not equal for the last batch
                em[loss_name]['train'][-1] += loss.item() * len(scans)

                with torch.no_grad():
                    for m_name, m_func in metrics.items():
                        if loss_name not in m_name:
                            value = m_func(out, y)
                            em[m_name]['train'][-1] += value.item() * len(scans)

                pbar_t.update(x.size()[0])

        for m_name, m_dict in em.items():
            if isinstance(m_dict, dict):
                m_dict['train'][-1] /= n_train
                tqdm.tqdm.write(f'{m_name} train: {m_dict["train"][-1] : .4f}')

        if checkpoints_dp is not None:
            torch.save(
                net.state_dict(),
                os.path.join(checkpoints_dp, f'cp_{loss_name}_epoch_{cur_epoch}.pth')
            )

        # ----------- validation ----------- #
        valid_gen = get_scans_and_masks_batches(indices_valid, scans_dp, labels_dp, None, aug_cnt=0, to_shuffle=False)
        evaluation_res = mu.evaluate_net(net, valid_gen, metrics, n_valid, device, f'epoch {cur_epoch}. valid')

        for m_name, m_dict in em.items():
            if isinstance(m_dict, dict):
                m_dict['valid'].append(evaluation_res[m_name]['mean'])
                tqdm.tqdm.write(f'{m_name} valid: {m_dict["valid"][-1] : .4f}')

        epoch_time = time.time() - epoch_time_start
        tqdm.tqdm.write(f'epoch completed in {epoch_time // 60}m {epoch_time % 60 : .2f}s')

        # ----------- early stopping ----------- #
        if em[loss_name]['valid'][-1] < best_loss_valid - es_tolerance:
            best_loss_valid = em[loss_name]['valid'][-1]
            best_net_wts = copy.deepcopy(net.state_dict())
            best_epoch_ix = cur_epoch
            es_cnt = 0
            tqdm.tqdm.write(f'epoch {cur_epoch}: new best loss valid: {best_loss_valid : .4f}')
        else:
            es_cnt += 1

        if es_cnt >= es_patience:
            tqdm.tqdm.write(const.SEPARATOR)
            tqdm.tqdm.write(f'Early Stopping! no improvements for {es_patience} epochs for {loss_name} metric')
            break

    # ----------- save metrics ----------- #
    em_dump_fp = f'results/{loss_name}_epoch_metrics.pickle'
    print(const.SEPARATOR)
    print(f'storing epoch metrics dict in "{em_dump_fp}"')
    with open(em_dump_fp, 'wb') as fout:
        pickle.dump(em, fout)

    # ----------- load best model ----------- #
    net.load_state_dict(best_net_wts)
    if checkpoints_dp is not None:
        torch.save(
            net.state_dict(),
            os.path.join(checkpoints_dp, f'cp_{loss_name}_best.pth')
        )

    print(const.SEPARATOR)
    train_time = time.time() - train_time_start
    print(f'training completed in {train_time // 60}m {train_time % 60 : .2f}s')
    print(f'best loss valid: {best_loss_valid : .4f}, best epoch: {best_epoch_ix}')

    return em


def build_total_hd_boxplot():
    for avg in [False, True]:
        hd = []
        for m_name in metrics:
            with open(f'hd_to_plot/{m_name}_hd{"_avg" if avg else ""}_valid.pickle', 'rb') as fin:
                hd_cur = pickle.load(fin)
                hd.append((m_name, hd_cur))
        mu.build_multiple_hd_boxplots([x[1] for x in hd], avg, [x[0] for x in hd])


class TrainPipeline:
    net = None

    def __init__(self, dataset_dp: str, device: str, n_epochs: int):
        self.dataset_dp = dataset_dp
        self.scans_dp = const.DataPaths.get_numpy_scans_dp(dataset_dp)
        self.masks_dp = const.DataPaths.get_numpy_masks_dp(dataset_dp)
        self.device = device
        self.n_epochs = n_epochs

        self.indices_train, self.indices_valid = get_train_valid_indices(self.dataset_dp)

    def create_net(self):
        self.net = UNet(n_channels=1, n_classes=1)
        self.net.to(device=self.device)
        return self.net

    def train(self):
        self.create_net()

        np.random.shuffle(self.indices_train)  # shuffle before training on partial dataset
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # TODO: remove upper bounds
        em = train_net(
            self.indices_train[:], self.indices_valid[:], self.scans_dp, self.masks_dp, self.net, optimizer,
            source_slices_per_batch=4, aug_cnt=1, device=self.device, n_epochs=self.n_epochs)
        plot_learning_curves(em)

    def load_net_from_weights(self, checkpoint_fp):
        """load model parameters from checkpoint .pth file"""
        print(const.SEPARATOR)
        print(f'loading model parameters from "{checkpoint_fp}"')

        self.create_net()

        state_dict = torch.load(checkpoint_fp)
        self.net.load_state_dict(state_dict)

    def evaluate_model(self):
        print(const.SEPARATOR)
        print('evaluate_model()')

        if self.net is None:
            raise ValueError('must call train() or load_model() before evaluating')

        # TODO: remove upper bound
        hd, hd_avg = mu.get_hd_for_valid_slices(
            self.net, self.device, loss_name, self.indices_valid[:], self.scans_dp, self.masks_dp
        )

        hd_list = [x[1] for x in hd]
        mu.build_hd_boxplot(hd_list, False, loss_name)
        mu.visualize_worst_best(self.net, hd, False, self.scans_dp, self.masks_dp, self.device, loss_name)

        hd_avg_list = [x[1] for x in hd_avg]
        mu.build_hd_boxplot(hd_avg_list, True, loss_name)
        mu.visualize_worst_best(self.net, hd_avg, True, self.scans_dp, self.masks_dp, self.device, loss_name)

    def segment_valid_scans(self, checkpoint_fp, segmented_masks_dp):
        print(const.SEPARATOR)
        print('segment_valid_scans()')

        self.load_net_from_weights(checkpoint_fp)
        fns_valid = load_split_from_json(self.dataset_dp)['valid']

        segment_scans(fns_valid, self.net, self.device, self.dataset_dp, segmented_masks_dp)


@click.command()
@click.option('--launch', help='launch location',
              type=click.Choice(['local', 'server']), default='local')
def main(launch):
    print(const.SEPARATOR)
    print('train_pipeline()')

    const.set_launch_type_env_var(launch == 'local')
    data_paths = const.DataPaths()
    processed_dp = data_paths.get_processed_dp(zoom_factor=0.25, mark_as_new=False)

    device = 'cuda:0'
    # device = 'cuda:1'

    os.makedirs('results', exist_ok=True)

    pipeline = TrainPipeline(dataset_dp=processed_dp, device=device, n_epochs=8)

    # TODO: add as option
    to_train = False

    if to_train:
        pipeline.train()
    else:
        checkpoint_fp = f'results/model_checkpoints/cp_NegDiceLoss_best.pth'
        pipeline.load_net_from_weights(checkpoint_fp)

    pipeline.evaluate_model()

    print(const.SEPARATOR)
    print('cuda memory stats:')
    print_cuda_memory_stats(device)


if __name__ == '__main__':
    main()
