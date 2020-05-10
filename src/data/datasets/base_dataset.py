from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def _init_slice_info(self):
        raise NotImplementedError

    @property
    def n_images(self):
        raise NotImplementedError
