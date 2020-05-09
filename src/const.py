import os

SEPARATOR = f'\n{"=" * 20}'

# changing groups order might change `utils.parse_image_id_from_filepath` function
IMAGE_FP_RE_PATTERN = r'(.*\/?)(id[\d]+)_*([^\/]*?)\.(npy|nii\.gz)'

BODY_THRESH_LOW = -1300
BODY_THRESH_HIGH = 1500

MASK_BINARIZATION_THRESH = -500

ZOOM_FACTOR = 0.25

# ----------- paths relative to project folder ----------- #

RESULTS_DP = 'results'
MODEL_CHECKPOINTS_DP = os.path.join(RESULTS_DP, 'model_checkpoints')
SEGMENTED_DP = os.path.join(RESULTS_DP, 'segmented')

# ----------- data paths from the data root directory ----------- #

ENV_IS_SERVER_LAUNCH = 'IS_SERVER_LAUNCH'


def set_launch_type_env_var(is_local_launch: bool):
    os.environ[ENV_IS_SERVER_LAUNCH] = '0' if is_local_launch else '1'
    print(f'\nset_launch_type_env_var()')
    print(f'{ENV_IS_SERVER_LAUNCH}: {os.environ[ENV_IS_SERVER_LAUNCH]}')


class DataPaths:
    ROOT_DATA_DP_LOCAL = '/media/rtn/storage/datasets/lungs/data'
    ROOT_DATA_DP_SERVER = '/media/data10T_1/datasets/CRDF_5_tmp/model_dataset'

    _is_server_launch = None
    _root_data_dp = None

    def __init__(self):
        env_server_launch = os.environ.get(ENV_IS_SERVER_LAUNCH)
        if env_server_launch is not None and env_server_launch == '1':
            self._is_local_launch = False
        else:
            self._is_local_launch = True

        self._root_data_dp = DataPaths.ROOT_DATA_DP_LOCAL if self._is_local_launch \
            else DataPaths.ROOT_DATA_DP_SERVER

        self._original_dp = os.path.join(self._root_data_dp, 'original')
        self._original_scans = os.path.join(self._original_dp, 'scans')
        self._original_masks = os.path.join(self._original_dp, 'masks')
        self._train_valid_split_fp = os.path.join(self._root_data_dp, 'train_valid_split.json')

    @property
    def root_data_dp(self):
        return self._root_data_dp

    @property
    def original_dp(self):
        return self._original_dp

    @property
    def scans_dp(self):
        return self._original_scans

    @property
    def masks_dp(self):
        return self._original_masks

    @property
    def train_valid_split_fp(self):
        return self._train_valid_split_fp

    def get_processed_dataset_dp(self, zoom_factor=None, mark_as_new=True):
        """
        get dir path where processed images are to be stored after dataset creation
        :param mark_as_new: whether to add '_new' postfix to avoid occasional overwrite on the the ready dataset
        """
        dirname = f'processed_z{zoom_factor}' if zoom_factor is not None else 'processed_no_zoom'
        if mark_as_new:
            dirname += '_new'
        return os.path.join(self._root_data_dp, dirname)


# ----------- paths for NumpyDataset ----------- #

def get_shapes_fp(dataset_dp):
    return os.path.join(dataset_dp, 'numpy', 'shapes.pickle')


def get_nifti_dp(dataset_dp):
    return os.path.join(dataset_dp, 'nifti')


def get_numpy_masks_dp(dataset_dp):
    return os.path.join(dataset_dp, 'numpy', 'masks')


def get_numpy_scans_dp(dataset_dp):
    return os.path.join(dataset_dp, 'numpy', 'scans')
