import os

SEPARATOR = f'\n{"=" * 20}'

# changing groups order might change `utils.parse_image_id_from_filepath` function
IMAGE_FP_RE_PATTERN = r'(.*\/?)(id[\d]+)_*([^\/]*?)\.(npy|nii\.gz)'

BODY_THRESH_LOW = -1300
BODY_THRESH_HIGH = 1500

MASK_BINARIZATION_THRESH = -500

ZOOM_FACTOR = 0.25

# ----------- paths relative to project folder ----------- #

RESULTS_DN = 'results'
MODEL_CHECKPOINTS_DN = 'model_checkpoints'
SEGMENTED_DN = 'segmented'

DOCS_DN = 'docs'
TRAIN_VALID_SPLIT_FP = os.path.join(DOCS_DN, 'train_valid_split.yaml')
HARD_CASES_MAPPING = os.path.join(DOCS_DN, 'hard_cases_mapping.csv')

# ----------- data paths from the data root directory ----------- #

ENV_IS_SERVER_LAUNCH = 'IS_SERVER_LAUNCH'
# DEFAULT_NUMPY_DATASET_DN = 'processed_z0.25'
DEFAULT_NUMPY_DATASET_DN = 'processed_no_zoom'


def set_launch_type_env_var(is_local_launch: bool):
    os.environ[ENV_IS_SERVER_LAUNCH] = '0' if is_local_launch else '1'
    print(f'\nset_launch_type_env_var()')
    print(f'{ENV_IS_SERVER_LAUNCH}: {os.environ[ENV_IS_SERVER_LAUNCH]}')


class DataPaths:
    ROOT_DATA_DP_LOCAL = '/media/rtn/storage/datasets/lungs/data'
    ROOT_DATA_DP_SERVER = '/media/data10T_1/datasets/CRDF_5_tmp/model_dataset'

    _is_server_launch = None
    _root_dp = None

    def __init__(self):
        env_server_launch = os.environ.get(ENV_IS_SERVER_LAUNCH)
        if env_server_launch is not None and env_server_launch == '1':
            self._is_local_launch = False
        else:
            self._is_local_launch = True

        self._root_dp = DataPaths.ROOT_DATA_DP_LOCAL if self._is_local_launch \
            else DataPaths.ROOT_DATA_DP_SERVER

        self._original_dp = os.path.join(self._root_dp, 'original')
        self._original_scans = os.path.join(self._original_dp, 'scans')
        self._original_masks = os.path.join(self._original_dp, 'masks')
        self._default_numpy_dataset_dp = os.path.join(self._root_dp, DEFAULT_NUMPY_DATASET_DN)

    @property
    def root_dp(self):
        return self._root_dp

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
    def default_numpy_dataset_dp(self):
        return self._default_numpy_dataset_dp

    def get_numpy_data_root_dp(self, zoom_factor=None):
        """
        get dir path where processed images are to be stored after dataset creation
        :param mark_as_new: whether to add '_new' postfix to avoid occasional overwrite on the the ready dataset
        """
        dirname = f'processed_z{zoom_factor}' \
            if zoom_factor is not None and zoom_factor != 1 \
            else 'processed_no_zoom'
        return os.path.join(self._root_dp, dirname)


# ----------- paths for NumpyDataset ----------- #

class NumpyDataPaths:
    def __init__(self, root_dp):
        self._root_dp = root_dp
        self._scans_dp = os.path.join(self._root_dp, 'numpy', 'scans')
        self._masks_dp = os.path.join(self._root_dp, 'numpy', 'masks')
        self._shapes_fp = os.path.join(self._root_dp, 'numpy', 'shapes.pickle')
        self._nifti_dp = os.path.join(self._root_dp, 'nifti')

    @property
    def root_dp(self):
        return self._root_dp

    @property
    def scans_dp(self):
        return self._scans_dp

    @property
    def masks_dp(self):
        return self._masks_dp

    @property
    def shapes_fp(self):
        return self._shapes_fp

    @property
    def nifti_dp(self):
        return self._nifti_dp
