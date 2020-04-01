import os

SEPARATOR = f'\n{"=" * 20}'

_ROOT_DATA_DP_LOCAL = '/media/rtn/storage/datasets/lungs/dataset'
_ROOT_DATA_DP_SERVER = '/media/data10T_1/datasets/CRDF_5_tmp/dataset'

NII_GZ_FP_RE_PATTERN = r'.*(id[\d]+).*\.nii\.gz'

BODY_THRESH_LOW = -1300
BODY_THRESH_HIGH = 1500

MASK_BINARIZATION_THRESH = -500

BODY_MIN_PIXELS_THRESH = 600
MASK_MIN_PIXELS_THRESH = 600

ZOOM_FACTOR = 0.25

ENV_IS_SERVER_LAUNCH = 'IS_SERVER_LAUNCH'


def set_launch_type_env_var(is_local_launch: bool):
    os.environ[ENV_IS_SERVER_LAUNCH] = '0' if is_local_launch else '1'
    print(SEPARATOR)
    print(f'set_launch_type_env_var(): {ENV_IS_SERVER_LAUNCH}: {os.environ[ENV_IS_SERVER_LAUNCH]}')


class DataPaths:
    _is_server_launch = None
    _root_data_dp = None

    def __init__(self):
        env_server_launch = os.environ.get(ENV_IS_SERVER_LAUNCH)
        if env_server_launch is not None and env_server_launch == '1':
            self._is_local_launch = False
        else:
            self._is_local_launch = True

        self._root_data_dp = _ROOT_DATA_DP_LOCAL if self._is_local_launch else _ROOT_DATA_DP_SERVER

    @property
    def root_data_dp(self):
        return self._root_data_dp

    @property
    def scans_dp(self):
        return os.path.join(self._root_data_dp, 'original', 'scans')

    @property
    def masks_dp(self):
        return os.path.join(self._root_data_dp, 'original', 'masks')

    @property
    def masks_bin_dp(self):
        return os.path.join(self._root_data_dp, 'original', 'masks_bin')

    @property
    def masks_raw_dp(self):
        return os.path.join(self._root_data_dp, 'original', 'masks_raw')

    def get_processed_dp(self, zoom_factor=None, mark_as_new=True):
        """
        get dir path where processed images are to be stored after dataset creation
        :param mark_as_new: whether to add '_new' postfix to avoid occasional overwrite on the the ready dataset
        """
        dirname = f'processed_z{zoom_factor}' if zoom_factor is not None else 'processed_no_zoom'
        if mark_as_new:
            dirname += '_new'
        return os.path.join(self._root_data_dp, dirname)

    @staticmethod
    def get_numpy_scans_dp(processed_dp):
        return os.path.join(processed_dp, 'numpy', 'scans')

    @staticmethod
    def get_numpy_masks_dp(processed_dp):
        return os.path.join(processed_dp, 'numpy', 'masks')

    @staticmethod
    def get_nifti_dp(processed_dp):
        return os.path.join(processed_dp, 'nifti')

    @staticmethod
    def get_train_valid_split_fp(processed_dp, is_random_split=False):
        split_fn = 'train_valid_split.json' if not is_random_split else 'train_valid_split_random.json'
        return os.path.join(processed_dp, split_fn)

    @staticmethod
    def get_images_z_dimensions_fp(processed_dp):
        return os.path.join(processed_dp, 'images_z.pickle')
