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

    def get_processed_dir(self, zoom_factor=None):
        dirname = f'processed_z{zoom_factor}' if zoom_factor is not None else 'processed_no_zoom'
        return os.path.join(self._root_data_dp, dirname)
