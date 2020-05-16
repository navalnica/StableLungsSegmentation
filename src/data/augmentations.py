import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    written by Eduard Sniazko.
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    k_size = 1  # number of slice and labels pairs
    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size]
    )
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    im_ = image[:, :, :-k_size].copy()
    roi_ = image[:, :, -k_size:].copy()
    shape_im = im_.shape
    shape_roi = roi_.shape

    # cv2 does not accept numpy's dtypes
    im_min_value = float(np.min(im_))
    roi_min_value = float(np.min(roi_))

    for i in range(im_.shape[2]):
        im_t = im_[:, :, i]
        im_t = cv2.warpAffine(
            im_t, M, shape_size[::-1], flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=im_min_value)
        im_[:, :, i] = im_t[:]

    for i in range(roi_.shape[2]):
        roi_t = roi_[:, :, i]
        roi_t = cv2.warpAffine(
            roi_t, M, shape_size[::-1], flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=roi_min_value)
        roi_[:, :, i] = roi_t[:]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape_im[1]), np.arange(shape_im[0]), np.arange(shape_im[2]))
    indices = np.reshape(y + dy[:, :, :-k_size], (-1, 1)), \
              np.reshape(x + dx[:, :, :-k_size], (-1, 1)), \
              np.reshape(z, (-1, 1))
    im_o = map_coordinates(
        im_, indices, order=0, mode='constant', cval=im_min_value, prefilter=False).reshape(shape_im)

    x, y, z = np.meshgrid(np.arange(shape_roi[1]), np.arange(shape_roi[0]), np.arange(shape_roi[2]))
    indices = np.reshape(y + dy[:, :, -k_size:], (-1, 1)), \
              np.reshape(x + dx[:, :, -k_size:], (-1, 1)), \
              np.reshape(z, (-1, 1))
    roi_o = map_coordinates(
        roi_, indices, order=0, mode='constant', cval=roi_min_value, prefilter=False).reshape(shape_roi)

    image[:, :, :-k_size] = im_o
    image[:, :, -k_size:] = roi_o

    return image


def gen_aug_vector():
    """
    generate vector with augmentation params.
    written by Eduard Sniazko.
    """
    angles = [180 * i / 36 for i in range(-2, 3)]
    v_flips = np.arange(2)
    h_flips = np.arange(2)
    to_deform = np.arange(2)
    clahe_coeffs = np.arange(1, 3)
    inten_shifts = np.arange(-16, 17)

    aug_vector = [0, 0, 0, 0, 0, 0]
    #     aug_vector[0] = np.random.choice(angles, 1)[0]
    #     aug_vector[1] = np.random.randint(0, 2)
    #     aug_vector[2] = np.random.randint(0, 2)
    aug_vector[3] = 1  # elastic transform
    aug_vector[4] = np.random.choice(inten_shifts, 1)[0]
    # aug_vector[5] = np.random.choice(clahe_coeffs, 1)[0]
    return aug_vector


def apply_aug(_img, _index, _value):
    """
    written by Eduard Sniazko.
    """
    k_size = 1

    # apply only elastic transformation
    if _index == 0:  # angles
        # _tim1 = _img[:, :, :-k_size]
        # _tim2 = _img[:, :, -k_size]
        #
        # _tim1 = rotate(_tim1, _value, order=3, reshape=False, mode='constant', cval=np.min(_tim1), prefilter=True)
        # _tim2 = rotate(_tim2, _value, order=0, reshape=False, mode='constant', cval=np.min(_tim2))
        #
        # # _img = rotate(_img, _value, order=0, reshape=False)
        # _img[:, :, :-k_size] = _tim1
        # _img[:, :, -k_size] = _tim2
        return _img
    if _index == 1:  # vertical flip
        # _img = np.flipud(_img)
        return _img
    if _index == 2:  # horizontal flip
        # _img = np.fliplr(_img)
        return _img
    if _index == 3:  # elastic_transform
        if _value == 1:
            # _img = elastic_transform(_img, _img.shape[1] * 0.4, _img.shape[1] * 0.05, _img.shape[1] * 0.05)
            _img = elastic_transform(_img, _img.shape[1] * 0.4, _img.shape[1] * 0.07, _img.shape[1] * 0.07)
        return _img
    if _index == 4:  # intensity shift
        _tim = _img[:, :, :-k_size]
        # _tim[_tim > 1] = _tim[_tim > 1] + _value
        _tim += _value
        _img[:, :, :-k_size] = _tim
        return _img


def augment(_img, aug_vector):
    """
    written by Eduard Sniazko.
    """
    for idx in range(0, 5):
        _img = apply_aug(_img, idx, aug_vector[idx])
    #     roi_ = _img[:, :, -1]
    #     roi_[roi_ == 0] = 1
    return _img


def get_single_augmentation(scan: np.ndarray, mask: np.ndarray):
    """
    Get single augmentation for 2D scan slice and mask slice.
    """
    aug_vector = gen_aug_vector()
    stacked = np.dstack([scan, mask])
    stacked_augmented = augment(stacked, aug_vector)
    scan_new = stacked_augmented[:, :, 0]
    mask_new = stacked_augmented[:, :, 1]

    # use following assert to check if augmentations are performed
    # as intended during debug. no need to perform extra-computations during real training.

    # check that no values besides {0, 1} are present in labels array
    labels_unique_values = np.unique(mask_new)
    assert np.setdiff1d(labels_unique_values, [0, 1]).size == 0, \
        f'unique values in labels array: {labels_unique_values}'

    mask_new = mask_new.astype(np.uint8)

    return scan_new, mask_new


def get_multiple_augmentations(scan: np.ndarray, mask: np.ndarray, aug_cnt: int):
    """
    Augment 2D scan slice and mask slice. Include original slices into return lists.
    :return: list of augmented slices and masks.
    """

    if aug_cnt < 0:
        raise ValueError(f'aug_cnt must be >= 0. passed {aug_cnt}')

    scan_augs = [scan]
    mask_augs = [mask]

    for i in range(aug_cnt):
        scan_cur_aug, mask_cur_aug = get_single_augmentation(scan, mask)
        scan_augs.append(scan_cur_aug)
        mask_augs.append(mask_cur_aug)

    return scan_augs, mask_augs
