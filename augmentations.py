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
    inten_shifts = np.arange(-16, 16)

    aug_vector = [0, 0, 0, 0, 0, 0]
    #     aug_vector[0] = np.random.choice(angles, 1)[0]
    #     aug_vector[1] = np.random.randint(0, 2)
    #     aug_vector[2] = np.random.randint(0, 2)
    aug_vector[3] = 1  # elastic transform
    #     aug_vector[4] = np.random.choice(intent_shifts, 1)[0]
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
        # _tim = _img[:, :, :-k_size]
        # _tim[_tim > 1] = _tim[_tim > 1] + _value
        # _img[:, :, :-k_size] = _tim
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


# example. written by Eduard Sniazko.
# im_s = np.dstack((tmp_src_img_left, msk_left))
# for a_ in range(number_of_augmented_):
#    cur_aug = gen_aug_vector()
#    print(cur_aug)
#    im_s3 = deepcopy(im_s)
#    im_s2 = augment(im_s3, cur_aug)
#    im1_ = im_s2[:, :, :-1]
#    mask1_ = im_s2[:, :, -1:].astype(np.uint8)


def augment_slice_with_elastic_transform(scan, labels, aug_cnt=5):
    """
    augment 2D slices from scan and labels
    :return: list of augmented images for scan slice and for labels slice
    """
    scans_aug = [scan]
    labels_aug = [labels]

    im_s = np.dstack([scan, labels])
    for i in range(aug_cnt):
        aug_vector = gen_aug_vector()
        im_s_copy = im_s.copy()
        im_s_aug = augment(im_s_copy, aug_vector)
        scan_t = im_s_aug[:, :, 0]
        label_t = im_s_aug[:, :, 1]
        assert sorted(np.unique(label_t)) == [0.0, 1.0]
        label_t = label_t.astype(np.uint8)
        scans_aug.append(scan_t)
        labels_aug.append(label_t)

    return scans_aug, labels_aug
