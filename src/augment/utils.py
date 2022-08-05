from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2


def shift_image(img: npt.NDArray, shift_pnt: Tuple[int, int]) -> npt.NDArray:
    """
    shifting image, the empty parts are replaced by reflection
    :param img: image
    :param shift_pnt: (shift in x-axis, shift in y-axis)
    :return: transformed image
    """
    M = np.float32([[1, 0, shift_pnt[0]], [0, 1, shift_pnt[1]]])
    res = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return res


def rotate_image(image: npt.NDArray, angle: float, scale: float, rot_pnt: Tuple[int, int]) -> npt.NDArray:
    """
    rotation around a given point + scaling
    :param image: image
    :param angle: angle
    :param scale: scale
    :param rot_pnt: rotation point
    :return: transformed image
    """
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)  # INTER_NEAREST
    return result


def gauss_noise(img: npt.NDArray, var: float = 30) -> npt.NDArray:
    """
    Gaussian Noise
    :param img: input image
    :param var: variance
    :return: image with noise
    """
    row, col, ch = img.shape
    mean = var
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return np.clip(img.astype(np.int32) + gauss, 0, 255).astype('uint8')


def clahe(img: npt.NDArray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (5, 5)) -> npt.NDArray:
    """
    Contrast Limited AHE (CLAHE)
    :param img: input image
    :param clip_limit: clip limit
    :param tile_grid_size: tile grid size
    :return: transformed image
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    transformation = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = transformation.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def change_hsv(img: npt.NDArray, h_shift: int, s_shift: int, v_shift: int) -> npt.NDArray:
    """
    :param img: input image
    :param h_shift:
    :param s_shift:
    :param v_shift:
    :return: image which is shifted in hsv
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:, :, 0] += h_shift
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 255)
    hsv[:, :, 1] += s_shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] += v_shift
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def shift_channels(img: npt.NDArray, b_shift: int, g_shift: int, r_shift: int) -> npt.NDArray:
    """
    :param img: image
    :param b_shift: shift in blue channel
    :param g_shift: shift in green chanel
    :param r_shift: shift in red channel
    :return: image with shifted channels
    """
    img = img.astype(int)
    img[:, :, 0] += b_shift
    img[:, :, 0] = np.clip(img[:, :, 0], 0, 255)
    img[:, :, 1] += g_shift
    img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
    img[:, :, 2] += r_shift
    img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)
    img = img.astype('uint8')
    return img


def blur(img: npt.NDArray, ksize: Tuple[int, int] = (3, 3)) -> npt.NDArray:
    """
    blur image
    :param img: input image
    :param ksize: A tuple representing the blurring kernel size.
    :return: blured image
    """
    return cv2.blur(img, ksize)


def _blend(img1: npt.NDArray, img2: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    belnd two images
    :param img1: image 1
    :param img2: image 2
    :param alpha: share of image one in blend, between 0 and 1
    :return: blending result
    """
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


def _grayscale(img: npt.NDArray) -> npt.NDArray:
    """
    :param img: input image
    :return: grayscale image
    """
    _alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    change image saturation
    :param img: input image
    :param alpha: change factor
    :return: saturated image
    """
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    change image brightness
    :param img: input image
    :param alpha: change factor
    :return: transformed image
    """
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    change image contrast
    :param img:  input image
    :param alpha: change factor
    :return: transformed image
    """
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 3)
    return _blend(img, gs, alpha)


def invert(img: npt.NDArray) -> npt.NDArray:
    """
    :param img: input image
    :return: inverted image
    """
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img
