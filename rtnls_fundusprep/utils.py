import numpy as np
import pydicom
from PIL import Image

from rtnls_fundusprep.transformation import get_affine_transform


def open_image(filename):
    try:
        return np.array(Image.open(filename))
    except:
        return pydicom.read_file(filename, force=True).pixel_array


def get_gray_scale(array):
    assert array.dtype == np.uint8, f"Expected uint8, got {array.dtype}"
    if len(array.shape) == 3:
        return array[:, :, 0]  # red channel
    elif len(array.shape) == 2:
        return array
    else:
        raise ValueError("Unknown image format")


def rescale(image, resolution=1024):
    """
    Rescale image to resolution x resolution
    """
    h, w = image.shape
    s = min(resolution / h, resolution / w)
    out_size = resolution, resolution
    rotate = 0
    scale = s, s
    center = h // 2, w // 2
    init_transform = get_affine_transform(out_size, rotate, scale, center)
    im_scaled = init_transform.warp(image, out_size)
    return init_transform, im_scaled


def to_uint8(image):
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)
