from __future__ import annotations

import numpy as np
from PIL import Image

from rtnls_fundusprep.transformation import get_affine_transform


def _read_dicom_pixel_array(filename):
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Reading DICOM images requires the optional dependency 'pydicom'. "
            "Install pydicom, or use PNG/JPEG/TIFF image inputs."
        ) from exc

    return pydicom.dcmread(filename, force=True).pixel_array


def open_image(filename):
    try:
        return np.array(Image.open(filename))
    except Exception:
        return _read_dicom_pixel_array(filename)


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
    h, w = image.shape[:2]
    in_size = h, w
    s = min(resolution / h, resolution / w)
    rotate = 0
    scale = s, s
    center = h // 2, w // 2
    init_transform = get_affine_transform(in_size, resolution, rotate, scale, center)
    im_scaled = init_transform.warp(image)
    return init_transform, im_scaled


def to_uint8(image):
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)
