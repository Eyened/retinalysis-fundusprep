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
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[:, :, 0]  # red channel; works for RGB and RGBA
    raise ValueError(f"Expected 2D or 3D image, got shape {array.shape}")


def spatial_gaussian_sigma(image: np.ndarray, sigma: float) -> tuple[float, ...]:
    """Sigma tuple for scipy.ndimage.gaussian_filter on spatial axes only."""
    if image.ndim < 2:
        raise ValueError(f"Expected at least 2D image, got ndim={image.ndim}")
    if image.ndim == 2:
        return (sigma, sigma)
    return (sigma, sigma, *([0] * (image.ndim - 2)))


def as_cv_color(image: np.ndarray, value: int = 255) -> int | tuple[int, ...]:
    """OpenCV color value matching image channel count."""
    if image.ndim == 2:
        return value
    return tuple([value] * image.shape[2])


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
