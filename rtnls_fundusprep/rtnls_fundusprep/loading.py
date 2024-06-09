from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_filter

from rtnls_fundusprep.colors import (
    contrast_enhance,
    rgb_to_luminance,
    to_uint8,
    vessel_enhance,
)
from rtnls_fundusprep.mask_extraction import extract_bounds


def load_image(path: Union[str, Path], normalize=True):
    """

    Args:
        path (_type_): file location
        normalize (bool, optional): if True, scale [0-255] to [0-1]. Defaults to True.

    Returns:
        np.array: 2D or 3D numpy array
    """
    if path.endswith(".png"):
        array = np.array(Image.open(path))
    else:
        ds = pydicom.read_file(path)
        array = ds.pixel_array

    if normalize:
        return array / 255
    else:
        return array


def preprocess_ce(image, target_diameter=1024, patch_size=(1024, 1024)):
    """

    Args:
        image :

    Returns:

    """
    bounds = extract_bounds(image)
    mask = bounds.make_binary_mask()
    affine = bounds.get_cropping_matrix(target_diameter, patch_size)

    sigma = 0.05 * bounds.radius
    mirrored = bounds.background_mirroring()
    ce = contrast_enhance(mirrored, mask, sigma)

    return bounds, affine, *[affine.warp(im, patch_size) for im in (image, ce, mask)]


def preprocess_vessels(
    image, target_diameter=1024, patch_size=(1024, 1024), sigmas=[9], gamma=0.8
):
    """

    Args:
        image :

    Returns:

    """

    bounds = extract_bounds(image)
    mask = bounds.make_binary_mask(mask_shrink_pixels=0.01 * bounds.radius)
    affine = bounds.get_cropping_matrix(target_diameter, patch_size)

    cropped = affine.warp(image, patch_size)
    mask_cropped = affine.warp(mask, patch_size)

    if len(image.shape) == 2:
        gray = cropped
    else:
        gray = rgb_to_luminance(cropped)

    gray[~mask_cropped] = 0

    vessel = vessel_enhance(gray, mask_cropped, sigmas=sigmas, gamma=gamma)

    return bounds, affine, cropped, vessel, mask_cropped


def preprocess_vessels_multiple(
    image, target_diameter=512, patch_size=(512, 512), sigmas=range(4, 9), gamma=0.8
):
    f = target_diameter / 1024

    bounds = extract_bounds(image)
    mask = bounds.make_binary_mask(mask_shrink_pixels=0.01 * bounds.radius)
    affine = bounds.get_cropping_matrix(target_diameter, patch_size)

    if len(image.shape) == 2:
        gray = image
    else:
        gray = rgb_to_luminance(image)

    mirrored = bounds.background_mirroring(gray)
    mirrored = cv2.medianBlur(to_uint8(mirrored), 5) / 255

    s = bounds.radius / 20
    blurred = gaussian_filter(mirrored, sigma=(s, s))
    equalized = mirrored - blurred

    # z = (equalized - equalized.mean()) / equalized.std()

    gray = equalized

    gray_cropped = affine.warp(gray, patch_size)
    mask_cropped = affine.warp(mask, patch_size)

    gray_cropped[~mask_cropped] = 0

    vessel_base = [
        vessel_enhance(gray_cropped, mask_cropped, sigmas=[f * sigma], gamma=gamma)
        for sigma in sigmas
    ]
    # a = 0.2
    # vessels = [
    #     a * gray_cropped + (1 - a) * np.clip(0.2 * v / v.std(), 0, 1)
    #     for v in vessel_base
    # ]
    vessels = [np.clip(0.25 * v / v.std(), 0, 1) for v in vessel_base]
    return bounds, affine, mask_cropped, vessels
