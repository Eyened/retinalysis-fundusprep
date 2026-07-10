from __future__ import annotations

import numpy as np
import pytest

from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.utils import get_gray_scale, open_image, spatial_gaussian_sigma
from tests.regression_helpers import SAMPLES_DIR


def _sample_rgb() -> np.ndarray:
    path = next(SAMPLES_DIR.glob("*.png"))
    return open_image(path)


def test_get_gray_scale_accepts_common_formats() -> None:
    rgb = _sample_rgb()
    gray = get_gray_scale(rgb)
    assert gray.ndim == 2
    assert gray.shape == rgb.shape[:2]

    assert get_gray_scale(gray).shape == gray.shape

    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2)
    assert get_gray_scale(rgba).shape == rgb.shape[:2]


def test_spatial_gaussian_sigma_matches_rank() -> None:
    rgb = _sample_rgb()
    gray = get_gray_scale(rgb)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2)

    assert spatial_gaussian_sigma(gray, 1.0) == (1.0, 1.0)
    assert spatial_gaussian_sigma(rgb, 1.0) == (1.0, 1.0, 0.0)
    assert spatial_gaussian_sigma(rgba, 1.0) == (1.0, 1.0, 0.0)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_contrast_enhancement_across_formats(channels: int) -> None:
    rgb = _sample_rgb()
    if channels == 1:
        image = get_gray_scale(rgb)
    elif channels == 3:
        image = rgb
    else:
        image = np.concatenate(
            [rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2
        )

    bounds = get_cfi_bounds(image)
    ce = bounds.contrast_enhanced_5
    assert ce.dtype == np.uint8
    if channels == 1:
        assert ce.ndim == 2
    else:
        assert ce.shape == image.shape
