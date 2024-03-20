import numpy as np

from rtnls_fundusprep.colors import contrast_enhance
from rtnls_fundusprep.mask_extraction import extract_bounds


class FundusPreprocessor:
    def __init__(
        self,
        square_size=None,
        contrast_enhance=False,
        target_prep_fn=None,
    ):
        self.square_size = square_size
        self.contrast_enhance = contrast_enhance
        self.target_prep_fn = target_prep_fn

    def __call__(self, image, mask=None, **kwargs):
        assert image.dtype == np.float32

        orig_bounds = extract_bounds(image)

        if self.target_prep_fn is not None:
            mask = self.target_prep_fn(mask)
            assert mask.dtype in [np.uint8, bool, float]

        if self.square_size is not None:
            diameter = self.square_size
            M = orig_bounds.get_cropping_matrix(diameter)
            bounds = orig_bounds.warp(M, (diameter, diameter))
            image = M.warp(image, (diameter, diameter))

            if mask is not None:
                mask = M.warp(mask, (diameter, diameter))
        else:
            bounds = orig_bounds

        if self.contrast_enhance:
            mask = bounds.make_binary_mask(0.01 * bounds.radius)
            mirrored = bounds.background_mirroring(image)
            sigma = 0.05 * bounds.radius
            ce = contrast_enhance(mirrored, mask, sigma)
        else:
            ce = None

        item = {"image": image, "bounds": orig_bounds, **kwargs}
        if mask is not None:
            item["mask"] = mask
        if ce is not None:
            item["ce"] = ce

        return item
