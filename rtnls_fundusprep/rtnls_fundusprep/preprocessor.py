import cv2
import numpy as np

from rtnls_fundusprep.colors import contrast_enhance
from rtnls_fundusprep.mask_extraction import extract_bounds


class FundusPreprocessor:
    def __init__(
        self,
        square_size=None,
        contrast_enhance=False,
        target_prep_fn=None,
        dilation_iterations=0,
    ):
        self.square_size = square_size
        self.contrast_enhance = contrast_enhance
        self.target_prep_fn = target_prep_fn
        self.dilation_iterations = dilation_iterations

    def __call__(self, image, mask=None, keypoint=None, **kwargs):
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
                # we dilate the mask to better preserve connectivity
                if self.dilation_iterations > 0:
                    mask = cv2.dilate(
                        mask, np.ones((3, 3)), iterations=self.dilation_iterations
                    )
                mask = M.warp(mask, (diameter, diameter))
            if keypoint is not None:
                keypoint = tuple(M.apply([keypoint])[0])
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
        if keypoint is not None:
            item["keypoint"] = keypoint
        if ce is not None:
            item["ce"] = ce

        return item


class FundusItemPreprocessor(FundusPreprocessor):
    def __call__(self, item):
        prep_data = super().__call__(**item)
        bounds = prep_data["bounds"]
        del prep_data["bounds"]
        return {**item, **prep_data}, bounds.to_dict()
