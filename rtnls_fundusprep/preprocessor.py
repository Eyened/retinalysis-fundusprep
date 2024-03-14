import numpy as np

from rtnls_fundusprep.mask_extraction import extract_bounds


class FundusPreprocessor:
    def __init__(
        self,
        square_size=None,
        target_prep_fn=None,
    ):
        self.square_size = square_size
        self.target_prep_fn = target_prep_fn

    def __call__(self, im, tg=None):
        if im.dtype == np.uint8:
            im = im / 255

        bounds = extract_bounds(im)

        if self.target_prep_fn is not None:
            tg = self.target_prep_fn(tg)
            assert tg.dtype in [np.uint8, bool, float]

        if self.square_size is not None:
            diameter = self.square_size
            M = bounds.get_cropping_matrix(diameter)
            im = M.warp(im, (diameter, diameter))

            if tg is not None:
                tg = M.warp(tg, (diameter, diameter))

        if im is not None:
            im = (255 * im).astype(np.uint8)

        return im, tg, bounds
