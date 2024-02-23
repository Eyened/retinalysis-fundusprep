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

    def __call__(self, im, tg=None):
        bounds = extract_bounds(im / 255)

        if self.contrast_enhance:
            mask = bounds.make_binary_mask(0.01 * bounds.radius)
            mirrored = bounds.background_mirroring(im)
            sigma = 0.05 * bounds.radius
            ce = contrast_enhance(mirrored, mask, sigma)
        else:
            ce = None

        if self.target_prep_fn is not None:
            tg = self.target_prep_fn(tg)

        if self.square_size is not None:
            diameter = self.square_size
            M = bounds.get_cropping_matrix(diameter)
            im = M.warp(im, (diameter, diameter))
            ce = M.warp(ce, (diameter, diameter))
            if tg is not None:
                tg = M.warp(tg, (diameter, diameter))

        return (im, ce), tg, bounds
