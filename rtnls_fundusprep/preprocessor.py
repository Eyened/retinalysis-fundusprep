import os
from typing import List
import cv2
from joblib import Parallel, delayed
import numpy as np
from PIL import Image
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.utils import open_image
from tqdm import tqdm

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
        # assert image.dtype == np.float32

        orig_bounds = get_cfi_bounds(image)

        if self.target_prep_fn is not None:
            mask = self.target_prep_fn(mask)
            assert mask.dtype in [np.uint8, bool, float]

        if self.square_size is not None:
            diameter = self.square_size
            M, bounds = orig_bounds.crop(diameter)
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
            mask = bounds.mask
            ce = bounds.contrast_enhanced_5
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


def preprocess_one(img_path, rgb_path, ce_path, square_size):
    preprocessor = FundusPreprocessor(
        square_size=square_size, contrast_enhance=ce_path is not None
    )

    try:
        image = open_image(img_path)
        prep = preprocessor(image, None)
    except Exception:
        print(f"Error with image {img_path}")
        return False, {}

    if rgb_path is not None:
        Image.fromarray((prep["image"] * 255).astype(np.uint8)).save(rgb_path)
    if ce_path is not None:
        Image.fromarray((prep["ce"] * 255).astype(np.uint8)).save(ce_path)
    bounds = prep["bounds"].to_dict()
    
    return True, bounds


def parallel_preprocess(
    files: List,
    ids: List = None,
    square_size=1024,
    ce_path=None,
    rgb_path=None,
    n_jobs=-1,
):
    if ids is not None:
        assert len(files) == len(ids)
    else:
        ids = [Path(f).stem for f in files]
    if ce_path is not None:
        if not os.path.exists(ce_path):
            os.makedirs(ce_path)

        ce_paths = [os.path.join(ce_path, str(id) + ".png") for id in ids]
    else:
        ce_paths = [None for f in files]

    if rgb_path is not None:
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        rgb_paths = [os.path.join(rgb_path, str(id) + ".png") for id in ids]
    else:
        rgb_paths = [None for f in files]

    items = zip(files, rgb_paths, ce_paths)

    meta = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(preprocess_one)(*item, square_size=square_size) for item in tqdm(items)
    )

    return [
        {"id": id, "success": success, "bounds": bounds}
        for (success, bounds), id in zip(meta, ids)
    ]