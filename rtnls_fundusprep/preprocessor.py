import os
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.transformation import Interpolation
from rtnls_fundusprep.utils import open_image


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

    def __call__(self, image, mask=None, keypoints=None, **kwargs):
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
                mask = M.warp(mask, (diameter, diameter), mode=Interpolation.NEAREST)

            if keypoints is not None and len(keypoints) > 0:
                # print(keypoints)
                # Convert list of tuples to numpy array for transformation
                keypoints_array = np.array(keypoints)
                transformed_keypoints = M.apply(keypoints_array)
                # Convert back to list of tuples
                keypoints = [tuple(point) for point in transformed_keypoints]
        else:
            bounds = orig_bounds

        if self.contrast_enhance:
            mask = bounds.mask
            ce = bounds.contrast_enhanced_5
        else:
            ce = None

        item = {"image": image, "metadata": {"bounds": orig_bounds.to_dict()}, **kwargs}
        if mask is not None:
            item["mask"] = mask
        if keypoints is not None:
            item["keypoints"] = keypoints
        if ce is not None:
            item["ce"] = ce

        return item


class FundusItemPreprocessor(FundusPreprocessor):
    def __call__(self, item):
        prep_data = super().__call__(**item)
        bounds = prep_data["bounds"]
        del prep_data["bounds"]
        return {**item, **prep_data}, bounds.to_dict()


def preprocess_one(id, img_path, rgb_path, ce_path, square_size):
    preprocessor = FundusPreprocessor(
        square_size=square_size, contrast_enhance=ce_path is not None
    )

    try:
        image = open_image(img_path)
        prep = preprocessor(image, None)
    except Exception:
        print(f"Error with image {img_path} with id {id}")
        return False, {}

    if rgb_path is not None:
        Image.fromarray((prep["image"]).astype(np.uint8)).save(rgb_path)
    if ce_path is not None:
        Image.fromarray((prep["ce"]).astype(np.uint8)).save(ce_path)
    bounds = prep["metadata"]["bounds"]

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

    items = zip(ids, files, rgb_paths, ce_paths)

    meta = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(preprocess_one)(*item, square_size=square_size) for item in tqdm(items)
    )

    return [
        {"id": id, "success": success, "bounds": bounds}
        for (success, bounds), id in zip(meta, ids)
    ]
