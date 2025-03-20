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
        contrast_enhance=False
    ):
        '''
        params:
            square_size: int, size of the square to crop the image to
            contrast_enhance: bool or int, whether to apply contrast enhancement
        '''
        self.square_size = square_size
        self.contrast_enhance = contrast_enhance

    def __call__(self, image, masks=None, keypoints=None, **kwargs):
        '''
        params:
            image: np.array (h, w, 3, dtype=np.uint8)
            masks: np.array (h, w, dtype=np.uint8 or bool) or list of masks
            keypoints: list of tuples (x, y)
            
        returns:
            dict {
                "image": original image (cropped to square if square_size is not None)
                "metadata": dict {
                    "bounds": dict {
                        "center": np.array (np.float64 (2,))
                        "radius": float
                        "top"?: np.array (np.float64 (2,))
                        "bottom"?: np.array (np.float64 (2,))
                        "left"?: np.array (np.float64 (2,))
                        "right"?: np.array (np.float64 (2,))
                    }
                }
                "masks"?: mask cropped to square if square_size is not None
                "keypoints"?: keypoints mapped to the new coordinates
                "ce"?: contrast enhanced image if contrast_enhance is True or int
            }
        '''
        # apply bounds extraction
        orig_bounds = get_cfi_bounds(image)


        item = {
            "image": image,
            "metadata": {"bounds": orig_bounds.to_dict()},
        
            **kwargs
        }
        
        if self.square_size is not None:
            # resize the image to a square
            M, bounds, image, masks, keypoints = self._apply_crop(
                orig_bounds, image, masks, keypoints)
            item["metadata"]["transform"] = M
        else:
            bounds = orig_bounds
        item["bounds"] = bounds

        
        if masks is not None:
            item["masks"] = masks
        if keypoints is not None:
            item["keypoints"] = keypoints

        if self.contrast_enhance:
            if type(self.contrast_enhance) == int:
                sigma_fraction = self.contrast_enhance / 100
                ce = bounds.make_contrast_enhanced_res256(
                    sigma_fraction=sigma_fraction)
            else:
                # Use default sigma_fraction
                ce = bounds.contrast_enhanced_5
            item["ce"] = ce
        return item

    def _apply_crop(self, orig_bounds, image, masks=None, keypoints=None):
        diameter = self.square_size
        # M is the transformation matrix to crop the image to a square
        # bounds is the new bounds (cropped to a square)
        M, bounds = orig_bounds.crop(diameter)
        
        # Crop the image to a square        
        image = M.warp(image, (diameter, diameter))

        if masks is not None:
            # Check if we have a single mask or multiple masks
            is_single_mask = isinstance(masks, np.ndarray) and masks.ndim == 2

            masks_list = [masks] if is_single_mask else masks

            # Crop the masks to the same square as the image
            warped_masks = [
                M.warp(mask, (diameter, diameter), mode=Interpolation.NEAREST)
                for mask in masks_list
            ]
            masks = warped_masks[0] if is_single_mask else warped_masks

        if keypoints is not None and len(keypoints) > 0:
            # Convert list of tuples to numpy array for transformation
            keypoints_array = np.array(keypoints)
            transformed_keypoints = M.apply(keypoints_array)
            # Convert back to list of tuples
            keypoints = [tuple(point) for point in transformed_keypoints]
        return M, bounds, image, masks, keypoints


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

    items = zip(files, rgb_paths, ce_paths)

    meta = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(preprocess_one)(*item, square_size=square_size) for item in tqdm(items)
    )

    return [
        {"id": id, "success": success, "bounds": bounds}
        for (success, bounds), id in zip(meta, ids)
    ]
