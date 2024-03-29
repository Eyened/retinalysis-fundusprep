import os
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from rtnls_utils.data_loading import load_image
from tqdm import tqdm

from rtnls_fundusprep import FundusPreprocessor


def preprocess_one(img_path, rgb_path, ce_path, square_size):
    preprocessor = FundusPreprocessor(
        square_size=square_size, contrast_enhance=ce_path is not None
    )

    try:
        image = load_image(img_path, np.float32)
        prep = preprocessor(image, None)
    except Exception:
        print(f"Error with image {img_path}")
        return False, {}

    if rgb_path is not None:
        Image.fromarray((prep["image"] * 255).astype(np.uint8)).save(rgb_path)
    if ce_path is not None:
        Image.fromarray((prep["ce"] * 255).astype(np.uint8)).save(ce_path)
    bounds = prep["bounds"]
    bounds = {
        "h": bounds.h,
        "w": bounds.w,
        "cy": bounds.cy,
        "cx": bounds.cx,
        "radius": bounds.radius,
        "min_x": bounds.min_x,
        "min_y": bounds.min_y,
        "max_x": bounds.max_x,
        "max_y": bounds.max_y,
    }
    return True, bounds


def preprocess_for_inference(
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
        {"id": id, "success": success, **bounds}
        for (success, bounds), id in zip(meta, ids)
    ]
