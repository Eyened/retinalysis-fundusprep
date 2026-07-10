from __future__ import annotations

from pathlib import Path

import numpy as np

from rtnls_fundusprep.mask_extraction import RESOLUTION, is_full_frame
from rtnls_fundusprep.utils import get_gray_scale, rescale

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = REPO_ROOT / "tests" / "full_frame_cases.txt"
NOT_CASES_FILE = REPO_ROOT / "tests" / "not_full_frame_cases.txt"


def is_full_frame_image(image: np.ndarray) -> bool:
    gray = get_gray_scale(image)
    _, scaled = rescale(gray, resolution=RESOLUTION)
    return is_full_frame(scaled)


def _read_case_paths(path: Path) -> list[Path]:
    if not path.is_file():
        return []
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def discover_full_frame_cases(
    *,
    cases_file: Path = CASES_FILE,
    cli_paths: list[str] | None = None,
) -> list[Path]:
    paths = _read_case_paths(cases_file)
    if cli_paths:
        paths.extend(Path(p.strip()) for p in cli_paths if p.strip())
    return paths


def discover_not_full_frame_cases(
    *,
    cases_file: Path = NOT_CASES_FILE,
    cli_paths: list[str] | None = None,
) -> list[Path]:
    paths = _read_case_paths(cases_file)
    if cli_paths:
        paths.extend(Path(p.strip()) for p in cli_paths if p.strip())
    return paths
