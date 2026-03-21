from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.utils import open_image

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples" / "original"
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"
REFERENCE_PATH = REFERENCE_DIR / "cfi_bounds.parquet"
SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm"}
LINE_LOCATIONS = ("top", "bottom", "left", "right")
ABS_TOL = 1e-3
REL_TOL = 1e-5
MAX_FAILURE_LINES = 25


def sample_paths() -> list[Path]:
    """Return the committed sample image paths used by the regression suite."""
    paths = sorted(
        path for path in SAMPLES_DIR.iterdir() if path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES
    )
    if not paths:
        raise AssertionError(f"No sample images found in {SAMPLES_DIR}")
    return paths


def flatten_bounds(sample_id: str, raw_bounds: dict[str, Any]) -> dict[str, Any]:
    """Flatten one bounds payload into a deterministic row."""
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "hw_y": int(raw_bounds["hw"][0]),
        "hw_x": int(raw_bounds["hw"][1]),
        "center_x": float(raw_bounds["center"][0]),
        "center_y": float(raw_bounds["center"][1]),
        "radius": float(raw_bounds["radius"]),
        "min_y": int(raw_bounds["min_y"]),
        "max_y": int(raw_bounds["max_y"]),
        "min_x": int(raw_bounds["min_x"]),
        "max_x": int(raw_bounds["max_x"]),
    }

    lines = raw_bounds["lines"]
    for location in LINE_LOCATIONS:
        line = lines.get(location)
        row[f"{location}_present"] = int(line is not None)
        if line is None:
            row[f"{location}_x0"] = np.nan
            row[f"{location}_y0"] = np.nan
            row[f"{location}_x1"] = np.nan
            row[f"{location}_y1"] = np.nan
            continue

        (p0, p1) = line
        row[f"{location}_x0"] = float(p0[0])
        row[f"{location}_y0"] = float(p0[1])
        row[f"{location}_x1"] = float(p1[0])
        row[f"{location}_y1"] = float(p1[1])

    return row


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows and columns to make comparisons deterministic."""
    return df.sort_index(axis=0).sort_index(axis=1)


def extract_bounds_frame() -> pd.DataFrame:
    """Run `get_cfi_bounds` on the committed sample set."""
    rows: list[dict[str, Any]] = []
    for path in sample_paths():
        image = open_image(path)
        bounds = get_cfi_bounds(image).to_dict_all()
        rows.append(flatten_bounds(path.stem, bounds))

    df = pd.DataFrame(rows).set_index("sample_id")
    return normalize_frame(df)


def load_reference_frame() -> pd.DataFrame:
    """Load the committed bounds baseline."""
    if not REFERENCE_PATH.exists():
        raise AssertionError(f"Missing reference parquet: {REFERENCE_PATH}")
    return normalize_frame(pd.read_parquet(REFERENCE_PATH))


def write_reference_frame(df: pd.DataFrame) -> None:
    """Persist the current bounds frame as the committed reference."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    normalize_frame(df).to_parquet(REFERENCE_PATH)


def assert_matches_reference(current: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Raise an assertion with concise mismatch lines when drift is detected."""
    current = normalize_frame(current)
    reference = normalize_frame(reference)
    failures: list[str] = []

    current_ids = set(current.index)
    reference_ids = set(reference.index)
    for sample_id in sorted(reference_ids - current_ids):
        failures.append(f"<sample> :: {sample_id} :: missing")
    for sample_id in sorted(current_ids - reference_ids):
        failures.append(f"<sample> :: {sample_id} :: unexpected")

    current_columns = set(current.columns)
    reference_columns = set(reference.columns)
    for column in sorted(reference_columns - current_columns):
        failures.append(f"<column> :: {column} :: missing")
    for column in sorted(current_columns - reference_columns):
        failures.append(f"<column> :: {column} :: unexpected")

    shared_ids = sorted(reference_ids & current_ids)
    shared_columns = sorted(reference_columns & current_columns)
    current = current.loc[shared_ids, shared_columns]
    reference = reference.loc[shared_ids, shared_columns]

    for column in shared_columns:
        current_series = current[column]
        reference_series = reference[column]

        if _is_integer_like(reference_series) and _is_integer_like(current_series):
            mismatch_mask = ~(
                (reference_series == current_series)
                | (reference_series.isna() & current_series.isna())
            )
        else:
            mismatch_mask = ~(
                np.isclose(
                    reference_series.to_numpy(dtype=float),
                    current_series.to_numpy(dtype=float),
                    rtol=REL_TOL,
                    atol=ABS_TOL,
                    equal_nan=True,
                )
            )

        if not np.any(mismatch_mask):
            continue

        mismatch_index = current.index[np.asarray(mismatch_mask)]
        for sample_id in mismatch_index:
            failures.append(
                f"{column} :: {sample_id} :: "
                f"ref={_format_value(reference.loc[sample_id, column])} "
                f"cur={_format_value(current.loc[sample_id, column])}"
            )

    if not failures:
        return

    shown_failures = failures[:MAX_FAILURE_LINES]
    remainder = len(failures) - len(shown_failures)
    lines = [f"{len(failures)} Fundusprep regression mismatches", *shown_failures]
    if remainder > 0:
        lines.append(f"... and {remainder} more")
    raise AssertionError("\n".join(lines))


def _is_integer_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    return bool(np.all(np.isclose(values, np.round(values), atol=0.0, rtol=0.0)))


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".6g")
    return str(value)
