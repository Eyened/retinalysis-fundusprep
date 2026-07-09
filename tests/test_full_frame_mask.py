from __future__ import annotations

from pathlib import Path

import pytest

from rtnls_fundusprep.utils import open_image
from tests.full_frame_helpers import (
    discover_full_frame_cases,
    discover_not_full_frame_cases,
    is_full_frame_image,
)


def pytest_generate_tests(metafunc):
    if "full_frame_image_path" in metafunc.fixturenames:
        request = metafunc.config
        cli_paths = request.getoption("--full-frame-images", default="").split(",")
        cases = discover_full_frame_cases(cli_paths=cli_paths)
        if not cases:
            metafunc.parametrize(
                "full_frame_image_path",
                [None],
                ids=["no_cases"],
                marks=pytest.mark.skip(reason="No paths in tests/full_frame_cases.txt"),
            )
        else:
            metafunc.parametrize(
                "full_frame_image_path",
                cases,
                ids=[p.name for p in cases],
            )

    if "not_full_frame_image_path" in metafunc.fixturenames:
        request = metafunc.config
        cli_paths = request.getoption("--not-full-frame-images", default="").split(",")
        cases = discover_not_full_frame_cases(cli_paths=cli_paths)
        if not cases:
            metafunc.parametrize(
                "not_full_frame_image_path",
                [None],
                ids=["no_cases"],
                marks=pytest.mark.skip(
                    reason="No paths in tests/not_full_frame_cases.txt"
                ),
            )
        else:
            metafunc.parametrize(
                "not_full_frame_image_path",
                cases,
                ids=[p.name for p in cases],
            )


@pytest.mark.full_frame
def test_full_frame_mask(full_frame_image_path: Path | None) -> None:
    assert full_frame_image_path is not None
    if not full_frame_image_path.exists():
        pytest.skip(f"Image not available on this machine: {full_frame_image_path}")

    image = open_image(full_frame_image_path)
    assert is_full_frame_image(image)


@pytest.mark.not_full_frame
def test_not_full_frame_mask(not_full_frame_image_path: Path | None) -> None:
    assert not_full_frame_image_path is not None
    if not not_full_frame_image_path.exists():
        pytest.skip(f"Image not available on this machine: {not_full_frame_image_path}")

    image = open_image(not_full_frame_image_path)
    assert not is_full_frame_image(image)
