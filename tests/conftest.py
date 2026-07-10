from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom options for reference test maintenance."""
    parser.addoption(
        "--accept-fundusprep-reference",
        action="store_true",
        default=False,
        help="Refresh stored Fundusprep regression references.",
    )
    parser.addoption(
        "--full-frame-images",
        action="store",
        default="",
        help=(
            "Comma-separated image paths added to tests/full_frame_cases.txt for "
            "full-frame mask tests."
        ),
    )
    parser.addoption(
        "--not-full-frame-images",
        action="store",
        default="",
        help=(
            "Comma-separated image paths added to tests/not_full_frame_cases.txt "
            "for negative full-frame mask tests."
        ),
    )


@pytest.fixture
def accept_fundusprep_reference(pytestconfig: pytest.Config) -> bool:
    """Expose whether the caller wants to refresh references."""
    return bool(pytestconfig.getoption("--accept-fundusprep-reference"))
