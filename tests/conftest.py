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


@pytest.fixture
def accept_fundusprep_reference(pytestconfig: pytest.Config) -> bool:
    """Expose whether the caller wants to refresh references."""
    return bool(pytestconfig.getoption("--accept-fundusprep-reference"))
