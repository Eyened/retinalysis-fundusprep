from __future__ import annotations

import pytest

from tests.regression_helpers import (
    assert_matches_reference,
    extract_bounds_frame,
    load_reference_frame,
    write_reference_frame,
)


@pytest.mark.reference
def test_cfi_bounds_regression(accept_fundusprep_reference: bool) -> None:
    """Compare extracted CFI bounds against the stored regression reference."""
    current = extract_bounds_frame()
    if accept_fundusprep_reference:
        write_reference_frame(current)

    reference = load_reference_frame()
    assert_matches_reference(current, reference)
