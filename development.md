# Development

`rtnls_fundusprep` targets Python `3.8` through `3.15`.

## Local Test Run

```bash
pip install -e ".[test]"
pytest
```

Useful variants:

```bash
pytest -m reference
pytest --accept-fundusprep-reference
```

## Tox

Run the full tox matrix:

```bash
tox
```

Run a single environment:

```bash
tox -e py312
tox -e pkg
```

## Updating the Reference

Refresh the committed `get_cfi_bounds` baseline with:

```bash
pytest --accept-fundusprep-reference
```

This rewrites `tests/reference/cfi_bounds.parquet` from the images in `samples/original`.

The committed `samples/original` and `tests/reference` fixtures are intentionally included so the tests work from a clean checkout and from the source release.
