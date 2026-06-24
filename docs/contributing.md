# Contributing

Contributions are welcome — bug reports, fixes, new segmentation backends, and
docs improvements all help. This page mirrors the repository's
[CONTRIBUTING.md](https://github.com/VANvonZHANG/SegSpy/blob/main/CONTRIBUTING.md).

## Development setup

```bash
git clone https://github.com/VANvonZHANG/SegSpy.git
cd SegSpy
pip install -e ".[dev]"
pre-commit install
```

For the SAM backend during development:

```bash
pip install -e ".[sam]"
```

## Tests and linting

```bash
pytest                          # tests + coverage (via addopts)
ruff check src tests            # lint
ruff format --check src tests   # format check
```

Pre-commit runs ruff automatically on commit.

## Pull requests

1. Branch from `main`: `git checkout -b feature/my-feature`.
2. Add tests for your change (test-driven where practical).
3. Ensure `pytest` and `ruff` pass.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description.

## Reporting issues

Please include your Python version, the SegSpy version
(`python -c "import SegSpy; print(SegSpy.__version__)"`), steps to reproduce,
and the full error traceback.
