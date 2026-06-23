# Contributing to SegSpy

Thank you for your interest in contributing! This document outlines the process and guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/VANvonZHANG/SegSpy.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

The optional `[sam]` extra (`pip install -e ".[sam]"`) adds `segment-anything`
and `torch` for the SAM backend. The package imports and the post-filter tests
work without it; full `SAMSegmenter.segment` tests are skipped when the extra is
absent.

## Running Tests

```bash
pytest
```

Coverage is enabled by default via `addopts`:

```bash
pytest --cov=SegSpy --cov-report=term-missing
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and import
sorting. Run before committing:

```bash
ruff check src tests
ruff format src tests
```

Pre-commit hooks are configured:

```bash
pre-commit install
pre-commit run --all-files
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests (test-driven where practical)
3. Ensure all tests pass: `pytest`
4. Ensure linting passes: `ruff check src tests && ruff format --check src tests`
5. Update `CHANGELOG.md` under `[Unreleased]`
6. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:

- Python version
- SegSpy version (`python -c "import SegSpy; print(SegSpy.__version__)"`)
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback if applicable
