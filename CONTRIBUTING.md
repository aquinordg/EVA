# Contributing to EVA

Thank you for your interest in contributing to EVA!

## Reporting bugs

Open an issue on [GitHub](https://github.com/aquinordga/EVA/issues) and include:

- A **minimal reproducible example** (code + data shape + expected vs. actual output)
- The full traceback
- Python version (`python --version`), EVA version (`python -c "import eva; print(eva.__version__)"`), and OS

## Requesting features

Open an issue describing the use case and why the feature would be broadly useful.
PRs implementing features are welcome once the issue has been discussed.

## Setting up a development environment

```bash
git clone https://github.com/aquinordga/EVA
cd eva
pip install -e ".[test]"
```

## Running the test suite

```bash
pytest tests/ -v
```

All tests must pass before opening a pull request.

## Code style

EVA uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Run both checks before submitting:

```bash
ruff check eva/
ruff format --check eva/
```

Configuration is in `pyproject.toml` (`[tool.ruff]`).

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Write or update tests covering the changed behaviour.
3. Ensure `pytest tests/` and `ruff check eva/` both pass.
4. Open a pull request with a clear description of *what* changed and *why*.

Amplitude values throughout the codebase are in **SI volts** (as returned by
MNE), not microvolts. Please follow this convention in new code and tests.
