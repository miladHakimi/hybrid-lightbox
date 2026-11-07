# Contributing

Thank you for helping improve Hybrid Lightbox Designer.

## Development setup

Use Python 3.11–3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pytest
```

## Pull requests

- Keep changes focused and explain the user-visible behavior.
- Add or update an automated test for every feature and bug fix.
- Run the complete test suite before submitting a pull request.
- Do not commit generated models, project archives, model checkpoints, customer files, or third-party artwork.
- Document behavior or format changes in the README or relevant file under `docs/`.

By contributing, you agree that your contribution is licensed under the MIT License.
