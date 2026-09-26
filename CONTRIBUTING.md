# Contributing to EVEWorld

Thank you for your interest in EVEWorld. Contributions of all kinds — bug
reports, documentation improvements, new evaluation protocols, and code
fixes — are welcome.

## Ground rules

- Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).
- This repository accompanies a research paper; the default branch is a
  pinned release. Open an issue before large changes.
- Never commit API keys, access tokens, or cluster-internal endpoints.
  Credentials belong in environment variables (see `docs/ENVIRONMENT.md`).
- Large artifacts (datasets, checkpoints, generated videos) stay out of the
  repository; share them via external storage links where needed.

## Development setup

```bash
conda env create -f environment.yml && conda activate EVEWorld
pip install -e ./giga-models
pip install pre-commit && pre-commit install
```

The pre-commit hooks (isort / black / flake8 / mdformat / docformatter) match
the upstream GigaAI style; run `pre-commit run --all-files` before submitting.

## Reporting issues

Include the command you ran, the environment (`pip list`, GPU/CUDA), and the
full error trace. For benchmark discrepancies, state the exact evaluation
protocol (detector settings, seeds, CFG weight) you used.
