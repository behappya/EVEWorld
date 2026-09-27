# Contributing to EVEWorld

Contributions are welcome — bug reports, documentation fixes, new evaluation
protocols and code changes. The repository accompanies a research paper and the
default branch tracks the released artefact, so open an issue before a large
change.

## Ground rules

- Never commit credentials. API keys, tokens and judge endpoints belong in
  environment variables; [`docs/installation.md`](docs/installation.md) lists
  the ones the code reads.
- `third_party/` and `outputs/` are never committed, and neither are datasets,
  checkpoints or generated videos. Those trees are gitignored, and anything
  large or redistributable belongs outside the repository.
- A reported number should be reproducible from the commands in
  [`docs/reproduction.md`](docs/reproduction.md). When you report a benchmark
  discrepancy, state the protocol you used: detector settings, seeds, guidance
  weight and checkpoint step.

## Development setup

The library is a `src/` package; for code work the evaluation environment is
enough:

```bash
conda env create -f envs/evaluation.yaml
conda activate eveworld-eval
pip install -e ".[dev]"
pre-commit install
```

[`docs/installation.md`](docs/installation.md) describes the other two
environments (`gigaworld`, `flowwam`), the extras that pull in the training and
evaluation stacks, and the environment variables. The `dev` extra adds pytest
and pre-commit, and any of the three environments can host development, because
`pip install -e .` links the checkout from `src/` instead of copying it.

## Style and hooks

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) wires the formatting
and linting hooks into `git commit`: whitespace, end-of-file and YAML
checks, isort, black and flake8, with the settings from
[`pyproject.toml`](pyproject.toml) — isort and black at 150 columns, and
flake8 at the same limit ignoring `E203`, `W503` and `E501`. Run the hooks
over the tree before submitting:

```bash
pre-commit run --all-files
```

## Tests

```bash
pytest -q
```

The suite under `tests/` covers the corruption builders, the IGR weight map,
the TIA transport adapter and the MLR occlusion and persistence rules. The
pre-commit configuration also exposes the suite as a manual-stage hook:

```bash
pre-commit run --hook-stage manual pytest
```

Add a test alongside new behaviour, in the style of the surrounding file;
`pyproject.toml` points pytest at `tests/`.

## Commits and pull requests

Use conventional-commit prefixes — `feat:`, `fix:`, `docs:`, `test:`,
`chore:`, `refactor:` — with an imperative subject line, and keep a change
focused on one concern. Reference the issue or the paper artefact (table or
figure) that a change affects.

## Reporting issues

Include the command you ran, the environment (`pip list`, GPU and CUDA
version) and the full error trace. For benchmark discrepancies, state the
evaluation protocol you used.
