# StackOverflow Tag Prediction

Given the Title of a StackOverflow Question, this project returns a ranked Suggestion list of
Tags with their Scores. Two Taggers are trained on historical tagged Questions: a Linear tagger
built on classic machine learning, and a Transformer tagger built on a fine-tuned pretrained
language model. See the [glossary](GLOSSARY.md) for the exact meaning of these terms and the
[spec issue #2](https://github.com/thomas-hinton/stackoverflow-tag-prediction/issues/2) for the
full scope.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if it is missing.

```bash
uv sync            # create .venv with the package and the dev tooling
uv run so-tagger --help
uv run so-tagger --version
```

Use `uv sync --frozen` to install exactly what `uv.lock` pins (this is what CI does).

Settings shared by the pipeline, such as the seed, the Supported tag frequency cut-off, the
split ratios and the Linear tagger's hyperparameters, live in the versioned
[`config.toml`](config.toml).

## Quickstart (CLI)

From a fresh clone, after `uv sync`:

```bash
uv run so-tagger prepare                  # raw CSV -> data/prepared/ (Questions with Supported tags)
uv run so-tagger split                    # -> data/splits/{train,validation,test}.jsonl
uv run so-tagger train --tagger linear    # -> artifacts/linear/ (model + metadata.json)
uv run so-tagger suggest "How do I parse JSON in Python?"
uv run so-tagger evaluate                 # -> metrics/linear-test.json
```

```text
Tag     Score
python  1.000
json    0.990
```

`train` tunes the Score cut-off on the validation split and saves it with the Tagger; `evaluate`
measures the Tagger on the test split and writes a metrics file. The tuning rule and the metrics
file schema are documented in [docs/metrics.md](docs/metrics.md). The Linear tagger's baseline
on the test split:

```text
micro-F1  0.509  (precision 0.527, recall 0.491)
macro-F1  0.463
P@1 0.611  R@1 0.453  P@3 0.306  R@3 0.619  P@5 0.211  R@5 0.694
Suggestions per Question: 1.47
```

Add `--json` to `suggest` for machine-readable output. Every command has `--help`, takes its
input and output locations as options, and reads its settings from `config.toml` (`--config`
selects another file).

The frozen split in `data/splits/` is committed: it is the reference on which both Taggers are
compared. `split` is stratified on Tags ([ADR 0004](docs/adr/0004-iteratively-stratified-frozen-split.md))
and re-running it with the same configuration reproduces the files byte for byte.

## Data exploration

The [exploration notebook](notebooks/01_data_exploration.ipynb) checks the dataset's integrity
and justifies the Supported tag cut-off. Its libraries live in a separate `explore` dependency
group that the runtime install never pulls:

```bash
uv sync --group explore
uv run jupyter lab notebooks/
```

## Running the checks

```bash
uv run ruff check .            # lint (notebooks included)
uv run ruff format --check .   # formatting
uv run mypy                    # strict type checking
uv run pytest                  # tests with coverage
uv run pytest -m "not slow"    # skip heavy tests
```

CI runs the same commands on every pull request to `main` and on every push to `main`.

## Contributing

- **Branches** are named `<type>/<issue-number>-<slug>`, e.g. `feat/7-linear-tagger`.
- **Commits** and pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, ...).
- **One pull request per sub-issue**, and its description closes that issue with `Closes #N`.
- **Squash merge** only; head branches are deleted after merge.
- All repository content is written in English.

### Why required approvals are 0

`main` is protected: changes land only through a pull request, the `ci` check must pass on a
branch that is up to date with `main`, and force pushes and deletions are blocked. Required
approvals are set to 0 because the project has a single maintainer and GitHub does not let
authors approve their own pull requests; requiring an approval would block every merge. Review
still happens on every pull request: the author posts a self-review comment before merging.

## Design records

- [Glossary](GLOSSARY.md): the domain language used in code, docs and issues.
- [Score cut-off tuning and the metrics file](docs/metrics.md).
- Architecture decisions:
  - [ADR 0001: DistilBERT fine-tuned on CPU](docs/adr/0001-distilbert-fine-tuned-on-cpu.md)
  - [ADR 0002: GitHub Releases for model artifacts](docs/adr/0002-github-releases-for-model-artifacts.md)
  - [ADR 0003: Supported tag frequency cut-off](docs/adr/0003-supported-tag-frequency-cutoff.md)
  - [ADR 0004: Iteratively stratified frozen split](docs/adr/0004-iteratively-stratified-frozen-split.md)
