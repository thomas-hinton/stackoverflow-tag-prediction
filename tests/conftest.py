import csv
import re
import tarfile
from pathlib import Path

import pytest

from so_tagger.config import load_config
from so_tagger.dataset import prepare_questions, read_raw_questions
from so_tagger.release import package_tagger
from so_tagger.split import split_questions
from so_tagger.taggers import Tagger, load_tagger, train_tagger

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DATASET = FIXTURES / "stackoverflow_posts_fixture.csv"

# The Transformer tagger fine-tunes a tiny checkpoint built by the `tiny_checkpoint` fixture, so
# that it trains in seconds on CPU without any download.
FIXTURE_CONFIG = FIXTURES / "config.toml"

FIXTURE_SUPPORTED_TAGS = ("css", "javascript", "python", "sql")


def fixture_config_text(checkpoint: Path) -> str:
    template = FIXTURE_CONFIG.read_text(encoding="utf-8")
    return template.replace("{checkpoint}", checkpoint.as_posix())


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised 2-layer DistilBERT and a WordPiece tokenizer, saved locally.

    Its vocabulary holds the words of the fixture Titles plus single characters as a
    fallback, so no network access is needed. Without the `transformer` extra, the returned
    directory is empty and the Transformer tagger's tests skip themselves.
    """
    directory = tmp_path_factory.mktemp("tiny-distilbert")
    try:
        import torch
        from transformers import BertTokenizer, DistilBertConfig, DistilBertModel
    except ModuleNotFoundError:
        return directory

    with FIXTURE_DATASET.open(encoding="utf-8", newline="") as file:
        titles = {row["title"] for row in csv.DictReader(file)}
    words = sorted({word for title in titles for word in re.findall(r"\w+", title.lower())})
    characters = sorted({c for title in titles for c in title.lower() if not c.isspace()})
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        *dict.fromkeys([*characters, *(f"##{c}" for c in characters), *words]),
    ]
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True).save_pretrained(directory)

    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(vocabulary),
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
    )
    DistilBertModel(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def fixture_config_toml(tiny_checkpoint: Path) -> str:
    return fixture_config_text(tiny_checkpoint)


@pytest.fixture
def fixture_config(tmp_path: Path, fixture_config_toml: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(fixture_config_toml, encoding="utf-8")
    return path


def needs_transformer_extra(kind: str) -> None:
    """Skip the current test when it targets the Transformer tagger without its extra."""
    if kind == "transformer":
        pytest.importorskip("torch")
        pytest.importorskip("transformers")


def train_fixture_tagger(kind: str, config_toml: str, directory: Path) -> Tagger:
    """A Tagger of ``kind`` trained and tuned on the fixture dataset."""
    config_path = directory / "config.toml"
    config_path.write_text(config_toml, encoding="utf-8")
    config = load_config(config_path)
    dataset = prepare_questions(read_raw_questions(FIXTURE_DATASET), config.data.frequency_cutoff)
    dataset_split = split_questions(
        dataset.questions, dataset.supported_tags, config.split, config.seed
    )
    return train_tagger(kind, dataset_split.train, dataset_split.validation, config)


@pytest.fixture(scope="session")
def linear_tagger(tmp_path_factory: pytest.TempPathFactory, fixture_config_toml: str) -> Tagger:
    """A small Linear tagger trained and tuned on the fixture dataset, shared by the session."""
    return train_fixture_tagger(
        "linear", fixture_config_toml, tmp_path_factory.mktemp("linear-config")
    )


@pytest.fixture(scope="session")
def transformer_tagger(
    tmp_path_factory: pytest.TempPathFactory, fixture_config_toml: str
) -> Tagger:
    """A Transformer tagger fine-tuned from the tiny checkpoint, shared by the session.

    Skips the test without the `transformer` extra.
    """
    needs_transformer_extra("transformer")
    return train_fixture_tagger(
        "transformer", fixture_config_toml, tmp_path_factory.mktemp("transformer-config")
    )


RELEASE_TAG = "v1.2.3"


def extract_archive(archive: Path, directory: Path) -> Path:
    """Extract a Tagger archive the way the Docker image does: files at the directory root."""
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")
    return directory


def release_fixture_tagger(tagger: Tagger, root: Path) -> Tagger:
    """``tagger`` packaged for Release ``RELEASE_TAG``, then extracted and loaded back."""
    tagger.save(root / "trained")
    assets = package_tagger(root / "trained", RELEASE_TAG, root / "dist")
    return load_tagger(extract_archive(assets.archive, root / "extracted"))


@pytest.fixture(scope="session")
def released_linear_tagger(
    tmp_path_factory: pytest.TempPathFactory, linear_tagger: Tagger
) -> Tagger:
    """The fixture Linear tagger packaged for Release ``RELEASE_TAG``, then extracted and loaded."""
    return release_fixture_tagger(linear_tagger, tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="session")
def released_transformer_tagger(
    tmp_path_factory: pytest.TempPathFactory, transformer_tagger: Tagger
) -> Tagger:
    """The fixture Transformer tagger packaged for Release ``RELEASE_TAG``, extracted and loaded."""
    return release_fixture_tagger(transformer_tagger, tmp_path_factory.mktemp("release"))
