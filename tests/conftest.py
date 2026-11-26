import csv
import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DATASET = FIXTURES / "stackoverflow_posts_fixture.csv"

# Same shape as the project config.toml, scaled down to the fixture dataset. The Transformer
# tagger fine-tunes a tiny checkpoint built by the `tiny_checkpoint` fixture, so that it trains
# in seconds on CPU without any download.
FIXTURE_CONFIG_TEMPLATE = """\
seed = 7

[data]
frequency_cutoff = 5

[split]
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

[linear_tagger]
word_ngram_range = [1, 2]
char_ngram_range = [2, 4]
min_df = 1
sublinear_tf = true
C = 10.0
class_weight = "balanced"
max_iter = 1000

[transformer_tagger]
checkpoint = "{checkpoint}"
learning_rate = 1e-3
batch_size = 8
epochs = 2
warmup_ratio = 0.1
weight_decay = 0.01
patience = 1
max_length_percentile = 99
inference_batch_size = 16
num_threads = 2
"""

FIXTURE_SUPPORTED_TAGS = ("css", "javascript", "python", "sql")


def fixture_config_text(checkpoint: Path) -> str:
    return FIXTURE_CONFIG_TEMPLATE.format(checkpoint=checkpoint.as_posix())


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
