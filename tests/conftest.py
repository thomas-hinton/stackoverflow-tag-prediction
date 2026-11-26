from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DATASET = FIXTURES / "stackoverflow_posts_fixture.csv"

# Same shape as the project config.toml, scaled down to the fixture dataset.
FIXTURE_CONFIG = """\
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
score_cutoff = 0.5
"""

FIXTURE_SUPPORTED_TAGS = ("css", "javascript", "python", "sql")


@pytest.fixture
def fixture_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(FIXTURE_CONFIG, encoding="utf-8")
    return path
