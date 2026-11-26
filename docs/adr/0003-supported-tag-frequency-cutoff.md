# Supported tags are Tags carried by at least 20 Questions

A Tag becomes a Supported tag when at least N = 20 distinct Questions carry it. On the current dataset this gives **86 Supported tags** out of 682, keeps **89.0% of Questions** (509 of 4,609 are dropped) and **64.6% of (Question, Tag) pairs**. Questions left with no Supported tag are dropped during data preparation, since they have nothing to learn from. N = 20 is the smallest cut-off at which the rarest Supported tag can expect about 3 Questions in the 15% test split and about 14 in training: enough to be learned from Titles, and enough for its per-Tag score (and so macro-F1) to mean something. The analysis is in `notebooks/01_data_exploration.ipynb`.

The cut-off is configurable as `data.frequency_cutoff` in `config.toml`; 20 is its default, and changing it changes the set of Supported tags of both Taggers.

## Considered Options

- **Keep all 682 Tags**: rejected, because 83 Tags appear in a single Question and most Tags appear in fewer than 10, so they cannot be learned, and with under one expected test example their per-Tag scores are noise that swamps macro-F1.
- **Top-K most frequent Tags**: rejected, because K is arbitrary with respect to learnability; a frequency floor states the requirement directly and keeps its meaning if the dataset grows.
- **N = 10** (201 Tags, 95.7% of Questions): rejected, because the rarest Tags would get about 1.5 test examples, too few to evaluate.
- **N = 50** (33 Tags, 78.6% of Questions): rejected, because it drops over a fifth of Questions and half of the pairs for little gain in learnability over N = 20.
