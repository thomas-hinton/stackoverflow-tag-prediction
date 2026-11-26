# Supported tags are Tags carried by at least 20 Questions

A Tag becomes a Supported tag when at least N = 20 distinct Questions carry it. On the current dataset this gives **86 Supported tags** out of 682, keeps **89.0% of Questions** (509 of 4,609 are dropped) and **64.6% of (Question, Tag) pairs**. Questions left with no Supported tag are dropped during data preparation, since they have nothing to learn from. The rule behind N: every Supported tag must expect at least 3 Questions in the 15% test split (0.15 × N ≥ 3), so that its per-Tag score, and so macro-F1, means something; with a 70/15/15 split it also gets about 14 training Questions to be learned from. N = 20 is the smallest cut-off that meets it. Three is an expected count, so the split should be stratified on Tags for the rarest Supported tags to actually reach the test split. The analysis is in `notebooks/01_data_exploration.ipynb`.

The cut-off is configurable as `data.frequency_cutoff` in `config.toml`; 20 is its default, and changing it changes the set of Supported tags of both Taggers.

## Considered Options

- **Keep all 682 Tags**: rejected, because 83 Tags appear in a single Question and 481 of the 682 appear in fewer than 10, so they cannot be learned, and with under 1.5 expected test examples their per-Tag scores are noise that swamps macro-F1.
- **The K most frequent Tags**: rejected, because on a fixed dataset any K is just some N in disguise (the top 86 is N = 20, the top 33 is N = 50), and K says nothing about how many examples the rarest Tag has; a frequency floor states the learnability requirement directly and keeps its meaning if the dataset grows.
- **N = 10** (201 Tags, 95.7% of Questions): rejected, because the rarest Tags would get about 1.5 test examples, too few to evaluate.
- **N = 50** (33 Tags, 78.6% of Questions): rejected, because it drops over a fifth of Questions and half of the pairs for little gain in learnability over N = 20.
