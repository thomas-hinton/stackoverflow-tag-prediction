# StackOverflow Tag Prediction

Suggests relevant tags for a StackOverflow question, learned from historical tagged questions.

## Language

### Data

**Question**:
A StackOverflow post as represented in the dataset, identified by its post id and described only by its title.
_Avoid_: Post, sample, row

**Title**:
The short English sentence summarising a Question; the only text the models read.
_Avoid_: Body, text, content

**Tag**:
A topic label attached to a Question by its author (e.g. `python`, `mysql`); a Question carries one to five Tags.
_Avoid_: Label, category, class

**Supported tag**:
A Tag frequent enough in the dataset to be learned and suggested; Tags below the frequency cut-off are never suggested.
_Avoid_: Vocabulary, known tag, target

### Prediction

**Tagger**:
A trained model that turns a Title into a Suggestion list. The project has exactly two.
_Avoid_: Classifier, predictor, model (alone)

**Linear tagger**:
The Tagger built on classic machine learning; it is the assignment's "machine learning model".
_Avoid_: ML model, baseline

**Transformer tagger**:
The Tagger built on a fine-tuned pretrained language model; it is the assignment's "deep learning model".
_Avoid_: DL model, BERT model

**Suggestion**:
A Supported tag proposed for a Title, together with its Score.
_Avoid_: Prediction, output, result

**Score**:
The estimated probability that a Suggestion applies to the Title.
_Avoid_: Confidence, weight, logit

**Score cut-off**:
The minimum Score a Suggestion needs to enter the Suggestion list; each Tagger has its own.
_Avoid_: Threshold (alone), confidence level

**Suggestion policy**:
The shared rule that turns per-Tag Scores into a Suggestion list: rank by Score, keep those at or above the Score cut-off, never fewer than one, never more than five.
_Avoid_: Post-processing, decoding

**Suggestion list**:
The ranked Suggestions returned for one Title: those whose Score passes the cut-off, never fewer than one and never more than five.
_Avoid_: Top-k, prediction set
