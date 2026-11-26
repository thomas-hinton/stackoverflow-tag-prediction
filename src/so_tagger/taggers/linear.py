"""The Linear tagger: TF-IDF features and one-vs-rest logistic regression."""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from so_tagger.config import ProjectConfig
from so_tagger.domain import Question
from so_tagger.taggers.base import ScoreMatrix, Tagger, TaggerMetadata

MODEL_FILE = "model.joblib"


class LinearTagger(Tagger):
    """Word and character n-gram TF-IDF, then one logistic regression per Supported tag.

    A Score is the predicted probability of its Tag's logistic regression.
    """

    kind = "linear"

    def __init__(self, metadata: TaggerMetadata, pipeline: Any) -> None:
        super().__init__(metadata)
        self._pipeline = pipeline

    @classmethod
    def train(
        cls,
        train: Sequence[Question],
        validation: Sequence[Question],
        config: ProjectConfig,
    ) -> Self:
        settings = config.linear_tagger
        metadata = TaggerMetadata.at_training(
            cls.kind,
            train,
            settings.score_cutoff,
            config,
            training_config={**asdict(settings), "seed": config.seed},
        )
        tfidf = {"min_df": settings.min_df, "sublinear_tf": settings.sublinear_tf}
        pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "words",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=settings.word_ngram_range,
                                    token_pattern=r"(?u)\b\w+\b",
                                    **tfidf,
                                ),
                            ),
                            (
                                "characters",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=settings.char_ngram_range,
                                    **tfidf,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=settings.C,
                            class_weight=settings.class_weight,
                            max_iter=settings.max_iter,
                            solver="liblinear",
                            random_state=config.seed,
                        )
                    ),
                ),
            ]
        )
        binarizer = MultiLabelBinarizer(classes=list(metadata.supported_tags))
        targets = binarizer.fit_transform([question.tags for question in train])
        pipeline.fit([question.title for question in train], targets)
        return cls(metadata, pipeline)

    def score(self, titles: Sequence[str]) -> ScoreMatrix:
        return np.asarray(self._pipeline.predict_proba(list(titles)), dtype=np.float64)

    def _save_model(self, directory: Path) -> None:
        joblib.dump(self._pipeline, directory / MODEL_FILE)

    @classmethod
    def _load_model(cls, directory: Path, metadata: TaggerMetadata) -> Self:
        # joblib unpickles the file: only load Tagger directories you trust.
        return cls(metadata, joblib.load(directory / MODEL_FILE))
