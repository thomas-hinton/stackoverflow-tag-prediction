"""The Transformer tagger: a pretrained language model fine-tuned with a multi-label head.

DistilBERT by default (ADR 0001), fine-tuned on CPU: one output per Supported tag, a sigmoid
activation and a binary cross-entropy loss. A Score is the sigmoid of its Tag's output.

Needs the ``transformer`` extra (torch, transformers): ``uv sync --extra transformer``.
"""

import copy
import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.utils import logging as transformers_logging

from so_tagger.config import ProjectConfig, TransformerTaggerConfig
from so_tagger.domain import Question
from so_tagger.evaluation import tune_score_cutoff
from so_tagger.taggers.base import ScoreMatrix, Tagger, TaggerMetadata

MODEL_DIR = "model"
"""Subdirectory of the Tagger directory holding the model and tokenizer files."""

logger = logging.getLogger(__name__)

_PROGRESS_EVERY_STEPS = 50
"""How often a training epoch logs its progress, so that a long run can be followed live."""


class TransformerTagger(Tagger):
    """A fine-tuned sequence classification model with one sigmoid output per Supported tag."""

    kind = "transformer"

    def __init__(
        self,
        metadata: TaggerMetadata,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(metadata)
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = int(metadata.training_details["max_length"])
        self._batch_size = int(metadata.training_config["inference_batch_size"])
        # A fast (Rust) tokenizer sets its truncation and padding state on every call and
        # raises "Already borrowed" when two threads call it at once: the API serves requests
        # from a thread pool, so tokenisation is serialised. The forward pass runs in parallel.
        self._tokenizer_lock = threading.Lock()

    @classmethod
    def train(
        cls,
        train: Sequence[Question],
        validation: Sequence[Question],
        config: ProjectConfig,
    ) -> Self:
        settings = config.transformer_tagger
        if settings.num_threads:
            torch.set_num_threads(settings.num_threads)
        set_seed(config.seed)  # Python, NumPy and torch
        metadata = TaggerMetadata.at_training(
            cls.kind, train, validation, config, training_config=asdict(settings)
        )
        tags = metadata.supported_tags
        tokenizer = AutoTokenizer.from_pretrained(settings.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            settings.checkpoint,
            num_labels=len(tags),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(tags)),
            label2id={tag: index for index, tag in enumerate(tags)},
        )
        titles = [question.title for question in train]
        max_length = _max_length(tokenizer, model, titles, settings.max_length_percentile)
        logger.info(
            "Maximum sequence length: %d tokens (covers the %gth percentile of training Titles).",
            max_length,
            settings.max_length_percentile,
        )
        tagger = cls(
            replace(metadata, training_details={"max_length": max_length}), model, tokenizer
        )
        details = tagger._fine_tune(train, validation, settings, config.seed)
        tagger._metadata = replace(
            tagger.metadata, training_details={"max_length": max_length, **details}
        )
        return tagger

    def _fine_tune(
        self,
        train: Sequence[Question],
        validation: Sequence[Question],
        settings: TransformerTaggerConfig,
        seed: int,
    ) -> dict[str, Any]:
        """Fine-tune the model in place and keep the epoch with the best validation micro-F1.

        The validation micro-F1 of an epoch is the one its tuned Score cut-off reaches, as
        computed by the shared tuning code.
        """
        tags = self.supported_tags
        column = {tag: index for index, tag in enumerate(tags)}
        targets = torch.zeros((len(train), len(tags)), dtype=torch.float32)
        for row, question in enumerate(train):
            targets[row, [column[tag] for tag in question.tags]] = 1.0
        titles = [question.title for question in train]
        generator = torch.Generator().manual_seed(seed)
        steps_per_epoch = math.ceil(len(train) / settings.batch_size)
        total_steps = steps_per_epoch * settings.epochs
        optimizer = torch.optim.AdamW(
            _parameter_groups(self._model, settings.weight_decay), lr=settings.learning_rate
        )
        schedule = get_linear_schedule_with_warmup(
            optimizer, round(settings.warmup_ratio * total_steps), total_steps
        )

        logger.info(
            "Fine-tuning for up to %d epochs of %d steps on %d CPU threads.",
            settings.epochs,
            steps_per_epoch,
            torch.get_num_threads(),
        )
        start = time.perf_counter()
        history: list[dict[str, float]] = []
        best_f1, best_epoch, best_state = -1.0, 0, copy.deepcopy(self._model.state_dict())
        for epoch in range(1, settings.epochs + 1):
            self._model.train()
            losses: list[float] = []
            order = torch.randperm(len(train), generator=generator)
            for indices in order.split(settings.batch_size):
                encoded = self._encode([titles[int(i)] for i in indices])
                loss = self._model(**encoded, labels=targets[indices]).loss
                loss.backward()
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                losses.append(float(loss.item()))
                if len(losses) % _PROGRESS_EVERY_STEPS == 0:
                    logger.info(
                        "Epoch %d: step %d/%d, mean loss %.4f, %.0fs elapsed.",
                        epoch,
                        len(losses),
                        steps_per_epoch,
                        np.mean(losses),
                        time.perf_counter() - start,
                    )
            self._model.eval()
            micro_f1 = tune_score_cutoff(self, validation).validation_micro_f1
            entry = {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 6),
                "validation_micro_f1": round(micro_f1, 6),
                "elapsed_seconds": round(time.perf_counter() - start, 1),
            }
            history.append(entry)
            logger.info(
                "Epoch %d/%d: loss %.4f, validation micro-F1 %.4f, %.0fs elapsed.",
                epoch,
                settings.epochs,
                entry["loss"],
                micro_f1,
                entry["elapsed_seconds"],
            )
            if micro_f1 > best_f1:
                best_f1, best_epoch = micro_f1, epoch
                best_state = copy.deepcopy(self._model.state_dict())
            elif epoch - best_epoch >= settings.patience:
                logger.info(
                    "Early stopping: no better validation micro-F1 for %d epoch(s).",
                    epoch - best_epoch,
                )
                break
        self._model.load_state_dict(best_state)
        self._model.eval()
        logger.info("Kept epoch %d (validation micro-F1 %.4f).", best_epoch, best_f1)
        return {"best_epoch": best_epoch, "epochs_run": len(history), "history": history}

    def score(self, titles: Sequence[str]) -> ScoreMatrix:
        """Scores in inference mode. Holds no per-call state on the Tagger, so concurrent calls
        are safe: the model must already be in eval mode, as after training or loading."""
        rows: list[ScoreMatrix] = [np.empty((0, len(self.supported_tags)), dtype=np.float64)]
        with torch.inference_mode():
            for start in range(0, len(titles), self._batch_size):
                encoded = self._encode(list(titles[start : start + self._batch_size]))
                logits = self._model(**encoded).logits
                rows.append(torch.sigmoid(logits).double().numpy())
        return np.concatenate(rows)

    def _encode(self, titles: list[str]) -> dict[str, torch.Tensor]:
        """Tokenise a batch, padded to its own longest Title (dynamic padding)."""
        with self._tokenizer_lock:
            encoded = self._tokenizer(
                titles,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            )
        return {key: value for key, value in encoded.items() if key != "token_type_ids"}

    def _save_model(self, directory: Path) -> None:
        self._model.save_pretrained(directory / MODEL_DIR)
        self._tokenizer.save_pretrained(directory / MODEL_DIR)

    @classmethod
    def _load_model(cls, directory: Path, metadata: TaggerMetadata) -> Self:
        model_dir = directory / MODEL_DIR
        # Keep `suggest` output clean: no "Loading weights" progress bar on every call.
        transformers_logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        # A loaded Tagger only serves: eval mode (no dropout) and no gradient on any weight.
        model.eval()
        model.requires_grad_(False)
        return cls(metadata, model, tokenizer)


def _max_length(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    titles: list[str],
    percentile: float,
) -> int:
    """The token length (special tokens included) covering ``percentile`` of the Titles.

    Capped at the longest sequence the model accepts.
    """
    lengths = [len(ids) for ids in tokenizer(titles, truncation=False)["input_ids"]]
    covering = math.ceil(float(np.percentile(lengths, percentile)))
    return min(covering, int(model.config.max_position_embeddings))


def _parameter_groups(model: PreTrainedModel, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on the weight matrices only, not on biases and LayerNorm weights."""
    decayed, not_decayed = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or "LayerNorm" in name or "layer_norm" in name:
            not_decayed.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": not_decayed, "weight_decay": 0.0},
    ]
