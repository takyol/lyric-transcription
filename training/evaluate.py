import re
from typing import Callable

_wer_metric = None
_g2p = None


def _normalize(text: str) -> str:
    """Normalize text: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_phonemes(text: str) -> str:
    """Convert normalized text to a space-joined phoneme string via g2p_en."""
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    phonemes = _g2p(text)
    return " ".join(p for p in phonemes if p.strip())


def _get_wer_metric():
    global _wer_metric
    if _wer_metric is None:
        import evaluate as hf_evaluate
        _wer_metric = hf_evaluate.load("wer")
    return _wer_metric


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """WER after normalisation (lowercase, strip punctuation)."""
    return _get_wer_metric().compute(
        predictions=[_normalize(p) for p in predictions],
        references=[_normalize(r) for r in references],
    )


def compute_per(predictions: list[str], references: list[str]) -> float:
    """Phoneme Error Rate — WER computed on g2p_en phoneme sequences."""
    return _get_wer_metric().compute(
        predictions=[_to_phonemes(_normalize(p)) for p in predictions],
        references=[_to_phonemes(_normalize(r)) for r in references],
    )


def compute_metrics_fn(tokenizer) -> Callable:
    """Return a compute_metrics callable for use with Seq2SeqTrainer.

    Works with predict_with_generate=False: predictions are raw logits (3D),
    so we take argmax for greedy decoding before computing WER/PER.
    """

    def compute_metrics(pred) -> dict[str, float]:
        import numpy as np

        pred_ids = pred.predictions
        # Whisper returns a tuple of outputs when predict_with_generate=False;
        # the first element is the logits
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids

        # When predict_with_generate=False, predictions are logits shape (B, T, V)
        if pred_ids.ndim == 3:
            pred_ids = pred_ids.argmax(-1)

        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": compute_wer(pred_str, label_str),
            "per": compute_per(pred_str, label_str),
        }

    return compute_metrics
