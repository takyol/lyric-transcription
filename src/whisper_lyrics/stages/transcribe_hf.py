import json
from pathlib import Path

from ..exceptions import TranscriptionError

# Words separated by a pause longer than this start a new segment
_SEGMENT_GAP_SECONDS = 1.0
_MAX_SEGMENT_WORDS = 30


def _build_raw_result(word_chunks: list[dict]) -> dict:
    """Convert HF pipeline word chunks into openai-whisper-shaped raw output.

    Words are grouped into segments at pauses longer than _SEGMENT_GAP_SECONDS
    (or when a segment grows past _MAX_SEGMENT_WORDS). The HF path exposes no
    per-word confidence, so probability is fixed at 1.0.
    """
    segments: list[dict] = []
    current: list[dict] = []

    def _flush() -> None:
        if not current:
            return
        segments.append({
            "text": " ".join(w["word"].strip() for w in current),
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "words": list(current),
        })
        current.clear()

    prev_end = None
    for chunk in word_chunks:
        start, end = chunk["timestamp"]
        if end is None:  # final word can lack an end timestamp
            end = start
        if prev_end is not None and (
            start - prev_end > _SEGMENT_GAP_SECONDS
            or len(current) >= _MAX_SEGMENT_WORDS
        ):
            _flush()
        current.append({
            "word": chunk["text"],
            "start": start,
            "end": end,
            "probability": 1.0,
        })
        prev_end = end
    _flush()

    return {"language": "en", "segments": segments}


def transcribe_hf(
    vocals_path: Path,
    cache_dir: Path,
    model_name: str,
    adapter_dir: Path,
) -> Path:
    """Transcribe vocals with a HF Whisper model + LoRA adapter.

    Produces the same {stem}.raw.json artifact as the openai-whisper stage,
    so cleanup consumes it unchanged. model_name uses the CLI naming
    ("large-v3") and is mapped to the HF hub id ("openai/whisper-large-v3").
    """
    stem = vocals_path.stem.removesuffix(".vocals")
    output_path = cache_dir / f"{stem}.raw.json"
    if output_path.exists():
        return output_path

    try:
        import librosa
        import torch
        from peft import PeftModel
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            pipeline,
        )

        hf_model_name = f"openai/whisper-{model_name}"
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        base = WhisperForConditionalGeneration.from_pretrained(
            hf_model_name, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        processor = WhisperProcessor.from_pretrained(
            hf_model_name, language="English", task="transcribe"
        )
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device=0 if use_cuda else -1,
            torch_dtype=dtype,
        )
        # Decode audio ourselves so transformers never needs torchcodec
        audio, _ = librosa.load(str(vocals_path), sr=16000, mono=True)
        out = asr({"raw": audio, "sampling_rate": 16000}, return_timestamps="word")
        result = _build_raw_result(out["chunks"])
    except Exception as e:
        raise TranscriptionError(f"LoRA transcription failed: {e}") from e

    output_path.write_text(json.dumps(result, indent=2))
    return output_path
