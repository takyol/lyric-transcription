import json
from pathlib import Path
from ..exceptions import CleanupError

_LOW_CONFIDENCE_THRESHOLD = 0.6


def cleanup(raw_json_path: Path, source_file: Path, model_name: str) -> Path:
    output_path = raw_json_path.parent / raw_json_path.name.replace(".raw.json", ".lyrics.json")
    if output_path.exists():
        return output_path

    try:
        raw = json.loads(raw_json_path.read_text())
        result = _build_output(raw, source_file, model_name)
        output_path.write_text(json.dumps(result, indent=2))
    except Exception as e:
        raise CleanupError(f"Cleanup failed: {e}") from e
    return output_path


def _build_output(raw: dict, source_file: Path, model_name: str) -> dict:
    segments = _deduplicate_segments(raw.get("segments", []))
    words = _extract_words(segments)
    duration = float(segments[-1]["end"]) if segments else 0.0

    return {
        "source_file": str(source_file),
        "model": model_name,
        "language": raw.get("language", "en"),
        "duration_seconds": round(duration, 2),
        "words": words,
        "segments": [
            {
                "text": seg["text"].strip(),
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "words": [w["word"].strip() for w in seg.get("words", [])],
            }
            for seg in segments
        ],
    }


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    if not segments:
        return segments
    deduped = [segments[0]]
    for seg in segments[1:]:
        if seg["text"].strip().lower() != deduped[-1]["text"].strip().lower():
            deduped.append(seg)
    return deduped


def _extract_words(segments: list[dict]) -> list[dict]:
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            confidence = round(float(w.get("probability", 1.0)), 4)
            words.append({
                "word": w["word"].strip(),
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "confidence": confidence,
                "low_confidence": confidence < _LOW_CONFIDENCE_THRESHOLD,
            })
    return words
