import json
from pathlib import Path
from ..exceptions import TranscriptionError

try:
    import whisper
except ImportError:  
    whisper = None  


_LYRIC_PROMPT = "Song lyrics:"


class _FloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


def transcribe(
    vocals_path: Path,
    cache_dir: Path,
    model_name: str = "large-v3",
) -> Path:
    stem = vocals_path.stem.removesuffix(".vocals")
    output_path = cache_dir / f"{stem}.raw.json"
    if output_path.exists():
        return output_path

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(
            str(vocals_path),
            language="en",
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=_LYRIC_PROMPT,
        )
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}") from e

    output_path.write_text(json.dumps(result, cls=_FloatEncoder, indent=2))
    return output_path
