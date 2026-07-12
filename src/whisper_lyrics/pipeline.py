from pathlib import Path
from .stages.cleanup import cleanup
from .stages.separate import separate
from .stages.transcribe import transcribe

_VALID_STAGES = {"separate", "transcribe", "cleanup"}


def run_pipeline(
    input_path: Path,
    cache_dir: Path,
    output_path: Path,
    model_name: str = "large-v3",
    force: bool = False,
    stage: str | None = None,
    adapter_dir: Path | None = None,
) -> Path:
    if stage is not None and stage not in _VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Must be one of: {', '.join(sorted(_VALID_STAGES))}")

    if force:
        _clear_artifacts(input_path, cache_dir, output_path)

    if stage is None or stage == "separate":
        vocals_path = separate(input_path, cache_dir)
        if stage == "separate":
            return vocals_path
    else:
        vocals_path = cache_dir / f"{input_path.stem}.vocals.wav"

    if stage is None or stage == "transcribe":
        if adapter_dir is not None:
            # Lazy import: the HF/PEFT stack is only needed in adapter mode
            from .stages.transcribe_hf import transcribe_hf
            raw_json_path = transcribe_hf(vocals_path, cache_dir, model_name, adapter_dir)
        else:
            raw_json_path = transcribe(vocals_path, cache_dir, model_name)
        if stage == "transcribe":
            return raw_json_path
    else:
        raw_json_path = cache_dir / f"{input_path.stem}.raw.json"

    # stage is None or "cleanup"
    lyrics_path = cleanup(raw_json_path, input_path, model_name)
    if output_path != lyrics_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lyrics_path.read_text())
    return output_path


def _clear_artifacts(input_path: Path, cache_dir: Path, output_path: Path) -> None:
    stem = input_path.stem
    for artifact in [
        cache_dir / f"{stem}.vocals.wav",
        cache_dir / f"{stem}.raw.json",
        cache_dir / f"{stem}.lyrics.json",
        output_path,
    ]:
        artifact.unlink(missing_ok=True)
