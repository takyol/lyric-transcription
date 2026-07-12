import argparse
import shutil
import sys
from pathlib import Path
from .exceptions import CleanupError, SeparationError, TranscriptionError
from .pipeline import run_pipeline

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="whisper-lyrics",
        description="Transcribe song lyrics from an audio file to structured JSON.",
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["separate", "transcribe", "cleanup"])
    parser.add_argument("--adapter-dir", type=Path,
                        help="LoRA adapter directory (e.g. models/lora); transcribes "
                             "with the HF Whisper model + adapter instead of openai-whisper")
    args = parser.parse_args()

    input_path: Path = args.audio_file.resolve()
    _validate_input(input_path)

    adapter_dir: Path | None = None
    if args.adapter_dir is not None:
        adapter_dir = args.adapter_dir.resolve()
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"Error: No adapter_config.json in {adapter_dir} — not a LoRA adapter directory.",
                  file=sys.stderr)
            sys.exit(1)

    cache_dir: Path = (args.cache_dir or input_path.parent).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path: Path = (
        args.output or input_path.parent / f"{input_path.stem}.lyrics.json"
    ).resolve()

    try:
        result = run_pipeline(
            input_path=input_path,
            cache_dir=cache_dir,
            output_path=output_path,
            model_name=args.model,
            force=args.force,
            stage=args.stage,
            adapter_dir=adapter_dir,
        )
        print(f"Output written to {result}")
    except (SeparationError, TranscriptionError, CleanupError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def _validate_input(path: Path) -> None:
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        print(
            f"Error: Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
            file=sys.stderr,
        )
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH. Install it via your system package manager.", file=sys.stderr)
        sys.exit(1)
