import json
import pickle
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from .config import TrainingConfig
from .exceptions import PreprocessError
import soundfile as sf 


def _chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_dur: float = 30.0,
    stride: float = 5.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio into overlapping windows. Returns (start_sec, end_sec, chunk)."""
    chunk_samples = int(chunk_dur * sr)
    stride_samples = int(stride * sr)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunks.append((start / sr, end / sr, audio[start:end]))
        if end >= len(audio):
            break
        start += stride_samples
    return chunks


def _transcript_for_chunk(
    words: list[dict],
    start_sec: float,
    end_sec: float,
    min_words: int = 3,
) -> Optional[str]:
    """Collect words whose midpoint falls in [start_sec, end_sec).
    Returns None if fewer than min_words words qualify."""
    in_chunk = [
        w["text"]
        for w in words
        if start_sec <= (w["time"][0] + w["time"][1]) / 2 < end_sec
    ]
    if len(in_chunk) < min_words:
        return None
    return " ".join(in_chunk)


def _load_dali_words(annotation_path: Path) -> list[dict]:
    """Load word-level annotations from a DALI annotation file (gzip-compressed pickle).

    Each returned dict has the shape: {"text": str, "time": [start_sec, end_sec]}.
    The DALI entry object lives at top level; words are at
    entry.annotations['annot']['words'].
    """
    import gzip
    with gzip.open(annotation_path, "rb") as f:
        entry = pickle.load(f)
    return entry.annotations["annot"]["words"]


def _read_audio(audio_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono float32 at target_sr.
    """

    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import resampy  # lazy import
        audio = resampy.resample(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def preprocess(config: TrainingConfig) -> None: 
    """Main preprocessing pipeline.
    Iterates all song directories in ``config.dali_data_dir``, separates vocals,
    chunks audio, aligns transcripts, and saves an HF DatasetDict to
    ``config.processed_dir``.
    """
    raw_dir = Path(config.dali_data_dir)
    processed_dir = Path(config.processed_dir)
    errors_path = raw_dir.parent / "preprocess_errors.jsonl"
    chunks_dir = processed_dir / "audio_chunks"

    # Skip if already done 
    if processed_dir.exists() and (processed_dir / "train").exists():
        print(f"Processed dataset already at {processed_dir}, skipping.")
        return

    if separate is None:
        raise PreprocessError(
            "Vocal separation is not available. "
            "Ensure whisper_lyrics is installed: pip install -e ."
        )

    _SKIP_DIRS = {"audio", "info"}
    song_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name not in _SKIP_DIRS)
    if not song_dirs:
        raise PreprocessError(f"No songs found in {raw_dir}")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    errors: list[dict] = []

    for song_dir in song_dirs:
        song_id = song_dir.name
        try:
            # Audio is downloaded to raw_dir/audio/ by training.download
            audio_dir = raw_dir / "audio"
            audio_files = list(audio_dir.glob(f"{song_id}.*")) if audio_dir.exists() else []
            # DALI annotation files have no extension — file named after song ID inside song dir
            ann_files = [f for f in song_dir.iterdir()
                         if f.is_file() and f.suffix == "" and f.name == song_id]
            if not audio_files:
                raise ValueError(f"No audio file found in {audio_dir} for song {song_id}")
            if not ann_files:
                raise ValueError("No annotation file found (expected extension-less file named after song ID)")

            vocals_path = separate(audio_files[0], raw_dir / "audio")
            audio, sr = _read_audio(vocals_path)
            words = _load_dali_words(ann_files[0])

            for chunk_idx, (start_sec, end_sec, chunk) in enumerate(
                _chunk_audio(audio, sr, config.chunk_duration, config.chunk_stride)
            ):
                text = _transcript_for_chunk(
                    words, start_sec, end_sec, config.min_words_per_chunk
                )
                if text is None:
                    continue
                chunk_path = chunks_dir / f"{song_id}_chunk_{chunk_idx:04d}.wav"
                sf.write(str(chunk_path), chunk, sr)
                records.append({"audio": str(chunk_path), "text": text, "song_id": song_id})

        except Exception as e:
            errors.append({"song_id": song_id, "error": str(e)})

    if errors:
        with open(errors_path, "w") as f:
            for err in errors:
                f.write(json.dumps(err) + "\n")
        error_rate = len(errors) / len(song_dirs)
        if error_rate > 0.20:
            raise PreprocessError(
                f"{len(errors)}/{len(song_dirs)} songs failed preprocessing "
                f"({error_rate:.0%} > 20% threshold). Check {errors_path}."
            )
        print(f"Warning: {len(errors)} songs failed preprocessing (logged to {errors_path})")

    if not records:
        raise PreprocessError("No valid chunks were produced.")

    from datasets import Audio, Dataset, DatasetDict

    # Song-level split. Sort song IDs for reproducibility
    song_ids = sorted(set(r["song_id"] for r in records))
    n = len(song_ids)
    train_ids = set(song_ids[: int(0.8 * n)])
    val_ids = set(song_ids[int(0.8 * n) : int(0.9 * n)])
    test_ids = set(song_ids) - train_ids - val_ids

    def build_split(ids: set) -> Dataset:
        rows = [r for r in records if r["song_id"] in ids]
        if not rows:
            return Dataset.from_dict({"audio": [], "text": [], "song_id": []})
        ds = Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    dataset_dict = DatasetDict({
        "train": build_split(train_ids),
        "validation": build_split(val_ids),
        "test": build_split(test_ids),
    })
    dataset_dict.save_to_disk(str(processed_dir))

    n_chunks = len(records)
    print(
        f"Preprocessed {n_chunks} chunks from {len(song_ids)} songs → {processed_dir} "
        f"(train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)} songs)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess DALI dataset for training")
    parser.add_argument("--dali-data-dir", default="data/dali/raw")
    parser.add_argument("--processed-dir", default="data/dali/processed")
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    parser.add_argument("--chunk-stride", type=float, default=5.0)
    parser.add_argument("--min-words-per-chunk", type=int, default=3)
    args = parser.parse_args()

    config = TrainingConfig(
        dali_data_dir=args.dali_data_dir,
        processed_dir=args.processed_dir,
        chunk_duration=args.chunk_duration,
        chunk_stride=args.chunk_stride,
        min_words_per_chunk=args.min_words_per_chunk,
    )
    try:
        preprocess(config)
    except PreprocessError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
