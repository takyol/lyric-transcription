import gzip
import pickle
import subprocess
import sys
from pathlib import Path

from .exceptions import DownloadError

_INFO_SUBPATH = "info/DALI_DATA_INFO.gz"
_SKIP_DIRS = {"audio", "info"}


def _audio_dir(data_dir: Path) -> Path:
    return data_dir / "audio"


def _get_youtube_url(song_dir: Path, song_id: str) -> str:
    ann_file = song_dir / song_id
    if not ann_file.exists():
        raise DownloadError(f"Annotation file not found: {ann_file}")
    with gzip.open(ann_file, "rb") as f:
        entry = pickle.load(f)
    return entry.info["audio"]["url"]


def _download_track(url: str, song_id: str, audio_dir: Path) -> None:
    output_template = str(audio_dir / f"{song_id}.%(ext)s")
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
         "-o", output_template, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise DownloadError(f"yt-dlp failed for {song_id}: {result.stderr.strip()}")


def download(data_dir: Path) -> None:
    """Download DALI audio from YouTube into data_dir/audio/ using yt-dlp.

    The DALI annotation gz files must already be present as data_dir/<song_id>/<song_id>
    (download from https://zenodo.org/records/2577915 after requesting access).
    """
    data_dir = Path(data_dir).resolve()

    song_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in _SKIP_DIRS]
    if not song_dirs:
        raise DownloadError(
            f"No song ID directories found in {data_dir}. "
            "Extract the DALI gz files from Zenodo into per-song subdirectories first."
        )

    audio_dir = _audio_dir(data_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for song_dir in sorted(song_dirs):
        song_id = song_dir.name
        existing = list(audio_dir.glob(f"{song_id}.*"))
        if existing:
            print(f"  {song_id}: already downloaded, skipping.")
            continue
        try:
            url = _get_youtube_url(song_dir, song_id)
            print(f"  {song_id}: downloading from {url}")
            _download_track(url, song_id, audio_dir)
        except DownloadError as e:
            print(f"  {song_id}: failed — {e}")
            errors.append(song_id)

    if errors:
        print(f"Warning: {len(errors)}/{len(song_dirs)} tracks failed: {errors}")
    else:
        print(f"Audio download complete → {audio_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DALI audio from YouTube via yt-dlp")
    parser.add_argument("--data-dir", default="data/dali/raw", type=Path,
                        help="Directory containing per-song DALI annotation subdirectories")
    args = parser.parse_args()

    try:
        download(args.data_dir)
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
