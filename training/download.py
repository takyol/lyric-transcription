import sys
from pathlib import Path

from .exceptions import DownloadError

_INFO_SUBPATH = "info/DALI_DATA_INFO.gz"


def _audio_dir(data_dir: Path) -> Path:
    return data_dir / "audio"


def download(data_dir: Path) -> None:
    """Download DALI audio from YouTube into data_dir/audio/.

    The DALI annotation gz files must already be present in data_dir — download
    them manually from https://zenodo.org/records/2577915 after requesting access.
    Audio is then fetched from YouTube via dali_code.get_audio().
    """
    data_dir = Path(data_dir).resolve()  # DALI requires absolute paths
    info_file = data_dir / _INFO_SUBPATH

    if not info_file.exists():
        raise DownloadError(
            f"DALI annotation data not found at {info_file}. "
            "Download the gz files manually from https://zenodo.org/records/2577915 "
            "(access requires registration) and extract them into that directory."
        )

    audio_dir = _audio_dir(data_dir)
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"Audio already present at {audio_dir}, skipping download.")
        return

    try:
        import DALI as dali_code
    except ImportError:
        raise DownloadError(
            "dali-dataset is not installed. Run: pip install dali-dataset"
        )

    try:
        dali_info = dali_code.get_info(str(info_file))
        audio_dir.mkdir(parents=True, exist_ok=True)
        errors = dali_code.get_audio(dali_info, str(audio_dir))
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"DALI audio download failed: {e}") from e

    if errors:
        print(f"Warning: {len(errors)} tracks unavailable on YouTube.")
    print(f"Audio download complete → {audio_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DALI audio from YouTube")
    parser.add_argument("--data-dir", default="data/dali/raw", type=Path,
                        help="Directory containing the DALI gz annotation files")
    args = parser.parse_args()

    try:
        download(args.data_dir)
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
