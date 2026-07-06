import os
import sys
from pathlib import Path

from .exceptions import DownloadError


def _run_dali_downloader(data_dir: Path, token: str) -> None:
    """Wraps DALI_python_core downloader. Downloads audio + annotations into data_dir."""
    try:
        import DALI as dali
    except ImportError:
        raise DownloadError(
            "DALI_python_core is not installed. Run: pip install DALI-python-core"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    # DALI_python_core downloads annotations from Zenodo and audio via the token.
    try:
        dali.utils.get_audio(
            dali_data_path=str(data_dir),
            audio_path=str(data_dir),
            token=token,
        )
    except Exception as e:
        raise DownloadError(f"DALI downloader call failed: {e}") from e


def download(data_dir: Path) -> None:
    """Download DALI dataset into data_dir. Skip if already present."""
    data_dir = Path(data_dir)  # accept str or Path
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"DALI data already present at {data_dir}, skipping download.")
        return

    token = os.environ.get("DALI_AUDIO_TOKEN")
    if not token:
        raise DownloadError(
            "DALI_AUDIO_TOKEN environment variable is not set. "
            "Register at https://zenodo.org/record/2658420 to obtain a token."
        )

    try:
        _run_dali_downloader(data_dir, token)
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"DALI download failed: {e}") from e

    print(f"Download complete → {data_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DALI dataset")
    parser.add_argument("--data-dir", default="data/dali/raw", type=Path)
    args = parser.parse_args()

    try:
        download(args.data_dir)
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
