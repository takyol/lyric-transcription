import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from ..exceptions import SeparationError


def separate(input_path: Path, cache_dir: Path) -> Path:
    output_path = cache_dir / f"{input_path.stem}.vocals.wav"
    if output_path.exists():
        return output_path

    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir, prefix="demucs_"))
    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems", "vocals",
                "--out", str(tmp_dir),
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SeparationError(f"Demucs failed: {e.stderr}") from e

    vocals_src = tmp_dir / "htdemucs" / input_path.stem / "vocals.wav"
    if not vocals_src.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SeparationError(f"Demucs output not found at {vocals_src}")

    shutil.copy(vocals_src, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_path
