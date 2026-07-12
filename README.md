# whisperai-lyric-transcription

Transcribe song lyrics from audio files to structured JSON using a 3-stage pipeline (vocal separation, Whisper transcription, cleanup), with an optional LoRA fine-tuning pipeline for adapting Whisper to lyric transcription on the DALI dataset.

---

## Inference pipeline

### How it works

```

[1] separate   : Demucs (htdemucs) strips backing track, outputs vocals WAV
[2] transcribe : Whisper large-v3 transcribes vocals with word timestamps
[3] cleanup    : deduplicates segments, flags low-confidence words
```

Each stage caches its output to disk. Re-running skips completed stages automatically. Use `--force` to clear caches and rerun from scratch.

### Output format

```json
{
  "source_file": "/path/to/song.mp3",
  "model": "large-v3",
  "language": "en",
  "duration_seconds": 214.5,
  "words": [
    { "word": "hello", "start": 1.23, "end": 1.56, "confidence": 0.98, "low_confidence": false }
  ],
  "segments": [
    { "text": "hello world", "start": 1.23, "end": 2.10, "words": ["hello", "world"] }
  ]
}
```

### Installation

Requires Python ≥ 3.11, `ffmpeg` on your PATH, and [`uv`](https://github.com/astral-sh/uv) (recommended over pip for faster dependency resolution).

**macOS / Linux:**
```bash
brew install ffmpeg      # macOS
pip install uv
pip install -e .
```

**Windows (conda):**
```bash
conda create -n whisper-lyrics python=3.11
conda activate whisper-lyrics
pip install uv
conda install ffmpeg
uv pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

### Usage

```bash
# Full pipeline (separate → transcribe → cleanup)
whisper-lyrics song.mp3

# Run a single stage
whisper-lyrics song.mp3 --stage separate
whisper-lyrics song.mp3 --stage transcribe
whisper-lyrics song.mp3 --stage cleanup

# Use a different Whisper model
whisper-lyrics song.mp3 --model medium

# Specify output path and cache directory
whisper-lyrics song.mp3 --output ./lyrics/song.json --cache-dir ./cache

# Force rerun all stages
whisper-lyrics song.mp3 --force
```

Supported formats: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`

---

## LoRA fine-tuning pipeline

Fine-tunes Whisper large-v3 on the [DALI v2](https://zenodo.org/records/2577915) lyric dataset ([GitHub](https://github.com/gabolsgabs/DALI)) using LoRA (Low-Rank Adaptation) applied to the decoder's Q/V projections only.

### Pipeline stages

```

[1] download    : fetches DALI audio + annotations via DALI_python_core
[2] preprocess  : Demucs vocal separation → 30s chunks → HF DatasetDict on disk
[3] train       : Seq2SeqTrainer with LoRA adapter, saves to models/lora/
adapter_config.json + adapter_model.safetensors + eval_results.json
```

### Installation

**Windows (conda):**
```bash
conda create -n whisper-lyrics python=3.11
conda activate whisper-lyrics
pip install uv
uv pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install demucs==4.0.1 openai-whisper
uv pip install -e ".[training]"
```

**macOS / Linux:**
```bash
pip install uv
uv pip install -e ".[training]"
```

DALI annotation data must be downloaded manually:

1. Request access at [Zenodo](https://zenodo.org/records/2577915) (requires registration and institutional affiliation)
2. Extract the downloaded gz files into `data/dali/raw/` so that `data/dali/raw/info/DALI_DATA_INFO.gz` exists

Audio is then fetched from YouTube by the download stage via `dali_code.get_audio()`.

### Training

```bash
# Download audio from YouTube (annotations must already be in data/dali/raw/)
python -m training.download --data-dir data/dali/raw

# Preprocess (vocal separation + chunking + alignment)
python -m training.preprocess \
  --dali-data-dir data/dali/raw \
  --processed-dir data/dali/processed

# Train
python -m training.train \
  --model-name openai/whisper-large-v3 \
  --processed-dir data/dali/processed \
  --lora-output-dir models/lora \
  --lora-rank 8 \
  --lora-alpha 16 \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 5000 \
  --eval-steps 500
```

### LoRA configuration

| Parameter | Default | Description |
| `--lora-rank` | 8 | LoRA rank (must be a positive power of 2) |
| `--lora-alpha` | 16 | LoRA scaling factor |
| `--lora-dropout` | 0.05 | Dropout on LoRA layers |
| `--max-steps` | 5000 | Total training steps |
| `--eval-steps` | 500 | Evaluate every N steps |
| `--learning-rate` | 1e-4 | AdamW learning rate |
| `--warmup-steps` | 500 | Linear warmup steps |

The encoder is fully frozen. Only decoder Q/V projections receive LoRA adapters (~1–2% of total parameters).

### Evaluation metrics

- **WER** — Word Error Rate after lowercasing and stripping punctuation
- **PER** — Phoneme Error Rate (WER computed on g2p_en phoneme sequences; rewards phonemically similar transcriptions)

Results are written to `models/lora/eval_results.json`.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"
```

### Project structure

```
src/whisper_lyrics/       # inference pipeline
  stages/
    separate.py           # Demucs vocal separation
    transcribe.py         # Whisper transcription
    cleanup.py            # segment dedup + confidence flagging
  pipeline.py             # orchestrator with caching
  cli.py                  # whisper-lyrics CLI entry point
  exceptions.py           # SeparationError, TranscriptionError, CleanupError

training/                 # LoRA fine-tuning pipeline
  config.py               # TrainingConfig dataclass
  exceptions.py           # DownloadError, PreprocessError, TrainingError
  download.py             # DALI dataset download
  preprocess.py           # vocal sep → 30s chunks → HF DatasetDict
  evaluate.py             # WER + PER metrics
  train.py                # _build_lora_model, DataCollator, Seq2SeqTrainer loop
```
