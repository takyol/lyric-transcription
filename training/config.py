from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Paths
    dali_data_dir: str = "data/dali/raw"
    processed_dir: str = "data/dali/processed"
    lora_output_dir: str = "models/lora"

    # Preprocessing
    chunk_duration: float = 30.0
    chunk_stride: float = 5.0
    min_words_per_chunk: int = 3

    # Model
    model_name: str = "openai/whisper-large-v3"

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 5000
    eval_steps: int = 500

    def __post_init__(self) -> None:
        if self.lora_rank <= 0:
            raise ValueError(
                f"lora_rank must be positive, got {self.lora_rank}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.min_words_per_chunk < 1:
            raise ValueError(
                f"min_words_per_chunk must be >= 1, got {self.min_words_per_chunk}"
            )
