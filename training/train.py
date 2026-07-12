# training/train.py
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from peft import LoraConfig, TaskType, get_peft_model
from .config import TrainingConfig
from .exceptions import TrainingError

_log = logging.getLogger(__name__)


def _build_lora_model(base_model, config: TrainingConfig):
    """Apply LoRA to decoder Q/V projections only. Freeze all encoder parameters."""
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)

    for _name, param in model.named_parameters():
        if param.requires_grad and "encoder" in _name:
            param.requires_grad_(False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]

    encoder_trainable = [n for n in trainable if "encoder" in n]
    if encoder_trainable:
        raise TrainingError(
            f"Encoder params still trainable after freeze: {encoder_trainable[:3]}"
        )

    decoder_trainable = [n for n in trainable if "decoder" in n]
    if not decoder_trainable:
        raise TrainingError(
            "No decoder LoRA params are trainable, check target_modules configuration."
        )

    total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.debug("Trainable params: %d / %d (%.2f%%)", n_trainable, total, 100 * n_trainable / total)
    return model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # WhisperProcessor typed as Any to avoid top-level transformers import

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Strip BOS per-row before padding so behaviour is independent of padding side
        bos_id = self.processor.tokenizer.bos_token_id
        label_features = [
            {"input_ids": f["labels"][1:] if f["labels"] and f["labels"][0] == bos_id else f["labels"]}
            for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def _prepare_dataset(dataset, processor):
    """Map raw audio + text columns to input_features + labels for the Trainer."""

    def _map_fn(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    remove_cols = list(set(dataset.column_names) & {"audio", "text", "song_id"})
    return dataset.map(_map_fn, remove_columns=remove_cols)


def train(config: TrainingConfig) -> None:
    from datasets import load_from_disk
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from .evaluate import compute_metrics_fn

    output_dir = Path(config.lora_output_dir)
    checkpoints_dir = output_dir / "checkpoints"

    dataset = load_from_disk(config.processed_dir)

    processor = WhisperProcessor.from_pretrained(
        config.model_name, language="English", task="transcribe"
    )

    # Feature-extract if raw audio columns are present
    if "audio" in dataset["train"].column_names:
        dataset = {split: _prepare_dataset(dataset[split], processor) for split in dataset}

    base_model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.use_cache = False
    base_model.generation_config.language = "english"
    base_model.generation_config.task = "transcribe"
    base_model.generation_config.forced_decoder_ids = None

    model = _build_lora_model(base_model, config)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=25,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor.tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(processor.tokenizer),
    )

    try:
        trainer.train()
    except Exception as e:
        raise TrainingError(f"Training loop failed: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    # Final evaluation on test split via direct generation (bypasses Trainer generate bug)
    from .evaluate import compute_wer, compute_per
    model.eval()
    device = next(model.parameters()).device
    preds, refs = [], []
    for example in dataset["test"]:
        input_features = torch.tensor(example["input_features"]).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=225)
        preds.append(processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True))
        label_ids = [t for t in example["labels"] if t != -100]
        refs.append(processor.tokenizer.decode(label_ids, skip_special_tokens=True))
    wer_val = compute_wer(preds, refs)
    per_val = compute_per(preds, refs)

    eval_results = {
        "split": "test",
        "wer": wer_val,
        "per": per_val,
        "num_samples": len(dataset["test"]),
    }
    (output_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))
    _log.info("Test WER: %.4f | Test PER: %.4f", wer_val, per_val)
    _log.info("Adapter saved to %s", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on DALI")
    parser.add_argument("--model-name", default="openai/whisper-large-v3")
    parser.add_argument("--processed-dir", default="data/dali/processed")
    parser.add_argument("--lora-output-dir", default="models/lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        processed_dir=args.processed_dir,
        lora_output_dir=args.lora_output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
    )
    try:
        train(cfg)
    except TrainingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
