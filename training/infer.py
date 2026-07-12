import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with a LoRA-adapted Whisper model"
    )
    parser.add_argument("audio", help="Path to audio file (wav/mp3/flac/...)")
    parser.add_argument("--adapter-dir", default="models/lora",
                        help="LoRA adapter dir: models/lora (final) or "
                             "models/lora/checkpoints/checkpoint-<step>")
    parser.add_argument("--model-name", default="openai/whisper-large-v3",
                        help="Base model the adapter was trained on")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    base = WhisperForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.merge_and_unload()  # bake adapter into weights for plain inference

    processor = WhisperProcessor.from_pretrained(
        args.model_name, language="English", task="transcribe"
    )
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=0 if use_cuda else -1,
        torch_dtype=dtype,
    )
    result = asr(args.audio, return_timestamps=True)
    print(result["text"].strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
