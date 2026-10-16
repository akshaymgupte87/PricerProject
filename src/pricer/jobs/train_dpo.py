"""Export reviewed price preferences and run an explicit QLoRA DPO job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pricer.config import Settings
from pricer.persistence.intelligence_store import IntelligenceStore


def export_preferences(database: Path, output: Path, minimum_examples: int = 3) -> int:
    """Write corrected-price preference pairs as JSONL without loading ML packages."""
    examples = IntelligenceStore(database).preference_examples()
    if len(examples) < minimum_examples:
        raise ValueError(
            "Need at least "
            f"{minimum_examples} distinct corrected-price examples; found {len(examples)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def train(dataset_path: Path, output: Path, base_model: str, args) -> None:
    """Run QLoRA DPO; heavyweight training dependencies load only on this command."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA DPO requires a CUDA-enabled PyTorch environment")

    raw = load_dataset("json", data_files=str(dataset_path), split="train")
    if len(raw) < 3:
        raise ValueError("DPO training requires at least three examples for train and validation splits")
    raw = raw.select_columns(["prompt", "chosen", "rejected"])
    data = raw.train_test_split(
        test_size=max(1, round(len(raw) * args.validation_fraction)),
        seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    training_args = DPOConfig(
        output_dir=str(output),
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "train", "evaluate"))
    parser.add_argument("--database", type=Path, default=Settings.from_env().intelligence_path)
    parser.add_argument(
        "--dataset", type=Path,
        default=Settings.from_env().home / "artifacts" / "price_preferences.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Settings.from_env().home / "artifacts" / "qwen-price-dpo",
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("PRICER_FINETUNED_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--minimum-examples", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument(
        "--adapter", type=Path,
        default=Path(os.getenv("PRICER_FINETUNED_ADAPTER", str(Settings.from_env().home / "artifacts" / "qwen-price-dpo"))),
    )
    parser.add_argument(
        "--training-data", type=Path,
        help="Preference JSONL used for overlap checks during independent evaluation.",
    )
    parser.add_argument(
        "--report", type=Path,
        default=Settings.from_env().home / "artifacts" / "dpo-evaluation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "export":
        count = export_preferences(args.database, args.dataset, args.minimum_examples)
        print(f"Exported {count} DPO preference pairs to {args.dataset}")
    elif args.action == "train":
        if not 0 < args.validation_fraction < 0.5:
            raise ValueError("--validation-fraction must be between 0 and 0.5")
        train(args.dataset, args.output, args.base_model, args)
        print(f"Saved DPO adapter to {args.output}")
    else:
        if args.training_data is None:
            raise ValueError("dpo evaluate requires --training-data for overlap checks")
        from pricer.jobs.evaluate_dpo import evaluate

        report = evaluate(args.dataset, args.training_data, args.base_model, args.adapter, args.report)
        print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
        print(f"Saved paired evaluation report to {args.report}")


if __name__ == "__main__":
    main()
