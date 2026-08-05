"""Lazy local inference for the fine-tuned Qwen price model.

The filename is historical. This module loads Qwen2.5 3B in four-bit
precision, attaches the project's PEFT adapter, and caches both objects.
"""

import os
import threading
from pathlib import Path


BASE_MODEL = os.getenv("PRICER_FINETUNED_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_ADAPTER_ID = "akshaymgupte87/price-2026-08-05_11.17.28-lite"
LOCAL_ADAPTER = (
    Path(__file__).resolve().parent
    / "notebooks"
    / "price-2026-08-05_11.17.28-lite"
)

_model = None
_tokenizer = None
_inference_lock = threading.RLock()


def adapter_source() -> str:
    """Return the configured adapter, preferring the checked-out local copy."""
    configured = os.getenv("PRICER_FINETUNED_ADAPTER")
    if configured:
        return configured
    if (LOCAL_ADAPTER / "adapter_config.json").is_file():
        return str(LOCAL_ADAPTER)
    return DEFAULT_ADAPTER_ID


def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _inference_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The fine-tuned Qwen specialist requires an NVIDIA CUDA GPU. "
                "Set PRICER_SPECIALIST_BACKEND=ollama to use the fallback backend."
            )

        source = adapter_source()
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(source)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization,
            device_map="auto",
        )
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, source)
        model.eval()

        _tokenizer = tokenizer
        _model = model
        return _model, _tokenizer


def generate(prompt: str) -> str:
    """Generate only the deterministic completion for a pricing prompt."""
    import torch

    with _inference_lock:
        model, tokenizer = _load_model()
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
