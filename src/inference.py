"""Adapter reload and local inference helpers."""
from __future__ import annotations

from pathlib import Path


def load_adapter_for_inference(model_id: str, adapter_dir: str | Path,
                               tokenizer):
    """Reload frozen 4-bit base weights plus the saved PEFT adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Adapter inference requires a CUDA runtime for this project setup.")
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=qconfig,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    return model
