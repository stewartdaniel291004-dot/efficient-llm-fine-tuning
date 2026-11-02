"""Memory-aware QLoRA loading and training helpers for Qwen causal LMs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def require_t4_cuda() -> torch.dtype:
    """Return FP16 compute dtype after checking CUDA is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Colab choose Runtime > Change runtime type > T4 GPU."
        )
    name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {name}")
    if "T4" not in name.upper():
        print("Note: this workflow is tuned for T4; detected a different CUDA GPU.")
    return torch.float16


def cuda_memory_gib() -> dict[str, float] | None:
    """Read current and peak VRAM metrics when CUDA is available."""
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def load_base_for_baseline(model_id: str, tokenizer: Any):
    """Load a non-quantized FP16 base model for the pre-training baseline."""
    from transformers import AutoModelForCausalLM

    dtype = require_t4_cuda()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "The baseline model did not fit in GPU memory. Restart the runtime, "
            "close other GPU workloads, or reduce evaluation batch size."
        ) from exc
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_qlora_model(model_id: str, tokenizer: Any, *, lora_r: int = 8,
                     lora_alpha: int = 16, lora_dropout: float = 0.05,
                     target_modules: str | list[str] = "all-linear",
                     gradient_checkpointing: bool = True):
    """Load NF4 4-bit base weights, prepare them, and attach LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = require_t4_cuda()
    if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_available():
        raise RuntimeError("CUDA/bitsandbytes 4-bit QLoRA requires a supported NVIDIA GPU runtime.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "The 4-bit model still ran out of VRAM while loading. Restart the runtime, "
            "check for other GPU allocations, and use a T4 with sufficient free memory."
        ) from exc
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "Could not load the model in bitsandbytes 4-bit mode. Confirm this is a "
            "CUDA Colab runtime, restart after package installation if requested, and "
            "check that the installed bitsandbytes wheel supports the runtime. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def train_qlora(model, tokenizer: Any, train_dataset: Any, eval_dataset: Any,
                *, output_dir: str | Path = "artifacts/checkpoints",
                learning_rate: float = 2e-4, num_train_epochs: float = 1,
                max_steps: int = -1, batch_size: int = 1,
                gradient_accumulation_steps: int = 8, logging_steps: int = 5,
                save_steps: int = 25, seed: int = 42, resume: bool = False):
    """Train adapters with Trainer and return (trainer, train_result)."""
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    kwargs = dict(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="epoch",
        eval_accumulation_steps=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=0.3,
    )
    args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    try:
        result = trainer.train(resume_from_checkpoint=True if resume else None)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory during training. Reduce MAX_INPUT_LENGTH, "
            "use batch size 1, increase gradient accumulation to preserve effective "
            "batch size, or reduce the training subset. Then restart the Colab runtime."
        ) from exc
    return trainer, result


def save_adapter(model, tokenizer: Any, adapter_dir: str | Path) -> Path:
    """Save PEFT adapter and tokenizer (no full base weights)."""
    path = Path(adapter_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path
