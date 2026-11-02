"""Fair generation and ROUGE evaluation helpers for SAMSum."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from data_preprocessing import build_prompt


@torch.inference_mode()
def generate_summaries(model: Any, tokenizer: Any, dialogues: Sequence[str],
                       *, batch_size: int = 1, max_input_length: int = 1024,
                       max_new_tokens: int = 128, num_beams: int = 1) -> list[str]:
    """Generate summaries using the same chat prompt and deterministic decoding."""
    if not dialogues:
        return []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # Decoder-only batched generation needs left padding.
    model.eval()
    model.config.use_cache = True
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(dialogue)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for dialogue in dialogues
    ]
    outputs: list[str] = []
    device = next(model.parameters()).device
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                add_special_tokens=False,
            ).to(device)
            generated = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            continuations = generated[:, tokens["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(continuations, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_padding_side
    model.config.use_cache = False
    return [text.strip() for text in outputs]


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute ROUGE-1/2/L/Lsum; preserve raw [0,1] values."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same number of rows.")
    if not predictions:
        raise ValueError("Cannot compute ROUGE for an empty evaluation set.")
    import evaluate

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=list(predictions),
        references=list(references),
        use_stemmer=True,
        use_aggregator=True,
    )
    keys = ("rouge1", "rouge2", "rougeL", "rougeLsum")
    return {key: float(scores[key]) for key in keys if key in scores}
