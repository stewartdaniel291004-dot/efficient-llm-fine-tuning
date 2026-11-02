"""SAMSum access, data checks, preview, and causal-LM preprocessing helpers."""
from __future__ import annotations

from collections import Counter
from typing import Any

DATASET_ID = "Samsung/samsum"
REQUIRED_COLUMNS = ("id", "dialogue", "summary")


def load_samsum(dataset_id: str = DATASET_ID):
    """Load SAMSum and turn common Hub/network problems into useful messages."""
    from datasets import load_dataset

    try:
        dataset = load_dataset(dataset_id)
    except Exception as exc:
        message = (
            f"Could not load dataset {dataset_id!r}. Check the runtime's internet "
            "connection, the dataset ID on the Hugging Face Hub, and that the "
            "datasets package is installed and current. If the Hub reports a "
            "temporary outage, retry in a few minutes. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise RuntimeError(message) from exc

    if not dataset:
        raise RuntimeError(f"Dataset {dataset_id!r} loaded but contains no splits.")
    return dataset


def inspect_dataset(dataset: Any) -> dict[str, Any]:
    """Return/report split schemas and require the expected SAMSum columns."""
    report = {}
    for split_name, split in dataset.items():
        columns = list(split.column_names)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
        report[split_name] = {
            "rows": len(split),
            "columns": columns,
            "features": split.features,
            "missing_required_columns": missing_columns,
        }
        print(f"{split_name}: {len(split):,} rows")
        print(f"  columns: {columns}")
        print(f"  features: {split.features}")
        if missing_columns:
            raise ValueError(
                f"Split {split_name!r} is missing required columns: {missing_columns}. "
                "Expected id, dialogue, summary. Verify the dataset ID/schema."
            )
    return report


def check_data_quality(split: Any, split_name: str = "split") -> dict[str, Any]:
    """Count blank/missing cells and duplicate IDs and dialogue-summary pairs."""
    absent = [column for column in REQUIRED_COLUMNS if column not in split.column_names]
    if absent:
        raise ValueError(f"{split_name} lacks required columns: {absent}")

    null_or_blank = Counter()
    ids, pairs = [], []
    for row in split:
        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                null_or_blank[column] += 1
        ids.append(row.get("id"))
        pairs.append((row.get("dialogue"), row.get("summary")))

    duplicate_ids = len(ids) - len(set(ids))
    duplicate_pairs = len(pairs) - len(set(pairs))
    result = {
        "split": split_name,
        "rows": len(split),
        "missing_or_blank": dict(null_or_blank),
        "duplicate_ids": duplicate_ids,
        "duplicate_dialogue_summary_pairs": duplicate_pairs,
    }
    print(result)
    return result


def show_examples(split: Any, n: int = 3, seed: int = 42) -> None:
    """Display real dialogue/reference pairs from the supplied split."""
    import pandas as pd

    if len(split) == 0:
        print("This split is empty; no examples to display.")
        return
    indices = list(range(min(n, len(split))))
    for number, index in enumerate(indices, start=1):
        row = split[index]
        print(f"Example {number} (id={row['id']})")
        print("Conversation:\n" + row["dialogue"])
        print("\nReference summary:\n" + row["summary"])
        print("\n" + "-" * 80 + "\n")


def select_splits(dataset: Any, seed: int = 42) -> tuple[Any, Any, Any]:
    """Use official splits where available; never source training rows from test."""
    if "train" not in dataset:
        raise ValueError("SAMSum has no train split; cannot train without an official train split.")
    if "validation" not in dataset:
        raise ValueError("SAMSum has no validation split; please verify the loaded dataset revision.")
    if "test" not in dataset:
        raise ValueError("SAMSum has no test split; refusing to substitute training data for final evaluation.")
    return dataset["train"], dataset["validation"], dataset["test"]


def limit_split(split: Any, limit: int | None, seed: int = 42):
    """Take a reproducible subset; None retains all rows."""
    if limit is None or limit >= len(split):
        return split
    if limit < 1:
        raise ValueError("Subset size must be positive.")
    return split.shuffle(seed=seed).select(range(limit))


def build_prompt(dialogue: str) -> str:
    """Create a consistent instruction prompt using the model chat template."""
    return (
        "Summarize the following conversation accurately and concisely. "
        "Include the key decisions and relevant details.\n\n"
        f"Conversation:\n{dialogue}"
    )


def preprocess_example(example: dict[str, str], tokenizer: Any,
                       max_input_length: int = 1024,
                       max_target_length: int = 192) -> dict[str, Any]:
    """Tokenize prompt/target separately and mask prompt labels with -100."""
    prompt = build_prompt(example["dialogue"])
    # Chat template places the assistant generation boundary at the end of prompt.
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    target_ids = tokenizer(
        example["summary"] + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )["input_ids"]

    # Reserve space for at least the first target tokens, keeping the prompt end.
    prompt_budget = max(1, max_input_length - len(target_ids))
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
