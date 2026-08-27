"""
dataset_providers.py
--------------------
Dataset provider and loader utilities for benchmark evaluation and calibration.
Supplies formatted prompts for multiple-choice tasks (MMLU-Pro) and streaming prefill text (C4).
"""

import random
from typing import Any, Dict, List

from datasets import load_dataset


def load_mmlu_pro_prompts(task_name: str = "mmlu_pro",
                          limit: int = 100,
                          split: str = "test",
                          seed: int = 42) -> List[str]:
    """
    Loads MMLU-Pro multiple-choice benchmark questions and formats them as standard few-shot/zero-shot prompts.
    Supports filtering by specific category (e.g., mmlu_pro_math, mmlu_pro_biology).
    """
    random.seed(seed)

    # Extract sub-category if specified (e.g., mmlu_pro_philosophy -> philosophy)
    category = None
    if "_" in task_name and task_name != "mmlu_pro":
        category = task_name.split("mmlu_pro_")[-1]

    # Load dataset from HuggingFace
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    if category:
        ds = ds.filter(lambda x: x["category"].lower() == category.lower())

    prompts = []
    options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # Format each sample into a standard question-options prompt
    for item in ds:
        q = item["question"]
        opts = item["options"]
        formatted_opts = "\n".join(
            [f"{options_letters[i]}. {opt}" for i, opt in enumerate(opts)])
        prompt_text = f"Question:\n{q}\n\nOptions:\n{formatted_opts}\n\nAnswer:"
        prompts.append(prompt_text)

        # Break when limit is reached
        if limit and len(prompts) >= limit:
            break

    return prompts


def get_evaluation_prompts(dataset_config: Dict[str, Any]) -> List[str]:
    """
    Generic dispatcher for retrieving benchmark prompt lists based on configuration dict.
    Supports mmlu_pro and c4 dataset streams.
    """
    task = dataset_config.get("task", "mmlu_pro")
    limit = dataset_config.get("limit", 100)

    # Dispatch to appropriate dataset provider
    if "mmlu_pro" in task:
        return load_mmlu_pro_prompts(task_name=task, limit=limit)
    elif "c4" in task:
        # Stream raw text slices from C4 validation split
        ds = load_dataset("allenai/c4",
                          "en",
                          split="validation",
                          streaming=True)
        prompts = [
            item["text"][:1024] for i, item in enumerate(ds) if i < limit
        ]
        return prompts
    else:
        raise ValueError(f"Unsupported dataset task: {task}")
