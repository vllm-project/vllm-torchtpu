"""
datautils.py
------------
Fast streaming calibration dataloaders for SparseGPT.
"""

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model, use_fast=False)


def get_c4(nsamples: int, seed: int, seqlen: int, model: str):
    print(
        f">>> [DATAUTILS] Streaming {nsamples} calibration sequences (seqlen={seqlen}) from C4..."
    )
    traindata = load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True)
    tokenizer = get_tokenizer(model)
    random.seed(seed)
    trainloader = []

    for item in traindata:
        trainenc = tokenizer(item["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            if len(trainloader) == nsamples:
                break

    print(
        f">>> [DATAUTILS] Successfully tokenized {len(trainloader)} calibration batches."
    )
    return trainloader, None


def get_loaders(name: str,
                nsamples: int = 128,
                seed: int = 0,
                seqlen: int = 2048,
                model: str = ""):
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
