"""
sparsity package
----------------
Structured and unstructured weight pruning algorithms, Hessian estimation utilities,
and TPU-accelerated SparseGPT kernel integrations for LLMs (Gemma 4, LLaMA).
"""

import os
import shutil
import time

import torch
from transformers import AutoTokenizer

from .datautils import get_loaders
from .modelutils import find_layers
from .quant import Quantizer
from .sparsegpt import SparseGPT


def prune_gemma4(model_path: str,
                 dataset: str = "c4",
                 minlayer: int = 0,
                 maxlayer: int = 60,
                 prunen: int = 2,
                 prunem: int = 4,
                 nsamples: int = 64,
                 seed: int = 0,
                 percdamp: float = 0.01,
                 blocksize: int = 128,
                 save_path: str = "",
                 dev: str = "cpu") -> str:
    """
    Main entry point for SparseGPT pruning on Gemma 4 architectures.
    Captures layer activations sequentially, calculates second-order Hessian matrices,
    and applies N:M structured sparsity on host CPU before serving on TPU.
    """
    from .gemma4 import gemma4_sequential, get_gemma4

    device = torch.device(dev)
    print(f">>> [SPARSITY ENGINE] Pruning Gemma 4 on device: {device}")
    model = get_gemma4(model_path)
    model.eval()

    # 1. Load calibration dataset (C4, WikiText2, etc.)
    dataloader, testloader = get_loaders(
        dataset,
        nsamples=nsamples,
        seed=seed,
        model=model_path,
        seqlen=model.seqlen,
    )

    # 2. Assemble pruning parameters
    class Args:
        pass

    args = Args()
    args.sparsity = 0
    args.nsamples = nsamples
    args.prunen = prunen
    args.prunem = prunem
    args.wbits = 16
    args.percdamp = percdamp
    args.blocksize = blocksize
    args.minlayer = minlayer
    args.maxlayer = maxlayer
    args.prune_only = ""
    args.invert = False
    args.true_sequential = False

    # 3. Execute sequential layer-by-layer Hessian collection and pruning
    tick = time.time()
    gemma4_sequential(model, dataloader, device, args)
    tock = time.time()
    print(f">>> [SPARSITY ENGINE] Pruning finished in {tock - tick:.2f}s")

    # 4. Save sparsified model and tokenizer to disk
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(
            f">>> [SPARSITY ENGINE] Saving pruned checkpoint to {save_path}..."
        )
        model.save_pretrained(save_path)
        # Ensure multimodal processor and tokenizer configs are saved alongside sparse weights.
        # Gemma 4 uses AutoProcessor (multimodal) which serializes processor_config.json,
        # fallback to AutoTokenizer for text-only backends.
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(model_path).save_pretrained(
                save_path)
        except (OSError, ValueError):
            # transformers raises OSError/ValueError when no processor
            # config exists; text-only checkpoints only ship a tokenizer.
            print(">>> [SPARSITY ENGINE] No AutoProcessor found, "
                  "falling back to AutoTokenizer")
            AutoTokenizer.from_pretrained(model_path).save_pretrained(
                save_path)
        # Preserve original base model config.json and processor_config.json to retain
        # exact architecture specifications (e.g. Gemma4ForConditionalGeneration) that
        # AutoModel.save_pretrained might otherwise overwrite or omit.
        for cfg_name in ["config.json", "processor_config.json"]:
            src_cfg = os.path.join(model_path, cfg_name)
            if os.path.exists(src_cfg):
                shutil.copy2(src_cfg, os.path.join(save_path, cfg_name))
        print(">>> [SPARSITY ENGINE] Checkpoint successfully saved.")

    return save_path


__all__ = [
    "SparseGPT",
    "find_layers",
    "Quantizer",
    "get_loaders",
    "prune_gemma4",
]
