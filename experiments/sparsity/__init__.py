from typing import Optional, Set
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
                 dev: str = "cpu",
                 sparse_layer_indices: Optional[Set[int]] = None,
                 use_lrc: bool = False,
                 lrc_rank: int = 128,
                 lrc_iters: int = 3,
                 save_prefix24_path: Optional[str] = None) -> str:
    """
    Main entry point for SparseGPT pruning on Gemma 4 architectures.
    Captures layer activations sequentially, calculates second-order Hessian matrices,
    and applies N:M structured sparsity on host CPU before serving on TPU.

    Args:
        model_path: Path to HuggingFace or checkpoint directory.
        dataset: Calibration dataset name ('c4', 'wikitext2').
        minlayer: Starting layer index for pruning range.
        maxlayer: Ending layer index (exclusive) for pruning range.
        prunen: N in N:M structured sparsity (e.g. 2).
        prunem: M in N:M structured sparsity (e.g. 4).
        nsamples: Number of calibration sequences.
        seed: Random seed for sample extraction.
        percdamp: Percentage dampening factor for Hessian regularization.
        blocksize: Column blocksize for Lazy OBS updates.
        save_path: Output directory to serialize sparsified weights and configs.
        dev: Device string ('cpu' or 'cuda'/'tpu').
        sparse_layer_indices: Optional explicit set of layer indices to prune.
        use_lrc: Whether to apply Low-Rank Compression alternating minimization.
        lrc_rank: Low-rank rank k for LRC approximation.
        lrc_iters: Alternating minimization iterations T for LRC.
        save_prefix24_path: Optional path to save prefix-24 intermediate checkpoint.
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
    args.sparse_layer_indices = sparse_layer_indices
    args.use_lrc = use_lrc
    args.lrc_rank = lrc_rank
    args.lrc_iters = lrc_iters
    args.save_prefix24_path = save_prefix24_path
    args.model_path = model_path
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
