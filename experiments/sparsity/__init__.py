"""
sparsity package
----------------
Structured and unstructured weight pruning algorithms, Hessian estimation utilities,
and SparseGPT calibration for Gemma 4 on CPU (reference) or TPU (JAX).
"""

import dataclasses
import importlib
import os
import shutil
import time
from typing import Any, Callable, FrozenSet, Optional, Set

import torch
from transformers import AutoTokenizer

from .datautils import get_loaders
from .lrc import LrcResult, lrc_prune, make_lrc_pruner
from .modelutils import find_layers
from .quant import Quantizer
from .sparsegpt import SparseGPT


@dataclasses.dataclass(frozen=True)
class PruneSpec:
    """What to prune and how; the contract every calibrator receives.

    ``sparse_layer_indices`` wins over ``[minlayer, maxlayer)`` when given.
    ``sparsity`` is only used when ``prunen == 0`` (unstructured pruning).
    """
    prunen: int = 2
    prunem: int = 4
    sparsity: float = 0.0
    percdamp: float = 0.01
    blocksize: int = 128
    nsamples: int = 64
    minlayer: int = 0
    maxlayer: int = 60
    sparse_layer_indices: Optional[FrozenSet[int]] = None
    prune_only: str = ""
    prune_fn: Optional[Callable[[SparseGPT], Any]] = None

    def layer_set(self, total_layers: int) -> Set[int]:
        idx = (set(self.sparse_layer_indices) if self.sparse_layer_indices
               is not None else set(range(self.minlayer, self.maxlayer)))
        return {i for i in idx if 0 <= i < total_layers}


# A calibrator is ``f(lm, inps, position_embeddings, spec, dev) -> None`` that
# prunes ``lm`` in place, starting from the layer-0 inputs captured by
# ``gemma4.capture_layer0_inputs``. Resolved lazily: the TPU one imports JAX,
# which CPU-only callers must never load.
CALIBRATORS = {
    "cpu": ("sparsity.gemma4", "calibrate_layers_cpu"),
    "tpu": ("sparsity.tpu_gemma4", "calibrate_layers_tpu"),
}


def get_calibrator(name: str) -> Callable:
    if name not in CALIBRATORS:
        raise ValueError(
            f"unknown calibrator {name!r}; choose from {sorted(CALIBRATORS)}")
    module, fn = CALIBRATORS[name]
    return getattr(importlib.import_module(module), fn)


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
                 prune_fn: Optional[Callable[[SparseGPT], Any]] = None,
                 calibrator: str = "cpu") -> str:
    """
    Main entry point for SparseGPT pruning on Gemma 4 architectures.
    Captures layer activations sequentially, calculates second-order Hessian matrices,
    and applies N:M structured sparsity, then saves a checkpoint for serving.

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
        dev: Device for the model and the CPU calibrator ('cpu' or 'cuda').
        sparse_layer_indices: Optional explicit set of layer indices to prune.
        prune_fn: Optional per-layer matrix solver callback ``prune_fn(gpt)``
            (defaults to ``SparseGPT.fasterprune``).
        calibrator: Key of ``CALIBRATORS``: "cpu" (PyTorch reference) or
            "tpu" (JAX; holds the TPU until the process exits, so callers
            that start vLLM afterwards run it in a child process).
    """
    from .gemma4 import capture_layer0_inputs, get_gemma4

    calibrate = get_calibrator(calibrator)  # fail before loading 60 GB
    device = torch.device(dev)
    print(f">>> [SPARSITY ENGINE] Pruning Gemma 4 with the {calibrator} "
          f"calibrator (model on {device})")
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
    spec = PruneSpec(
        prunen=prunen,
        prunem=prunem,
        percdamp=percdamp,
        blocksize=blocksize,
        nsamples=nsamples,
        minlayer=minlayer,
        maxlayer=maxlayer,
        sparse_layer_indices=(frozenset(int(i) for i in sparse_layer_indices)
                              if sparse_layer_indices is not None else None),
        prune_fn=prune_fn,
    )

    # 3. Capture layer-0 inputs once, then run the calibrator over the layers
    tick = time.time()
    lm, inps, position_embeddings = capture_layer0_inputs(
        model, dataloader, device, nsamples)
    calibrate(lm, inps, position_embeddings, spec, device)
    tock = time.time()
    print(f">>> [SPARSITY ENGINE] Pruning finished in {tock - tick:.2f}s")

    # 4. Save sparsified model and tokenizer to disk
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(
            f">>> [SPARSITY ENGINE] Saving pruned checkpoint to {save_path}..."
        )
        base_dir = model_path
        if not os.path.exists(base_dir):
            try:
                from core.model_utils import resolve_model_path
                base_dir = resolve_model_path(model_path)
            except Exception:
                base_dir = model_path
        has_safetensors = os.path.exists(base_dir) and (os.path.exists(
            os.path.join(base_dir, "model.safetensors.index.json")) or any(
                f.endswith(".safetensors") for f in os.listdir(base_dir)))
        if has_safetensors:
            from resolvers.model_splicer import ModelSplicer
            sparse_indices = spec.layer_set(len(lm.layers))
            ModelSplicer.save_substituted_checkpoint(
                base_model_dir=base_dir,
                output_dir=save_path,
                replacement_state_dict=model.state_dict(),
                target_layer_indices=sparse_indices,
            )
            ModelSplicer.assert_checkpoint_integrity(base_dir, save_path)
        else:
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
    "LrcResult",
    "lrc_prune",
    "make_lrc_pruner",
    "PruneSpec",
    "CALIBRATORS",
    "get_calibrator",
    "find_layers",
    "Quantizer",
    "get_loaders",
    "prune_gemma4",
]
