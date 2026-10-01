"""
gemma4.py
---------
Structured and unstructured SparseGPT pruning implementation for Google Gemma 4 models.

Attribution Notice:
    Adapted from SparseGPT (Frantar & Alistarh, 2023 - https://github.com/IST-DASLab/sparsegpt).
    Modifications for Gemma 4 architecture include:
      - Support for Gemma 4 hybrid attention (full attention and sliding window causal masks).
      - Rotary embedding caching across distinct unique layer types.
      - Support for shared KV / MLA projection weights.
      - Sequential layer-by-layer calibration propagation with detailed progress tracking.
"""

import time

import torch
import torch.nn as nn
from transformers import Gemma4ForConditionalGeneration

from .modelutils import find_layers
from .sparsegpt import SparseGPT


def get_gemma4(model_path: str) -> Gemma4ForConditionalGeneration:
    """Loads a Gemma 4 model with uninitialized random weight hooks disabled for fast instantiation."""
    print(f">>> [MODEL LOADER] Loading Gemma 4 from {model_path}...")

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.seqlen = 2048
    return model


@torch.no_grad()
def capture_layer0_inputs(model, dataloader, dev, nsamples):
    """Collect the inputs of decoder layer 0 for every calibration sample.

    Runs the model on each batch with ``layers[0]`` replaced by a catcher that
    records its input and aborts, so only the embedding stage executes.

    Returns ``(lm, inps, position_embeddings)``: the text stack, a
    ``(nsamples, seqlen, hidden)`` tensor on ``dev``, and the RoPE
    ``(cos, sin)`` tables keyed by layer type. Every calibrator (CPU or TPU)
    starts from this same state.
    """
    lm = model.model.language_model
    layers = lm.layers
    embed_tokens = lm.embed_tokens.to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.text_config.hidden_size
    seqlen = model.seqlen

    print(
        f">>> [CALIBRATION] Collecting initial token embeddings for {nsamples} samples..."
    )
    inps = torch.zeros((nsamples, seqlen, hidden_size),
                       dtype=dtype,
                       device=dev)
    position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps[self.cur_sample] = inp
            self.cur_sample += 1
            raise ValueError

    catcher = Catcher(layers[0])
    catcher.cur_sample = 0
    layers[0] = catcher

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = catcher.module
    layers[0] = layers[0].cpu()
    embed_tokens = embed_tokens.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Precompute RoPE position embeddings for each unique attention layer type
    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(
            inps[0:1], position_ids, layer_type)
    return lm, inps, position_embeddings


@torch.no_grad()
def gemma4_sequential(model, dataloader, dev, args, prune_fn=None):
    """
    Performs layer-by-layer sequential SparseGPT pruning on Gemma 4.
    Passes calibration activations through each layer, captures empirical Hessians,
    and applies second-order error compensation on all unpruned weights.

    Equivalent to ``capture_layer0_inputs`` followed by ``calibrate_layers_cpu``.
    """
    total_layers = len(model.model.language_model.layers)
    sparsity_ratio = f"{args.prunen}:{args.prunem}" if args.prunem > 0 else f"{args.sparsity * 100:.1f}%"
    print("=" * 85)
    print(">>> [SPARSITY ENGINE] Starting Gemma 4 Sequential Pruning")
    print(
        f"    • Structured Sparsity: {sparsity_ratio} ({args.prunen} kept / {args.prunem} block)"
    )
    print(
        f"    • Total Layers:        {total_layers} (pruning layers [{args.minlayer} .. {args.maxlayer}-1])"
    )
    print(f"    • Calibration Samples: {args.nsamples} sequences")
    print(f"    • Device:              {dev}")
    print("=" * 85)

    lm, inps, position_embeddings = capture_layer0_inputs(
        model, dataloader, dev, args.nsamples)
    calibrate_layers_cpu(lm, inps, position_embeddings, args, dev, prune_fn=prune_fn)


@torch.no_grad()
def calibrate_layers_cpu(lm, inps, position_embeddings, args, dev, prune_fn=None):
    """SparseGPT layer loop in PyTorch (the reference implementation).

    For each layer in order: accumulate the input Hessian of every linear with
    forward hooks (``SparseGPT.add_batch``), prune in place
    (``SparseGPT.fasterprune`` or custom ``prune_fn``) if the layer is selected, then
    propagate the calibration activations through the (now pruned) layer.

    Calibrator contract (shared with ``tpu_gemma4.calibrate_layers_tpu``):
    ``lm`` is the text stack with weights on CPU, ``inps`` the layer-0 inputs
    from ``capture_layer0_inputs``, ``position_embeddings`` the RoPE tables,
    ``args`` a ``PruneSpec``-like namespace. Weights are modified in place.
    """
    overall_start_time = time.time()
    layers = lm.layers
    total_layers = len(layers)
    sparsity_ratio = f"{args.prunen}:{args.prunem}" if args.prunem > 0 else f"{args.sparsity * 100:.1f}%"
    seqlen = inps.shape[1]
    position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)
    outs = torch.zeros_like(inps)

    from transformers.models.gemma4.modeling_gemma4 import (
        create_causal_mask, create_sliding_window_causal_mask)
    causal_mask_mapping = {
        "full_attention":
        create_causal_mask(
            config=lm.config,
            inputs_embeds=inps[0:1],
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        ),
        "sliding_attention":
        create_sliding_window_causal_mask(
            config=lm.config,
            inputs_embeds=inps[0:1],
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        ),
    }
    for k in causal_mask_mapping:
        if causal_mask_mapping[k] is not None and torch.is_tensor(
                causal_mask_mapping[k]):
            causal_mask_mapping[k] = causal_mask_mapping[k].to(dev)

    # Process each Transformer decoder layer sequentially
    for i in range(len(layers)):
        layer_start_time = time.time()
        layer_type = lm.config.layer_types[i]
        layer = layers[i].to(dev)
        progress_pct = (i / total_layers) * 100.0

        pos_emb = position_embeddings[layer_type]
        attn_mask = causal_mask_mapping[layer_type]

        subset = find_layers(layer)
        gpts = {}

        sparse_indices = getattr(args, "sparse_layer_indices", None)
        is_pruned = (i in sparse_indices) if sparse_indices is not None else (
            args.minlayer <= i < args.maxlayer)
        status_tag = f"PRUNING ({sparsity_ratio} SparseGPT)" if is_pruned else "DENSE (Preserved)"
        print(
            f"\n>>> [Layer {i:02d}/{total_layers:02d} - {progress_pct:4.1f}%] Type: {layer_type:<18} | Status: {status_tag}"
        )

        if is_pruned:
            for name in subset:
                if args.prune_only and args.prune_only not in name:
                    continue
                gpts[name] = SparseGPT(subset[name])

            # Hook layer input activations (inp[0].data is X in paper) to update Hessian.
            # When layer(...) runs, this hook calls gpts[name].add_batch(), where
            # gpts[name].H is updated via: H += (sqrt(2/nsamples) * X) @ (sqrt(2/nsamples) * X)^T
            def add_batch(name):

                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(
                    add_batch(name)))

            # Step 1: Forward calibration batches (inps[j] is X).
            # The forward hooks fire during this loop, accumulating H = 2*X*X^T into gpts[name].H.
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    position_embeddings=pos_emb,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    shared_kv_states={},
                )[0]

            for h in handles:
                h.remove()

            # Step 2: Prune each Linear layer in place using the accumulated
            # empirical Hessian gpts[name].H (defaults to SparseGPT OBS solver
            # fasterprune; callers such as LRC can pass a custom prune_fn).
            matrix_pruner = prune_fn or getattr(args, "prune_fn", None)

            for name in subset:
                if name in gpts:
                    mat_start = time.time()
                    w_shape = list(subset[name].weight.shape)
                    print(f"    ├─ [{name:<24}] shape {str(w_shape):<16} ...",
                          end="",
                          flush=True)
                    if matrix_pruner is not None:
                        matrix_pruner(gpts[name])
                    else:
                        gpts[name].fasterprune(
                            args.sparsity,
                            prunen=args.prunen,
                            prunem=args.prunem,
                            percdamp=args.percdamp,
                            blocksize=args.blocksize,
                        )
                    gpts[name].free()
                    mat_duration = time.time() - mat_start
                    print(f" done in {mat_duration:.2f}s")

        # Step 3 (Sequential layer-by-layer propagation):
        # Forward inps through the newly pruned layer to get outs (distorted X_{l+1}).
        # Below, inps, outs = outs, inps feeds this into layer l+1 so downstream
        # layers accumulate Hessians on the actual distorted activations.
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=position_ids,
                shared_kv_states={},
            )[0]

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

        layer_duration = time.time() - layer_start_time
        elapsed_total = time.time() - overall_start_time
        avg_per_layer = elapsed_total / (i + 1)
        remaining_layers = total_layers - (i + 1)
        eta_seconds = avg_per_layer * remaining_layers
        print(
            f"    └─ Layer {i:02d} completed in {layer_duration:.2f}s (Elapsed: {elapsed_total/60:.1f}m | ETA: {eta_seconds/60:.1f}m)"
        )

    total_time = time.time() - overall_start_time
    print("\n" + "=" * 85)
    print(
        f">>> [SPARSITY ENGINE] Sparsification of all {total_layers} layers complete in {total_time/60:.2f} minutes!"
    )
    print("=" * 85)
