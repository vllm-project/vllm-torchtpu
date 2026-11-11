# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Universal vLLM model adapter implementing the `ExportableTextOnlyLLM` protocol."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import itertools
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from vllm_torchtpu import rope_patch
from vllm_torchtpu.experimental.export import (
    api,
    draft_model,
    sampling,
    tpu_lowering,
    weight_mapper,
)
from vllm_torchtpu.experimental.export.trace import trace
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

# The vLLM TPU runner's smallest request padding (`tpu_runner.MIN_NUM_SEQS`).
_MIN_REQ_BUCKET = 8


def default_req_buckets(max_num_seqs: int) -> tuple[int, ...]:
    """Request buckets as the vLLM TPU runner pads `num_reqs` (`_get_req_paddings`).

    Powers of two from 8 up to, and always ending with, `max_num_seqs`, so a batch is
    padded to at most twice its size. A fixed list with a jump to `max_num_seqs`
    (8, 16, 32, 64, 512) would run the LM head and the sampler on 512 rows for any
    batch above 64.
    """
    buckets = []
    size = _MIN_REQ_BUCKET
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    buckets.append(max_num_seqs)
    return tuple(buckets)


def make_attention_metadata(
    *,
    input_positions: Any,
    block_tables: Any = None,
    seq_lens: Any = None,
    query_start_loc: Any = None,
    request_distribution: Any = None,
    mamba_state_indices: Any = None,
    mamba_slot_read_offsets: Any = None,
    mamba_ckpt_indices: Any = None,
    mamba_request_distribution: Any = None,
) -> AttentionMetadata:
    """Constructs an `AttentionMetadata` instance directly from control tensors."""
    return AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
        mamba_state_indices=mamba_state_indices,
        mamba_slot_read_offsets=mamba_slot_read_offsets,
        mamba_ckpt_indices=mamba_ckpt_indices,
        mamba_request_distribution=mamba_request_distribution,
    )


def _model_config_value(model_cfg: Any, accessor: str) -> Any:
    """Calls `model_cfg.<accessor>()` if vLLM's ModelConfig provides it."""
    fn = getattr(model_cfg, accessor, None)
    return fn() if callable(fn) else None


def _extract_hf_field(hf_config: Any, *keys: str, default: Any = None) -> Any:
    """Reads the first present attribute or dict entry from `hf_config`."""
    if hf_config is None:
        return default
    text_cfg = getattr(hf_config, "text_config", None)
    for target in (hf_config, text_cfg):
        if target is None:
            continue
        for k in keys:
            if isinstance(target, dict) and k in target and target[k] is not None:
                return target[k]
            val = getattr(target, k, None)
            if val is not None:
                return val
    return default


def _min_smem_page_size(
    parallel_cfg: Any, max_num_seqs: int, max_model_len: int
) -> int:
    """Smallest power-of-two page size whose block table fits half of SMEM.

    Mirrors `get_tpu_min_page_size`, which serving applies to the block size
    once the model is loaded; the exporter builds its config before that, so
    it takes the already-resolved values rather than a full `VllmConfig`.
    """
    if int(getattr(parallel_cfg, "prefill_context_parallel_size", 1) or 1) > 1:
        return 1
    max_pages_per_req = 1024 * 1024 // 2 // max_num_seqs // 4
    dcp = int(getattr(parallel_cfg, "decode_context_parallel_size", 1) or 1)
    local_max_model_len = -(-max_model_len // dcp)
    min_page = -(-local_max_model_len // max_pages_per_req)
    return 1 << (min_page - 1).bit_length()


def _kernel_block_size(vllm_config: Any, block_size: int) -> int:
    """The page size the attention kernels run at for KV blocks of `block_size`.

    Mirrors the runner's `prepare_kernel_block_sizes`: a hybrid model's unified
    block pool sizes its blocks to hold a recurrent state (e.g. 1152 tokens), and
    the runner splits each into pages the attention backend supports (128 or 256
    for RPA) before the kernel sees them. The export keeps recurrent states in
    their own pools, so it pages its KV caches at the kernel's size directly; the
    RPA kernel does not reject an unsupported page size, it computes wrong
    attention.
    """
    from vllm.model_executor.layers.attention import Attention

    ctx = getattr(
        getattr(vllm_config, "compilation_config", None), "static_forward_context", {}
    )
    backends = list(
        {
            layer.attn_backend
            for layer in (ctx or {}).values()
            if isinstance(layer, Attention) and getattr(layer, "attn_backend", None)
        }
    )
    if not backends:
        return block_size
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.utils import select_common_block_size

    # Backends read their supported sizes from the current config.
    with set_current_vllm_config(vllm_config):
        return select_common_block_size(block_size, backends)


def check_supported_parallelism(parallel_cfg: Any) -> tuple[int, int]:
    """Returns `(dp, pp)` after rejecting data or pipeline parallel sizes above 1.

    Takes a `ParallelConfig` or the parsed CLI arguments.
    """
    dp_size = int(getattr(parallel_cfg, "data_parallel_size", 1) or 1)
    pp_size = int(getattr(parallel_cfg, "pipeline_parallel_size", 1) or 1)
    if (dp_size, pp_size) != (1, 1):
        raise NotImplementedError(
            f"export supports only data/pipeline parallel size 1, got "
            f"dp={dp_size} pp={pp_size}"
        )
    return dp_size, pp_size


def resolve_hf_checkpoint_dir(model: str | None) -> str | None:
    """Returns the local checkpoint directory for a path or a Hub repo id.

    The companion files (`exporter.HF_COMPANION_FILES`: `config.json`,
    `tokenizer.json`, `generation_config.json`, `tokenizer_config.json`) and the
    safetensors metadata are read from this directory. For a repo id the
    companion files are fetched into the Hub cache first: loading dummy weights
    caches `config.json` alone, so the tokenizer would otherwise be missing. A
    repo without `config.json` is an error rather than a silently thinner
    artifact.
    """
    if not model or os.path.isdir(model):
        return model
    from transformers.utils import cached_file

    from vllm_torchtpu.experimental.export import exporter

    hint = (
        f"--model {model!r} is neither a local directory nor a HuggingFace Hub "
        "repo with a config.json; pass the checkpoint directory"
    )
    try:
        config_path = cached_file(model, "config.json")
    except OSError as e:
        raise FileNotFoundError(hint) from e
    if config_path is None:
        raise FileNotFoundError(hint)
    # The exporter reports a missing tokenizer.json or tokenizer_config.json.
    for name in exporter.HF_COMPANION_FILES:
        if name != "config.json":
            cached_file(model, name, _raise_exceptions_for_missing_entries=False)
    return os.path.dirname(config_path)


def _detect_target_tpu() -> str:
    """The TPU generation of this host, as `ChipVersion` names it ('7x', 'v6e').

    The artifact's kernels are tuned for one generation, so an export that
    cannot tell which one it runs on fails instead of guessing.
    """
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu import tpu_info

    accelerator_type = tpu_info.get_tpu_type()
    if not accelerator_type:
        raise RuntimeError(
            "Cannot detect the TPU generation (neither TPU_ACCELERATOR_TYPE nor "
            "the GCE metadata accelerator-type is available); pass --target-tpu."
        )
    name = accelerator_type.strip().lower().split("-")[0].removeprefix("tpu")
    name = {"v5litepod": "v5e", "v7x": "7x"}.get(name, name)
    try:
        return str(pltpu.ChipVersion(name).value)
    except ValueError as e:
        raise RuntimeError(
            f"Unrecognized TPU accelerator type {accelerator_type!r}; "
            "pass --target-tpu."
        ) from e


@trace("export.build_export_config")
def build_export_config_from_vllm(
    vllm_config: Any,
    *,
    model_name: str | None = None,
    hf_checkpoint_dir: str | None = None,
    target_tpu: str | None = None,
    token_buckets: Sequence[int] | None = None,
    req_buckets: Sequence[int] | None = None,
    max_verify_tokens: int = 0,
    num_speculative_tokens: int = 0,
) -> api.TextOnlyLLMExportConfig:
    """Derives a `TextOnlyLLMExportConfig` from a `VllmConfig` (or mock config).

    With `max_verify_tokens`, `vllm_config`'s speculative config is the one that
    makes vLLM keep that many recurrent checkpoints
    (`verify_window_speculative_config`), and `num_speculative_tokens` is what the
    export's drafter proposes, if any.
    """
    model_cfg = getattr(vllm_config, "model_config", None)
    cache_cfg = getattr(vllm_config, "cache_config", None)
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    hf_cfg = getattr(model_cfg, "hf_config", None)

    resolved_model_name = (
        model_name
        or getattr(model_cfg, "model", None)
        or _extract_hf_field(hf_cfg, "_name_or_path", "model_type", default="model")
    )
    resolved_hf_dir = resolve_hf_checkpoint_dir(
        hf_checkpoint_dir or getattr(model_cfg, "model", None)
    )

    tp_size = int(getattr(parallel_cfg, "tensor_parallel_size", 1) or 1)
    dp_size, pp_size = check_supported_parallelism(parallel_cfg)

    if target_tpu is None:
        target_tpu = _detect_target_tpu()

    vocab_size = int(
        (
            getattr(model_cfg, "get_vocab_size", lambda: None)()
            if hasattr(model_cfg, "get_vocab_size")
            else None
        )
        or _extract_hf_field(hf_cfg, "vocab_size", default=151936)
    )
    hidden_size = int(
        (
            getattr(model_cfg, "get_hidden_size", lambda: None)()
            if hasattr(model_cfg, "get_hidden_size")
            else None
        )
        or _extract_hf_field(hf_cfg, "hidden_size", default=1024)
    )
    # vLLM's ModelConfig resolves fields a checkpoint declares per layer (e.g.
    # Gemma 4's KV heads and head_dim), which transformers refuses to read as
    # one global value, so its accessors come before the raw HF fields.
    num_layers = int(
        _model_config_value(model_cfg, "get_total_num_hidden_layers")
        or _extract_hf_field(
            hf_cfg, "num_hidden_layers", "num_layers", "n_layer", default=28
        )
    )
    num_q_heads = int(
        _extract_hf_field(
            hf_cfg, "num_attention_heads", "num_heads", "n_head", default=16
        )
    )
    num_kv_heads = int(
        _model_config_value(model_cfg, "get_total_num_kv_heads")
        or _extract_hf_field(
            hf_cfg,
            "num_key_value_heads",
            "num_kv_heads",
            default=num_q_heads,
        )
    )
    head_dim = int(
        (
            getattr(model_cfg, "get_head_size", lambda: None)()
            if hasattr(model_cfg, "get_head_size")
            else None
        )
        or _extract_hf_field(
            hf_cfg,
            "head_dim",
            default=max(1, hidden_size // max(1, num_q_heads)),
        )
    )
    block_size = int(getattr(cache_cfg, "block_size", 16) or 16)
    num_blocks = int(
        getattr(cache_cfg, "num_gpu_blocks", None)
        or getattr(cache_cfg, "num_gpu_blocks_override", None)
        or 512
    )
    max_num_seqs = int(getattr(sched_cfg, "max_num_seqs", 64) or 64)
    max_model_len = int(
        getattr(model_cfg, "max_model_len", None)
        or _extract_hf_field(hf_cfg, "max_position_embeddings", default=2048)
    )
    block_size = max(
        block_size, _min_smem_page_size(parallel_cfg, max_num_seqs, max_model_len)
    )
    kernel_block_size = _kernel_block_size(vllm_config, block_size)
    num_blocks *= block_size // kernel_block_size
    block_size = kernel_block_size

    raw_eos = _extract_hf_field(hf_cfg, "eos_token_id", default=151645)
    if isinstance(raw_eos, (list, tuple)):
        eos_token_id = int(raw_eos[0]) if raw_eos else 151645
    else:
        eos_token_id = int(raw_eos)

    raw_dtype = getattr(model_cfg, "dtype", None) or "bf16"
    dtype_str = tpu_lowering.dtype_name(raw_dtype)

    if token_buckets is None:
        comp_sizes = getattr(comp_cfg, "compile_sizes", None)
        if comp_sizes:
            token_buckets = tuple(sorted(int(s) for s in comp_sizes))
        else:
            token_buckets = (16, 32, 64, 128, 256)
    else:
        token_buckets = tuple(int(b) for b in token_buckets)

    if req_buckets is None:
        # A speculative step runs the LM head on every request's verify rows, its
        # drafts and the token after them, as vLLM's sampler does; they are some of
        # the step's tokens, so no more than its largest token bucket.
        spec = getattr(vllm_config, "speculative_config", None)
        drafted = num_speculative_tokens
        if spec is not None and not max_verify_tokens:
            drafted = spec.num_speculative_tokens
        # Tokens verified beyond the drafter's are forced, and skip the LM head.
        rows = max_num_seqs
        if drafted:
            verify_rows = max_num_seqs * (drafted + 1)
            rows = max(rows, min(verify_rows, max(token_buckets)))
        req_buckets = default_req_buckets(rows)
    else:
        req_buckets = tuple(int(r) for r in req_buckets)

    rope_scaling = _extract_hf_field(
        hf_cfg, "rope_scaling", "rope_parameters", default=None
    )
    uses_mrope = bool(getattr(model_cfg, "uses_mrope", False)) or (
        isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling
    )

    return api.TextOnlyLLMExportConfig(
        model_name=str(resolved_model_name),
        target_tpu=str(target_tpu),
        num_replicas=1,
        num_partitions=tp_size,
        use_spmd_partitioning=True,
        data_parallel_size=dp_size,
        pipeline_parallel_size=pp_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        eos_token_id=eos_token_id,
        dtype=dtype_str,
        token_buckets=token_buckets,
        req_buckets=req_buckets,
        emit_byte_tokenizer=False,
        hf_checkpoint_dir=str(resolved_hf_dir) if resolved_hf_dir else None,
        uses_mrope=uses_mrope,
        max_verify_tokens=max_verify_tokens,
    )


def _create_single_device_tpu_mesh() -> Any:
    """Creates a 1-device mesh on `axis_names=('model',)` for Pallas kernels."""
    with contextlib.suppress(Exception):
        import jax
        import numpy as np
        from jax.sharding import Mesh

        local_devices = list(jax.local_devices())
        if local_devices:
            return Mesh(
                np.asarray(local_devices[:1]).reshape((1,)),
                axis_names=("model",),
            )
    return None


def init_export_distributed_environment(vllm_config: Any) -> None:
    """Joins the process group the export launcher set up for this rank.

    The launcher (`cli.main`) starts one worker per tensor-parallel rank and sets
    `RANK`, `LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT`; a single-device export
    is the one-worker case.
    """
    from vllm.distributed import init_distributed_environment

    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    world_size = vllm_config.parallel_config.tensor_parallel_size
    init_distributed_environment(
        world_size=world_size,
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        distributed_init_method="env://",
        backend=TpuPlatform.get_worker_distributed_backend(world_size),
    )


def run_model_loader(vllm_config: Any) -> nn.Module:
    """Builds the model and loads its weights with vLLM's loader for `load_config`.

    Logged as an `export.load_safetensors` span (`export.load_dummy_weights` for
    dummy weights): reading a large checkpoint is often the longest stage.
    """
    from vllm.model_executor.model_loader import get_model_loader

    load_format = vllm_config.load_config.load_format
    span = (
        "export.load_dummy_weights"
        if load_format == "dummy"
        else "export.load_safetensors"
    )
    with trace(span):
        loader = get_model_loader(vllm_config.load_config)
        # `@instrument` on `load_model` hides its signature from type checkers.
        load_model: Any = loader.load_model
        model: nn.Module = load_model(
            vllm_config=vllm_config, model_config=vllm_config.model_config
        )
    return model


def load_dummy_vllm_model(vllm_config: Any) -> nn.Module:
    """Loads a vLLM model with `load_format='dummy'` on TPU (zero disk weight IO)."""
    return load_vllm_model(vllm_config, load_format="dummy")


@trace("export.create_vllm_config")
def create_export_vllm_config(args: Any) -> Any:
    """Creates the `VllmConfig` an export runs with from parsed `EngineArgs`.

    Prefix caching is off for hybrid recurrent models: the artifact keeps their
    recurrent state per layer, which vLLM only prefix-caches in its unified KV
    pool, and the serving runtime does its own prefix caching.
    """
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.create_model_config().is_hybrid:
        engine_args.enable_prefix_caching = False
    return engine_args.create_engine_config()


def load_vllm_model(vllm_config: Any, *, load_format: str) -> nn.Module:
    """Loads a vLLM model with `load_format`, running vLLM's weight processing.

    The model is placed on TPU when one is available; on a host without one
    (`PJRT_DEVICE=CPU`) it stays on CPU.
    """
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    vllm_config.load_config.load_format = load_format
    if getattr(vllm_config, "device_config", None) is not None:
        with contextlib.suppress(Exception):
            vllm_config.device_config.device = torch.device("tpu")

    with set_current_vllm_config(vllm_config):
        if not dist.is_initialized():
            init_export_distributed_environment(vllm_config)
        if not model_parallel_is_initialized():
            ensure_model_parallel_initialized(
                vllm_config.parallel_config.tensor_parallel_size, 1
            )

        mesh = _create_single_device_tpu_mesh()
        with contextlib.ExitStack() as stack:
            with contextlib.suppress(Exception):
                from vllm_torchtpu.layers.adapter.quantization import (
                    get_tpu_quantization_config,
                )
                from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
                    set_vllm_model_wrapper_context,
                )

                vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
                stack.enter_context(
                    set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_config)
                )

            model = run_model_loader(vllm_config)
            draft = _load_mtp_draft(vllm_config, model)
            separate_draft = draft_model.load_draft_model(vllm_config, model, mesh)
            # Same step vLLM's executors run right after `load_model`: it picks
            # the KV block size from the model's attention backend, which serving
            # uses and which keeps the block table within SMEM.
            from vllm.platforms import current_platform

            current_platform.update_block_size_for_backend(vllm_config)

            with contextlib.suppress(Exception):
                from vllm_torchtpu.layers.adapter.attention import (
                    PallasAttentionBackendImpl,
                )

                layers = vllm_config.compilation_config.static_forward_context
                for attn_layer in layers.values():
                    if isinstance(attn_layer, AttentionLayerBase):
                        impl = getattr(attn_layer, "impl", None)
                        if isinstance(impl, PallasAttentionBackendImpl):
                            impl.initialize_kernel(attn_layer)  # type: ignore[arg-type]

            model._export_tpu_mesh = mesh  # type: ignore[attr-defined]
            if draft is not None:
                # Kept off the target's module tree, so its parameters are not
                # the target's.
                object.__setattr__(model, MTP_DRAFT_ATTR, draft)
            if separate_draft is not None:
                object.__setattr__(model, draft_model.DRAFT_MODEL_ATTR, separate_draft)
            return model


# The attribute of a loaded target model holding its MTP draft, if it has one.
MTP_DRAFT_ATTR = "_export_mtp_draft"


def _load_mtp_draft(vllm_config: Any, target: nn.Module) -> nn.Module | None:
    """Loads the MTP draft `vllm_config.speculative_config` declares, or None.

    Built as vLLM's MTP proposer builds it, from the speculative config's draft
    model config and the same loader, so it reads the checkpoint's `mtp.*`
    weights, or the separate checkpoint the config names (Gemma 4's assistant).
    It then shares the target's embedding and, unless it has its own, LM head,
    as the serving runtime binds them, so the artifact stores those once; and
    a draft whose attention reads the target's KV is wired to it.
    """
    import copy

    from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.model_loader import get_model_loader

    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "mtp":
        return None
    target_attn_layer_names = set(
        get_layers_from_vllm_config(vllm_config, AttentionLayerBase)  # type: ignore[type-abstract]
    )
    draft_vllm_config = copy.copy(vllm_config)
    draft_model_config = copy.copy(spec.draft_model_config)
    draft_model_config.runner_type = "draft"
    with trace("export.load_mtp_draft"), set_current_vllm_config(draft_vllm_config):
        loader = cast(
            Callable[..., torch.nn.Module],
            get_model_loader(vllm_config.load_config).load_model,
        )
        draft = loader(vllm_config=draft_vllm_config, model_config=draft_model_config)
    _share_embedding_and_lm_head(draft, target)
    _share_target_kv_cache(draft, target, target_attn_layer_names)
    return draft


def _share_embedding_and_lm_head(draft: nn.Module, target: nn.Module) -> None:
    """Points the draft's embedding and LM head at the target's modules, as
    the TPU runner's MTP proposer does (`maybe_share_embeddings`,
    `maybe_share_lm_head`): a draft declaring `has_own_lm_head` (Gemma 4's
    assistant) keeps its own head. The draft runs sharded on the target's
    tensor-parallel ranks (the CLI requires it), so their layouts match and
    the draft's logits processor gathers over the group it is sharded on.
    """
    from vllm_torchtpu.spec_decode.utils import (
        maybe_share_embeddings,
        maybe_share_lm_head,
    )

    maybe_share_embeddings(draft, target, tp_layout_matches=True)
    maybe_share_lm_head(
        draft,
        target,
        draft_replicated=False,
        tp_layout_matches=True,
        materialize_if_mismatched=False,
    )


def _share_target_kv_cache(
    draft: Any, target: Any, target_attn_layer_names: set[str]
) -> None:
    """Wires the draft's Q-only attention layers to read the target's KV cache.

    A copy of the TPU runner's `Eagle3Proposer._setup_gemma4_kv_sharing`,
    which a parity test holds it to: each draft layer declaring
    `is_kv_shared_layer` (Gemma 4's assistant) reads the last target layer of
    its attention type among those keeping their own KV, and its kernel is
    built without the KV write (`skip_kv_update`). A no-op for an MTP draft
    that keeps its own KV.
    """
    from collections import defaultdict

    if not (hasattr(draft, "model") and hasattr(draft.model, "layers")):
        return
    shared_layers = [
        (idx, layer)
        for idx, layer in enumerate(draft.model.layers)
        if getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False)
    ]
    if not shared_layers:
        return

    draft_text_config = draft.config.get_text_config()
    target_text_config = target.config.get_text_config()
    target_layer_types = getattr(target_text_config, "layer_types", [])
    target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
    num_non_shared = len(target_layer_types) - target_num_kv_shared
    type_to_target_indices: dict[str, list[int]] = defaultdict(list)
    for idx, lt in enumerate(target_layer_types[:num_non_shared]):
        type_to_target_indices[lt].append(idx)

    target_prefix = "model.layers"
    for name in sorted(target_attn_layer_names):
        if ".layers." in name:
            target_prefix = name.split(".layers.")[0] + ".layers"
            break

    draft_layer_types = getattr(draft_text_config, "layer_types", [])
    for draft_idx, layer in shared_layers:
        attn = getattr(layer.self_attn, "attn", None)
        if attn is None:
            raise RuntimeError(
                f"Draft layer {draft_idx} declares is_kv_shared_layer but "
                "exposes no self_attn.attn to wire; cannot guarantee it "
                "will not write into the target's KV cache."
            )
        draft_layer_type = (
            draft_layer_types[draft_idx]
            if draft_idx < len(draft_layer_types)
            else "full_attention"
        )
        candidates = type_to_target_indices.get(draft_layer_type, [])
        if not candidates:
            raise RuntimeError(
                f"No target layer of type '{draft_layer_type}' to share KV "
                f"with for draft layer {draft_idx}. Leaving it unwired "
                "would let it write dummy K/V into the target's cache."
            )
        target_layer_name = f"{target_prefix}.{candidates[-1]}.self_attn.attn"
        attn.kv_sharing_target_layer_name = target_layer_name
        impl = getattr(attn, "impl", None)
        if impl is None:
            raise RuntimeError(
                f"Draft attention layer {draft_idx} has no `impl`; cannot "
                "enable skip_kv_update for cross-model KV sharing."
            )
        impl.kv_sharing_target_layer_name = target_layer_name


def mtp_draft_names(model: nn.Module) -> dict[str, str]:
    """The export name of each parameter and buffer of `model`'s MTP draft.

    The weights the draft shares with the target keep the target's names; its
    own are named as in the checkpoint, `mtp.` for the draft's `model.`. Empty
    for a model without a draft.
    """
    return _draft_export_names(model, MTP_DRAFT_ATTR, "mtp.")


def draft_model_names(model: nn.Module) -> dict[str, str]:
    """The export name of each parameter and buffer of `model`'s separate draft
    model: the target's names for weights it shares, else `draft.` for its
    `model.`. Empty for a model without one."""
    return _draft_export_names(
        model, draft_model.DRAFT_MODEL_ATTR, draft_model.DRAFT_PARAM_PREFIX
    )


def _draft_export_names(model: nn.Module, attr: str, prefix: str) -> dict[str, str]:
    """The export names of the draft at `model.<attr>`: shared tensors keep the
    target's names, its own are `prefix` for its `model.`."""
    draft = getattr(model, attr, None)
    if draft is None:
        return {}
    # A tensor the target holds under two names (a tied LM head) keeps the first.
    target: dict[int, str] = {}
    for n, t in itertools.chain(
        model.named_parameters(remove_duplicate=False),
        model.named_buffers(remove_duplicate=False),
    ):
        target.setdefault(id(t), n)
    names: dict[str, str] = {}
    for n, t in itertools.chain(
        draft.named_parameters(remove_duplicate=False),
        draft.named_buffers(remove_duplicate=False),
    ):
        own = prefix + n.removeprefix("model.")
        names[n] = target.get(id(t), own)
    return names


def mtp_draft_tensors(model: nn.Module) -> dict[str, Any]:
    """The MTP draft's own parameters and buffers by export name."""
    return _draft_own_tensors(model, MTP_DRAFT_ATTR, "mtp.")


def draft_model_tensors(model: nn.Module) -> dict[str, Any]:
    """The separate draft model's own parameters and buffers by export name."""
    return _draft_own_tensors(
        model, draft_model.DRAFT_MODEL_ATTR, draft_model.DRAFT_PARAM_PREFIX
    )


def _draft_own_tensors(model: nn.Module, attr: str, prefix: str) -> dict[str, Any]:
    draft = getattr(model, attr, None)
    if draft is None:
        return {}
    tensors = dict(draft.named_buffers())
    tensors.update(draft.named_parameters())
    names = _draft_export_names(model, attr, prefix)
    return {names[n]: t for n, t in tensors.items() if names[n].startswith(prefix)}


def _current_vllm_config(vllm_config: Any) -> contextlib.AbstractContextManager[Any]:
    """Makes `vllm_config` current, which attention backends read their KV
    cache layout from; a no-op for None."""
    if vllm_config is None:
        return contextlib.nullcontext()
    from vllm.config import set_current_vllm_config

    return set_current_vllm_config(vllm_config)


# Constant values are gathered from every rank to lay them out, so only small
# buffers qualify.
_MAX_CONSTANT_ELEMENTS = 4096


# Per-rank constant naming the rank's slice of a `RowParallelLinear` input.
_TP_RANK_BUFFER = "export_tp_rank"


def _patch_row_parallel_input_split(model: nn.Module) -> int:
    """Selects each `RowParallelLinear`'s input slice by a per-rank constant.

    With `input_is_parallel=False` the layer takes the full input and keeps
    `split[self.tp_rank]`, a Python int the trace bakes into the rank's module,
    while every device runs the one module; Muse Spark's `post_expert_proj` is
    such a layer. Each one gains a `[1]` buffer holding its rank, a constant the
    export lays out per device, and its forward picks the slice with it. A layer
    with a bias is refused: only rank 0 adds it, which the trace bakes as well.
    Returns the number of layers patched.
    """
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.layers.linear import RowParallelLinear

    if not model_parallel_is_initialized():
        return 0
    tp = get_tensor_model_parallel_world_size()
    if tp == 1:
        return 0
    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, RowParallelLinear) or module.input_is_parallel:
            continue
        if module.bias is not None and not module.skip_bias_add:
            raise NotImplementedError(
                f"{name}: a RowParallelLinear that splits its input and adds a bias "
                "cannot be exported at tensor parallel size > 1"
            )
        device = next(module.parameters()).device
        module.register_buffer(
            _TP_RANK_BUFFER,
            torch.tensor([get_tensor_model_parallel_rank()], device=device),
            persistent=False,
        )
        forward = module.forward

        def split_forward(input_: Any, module=module, forward=forward) -> Any:
            rank = getattr(module, _TP_RANK_BUFFER)
            local = input_.unflatten(-1, (tp, -1)).index_select(-2, rank).squeeze(-2)
            module.input_is_parallel = True
            try:
                return forward(local)
            finally:
                module.input_is_parallel = False

        module.forward = split_forward
        patched += 1
    return patched


def _backbone_kwargs(
    backbone: nn.Module,
    *,
    input_ids: Any,
    positions: Any,
    kv_caches: Sequence[Any],
    attn_metadata: Any,
) -> dict[str, Any]:
    """Keyword arguments for `backbone.forward`, chosen from its signature.

    vLLM backbones take `intermediate_tensors` (positionally, often with no
    default) and read the KV caches and metadata from the forward context; the
    caches and metadata are passed directly only to a forward that declares them.
    """
    params = inspect.signature(backbone.forward).parameters
    kwargs: dict[str, Any] = {"input_ids": input_ids, "positions": positions}
    if "intermediate_tensors" in params:
        kwargs["intermediate_tensors"] = None
    if "kv_caches" in params:
        kwargs["kv_caches"] = kv_caches
    if "attn_metadata" in params:
        kwargs["attn_metadata"] = attn_metadata
    return kwargs


@contextlib.contextmanager
def _weights_bound_in_place(module: nn.Module, weights: Mapping[str, Any]) -> Any:
    """Rebinds `weights` on plain objects and restores shared submodules on exit.

    `functional_call` rebinds module attributes only, while some layers hand a
    parameter to a plain attribute object (vLLM MoE routers keep the gate's
    `e_score_correction_bias`). In addition, when a submodule is reachable at
    two paths in `named_modules()` (vLLM's `MoERunner` sets `self.gate = gate`
    alongside `mlp.gate`), `functional_call` swaps the submodule's parameter
    twice and leaves the bound `torch.Tensor` in `sub.__dict__` instead of
    restoring `sub._parameters`, breaking the second forward pass. Snapshot
    each submodule's parameter/buffer/attribute tables and restore them on exit.
    """
    bound: dict[int, Any] = {}
    tensors = itertools.chain(
        module.named_parameters(remove_duplicate=False),
        module.named_buffers(remove_duplicate=False),
    )
    for name, tensor in tensors:
        value = weights.get(name)
        if isinstance(value, torch.Tensor):
            bound[id(tensor)] = value
    sub_snapshots: list[
        tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    restore: list[tuple[Any, str, Any]] = []
    try:
        for sub in module.modules():
            sub_snapshots.append(
                (
                    sub,
                    dict(sub._parameters),
                    dict(sub._buffers),
                    dict(vars(sub)),
                )
            )
            for holder in vars(sub).values():
                if isinstance(holder, (nn.Module, torch.Tensor)) or not hasattr(
                    holder, "__dict__"
                ):
                    continue
                for attr, held in list(vars(holder).items()):
                    if isinstance(held, torch.Tensor) and id(held) in bound:
                        restore.append((holder, attr, held))
                        setattr(holder, attr, bound[id(held)])
        yield
    finally:
        for holder, attr, held in reversed(restore):
            setattr(holder, attr, held)
        for sub, orig_params, orig_buffers, orig_vars in reversed(sub_snapshots):
            sub._parameters.clear()
            sub._parameters.update(orig_params)
            sub._buffers.clear()
            sub._buffers.update(orig_buffers)
            sub_vars = vars(sub)
            for k in list(sub_vars):
                if k not in orig_vars:
                    del sub_vars[k]
            sub_vars.update(orig_vars)


def _split_req_dist(req_dist: Any) -> tuple[Any, Any]:
    """The attention and recurrent distributions of `req_dist`: its first three
    elements and, for six, the last three (None for three)."""
    if int(req_dist.shape[0]) == 6:
        return req_dist[:3], req_dist[3:]
    return req_dist, None


def _is_recurrent_layer(layer: Any) -> bool:
    """Whether `layer` keeps fixed-size recurrent state (e.g. GDN linear attention)."""
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase
    except ImportError:
        return False
    return isinstance(layer, MambaBase)


def _check_recurrent_layer_supported(layer: Any) -> None:
    """Raises unless export can bind `layer`'s state and drive its forward.

    Export binds a recurrent layer's state as a conv/recurrent pair and hands
    it the gated-delta-net style metadata, which only `GatedDeltaNetAttention`
    layers read. Other recurrent layers (Mamba mixers, ShortConv) keep other
    states and expect their own metadata type.
    """
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

    if not isinstance(layer, GatedDeltaNetAttention):
        raise NotImplementedError(
            f"recurrent layer {type(layer).__name__} is not supported by export; "
            "supported: GatedDeltaNetAttention (gated delta net linear attention, "
            "e.g. Qwen3-Next)"
        )


def _recurrent_state_specs(
    layer: Any, num_slots: int, tp_size: int
) -> list[api.ShapeDtypeStruct]:
    """Device-local `[num_slots, ...]` specs of a recurrent layer's two states.

    `get_state_shape` gives one block's device-local shapes. The conv state is
    split over the model axis along its channel dimension and the recurrent
    state along its value heads, its first dimension.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    conv_shape, recurrent_shape = layer.get_state_shape()
    conv_dtype, recurrent_dtype = layer.get_state_dtype()
    conv_channel_dim = 0 if is_conv_state_dim_first() else len(conv_shape) - 1
    specs = []
    for shape, dtype, split_dim in (
        (conv_shape, conv_dtype, conv_channel_dim),
        (recurrent_shape, recurrent_dtype, 0),
    ):
        sharding: list[str | None] = [None] * (len(shape) + 1)
        if tp_size > 1:
            sharding[split_dim + 1] = "model"
        specs.append(
            api.ShapeDtypeStruct(
                shape=[num_slots, *(int(d) for d in shape)],
                dtype=tpu_lowering.dtype_name(dtype),
                sharding=sharding,
            )
        )
    return specs


class UniversalVllmExportableLLM(api.ExportableTextOnlyLLM):
    """Universal adapter wrapping any vLLM causal LM for standardized AOT export.

    Works generically across all vLLM model architectures (Qwen, Llama, DeepSeek,
    Gemma, Mistral, Mixtral, Kimi, etc.) without model-specific hand-written layers:
    - `abstract_params()` inspects `self.model.named_parameters()` and infers TP/EP
      sharding specs via `weight_mapper.infer_weight_sharding`.
    - `forward()` converts the 6 standard control inputs into `AttentionMetadata`
      via `make_attention_metadata(...)` and drives the top-level `self.model(...)`
      inside `vllm.forward_context.set_forward_context(...)`.
    - `compute_logits()` invokes the model's own `self.model.compute_logits`.
    - `sample_tokens()` executes greedy, dynamic stochastic, or penalized token
      sampling.
    """

    # The MTP draft the target was loaded with (`MTP_DRAFT_ATTR`), if any.
    _mtp_draft: nn.Module | None = None
    # The separate draft model (DSpark) the target was loaded with
    # (`draft_model.DRAFT_MODEL_ATTR`), if any.
    _draft_model: nn.Module | None = None
    # `kv_cache_groups()`, computed on first use.
    _kv_cache_groups: list[api.KvCacheGroup] | None = None

    def __init__(
        self,
        model: nn.Module | None = None,
        export_config: api.TextOnlyLLMExportConfig | None = None,
        *,
        vllm_config: Any = None,
    ) -> None:
        if model is None and vllm_config is None:
            raise ValueError(
                "UniversalVllmExportableLLM requires either `model` or `vllm_config`"
            )
        self._vllm_config = vllm_config
        # Load first: loading finalizes `cache_config.block_size`.
        if model is None:
            model = load_dummy_vllm_model(vllm_config)
        if export_config is None:
            if vllm_config is None:
                raise ValueError(
                    "export_config must be provided when vllm_config is None"
                )
            export_config = build_export_config_from_vllm(vllm_config)
        self._export_config = export_config
        self._model = model
        self._mtp_draft = getattr(model, MTP_DRAFT_ATTR, None)
        self._draft_model = getattr(model, draft_model.DRAFT_MODEL_ATTR, None)
        self._patched_rope_modules = rope_patch.patch_in_graph_rotary_embedding(
            self._model
        )
        for draft in (self._mtp_draft, self._draft_model):
            if draft is not None:
                self._patched_rope_modules += (
                    rope_patch.patch_in_graph_rotary_embedding(draft)
                )
        _patch_row_parallel_input_split(self._model)
        for draft in (self._mtp_draft, self._draft_model):
            if draft is not None:
                # Kept off the target's module tree, like the RoPE patch above.
                _patch_row_parallel_input_split(draft)
        if not self._export_config.uses_mrope and any(
            getattr(m, "_in_graph_rope_dim_modalities", None) is not None
            or rope_patch.compute_mrope_dim_modalities(m) is not None
            for _, m in self._model.named_modules()
        ):
            self._export_config = dataclasses.replace(
                self._export_config, uses_mrope=True
            )

    @property
    def model(self) -> nn.Module:
        """Returns the underlying PyTorch `nn.Module`."""
        return self._model

    @property
    def vllm_config(self) -> Any:
        """Returns the optional `VllmConfig`."""
        return self._vllm_config

    @property
    def pallas_interpret(self) -> bool:
        return False

    @property
    def export_config(self) -> api.TextOnlyLLMExportConfig:
        return self._export_config

    @trace("export.fit_num_blocks")
    def fit_num_blocks_to_hbm(
        self,
        gpu_memory_utilization: float,
        kv_cache_memory_bytes: int | None = None,
        devices: Sequence[Any] | None = None,
    ) -> int:
        """Sets `num_blocks` to the KV blocks that fit this rank's KV budget.

        Same budget as `TPUWorker.determine_available_memory`: `kv_cache_memory_bytes`
        when given, otherwise the HBM limit of `devices` (default: this rank's TPU)
        times `gpu_memory_utilization`, minus the HBM already in use (the loaded
        model). Recurrent state pools are sized by `num_state_slots`, not by
        `num_blocks`, so their bytes come off the budget first and the rest is
        divided among KV blocks. Export runs no profile pass, so activation scratch
        has to fit in the headroom `gpu_memory_utilization` leaves. Collective: rank
        0's artifact serves every rank, so all ranks take the minimum.
        """
        from vllm_torchtpu import utils

        if kv_cache_memory_bytes:
            available = kv_cache_memory_bytes
            limit_msg = f"--kv-cache-memory-bytes {kv_cache_memory_bytes}"
        else:
            if devices is None:
                devices = [torch.device("tpu")]
            budget = utils.compute_hbm_budget(devices, gpu_memory_utilization)
            available = budget.available
            limit_msg = (
                f"{budget.total_used} bytes in use of {budget.cap} allowed by "
                f"--gpu-memory-utilization {gpu_memory_utilization} "
                f"({budget.total_limit} bytes total)"
            )
        num_blocks = self._export_config.num_blocks
        total_bytes = self._state_bytes(num_blocks)
        block_bytes = self._state_bytes(num_blocks + 1) - total_bytes
        fixed_bytes = total_bytes - num_blocks * block_bytes
        fitted = max(available - fixed_bytes, 0) // block_bytes
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            from vllm.distributed import get_world_group

            t = torch.tensor([fitted], dtype=torch.int64)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MIN, group=get_world_group().cpu_group
            )
            fitted = int(t.item())
        if fitted < 1:
            raise ValueError(
                f"no KV cache block ({block_bytes} bytes) fits in the KV budget after "
                f"{fixed_bytes} bytes of recurrent state: {limit_msg}; raise the "
                "budget or pass --num-blocks"
            )
        self._export_config = dataclasses.replace(
            self._export_config, num_blocks=fitted
        )
        return fitted

    def _state_bytes(self, num_blocks: int) -> int:
        """Bytes of this rank's state tensors when exported with `num_blocks`."""
        saved = self._export_config
        self._export_config = dataclasses.replace(saved, num_blocks=num_blocks)
        try:
            states = list(self.abstract_states())
            # A draft reading the target's KV keeps none of its own.
            if self._drafts_with_mtp() and self.mtp_state_indices() is None:
                states += self.mtp_abstract_states()
            if self._drafts_with_draft_model():
                states += self.dspark_abstract_states()
            return sum(
                math.prod(s.shape) * tpu_lowering.torch_dtype(s.dtype).itemsize
                for s in states
            )
        finally:
            self._export_config = saved

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        """Builds abstract specs for the model's parameters and buffers.

        Buffers are constants: not in the checkpoint, their per-rank values come
        from `constant_axis_values`. The exporter keeps only the tensors that a
        subgraph actually reads.
        """
        params: dict[str, api.ShapeDtypeStruct] = {}
        tp_size = self._export_config.tensor_parallel_size
        named_params = list(self._model.named_parameters())
        has_ep = any(".routed_experts." in name for name, _ in named_params)

        for name, param in named_params:
            shape = tuple(int(d) for d in param.shape)
            dt_name = tpu_lowering.dtype_name(param.dtype)
            sharding = weight_mapper.infer_weight_sharding(
                name,
                shape,
                model=self._model,
                tp_size_hint=tp_size,
                has_ep_hint=has_ep,
            )
            params[name] = api.ShapeDtypeStruct(
                shape=shape,
                dtype=dt_name,
                sharding=sharding,
            )

        for name, buf in self._model.named_buffers():
            params.setdefault(
                name,
                api.ShapeDtypeStruct(
                    shape=tuple(int(d) for d in buf.shape),
                    dtype=tpu_lowering.dtype_name(buf.dtype),
                ),
            )

        # A tied LM head projects with the embedding; give it its own name, so the
        # embedding stays a forward input and the head a compute_logits one.
        tied = weight_mapper.tied_lm_head_alias(params)
        if tied:
            emb_spec = params[tied[1]]
            params[tied[0]] = api.ShapeDtypeStruct(
                shape=emb_spec.shape,
                dtype=emb_spec.dtype,
                sharding=emb_spec.sharding,
            )
        params.update(
            self._draft_abstract_params(
                self._mtp_draft, mtp_draft_names(self._model), "mtp.", tp_size, has_ep
            )
        )
        params.update(
            self._draft_abstract_params(
                self._draft_model,
                draft_model_names(self._model),
                draft_model.DRAFT_PARAM_PREFIX,
                tp_size,
                has_ep,
            )
        )
        return params

    @staticmethod
    def _draft_abstract_params(
        draft: nn.Module | None,
        names: Mapping[str, str],
        prefix: str,
        tp_size: int,
        has_ep: bool,
    ) -> dict[str, api.ShapeDtypeStruct]:
        """A draft's own parameters and buffers, by their export `names` (those
        starting with `prefix`); the ones it shares are the target's."""
        if draft is None:
            return {}
        params: dict[str, api.ShapeDtypeStruct] = {}
        for name, param in draft.named_parameters():
            if not names[name].startswith(prefix):
                continue
            shape = tuple(int(d) for d in param.shape)
            params[names[name]] = api.ShapeDtypeStruct(
                shape=shape,
                dtype=tpu_lowering.dtype_name(param.dtype),
                sharding=weight_mapper.infer_weight_sharding(
                    name,
                    shape,
                    model=draft,
                    tp_size_hint=tp_size,
                    has_ep_hint=has_ep,
                ),
            )
        for name, buf in draft.named_buffers():
            if names[name].startswith(prefix):
                params.setdefault(
                    names[name],
                    api.ShapeDtypeStruct(
                        shape=tuple(int(d) for d in buf.shape),
                        dtype=tpu_lowering.dtype_name(buf.dtype),
                    ),
                )
        return params

    def _buffers(self) -> dict[str, Any]:
        """The target's buffers and its drafts' own, by export name."""
        bufs = dict(self._model.named_buffers())
        for draft, names in (
            (self._mtp_draft, mtp_draft_names(self._model)),
            (self._draft_model, draft_model_names(self._model)),
        ):
            if draft is not None:
                for name, buf in draft.named_buffers():
                    bufs.setdefault(names[name], buf)
        return bufs

    def constant_names(self) -> set[str]:
        """Names in `abstract_params()` that are buffers rather than parameters."""
        params = {name for name, _ in self._model.named_parameters()}
        return {name for name in self._buffers() if name not in params}

    def constant_axis_values(self, name: str) -> list[Any]:
        """Values of buffer `name` on every tensor-parallel rank, in rank order.

        Collective: every rank must call it for the same names in the same order.
        """
        from vllm.distributed import get_tp_group

        buf = self._buffers()[name]
        if buf.numel() > _MAX_CONSTANT_ELEMENTS:
            raise ValueError(
                f"buffer {name} has {buf.numel()} elements; constants are limited "
                f"to {_MAX_CONSTANT_ELEMENTS}"
            )
        local = buf.detach().cpu().flatten().tolist()
        value = local[0] if buf.dim() == 0 else local
        group = get_tp_group()
        values: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(values, value, group=group.cpu_group)
        return values

    def _named_attention_layers(self) -> list[tuple[str, Any]]:
        """Returns the ordered `(layer name, layer)` attention layers of the
        target, without its drafts'."""
        draft = {
            id(m)
            for d in (self._mtp_draft, self._draft_model)
            if d is not None
            for m in d.modules()
        }
        return [(n, m) for n, m in self._all_attention_layers() if id(m) not in draft]

    def _draft_model_attention_layers(self) -> list[tuple[str, Any]]:
        """The ordered `(layer name, layer)` attention layers of the separate
        draft model."""
        if self._draft_model is None:
            return []
        draft = {id(m) for m in self._draft_model.modules()}
        return [(n, m) for n, m in self._all_attention_layers() if id(m) in draft]

    def _mtp_attention_layers(self) -> list[tuple[str, Any]]:
        """The ordered `(layer name, layer)` attention layers of the MTP draft."""
        if self._mtp_draft is None:
            return []
        draft = {id(m) for m in self._mtp_draft.modules()}
        return [(n, m) for n, m in self._all_attention_layers() if id(m) in draft]

    def _all_attention_layers(self) -> list[tuple[str, Any]]:
        """The ordered `(layer name, layer)` attention layers vLLM registered."""
        if self._vllm_config is not None:
            with contextlib.suppress(Exception):
                from vllm.model_executor.layers.attention_layer_base import (
                    AttentionLayerBase,
                )

                ctx = self._vllm_config.compilation_config.static_forward_context
                layers = [
                    (n, m) for n, m in ctx.items() if isinstance(m, AttentionLayerBase)
                ]
                if layers:
                    return layers
        return [
            (n, m)
            for n, m in self._model.named_modules()
            if hasattr(m, "kv_cache") and isinstance(m.kv_cache, list)
        ]

    def _attention_layers(self) -> list[Any]:
        """Returns the ordered attention layers of the wrapped model."""
        return [m for _, m in self._named_attention_layers()]

    def _has_recurrent_layers(self) -> bool:
        return any(_is_recurrent_layer(m) for m in self._attention_layers())

    def kv_cache_groups(self) -> list[api.KvCacheGroup]:
        """The KV cache groups of a model whose layers vLLM splits into several.

        vLLM's hybrid KV cache manager gives sliding-window and full-attention
        layers their own page tables (`get_kv_cache_groups`), with layer `j` of
        every group sharing one KV tensor. The export takes vLLM's groups as they
        are. A model with one group, or with layers other than paged decoder
        attention of one KV shape, has none: every layer reads `block_tables`.
        """
        if self._kv_cache_groups is None:
            self._kv_cache_groups = self._compute_kv_cache_groups()
        return self._kv_cache_groups

    def _compute_kv_cache_groups(self) -> list[api.KvCacheGroup]:
        # A drafter takes one page table, so a drafting model keeps every layer
        # on `block_tables`, as it exported before groups existed.
        if (
            self._vllm_config is None
            or self._drafts_with_mtp()
            or self._drafts_with_draft_model()
        ):
            return []
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        from vllm_torchtpu.kv_cache_spec_normalizer import (
            normalize_kv_cache_specs_for_tpu,
        )
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        named_layers = self._named_attention_layers()
        if not named_layers or not all(
            isinstance(m, Attention)
            and m.attn_type == AttentionType.DECODER
            and m.kv_sharing_target_layer_name is None
            for _, m in named_layers
        ):
            return []
        # Decoder attention splits into groups only around sliding-window layers.
        # Without one, nothing below runs, so such a model exports exactly as it
        # did before groups existed (the spec path touches the backend, which can
        # shift the KV budget fit by a block).
        if all(m.sliding_window is None for _, m in named_layers):
            return []
        per_layer = self._layer_state_specs([m for _, m in named_layers])
        if (
            per_layer is None
            or len({tuple((t.shape, t.dtype) for t in specs) for specs in per_layer})
            != 1
        ):
            return []
        # The runner's specs: each layer's own `get_kv_cache_spec`, normalized to
        # the TPU page geometry (`KVCacheManager.get_kv_cache_spec`), so the export
        # groups the layers exactly as serving does.
        from vllm.config import set_current_vllm_config

        specs: dict[str, Any] = {}
        # The TPU backend's `customize_spec` reads the current config, as it does
        # in the runner.
        with set_current_vllm_config(self._vllm_config):
            for name, layer in named_layers:
                spec = layer.get_kv_cache_spec(self._vllm_config)
                if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    return []
                specs[name] = spec
            specs = normalize_kv_cache_specs_for_tpu(
                specs,
                tpu_lowering.torch_dtype(per_layer[0][0].dtype),
                attention_backend=cast(
                    Any, TpuPlatform._find_non_ssm_backend(self._vllm_config)
                ),
            )
            groups = get_kv_cache_groups(self._vllm_config, specs)
        if len(groups) <= 1:
            return []
        result = []
        for i, group in enumerate(groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                raise NotImplementedError(
                    f"KV cache group {i} has a {type(spec).__name__}; export supports "
                    "full-attention and sliding-window groups"
                )
            window = (
                spec.sliding_window if isinstance(spec, SlidingWindowSpec) else None
            )
            result.append(
                api.KvCacheGroup(
                    name=f"kv{i}", layer_names=tuple(group.layer_names), window=window
                )
            )
        return result

    def _kv_cache_slots(self) -> dict[str, int]:
        """The state tensor each layer of a grouped model keeps its KV in."""
        return {
            name: j
            for group in self.kv_cache_groups()
            for j, name in enumerate(group.layer_names)
        }

    def _layer_state_specs(
        self, layers: Sequence[Any] | None = None
    ) -> list[list[api.ShapeDtypeStruct]] | None:
        """The state tensors of each attention layer, or None if any is unknown.

        A paged attention layer holds one KV cache; a recurrent (GDN) layer holds
        its conv and recurrent states, `[num_state_slots, ...]` each and indexed by
        the `mamba_state_indices` slot of each request.
        """
        cfg = self._export_config
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        attn_layers = self._attention_layers() if layers is None else list(layers)
        if not attn_layers:
            return None
        try:
            from vllm_torchtpu.layers.adapter.attention import (
                PallasAttentionBackend,
                PallasAttentionBackendImpl,
                PallasMLAttentionBackend,
                PallasMLAttentionBackendImpl,
            )
        except Exception:
            return None
        if any(
            isinstance(getattr(layer, "impl", None), PallasMLAttentionBackendImpl)
            and getattr(layer, "use_sparse", False)
            for layer in attn_layers
        ):
            raise NotImplementedError(
                "sparse MLA is not supported by export; supported: "
                "dense MLA (e.g. DeepSeek-V2, DeepSeek-V3, Kimi-K2)"
            )
        per_layer: list[list[api.ShapeDtypeStruct]] = []
        for layer in attn_layers:
            impl = getattr(layer, "impl", None)
            if isinstance(impl, PallasAttentionBackendImpl):
                # An fp8 KV cache (`--kv-cache-dtype fp8`) stores quantized KV.
                kv_dtype = getattr(
                    impl, "kv_cache_quantized_dtype", None
                ) or tpu_lowering.torch_dtype(act_dtype_str)
                # The shape the runner allocates: the layer's backend's, which
                # for the batched RPA backend (`--attention-backend CUSTOM`)
                # follows the configured KV cache layout.
                backend = getattr(layer, "attn_backend", None)
                if not (
                    isinstance(backend, type)
                    and issubclass(backend, PallasAttentionBackend)
                ):
                    backend = PallasAttentionBackend
                with _current_vllm_config(getattr(self, "_vllm_config", None)):
                    kv_shape = backend.get_kv_cache_shape(
                        cfg.num_blocks,
                        cfg.block_size,
                        impl.num_kv_heads,
                        impl.head_size,
                        kv_dtype,
                    )
                per_layer.append(
                    [
                        api.ShapeDtypeStruct(
                            shape=tuple(int(d) for d in kv_shape),
                            dtype=tpu_lowering.dtype_name(kv_dtype),
                        )
                    ]
                )
            elif isinstance(impl, PallasMLAttentionBackendImpl) and not getattr(
                layer, "use_sparse", False
            ):
                # MLA keeps one latent KV head of `head_size` (kv_lora_rank +
                # qk_rope_head_dim), packed the way the MLA kernel reads it and
                # replicated on every TP rank.
                kv_dtype = getattr(
                    layer, "kv_cache_quantized_dtype", None
                ) or tpu_lowering.torch_dtype(act_dtype_str)
                kv_shape = PallasMLAttentionBackend.get_kv_cache_shape(
                    cfg.num_blocks, cfg.block_size, 1, layer.head_size, kv_dtype
                )
                per_layer.append(
                    [
                        api.ShapeDtypeStruct(
                            shape=tuple(int(d) for d in kv_shape),
                            dtype=tpu_lowering.dtype_name(kv_dtype),
                        )
                    ]
                )
            elif _is_recurrent_layer(layer):
                _check_recurrent_layer_supported(layer)
                per_layer.append(
                    _recurrent_state_specs(
                        layer, cfg.num_state_slots, cfg.tensor_parallel_size
                    )
                )
            else:
                return None
        return per_layer

    def attention_decode_query_size(self) -> int:
        """The longest query, in tokens, the attention kernel runs as a decode.

        `TPUModelRunner._initialize_attention_kernels` sets
        `reorder_batch_threshold` to the narrowest decode stage of its Pallas
        attention layers, target and draft alike, and the step puts every
        leading request with a query at most that long in the decode segment
        of `req_dist`. 1 without Pallas attention layers.
        """
        try:
            from vllm_torchtpu.layers.adapter.attention import (
                PallasAttentionBackendImpl,
            )
        except Exception:
            return 1
        impls = (getattr(m, "impl", None) for _, m in self._all_attention_layers())
        return min(
            (
                int(impl.decode_query_size)
                for impl in impls
                if isinstance(impl, PallasAttentionBackendImpl)
            ),
            default=1,
        )

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        """Control inputs beyond the standard six that `forward` takes.

        Recurrent layers read each request's state slot from
        `mamba_state_indices`.
        """
        cfg = self._export_config
        groups = self.kv_cache_groups()
        if groups:
            # The first group's page table takes the place of `block_tables`.
            return [
                (f"{g.name}_block_tables", (cfg.max_num_pages,)) for g in groups[1:]
            ]
        if not self._has_recurrent_layers():
            return []
        inputs: list[tuple[str, tuple[int, ...]]] = [
            ("mamba_state_indices", (cfg.max_num_seqs,))
        ]
        if cfg.recurrent_window > 1:
            # Speculative verify windows: each request's `recurrent_window`
            # consecutive slots, and per slot the checkpoint to resume from.
            inputs += [
                ("mamba_slot_read_offsets", (cfg.num_state_slots,)),
                ("mamba_ckpt_indices", (cfg.max_num_seqs, cfg.recurrent_window)),
            ]
        return inputs

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """Returns the state tensors of every attention layer, in layer order.

        A model with KV cache groups has one KV tensor per layer slot of a group
        instead, shared by layer `j` of every group.
        """
        cfg = self._export_config
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        per_layer = self._layer_state_specs()
        groups = self.kv_cache_groups()
        if groups and per_layer is not None:
            num_slots = max(len(g.layer_names) for g in groups)
            return [per_layer[0][0]] * num_slots
        if per_layer is not None:
            return [spec for specs in per_layer for spec in specs]

        default_kv_shape = (
            cfg.num_blocks,
            cfg.block_size,
            cfg.num_kv_heads * 2,
            cfg.head_dim,
        )
        return [
            api.ShapeDtypeStruct(default_kv_shape, act_dtype_str)
            for _ in range(cfg.num_layers)
        ]

    def _drafts_with_mtp(self) -> bool:
        spec = self._export_config.speculative_config
        return (
            self._mtp_draft is not None and bool(spec) and spec.get("method") == "mtp"
        )

    def _require_mtp_draft(self) -> nn.Module:
        if self._mtp_draft is None:
            raise ValueError(
                "the model has no MTP draft: load it with a vLLM speculative_config "
                "whose method is 'mtp'"
            )
        return self._mtp_draft

    def _mtp_kv_targets(self) -> list[str] | None:
        """The target layer each MTP draft attention layer reads its KV from
        (`_share_target_kv_cache`), in layer order; None for a draft keeping
        its own KV."""
        targets = [
            getattr(m, "kv_sharing_target_layer_name", None)
            for _, m in self._mtp_attention_layers()
        ]
        if not any(targets):
            return None
        if not all(targets):
            raise ValueError(
                "the MTP draft reads the target's KV in some attention layers "
                f"and keeps its own in others ({targets}); export supports one "
                "or the other"
            )
        return cast(list[str], targets)

    def _target_state_indices(self) -> dict[str, int]:
        """The `abstract_states()` index of each target attention layer's
        first state."""
        if self.kv_cache_groups():
            raise NotImplementedError(
                "a draft reading the target's KV needs every layer on one page "
                "table, not KV cache groups"
            )
        named = self._named_attention_layers()
        per_layer = self._layer_state_specs([m for _, m in named])
        sizes = [1] * len(named) if per_layer is None else [len(s) for s in per_layer]
        offsets = itertools.accumulate([0, *sizes])
        return {name: offset for (name, _), offset in zip(named, offsets)}

    def mtp_state_indices(self) -> list[int] | None:
        """For an MTP draft reading the target's KV (Gemma 4's assistant), the
        target states it reads, by `abstract_states()` index, in the order its
        layers first read them: its states are those buffers. None for a draft
        keeping its own KV, whose states are new ones."""
        targets = self._mtp_kv_targets()
        if targets is None:
            return None
        index = self._target_state_indices()
        missing = sorted(set(targets) - set(index))
        if missing:
            raise ValueError(
                f"the MTP draft reads the KV of {missing}, which are not target "
                f"attention layers ({sorted(index)})"
            )
        return list(dict.fromkeys(index[t] for t in targets))

    def mtp_abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """The paged KV of each attention layer of the MTP draft, in layer
        order; for a draft reading the target's KV, the target states it reads
        (`mtp_state_indices()`)."""
        self._require_mtp_draft()
        shared = self.mtp_state_indices()
        if shared is not None:
            states = list(self.abstract_states())
            return [states[i] for i in shared]
        layers = [m for _, m in self._mtp_attention_layers()]
        specs = []
        for layer in layers:
            per_layer = self._layer_state_specs([layer])
            if per_layer is None or len(per_layer[0]) != 1:
                raise NotImplementedError(
                    f"MTP draft attention layer {type(layer).__name__} keeps no "
                    "paged KV cache export can bind"
                )
            specs.append(per_layer[0][0])
        return specs

    def _drafts_with_draft_model(self) -> bool:
        if self._draft_model is None:
            return False
        spec = self._export_config.speculative_config
        return bool(spec) and spec.get("method") in draft_model.DRAFT_MODEL_METHODS

    def _require_draft_model(self) -> nn.Module:
        if self._draft_model is None:
            raise ValueError(
                "the model has no draft model: load it with a vLLM "
                "speculative_config whose method is one of "
                f"{', '.join(draft_model.DRAFT_MODEL_METHODS)}"
            )
        return self._draft_model

    def forward_extra_outputs(self) -> list[tuple[str, int]]:
        """`(name, width)` of the `[B, width]` intermediates `forward` returns
        after `hidden_states`: with a separate draft model,
        `aux_hidden_states`, the residual stream entering each of the draft's
        target layers (after layer `target_layer_ids[i]`), concatenated, as
        wide as the draft's `fc` input."""
        if not self._drafts_with_draft_model():
            return []
        fc = getattr(self._require_draft_model().model, "fc", None)
        if fc is None:
            raise NotImplementedError(
                "a draft model that reads no target hidden states "
                "(use_aux_hidden_state=False) is not supported by export"
            )
        return [(draft_model.AUX_HIDDEN_STATES, int(fc.input_size))]

    def dspark_abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """The paged KV of each attention layer of the draft model, in layer
        order, paged like the target's (`num_blocks` of `block_size`)."""
        self._require_draft_model()
        specs = []
        for _, layer in self._draft_model_attention_layers():
            per_layer = self._layer_state_specs([layer])
            if per_layer is None or len(per_layer[0]) != 1:
                raise NotImplementedError(
                    f"draft model attention layer {type(layer).__name__} keeps no "
                    "paged KV cache export can bind"
                )
            specs.append(per_layer[0][0])
        return specs

    def dspark_draft(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        block_tables: Any,
        query_start_loc: Any,
        req_dist: Any,
        last_token_indices: Any,
        aux_hidden_states: Any,
        draft_slots: int | None = None,
    ) -> tuple[Sequence[Any], Any]:
        """Drafts `K` tokens per request with the separate draft model (DSpark or
        DFlash).

        Every row of the step writes its context keys and values, projected from
        its `aux_hidden_states`, to the draft's paged KV at its position, through
        the step's own attention metadata. Then each request drafts one block:
        its anchor, `input_ids[last_token_indices[r]]` (the token sampled after
        its last accepted one), at the next position, followed by mask tokens,
        attending to its context and, non-causally, to the block. The block's
        own keys and values land past the request's accepted tokens, where the
        next step's context rows overwrite them. A step of `B` tokens holds at
        most `min(B, max_num_seqs)` requests, so only those draft;
        `draft_tokens` rows past them are 0. With `draft_slots` `N`, only the
        first `N` requests draft: a variant for steps of at most `N` requests
        (`draft_slot_counts`).
        """
        draft: Any = self._require_draft_model()
        cfg = self._export_config
        k = int(cfg.speculative_config["num_speculative_tokens"])  # type: ignore[index]
        block = draft_model.draft_block_size(draft, k)
        mask_token_id = int(draft.config.dflash_config["mask_token_id"])
        names = draft_model_names(self._model)
        call_weights = {d: weights[e] for d, e in names.items() if e in weights}
        layers = self._draft_model_attention_layers()
        for (_, layer), kv in zip(layers, kv_caches, strict=True):
            layer.kv_cache = kv
        if positions.dim() > 1:
            positions = positions[0]

        attention_dist, _ = _split_req_dist(req_dist)
        context_metadata = make_attention_metadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=kv_lens,
            query_start_loc=query_start_loc,
            request_distribution=attention_dist,
        )
        rows = min(int(input_ids.shape[0]), cfg.max_num_seqs)
        if draft_slots is not None:
            rows = min(rows, draft_slots)
        block_ids, block_positions, block_metadata = self._draft_block_inputs(
            input_ids=input_ids,
            positions=positions,
            block_tables=block_tables,
            num_seqs=attention_dist[2],
            last_token_indices=last_token_indices[:rows],
            block=block,
            mask_token_id=mask_token_id,
        )

        with contextlib.ExitStack() as stack:
            if self._vllm_config is not None:
                mesh = getattr(self._model, "_export_tpu_mesh", None)
                if mesh is not None:
                    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (  # noqa: E501
                        set_vllm_model_wrapper_context,
                    )

                    stack.enter_context(
                        set_vllm_model_wrapper_context(
                            mesh=mesh, vllm_config=self._vllm_config
                        )
                    )
            from torch.nn.utils.stateless import _reparametrize_module

            stack.enter_context(_weights_bound_in_place(draft, call_weights))
            stack.enter_context(_reparametrize_module(draft, call_weights))
            draft_model.bind_fused_kv_weight(draft)
            draft_model.precompute_context_kv(
                draft,
                aux_hidden_states,
                positions,
                [context_metadata] * len(layers),
            )
            with self._draft_forward_context(layers, block_metadata, block_ids):
                tokens = draft_model.draft_forward_and_sample(
                    draft, block_ids, block_positions, block
                )
        updated = [
            layer.kv_cache[0] if isinstance(layer.kv_cache, list) else layer.kv_cache
            for _, layer in layers
        ]
        tokens = tokens.to(torch.int32)
        if rows < cfg.max_num_seqs:
            tokens = torch.cat(
                [tokens, tokens.new_zeros((cfg.max_num_seqs - rows, k))], dim=0
            )
        return updated, tokens

    def _draft_block_inputs(
        self,
        *,
        input_ids: Any,
        positions: Any,
        block_tables: Any,
        num_seqs: Any,
        last_token_indices: Any,
        block: int,
        mask_token_id: int,
    ) -> tuple[Any, Any, Any]:
        """The draft block of each of `len(last_token_indices)` request rows:
        `(input_ids, positions, attention metadata)`.

        Request `r`'s block is its anchor `input_ids[last_token_indices[r]]` at
        `positions[last_token_indices[r]] + 1`, then mask tokens, laid out by
        `draft_model.layout_draft_block`. Requests past `num_seqs` attend to
        nothing (`seq_lens` 0) and are outside `query_start_loc`'s live range.
        """
        cfg = self._export_config
        rows = int(last_token_indices.shape[0])
        idx = last_token_indices.to(torch.int64)
        live = torch.arange(rows, device=idx.device) < num_seqs
        base_tokens = torch.where(live, input_ids[idx], mask_token_id).to(torch.int32)
        base_pos = torch.where(live, positions[idx] + 1, 0).to(torch.int32)
        block_ids, block_positions, seq_lens = draft_model.layout_draft_block(
            base_tokens,
            base_pos.unsqueeze(1),
            block,
            rows * block,
            mask_token_id,
            cfg.max_model_len,
        )
        query_start_loc = (
            torch.clamp(torch.arange(rows + 1, device=idx.device), max=num_seqs) * block
        ).to(torch.int32)
        if rows < cfg.max_num_seqs:
            pad = cfg.max_num_seqs - rows
            seq_lens = torch.cat([seq_lens, seq_lens.new_zeros(pad)])
            query_start_loc = torch.cat(
                [query_start_loc, query_start_loc[-1:].expand(pad)]
            )
        seq_lens = torch.where(
            torch.arange(cfg.max_num_seqs, device=idx.device) < num_seqs, seq_lens, 0
        ).to(torch.int32)
        zero = torch.zeros((), dtype=torch.int32, device=idx.device)
        metadata = make_attention_metadata(
            input_positions=block_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=torch.stack(
                [zero, zero, torch.as_tensor(num_seqs).to(torch.int32)]
            ),
        )
        return block_ids, block_positions, metadata

    @contextlib.contextmanager
    def _draft_forward_context(
        self, layers: Sequence[tuple[str, Any]], metadata: Any, input_ids: Any
    ) -> Any:
        """The forward context a draft model pass reads its layers' metadata from."""
        if self._vllm_config is None:
            yield
            return
        from vllm.forward_context import set_forward_context

        with set_forward_context(
            {name: metadata for name, _ in layers},
            self._vllm_config,
            num_tokens=int(input_ids.shape[0]),
        ):
            yield

    def _lm_head_weight_name(self) -> str:
        """The `abstract_params` name `compute_logits` projects with."""
        names = [n for n, _ in self._model.named_parameters()]
        tied = weight_mapper.tied_lm_head_alias(names)
        if tied:
            return tied[0]
        heads = [
            n for n in names if n == "lm_head.weight" or n.endswith(".lm_head.weight")
        ]
        if len(heads) != 1:
            raise ValueError(f"expected one LM head weight, found {heads}")
        return heads[0]

    def mtp_draft(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        block_tables: Any,
        query_start_loc: Any,
        req_dist: Any,
        last_token_indices: Any,
        hidden_states: Any,
        draft_slots: int | None = None,
    ) -> tuple[Sequence[Any], Any]:
        """Drafts `K` tokens per request with the MTP draft.

        The first pass runs the draft over every row of the step, `input_ids`
        shifted by one token and the target's `hidden_states`, which writes the
        draft's KV for each row; each request's first draft comes from its row
        `last_token_indices`. Each further pass feeds every request its last
        draft and the draft's own hidden state, one token per request at the
        next position, as vLLM's MTP proposer chains them. With `draft_slots`
        `N`, only the first `N` requests draft (the first pass still covers every
        row): a variant for steps of at most `N` requests (`draft_slot_counts`).

        A draft reading the target's KV (Gemma 4's assistant) takes the target
        states it reads (`mtp_state_indices()`) and writes none, so each further
        pass stays at the first pass's last position and KV length, as the TPU
        runner's proposer does for it (`advance_draft_positions`). A draft
        returning `(hidden, carry)` picks tokens from `hidden` and feeds `carry`
        on (`Eagle3Proposer._unwrap_model_out`), and a draft with its own LM
        head projects with it.
        """
        from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

        draft = self._require_mtp_draft()
        if getattr(draft, "masked_embedding", None) is not None:
            raise NotImplementedError(
                "an MTP draft with ordered embeddings (centroid-masked logits, "
                "use_ordered_embeddings) is not supported by export"
            )
        if getattr(draft, "_suppress_idx", None) is not None:
            raise NotImplementedError(
                "an MTP draft suppressing tokens (generation_config "
                "suppress_tokens) is not supported by export"
            )
        cfg = self._export_config
        k = int(cfg.speculative_config["num_speculative_tokens"])  # type: ignore[index]
        names = mtp_draft_names(self._model)
        call_weights = {d: weights[e] for d, e in names.items() if e in weights}
        if getattr(draft, "has_own_lm_head", False):
            lm_head = weights[names["lm_head.weight"]]
        else:
            lm_head = weights[self._lm_head_weight_name()]
        layers = self._mtp_attention_layers()
        shared = self.mtp_state_indices()
        if shared is None:
            layer_states = list(range(len(layers)))
            if len(kv_caches) != len(layers):
                raise ValueError(
                    f"{len(kv_caches)} MTP KV states for {len(layers)} layers"
                )
        else:
            index = self._target_state_indices()
            targets = cast(list[str], self._mtp_kv_targets())
            layer_states = [shared.index(index[t]) for t in targets]
        for (_, layer), s_i in zip(layers, layer_states, strict=True):
            layer.kv_cache = kv_caches[s_i]
        # Whether each further pass moves one token on: not for a draft that
        # writes no KV.
        advance = int(shared is None)

        def run(ids: Any, pos: Any, hidden: Any, metadata: Any) -> Any:
            forward_metadata = {name: metadata for name, _ in layers}
            prev = getattr(draft, "do_not_compile", None)
            if prev is not None:
                object.__setattr__(draft, "do_not_compile", True)
            try:
                with contextlib.ExitStack() as stack:
                    if self._vllm_config is not None:
                        from vllm.forward_context import set_forward_context

                        mesh = getattr(self._model, "_export_tpu_mesh", None)
                        if mesh is not None:
                            from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (  # noqa: E501
                                set_vllm_model_wrapper_context,
                            )

                            stack.enter_context(
                                set_vllm_model_wrapper_context(
                                    mesh=mesh, vllm_config=self._vllm_config
                                )
                            )
                        stack.enter_context(
                            set_forward_context(
                                forward_metadata,
                                self._vllm_config,
                                num_tokens=int(ids.shape[0]),
                            )
                        )
                    stack.enter_context(_weights_bound_in_place(draft, call_weights))
                    return torch.func.functional_call(
                        draft, call_weights, args=(ids, pos, hidden)
                    )
            finally:
                if prev is not None:
                    object.__setattr__(draft, "do_not_compile", prev)

        def greedy(hidden: Any) -> Any:
            return self.greedy_tokens(hidden, lm_head)

        last, hidden = Eagle3Proposer._unwrap_model_out(
            run(
                input_ids,
                positions,
                hidden_states,
                make_attention_metadata(
                    input_positions=positions,
                    block_tables=block_tables,
                    seq_lens=kv_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=_split_req_dist(req_dist)[0],
                ),
            )
        )
        all_rows = last_token_indices.to(torch.int64)
        slots = cfg.max_num_seqs if draft_slots is None else draft_slots
        rows = all_rows[:slots]
        last, hidden = last[rows], hidden[rows]
        drafts = [greedy(last)]
        if k > 1:
            num_seqs = req_dist[2]
            live = torch.arange(cfg.max_num_seqs, device=rows.device) < num_seqs
            # KV index of each request's row `last_token_indices`: its KV length
            # less the rows after it in the step.
            after = query_start_loc[1:] - all_rows.to(kv_lens.dtype) - 1
            last_kv = kv_lens - after - 1
            one_per_seq = torch.clamp(
                torch.arange(cfg.max_num_seqs + 1, device=rows.device), max=num_seqs
            ).to(query_start_loc.dtype)
            all_decode = torch.stack([num_seqs, num_seqs, num_seqs]).to(req_dist.dtype)
            last_pos = positions[..., rows]
            for step in range(1, k):
                delta = step * advance
                pos = last_pos + delta
                metadata = make_attention_metadata(
                    input_positions=pos,
                    block_tables=block_tables,
                    seq_lens=torch.where(live, last_kv + delta + 1, 0).to(
                        kv_lens.dtype
                    ),
                    query_start_loc=one_per_seq,
                    request_distribution=all_decode,
                )
                last, hidden = Eagle3Proposer._unwrap_model_out(
                    run(drafts[-1], pos, hidden, metadata)
                )
                drafts.append(greedy(last))
        # Each state as its first layer holds it after the passes.
        holder: dict[int, Any] = {}
        for (_, layer), s_i in zip(layers, layer_states):
            holder.setdefault(s_i, layer)
        updated = [
            holder[s_i].kv_cache[0]
            if isinstance(holder[s_i].kv_cache, list)
            else holder[s_i].kv_cache
            for s_i in range(len(kv_caches))
        ]
        tokens = torch.stack(drafts, dim=1)
        if slots < cfg.max_num_seqs:
            tokens = torch.cat(
                [tokens, tokens.new_zeros((cfg.max_num_seqs - slots, k))], dim=0
            )
        return updated, tokens

    def forward(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        query_start_loc: Any,
        req_dist: Any,
        block_tables: Any = None,
        **extra_controls: Any,
    ) -> tuple[Any, ...]:
        """Executes one flattened continuous-batching forward pass.

        Returns the updated states and `hidden_states`, then the
        `forward_extra_outputs()`: with a separate draft model,
        `aux_hidden_states`, the target layers' residual stream concatenated
        per row.

        A 6-element `req_dist` is the attention kernel's distribution, then the
        recurrent kernels', whose windowed segment takes the verify windows too.
        A model with KV cache groups takes each group's page table as
        `{name}_block_tables` instead of `block_tables`.
        """
        num_tokens = int(input_ids.shape[0])
        attention_dist, recurrent_dist = _split_req_dist(req_dist)
        attn_metadata = make_attention_metadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=kv_lens,
            query_start_loc=query_start_loc,
            request_distribution=attention_dist,
            mamba_state_indices=extra_controls.get("mamba_state_indices"),
            mamba_slot_read_offsets=extra_controls.get("mamba_slot_read_offsets"),
            mamba_ckpt_indices=extra_controls.get("mamba_ckpt_indices"),
            mamba_request_distribution=recurrent_dist,
        )

        # The model's own forward, as `tpu_runner.forward_model` calls it: whatever a
        # top-level forward does around its decoder (multimodal embedding merges,
        # positions handling) is in the artifact, and weights keep their names.
        backbone: Any = self._model
        call_weights = {
            k: v for k, v in weights.items() if not k.startswith("lm_head.")
        }

        named_layers = self._named_attention_layers()
        attn_layers = [m for _, m in named_layers]
        working_kvs = list(kv_caches)
        groups = self.kv_cache_groups()
        per_layer = None if groups else self._layer_state_specs()
        # Each recurrent layer binds its `[conv_state, recurrent_state]` list.
        layer_kvs: list[list[Any]] | None = None
        # Of a model with KV cache groups, one layer per shared KV tensor and the
        # metadata of each layer's group.
        slot_layers: list[Any] = []
        metadata_by_layer: dict[str, Any] = {}
        if groups:
            slots = self._kv_cache_slots()
            group_metadata = {
                g.name: make_attention_metadata(
                    input_positions=positions,
                    block_tables=extra_controls[f"{g.name}_block_tables"],
                    seq_lens=kv_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=attention_dist,
                )
                for g in groups
            }
            metadata_by_layer = {
                name: group_metadata[g.name] for g in groups for name in g.layer_names
            }
            # Layers that share a tensor hold the same object, so the in-place KV
            # write of one is what the next one reads.
            for name, layer in named_layers:
                layer.kv_cache = working_kvs[slots[name]]
            first_by_slot: dict[int, Any] = {}
            for name, layer in named_layers:
                first_by_slot.setdefault(slots[name], layer)
            slot_layers = [first_by_slot[j] for j in range(len(working_kvs))]
        elif per_layer is not None and sum(map(len, per_layer)) == len(working_kvs):
            it = iter(working_kvs)
            layer_kvs = [[next(it) for _ in specs] for specs in per_layer]
            for layer, bufs in zip(attn_layers, layer_kvs, strict=True):
                layer.kv_cache = bufs if _is_recurrent_layer(layer) else bufs[0]
        elif len(attn_layers) == len(working_kvs):
            for layer, kv_buf in zip(attn_layers, working_kvs, strict=True):
                layer.kv_cache = kv_buf
        forward_metadata: Any = attn_metadata
        if groups:
            forward_metadata = metadata_by_layer
        elif any(_is_recurrent_layer(m) for m in attn_layers):
            # Recurrent layers look their metadata up by layer name, as the
            # runner's per-layer metadata provides it.
            forward_metadata = {name: attn_metadata for name, _ in named_layers}

        def _invoke_backbone() -> tuple[Any, ...]:
            use_func = bool(call_weights) and all(
                isinstance(v, torch.Tensor) for v in call_weights.values()
            )
            prev_do_not_compile = getattr(backbone, "do_not_compile", None)
            if prev_do_not_compile is not None:
                backbone.do_not_compile = True

            try:

                def _call(kw: dict[str, Any]) -> Any:
                    if use_func:
                        with _weights_bound_in_place(backbone, call_weights):
                            return torch.func.functional_call(
                                backbone, call_weights, args=(), kwargs=kw
                            )
                    return backbone(**kw)

                out = _call(
                    _backbone_kwargs(
                        backbone,
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=working_kvs,
                        attn_metadata=forward_metadata,
                    )
                )
            finally:
                if prev_do_not_compile is not None:
                    backbone.do_not_compile = prev_do_not_compile

            if slot_layers:
                updated_kvs = [layer.kv_cache for layer in slot_layers]
            elif layer_kvs is not None:
                updated_kvs = [
                    buf
                    for layer in attn_layers
                    for buf in (
                        layer.kv_cache
                        if _is_recurrent_layer(layer)
                        else [
                            layer.kv_cache[0]
                            if isinstance(layer.kv_cache, list)
                            else layer.kv_cache
                        ]
                    )
                ]
            elif len(attn_layers) == len(working_kvs):
                updated_kvs = [
                    layer.kv_cache[0]
                    if isinstance(layer.kv_cache, list)
                    else layer.kv_cache
                    for layer in attn_layers
                ]
            else:
                updated_kvs = working_kvs
            aux = None
            if (
                isinstance(out, tuple)
                and len(out) == 2
                and isinstance(out[0], torch.Tensor)
                and isinstance(out[1], (list, tuple))
            ):
                # A target tagged with a draft model's target layers
                # (`set_aux_hidden_state_layers`) also returns their residual
                # stream.
                out, aux = out[0], list(out[1])
            if (
                isinstance(out, tuple)
                and len(out) == 2
                and isinstance(out[0], (list, tuple))
            ):
                kvs, hidden = list(out[0]), out[1]
            elif isinstance(out, tuple):
                kvs, hidden = updated_kvs, out[0]
            else:
                kvs, hidden = updated_kvs, out
            if not self.forward_extra_outputs():
                return kvs, hidden
            if not aux:
                raise ValueError(
                    f"{type(backbone).__name__} returned no residual stream of the "
                    "draft model's target layers"
                )
            return kvs, hidden, torch.cat(aux, dim=-1)

        if self._vllm_config is not None:
            from vllm.forward_context import set_forward_context

            mesh = getattr(self._model, "_export_tpu_mesh", None)
            with contextlib.ExitStack() as stack:
                if mesh is not None:
                    with contextlib.suppress(Exception):
                        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (  # noqa: E501
                            set_vllm_model_wrapper_context,
                        )

                        stack.enter_context(
                            set_vllm_model_wrapper_context(
                                mesh=mesh, vllm_config=self._vllm_config
                            )
                        )
                stack.enter_context(
                    set_forward_context(
                        forward_metadata, self._vllm_config, num_tokens=num_tokens
                    )
                )
                return _invoke_backbone()
        return _invoke_backbone()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Projects `hidden_states[R, hidden_size]` to `logits[R, vocab_size]`.

        Each rank holds one vocabulary shard of `lm_head_weight`; the shards are
        gathered and the vocabulary padding dropped, so every rank returns the full
        vocabulary (at tensor parallel size 1 the gather returns its input).

        A vLLM model's own `compute_logits` runs, with `lm_head_weight` bound to its
        LM head, so whatever it does past the projection (Muse Spark's output
        multiplier and final logit soft-capping) is in the artifact too.
        """
        from vllm.distributed import tensor_model_parallel_all_gather

        model_logits = getattr(self._model, "compute_logits", None)
        if self._vllm_config is not None and callable(model_logits):
            head = self._bound_head_name()
            if head is not None:
                from torch.nn.utils.stateless import _reparametrize_module

                with _reparametrize_module(self._model, {head: lm_head_weight}):
                    logits = model_logits(hidden_states)
                return logits[..., : self._export_config.vocab_size]
        local = self._local_logits(hidden_states, lm_head_weight)
        gathered = tensor_model_parallel_all_gather(local, dim=-1)
        return gathered[..., : self._export_config.vocab_size]

    def greedy_tokens(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """The argmax token `i32[R]` of `compute_logits(hidden_states, lm_head_weight)`.

        Each rank takes the maximum of its own vocabulary shard, and the ranks
        gather only those `[R, tp]` maxima and their indices instead of the
        `[R, vocab_size]` logits; the first shard holding the overall maximum
        names the token, so ties resolve to the lowest index, as a full argmax
        does. With vocabulary padding the last shard's padded columns would join
        the race, and the shard a rank holds is unknown while tracing, so a
        padded vocabulary gathers the logits instead.

        It takes the raw projection, not `compute_logits`'s post-processing: a
        monotonic step (output multiplier, soft-capping) leaves the argmax alone,
        except where bf16 soft-capping saturates several logits to one value,
        which can change a draft token, never a verified one.
        """
        from vllm.distributed import tensor_model_parallel_all_gather

        local = self._local_logits(hidden_states, lm_head_weight)
        shard = int(local.shape[-1])
        tp = self._export_config.tensor_parallel_size
        if shard * tp != self._export_config.vocab_size:
            logits = self.compute_logits(hidden_states, lm_head_weight)
            return logits.argmax(dim=-1).to(torch.int32)
        index = local.argmax(dim=-1, keepdim=True)
        best = local.gather(-1, index)
        best = tensor_model_parallel_all_gather(best, dim=-1)
        index = tensor_model_parallel_all_gather(index.to(torch.int32), dim=-1)
        rank = best.argmax(dim=-1, keepdim=True)
        token = rank.to(torch.int32) * shard + index.gather(-1, rank)
        return token.squeeze(-1)

    def _bound_head_name(self) -> str | None:
        """The parameter `compute_logits` projects with, as the model names it.

        `_lm_head_weight_name()` may be the alias of a tied head the model holds
        only as its embedding (no `lm_head` attribute); then the embedding's
        parameter is bound. None when the model holds neither.
        """
        names = {n for n, _ in self._model.named_parameters(remove_duplicate=False)}
        head = self._lm_head_weight_name()
        if head in names:
            return head
        tied = weight_mapper.tied_lm_head_alias(names)
        return tied[1] if tied and tied[0] == head else None

    def _local_logits(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        lm_head_mod = getattr(self._model, "lm_head", None)
        if isinstance(lm_head_mod, nn.Module) and hasattr(lm_head_mod, "weight"):
            with contextlib.suppress(Exception):
                return torch.func.functional_call(
                    lm_head_mod,
                    {"weight": lm_head_weight},
                    (hidden_states,),
                )
        if lm_head_weight.dim() == 2:
            if lm_head_weight.shape[0] == hidden_states.shape[-1]:
                return torch.matmul(hidden_states, lm_head_weight)
            return torch.nn.functional.linear(hidden_states, lm_head_weight)
        return torch.matmul(hidden_states, lm_head_weight.transpose(-1, -2))

    def sample_tokens(
        self,
        logits: Any,
        *,
        temperature: Any | None = None,
        top_k: Any | None = None,
        top_p: Any | None = None,
        min_p: Any | None = None,
        uniform: Any | None = None,
        repetition_penalty: Any | None = None,
        presence_penalty: Any | None = None,
        frequency_penalty: Any | None = None,
        history: Any | None = None,
        output_start: Any | None = None,
    ) -> Any:
        """Samples token ids `i64[R, 1]` from `logits[R, vocab_size]`."""
        return sampling.dispatch_sample_tokens(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniform=uniform,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            history=history,
            output_start=output_start,
        )
