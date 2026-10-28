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
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import (
    api,
    rope_patch,
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

    The companion files (`config.json`, `tokenizer.json`, `generation_config.json`)
    and the safetensors metadata are read from this directory. For a repo id the
    companion files are fetched into the Hub cache first: loading dummy weights
    caches `config.json` alone, so the tokenizer would otherwise be missing. A
    repo without `config.json` is an error rather than a silently thinner
    artifact.
    """
    if not model or os.path.isdir(model):
        return model
    from transformers.utils import cached_file

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
    # A missing tokenizer.json is reported by the exporter, which requires it.
    for name in ("tokenizer.json", "generation_config.json"):
        cached_file(model, name, _raise_exceptions_for_missing_entries=False)
    return os.path.dirname(config_path)


@trace("export.build_export_config")
def build_export_config_from_vllm(
    vllm_config: Any,
    *,
    model_name: str | None = None,
    hf_checkpoint_dir: str | None = None,
    target_tpu: str | None = None,
    token_buckets: Sequence[int] | None = None,
    req_buckets: Sequence[int] | None = None,
) -> api.TextOnlyLLMExportConfig:
    """Derives a `TextOnlyLLMExportConfig` from a `VllmConfig` (or mock config)."""
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
        with contextlib.suppress(Exception):
            from vllm_torchtpu.tpu_info import get_chip_version

            target_tpu = str(get_chip_version().value)
        if not target_tpu:
            target_tpu = "v6e"

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
    num_layers = int(
        _extract_hf_field(
            hf_cfg, "num_hidden_layers", "num_layers", "n_layer", default=28
        )
    )
    num_q_heads = int(
        _extract_hf_field(
            hf_cfg, "num_attention_heads", "num_heads", "n_head", default=16
        )
    )
    num_kv_heads = int(
        _extract_hf_field(
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
        req_buckets = default_req_buckets(max_num_seqs)
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
        return load_model(
            vllm_config=vllm_config, model_config=vllm_config.model_config
        )


def load_dummy_vllm_model(vllm_config: Any) -> nn.Module:
    """Loads a vLLM model with `load_format='dummy'` on TPU (zero disk weight IO)."""
    return load_vllm_model(vllm_config, load_format="dummy")


@trace("export.load_model")
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
            return model


# Constant values are gathered from every rank to lay them out, so only small
# buffers qualify.
_MAX_CONSTANT_ELEMENTS = 4096


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
      via `make_attention_metadata(...)` and drives `self.model.model(...)` (or
      `self.model(...)`) inside `vllm.forward_context.set_forward_context(...)`.
    - `compute_logits()` invokes `self.model.compute_logits` or `self.model.lm_head`.
    - `sample_tokens()` executes greedy, dynamic stochastic, or penalized token
      sampling.
    """

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
        self._patched_rope_modules = rope_patch.patch_in_graph_rotary_embedding(
            self._model
        )
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
            return sum(
                math.prod(s.shape) * tpu_lowering.torch_dtype(s.dtype).itemsize
                for s in self.abstract_states()
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
        return params

    def constant_names(self) -> set[str]:
        """Names in `abstract_params()` that are buffers rather than parameters."""
        params = {name for name, _ in self._model.named_parameters()}
        return {name for name, _ in self._model.named_buffers() if name not in params}

    def constant_axis_values(self, name: str) -> list[Any]:
        """Values of buffer `name` on every tensor-parallel rank, in rank order.

        Collective: every rank must call it for the same names in the same order.
        """
        from vllm.distributed import get_tp_group

        buf = dict(self._model.named_buffers())[name]
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
        """Returns the ordered `(layer name, layer)` attention layers of the model."""
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

    def _layer_state_specs(self) -> list[list[api.ShapeDtypeStruct]] | None:
        """The state tensors of each attention layer, or None if any is unknown.

        A paged attention layer holds one KV cache; a recurrent (GDN) layer holds
        its conv and recurrent states, `[num_state_slots, ...]` each and indexed by
        the `mamba_state_indices` slot of each request.
        """
        cfg = self._export_config
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        attn_layers = self._attention_layers()
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
                kv_shape = PallasAttentionBackend.get_kv_cache_shape(
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

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        """Control inputs beyond the standard six that `forward` takes.

        Recurrent layers read each request's state slot from
        `mamba_state_indices`.
        """
        if self._has_recurrent_layers():
            return [("mamba_state_indices", (self._export_config.max_num_seqs,))]
        return []

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """Returns the state tensors of every attention layer, in layer order."""
        cfg = self._export_config
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        per_layer = self._layer_state_specs()
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

    def forward(
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
        **extra_controls: Any,
    ) -> tuple[Sequence[Any], Any]:
        """Executes one flattened continuous-batching forward pass."""
        num_tokens = int(input_ids.shape[0])
        attn_metadata = make_attention_metadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=kv_lens,
            query_start_loc=query_start_loc,
            request_distribution=req_dist,
            mamba_state_indices=extra_controls.get("mamba_state_indices"),
            mamba_slot_read_offsets=extra_controls.get("mamba_slot_read_offsets"),
            mamba_ckpt_indices=extra_controls.get("mamba_ckpt_indices"),
            mamba_request_distribution=extra_controls.get("mamba_request_distribution"),
        )

        backbone: Any = getattr(self._model, "model", self._model)
        call_weights: dict[str, Any] = {}
        if backbone is not self._model:
            prefix = "model."
            for k, v in weights.items():
                if k.startswith(prefix):
                    call_weights[k[len(prefix) :]] = v
                elif not k.startswith("lm_head."):
                    call_weights[k] = v
        else:
            call_weights = {
                k: v for k, v in weights.items() if not k.startswith("lm_head.")
            }

        named_layers = self._named_attention_layers()
        attn_layers = [m for _, m in named_layers]
        working_kvs = list(kv_caches)
        per_layer = self._layer_state_specs()
        # Each recurrent layer binds its `[conv_state, recurrent_state]` list.
        layer_kvs: list[list[Any]] | None = None
        if per_layer is not None and sum(map(len, per_layer)) == len(working_kvs):
            it = iter(working_kvs)
            layer_kvs = [[next(it) for _ in specs] for specs in per_layer]
            for layer, bufs in zip(attn_layers, layer_kvs, strict=True):
                layer.kv_cache = bufs if _is_recurrent_layer(layer) else bufs[0]
        elif len(attn_layers) == len(working_kvs):
            for layer, kv_buf in zip(attn_layers, working_kvs, strict=True):
                layer.kv_cache = kv_buf
        forward_metadata: Any = attn_metadata
        if any(_is_recurrent_layer(m) for m in attn_layers):
            # Recurrent layers look their metadata up by layer name, as the
            # runner's per-layer metadata provides it.
            forward_metadata = {name: attn_metadata for name, _ in named_layers}

        def _invoke_backbone() -> tuple[Sequence[Any], Any]:
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

            if layer_kvs is not None:
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
            if (
                isinstance(out, tuple)
                and len(out) == 2
                and isinstance(out[0], (list, tuple))
            ):
                return list(out[0]), out[1]
            if isinstance(out, tuple):
                return updated_kvs, out[0]
            return updated_kvs, out

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
        """
        from vllm.distributed import tensor_model_parallel_all_gather

        local = self._local_logits(hidden_states, lm_head_weight)
        gathered = tensor_model_parallel_all_gather(local, dim=-1)
        return gathered[..., : self._export_config.vocab_size]

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
