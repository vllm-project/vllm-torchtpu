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
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


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


def resolve_hf_checkpoint_dir(model: str | None) -> str | None:
    """Returns the local checkpoint directory for a path or a cached Hub repo id.

    The companion files (`config.json`, `tokenizer.json`, `generation_config.json`)
    and the safetensors metadata are read from this directory, so a repo id that is
    not in the local Hub cache is an error rather than a silently thinner artifact.
    """
    if not model or os.path.isdir(model):
        return model
    from transformers.utils import cached_file

    hint = (
        f"--model {model!r} is neither a local directory nor a repo in the "
        "HuggingFace cache; download it first or pass the checkpoint directory"
    )
    try:
        config_path = cached_file(model, "config.json", local_files_only=True)
    except OSError as e:
        raise FileNotFoundError(hint) from e
    if config_path is None:
        raise FileNotFoundError(hint)
    return os.path.dirname(config_path)


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
    dp_size = int(getattr(parallel_cfg, "data_parallel_size", 1) or 1)
    pp_size = int(getattr(parallel_cfg, "pipeline_parallel_size", 1) or 1)
    if (dp_size, pp_size) != (1, 1):
        raise NotImplementedError(
            f"export supports only data/pipeline parallel size 1, got "
            f"dp={dp_size} pp={pp_size}"
        )

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
        default_reqs = [8, 16, 32, 64]
        filtered = [r for r in default_reqs if r <= max_num_seqs]
        if not filtered or filtered[-1] != max_num_seqs:
            filtered.append(max_num_seqs)
        req_buckets = tuple(sorted(set(filtered)))
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


def load_dummy_vllm_model(vllm_config: Any) -> nn.Module:
    """Loads a vLLM model with `load_format='dummy'` on TPU (zero disk weight IO)."""
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.model_loader import get_model_loader

    vllm_config.load_config.load_format = "dummy"
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

            loader = get_model_loader(vllm_config.load_config)
            # `@instrument` on `load_model` hides its signature from type checkers.
            load_model: Any = loader.load_model
            model: nn.Module = load_model(
                vllm_config=vllm_config,
                model_config=vllm_config.model_config,
            )

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


# Constants are written inline into weight.json, so only small buffers qualify.
_MAX_CONSTANT_ELEMENTS = 4096


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
        if export_config is None:
            if vllm_config is None:
                raise ValueError(
                    "export_config must be provided when vllm_config is None"
                )
            export_config = build_export_config_from_vllm(vllm_config)
        self._export_config = export_config
        if model is None:
            model = load_dummy_vllm_model(vllm_config)
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

        # Ensure lm_head.weight is present even when tied to embed_tokens.weight
        if "lm_head.weight" not in params:
            for cand in (
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "transformer.wte.weight",
            ):
                if cand in params:
                    emb_spec = params[cand]
                    params["lm_head.weight"] = api.ShapeDtypeStruct(
                        shape=emb_spec.shape,
                        dtype=emb_spec.dtype,
                        sharding=emb_spec.sharding,
                    )
                    break
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
                f"buffer {name} has {buf.numel()} elements; constants are declared "
                f"inline in weight.json and are limited to {_MAX_CONSTANT_ELEMENTS}"
            )
        local = buf.detach().cpu().flatten().tolist()
        value = local[0] if buf.dim() == 0 else local
        group = get_tp_group()
        values: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(values, value, group=group.cpu_group)
        return values

    def _attention_layers(self) -> list[Any]:
        """Returns the ordered attention layers of the wrapped model."""
        if self._vllm_config is not None:
            with contextlib.suppress(Exception):
                from vllm.model_executor.layers.attention_layer_base import (
                    AttentionLayerBase,
                )

                ctx = self._vllm_config.compilation_config.static_forward_context
                layers = [m for m in ctx.values() if isinstance(m, AttentionLayerBase)]
                if layers:
                    return layers
        return [
            m
            for _, m in self._model.named_modules()
            if hasattr(m, "kv_cache") and isinstance(m.kv_cache, list)
        ]

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """Returns per-layer KV cache specs matching the active attention backend."""
        cfg = self._export_config
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        attn_layers = self._attention_layers()
        if attn_layers:
            with contextlib.suppress(Exception):
                from vllm_torchtpu.layers.adapter.attention import (
                    PallasAttentionBackend,
                    PallasAttentionBackendImpl,
                )

                specs: list[api.ShapeDtypeStruct] = []
                for layer in attn_layers:
                    impl = getattr(layer, "impl", None)
                    if isinstance(impl, PallasAttentionBackendImpl):
                        kv_shape = PallasAttentionBackend.get_kv_cache_shape(
                            cfg.num_blocks,
                            cfg.block_size,
                            impl.num_kv_heads,
                            impl.head_size,
                            tpu_lowering.torch_dtype(act_dtype_str),
                        )
                        specs.append(
                            api.ShapeDtypeStruct(
                                shape=tuple(int(d) for d in kv_shape),
                                dtype=act_dtype_str,
                            )
                        )
                if len(specs) == cfg.num_layers:
                    return specs

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

        attn_layers = self._attention_layers()
        working_kvs = list(kv_caches)
        if len(attn_layers) == len(working_kvs):
            for layer, kv_buf in zip(attn_layers, working_kvs, strict=True):
                layer.kv_cache = kv_buf

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
                        return torch.func.functional_call(
                            backbone, call_weights, args=(), kwargs=kw
                        )
                    return backbone(**kw)

                try:
                    out = _call(
                        {
                            "input_ids": input_ids,
                            "positions": positions,
                            "kv_caches": working_kvs,
                            "attn_metadata": attn_metadata,
                        }
                    )
                except TypeError:
                    out = _call({"input_ids": input_ids, "positions": positions})
            finally:
                if prev_do_not_compile is not None:
                    backbone.do_not_compile = prev_do_not_compile

            updated_kvs = (
                [
                    layer.kv_cache[0]
                    if isinstance(layer.kv_cache, list)
                    else layer.kv_cache
                    for layer in attn_layers
                ]
                if len(attn_layers) == len(working_kvs)
                else working_kvs
            )
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
                        attn_metadata, self._vllm_config, num_tokens=num_tokens
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
