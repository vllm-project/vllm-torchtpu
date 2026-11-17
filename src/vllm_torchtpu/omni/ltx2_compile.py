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
"""Dynamo-traceable sequence/CFG-parallel primitives and TPU diffusion compile.

Provides:
- Dynamo-traceable Ulysses / video-to-audio SP collectives and symmetric CFG
  perturbation masks for LTX-2 / LTX-2.3 DiT.
- Shared-block ``torch.compile(backend='tpu', fullgraph=True, dynamic=False)``
  for ``LTX2Pipeline`` and ``LTX2TwoStagePipeline`` plus allowlisted
  ``backend='tpu'`` regional/full compilation for verified diffusion families
  (``FluxTransformer2DModel`` and ``QwenImageTransformer2DModel``) without
  enabling ``OmniTpuPlatform.supports_torch_inductor()``.
"""

from __future__ import annotations

import functools
import types
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.nn.utils import stateless

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches import _ulysses_all_to_all_4d_layout
from vllm_torchtpu.omni.rope import _first_half_lanes
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

_V2A_SP_PATCHED = "_tpu_v2a_sp_patched"
_PERTURBATION_SYM_PATCHED = "_tpu_perturbation_sym_patched"
_LTX2_COMPILE_PATCHED = "_tpu_ltx2_compile_patched"
_TPU_DIFFUSION_COMPILE_PATCHED = "_tpu_diffusion_compile_patched"
_MXU_GATE_PATCHED = "_tpu_ltx2_mxu_gate_patched"
_ROPE_KEYS = frozenset(
    {
        "video_rotary_emb",
        "audio_rotary_emb",
        "ca_video_rotary_emb",
        "ca_audio_rotary_emb",
    }
)
# Caps compiled rotary-table copies across changing resolutions/layouts when
# slots change shape instead of overwriting in place.
_MAX_COMPILED_ROPE_CACHE_ENTRIES = 16


def _funcol_all_to_all_single(send_buf: torch.Tensor, group: Any) -> torch.Tensor:
    return funcol.all_to_all_single(send_buf, None, None, group)


def _build_gate_expand_matrix(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cols = torch.arange(heads * head_dim, device=device) // head_dim
    rows = torch.arange(heads, device=device).unsqueeze(1)
    return (rows == cols.unsqueeze(0)).to(dtype=dtype)


@functools.cache
def _cached_gate_expand_matrix(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _build_gate_expand_matrix(heads, head_dim, dtype, device)


def _patch_ltx2_mxu_gated_attention() -> None:
    """Expand LTX-2.3 per-head attention gates on MXU without 4D relayout copies."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    proc_cls = ltx2_transformer.LTX2AudioVideoAttnProcessor
    if getattr(proc_cls.__call__, _MXU_GATE_PATCHED, False):
        return

    def _mxu_gated_attn_call(
        self: Any,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_self_attention = encoder_hidden_states is None
        batch_size, sequence_length, _ = (
            hidden_states.shape if is_self_attention else encoder_hidden_states.shape
        )

        attention_mask = self._prepare_attention_mask(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        gate_logits: torch.Tensor | None = None
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)
            if isinstance(gate_logits, tuple):
                gate_logits = gate_logits[0]

        if is_self_attention:
            encoder_hidden_states = hidden_states

        query, key, value = self._project_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            is_self_attention=is_self_attention,
        )

        query = attn.norm_q(query).to(dtype=value.dtype)
        key = attn.norm_k(key).to(dtype=value.dtype)

        if query_rotary_emb is not None:
            query_rotary_emb = self._slice_rope_for_tp(query_rotary_emb, attn)
            if key_rotary_emb is not None:
                key_rotary_emb = self._slice_rope_for_tp(key_rotary_emb, attn)
            if attn.rope_type == "interleaved":
                query = ltx2_transformer.apply_interleaved_rotary_emb(
                    query, query_rotary_emb
                )
                key = ltx2_transformer.apply_interleaved_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                )
            elif attn.rope_type == "split":
                query = ltx2_transformer.apply_split_rotary_emb(
                    query, query_rotary_emb, head_dim=attn.head_dim
                )
                key = ltx2_transformer.apply_split_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                    head_dim=attn.head_dim,
                )

        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)

        query = query.unflatten(2, (attn.heads, attn.head_dim))
        key = key.unflatten(2, (attn.heads, attn.head_dim))
        value_heads = value.unflatten(2, (attn.heads, attn.head_dim))

        attn_metadata = (
            AttentionMetadata(attn_mask=attention_mask)
            if attention_mask is not None
            else None
        )
        hidden_states = attn.attn(query, key, value_heads, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        if perturbation_mask is not None:
            hidden_states = hidden_states * perturbation_mask + value * (
                1 - perturbation_mask
            )

        if gate_logits is not None:
            gates = 2.0 * torch.sigmoid(gate_logits)
            if hidden_states.dtype == torch.bfloat16 and gates.dtype == torch.bfloat16:
                expand_mat = (
                    _build_gate_expand_matrix(
                        attn.heads,
                        attn.head_dim,
                        gates.dtype,
                        hidden_states.device,
                    )
                    if torch.compiler.is_compiling()
                    else _cached_gate_expand_matrix(
                        attn.heads,
                        attn.head_dim,
                        gates.dtype,
                        hidden_states.device,
                    )
                )
                hidden_states = hidden_states * torch.matmul(gates, expand_mat)
            else:
                hidden_states = (
                    hidden_states.unflatten(2, (attn.heads, attn.head_dim))
                    * gates.unsqueeze(-1)
                ).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    setattr(_mxu_gated_attn_call, _MXU_GATE_PATCHED, True)
    proc_cls.__call__ = _mxu_gated_attn_call
    logger.info("Applied TPU patch: MXU channel-expanded LTX-2.3 gated attention.")


def _functional_all_to_all_4d(
    input_t: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: dist.ProcessGroup | None = None,
    use_sync: bool = False,
) -> torch.Tensor:
    """Functional ``all_to_all_4D`` lowering cleanly to ``_c10d_functional``."""
    del use_sync
    seq_world_size = dist.get_world_size(group)
    if seq_world_size <= 1:
        return input_t
    return _ulysses_all_to_all_4d_layout(
        input_t,
        scatter_idx,
        gather_idx,
        seq_world_size,
        group,
        _funcol_all_to_all_single,
    )


def _patch_ltx2_traceable_collectives() -> None:
    """Make Ulysses and LTX-2 video-to-audio SP collectives Dynamo-traceable.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/distributed/comm.py`` (``SeqAllToAll4D``),
    ``vllm_omni/diffusion/attention/parallel/ulysses.py``, and
    ``vllm_omni/diffusion/models/ltx2/ltx2_sequence_parallel.py``
    (``LTX2VideoToAudioParallelAttention.pre_attention`` and ``post_attention``).

    Activated by stacked PR #3444 on the compiled LTX-2 path, which also
    materializes ``LTX2VideoToAudioParallelAttention._tpu_head_idx`` per rank
    outside ``torch.compile``. Eager calls delegate unchanged to upstream.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni lowers Ulysses ``SeqAllToAll4D`` and
    ``LTX2VideoToAudioParallelAttention`` collectives through
    ``torch.distributed._functional_collectives`` when compiling.
    """
    from vllm_omni.diffusion.attention.parallel import ulysses as ulysses_mod
    from vllm_omni.diffusion.distributed import comm as comm_mod
    from vllm_omni.diffusion.models.ltx2 import ltx2_sequence_parallel as ltx2_sp_mod

    v2a_cls = ltx2_sp_mod.LTX2VideoToAudioParallelAttention
    if getattr(v2a_cls.pre_attention, _V2A_SP_PATCHED, False):
        return

    orig_seq_a2a = comm_mod.SeqAllToAll4D

    class _TraceableSeqAllToAll4D:
        @staticmethod
        def apply(
            group: dist.ProcessGroup,
            input_t: torch.Tensor,
            scatter_idx: int,
            gather_idx: int,
            use_sync: bool = False,
        ) -> torch.Tensor:
            if torch.compiler.is_compiling():
                return _functional_all_to_all_4d(
                    input_t,
                    scatter_idx=scatter_idx,
                    gather_idx=gather_idx,
                    group=group,
                    use_sync=use_sync,
                )
            return orig_seq_a2a.apply(group, input_t, scatter_idx, gather_idx, use_sync)

    # Replace the class binding (not just ``SeqAllToAll4D.apply``): TorchDynamo
    # intercepts ``.apply`` on ``torch.autograd.Function`` subclasses and
    # dispatches straight to ``.forward``, bypassing a monkeypatched ``.apply``.
    comm_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]
    ulysses_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]
    ltx2_sp_mod.SeqAllToAll4D = _TraceableSeqAllToAll4D  # type: ignore[assignment]

    orig_pre = v2a_cls.pre_attention
    orig_post = v2a_cls.post_attention

    def _traceable_v2a_pre(
        self: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any,
    ) -> Any:
        if not torch.compiler.is_compiling():
            return orig_pre(self, query, key, value, attn_metadata)
        world_size = self._sp_group.ulysses_world_size
        query_heads = int(query.shape[2])
        heads_per_rank = query_heads // world_size
        head_idx = getattr(self, "_tpu_head_idx", None)
        if head_idx is None:
            raise RuntimeError(
                "LTX2VideoToAudioParallelAttention._tpu_head_idx must be initialized "
                "outside torch.compile before compiled execution."
            )
        if head_idx.shape[0] != heads_per_rank:
            raise ValueError(
                f"Expected {heads_per_rank} heads in _tpu_head_idx, "
                f"got {head_idx.shape[0]}"
            )
        q_local = torch.index_select(query, 2, head_idx)
        k_full = _functional_all_to_all_4d(key, 2, 1, group=self._ulysses_pg)
        v_full = _functional_all_to_all_4d(value, 2, 1, group=self._ulysses_pg)
        ctx = ltx2_sp_mod._VideoToAudioContext(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            use_sync=self._use_sync,
        )
        return q_local, k_full, v_full, attn_metadata, ctx

    def _traceable_v2a_post(
        self: Any,
        attn_output: torch.Tensor,
        ctx: Any,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            return orig_post(self, attn_output, ctx)
        local_heads_first = attn_output.permute(2, 0, 1, 3).contiguous()
        gathered = funcol.all_gather_tensor(local_heads_first, 0, ctx.ulysses_pg)
        return gathered.permute(1, 2, 0, 3).contiguous()

    setattr(_traceable_v2a_pre, _V2A_SP_PATCHED, True)
    v2a_cls.pre_attention = _traceable_v2a_pre
    v2a_cls.post_attention = _traceable_v2a_post
    logger.info("Applied TPU patch: Dynamo-traceable LTX-2 Ulysses collectives.")


def _patch_ltx2_symmetric_perturbation_masks() -> None:
    """Keep perturbation masks structurally identical across all CFG-parallel ranks.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_guidance.py``
    (``LTXGuidanceExecutor.predict_parallel_guidance`` and
    ``build_perturbation_kwargs``).

    Activated by stacked PR #3444 on the compiled LTX-2 path. Under
    multi-guidance CFG parallelism, ranks whose local plan lacks ``ptb`` or
    ``mod`` omit those perturbation masks, which would otherwise produce a
    different compiled block graph signature per CFG rank and deadlock
    Ulysses collectives on step 0. Injecting all-ones masks of shape
    ``(len(local_plan.passes) * batch_size, 1, 1)`` keeps the compiled
    signature symmetric while leaving attention outputs unchanged
    (multiplying by ``1.0``).

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni supplies symmetric perturbation masks across
    CFG-parallel ranks or compiles rank-local graphs without cross-rank
    signature mismatch.
    """
    from vllm_omni.diffusion.models.ltx2 import ltx2_guidance

    exec_cls = ltx2_guidance.LTXGuidanceExecutor
    if getattr(exec_cls.predict_parallel_guidance, _PERTURBATION_SYM_PATCHED, False):
        return

    orig_build_perturbation = ltx2_guidance.build_perturbation_kwargs
    orig_predict_parallel = exec_cls.predict_parallel_guidance

    def _sym_predict_parallel_guidance(
        self: Any,
        pipeline: Any,
        plan: Any,
        index: int,
        timestep: torch.Tensor,
        state: Any,
        forward_ctx: Any,
        denoise_ctx: Any,
        preserve_positive_velocity: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def _sym_build_perturbation_kwargs(
            local_plan: Any,
            batch_size: int,
            reference: torch.Tensor,
        ) -> dict[str, Any]:
            kw = orig_build_perturbation(local_plan, batch_size, reference)
            local_total = len(local_plan.passes) * batch_size
            ones: torch.Tensor | None = None

            def _get_ones() -> torch.Tensor:
                nonlocal ones
                if ones is None:
                    ones = reference.new_ones((local_total, 1, 1))
                    # Materialize as a fresh zero-offset device buffer before
                    # entering the compiled block graph.
                    synchronize_tensors(ones, wait=False)
                return ones

            if "ptb" in plan.names and "ptb" not in local_plan.names:
                if plan.spec.video.do_stg:
                    kw["video_self_attention_mask"] = _get_ones()
                    kw["video_self_attention_blocks"] = plan.spec.video.stg_blocks
                if plan.spec.audio.do_stg:
                    kw["audio_self_attention_mask"] = _get_ones()
                    kw["audio_self_attention_blocks"] = plan.spec.audio.stg_blocks
            if "mod" in plan.names and "mod" not in local_plan.names:
                kw["a2v_cross_attention_mask"] = _get_ones()
                kw["v2a_cross_attention_mask"] = _get_ones()
            return kw

        ltx2_guidance.build_perturbation_kwargs = _sym_build_perturbation_kwargs
        try:
            return orig_predict_parallel(
                self,
                pipeline,
                plan,
                index,
                timestep,
                state,
                forward_ctx,
                denoise_ctx,
                preserve_positive_velocity=preserve_positive_velocity,
            )
        finally:
            ltx2_guidance.build_perturbation_kwargs = orig_build_perturbation

    setattr(_sym_predict_parallel_guidance, _PERTURBATION_SYM_PATCHED, True)
    exec_cls.predict_parallel_guidance = _sym_predict_parallel_guidance
    logger.info("Applied TPU patch: symmetric LTX-2 CFG perturbation masks.")


def _ensure_zero_offset(
    val: Any,
    rope_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] | None = None,
    slot: Any = None,
) -> Any:
    """Materialize strided or non-zero-offset views before compiled block entry."""
    if isinstance(val, torch.Tensor):
        if (
            val.is_contiguous()
            and val.storage_offset() == 0
            and val.untyped_storage().nbytes() == val.numel() * val.element_size()
        ):
            return val
        if rope_cache is not None:
            ptr = val.untyped_storage().data_ptr()
            if ptr != 0:
                # One entry per rotary slot and view layout: each request builds
                # new tables, whose copy replaces the previous request's entry
                # (and releases its full tables) instead of piling up next to it.
                key = (
                    slot,
                    val.storage_offset(),
                    tuple(val.shape),
                    val.stride(),
                    val.dtype,
                    val.device,
                )
                entry = rope_cache.get(key)
                if entry is not None and entry[0].untyped_storage().data_ptr() == ptr:
                    return entry[1]
                mat = fresh_buffer(val)
                synchronize_tensors(mat, wait=False)
                if (
                    entry is None
                    and len(rope_cache) >= _MAX_COMPILED_ROPE_CACHE_ENTRIES
                ):
                    rope_cache.clear()
                rope_cache[key] = (val, mat)
                return mat
        mat = torch.empty(val.shape, dtype=val.dtype, device=val.device)
        mat.copy_(val)
        synchronize_tensors(mat, wait=False)
        return mat
    if isinstance(val, tuple):
        return tuple(
            _ensure_zero_offset(x, rope_cache, (slot, i)) for i, x in enumerate(val)
        )
    return val


def _install_shared_block_compile(transformer: torch.nn.Module) -> None:
    """Compile ONE shared block function and bind all LTX-2 blocks to it.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_transformer.py``
    (``LTX2VideoTransformer3DModel.transformer_blocks`` and
    ``LTX2VideoTransformerBlock``).

    Compiling each block separately would trigger N distinct XLA compilations
    of the same block graph. ``stateless._reparametrize_module`` swaps each
    block's parameters and buffers into ``blocks[0]`` so every block executes
    the single compiled graph. If any block has a different parameter or buffer
    key set (for example, heterogeneous phase-adapter LoRA targets), compilation
    is skipped so blocks never silently reuse ``blocks[0]`` tensors.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni supports shared-block ``torch.compile`` on TPU for
    LTX-2 transformer blocks natively.
    """
    blocks = transformer.transformer_blocks
    if len(blocks) == 0:
        return
    if getattr(transformer, "_tpu_shared_blocks_compiled", False):
        return

    template_block = blocks[0]
    expected_param_keys = {k for k, _ in template_block.named_parameters()}
    expected_buffer_keys = {k for k, _ in template_block.named_buffers()}
    for idx, blk in enumerate(blocks[1:], start=1):
        param_keys = {k for k, _ in blk.named_parameters()}
        buffer_keys = {k for k, _ in blk.named_buffers()}
        if param_keys != expected_param_keys or buffer_keys != expected_buffer_keys:
            logger.warning(
                "Skipping shared-block TPU torch.compile: block %d has different "
                "parameter/buffer keys from block 0; running eager.",
                idx,
            )
            return

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_world_size,
        get_sp_group,
    )

    device = next(transformer.parameters()).device
    cfg = transformer.config
    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    audio_inner_dim = cfg.audio_num_attention_heads * cfg.audio_attention_head_dim
    _first_half_lanes(inner_dim, cfg.attention_head_dim, device)
    _first_half_lanes(audio_inner_dim, cfg.audio_attention_head_dim, device)

    sp_world = get_sequence_parallel_world_size()
    if sp_world > 1:
        sp_group = get_sp_group()
        uworld = sp_group.ulysses_world_size
        if uworld > 1:
            if cfg.audio_num_attention_heads % uworld != 0:
                raise ValueError(
                    "LTX-2 shared-block compile requires audio_num_attention_heads "
                    f"divisible by ulysses_world_size, got "
                    f"audio_num_attention_heads={cfg.audio_num_attention_heads} "
                    f"and ulysses_world_size={uworld}."
                )
            urank = sp_group.ulysses_rank
            heads_per_rank = cfg.audio_num_attention_heads // uworld
            start = urank * heads_per_rank
            head_idx = torch.arange(
                start, start + heads_per_rank, dtype=torch.int32, device=device
            )
            synchronize_tensors(head_idx, wait=False)
            from vllm_omni.diffusion.models.ltx2.ltx2_sequence_parallel import (
                LTX2VideoToAudioParallelAttention,
            )

            for blk in blocks:
                inner_attn = blk.video_to_audio_attn.attn
                if inner_attn.ltx_sp_mode == "video_to_audio":
                    if inner_attn._video_to_audio_strategy is None:
                        inner_attn._video_to_audio_strategy = (
                            LTX2VideoToAudioParallelAttention(
                                sp_group,
                                use_sync=inner_attn.use_sync,
                            )
                        )
                    inner_attn._video_to_audio_strategy._tpu_head_idx = head_idx

    orig_block_forward = type(template_block).forward
    rope_view_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _compiled_block_step(
        params_and_buffers: dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        block_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with stateless._reparametrize_module(template_block, params_and_buffers):
            return orig_block_forward(
                template_block,
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                **block_kwargs,
            )

    for blk in blocks:
        state_dict_tensors: dict[str, torch.Tensor] = {}
        for k, p in blk.named_parameters():
            state_dict_tensors[k] = p.detach()
        for k, b in blk.named_buffers():
            state_dict_tensors[k] = b.detach()

        def _make_bound_forward(pb: dict[str, torch.Tensor]) -> Any:
            @functools.wraps(orig_block_forward)
            def _bound_forward(
                _self: Any,
                hidden_states: torch.Tensor,
                audio_hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                audio_encoder_hidden_states: torch.Tensor,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                hidden_states = _ensure_zero_offset(hidden_states)
                audio_hidden_states = _ensure_zero_offset(audio_hidden_states)
                encoder_hidden_states = _ensure_zero_offset(encoder_hidden_states)
                audio_encoder_hidden_states = _ensure_zero_offset(
                    audio_encoder_hidden_states
                )
                clean_kwargs = {
                    k: _ensure_zero_offset(
                        v, rope_view_cache if k in _ROPE_KEYS else None, k
                    )
                    for k, v in kwargs.items()
                }
                return _compiled_block_step(
                    pb,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    clean_kwargs,
                )

            return _bound_forward

        bound_fn = _make_bound_forward(state_dict_tensors)
        blk.forward = types.MethodType(bound_fn, blk)  # type: ignore[method-assign]

    transformer._tpu_shared_blocks_compiled = True  # type: ignore[attr-defined]
    logger.info(
        "Installed shared torch.compile(backend='tpu', fullgraph=True, dynamic=False) "
        "callable across %d LTX2VideoTransformerBlock modules.",
        len(blocks),
    )


def _activate_ltx2_compile_runtime_hooks() -> None:
    """Install traceable collectives, symmetric masks, and fresh-timestep forward.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_transformer.py``
    (``LTX2VideoTransformer3DModel.forward``).

    Clones ``timestep`` / ``sigma`` views into fresh zero-offset buffers before
    entering the compiled transformer blocks (preventing per-step view-offset
    recompilations) and issues a non-blocking flush of transformer outputs
    before CFG-parallel guidance ``all_gather``.

    REMOVAL / FIX CONDITION:
    Remove once TorchTPU handles non-zero-offset scalar views without
    recompiling and flushes ``ulysses_pg`` outputs before ``cfg_pg``
    collectives automatically.
    """
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    model_cls = ltx2_transformer.LTX2VideoTransformer3DModel
    if getattr(model_cls.forward, _LTX2_COMPILE_PATCHED, False):
        return

    _patch_ltx2_traceable_collectives()
    _patch_ltx2_symmetric_perturbation_masks()

    orig_model_forward = model_cls.forward

    @functools.wraps(orig_model_forward)
    def _fresh_ltx2_forward(
        self: Any,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        *args: Any,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        to_sync: list[torch.Tensor] = []

        def _fresh(t: torch.Tensor | None) -> torch.Tensor | None:
            if isinstance(t, torch.Tensor):
                c = t.clone(memory_format=torch.contiguous_format)
                to_sync.append(c)
                return c
            return t

        timestep = _fresh(timestep)  # type: ignore[assignment]
        audio_timestep = _fresh(audio_timestep)
        sigma = _fresh(sigma)
        audio_sigma = _fresh(audio_sigma)
        if to_sync:
            synchronize_tensors(to_sync, wait=False)

        res = orig_model_forward(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            audio_encoder_hidden_states,
            timestep,
            audio_timestep,
            *args,
            sigma=sigma,
            audio_sigma=audio_sigma,
            **kwargs,
        )
        # Flush ulysses_pg output tensors before cfg_pg guidance all_gather.
        out_tensors: list[torch.Tensor] = []
        if isinstance(res, tuple):
            out_tensors = [t for t in res if isinstance(t, torch.Tensor)]
        else:
            for attr in ("sample", "audio_sample"):
                t = getattr(res, attr, None)
                if isinstance(t, torch.Tensor):
                    out_tensors.append(t)
        if out_tensors:
            synchronize_tensors(out_tensors, wait=False)
        return res

    setattr(_fresh_ltx2_forward, _LTX2_COMPILE_PATCHED, True)
    model_cls.forward = _fresh_ltx2_forward


@functools.cache
def _verified_tpu_compile_transformer_types() -> tuple[type[torch.nn.Module], ...]:
    """Diffusion transformer classes verified with ``backend='tpu'`` compile.

    Flux and Qwen-Image pass end-to-end TPU v7x verification with regional
    ``backend='tpu'`` compilation. Unverified models (including Wan, whose
    ``@run_once`` Pallas custom ops fail Dynamo schema tracing) stay eager.
    LTX-2 uses shared-block compilation in ``_apply_tpu_diffusion_compile``.
    """
    from vllm_omni.diffusion.models.flux.flux_transformer import (
        FluxTransformer2DModel,
    )
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    return (FluxTransformer2DModel, QwenImageTransformer2DModel)


def _compile_verified_tpu_transformer(
    pipeline: Any, od_config: Any, attr_name: str
) -> None:
    """Compile one allowlisted diffusion transformer with ``backend='tpu'``.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/worker/diffusion_model_runner.py``
    (``DiffusionModelRunner._compile_transformer``) and
    ``vllm_omni/diffusion/compile.py`` (``regionally_compile``).

    Mirrors upstream granularity dispatch (``full`` vs ``regional``) with
    ``backend='tpu', dynamic=False`` for verified diffusion transformer
    families only; unverified models log once and stay eager.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni supports platform-provided ``torch.compile`` kwargs
    and per-model TPU compile allowlisting in ``DiffusionModelRunner``.
    """
    model = getattr(pipeline, attr_name, None)
    if model is None:
        return
    if not isinstance(model, _verified_tpu_compile_transformer_types()):
        logger.info_once(
            "Model runner: skipping TPU torch.compile for unverified "
            "transformer %s (%s); running eager.",
            attr_name,
            type(model).__name__,
        )
        return
    from vllm_omni.diffusion import compile as diffusion_compile

    compile_kwargs = {"backend": "tpu", "dynamic": False}
    if od_config.diffusion_compile_granularity == "full":
        model.compile(**compile_kwargs)
        compiled_model = model
    else:
        compiled_model = diffusion_compile.regionally_compile(model, **compile_kwargs)
    setattr(pipeline, attr_name, compiled_model)


def _apply_tpu_diffusion_compile(pipeline: Any, od_config: Any) -> None:
    """Apply ``backend='tpu'`` compilation to verified diffusion pipelines."""
    if od_config.enforce_eager or pipeline is None:
        return
    from vllm_omni.diffusion.models.ltx2 import (
        pipeline_ltx2,
        pipeline_ltx2_two_stage,
    )

    # The two-stage pipeline shares LTXRuntime and the LTX-2 transformer block
    # with LTX2Pipeline but does not subclass it. Both expose ``transformer``
    # (``dit_modules=("transformer",)``) with no ``transformer_2``.
    if isinstance(
        pipeline,
        (pipeline_ltx2.LTX2Pipeline, pipeline_ltx2_two_stage.LTX2TwoStagePipeline),
    ):
        _activate_ltx2_compile_runtime_hooks()
        if pipeline.transformer is not None:
            _install_shared_block_compile(pipeline.transformer)
        return

    transformer_attrs = getattr(pipeline, "_dit_modules", None) or (
        "transformer",
        "transformer_2",
    )
    for attr_name in transformer_attrs:
        _compile_verified_tpu_transformer(pipeline, od_config, attr_name)


def apply_ltx2_compile_patches() -> None:
    """Hook diffusion model loading to run ``backend='tpu'`` compilation.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/worker/diffusion_model_runner.py``
    (``DiffusionModelRunner.load_model`` calling ``enable_offload_backend``
    right before the ``torch.compile`` block).

    ``OmniTpuPlatform.supports_torch_inductor()`` stays ``False`` because
    non-diffusion Omni models (Qwen3 code predictor, VoxCPM2 talker) and
    upstream pipeline ``setup_compile()`` hooks use it to enable default-backend
    Inductor compile. Wrapping ``diffusion_model_runner.enable_offload_backend``
    delegates to the original offloader and runs ``backend='tpu'`` compilation
    for verified diffusion families at the exact post-load / pre-cache boundary.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni exposes a platform hook for non-Inductor diffusion
    ``torch.compile`` backends instead of gating diffusion compile on
    ``supports_torch_inductor()``.
    """
    from vllm_omni.diffusion.worker import diffusion_model_runner

    _patch_ltx2_mxu_gated_attention()

    orig_enable_offload = diffusion_model_runner.enable_offload_backend
    if getattr(orig_enable_offload, _TPU_DIFFUSION_COMPILE_PATCHED, False):
        return

    @functools.wraps(orig_enable_offload)
    def _tpu_enable_offload_and_compile(
        od_config: Any,
        pipeline: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pipeline, offload_backend = orig_enable_offload(
            od_config, pipeline, *args, **kwargs
        )
        _apply_tpu_diffusion_compile(pipeline, od_config)
        return pipeline, offload_backend

    setattr(_tpu_enable_offload_and_compile, _TPU_DIFFUSION_COMPILE_PATCHED, True)
    diffusion_model_runner.enable_offload_backend = _tpu_enable_offload_and_compile
    logger.info("Applied TPU patch: diffusion backend='tpu' torch.compile hook.")
