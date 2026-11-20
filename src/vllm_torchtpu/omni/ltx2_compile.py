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
"""Dynamo-traceable sequence/CFG-parallel primitives for LTX-2 / LTX-2.3 DiT.

Stacked PR #3444 activates ``_patch_ltx2_traceable_collectives`` and
``_patch_ltx2_symmetric_perturbation_masks`` on the compiled path
(``enforce_eager=False``) and initializes
``LTX2VideoToAudioParallelAttention._tpu_head_idx`` per rank outside
``torch.compile``. Leaving them inactive in eager mode avoids adding
compile-only mask allocations to eager runs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches import _ulysses_all_to_all_4d_layout
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_V2A_SP_PATCHED = "_tpu_v2a_sp_patched"
_PERTURBATION_SYM_PATCHED = "_tpu_perturbation_sym_patched"


def _funcol_all_to_all_single(send_buf: torch.Tensor, group: Any) -> torch.Tensor:
    return funcol.all_to_all_single(send_buf, None, None, group)


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
