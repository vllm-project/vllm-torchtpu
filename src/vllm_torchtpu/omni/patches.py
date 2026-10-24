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
"""TPU runtime patches for vLLM-Omni diffusion models.

vLLM-Omni diffusion workers run in separate worker processes spawned by
vLLM-Omni's DiffusionEngine rather than through TPUWorker's worker_init /
model_load lifecycle stages, so OmniTpuPlatform applies the Omni patches
itself: apply_omni_tpu_patches() when the platform is constructed, and
apply_omni_model_specific_patches() at attention backend selection
(get_diffusion_attn_backend_cls). Tracked in #1862 for unification with
patch_registry.py.
"""

from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def _optimization_barrier_op() -> Any:
    from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (
        optimization_barrier_op,
    )

    return optimization_barrier_op()


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend the async output timeout for TPU compilation and execution."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Lazy execution attributes failure to sync point: In PyTorch/XLA
    #    and torch_tpu, forward passes are queued lazily in the computation
    #    graph. diffusion_worker emits COMPUTE_DONE prematurely before the TPU
    #    hardware finishes execution. When the async output background loop
    #    triggers host synchronization (.cpu()), it blocks on multi-step diffusion
    #    graph execution and initial compilation.
    # 2. Recompilation storms & JIT compile latency: Initial JIT/AOT
    #    compilation and weight distribution for large diffusion models (e.g.
    #    Wan 14B) take >600s, exceeding vLLM-Omni's default 600s timeout.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed/reduced once:
    # 1. OmniTpuPlatform.record_device_event() implements hardware
    #    synchronization (torch.accelerator.synchronize()) to ensure execution
    #    completes before COMPUTE_DONE.
    # 2. AOT compilation caching stabilizes startup compile times.
    # -------------------------------------------------------------------------
    import os

    os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""
    _patch_diffusion_ipc_cpu_pack()
    _patch_qwen_image_rope_txt_freqs_clone()
    _patch_wan_pair_swap_rope()
    _patch_wan_ffn_gelu_fence()
    _patch_diffusion_startup_warmup_sync()
    _patch_wan_fused_qk_norm_rope()
    _patch_wan_fresh_timestep()
    _patch_sdpa_fallback_dispatch()


def _patch_diffusion_ipc_cpu_pack() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Pickling a device tensor aborts process: In torch_tpu, calling
    #    pickle.dumps on a device tensor hits
    #    ABSL_CHECK(tensor.storage().allocator() == nullptr) in
    #    csrc/eager/tensor_to_buffer.cc:260-261, terminating the worker process
    #    with SIGABRT (exit code 134) with no Python traceback.
    # 2. libtpu per-process lockfile prevents cross-process init:
    #    Holding TPU tensors across process boundaries causes the CPU-only
    #    orchestrator process to trigger PJRT backend initialization during
    #    unpickling, colliding with the worker on /tmp/libtpu_lockfile.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed once torch_tpu supports safe inter-process tensor
    # serialization / proxy unpickling without triggering PJRT initialization
    # in host processes that do not own TPU hardware.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion import ipc

    orig_pack_tensor = ipc._pack_tensor_if_large

    def _safe_pack_tensor_if_large(val: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return orig_pack_tensor(val.cpu(), *args, **kwargs)

    ipc._pack_tensor_if_large = _safe_pack_tensor_if_large
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")


def _patch_qwen_image_rope_txt_freqs_clone() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once torch_tpu supports torch.real/torch.imag
    # inside torch.inference_mode() on a complex tensor created outside it.
    #
    # Qwen-Image RoPE modules return txt_freqs as a slice of pos_freqs, which
    # is created outside inference mode. The attention blocks take
    # torch.real/torch.imag of that slice inside inference mode, which fails
    # on TPU with "RuntimeError: Cannot set version_counter for inference
    # tensor". vid_freqs comes from torch.cat inside inference mode and is not
    # affected. Cloning txt_freqs once per transformer forward gives the
    # attention blocks an inference tensor of the same values.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer

    for rope_cls in (
        qwen_image_transformer.QwenEmbedRope,
        qwen_image_transformer.QwenEmbedLayer3DRope,
    ):
        rope_cls.forward = _clone_txt_freqs(rope_cls.forward)
    logger.info("Applied TPU patch: Qwen-Image RoPE txt_freqs clone.")


def _clone_txt_freqs(forward: Any) -> Any:
    def _forward(self: Any, *args: Any, **kwargs: Any):
        vid_freqs, txt_freqs = forward(self, *args, **kwargs)
        return vid_freqs, txt_freqs.clone()

    return _forward


def _patch_wan_pair_swap_rope() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once torch_tpu lowers size-2 minor unflatten/stack
    # without relayouts, or vLLM-Omni's CustomOp gains an out-of-tree dispatch
    # hook (then make this the TPU forward of RotaryEmbeddingWan).
    #
    # RotaryEmbeddingWan.forward_native (the CustomOp path non-CUDA platforms
    # take) rotates interleaved pairs via unflatten(-1, (-1, 2)) / stack(-1).
    # torch_tpu lowers the size-2 minor dimension to sequence-minor layouts
    # plus strided relayout copies around every q/k rotation, costing ~7 ms
    # per Wan2.2-A14B layer at 480p on v7x (~9% of the layer).
    # The replacement computes the same rotation on the (..., head_dim)
    # layout with an exact signed pair-swap matmul (see omni/rope.py). It is
    # installed on the class because CustomOp binds _forward_method at init
    # and the rotary op is constructed before the attention layer that
    # triggers these patches.
    #
    # Complex sliced views crash on TPU under inference mode, which rules
    # out the complex-number RoPE form.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan

    from vllm_torchtpu.omni.rope import rotary_embedding_wan_forward

    RotaryEmbeddingWan.forward = rotary_embedding_wan_forward
    logger.info("Applied TPU patch: pair-swap matmul RotaryEmbeddingWan.")


def _patch_wan_ffn_gelu_fence() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once XLA stops duplicating the elementwise producer
    # into the matmul operand fusion here, or torch_tpu exposes a fusion fence.
    #
    # XLA fuses the Wan FFN up-projection's bias add and tanh-GELU into the
    # operand fusion of the down-projection matmul, so the GELU of the
    # (tokens, 13824) activation is recomputed for every output tile of the
    # down-projection instead of once. torch_tpu has no way to stop a
    # producer fusion. An XLA optimization barrier on the GELU output (an
    # identity that fusion passes cannot see through) makes XLA materialize
    # it once, in bf16, which is also the value eager PyTorch produces.
    # Measured on v7x, Wan2.2-I2V-A14B: 72.72 -> 71.39 ms per block at
    # 32,760 tokens on one core; 0.939 -> 0.927 s/step end to end at
    # 480x832x81 on 8 cores (CFG 2 x Ulysses 4).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        ColumnParallelGELU,
    )

    orig_forward = getattr(
        ColumnParallelGELU, "_tpu_orig_forward", ColumnParallelGELU.forward
    )
    ColumnParallelGELU._tpu_orig_forward = orig_forward

    def _fenced_gelu_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        return _optimization_barrier_op()(orig_forward(self, x))

    ColumnParallelGELU.forward = _fenced_gelu_forward
    logger.info("Applied TPU patch: fusion fence after Wan FFN GELU.")


# TODO: Remove once vLLM-Omni synchronizes the device at the end of its
# startup warmup (e.g. self.executor.collective_rpc("synchronize_device") in
# _dummy_run).
def _patch_diffusion_startup_warmup_sync() -> None:
    """Synchronize the device after vLLM-Omni's diffusion startup warmup.

    Root cause: ``DiffusionEngine._dummy_run`` (startup warmup) drops its output
    without a host sync. Under torch_tpu DEFER_AND_FUSE the warmup's tail ops
    (VAE decode) stay pending in torch_tpu's process-global deferred-op queue and
    are materialized together with the first real request, which then gets cut
    into different fused programs than later identical requests (those start
    from an empty queue because the previous output's ``.cpu()`` flushed it).
    The first request after startup therefore differs numerically from later
    same-seed requests.

    Scope: this only takes effect when the engine runs its worker in-process
    (the uniproc executor, the single-device default). With the mp executor,
    only the worker processes install this patch, and they never call
    run_startup_warmup.
    """
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    orig_run_startup_warmup = DiffusionEngine.run_startup_warmup

    def _synced_run_startup_warmup(self: Any) -> None:
        orig_run_startup_warmup(self)
        # TODO: Once the vllm-omni pin is >= 0.30, sync every worker with
        # self.executor.collective_rpc("synchronize_device") and install this
        # patch at platform init, so that mp engines are covered too.
        torch.accelerator.synchronize()

    DiffusionEngine.run_startup_warmup = _synced_run_startup_warmup
    logger.info("Applied TPU patch: device sync after diffusion startup warmup.")


def _patch_wan_fused_qk_norm_rope() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once XLA fuses the q/k RMSNorm and RoPE into one
    # pass at HBM bandwidth, or vLLM-Omni's CustomOp gains an out-of-tree dispatch
    # hook for a fused q/k norm + RoPE op.
    #
    # WanSelfAttention normalizes q and k (RMSNorm over all heads) and then
    # rotates them. XLA splits this into a row-reduction fusion for the norm
    # statistics, an apply fusion and the RoPE fusion, each streaming the
    # (tokens, 5120) q and k through HBM; the row reduction also runs well
    # below HBM bandwidth. Together ~0.78 ms per block at 8190 tokens per
    # core on v7x. A Pallas kernel that reads q and k once and writes them
    # once takes 0.23 ms and is closer to an f64 reference (single rounding).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
    )

    from vllm_torchtpu.omni.rope import (
        can_fuse_qk_norm_rope,
        wan_self_attention_fused,
    )

    if hasattr(WanSelfAttention, "_tpu_orig_forward"):
        return
    WanSelfAttention._tpu_orig_forward = WanSelfAttention.forward

    def _fused_self_attention_forward(
        self: Any,
        hidden_states: torch.Tensor,
        rotary_emb: Any = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        if not can_fuse_qk_norm_rope(self, hidden_states, rotary_emb):
            return WanSelfAttention._tpu_orig_forward(
                self, hidden_states, rotary_emb, attn_metadata
            )
        return wan_self_attention_fused(self, hidden_states, rotary_emb, attn_metadata)

    WanSelfAttention.forward = _fused_self_attention_forward
    logger.info("Applied TPU patch: fused q/k RMSNorm + RoPE for Wan attention.")


def _patch_wan_fresh_timestep() -> None:
    # Wan: give the transformer its own copy of the timestep.
    #
    # Problem: Wan pipelines pass `t.expand(batch)`, a view into
    #   scheduler.timesteps. torch_tpu bakes a view's offset into every graph
    #   that reads it, so the transformer recompiles at every denoising step.
    # Fix: clone the timestep on entry and flush the clone right away, so only
    #   a tiny copy graph sees the view.
    # Effect (v7x, 480x832x81, 8 cores): first 12-step request 53.0 s -> 26.9 s
    #   (warm: 15.3 s); output is bit-identical.
    # Scope: WanTransformer3DModel (T2V, I2V, TI2V, both A14B experts),
    #   WanVACETransformer3DModel (overrides forward) and
    #   WanS2VTransformer3DModel (separate class). The numbers above are T2V;
    #   the VACE and S2V classes are covered but not measured on TPU.
    # TODO: remove once torch_tpu passes view offsets as runtime inputs.
    from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
        WanS2VTransformer3DModel,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanTransformer3DModel,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import (
        WanVACETransformer3DModel,
    )

    for cls in (
        WanTransformer3DModel,
        WanVACETransformer3DModel,
        WanS2VTransformer3DModel,
    ):
        _wrap_fresh_timestep(cls)
    logger.info(
        "Applied TPU patch: fresh timestep buffer for WanTransformer3DModel, "
        "WanVACETransformer3DModel and WanS2VTransformer3DModel."
    )


_FRESH_TIMESTEP_PATCHED = "_tpu_fresh_timestep_patched"


def _wrap_fresh_timestep(cls: type) -> None:
    # The flag lives on the forward function, not the class:
    # WanVACETransformer3DModel subclasses WanTransformer3DModel but has its
    # own forward, which still needs the wrapper.
    if getattr(cls.forward, _FRESH_TIMESTEP_PATCHED, False):
        return
    forward = cls.forward

    def _fresh_timestep_forward(
        self: Any, hidden_states: Any, timestep: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(timestep, torch.Tensor):
            from vllm_torchtpu.utils import synchronize_tensors

            timestep = timestep.clone()
            synchronize_tensors(timestep, wait=False)
        return forward(self, hidden_states, timestep, *args, **kwargs)

    setattr(_fresh_timestep_forward, _FRESH_TIMESTEP_PATCHED, True)
    cls.forward = _fresh_timestep_forward


_SDPA_FALLBACK_PATCHED = "_tpu_sdpa_fallback_patched"


def _patch_sdpa_fallback_dispatch() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's AttentionImpl.forward dispatches
    # out-of-tree platforms instead of raising NotImplementedError.
    #
    # Problem: Attention builds `sdpa_fallback` from vLLM-Omni's SDPAImpl,
    #   whatever the selected backend. AttentionImpl.forward picks
    #   forward_cuda/_npu/... from the platform and raises NotImplementedError
    #   on TPU, so every sdpa_fallback call fails there.
    # Fix: SDPAImpl.forward is replaced by TpuSDPAImpl.forward, which runs
    #   SDPAImpl._forward_impl with broadcast key masks, as forward_cuda does.
    # Effect: sdpa_fallback calls run on TPU instead of raising
    #   NotImplementedError. LTX-2's _LTX2ParallelAttention sends masked calls
    #   to sdpa_fallback: with sequence parallelism the audio latents are
    #   padded to a multiple of the SP size, so audio self-attention and
    #   audio-to-video cross-attention carry a key-padding mask. SANA-Video,
    #   LingBot-Video and SenseNova-U1 also call sdpa_fallback.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    from vllm_torchtpu.omni.attention import TpuSDPAImpl

    if getattr(SDPAImpl.forward, _SDPA_FALLBACK_PATCHED, False):
        return
    # The flag lands on TpuSDPAImpl.forward, the function shared by both classes.
    setattr(TpuSDPAImpl.forward, _SDPA_FALLBACK_PATCHED, True)
    SDPAImpl.forward = TpuSDPAImpl.forward
    logger.info("Applied TPU patch: SDPAImpl.forward routes to TpuSDPAImpl.")
