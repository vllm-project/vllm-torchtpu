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
itself: apply_omni_tpu_patches() when the platform is constructed,
set_device() for worker runtime/RoPE/thread patches, and
apply_omni_model_specific_patches() at attention backend selection
(get_diffusion_attn_backend_cls). Tracked in #1862 for unification with
patch_registry.py.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

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
    _patch_flow_match_scheduler_step_flush()
    _patch_group_norm_tpu()
    _patch_vae_image_processor_pil_output()


@run_once
def apply_omni_worker_patches() -> None:
    """Apply diffusion-worker runtime patches on ``OmniTpuPlatform.set_device``."""
    _patch_diffusion_worker_runtime_threads()
    _patch_rotary_embedding_tpu()


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""
    _patch_diffusion_ipc_cpu_pack()
    _patch_qwen_image_rope_txt_freqs_clone()
    _patch_wan_pair_swap_rope()
    _patch_wan_ffn_gelu_fence()
    _patch_diffusion_worker_d2h_stream()
    _patch_diffusion_startup_warmup_sync()
    _patch_ltx2_split_rope()
    _patch_wan_fused_qk_norm_rope()
    _patch_wan_fresh_timestep()
    _patch_sdpa_fallback_dispatch()
    _patch_ltx2_vocoder_cpu_init()
    _patch_z_image_transformer_block_flush()
    _patch_z_image_prompt_encoder_tpu()


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


_D2H_STREAM_PATCHED = "_tpu_d2h_stream_patched"


def _patch_diffusion_worker_d2h_stream() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni creates the D2H stream of
    # WorkerProc._async_output_loop (vllm_omni/diffusion/worker/diffusion_worker.py)
    # on the current device index instead of
    # torch.device(torch.accelerator.current_accelerator().type, self.gpu_id).
    #
    # Problem: WorkerProc._async_output_loop creates its D2H stream on device
    #   index self.gpu_id. On TPU, OmniTpuPlatform.set_device selects device
    #   index 0 in every worker process, so a worker with gpu_id >= 1 asks for
    #   a stream on a device index that its process did not select.
    # Fix: run the loop with a proxy as self whose gpu_id is the current
    #   device index; other attribute reads and writes go to the worker.
    # Effect: the output thread of a worker with gpu_id >= 1 gets its stream
    #   instead of failing at startup.
    # Scope: the proxy is live for the whole output-thread lifetime, not only
    #   while the stream is created. Only attribute reads/writes are
    #   forwarded, so identity/type checks on self inside the loop would see
    #   the proxy (the upstream loop does none).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

    orig_async_output_loop = WorkerProc._async_output_loop
    if getattr(orig_async_output_loop, _D2H_STREAM_PATCHED, False):
        return

    def _tpu_async_output_loop(self: Any) -> None:
        orig_async_output_loop(_CurrentDeviceWorkerProxy(self))

    setattr(_tpu_async_output_loop, _D2H_STREAM_PATCHED, True)
    WorkerProc._async_output_loop = _tpu_async_output_loop
    logger.info("Applied TPU patch: diffusion D2H stream on the current device.")


class _CurrentDeviceWorkerProxy:
    """Forward attribute access to a worker, reporting the current device as
    ``gpu_id``."""

    def __init__(self, worker: Any) -> None:
        object.__setattr__(self, "_worker", worker)

    def __getattr__(self, name: str) -> Any:
        if name == "gpu_id":
            return torch.accelerator.current_device_index()
        return getattr(self._worker, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._worker, name, value)


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


_VOCODER_CPU_INIT_PATCHED = "_tpu_vocoder_cpu_init_patched"


def _patch_ltx2_vocoder_cpu_init() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once torch_tpu implements kaiser_window and sinc, or
    # vLLM-Omni builds the vocoder and runs
    # ltx2_runtime._restore_ltx_bwe_resampler_filter on CPU.
    #
    # Problem: DiffusersPipelineLoader._init_from_load_format constructs the
    #   pipeline under a `with torch.device("tpu:N"):` context. LTX-2.3 and
    #   LTX-2.5 use Diffusers LTX2VocoderWithBWE (`act_fn="snakebeta"`,
    #   `antialias=True`), whose __init__ builds AntiAliasAct1d -> UpSample1d
    #   -> kaiser_sinc_filter1d and calls torch.kaiser_window(...) on the
    #   active default device before weights are loaded, failing on TPU with
    #   `DispatchStub: missing kernel for tpu`. At decode,
    #   ltx2_runtime._restore_ltx_bwe_resampler_filter rebuilds the BWE
    #   resampler's FP32 Hann filter with torch.sinc on the filter's device,
    #   failing on TPU with `operator 'aten::sinc.out' is not implemented`.
    # Fix: Constructing the vocoder on CPU (matching text_encoder and
    #   latent_upsampler in ltx2_components.py) builds the filter buffers on
    #   the host before _place_aux_components moves the module to
    #   pipeline.device. The restore helper runs upstream's rebuild on a CPU
    #   stand-in filter and moves the result to the filter's device, so later
    #   decodes hit upstream's identity check and skip the rebuild.
    # Effect: LTX-2.3 loads and decodes on TPU.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_runtime

    _wrap_ltx2_bwe_filter_restore(ltx2_runtime)
    if getattr(ltx2_components._load_component, _VOCODER_CPU_INIT_PATCHED, False):
        return
    orig_load_component = ltx2_components._load_component

    def _cpu_vocoder_load_component(
        component_cls: Any,
        model: str,
        subfolder: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if subfolder == "vocoder":
            with torch.device("cpu"):
                return orig_load_component(
                    component_cls, model, subfolder, *args, **kwargs
                )
        return orig_load_component(component_cls, model, subfolder, *args, **kwargs)

    setattr(_cpu_vocoder_load_component, _VOCODER_CPU_INIT_PATCHED, True)
    ltx2_components._load_component = _cpu_vocoder_load_component
    logger.info("Applied TPU patch: CPU-context init for LTX-2 vocoder.")


_BWE_FILTER_CPU_RESTORE_PATCHED = "_tpu_bwe_filter_cpu_restore_patched"


def _wrap_ltx2_bwe_filter_restore(ltx2_runtime: Any) -> None:
    # _run_ltx_vocoder resolves the helper as an ltx2_runtime global at call
    # time, so replacing the module attribute covers every caller.
    restore = ltx2_runtime._restore_ltx_bwe_resampler_filter
    if getattr(restore, _BWE_FILTER_CPU_RESTORE_PATCHED, False):
        return

    def _cpu_restore_ltx_bwe_resampler_filter(vocoder: Any) -> None:
        resampler: Any = getattr(vocoder, "resampler", None)
        current: Any = getattr(resampler, "filter", None)
        previous = getattr(resampler, "_ltx_restored_filter", None)
        if (
            not isinstance(current, torch.Tensor)
            or current.device.type == "cpu"
            or current is previous
        ):
            return restore(vocoder)
        # Upstream reads only the filter's device, never its values.
        resampler.filter = torch.empty_like(current, device="cpu")
        try:
            restore(vocoder)
        finally:
            rebuilt: Any = getattr(resampler, "_ltx_restored_filter", None)
            if rebuilt is previous or resampler.filter is not rebuilt:
                resampler.filter = current
            else:
                rebuilt = rebuilt.to(current.device)
                resampler.filter = rebuilt
                resampler._ltx_restored_filter = rebuilt

    setattr(
        _cpu_restore_ltx_bwe_resampler_filter, _BWE_FILTER_CPU_RESTORE_PATCHED, True
    )
    ltx2_runtime._restore_ltx_bwe_resampler_filter = (
        _cpu_restore_ltx_bwe_resampler_filter
    )
    logger.info("Applied TPU patch: CPU rebuild of the LTX-2 BWE resampler filter.")


_SPLIT_ROPE_PATCHED = "_tpu_split_rope_patched"


def _patch_ltx2_split_rope() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTX-2 split RoPE rotates q and
    # k in their (B, T, H * D) layout instead of reshaping them per head.
    #
    # Problem: LTX2AudioVideoRotaryPosEmbed builds split-RoPE tables as
    #   (B, H, T, D/2), so apply_split_rotary_emb transposes every (B, T, H * D)
    #   q and k to (B, H, T, D) and back in each attention layer. The pipeline
    #   passes the same coords tensor at every denoising step, and the rope
    #   modules rebuild the tables each time.
    # Fix: the rope module returns the tables in the layout of q and k,
    #   (B, 1, T, H * D) with the sign of each half folded into sin
    #   (split_rope_natural_tables), and apply_split_rotary_emb rotates them
    #   elementwise. Each rope module keeps the tables of its last split-RoPE
    #   call and returns them while the key matches, until a call with
    #   another key replaces them.
    # Effect: LTX-2 T2V 704x1280x121, 40 steps, on TPU v7x: 0.770 -> 0.482
    #   s/step (-37%).
    # Scope:
    #   - T stays at dim 2, where the split-RoPE sequence-parallel (SP) plan
    #     shards the tables.
    #   - TP: _slice_rope_for_tp slices 4-D tables by heads on dim 1 and so
    #     passes these (B, 1, T, H * D) tables through whole; the rotation
    #     takes the lanes of the local heads.
    #   - Cache key: the storage of coords (its data pointer; Tensor.is_set_to
    #     is not implemented for TPU), the layout of coords on it (storage
    #     offset, shape, stride), its dtype, device and version counter, and
    #     the device and out_dtype arguments. Fresh views with the same layout,
    #     such as the cross-attention ropes' coords[:, 0:1, :] each step, hit
    #     in and out of inference mode. The cache holds the coords view, which
    #     keeps the storage alive, so its address is not reused while cached.
    #     Inference tensors have no version counter, so in-place edits of
    #     coords in inference mode are not seen (vLLM-Omni does not edit
    #     coords after building them).
    #   - Interleaved RoPE and (B, H, T, D/2) tables use the original
    #     functions.
    #   - synchronize_tensors materializes the cached tables once.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    from vllm_torchtpu.omni.rope import (
        apply_split_rotary_half_swap,
        split_rope_natural_tables,
    )
    from vllm_torchtpu.utils import synchronize_tensors

    rope_cls = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed
    if getattr(rope_cls.forward, _SPLIT_ROPE_PATCHED, False):
        return
    orig_rope_forward = rope_cls.forward
    orig_apply_split_rotary_emb = ltx2_transformer.apply_split_rotary_emb

    def _cached_rope_forward(
        self: Any,
        coords: torch.Tensor,
        device: str | torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_type != "split":
            return orig_rope_forward(self, coords, device, out_dtype)
        storage = coords.untyped_storage().data_ptr()
        version = None if coords.is_inference() else coords._version
        key = (
            version,
            coords.storage_offset(),
            tuple(coords.shape),
            coords.stride(),
            coords.dtype,
            coords.device,
            device,
            out_dtype,
        )
        cached = getattr(self, "_tpu_split_rope_cache", None)
        if (
            cached is not None
            and storage != 0
            and cached[0].untyped_storage().data_ptr() == storage
            and cached[1] == key
        ):
            return cached[2]
        tables = orig_rope_forward(self, coords, device, out_dtype)
        if tables[0].ndim != 4:
            return tables
        tables = split_rope_natural_tables(*tables)
        synchronize_tensors(list(tables), wait=False)
        self._tpu_split_rope_cache = (coords, key, tables)
        return tables

    def _apply_split_rotary_emb(
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        *,
        head_dim: int,
    ) -> torch.Tensor:
        cos, sin = freqs
        if cos.ndim != 4 or cos.shape[1] != 1 or cos.shape[-1] == head_dim // 2:
            return orig_apply_split_rotary_emb(x, freqs, head_dim=head_dim)
        width = x.shape[-1]
        if cos.shape[-1] != width:
            tp_size = ltx2_transformer.get_tensor_model_parallel_world_size()
            if cos.shape[-1] != width * tp_size:
                raise ValueError(
                    f"LTX-2 split RoPE tables of width {cos.shape[-1]} do not "
                    f"match q/k of width {width} on {tp_size} tensor-parallel "
                    "ranks; the tables must cover all heads of all ranks."
                )
            start = ltx2_transformer.get_tensor_model_parallel_rank() * width
            cos = cos[..., start : start + width]
            sin = sin[..., start : start + width]
        return apply_split_rotary_half_swap(x, cos, sin, head_dim)

    setattr(_cached_rope_forward, _SPLIT_ROPE_PATCHED, True)
    rope_cls.forward = _cached_rope_forward
    ltx2_transformer.apply_split_rotary_emb = _apply_split_rotary_emb
    logger.info("Applied TPU patch: cached LTX-2 split RoPE on the q/k layout.")


_ZIMAGE_BLOCK_PATCHED = "_tpu_zimage_block_patched"
_ZIMAGE_MODEL_PATCHED = "_tpu_zimage_model_patched"
_ZIMAGE_PROMPT_ENCODER_PATCHED = "_tpu_zimage_prompt_encoder_patched"
_VAE_PIL_OUTPUT_PATCHED = "_tpu_vae_pil_output_patched"
_WORKER_THREADS_PATCHED = "_tpu_worker_threads_patched"


def _pt_images_to_pil(images: torch.Tensor) -> list[Any]:
    """diffusers' ``numpy_to_pil(pt_to_numpy(images))`` computed in torch."""
    # pyrefly: ignore[missing-import]
    from PIL import Image

    u8 = images.detach().cpu().to(torch.float32, copy=True).mul_(255).round_()
    u8 = u8.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    if u8.shape[-1] == 1:
        return [Image.fromarray(im.squeeze(-1), mode="L") for im in u8]
    return [Image.fromarray(im) for im in u8]


def _patch_vae_image_processor_pil_output() -> None:
    # TODO: Remove once diffusers' pt_to_numpy / numpy_to_pil converts in
    # contiguous torch/numpy rather than strided NHWC loops.
    # pyrefly: ignore[missing-import]
    from diffusers.image_processor import VaeImageProcessor

    orig_postprocess = VaeImageProcessor.postprocess
    if getattr(orig_postprocess, _VAE_PIL_OUTPUT_PATCHED, False):
        return

    def _postprocess(
        self: Any,
        image: Any,
        output_type: str = "pil",
        do_denormalize: Any = None,
    ) -> Any:
        if (
            output_type != "pil"
            or not isinstance(image, torch.Tensor)
            or image.dim() != 4
        ):
            return orig_postprocess(
                self, image, output_type=output_type, do_denormalize=do_denormalize
            )
        image = orig_postprocess(
            self, image, output_type="pt", do_denormalize=do_denormalize
        )
        return _pt_images_to_pil(image)

    setattr(_postprocess, _VAE_PIL_OUTPUT_PATCHED, True)
    VaeImageProcessor.postprocess = _postprocess
    logger.info("Applied TPU patch: torch-native PIL output in VaeImageProcessor.")


_GROUP_NORM_PATCHED = "_tpu_group_norm_patched"
_SCHEDULER_STEP_PATCHED = "_tpu_scheduler_step_patched"
_ROTARY_EMBEDDING_PATCHED = "_tpu_rotary_embedding_patched"


def _patch_rotary_embedding_tpu() -> None:
    # TODO: Remove once vLLM-Omni's RotaryEmbedding.forward_native avoids
    # minor-dim strided slices or torch_tpu fuses them without relayouts.
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding

    from vllm_torchtpu.omni.rope import tpu_apply_rotary

    orig_forward_native = RotaryEmbedding.forward_native
    if getattr(orig_forward_native, _ROTARY_EMBEDDING_PATCHED, False):
        return

    def _forward_native(
        self: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if x.device.type != "tpu" or x.dtype != torch.bfloat16 or x.dim() < 3:
            return orig_forward_native(self, x, cos, sin)
        cos_h, sin_h = self._prepare_half_head_dim_cos_sin(cos, sin)
        if (
            cos_h.dim() != 2
            or cos_h.shape[-1] * 2 != x.shape[-1]
            or cos_h.shape != sin_h.shape
        ):
            return orig_forward_native(self, x, cos, sin)
        return tpu_apply_rotary(
            x, cos_h.to(x.dtype), sin_h.to(x.dtype), bool(self.interleaved)
        )

    setattr(_forward_native, _ROTARY_EMBEDDING_PATCHED, True)
    RotaryEmbedding.forward_native = _forward_native
    logger.info(
        "Applied TPU patch: matmul-based rotate_half in RotaryEmbedding.forward_native."
    )


def _tpu_group_norm(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """GroupNorm reduced over ``[N, G, C/G, *spatial]`` in fp32."""
    n, c = input.shape[0], input.shape[1]
    if c % num_groups != 0:
        raise ValueError(
            f"Expected channels ({c}) to be divisible by num_groups ({num_groups})."
        )
    spatial = tuple(input.shape[2:])
    x = input.reshape(n, num_groups, c // num_groups, *spatial).float()
    reduce_dims = tuple(range(2, x.dim()))
    mean = x.mean(reduce_dims, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(reduce_dims, keepdim=True)
    y = (xc * torch.rsqrt(var + eps)).reshape(input.shape)
    affine_shape = (1, c) + (1,) * len(spatial)
    if weight is not None:
        y = y * weight.reshape(affine_shape).float()
    if bias is not None:
        y = y + bias.reshape(affine_shape).float()
    return y.to(input.dtype)


def _patch_group_norm_tpu() -> None:
    # TODO: Remove once torch_tpu's native_group_norm lowering keeps spatial dims
    # separate instead of flattening each group into a serial XLA while-loop.
    orig_group_norm = torch.nn.functional.group_norm
    if getattr(orig_group_norm, _GROUP_NORM_PATCHED, False):
        return

    def _group_norm(
        input: torch.Tensor,
        num_groups: int,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and input.dim() >= 2:
            return _tpu_group_norm(input, num_groups, weight, bias, eps)
        return orig_group_norm(input, num_groups, weight, bias, eps)

    setattr(_group_norm, _GROUP_NORM_PATCHED, True)
    torch.nn.functional.group_norm = _group_norm
    logger.info("Applied TPU patch: fused-reduce torch.nn.functional.group_norm.")


def _patch_flow_match_scheduler_step_flush() -> None:
    # TODO: Remove once vLLM-Omni compiles the denoise loop on TPU and avoids
    # the step-0 (timesteps == timestep).nonzero() D2H sync.
    # pyrefly: ignore[missing-import]
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    orig_step = FlowMatchEulerDiscreteScheduler.step
    if getattr(orig_step, _SCHEDULER_STEP_PATCHED, False):
        return

    def _resolve_first_step_index(self: Any, timestep: Any) -> None:
        # Only fast-path full-schedule runs starting at step 0; mid-schedule
        # callers (e.g. img2img with sliced timesteps) have len(timesteps) !=
        # num_inference_steps or _begin_index set.
        if self._step_index is not None or self._begin_index is not None:
            return
        if not (isinstance(timestep, torch.Tensor) and timestep.device.type == "tpu"):
            return
        timesteps = self.timesteps
        if (
            isinstance(timesteps, torch.Tensor)
            and timesteps.numel() > 0
            and len(timesteps) == self.num_inference_steps
        ):
            self._step_index = 0

    def _step_and_flush(self: Any, *args: Any, **kwargs: Any) -> Any:
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        _resolve_first_step_index(self, timestep)
        out = orig_step(self, *args, **kwargs)
        sample = out[0] if isinstance(out, tuple) else out.prev_sample
        if isinstance(sample, torch.Tensor) and sample.device.type == "tpu":
            synchronize_tensors(sample, wait=False)
        return out

    setattr(_step_and_flush, _SCHEDULER_STEP_PATCHED, True)
    FlowMatchEulerDiscreteScheduler.step = _step_and_flush
    logger.info(
        "Applied TPU patch: async per-step flush + step-0 index on "
        "FlowMatchEulerDiscreteScheduler.step."
    )


def _patch_diffusion_worker_runtime_threads() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni passes local_world_size=num_gpus to
    # set_multiprocessing_worker_envs and/or calls set_torch_threads_for_runtime()
    # in DiffusionWorker after load_model.
    #
    # DiffusionMultiprocExecutor calls set_multiprocessing_worker_envs() with
    # local_world_size=1, so every spawned worker starts with OMP_NUM_THREADS =
    # all host CPUs. With N SP workers that causes N-fold OpenMP oversubscription
    # on per-image CPU ops, stalling the slowest rank at the first collective.
    # -------------------------------------------------------------------------
    from vllm.utils.torch_utils import set_torch_threads_for_runtime
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    orig_load_model = DiffusionWorker.load_model
    if getattr(orig_load_model, _WORKER_THREADS_PATCHED, False):
        return

    def _load_model_then_trim_threads(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_load_model(self, *args, **kwargs)
        set_torch_threads_for_runtime()
        return result

    setattr(_load_model_then_trim_threads, _WORKER_THREADS_PATCHED, True)
    DiffusionWorker.load_model = _load_model_then_trim_threads
    logger.info(
        "Applied TPU patch: steady-state torch CPU threads "
        "(set_torch_threads_for_runtime) after DiffusionWorker.load_model."
    )


# ZImageTransformer2DModel.__init__ numbers noise_refiner blocks starting at
# 1000 + layer_id; flush before block 0 of both noise_refiner and main/context.
_NOISE_REFINER_FIRST_LAYER_ID = 1000


@contextlib.contextmanager
def _tpu_compile_config() -> Iterator[None]:
    """Scope AOTAutograd/Dynamo overrides to TPU diffusion compilations."""
    # AOTAutograd's disk cache does not key on TPU world_size / mesh,
    # so an sp=1 entry would be replayed on sp=8 with replica_count=1.
    with (
        torch._functorch.config.patch(enable_autograd_cache=False),
        torch._dynamo.config.patch(recompile_limit=64, cache_size_limit=64),
    ):
        yield


def _has_sp_plan_hooks(module: Any) -> bool:
    registry = module.__dict__.get("_hook_registry")
    if registry is None:
        return False
    return bool(registry._hooks)


class _TpuCompiledZImageLayers:
    """Runs all ``ZImageTransformer2DModel.layers`` blocks as one TPU graph."""

    def __init__(self, layers: Any, orig_block_forward: Any) -> None:
        self._layers = list(layers)
        self._orig_block_forward = orig_block_forward
        with _tpu_compile_config():
            self._compiled = torch.compile(self._forward, backend="tpu", dynamic=False)

    def _forward(
        self,
        unified: torch.Tensor,
        unified_attn_mask: torch.Tensor,
        unified_cos: torch.Tensor,
        unified_sin: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self._layers:
            unified = self._orig_block_forward(
                layer,
                unified,
                unified_attn_mask,
                unified_cos,
                unified_sin,
                adaln_input,
            )
        return unified

    def __call__(
        self,
        unified: torch.Tensor,
        unified_attn_mask: torch.Tensor,
        unified_cos: torch.Tensor,
        unified_sin: torch.Tensor,
        adaln_input: torch.Tensor,
        qk_norm_rope_table: Any = None,
    ) -> torch.Tensor:
        if qk_norm_rope_table is not None:
            raise ValueError(
                "TPU compiled ZImage layers do not support pre-packed CUDA "
                "qk_norm_rope_table."
            )
        with _tpu_compile_config():
            out = self._compiled(
                unified, unified_attn_mask, unified_cos, unified_sin, adaln_input
            )
        synchronize_tensors(out, wait=False)
        return out


def _patch_z_image_transformer_block_flush() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni compiles ZImageTransformerBlock with
    # backend="tpu" (dynamic=False) and restores ForwardContext.sp_plan_hooks_applied
    # across per-step contexts.
    #
    # 1. Compiles ZImageTransformerBlock.forward with torch.compile(backend="tpu",
    #    dynamic=False) and flushes asynchronously at block boundaries so each
    #    block compiles to one static graph without mid-layer kSplitTraversal.
    # 2. Sets auto_pad=True on ZImageTransformer2DModel._sp_plan["unified_prepare"]
    #    to eliminate per-layer _all_gather_int host syncs in advanced_uaa mode
    #    (safe once each block flushes its collectives).
    # 3. Restores ForwardContext.sp_plan_hooks_applied = True in
    #    ZImageTransformer2DModel.forward when unified_prepare has SP hooks.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.attention.parallel.ulysses import (
        UlyssesParallelAttention,
    )
    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )
    from vllm_omni.diffusion.models.z_image.z_image_transformer import (
        ZImageTransformer2DModel,
        ZImageTransformerBlock,
    )

    sp_plan = ZImageTransformer2DModel._sp_plan
    unified_prepare_plan = sp_plan.get("unified_prepare")
    if isinstance(unified_prepare_plan, dict):
        sp_plan["unified_prepare"] = {
            idx: (
                dataclasses.replace(sp_input, auto_pad=True)
                if dataclasses.is_dataclass(sp_input) and not isinstance(sp_input, type)
                else sp_input
            )
            for idx, sp_input in unified_prepare_plan.items()
        }

    orig_block_forward = ZImageTransformerBlock.forward
    orig_model_forward = ZImageTransformer2DModel.forward
    if not getattr(orig_model_forward, _ZIMAGE_MODEL_PATCHED, False):

        def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            sp_active = _has_sp_plan_hooks(self.unified_prepare)
            if sp_active and is_forward_context_available():
                get_forward_context().sp_plan_hooks_applied = True
            x_arg = args[0] if args else kwargs.get("x")
            x_first: Any = (
                x_arg[0] if isinstance(x_arg, (list, tuple)) and x_arg else None
            )
            on_tpu = isinstance(x_first, torch.Tensor) and x_first.device.type == "tpu"
            if (
                not sp_active
                and on_tpu
                and not getattr(orig_block_forward, _ZIMAGE_BLOCK_PATCHED, False)
            ):
                runner = self.__dict__.get("_tpu_compiled_layers_runner")
                if runner is None:
                    runner = _TpuCompiledZImageLayers(self.layers, orig_block_forward)
                    self._tpu_compiled_layers_runner = runner
                had_dict_layers = "layers" in self.__dict__
                orig_dict_layers = self.__dict__.get("layers")
                object.__setattr__(self, "layers", [runner])
                try:
                    return orig_model_forward(self, *args, **kwargs)
                finally:
                    if had_dict_layers:
                        object.__setattr__(self, "layers", orig_dict_layers)
                    else:
                        object.__delattr__(self, "layers")
            return orig_model_forward(self, *args, **kwargs)

        setattr(_model_forward, _ZIMAGE_MODEL_PATCHED, True)
        ZImageTransformer2DModel.forward = _model_forward

    if getattr(orig_block_forward, _ZIMAGE_BLOCK_PATCHED, False):
        return

    compile_state: dict[str, Any] = {"fn": None}

    def _compiled_forward() -> Any:
        if compile_state["fn"] is None:
            # Keep Ulysses all_to_all_single collectives outside torch.compile
            # to avoid non-deterministic collective lowering on TPU, while still
            # compiling the surrounding norms, QKV, RoPE, attention, and MLP.
            # torch._dynamo.disable is a no-op outside torch.compile (harmless
            # in eager mode across non-ZImage callers).
            for method_name in ("pre_attention", "post_attention"):
                fn = getattr(UlyssesParallelAttention, method_name)
                if not getattr(fn, "_tpu_dynamo_disabled", False):
                    wrapped = torch._dynamo.disable(fn)
                    wrapped._tpu_dynamo_disabled = True
                    setattr(UlyssesParallelAttention, method_name, wrapped)
            with _tpu_compile_config():
                compile_state["fn"] = torch.compile(
                    orig_block_forward, backend="tpu", dynamic=False
                )
        return compile_state["fn"]

    def _forward_and_flush(self: Any, *args: Any, **kwargs: Any) -> Any:
        x = args[0] if args else kwargs.get("x")
        on_tpu = (
            x is not None and isinstance(x, torch.Tensor) and x.device.type == "tpu"
        )
        if on_tpu and self.layer_id in (0, _NOISE_REFINER_FIRST_LAYER_ID):
            synchronize_tensors(x, wait=False)
        if on_tpu:
            with _tpu_compile_config():
                out = _compiled_forward()(self, *args, **kwargs)
        else:
            out = orig_block_forward(self, *args, **kwargs)
        if on_tpu and isinstance(out, torch.Tensor):
            synchronize_tensors(out, wait=False)
        return out

    setattr(_forward_and_flush, _ZIMAGE_BLOCK_PATCHED, True)
    ZImageTransformerBlock.forward = _forward_and_flush
    logger.info(
        "Applied TPU patch: async per-block graph flush and torch.compile "
        "around ZImageTransformerBlock.forward."
    )


class _TpuCompiledPromptEncoder:
    """``hidden_states[-2]`` of a HF decoder-only LM as one compiled TPU program."""

    def __init__(self, text_encoder: Any) -> None:
        decoder = text_encoder.get_decoder()
        self._embed_tokens = decoder.embed_tokens
        self._rotary_emb = decoder.rotary_emb
        self._layers = list(decoder.layers)[:-1]
        with _tpu_compile_config():
            self._compiled = torch.compile(self._forward, backend="tpu", dynamic=False)

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._embed_tokens(input_ids)
        position_embeddings = self._rotary_emb(hidden, position_ids)
        for layer in self._layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return hidden

    @staticmethod
    def build_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """``[B, 1, S, S]`` bool mask: causal & key padding, diagonal kept."""
        seq = attention_mask.shape[-1]
        eye = torch.eye(seq, dtype=torch.bool, device=attention_mask.device)
        causal = torch.ones_like(eye).tril()
        return (causal & attention_mask.bool()[:, None, None, :]) | eye

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq = input_ids.shape
        device = input_ids.device
        mask = self.build_mask(attention_mask).to(device)
        position_ids = torch.arange(seq).unsqueeze(0).expand(bsz, seq).to(device)
        with torch.no_grad(), _tpu_compile_config():
            return self._compiled(input_ids, mask, position_ids)


def _is_tpu_device(device: Any) -> bool:
    return isinstance(device, torch.device) and device.type == "tpu"


def _patch_z_image_prompt_encoder_tpu() -> None:
    # TODO: Remove once vLLM-Omni compiles diffusion text encoders with
    # backend="tpu" (dynamic=False) and drops the boolean-mask D2H gather.
    from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

    orig_encode_prompt = ZImagePipeline._encode_prompt
    if getattr(orig_encode_prompt, _ZIMAGE_PROMPT_ENCODER_PATCHED, False):
        return

    def _get_encoder(pipe: Any) -> _TpuCompiledPromptEncoder:
        encoder = pipe.__dict__.get("_tpu_prompt_encoder")
        if encoder is None:
            encoder = _TpuCompiledPromptEncoder(pipe.text_encoder)
            pipe._tpu_prompt_encoder = encoder
        return encoder

    def _encode_prompt(
        self: Any,
        prompt: Any,
        device: torch.device | None = None,
        prompt_embeds: Any = None,
        max_sequence_length: int = 512,
    ) -> Any:
        device = device or self._execution_device
        if not (prompt_embeds is None and _is_tpu_device(device)):
            return orig_encode_prompt(
                self,
                prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                max_sequence_length=max_sequence_length,
            )

        encoder = _get_encoder(self)
        # Mirrors vllm-omni==0.32.0rc1
        # (vllm_omni/diffusion/models/z_image/pipeline_z_image.py,
        # ZImagePipeline._encode_prompt) chat-template formatting and
        # tokenization, then runs the compiled decoder[:-1] graph and slices
        # valid tokens using the host attention_mask to avoid device nonzero()
        # syncs.
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_item}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt_item in prompts
        ]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = text_inputs.attention_mask
        hidden = encoder(text_inputs.input_ids.to(device), attention_mask)
        embeddings_list = []
        for i in range(hidden.shape[0]):
            valid = attention_mask[i].nonzero().flatten()
            start = int(valid[0]) if valid.numel() else 0
            stop = int(valid[-1]) + 1 if valid.numel() else 0
            if stop - start == valid.numel():
                embeddings_list.append(hidden[i, start:stop])
            else:
                embeddings_list.append(hidden[i][attention_mask[i].bool()])
        return embeddings_list

    setattr(_encode_prompt, _ZIMAGE_PROMPT_ENCODER_PATCHED, True)
    ZImagePipeline._encode_prompt = _encode_prompt
    logger.info("Applied TPU patch: compiled Qwen3 prompt encoder for ZImagePipeline.")
