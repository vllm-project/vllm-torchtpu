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
    _patch_diffusion_worker_d2h_stream()
    _patch_diffusion_startup_warmup_sync()
    _patch_ltx2_split_rope()
    _patch_wan_fused_qk_norm_rope()
    _patch_wan_fresh_timestep()
    _patch_sdpa_fallback_dispatch()
    _patch_ltx2_vocoder_cpu_init()
    _patch_ring_attention()


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


_RING_ATTENTION_PATCHED = "_tpu_ring_attention_patched"


def _patch_ring_attention() -> None:
    """Route RingParallelAttention and hybrid Ring+Ulysses to TPU Pallas ring."""
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention

    from vllm_torchtpu.omni.attention import (
        can_fuse_tpu_ring_ulysses,
        can_run_tpu_ring_attention,
        tpu_ring_flash_attention,
    )

    if not getattr(RingParallelAttention.run_attention, _RING_ATTENTION_PATCHED, False):
        orig_run = RingParallelAttention.run_attention

        def _tpu_run_attention(
            self: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata: Any = None,
            softmax_scale: float | None = None,
            causal: bool = False,
        ) -> torch.Tensor:
            if not can_run_tpu_ring_attention(
                self, query, key, value, attn_metadata, causal=causal
            ):
                if int(self._sp_group.ring_world_size) > 1:
                    # The upstream fallback exchanges K/V with torch.distributed
                    # P2P sends, which hang on TPU, so fail loudly instead.
                    raise NotImplementedError(
                        "Ring attention call does not fit the TPU Pallas ring "
                        f"kernel (dtype={query.dtype}, causal={causal}, "
                        "attn_backend_pref="
                        f"{getattr(self, 'attn_backend_pref', None)}, "
                        f"metadata={type(attn_metadata).__name__}), and the "
                        "vLLM-Omni RingParallelAttention fallback (torch.distributed "
                        "P2P sends) is not supported on TPU."
                    )
                logger.warning_once(
                    "Ring attention call does not fit the TPU Pallas ring kernel "
                    "(dtype=%s, causal=%s, attn_backend_pref=%s, metadata=%s); "
                    "falling back to vLLM-Omni RingParallelAttention, which "
                    "exchanges K/V with torch.distributed P2P sends.",
                    str(query.dtype),
                    causal,
                    getattr(self, "attn_backend_pref", None),
                    type(attn_metadata).__name__,
                )
                return orig_run(
                    self,
                    query,
                    key,
                    value,
                    attn_metadata,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            sp_group = self._sp_group
            return tpu_ring_flash_attention(
                query,
                key,
                value,
                softmax_scale,
                ring_size=int(sp_group.ring_world_size),
                uly_size=int(getattr(sp_group, "ulysses_world_size", 1)),
                fuse_ulysses=False,
            )

        setattr(_tpu_run_attention, _RING_ATTENTION_PATCHED, True)
        RingParallelAttention.run_attention = _tpu_run_attention

    if not getattr(Attention._forward_impl, _RING_ATTENTION_PATCHED, False):
        orig_forward_impl = Attention._forward_impl

        def _tpu_attention_forward_impl(
            self: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata: Any = None,
        ) -> torch.Tensor:
            # Layers without a ring runner (every non-ring pipeline) return to
            # the upstream path before any strategy or metadata work.
            if not getattr(self, "use_ring", False) or (
                getattr(self, "ring_runner", None) is None
            ):
                return orig_forward_impl(self, query, key, value, attn_metadata)
            strategy = self._get_active_parallel_strategy()
            checked_meta = self._with_kv_cache_dtype(attn_metadata)
            if can_fuse_tpu_ring_ulysses(
                self, strategy, query, key, value, checked_meta
            ):
                sp_group = self.ring_runner._sp_group
                return tpu_ring_flash_attention(
                    query,
                    key,
                    value,
                    self.softmax_scale,
                    ring_size=int(sp_group.ring_world_size),
                    uly_size=int(sp_group.ulysses_world_size),
                    fuse_ulysses=True,
                )
            # Pure ring (ulysses_world_size <= 1) has no Ulysses to fuse: the
            # ring runner below is the intended TPU kernel path, so only warn
            # when a Ulysses all-to-all really runs outside the kernel.
            sp_group = getattr(self.ring_runner, "_sp_group", None)
            if (
                getattr(strategy, "name", None) == "ulysses"
                and int(getattr(sp_group, "ulysses_world_size", 1)) > 1
            ):
                logger.warning_once(
                    "Attention call cannot fuse Ulysses into the TPU Pallas ring "
                    "kernel (strategy=%s, dtype=%s); running vLLM-Omni's Ulysses "
                    "all-to-all around the ring attention runner instead.",
                    getattr(strategy, "name", None),
                    str(query.dtype),
                )
            return orig_forward_impl(self, query, key, value, attn_metadata)

        setattr(_tpu_attention_forward_impl, _RING_ATTENTION_PATCHED, True)
        Attention._forward_impl = _tpu_attention_forward_impl
    logger.info("Applied TPU patch: Pallas 2D Ring-Ulysses diffusion attention.")


_PARALLEL_DEVICE_ORDER_PATCHED = "_tpu_parallel_device_order_patched"
_CACHED_DEVICE_ID_ORDER: tuple[int, list[int], list[int]] | None = None
_DEVICE_IDS_BY_RANK_OVERRIDE: list[int] | None = None
_ACTIVE_INIT_MP_DEVICE_ORDER: bool | None = None


def _query_local_tpu_device_id() -> int:
    """PJRT device ID of this rank's TPU."""
    return int(torch.tpu.get_local_device_attributes()["id"])


def _resolve_device_ids_by_rank(world_size: int) -> tuple[list[int], list[int]] | None:
    global _CACHED_DEVICE_ID_ORDER
    from vllm_torchtpu.omni.attention import ranks_ordered_by_device_id

    if _DEVICE_IDS_BY_RANK_OVERRIDE is not None:
        if len(_DEVICE_IDS_BY_RANK_OVERRIDE) != world_size:
            return None
        dev_ids = [int(d) for d in _DEVICE_IDS_BY_RANK_OVERRIDE]
        return dev_ids, ranks_ordered_by_device_id(dev_ids)

    if _CACHED_DEVICE_ID_ORDER is not None and _CACHED_DEVICE_ID_ORDER[0] == world_size:
        return _CACHED_DEVICE_ID_ORDER[1], _CACHED_DEVICE_ID_ORDER[2]

    if world_size <= 1 or not torch.distributed.is_initialized():
        return None

    # Every rank must reach all_gather_object, or the ranks whose query worked
    # wait on it forever, so a local failure is gathered as None and raised
    # below on every rank.
    local_val: int | None = None
    local_exc: Exception | None = None
    try:
        local_val = _query_local_tpu_device_id()
    except Exception as exc:
        local_exc = exc

    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    gathered: list[int | None] = [None] * world_size
    torch.distributed.all_gather_object(
        gathered, local_val, group=get_world_group().cpu_group
    )
    if any(x is None for x in gathered):
        raise RuntimeError(
            "torch.tpu.get_local_device_attributes() failed on one or more "
            f"ranks (gathered device IDs by rank: {gathered}). Ring attention "
            "(ring_degree > 1) lays out the parallel groups by TPU device ID "
            "and cannot run with rank-ordered groups."
        ) from local_exc
    dev_ids = [int(x) for x in gathered if x is not None]
    try:
        ranks_by_dev = ranks_ordered_by_device_id(dev_ids)
    except ValueError as exc:
        raise RuntimeError(
            f"TPU device IDs gathered by rank ({dev_ids}) are not a "
            "permutation of 0..world_size-1. Ring attention (ring_degree > 1) "
            "lays out the parallel groups by TPU device ID and cannot run "
            "with rank-ordered groups."
        ) from exc
    _CACHED_DEVICE_ID_ORDER = (world_size, dev_ids, ranks_by_dev)
    logger.info(
        "Resolved TPU device-ID rank ordering: dev_ids=%s, ranks_by_dev=%s",
        dev_ids,
        ranks_by_dev,
    )
    return dev_ids, ranks_by_dev


def _validate_ascending_c10d_groups(layout: dict[str, Any]) -> None:
    """Ensure c10d all-to-all / group-rank groups stay sorted by global rank."""
    for group_kind, key in (
        ("ulysses", "ulysses_groups"),
        ("tp", "tp_groups"),
        ("pp", "pp_groups"),
    ):
        for group in layout[key]:
            ranks = [int(r) for r in group]
            if ranks != sorted(ranks):
                raise RuntimeError(
                    f"TPU device-ID parallel group ordering produced non-ascending "
                    f"{group_kind} group {ranks} (torch.distributed.new_group "
                    f"sorts ranks); dev_ids={layout['device_ids_by_rank']}, "
                    f"ranks_by_dev={layout['ranks_by_device_id']}."
                )


def _patch_parallel_state_device_order() -> None:
    """Lay out vLLM-Omni parallel groups by PJRT device ID on TPU."""
    from vllm_omni.diffusion.distributed import group_coordinator, parallel_state

    if not getattr(
        parallel_state.RankGenerator.get_ranks, _PARALLEL_DEVICE_ORDER_PATCHED, False
    ):
        orig_get_ranks = parallel_state.RankGenerator.get_ranks

        def _tpu_get_ranks(
            self: Any, token: str, *args: Any, **kwargs: Any
        ) -> list[list[int]]:
            groups = orig_get_ranks(self, token, *args, **kwargs)
            if _ACTIVE_INIT_MP_DEVICE_ORDER is False:
                return groups
            world_size = getattr(self, "world_size", None)
            if world_size is None:
                world_size = sum(len(g) for g in groups)
            resolved = _resolve_device_ids_by_rank(int(world_size))
            if resolved is None:
                return groups
            _, ranks_by_dev = resolved
            return [[ranks_by_dev[int(pos)] for pos in g] for g in groups]

        setattr(_tpu_get_ranks, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        parallel_state.RankGenerator.get_ranks = _tpu_get_ranks

    if not getattr(
        parallel_state._initialize_model_parallel,
        _PARALLEL_DEVICE_ORDER_PATCHED,
        False,
    ):
        orig_init_mp = parallel_state._initialize_model_parallel

        def _tpu_initialize_model_parallel(
            data_parallel_size: int | None = None,
            cfg_parallel_size: int = 1,
            sequence_parallel_size: int | None = None,
            ulysses_degree: int = 1,
            ring_degree: int = 1,
            allgather_degree: int = 1,
            tensor_parallel_size: int = 1,
            pipeline_parallel_size: int = 1,
            fully_shard_degree: int = 1,
            enable_expert_parallel: bool = False,
            use_hsdp: bool = False,
            backend: str | None = None,
        ) -> None:
            global _ACTIVE_INIT_MP_DEVICE_ORDER
            use_device_order = (ring_degree > 1) or (
                _DEVICE_IDS_BY_RANK_OVERRIDE is not None
            )
            if not use_device_order:
                prev_active = _ACTIVE_INIT_MP_DEVICE_ORDER
                _ACTIVE_INIT_MP_DEVICE_ORDER = False
                try:
                    orig_init_mp(
                        data_parallel_size=data_parallel_size,
                        cfg_parallel_size=cfg_parallel_size,
                        sequence_parallel_size=sequence_parallel_size,
                        ulysses_degree=ulysses_degree,
                        ring_degree=ring_degree,
                        allgather_degree=allgather_degree,
                        tensor_parallel_size=tensor_parallel_size,
                        pipeline_parallel_size=pipeline_parallel_size,
                        fully_shard_degree=fully_shard_degree,
                        enable_expert_parallel=enable_expert_parallel,
                        use_hsdp=use_hsdp,
                        backend=backend,
                    )
                finally:
                    _ACTIVE_INIT_MP_DEVICE_ORDER = prev_active
                return

            expected_sp = ring_degree * ulysses_degree * allgather_degree
            layout: dict[str, Any] | None = None
            if allgather_degree == 1 and (
                sequence_parallel_size is None or sequence_parallel_size == expected_sp
            ):
                world_size = (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else (
                        len(_DEVICE_IDS_BY_RANK_OVERRIDE)
                        if _DEVICE_IDS_BY_RANK_OVERRIDE is not None
                        else 1
                    )
                )
                resolved = _resolve_device_ids_by_rank(world_size)
                if resolved is not None:
                    from vllm_torchtpu.omni.attention import (
                        compute_device_id_group_layout,
                        validate_ring_cfg_lanes,
                    )

                    dev_ids, _ = resolved
                    dp_arg = (
                        1
                        if (use_hsdp and data_parallel_size is None)
                        else data_parallel_size
                    )
                    non_dp = (
                        tensor_parallel_size
                        * expected_sp
                        * pipeline_parallel_size
                        * cfg_parallel_size
                    )
                    if (
                        non_dp > 0
                        and world_size % non_dp == 0
                        and (dp_arg is None or dp_arg * non_dp == world_size)
                    ):
                        layout = compute_device_id_group_layout(
                            dev_ids,
                            cfg_size=cfg_parallel_size,
                            ring_size=ring_degree,
                            uly_size=ulysses_degree,
                            tp_size=tensor_parallel_size,
                            pp_size=pipeline_parallel_size,
                            dp_size=dp_arg,
                        )
                        if ring_degree > 1:
                            _validate_ascending_c10d_groups(layout)
                            outer_size = max(
                                1, world_size // (ring_degree * ulysses_degree)
                            )
                            validate_ring_cfg_lanes(
                                cfg_groups=layout["cfg_groups"],
                                ring_groups=layout["ring_groups"],
                                cfg_lane_by_rank=layout["cfg_lane_by_rank"],
                                device_ids_by_rank=dev_ids,
                                mesh_device_ids=list(range(world_size)),
                                outer_size=outer_size,
                                ring_size=ring_degree,
                                uly_size=ulysses_degree,
                            )

            prev_active = _ACTIVE_INIT_MP_DEVICE_ORDER
            _ACTIVE_INIT_MP_DEVICE_ORDER = True
            try:
                orig_init_mp(
                    data_parallel_size=data_parallel_size,
                    cfg_parallel_size=cfg_parallel_size,
                    sequence_parallel_size=sequence_parallel_size,
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    allgather_degree=allgather_degree,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    fully_shard_degree=fully_shard_degree,
                    enable_expert_parallel=enable_expert_parallel,
                    use_hsdp=use_hsdp,
                    backend=backend,
                )
            finally:
                _ACTIVE_INIT_MP_DEVICE_ORDER = prev_active

            if layout is not None:
                from vllm_torchtpu.omni.attention import set_ring_cfg_validation_state

                set_ring_cfg_validation_state(layout)

        setattr(_tpu_initialize_model_parallel, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        parallel_state._initialize_model_parallel = _tpu_initialize_model_parallel

    if not getattr(
        parallel_state.destroy_model_parallel, _PARALLEL_DEVICE_ORDER_PATCHED, False
    ):
        orig_destroy_mp = parallel_state.destroy_model_parallel

        def _tpu_destroy_model_parallel() -> None:
            global _CACHED_DEVICE_ID_ORDER
            from vllm_torchtpu.omni.attention import clear_ring_cfg_validation_state

            _CACHED_DEVICE_ID_ORDER = None
            clear_ring_cfg_validation_state()
            orig_destroy_mp()

        setattr(_tpu_destroy_model_parallel, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        parallel_state.destroy_model_parallel = _tpu_destroy_model_parallel

    # With ring_degree > 1 the cfg/dp groups can list ranks out of ascending
    # order (e.g. [6, 4]) while torch.distributed.new_group sorts them. In
    # vLLM-Omni's GroupCoordinator every src/dst argument is mapped through
    # self.ranks and all_reduce/barrier do not depend on member order, so only
    # all_gather and gather return data in sorted c10d order; both are reordered
    # below. test_group_coordinator_methods_are_audited_for_reorder fails when
    # upstream adds a method that has not been audited.
    if not getattr(
        group_coordinator.GroupCoordinator.all_gather,
        _PARALLEL_DEVICE_ORDER_PATCHED,
        False,
    ):
        orig_all_gather = group_coordinator.GroupCoordinator.all_gather

        def _tpu_all_gather(
            self: Any,
            input_: torch.Tensor,
            dim: int = 0,
            separate_tensors: bool = False,
            group: Any | None = None,
        ) -> torch.Tensor | list[torch.Tensor]:
            is_own_group = group is None or group is getattr(self, "device_group", None)
            ranks = list(getattr(self, "ranks", []))
            if is_own_group and len(ranks) > 1 and ranks != sorted(ranks):
                sorted_ranks = sorted(ranks)
                order = [sorted_ranks.index(r) for r in ranks]
                chunks = orig_all_gather(
                    self, input_, dim=dim, separate_tensors=True, group=group
                )
                reordered = [chunks[i] for i in order]
                if separate_tensors:
                    return reordered
                if dim < 0:
                    dim += input_.dim()
                return torch.cat(reordered, dim=dim)
            return orig_all_gather(
                self,
                input_,
                dim=dim,
                separate_tensors=separate_tensors,
                group=group,
            )

        setattr(_tpu_all_gather, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        group_coordinator.GroupCoordinator.all_gather = _tpu_all_gather

    if not getattr(
        group_coordinator.GroupCoordinator.gather,
        _PARALLEL_DEVICE_ORDER_PATCHED,
        False,
    ):
        orig_gather = group_coordinator.GroupCoordinator.gather

        def _tpu_gather(
            self: Any,
            input_: torch.Tensor,
            dst: int = 0,
            dim: int = -1,
        ) -> torch.Tensor | None:
            ranks = list(getattr(self, "ranks", []))
            if len(ranks) > 1 and ranks != sorted(ranks):
                out = orig_gather(self, input_, dst=dst, dim=dim)
                if out is None:
                    return None
                norm_dim = dim + input_.dim() if dim < 0 else dim
                chunk_size = input_.size(norm_dim)
                if chunk_size == 0:
                    return out
                sorted_ranks = sorted(ranks)
                order = [sorted_ranks.index(r) for r in ranks]
                chunks = out.split(chunk_size, dim=norm_dim)
                return torch.cat([chunks[i] for i in order], dim=norm_dim)
            return orig_gather(self, input_, dst=dst, dim=dim)

        setattr(_tpu_gather, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        group_coordinator.GroupCoordinator.gather = _tpu_gather

    if not getattr(
        group_coordinator.SequenceParallelGroupCoordinator.__init__,
        _PARALLEL_DEVICE_ORDER_PATCHED,
        False,
    ):
        orig_sp_init = group_coordinator.SequenceParallelGroupCoordinator.__init__

        def _tpu_sp_init(
            self: Any,
            group_ranks: list[list[int]],
            local_rank: int,
            torch_distributed_backend: Any,
            **kwargs: Any,
        ) -> None:
            orig_sp_init(
                self,
                group_ranks=group_ranks,
                local_rank=local_rank,
                torch_distributed_backend=torch_distributed_backend,
                **kwargs,
            )
            if _ACTIVE_INIT_MP_DEVICE_ORDER is False:
                return
            ranks = list(getattr(self, "ranks", []))
            uly_w = int(getattr(self, "ulysses_world_size", 1))
            ring_w = int(getattr(self, "ring_world_size", 1))
            ag_w = int(getattr(self, "allgather_world_size", 1))
            if ring_w <= 1 and _DEVICE_IDS_BY_RANK_OVERRIDE is None:
                return
            if (
                len(ranks) > 1
                and ag_w == 1
                and uly_w * ring_w == len(ranks)
                and self.rank in ranks
            ):
                pos = ranks.index(self.rank)
                u_idx = pos % uly_w
                r_idx = pos // uly_w
                self.ulysses_ranks = ranks[r_idx * uly_w : (r_idx + 1) * uly_w]
                self.ring_ranks = ranks[u_idx::uly_w]
                self.ulysses_rank = u_idx
                self.ring_rank = r_idx

        setattr(_tpu_sp_init, _PARALLEL_DEVICE_ORDER_PATCHED, True)
        group_coordinator.SequenceParallelGroupCoordinator.__init__ = _tpu_sp_init

    logger.info("Applied TPU patch: PJRT device-ID ordering for Omni parallel groups.")
