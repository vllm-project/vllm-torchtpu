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
"""TPU runtime patches for vLLM-Omni models.

vLLM-Omni diffusion workers run in separate worker processes spawned by
vLLM-Omni's DiffusionEngine rather than through TPUWorker's worker_init /
model_load lifecycle stages, so OmniTpuPlatform applies the diffusion patches
itself: apply_omni_tpu_patches() when the platform is constructed, and
apply_omni_model_specific_patches() at attention backend selection
(get_diffusion_attn_backend_cls).

Patches for autoregressive stages served by TPUARWorker live in one submodule
per model of this package and are applied from TPUARModelRunner.load_model()
until Omni patches are unified with patch_registry.py (tracked in #1862); see
qwen3omni_thinker.py.
"""

import weakref
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
    _patch_ltx2_per_batch_timestep()
    _patch_ltx2_guidance_sigma_scalar()
    _patch_ltx2_async_guidance_step()
    _patch_ltx2_fast_attention_dispatch()
    _patch_ulysses_inference_all_to_all_4d()


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
# Upper bounds on the split-RoPE rotation caches. A single LTX-2 request uses
# a small handful of distinct table slices (video, cross-attention, audio) and
# tensor objects. A rope module that replaces its tables releases their owned
# copies itself; clearing on overflow bounds HBM for tables that are never
# replaced (a dropped pipeline).
_MAX_OWNED_ROPE_LAYOUTS = 8
_MAX_ROPE_VIEW_ENTRIES = 16


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
    #   - SP hands each rank a dim-2 slice of the tables, and TP the lanes of
    #     its heads. torch_tpu passes a program the buffer behind each
    #     argument, so a rotation that reads a slice takes the full table as
    #     an input and slices it again, in every attention layer. The
    #     rotation reads fresh_buffer copies with a buffer of the slice's
    #     shape instead, built once per table and slice layout and released
    #     when the rope module replaces the tables; a slice whose storage has
    #     no data pointer is rotated as before.
    #   - The tables are not edited after the rope module builds them
    #     (vLLM-Omni only slices them): the rotation caches below are keyed
    #     on storage and layout, not on a version counter.
    #   - The torch.compiler.is_compiling() branches keep these caches and
    #     their storage queries out of a traced graph. No caller compiles the
    #     LTX-2 blocks today, so they are only reached by the unit tests.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    from vllm_torchtpu.omni.rope import (
        apply_split_rotary_half_swap,
        split_rope_natural_tables,
    )
    from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

    rope_cls = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed
    if getattr(rope_cls.forward, _SPLIT_ROPE_PATCHED, False):
        return
    orig_rope_forward = rope_cls.forward
    orig_apply_split_rotary_emb = ltx2_transformer.apply_split_rotary_emb

    # (cos, sin) the rotation is handed -> the tables it reads.
    #   - tables_by_view finds them by identity. The same tensor objects reach
    #     every attention layer of a forward, so the hot path makes no storage
    #     or layout query. An entry holds the pair by weak reference (a dead
    #     reference never matches a tensor that reuses its id), the q/k width
    #     and TP layout its lanes were cut for, and the owned tables, or None
    #     for a pair that is rotated as is (never one whose lanes were cut).
    #   - owned_tables holds one pair of owned tables per slice of a pair of
    #     tables: the key is the storage of cos and sin (data pointers) and
    #     the slice layout on it (storage offset, shape and stride of cos and
    #     sin, dtype, device). Each forward slices fresh views of the same
    #     tables, which hit here. Tables of the same shape (under TP the audio
    #     and audio cross-attention tables) have separate entries. An entry
    #     holds its slices, which keeps the storage alive, so a data pointer
    #     is not reused while it is a key.
    tables_by_view: dict[
        int,
        tuple[
            weakref.ref[torch.Tensor],
            weakref.ref[torch.Tensor],
            int,
            tuple[int, int] | None,
            tuple[torch.Tensor, torch.Tensor] | None,
        ],
    ] = {}
    owned_tables: dict[
        tuple[Any, ...],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}

    def _release_owned_tables(tables: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Drops the slices and owned copies cut from ``tables``."""
        storage = (
            tables[0].untyped_storage().data_ptr(),
            tables[1].untyped_storage().data_ptr(),
        )
        stale = [key for key in owned_tables if key[:2] == storage]
        if not stale:
            return
        for key in stale:
            del owned_tables[key]
        tables_by_view.clear()  # its entries hold the owned copies too

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
        if cached is not None:
            # Release the replaced tables before the new ones are built: a
            # call with another token count (two-stage refinement, another
            # resolution) slices the new tables in another layout, which
            # would otherwise leave the old slices pinning the old tables.
            self._tpu_split_rope_cache = None
            _release_owned_tables(cached[2])
            del cached
        tables = orig_rope_forward(self, coords, device, out_dtype)
        if tables[0].ndim != 4:
            return tables
        tables = split_rope_natural_tables(*tables)
        synchronize_tensors(list(tables), wait=False)
        self._tpu_split_rope_cache = (coords, key, tables)
        return tables

    def _is_view(t: torch.Tensor) -> bool:
        return (
            not t.is_contiguous()
            or t.storage_offset() != 0
            or t.untyped_storage().nbytes() != t.numel() * t.element_size()
        )

    def _tp_layout() -> tuple[int, int]:
        return (
            ltx2_transformer.get_tensor_model_parallel_world_size(),
            ltx2_transformer.get_tensor_model_parallel_rank(),
        )

    def _local_lanes(
        cos: torch.Tensor, sin: torch.Tensor, width: int, tp_layout: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_size, tp_rank = tp_layout
        if cos.shape[-1] != width * tp_size:
            raise ValueError(
                f"LTX-2 split RoPE tables of width {cos.shape[-1]} do not "
                f"match q/k of width {width} on {tp_size} tensor-parallel "
                "ranks; the tables must cover all heads of all ranks."
            )
        start = tp_rank * width
        return cos[..., start : start + width], sin[..., start : start + width]

    def _rotation_tables(
        cos: torch.Tensor, sin: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tables for the rotation of ``width`` channels with ``(cos, sin)``."""
        entry = tables_by_view.get(id(cos))
        if (
            entry is not None
            and entry[0]() is cos
            and entry[1]() is sin
            and entry[2] == width
            and (entry[3] is None or entry[3] == _tp_layout())
        ):
            return entry[4] or (cos, sin)
        tp_layout = None if cos.shape[-1] == width else _tp_layout()
        local = (
            (cos, sin)
            if tp_layout is None
            else _local_lanes(cos, sin, width, tp_layout)
        )
        owned = None
        if _is_view(local[0]) or _is_view(local[1]):
            cos_ptr = local[0].untyped_storage().data_ptr()
            sin_ptr = local[1].untyped_storage().data_ptr()
            if cos_ptr == 0 or sin_ptr == 0:
                return local  # Unknown storage: rotate with the views.
            key = (
                cos_ptr,
                sin_ptr,
                local[0].storage_offset(),
                tuple(local[0].shape),
                local[0].stride(),
                local[1].storage_offset(),
                tuple(local[1].shape),
                local[1].stride(),
                local[0].dtype,
                local[0].device,
            )
            cached = owned_tables.get(key)
            if cached is None:
                owned = (fresh_buffer(local[0]), fresh_buffer(local[1]))
                synchronize_tensors(list(owned), wait=False)
                if len(owned_tables) >= _MAX_OWNED_ROPE_LAYOUTS:
                    owned_tables.clear()
                    tables_by_view.clear()
                # The entry keeps the views: their storage, and with it the
                # data pointers in the key, stays valid while it is cached.
                owned_tables[key] = (*local, *owned)
            else:
                owned = (cached[2], cached[3])
        if owned is None and tp_layout is not None:
            return local  # Lanes without an owned copy are cut again next time.
        if len(tables_by_view) >= _MAX_ROPE_VIEW_ENTRIES:
            tables_by_view.clear()
        tables_by_view[id(cos)] = (
            weakref.ref(cos),
            weakref.ref(sin),
            width,
            tp_layout,
            owned,
        )
        return owned or local

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
        if not torch.compiler.is_compiling():
            cos, sin = _rotation_tables(cos, sin, width)
        elif cos.shape[-1] != width:
            cos, sin = _local_lanes(cos, sin, width, _tp_layout())
        return apply_split_rotary_half_swap(x, cos, sin, head_dim)

    setattr(_cached_rope_forward, _SPLIT_ROPE_PATCHED, True)
    rope_cls.forward = _cached_rope_forward
    ltx2_transformer.apply_split_rotary_emb = _apply_split_rotary_emb
    logger.info("Applied TPU patch: cached LTX-2 split RoPE on the q/k layout.")


_LTX2_PER_BATCH_TIMESTEP_PATCHED = "_tpu_ltx2_per_batch_timestep_patched"


def _patch_ltx2_per_batch_timestep() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTXRuntime._denoise_timestep_kwargs
    # stops expanding unconditioned timesteps for sequence parallelism.
    #
    # Problem: In vllm-omni 0.32.0rc1
    #   (vllm_omni/diffusion/models/ltx2/ltx2_runtime.py::
    #   LTXRuntime._denoise_timestep_kwargs), the method discards forward_ctx
    #   and denoise_ctx and calls guidance_executor.timestep_kwargs with
    #   expand_for_sequence_parallel=True. That expands each sample's scalar
    #   timestep to (B, T_video) and (B, T_audio), causing time_embed and every
    #   transformer block to project and materialize per-token AdaLN modulation
    #   tables even though all tokens share the same timestep.
    # Fix: Replace LTXRuntime._denoise_timestep_kwargs (copied from vllm-omni
    #   0.32.0rc1, since wrapping cannot override the hardcoded keyword
    #   argument) to pass expand_for_sequence_parallel=False. Unconditioned
    #   timesteps stay 1-D (B,), keeping temb at (B, 1, C) so it broadcasts
    #   into each block's norm modulation.
    # Scope:
    #   - LTX2VideoTransformer3DModel._build_sp_plan declares
    #     SequenceParallelInput(split_dim=1, expected_dims=2, split_output=False)
    #     on timestep, so 1-D (B,) tensors skip sequence sharding and stay
    #     replicated across SP ranks. They take the dimension-mismatch path of
    #     SequenceParallelSplitHook._prepare_sp_input
    #     (vllm_omni/diffusion/hooks/sequence_parallel.py), which logs
    #     "Expected tensor with 2 dims, got 1. Skipping split." through
    #     warning_once: under SP every rank prints it once for T2V, and it is
    #     expected.
    #   - Conditioned I2V requests with a conditioning_mask still override
    #     timestep with the 2-D (B, T_video) masked tensor in
    #     LTXI2VConditioningMixin._denoise_timestep_kwargs (which is sharded
    #     by the SP plan), while keeping audio_timestep 1-D (B,).
    # -------------------------------------------------------------------------
    import inspect

    from vllm_omni.diffusion.models.ltx2.ltx2_guidance import LTXGuidanceExecutor
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

    if getattr(
        LTXRuntime._denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        False,
    ):
        return

    # Fail loudly at patch-install time if upstream changes the replaced
    # method's signature or drops expand_for_sequence_parallel.
    expected_params = (
        "self",
        "ts",
        "forward_ctx",
        "denoise_ctx",
        "video_token_count",
        "audio_token_count",
    )
    actual_params = tuple(
        inspect.signature(LTXRuntime._denoise_timestep_kwargs).parameters
    )
    if actual_params != expected_params:
        raise RuntimeError(
            "LTXRuntime._denoise_timestep_kwargs signature drifted from "
            f"vllm-omni 0.32.0rc1: expected {expected_params}, got {actual_params}."
        )
    guidance_params = inspect.signature(LTXGuidanceExecutor.timestep_kwargs).parameters
    if "expand_for_sequence_parallel" not in guidance_params:
        raise RuntimeError(
            "LTXGuidanceExecutor.timestep_kwargs no longer accepts "
            "'expand_for_sequence_parallel'; review _patch_ltx2_per_batch_timestep."
        )

    def _per_batch_denoise_timestep_kwargs(
        self: Any,
        ts: torch.Tensor,
        forward_ctx: Any,
        denoise_ctx: Any,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return self.guidance_executor.timestep_kwargs(
            ts,
            video_token_count,
            audio_token_count,
            expand_for_sequence_parallel=False,
        )

    setattr(
        _per_batch_denoise_timestep_kwargs,
        _LTX2_PER_BATCH_TIMESTEP_PATCHED,
        True,
    )
    LTXRuntime._denoise_timestep_kwargs = _per_batch_denoise_timestep_kwargs
    logger.info("Applied TPU patch: per-batch LTX-2 unconditioned timestep.")


_LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED = "_tpu_ltx2_guidance_sigma_scalar_patched"
# Live entries are one video and one audio schedule per stage; the bound only
# drops entries whose sigma tensors are gone.
_MAX_SIGMA_CACHE_ENTRIES = 8
_LTX2_ASYNC_STEP_PATCHED = "_tpu_ltx2_async_step_patched"


def _patch_ltx2_guidance_sigma_scalar() -> None:
    """Pre-materialize LTX-2 scheduler sigmas on CPU to avoid per-step D2H syncs.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_guidance.py``
    (``LTXGuidanceExecutor.predict_noise``, ``predict_parallel_guidance``, and
    ``_guide_modality``) and ``ltx2_denoise.py`` (``prepare_scheduler_stage``).

    Upstream ``LTXGuidanceExecutor.predict_noise`` (and
    ``predict_parallel_guidance``, which ``predict_noise`` dispatches to when
    ``forward_ctx.guidance_parallel_ready`` is True) calls
    ``self._guide_modality(..., sigma=scheduler.sigmas[index])`` twice per
    denoise step (video and audio) without passing ``sigma_scalar``, so
    ``velocity_from_x0`` executes ``sigma.to(torch.float32).item()`` on a TPU
    tensor twice every step. In TorchTPU lazy execution mode, each ``.item()``
    forces a blocking ``TransferFromDevice`` that drains the in-flight
    execution queue and starves TensorCore between steps. Caching
    ``scheduler.sigmas`` as CPU ``float32`` scalars once per stage and
    supplying ``sigma_scalar`` eliminates all per-step D2H host syncs while
    preserving the exact IEEE-754 ``float32`` scalar value.

    The cache lives in this closure, keyed by the sigma tensor's identity (a
    weakref) and version counter, never by storage address and never on the
    scheduler: ``prepare_scheduler_stage`` deep-copies ``pipeline.scheduler``
    into ``audio_scheduler``, and a freed schedule's address can be reused by
    the next one. Each scheduler stage also refreshes it unconditionally.

    Limitations: ``combine_cfg_velocity`` (``ltx2_guidance.py``) still calls
    ``.item()`` per call. It is reached only through the generic
    ``combine_cfg_noise`` CFG-parallel hook, not through ``predict_noise``.
    Removing these reads also removes the per-step graph cut they made; the
    step flush in ``_patch_ltx2_async_guidance_step`` replaces it.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni pre-materializes scheduler sigmas on CPU and passes
    ``sigma_scalar`` from ``LTXGuidanceExecutor.predict_noise`` /
    ``predict_parallel_guidance`` into ``_guide_modality``.
    """
    from vllm_omni.diffusion.models.ltx2 import ltx2_denoise, ltx2_guidance

    executor_cls = ltx2_guidance.LTXGuidanceExecutor
    if getattr(
        executor_cls.predict_noise,
        _LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED,
        False,
    ):
        return

    orig_prepare_scheduler_stage = ltx2_denoise.prepare_scheduler_stage
    orig_predict_noise = executor_cls.predict_noise
    orig_guide_modality = executor_cls._guide_modality
    active_step_sigma_map: dict[int, float | None] = {}
    # id(sigmas) -> (weakref to sigmas, version counter or None, scalars).
    sigma_cache: dict[int, tuple[Any, int | None, tuple[float, ...]]] = {}

    def _ensure_sigma_scalars(
        scheduler: Any, *, refresh: bool = False
    ) -> tuple[float, ...] | None:
        sigmas_t = getattr(scheduler, "sigmas", None)
        if (
            sigmas_t is None
            or not isinstance(sigmas_t, torch.Tensor)
            or sigmas_t.ndim != 1
        ):
            return None
        key = id(sigmas_t)
        # Inference tensors have no version counter; the per-stage refresh
        # covers schedules rebuilt in inference mode.
        version = None if sigmas_t.is_inference() else sigmas_t._version
        entry = sigma_cache.get(key)
        if (
            not refresh
            and entry is not None
            and entry[0]() is sigmas_t
            and entry[1] == version
        ):
            return entry[2]
        scalars = tuple(
            float(x)
            for x in sigmas_t.detach().to(device="cpu", dtype=torch.float32).tolist()
        )
        if len(sigma_cache) >= _MAX_SIGMA_CACHE_ENTRIES:
            for dead in [k for k, e in sigma_cache.items() if e[0]() is None]:
                del sigma_cache[dead]
            if len(sigma_cache) >= _MAX_SIGMA_CACHE_ENTRIES:
                sigma_cache.clear()
        sigma_cache[key] = (weakref.ref(sigmas_t), version, scalars)
        return scalars

    def _cached_prepare_scheduler_stage(
        pipeline: Any,
        request_inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, torch.Tensor]:
        audio_scheduler, video_audio_step_adapter, timesteps_tensor = (
            orig_prepare_scheduler_stage(pipeline, request_inputs, *args, **kwargs)
        )
        _ensure_sigma_scalars(getattr(pipeline, "scheduler", None), refresh=True)
        _ensure_sigma_scalars(audio_scheduler, refresh=True)
        return audio_scheduler, video_audio_step_adapter, timesteps_tensor

    def _predict_noise_with_cached_sigma(
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
        v_scalars = _ensure_sigma_scalars(getattr(pipeline, "scheduler", None))
        a_scalars = _ensure_sigma_scalars(getattr(forward_ctx, "audio_scheduler", None))
        active_step_sigma_map[id(state.video)] = (
            v_scalars[index]
            if v_scalars is not None and 0 <= index < len(v_scalars)
            else None
        )
        active_step_sigma_map[id(state.audio)] = (
            a_scalars[index]
            if a_scalars is not None and 0 <= index < len(a_scalars)
            else None
        )
        try:
            return orig_predict_noise(
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
            active_step_sigma_map.clear()

    @staticmethod
    def _guide_modality_with_cached_sigma(
        sample: torch.Tensor,
        splits: dict[str, torch.Tensor],
        sigma: torch.Tensor,
        guidance: Any,
        *,
        model_sigma: torch.Tensor | None = None,
        sigma_scalar: float | None = None,
        preserve_positive_velocity: bool = False,
        rescale_token_count: int | None = None,
    ) -> torch.Tensor:
        if sigma_scalar is None:
            sigma_scalar = active_step_sigma_map.get(id(sample))
        return orig_guide_modality(
            sample,
            splits,
            sigma,
            guidance,
            model_sigma=model_sigma,
            sigma_scalar=sigma_scalar,
            preserve_positive_velocity=preserve_positive_velocity,
            rescale_token_count=rescale_token_count,
        )

    setattr(
        _predict_noise_with_cached_sigma,
        _LTX2_GUIDANCE_SIGMA_SCALAR_PATCHED,
        True,
    )
    ltx2_denoise.prepare_scheduler_stage = _cached_prepare_scheduler_stage
    executor_cls.predict_noise = _predict_noise_with_cached_sigma
    executor_cls._guide_modality = _guide_modality_with_cached_sigma
    logger.info(
        "Applied TPU patch: pre-materialized LTX-2 guidance sigma scalars on CPU."
    )


def _patch_ltx2_async_guidance_step() -> None:
    """Replace CUDA-only per-step device synchronize() with non-blocking flush.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_runtime.py``
    (``LTXRuntime._synchronize_guidance_parallel_step_output``).

    Upstream materializes contiguous latents and calls
    ``current_omni_platform.synchronize()`` at the end of every CFG-parallel
    denoise step to avoid CUDA async numerical drift. On TPU, that blocking
    sync stalls the host between steps and cannot be wrapped without invoking
    the sync itself, so this patch replaces the 5-line method body to keep the
    same CFG-world-size guard and ``.contiguous()`` materialization while
    flushing via ``synchronize_tensors(list(latents), wait=False)``.

    Every other step (``cfg_parallel_size == 1``, or a single-pass phase such
    as the two-stage refine) is flushed too, without the copy. Upstream skips
    the sync there, but its ``velocity_from_x0`` ``.item()`` cut the lazy graph
    every step; ``_patch_ltx2_guidance_sigma_scalar`` removes that read, so
    without this flush the whole loop and the decoder would trace into one
    TPU program. This runs in the host denoise loop, never under
    ``torch.compile``.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni gates the blocking ``synchronize()`` in
    ``LTXRuntime._synchronize_guidance_parallel_step_output`` to CUDA (or
    uses a non-blocking flush on TPU).
    """
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

    from vllm_torchtpu.utils import synchronize_tensors

    if getattr(
        LTXRuntime._synchronize_guidance_parallel_step_output,
        _LTX2_ASYNC_STEP_PATCHED,
        False,
    ):
        return

    def _async_guidance_step_output(
        self: Any,
        latents: tuple[torch.Tensor, torch.Tensor],
        guidance_parallel_ready: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self
        if (
            guidance_parallel_ready
            and parallel_state.get_classifier_free_guidance_world_size() > 1
        ):
            latents = tuple(tensor.contiguous() for tensor in latents)
        synchronize_tensors(list(latents), wait=False)
        return latents

    setattr(_async_guidance_step_output, _LTX2_ASYNC_STEP_PATCHED, True)
    LTXRuntime._synchronize_guidance_parallel_step_output = _async_guidance_step_output
    logger.info(
        "Applied TPU patch: non-blocking async guidance-parallel step boundary."
    )


_LTX2_FAST_ATTENTION_DISPATCH_PATCHED = "_tpu_ltx2_fast_attention_dispatch_patched"
_ULYSSES_INFERENCE_A2A_PATCHED = "_tpu_ulysses_inference_all_to_all_4d_patched"


def _patch_ltx2_fast_attention_dispatch() -> None:
    """Skip per-call ``inspect.signature`` reflection in ``LTX2Attention``.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/models/ltx2/ltx2_transformer.py``
    (``LTX2Attention.forward``).

    Upstream ``LTX2Attention.forward`` calls
    ``inspect.signature(self.processor.__call__)`` and builds filtered kwargs on
    all 288 attention calls per step (11,520 calls per 40-step run, about
    19 ms/step of host time per worker under 8-worker load). The plain 6-arg
    call on the default ``LTX2AudioVideoAttnProcessor`` (which accepts
    ``perturbation_mask``) goes straight to the processor. Any other processor,
    or extra kwargs, takes upstream's filtering path with the signature cached
    per processor class.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni caches the processor signature in
    ``LTX2Attention.forward``.
    """
    import inspect

    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    attention_cls = ltx2_transformer.LTX2Attention
    default_processor_cls = ltx2_transformer.LTX2AudioVideoAttnProcessor
    if getattr(attention_cls.forward, _LTX2_FAST_ATTENTION_DISPATCH_PATCHED, False):
        return

    sig_cache: dict[type, frozenset[str]] = {}

    def _fast_ltx2_attention_forward(
        self: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        perturbation_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        processor = self.processor
        if not kwargs and type(processor) is default_processor_cls:
            return processor(
                self,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                query_rotary_emb,
                key_rotary_emb,
                perturbation_mask=perturbation_mask,
            )
        if perturbation_mask is not None:
            kwargs["perturbation_mask"] = perturbation_mask
        proc_cls = type(processor)
        attn_parameters = sig_cache.get(proc_cls)
        if attn_parameters is None:
            attn_parameters = frozenset(
                inspect.signature(processor.__call__).parameters.keys()
            )
            sig_cache[proc_cls] = attn_parameters
        unused_kwargs = [k for k in kwargs if k not in attn_parameters]
        if unused_kwargs:
            logger.warning(
                "attention_kwargs %s are not expected by %s and will be ignored.",
                unused_kwargs,
                proc_cls.__name__,
            )
        return processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            query_rotary_emb,
            key_rotary_emb,
            **{k: v for k, v in kwargs.items() if k in attn_parameters},
        )

    setattr(_fast_ltx2_attention_forward, _LTX2_FAST_ATTENTION_DISPATCH_PATCHED, True)
    attention_cls.forward = _fast_ltx2_attention_forward
    logger.info("Applied TPU patch: fast LTX-2 attention processor dispatch.")


def _patch_ulysses_inference_all_to_all_4d() -> None:
    """Run inference Ulysses 4D all-to-all without autograd boxing.

    Upstream provenance (vllm-omni 0.32.0rc1):
    ``vllm_omni/diffusion/distributed/comm.py`` (``all_to_all_4D`` and
    ``SeqAllToAll4D``).

    Scope: process-wide. ``comm.all_to_all_4D`` and ``SeqAllToAll4D`` are shared
    by every Ulysses model (LTX-2, Wan, Kandinsky6, ...); every module that
    imports ``SeqAllToAll4D`` binds the same class, so setting ``apply`` once
    covers them all. ``SeqAllToAll4D`` defines no backward, so skipping
    ``torch.autograd.Function.apply`` (called 288 times per LTX-2 step under
    ``torch.no_grad()``) changes nothing but host time. The ``(2, 1)`` and
    ``(1, 2)`` layouts use one ``permute`` instead of chained transposes and
    are bit-exact with upstream. ``use_sync=True``, ``_output_destination``,
    a world size of 1 and other index pairs go to upstream ``all_to_all_4D``.
    Modules that imported ``all_to_all_4D`` by name keep upstream's function.

    REMOVAL / FIX CONDITION:
    Remove once vLLM-Omni's inference ``SeqAllToAll4D`` / ``all_to_all_4D``
    skip autograd boxing and the double transpose.
    """
    from vllm_omni.diffusion.distributed import comm

    if getattr(comm.all_to_all_4D, _ULYSSES_INFERENCE_A2A_PATCHED, False):
        return

    orig_all_to_all_4d = comm.all_to_all_4D

    def _fast_all_to_all_4d(
        input: torch.Tensor,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        group: Any = None,
        use_sync: bool = False,
        *,
        _output_destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input.dim() == 4, (
            f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"
        )
        seq_world_size = torch.distributed.get_world_size(group)
        if (
            (scatter_idx, gather_idx) not in ((2, 1), (1, 2))
            or seq_world_size <= 1
            or use_sync
            or _output_destination is not None
        ):
            return orig_all_to_all_4d(
                input,
                scatter_idx,
                gather_idx,
                group=group,
                use_sync=use_sync,
                _output_destination=_output_destination,
            )
        if scatter_idx == 2:
            bs, shard_seqlen, hc, hs = input.shape
            send_buf = (
                input.reshape(
                    bs, shard_seqlen, seq_world_size, hc // seq_world_size, hs
                )
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            recv_buf = torch.empty_like(send_buf)
            torch.distributed.all_to_all_single(recv_buf, send_buf, group=group)
            return (
                recv_buf.reshape(
                    shard_seqlen * seq_world_size, bs, hc // seq_world_size, hs
                )
                .transpose(0, 1)
                .contiguous()
            )
        bs, seqlen, shard_hc, hs = input.shape
        send_buf = (
            input.reshape(bs, seq_world_size, seqlen // seq_world_size, shard_hc, hs)
            .permute(1, 3, 2, 0, 4)
            .contiguous()
        )
        recv_buf = torch.empty_like(send_buf)
        torch.distributed.all_to_all_single(recv_buf, send_buf, group=group)
        return (
            recv_buf.reshape(
                shard_hc * seq_world_size, seqlen // seq_world_size, bs, hs
            )
            .permute(2, 1, 0, 3)
            .contiguous()
        )

    def _fast_seq_all_to_all_4d_apply(
        group: Any,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        return _fast_all_to_all_4d(
            input,
            scatter_idx,
            gather_idx,
            group=group,
            use_sync=use_sync,
        )

    setattr(_fast_all_to_all_4d, _ULYSSES_INFERENCE_A2A_PATCHED, True)
    comm.all_to_all_4D = _fast_all_to_all_4d
    comm.SeqAllToAll4D.apply = staticmethod(_fast_seq_all_to_all_4d_apply)
    logger.info("Applied TPU patch: inference Ulysses all_to_all_4D without autograd.")
