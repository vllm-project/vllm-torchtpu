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

vLLM-Omni diffusion workers do not run TPUWorker, so patch_registry.py
applies these patches at the diffusion stages that OmniTpuPlatform enters:
omni_platform, diffusion_worker and diffusion_model_load. Shared patches to
worker infrastructure and common layer modules live here and run for every
model. Model-specific patches live in submodules under omni/patches/, take
od_config, and return False unless od_config.model_class_name names a pipeline
that builds the patched classes, so a model's module is only imported when
that model is served.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_model_patch_applied = False


def is_model_class_in(
    od_config: "OmniDiffusionConfig | None", pipelines: frozenset[str]
) -> bool:
    """Return whether ``od_config.model_class_name`` is in ``pipelines``."""
    return od_config is not None and od_config.model_class_name in pipelines


def mark_model_patch_applied() -> None:
    """Record that at least one model-specific diffusion patch was applied."""
    global _model_patch_applied
    _model_patch_applied = True


def log_unpatched_pipeline(od_config: "OmniDiffusionConfig | None") -> None:
    """Logs the served pipeline when no model-specific TPU patch applies to it."""
    if _model_patch_applied:
        return
    logger.info(
        "No model-specific TPU diffusion patch applies to pipeline %s.",
        None if od_config is None else od_config.model_class_name,
    )


def _extend_async_output_timeout() -> None:
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


_IPC_CPU_PACK_PATCHED = "_tpu_ipc_cpu_pack_patched"


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
    if getattr(orig_pack_tensor, _IPC_CPU_PACK_PATCHED, False):
        return

    def _safe_pack_tensor_if_large(val: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return orig_pack_tensor(val.cpu(), *args, **kwargs)

    setattr(_safe_pack_tensor_if_large, _IPC_CPU_PACK_PATCHED, True)
    ipc._pack_tensor_if_large = _safe_pack_tensor_if_large
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")


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
    # layout with an exact signed pair-swap matmul (see omni/rope.py). It
    # replaces the class forward because CustomOp binds _forward_method at
    # init.
    #
    # Complex sliced views crash on TPU under inference mode, which rules
    # out the complex-number RoPE form.
    #
    # Scope: every model. RotaryEmbeddingWan lives in a shared layer module,
    # and the replacement falls back to _forward_method for calls that are not
    # interleaved, not half-dim or not bf16.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan

    from vllm_torchtpu.omni.rope import rotary_embedding_wan_forward

    RotaryEmbeddingWan.forward = rotary_embedding_wan_forward
    logger.info("Applied TPU patch: pair-swap matmul RotaryEmbeddingWan.")


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


_WARMUP_SYNC_PATCHED = "_tpu_warmup_sync_patched"


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
    if getattr(orig_run_startup_warmup, _WARMUP_SYNC_PATCHED, False):
        return

    def _synced_run_startup_warmup(self: Any) -> None:
        orig_run_startup_warmup(self)
        # TODO: Once the vllm-omni pin is >= 0.30, sync every worker with
        # self.executor.collective_rpc("synchronize_device") and install this
        # patch at platform init, so that mp engines are covered too.
        torch.accelerator.synchronize()

    setattr(_synced_run_startup_warmup, _WARMUP_SYNC_PATCHED, True)
    DiffusionEngine.run_startup_warmup = _synced_run_startup_warmup
    logger.info("Applied TPU patch: device sync after diffusion startup warmup.")


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
