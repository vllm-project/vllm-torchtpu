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

import sys
from contextlib import suppress
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
    os.environ.setdefault("VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")
    _patch_tpu_broadcast()
    _patch_tpu_empty_collectives()
    from vllm_torchtpu.omni.quantization import register_omni_tpu_quantization

    register_omni_tpu_quantization()
    _patch_block_sparse_attention_registry()


_TPU_BROADCAST_PATCHED = "_tpu_allreduce_bcast_patched"


def _patch_tpu_broadcast() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Under torch_tpu's default DEFER_AND_FUSE mode, ProcessGroupTpu::broadcast
    # (process_group_tpu.cc) fails across multi-rank diffusion workers for two
    # reasons:
    # 1. Rank-asymmetric deferred graph fusion: broadcast zeroes non-root buffers
    #    via tensor.copy_(at::zeros_like(tensor)) before emitting an in-place
    #    allreduce. When rank 0 queues rank-0-only preprocessing ops before
    #    dist.broadcast_object_list / dist.broadcast, or when non-root ranks
    #    fuse zero-copy + allreduce while root fuses only allreduce, ranks cut
    #    mismatched StableHLO collectives and hang in PjRt rendezvous.
    # 2. Tier-2 compilation cache key collision on root rank: the Tier-2 cache
    #    key hashes StableHLO IR without rootRank, so a sub-group or non-0 root
    #    broadcast can hit a cached executable compiled for a different root
    #    role and silently corrupt outputs to zeros.
    #
    # Fix: For eager TPU tensors on the "tpu_dist" backend, synchronize pending
    #   deferred ops on all ranks, explicitly zero_() non-root buffers and
    #   synchronize before issuing a symmetric c10d.all_reduce(SUM), and
    #   synchronize before returning (when not async_op). Bool and
    #   non-contiguous tensors run through a contiguous scratch buffer (int32
    #   for bool) and copy back.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once ProcessGroupTpu::broadcast flushes/isolates rank-asymmetric
    # deferred ops and includes the root-rank role in its compilation cache key.
    # -------------------------------------------------------------------------
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    orig_broadcast = c10d.broadcast
    if getattr(orig_broadcast, _TPU_BROADCAST_PATCHED, False):
        return

    def _tpu_safe_broadcast(
        tensor: torch.Tensor,
        src: int | None = None,
        group: Any = None,
        async_op: bool = False,
        group_src: int | None = None,
    ) -> Any:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "tpu"
            or torch.compiler.is_compiling()
        ):
            return orig_broadcast(
                tensor, src=src, group=group, async_op=async_op, group_src=group_src
            )

        if group_src is None:
            if src is None:
                raise ValueError("Either src or group_src must be specified")
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        if c10d._rank_not_in_group(group):
            return None

        try:
            backend = c10d.get_backend(group)
        except Exception:
            backend = None
        if backend != "tpu_dist":
            return orig_broadcast(
                tensor, src=src, group=group, async_op=async_op, group_src=group_src
            )

        if group_src is None:
            assert src is not None
            if group is None or group is c10d.GroupMember.WORLD:
                group_src = src
            else:
                group_src = c10d.get_group_rank(group, src)

        group_world_size = c10d.get_world_size(group)
        if not 0 <= group_src < group_world_size:
            raise ValueError(
                f"Invalid broadcast source rank {group_src} for group of size "
                f"{group_world_size}"
            )

        # Flush any rank-asymmetric deferred ops before touching collective inputs.
        torch.accelerator.synchronize()

        if group_world_size <= 1 or tensor.numel() == 0:
            return None

        my_group_rank = c10d.get_rank(group)
        needs_copy_back = tensor.dtype == torch.bool or not tensor.is_contiguous()
        if needs_copy_back:
            work_tensor = (
                tensor.to(dtype=torch.int32)
                if tensor.dtype == torch.bool
                else tensor.contiguous()
            )
            if my_group_rank != group_src:
                work_tensor.zero_()
            torch.accelerator.synchronize()
        else:
            work_tensor = tensor
            if my_group_rank != group_src:
                work_tensor.zero_()
                torch.accelerator.synchronize()

        work = c10d.all_reduce(
            work_tensor, op=c10d.ReduceOp.SUM, group=group, async_op=async_op
        )

        if needs_copy_back:
            if work is not None:
                work.wait()
            tensor.copy_(
                work_tensor.to(dtype=torch.bool)
                if tensor.dtype == torch.bool
                else work_tensor
            )

        if not async_op:
            torch.accelerator.synchronize()
        return work

    setattr(_tpu_safe_broadcast, _TPU_BROADCAST_PATCHED, True)
    c10d.broadcast = _tpu_safe_broadcast
    dist.broadcast = _tpu_safe_broadcast
    logger.info("Applied TPU patch: symmetric all_reduce broadcast for tpu_dist.")


_TPU_EMPTY_COLLECTIVES_PATCHED = "_tpu_empty_collectives_patched"


class _CompletedNoopWork:
    """Minimal completed Work handle returned for 0-element async collectives."""

    def wait(self, timeout: Any = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True


def _patch_tpu_empty_collectives() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Under hybrid SP x TP parallelism (e.g. Ulysses-2 x TP-4 on MiniMax-H3),
    # modality-specific projections (such as text/audio embeddings located in
    # one SP shard of a packed sequence) run TP collectives on 0-row tensors
    # (`shape = [0, hidden]`) across all ranks of the other SP shard's TP group.
    # ProcessGroupTpu lowers `all_gather_into_tensor` / `reduce_scatter_tensor`
    # on 0-sized tensors into StableHLO collectives whose `all_gather_dim` /
    # `scatter_dimension` has extent 0, failing MLIR verification with:
    #   `error: dimension size of operand at 'all_gather_dim' cannot be zero`
    #   `RuntimeError: failed to inline functions in MLIR module`.
    #
    # Fix: Short-circuit eager TPU `all_gather_single` (which backs
    #   `all_gather_into_tensor`), `reduce_scatter_tensor`, and `all_reduce`
    #   when both input and output tensors have `numel() == 0`.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once ProcessGroupTpu treats 0-element collectives as no-ops before
    # emitting StableHLO IR.
    # -------------------------------------------------------------------------
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    orig_all_gather_single = c10d.all_gather_single
    if getattr(orig_all_gather_single, _TPU_EMPTY_COLLECTIVES_PATCHED, False):
        return

    orig_reduce_scatter_tensor = c10d.reduce_scatter_tensor
    orig_all_reduce = c10d.all_reduce
    orig_barrier = c10d.barrier

    def _sync_eager_tpu() -> None:
        if not torch.compiler.is_compiling() and hasattr(torch, "accelerator"):
            with suppress(Exception):
                if torch.accelerator.is_available():
                    torch.accelerator.synchronize()

    def _is_empty_eager_tpu(*tensors: Any) -> bool:
        return (
            not torch.compiler.is_compiling()
            and bool(tensors)
            and all(
                isinstance(t, torch.Tensor)
                and t.device.type == "tpu"
                and t.numel() == 0
                for t in tensors
            )
        )

    def _tpu_safe_all_gather_single(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: Any = None,
        async_op: bool = False,
    ) -> Any:
        if _is_empty_eager_tpu(output_tensor, input_tensor):
            return _CompletedNoopWork() if async_op else None
        is_eager_tpu = (
            not torch.compiler.is_compiling()
            and isinstance(output_tensor, torch.Tensor)
            and output_tensor.device.type == "tpu"
        )
        if is_eager_tpu:
            _sync_eager_tpu()
        work = orig_all_gather_single(
            output_tensor, input_tensor, group=group, async_op=async_op
        )
        if is_eager_tpu and not async_op:
            _sync_eager_tpu()
        return work

    def _tpu_safe_reduce_scatter_tensor(
        output: torch.Tensor,
        input: torch.Tensor,
        op: Any = c10d.ReduceOp.SUM,
        group: Any = None,
        async_op: bool = False,
    ) -> Any:
        if _is_empty_eager_tpu(output, input):
            return _CompletedNoopWork() if async_op else None
        return orig_reduce_scatter_tensor(
            output, input, op=op, group=group, async_op=async_op
        )

    def _tpu_safe_all_reduce(
        tensor: torch.Tensor,
        op: Any = c10d.ReduceOp.SUM,
        group: Any = None,
        async_op: bool = False,
    ) -> Any:
        if _is_empty_eager_tpu(tensor):
            return _CompletedNoopWork() if async_op else None
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def _tpu_safe_barrier(*args: Any, **kwargs: Any) -> Any:
        _sync_eager_tpu()
        work = orig_barrier(*args, **kwargs)
        if not kwargs.get("async_op", False):
            _sync_eager_tpu()
        return work

    for fn in (
        _tpu_safe_all_gather_single,
        _tpu_safe_reduce_scatter_tensor,
        _tpu_safe_all_reduce,
        _tpu_safe_barrier,
    ):
        setattr(fn, _TPU_EMPTY_COLLECTIVES_PATCHED, True)

    c10d.all_gather_single = _tpu_safe_all_gather_single
    dist.all_gather_single = _tpu_safe_all_gather_single
    c10d.reduce_scatter_tensor = _tpu_safe_reduce_scatter_tensor
    dist.reduce_scatter_tensor = _tpu_safe_reduce_scatter_tensor
    c10d.all_reduce = _tpu_safe_all_reduce
    dist.all_reduce = _tpu_safe_all_reduce
    c10d.barrier = _tpu_safe_barrier
    dist.barrier = _tpu_safe_barrier
    logger.info("Applied TPU patch: short-circuit 0-element collectives for tpu_dist.")


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
    _patch_indexed_modulation()
    _patch_minimax_h3_fused_qk_norm_rope()
    _patch_diffusion_tpu_compile()
    _patch_minimax_h3_tpu_execution()
    _patch_ulysses_batched_qkv_a2a()
    _patch_tpu_fp16_sdpa()


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


_INDEXED_MODULATION_PATCHED = "_tpu_indexed_modulation_patched"


def _patch_indexed_modulation() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's indexed_modulation dispatches out-of-tree
    # platforms via CustomOp.register_oot instead of checking only `if x.is_cpu:`
    # before launching Triton kernels.
    #
    # Problem: vllm_omni/diffusion/layers/indexed_modulation.py (used by
    #   MiniMax-H3's DiT blocks and output norm) checks only `if x.is_cpu:`
    #   before launching Triton kernels (`triton.next_power_of_2(...)`). On TPU,
    #   `vllm.triton_utils.triton` is a placeholder (`HAS_TRITON = False`), so
    #   every DiT block fails with
    #   `AttributeError: module 'triton' has no attribute 'next_power_of_2'`.
    #   Moreover, naive PyTorch `index_select` + RMSNorm materializes `[S, D]`
    #   FP32 tables in HBM multiple times per block.
    # Fix: Route TPU BF16 aligned calls through fused Pallas AdaLN/gated-residual
    #   kernels (`vllm_torchtpu.omni.adaln`), falling back to FP32-intermediate
    #   PyTorch ATen ops for non-BF16 or unaligned shapes; non-TPU tensors
    #   delegate to upstream. Also rebind any direct symbol imports already
    #   cached in loaded `vllm_omni.*` modules.
    # -------------------------------------------------------------------------
    import sys

    from vllm_omni.diffusion.layers import indexed_modulation as im

    from vllm_torchtpu.omni.adaln import (
        register_adaln_pallas_ops,
        tpu_indexed_gate,
        tpu_indexed_gate_rms_norm_scale_shift,
        tpu_indexed_scale_shift_,
        tpu_rms_norm_indexed_scale_shift,
    )

    if getattr(im.rms_norm_indexed_scale_shift, _INDEXED_MODULATION_PATCHED, False):
        return

    register_adaln_pallas_ops()

    orig_scale_shift_ = im.indexed_scale_shift_
    orig_gate = im.indexed_gate
    orig_rms_scale_shift = im.rms_norm_indexed_scale_shift
    orig_gate_rms_scale_shift = im.indexed_gate_rms_norm_scale_shift

    def _tpu_indexed_scale_shift_(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if x.device.type != "tpu":
            return orig_scale_shift_(x, shift, scale, indices)
        return tpu_indexed_scale_shift_(x, shift, scale, indices)

    def _tpu_indexed_gate(
        x: torch.Tensor,
        gate: torch.Tensor,
        other: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if x.device.type != "tpu":
            return orig_gate(x, gate, other, indices)
        return tpu_indexed_gate(x, gate, other, indices)

    def _tpu_rms_norm_indexed_scale_shift(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if x.device.type != "tpu":
            return orig_rms_scale_shift(x, weight, shift, scale, indices, eps)
        return tpu_rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)

    def _tpu_indexed_gate_rms_norm_scale_shift(
        residual: torch.Tensor,
        gate: torch.Tensor,
        branch: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.device.type != "tpu":
            return orig_gate_rms_scale_shift(
                residual, gate, branch, weight, shift, scale, indices, eps
            )
        return tpu_indexed_gate_rms_norm_scale_shift(
            residual, gate, branch, weight, shift, scale, indices, eps
        )

    replacements = {
        "indexed_scale_shift_": (orig_scale_shift_, _tpu_indexed_scale_shift_),
        "indexed_gate": (orig_gate, _tpu_indexed_gate),
        "rms_norm_indexed_scale_shift": (
            orig_rms_scale_shift,
            _tpu_rms_norm_indexed_scale_shift,
        ),
        "indexed_gate_rms_norm_scale_shift": (
            orig_gate_rms_scale_shift,
            _tpu_indexed_gate_rms_norm_scale_shift,
        ),
    }
    for name, (_, patched_fn) in replacements.items():
        setattr(patched_fn, _INDEXED_MODULATION_PATCHED, True)
        setattr(im, name, patched_fn)

    for mod in list(sys.modules.values()):
        mod_name = getattr(mod, "__name__", "")
        if not mod_name.startswith("vllm_omni."):
            continue
        for name, (orig_fn, patched_fn) in replacements.items():
            if getattr(mod, name, None) is orig_fn:
                setattr(mod, name, patched_fn)

    logger.info("Applied TPU patch: Pallas-fused indexed_modulation for AdaLN.")


_MINIMAX_QKNORM_ROPE_PATCHED = "_tpu_minimax_qknorm_rope_patched"


def _patch_minimax_h3_fused_qk_norm_rope() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's fused_qk_norm_rope dispatches out-of-tree
    # platforms via CustomOp.register_oot.
    #
    # Problem: MiniMax-H3 Attention calls fused_qk_norm_rope with head_dim=128,
    #   rotary_dim=96, interleaved=False. On non-CUDA platforms it falls back to
    #   _eager_qk_norm_rope, which slices q/k into (rotary[:48], rotary[48:96],
    #   pass[96:128]) and concatenates them per head, forcing non-128-lane-aligned
    #   strided copies around every attention layer on TPU.
    # Fix: Route eligible TPU bf16 (T, H, 128) calls through the fused Pallas
    #   kernel in omni/rope.py (expressing the half-split rotation as an exact
    #   signed permutation matmul on the MXU).
    # Effect: Cuts QK RMSNorm + RoPE from ~0.68 ms to ~0.19 ms per block on TPU v7x.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fqkr

    from vllm_torchtpu.omni.rope import (
        can_fuse_minimax_qk_norm_rope,
        minimax_fused_qk_norm_rope,
        register_minimax_qk_norm_rope_op,
    )

    orig_impl = fqkr._fused_qk_norm_rope_impl
    if getattr(orig_impl, _MINIMAX_QKNORM_ROPE_PATCHED, False):
        return

    try:
        import torch_tpu  # noqa: F401

        register_minimax_qk_norm_rope_op()
    except ImportError:
        pass

    def _tpu_fused_qk_norm_rope_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rope_table: torch.Tensor,
        eps: float,
        head_dim: int,
        rotary_dim: int,
        interleaved: bool,
        cuda_rms_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if can_fuse_minimax_qk_norm_rope(
            q, k, q_weight, k_weight, rope_table, head_dim, rotary_dim, interleaved
        ):
            return minimax_fused_qk_norm_rope(
                q, k, q_weight, k_weight, rope_table, eps, rotary_dim
            )
        return orig_impl(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
            cuda_rms_norm,
        )

    setattr(_tpu_fused_qk_norm_rope_impl, _MINIMAX_QKNORM_ROPE_PATCHED, True)
    fqkr._fused_qk_norm_rope_impl = _tpu_fused_qk_norm_rope_impl
    logger.info(
        "Applied TPU patch: Pallas fused_qk_norm_rope for MiniMax-H3 half-split RoPE."
    )


_DIFFUSION_TPU_COMPILE_PATCHED = "_tpu_diffusion_compile_patched"
_DIFFUSION_TPU_EXEC_SYNC_PATCHED = "_tpu_diffusion_exec_sync_patched"


def _patch_diffusion_tpu_compile() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once DiffusionModelRunner._compile_transformer queries the
    # active OmniPlatform for its torch.compile backend ("tpu") and static-shape
    # policy (dynamic=False), and DiffusionWorker synchronizes non-primary ranks
    # at the end of execute_model.
    #
    # Problem:
    # 1. DiffusionModelRunner.load_model skips torch.compile when
    #    supports_torch_inductor() is False, and _compile_transformer defaults to
    #    Inductor with dynamic=True (which torch_tpu's Dynamo backend rejects).
    # 2. On multi-rank SP/TP stages, DiffusionWorker only copies outputs to CPU
    #    on rank 0 (`should_reply = True`), leaving tail ops pending in
    #    DEFER_AND_FUSE on ranks 1..N-1 across request boundaries.
    # Fix: Compile DiT blocks regionally with backend="tpu", dynamic=False when
    #   enforce_eager is False (initializing static LoRA adapters before compile
    #   when initialize_lora is provided), and synchronize all TPU ranks at the
    #   end of execute_model / execute_model_batch.
    # -------------------------------------------------------------------------
    try:
        from vllm_omni.diffusion.compile import regionally_compile
        from vllm_omni.diffusion.worker.diffusion_model_runner import (
            DiffusionModelRunner,
        )
    except ImportError:
        return

    from vllm_torchtpu.omni.minimax_h3 import disable_tpu_adaln_cache

    orig_compile_transformer = DiffusionModelRunner._compile_transformer
    orig_load_model = DiffusionModelRunner.load_model
    if getattr(
        orig_compile_transformer, _DIFFUSION_TPU_COMPILE_PATCHED, False
    ) and getattr(orig_load_model, _DIFFUSION_TPU_COMPILE_PATCHED, False):
        return

    def _tpu_compile_transformer(self: Any, attr_name: str) -> None:
        if getattr(self.device, "type", None) != "tpu":
            return orig_compile_transformer(self, attr_name)
        model = getattr(self.pipeline, attr_name, None)
        if not isinstance(model, torch.nn.Module):
            return
        disable_tpu_adaln_cache(model)
        dynamic = False
        compile_kwargs: dict[str, Any] = {"backend": "tpu", "dynamic": dynamic}
        compile_granularity = getattr(
            self.od_config, "diffusion_compile_granularity", "regional"
        )
        try:
            if compile_granularity == "full":
                model.compile(**compile_kwargs)
                compiled_model = model
            else:
                compiled_model = regionally_compile(model, **compile_kwargs)
            setattr(self.pipeline, attr_name, compiled_model)
        except Exception as exc:
            logger.warning(
                "Model runner: TPU %s torch.compile setup for %s failed: %s.",
                compile_granularity,
                attr_name,
                exc,
            )
            return
        logger.info(
            "Model runner: %s configured for lazy %s "
            "torch.compile(backend='tpu', dynamic=%s).",
            attr_name,
            compile_granularity,
            dynamic,
        )

    def _tpu_load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        initialize_lora = kwargs.get("initialize_lora")
        result = orig_load_model(self, *args, **kwargs)
        if getattr(self.device, "type", None) == "tpu" and self.pipeline is not None:
            if (
                callable(initialize_lora)
                and getattr(self.od_config, "lora_path", None) is not None
                and not getattr(self.pipeline, "requires_early_lora_init", False)
            ):
                initialize_lora()
            transformer_attrs = getattr(self.pipeline, "_dit_modules", None)
            if not transformer_attrs:
                transformer_attrs = ("transformer", "transformer_2")
            for attr_name in transformer_attrs:
                dit = getattr(self.pipeline, attr_name, None)
                if dit is not None:
                    disable_tpu_adaln_cache(dit)
            if not self.od_config.enforce_eager:
                if hasattr(self.pipeline, "setup_compile"):
                    try:
                        self.pipeline.setup_compile()
                    except Exception as exc:
                        logger.warning(
                            "Model runner: TPU setup_compile() failed (%s); "
                            "running without compile.",
                            exc,
                        )
                else:
                    for attr_name in transformer_attrs:
                        self._compile_transformer(attr_name)
        return result

    setattr(_tpu_compile_transformer, _DIFFUSION_TPU_COMPILE_PATCHED, True)
    DiffusionModelRunner._compile_transformer = _tpu_compile_transformer
    DiffusionModelRunner.load_model = _tpu_load_model

    for method_name in ("execute_model", "execute_model_batch"):
        orig_exec = getattr(DiffusionModelRunner, method_name, None)
        if orig_exec is None or getattr(
            orig_exec, _DIFFUSION_TPU_EXEC_SYNC_PATCHED, False
        ):
            continue

        def _make_synced_exec(fn: Any) -> Any:
            def _synced_exec(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    if getattr(self.device, "type", None) == "tpu":
                        torch.accelerator.synchronize()

            setattr(_synced_exec, _DIFFUSION_TPU_EXEC_SYNC_PATCHED, True)
            return _synced_exec

        setattr(DiffusionModelRunner, method_name, _make_synced_exec(orig_exec))

    logger.info("Applied TPU patch: regional torch.compile(backend='tpu') for DiT.")


def _patch_minimax_h3_tpu_execution() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's MiniMax-H3 pipeline avoids per-step TPU host
    # synchronizations (.item(), torch.unique, dynamic boolean masks) and
    # @torch.compiler.disable around single-request packed attention.
    #
    # Problem:
    # 1. ExactProjectionCache hashes timesteps on CPU and votes across TP ranks
    #    via all_reduce().item() on every DiT block.
    # 2. MiniMaxH3Attention._run_packed_attention is decorated with
    #    @torch.compiler.disable, splitting every MiniMaxH3DiTBlock into two
    #    compiled subgraphs plus an eager DEFER_AND_FUSE island.
    # 3. MiniMaxH3DenoiseBranch.forward_kwargs runs torch.unique(timesteps) on
    #    device each step (where step 0 has M=3 unique timesteps and steps >= 1
    #    have M=4, triggering a full 50-block recompile on step 1).
    # 4. minimax_h3_rf_v_to_x0 and euler_eta0_step execute 16 .item() finiteness
    #    checks per denoising step, and condition_proj + token_refiner recompute
    #    timestep-invariant prompt embeddings every step.
    # Fix: Install the TPU execution helpers in omni/minimax_h3.py.
    # Effect: MiniMax-H3 768x1344x124f (50 steps) on TPU v7x (Ulysses-8):
    #   10.49 s/step -> 0.962 s/step (BF16 dense).
    # -------------------------------------------------------------------------
    from vllm_torchtpu.omni.minimax_h3 import install_minimax_h3_tpu_execution_patches

    install_minimax_h3_tpu_execution_patches()


_ULYSSES_BATCHED_A2A_PATCHED = "_tpu_ulysses_batched_a2a_patched"


def _patch_ulysses_batched_qkv_a2a() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's UlyssesParallelAttention.pre_attention
    # batches matching Q, K, V tensors into a single SeqAllToAll4D collective
    # without mutating self._sp_ctx during TorchDynamo tracing.
    #
    # Problem: UlyssesParallelAttention.pre_attention issues three separate
    #   SeqAllToAll4D.apply calls for Q, K, V in every transformer block (150
    #   forward all-to-alls per step on 50-block DiTs).
    # Fix: When Q, K, V share shape and dtype on TPU under standard strict
    #   Ulysses SP (scatter_idx=2, gather_idx=1), concatenate (Q, K, V) along
    #   batch dim 0, issue one SeqAllToAll4D.apply, and chunk(3, dim=0),
    #   returning a fresh _UlyssesCtx in the 5-tuple without mutating self.
    # Effect: Reduces forward Ulysses all-to-all launches by 3x (150 -> 50/step).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.attention.parallel import ulysses as uly_mod
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    cls = uly_mod.UlyssesParallelAttention
    orig_pre = cls.pre_attention
    if getattr(orig_pre, _ULYSSES_BATCHED_A2A_PATCHED, False):
        return

    def _tpu_pre_attention(
        self: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> Any:
        ulysses_mode = getattr(self, "_ulysses_mode", None)
        if ulysses_mode is None and hasattr(uly_mod, "get_ulysses_mode"):
            ulysses_mode = uly_mod.get_ulysses_mode(default="strict")
        has_joint_or_gate = False
        has_2d_mask = False
        if attn_metadata is not None:
            if (
                attn_metadata.joint_query is not None
                or attn_metadata.joint_key is not None
                or attn_metadata.joint_value is not None
                or (
                    attn_metadata.extra is not None
                    and attn_metadata.extra.get("gate_compress") is not None
                )
            ):
                has_joint_or_gate = True
            if (
                attn_metadata.attn_mask is not None
                and attn_metadata.attn_mask.ndim == 2
            ) or (
                attn_metadata.joint_attn_mask is not None
                and attn_metadata.joint_attn_mask.ndim == 2
            ):
                has_2d_mask = True
        if (
            query.device.type != "tpu"
            or ulysses_mode == "advanced_uaa"
            or self._ulysses_a2a_permute
            or self._scatter_idx != 2
            or self._gather_idx != 1
            or has_joint_or_gate
            or has_2d_mask
            or query.shape != key.shape
            or query.shape != value.shape
            or query.dtype != key.dtype
            or query.dtype != value.dtype
        ):
            return orig_pre(
                self, query, key, value, attn_metadata, *extra_args, **extra_kwargs
            )

        sp_group = getattr(self, "_sp_group", None)
        ulysses_world_size = getattr(sp_group, "ulysses_world_size", None)
        if ulysses_world_size is None:
            ulysses_world_size = uly_mod.dist.get_world_size(self._ulysses_pg)
        if int(query.shape[2]) % ulysses_world_size != 0:
            return orig_pre(
                self, query, key, value, attn_metadata, *extra_args, **extra_kwargs
            )

        qkv_batched = SeqAllToAll4D.apply(
            self._ulysses_pg,
            torch.cat((query, key, value), dim=0),
            self._scatter_idx,
            self._gather_idx,
            self._use_sync,
        )
        q_ex, k_ex, v_ex = qkv_batched.chunk(3, dim=0)
        ctx = uly_mod._UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
        )
        return q_ex, k_ex, v_ex, attn_metadata, ctx

    setattr(_tpu_pre_attention, _ULYSSES_BATCHED_A2A_PATCHED, True)
    cls.pre_attention = _tpu_pre_attention
    logger.info("Applied TPU patch: batched QKV forward all_to_all for Ulysses SP.")


_BLOCK_SPARSE_REGISTRY_PATCHED = "_tpu_block_sparse_registry_patched"


def _patch_block_sparse_attention_registry() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni's DiffusionAttentionBackendEnum and
    # BLOCK_SPARSE_BACKENDS natively include TpuBlockSparseAttentionBackend and
    # ForwardContext step hooks notify registered attention backends directly.
    #
    # Problem:
    # 1. vLLM-Omni's AttentionSpec only forwards block_sparse kwargs when the
    #    backend name is in diff_data.BLOCK_SPARSE_BACKENDS.
    # 2. MiniMaxH3Pipeline calls set_forward_context_denoise_step_idx outside
    #    an active ForwardContext, so reading get_forward_context() inside a
    #    compiled block either misses the step index or triggers a TorchDynamo
    #    recompilation on every integer step 0..N-1.
    # Fix: Register TpuBlockSparseAttentionBackend for RAINFUSION_ATTN and
    #   FASTVIDEO_VSA, add TPU_BLOCK_SPARSE / BLOCK_SPARSE / FASTVIDEO_VSA to
    #   BLOCK_SPARSE_BACKENDS, and update each TpuBlockSparseAttentionImpl's
    #   boolean _sparse_step_active gate on host whenever the step index changes.
    # -------------------------------------------------------------------------
    import sys

    try:
        from vllm_omni.diffusion import data as diff_data
        from vllm_omni.diffusion import forward_context as fc
        from vllm_omni.diffusion.attention.backends.registry import (
            DiffusionAttentionBackendEnum,
            register_diffusion_backend,
        )
    except ImportError:
        return

    backend_path = "vllm_torchtpu.omni.attention.TpuBlockSparseAttentionBackend"
    for member_name in ("RAINFUSION_ATTN", "FASTVIDEO_VSA"):
        member = DiffusionAttentionBackendEnum.__members__.get(member_name)
        if member is not None:
            register_diffusion_backend(member, backend_path)

    diff_data.BLOCK_SPARSE_BACKENDS = frozenset(
        set(diff_data.BLOCK_SPARSE_BACKENDS)
        | {"FASTVIDEO_VSA", "TPU_BLOCK_SPARSE", "BLOCK_SPARSE"}
    )

    orig_set_step = fc.set_forward_context_denoise_step_idx
    if getattr(orig_set_step, _BLOCK_SPARSE_REGISTRY_PATCHED, False):
        return
    orig_set_total = fc.set_forward_context_denoise_total_steps

    _step_state: dict[str, int | None] = {"step": None, "total": None}

    def _sync_gates() -> None:
        from vllm_torchtpu.omni.attention import refresh_block_sparse_step_gates

        ctx = fc.get_forward_context() if fc.is_forward_context_available() else None
        step_idx = (
            getattr(ctx, "denoise_step_idx", None)
            if ctx is not None
            else _step_state["step"]
        )
        total = (
            getattr(ctx, "total_denoise_steps", None)
            if ctx is not None
            else _step_state["total"]
        )
        refresh_block_sparse_step_gates(step_idx, total)

    def _tpu_set_step(step_idx: int | None) -> None:
        _step_state["step"] = step_idx
        if step_idx is None:
            _step_state["total"] = None
        orig_set_step(step_idx)
        _sync_gates()

    def _tpu_set_total(total_steps: int | None) -> None:
        _step_state["total"] = total_steps
        orig_set_total(total_steps)
        _sync_gates()

    setattr(_tpu_set_step, _BLOCK_SPARSE_REGISTRY_PATCHED, True)
    setattr(_tpu_set_total, _BLOCK_SPARSE_REGISTRY_PATCHED, True)
    fc.set_forward_context_denoise_step_idx = _tpu_set_step
    fc.set_forward_context_denoise_total_steps = _tpu_set_total

    for mod in list(sys.modules.values()):
        mod_name = getattr(mod, "__name__", "")
        if not mod_name.startswith("vllm_omni."):
            continue
        if getattr(mod, "set_forward_context_denoise_step_idx", None) is orig_set_step:
            mod.set_forward_context_denoise_step_idx = _tpu_set_step
        if (
            getattr(mod, "set_forward_context_denoise_total_steps", None)
            is orig_set_total
        ):
            mod.set_forward_context_denoise_total_steps = _tpu_set_total


_TPU_FP16_SDPA_PATCHED = "_tpu_fp16_sdpa_patched"


def _patch_tpu_fp16_sdpa() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once torch_tpu's FLASH_ATTENTION SDPBackend supports float16
    # inputs or upcasts them internally.
    #
    # Problem: TorchTPU's FLASH_ATTENTION SDPBackend accepts only float32 or
    #   bfloat16 tensors and falls back with a warning on float16 (`Half`),
    #   which remote-code VAEs such as MiniMax-H3's ViT video decoder emit under
    #   `torch.autocast(..., dtype=torch.float16)`. Moreover, PyTorch's C++
    #   autocast dispatcher re-downcasts bfloat16 arguments back to float16 if
    #   autocast remains enabled around F.scaled_dot_product_attention.
    # Fix: For TPU float16 inputs, cast Q, K, V (and floating-point attn_mask)
    #   to bfloat16, run orig_sdpa under `torch.autocast("tpu", enabled=False)`,
    #   and cast the output back to float16.
    # Effect: Eliminates all <TorchTpu/Performance/> SDPA fallbacks during
    #   MiniMax-H3 Video VAE decode and cuts non-DiT decode wall time by ~1.1 s.
    # -------------------------------------------------------------------------
    from contextlib import nullcontext

    import torch.nn.functional as F

    orig_sdpa = F.scaled_dot_product_attention
    if getattr(orig_sdpa, _TPU_FP16_SDPA_PATCHED, False):
        return

    def _tpu_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if query.device.type != "tpu":
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        was_fp16 = (
            query.dtype == torch.float16
            and key.dtype == torch.float16
            and value.dtype == torch.float16
        )
        if was_fp16:
            attn_mask = (
                attn_mask.to(torch.bfloat16)
                if (attn_mask is not None and attn_mask.is_floating_point())
                else attn_mask
            )
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        ac_ctx = (
            torch.autocast(device_type="tpu", enabled=False)
            if (was_fp16 and torch._C._is_any_autocast_enabled())
            else nullcontext()
        )
        with ac_ctx:
            if attn_mask is None and dropout_p == 0.0 and query.ndim == 4:
                s_q = int(query.shape[-2])
                s_kv = int(key.shape[-2])
                d = int(query.shape[-1])
                if is_causal and s_q == s_kv and d % 128 == 0 and s_q % 128 != 0:
                    pad_len = ((s_q + 127) // 128) * 128 - s_q
                    q_pad = F.pad(query, (0, 0, 0, pad_len))
                    k_pad = F.pad(key, (0, 0, 0, pad_len))
                    v_pad = F.pad(value, (0, 0, 0, pad_len))
                    out = orig_sdpa(
                        q_pad,
                        k_pad,
                        v_pad,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=scale,
                        enable_gqa=enable_gqa,
                    )[..., :s_q, :]
                    return out.to(torch.float16) if was_fp16 else out
                if (
                    not is_causal
                    and s_q > 2048
                    and s_q * s_kv > 4_000_000
                    and (s_q % 128 != 0 or s_kv % 128 != 0 or d % 128 != 0)
                ):
                    q_chunk = 2048
                    out = torch.cat(
                        [
                            orig_sdpa(
                                query[..., start : start + q_chunk, :],
                                key,
                                value,
                                attn_mask=None,
                                dropout_p=0.0,
                                is_causal=False,
                                scale=scale,
                                enable_gqa=enable_gqa,
                            )
                            for start in range(0, s_q, q_chunk)
                        ],
                        dim=-2,
                    )
                    return out.to(torch.float16) if was_fp16 else out

            out = orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return out.to(torch.float16) if was_fp16 else out

    setattr(_tpu_sdpa, _TPU_FP16_SDPA_PATCHED, True)
    F.scaled_dot_product_attention = _tpu_sdpa
    for mod in list(sys.modules.values()):
        if getattr(mod, "scaled_dot_product_attention", None) is orig_sdpa:
            with suppress(Exception):
                mod.scaled_dot_product_attention = _tpu_sdpa
    logger.info(
        "Applied TPU patch: FP16->BF16 upcast for scaled_dot_product_attention."
    )
