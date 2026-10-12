"""Host copies of a step's Engram hash inputs, for Host-DRAM Engram tables.

With ``TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1`` the Engram rows are gathered on
the host, so the n-gram hashes must reach the host every step. Hashing on the
device and copying back (``gathered_hashes.to("cpu")``) blocks on the previous
step and stops async scheduling from overlapping host work with the device.
The runner already holds every hash input on the host before uploading it, so
V4.1 publishes those host copies for the model instead.

Like ``async_fence.py``, this wraps the runner's lookback upload rather than
editing the shared runner: ``forward_model`` calls
``_prepare_lookback_token_ids`` inside the step's forward context, after
``input_ids_cpu``, ``positions_cpu`` and ``query_start_loc_cpu`` are final.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_engram_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    engram_gathered_num_tokens,
)

HOST_ENGRAM_KEY = "deepseek_v41_host_engram"


@dataclass(frozen=True)
class HostEngramInputs:
    """Host views matching the device tensors of one forward step.

    ``input_ids``/``positions`` cover the padded token count from the start
    of the runner's host buffers (full-batch layout), ``query_start_loc`` the
    whole padded request table and ``lookback_token_ids`` the whole lookback
    buffer, so callers slice them to the device tensors' static shapes.
    ``all_prompt`` is True when every scheduled token of every request is a
    prompt token. ``host_ready`` indicates whether decode/mixed token and
    lookback inputs have been resolved on the host for ``hash_on_host``.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    lookback_token_ids: torch.Tensor
    all_prompt: bool
    host_ready: bool | None = None

    @property
    def can_hash_on_host(self) -> bool:
        return self.all_prompt if self.host_ready is None else self.host_ready


def _async_source_tokens_cpu(pre_async: Any) -> torch.Tensor | None:
    """Resolve the previous step's sampled tokens on CPU for async substitution."""
    wait_for_copy = getattr(pre_async, "wait_for_copy", None)
    waited: Any = wait_for_copy() if callable(wait_for_copy) else None
    copy_state = getattr(pre_async, "copy_state", None)
    chunks_cpu = getattr(copy_state, "chunks_cpu", None)
    if chunks_cpu:
        flat_chunks = [c.reshape(-1) for c in chunks_cpu]
        return (
            flat_chunks[0] if len(flat_chunks) == 1 else torch.cat(flat_chunks, dim=0)
        )
    if isinstance(waited, torch.Tensor):
        return waited.reshape(-1)
    next_tpu: Any = getattr(pre_async, "next_tokens_tpu", None)
    if isinstance(next_tpu, torch.Tensor):
        return next_tpu.to("cpu").reshape(-1)
    return None


def _resolve_decode_lookback_cpu(
    runner, batch, num_reqs: int, start_index: int
) -> torch.Tensor:
    """Fill lookback_token_ids on CPU for both prompt and generated positions."""
    lookback = runner.lookback_token_ids.cpu.clone()
    if num_reqs <= 0:
        return lookback
    depth = int(lookback.shape[1])
    starts = np.asarray(
        batch.num_computed_tokens_cpu[start_index : start_index + num_reqs],
        dtype=np.int64,
    )[:, None]
    pos = starts - np.arange(1, depth + 1, dtype=np.int64)
    token_ids_cpu = np.asarray(batch.token_ids_cpu)
    max_cols = int(token_ids_cpu.shape[1])
    valid = (pos >= 0) & (pos < max_cols)
    rows = np.arange(start_index, start_index + num_reqs, dtype=np.int64)[:, None]
    clipped = np.clip(pos, 0, max(max_cols - 1, 0))
    gathered = np.where(valid, token_ids_cpu[rows, clipped], -1)
    lookback[:num_reqs] = torch.from_numpy(
        np.ascontiguousarray(gathered, dtype=np.int64)
    ).to(dtype=lookback.dtype)
    return lookback


def capture_host_engram_inputs(runner, num_reqs: int) -> HostEngramInputs | None:
    """Snapshot the runner's host hash inputs, or None if they do not apply."""
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    plan = getattr(runner, "_last_sequence_layout_plan", None)
    if plan is None or plan.kind is not SequenceLayoutKind.ALL:
        return None
    ctx = getattr(runner, "_attn_metadata_builder_ctx", None)
    start_index = int(getattr(ctx, "start_index", getattr(plan, "start_index", 0)))
    chunk_num_reqs = min(
        num_reqs,
        int(getattr(ctx, "num_reqs", getattr(plan, "num_reqs", num_reqs))),
    )
    batch = runner.input_batch
    seq_lens = runner.seq_lens_np[:chunk_num_reqs]
    prompt_lens = batch.num_prompt_tokens[start_index : start_index + chunk_num_reqs]
    all_prompt = bool(np.all(seq_lens <= prompt_lens))
    if all_prompt and start_index == 0:
        return HostEngramInputs(
            input_ids=runner.input_ids_cpu,
            positions=runner.positions_cpu,
            query_start_loc=runner.query_start_loc_cpu,
            lookback_token_ids=runner.lookback_token_ids.cpu,
            all_prompt=True,
            host_ready=True,
        )

    pre_async = getattr(runner, "_pre_async_results", None)
    if (
        pre_async is not None
        and (
            getattr(pre_async, "spec_decode_num_rejected_tokens", None) is not None
            or getattr(runner, "speculative_config", None) is not None
        )
    ) or (
        getattr(batch, "num_computed_tokens_cpu", None) is None
        or getattr(batch, "token_ids_cpu", None) is None
    ):
        return HostEngramInputs(
            input_ids=runner.input_ids_cpu,
            positions=runner.positions_cpu,
            query_start_loc=runner.query_start_loc_cpu,
            lookback_token_ids=runner.lookback_token_ids.cpu,
            all_prompt=all_prompt,
            host_ready=False,
        )

    lookback = _resolve_decode_lookback_cpu(runner, batch, chunk_num_reqs, start_index)
    ids = runner.input_ids_cpu
    req_id_to_index = (
        getattr(pre_async, "req_id_to_index_copy", None)
        if pre_async is not None
        else None
    )
    req_ids = getattr(batch, "req_ids", None)
    if req_id_to_index and req_ids is not None:
        dst_idx: list[int] = []
        src_idx: list[int] = []
        qsl = runner.query_start_loc_cpu[: chunk_num_reqs + 1].tolist()
        for i in range(chunk_num_reqs):
            req_id = req_ids[start_index + i]
            if req_id in req_id_to_index:
                q_start = int(qsl[i])
                q_end = int(qsl[i + 1])
                if q_end > q_start:
                    n_sched = q_end - q_start
                    src_start = int(req_id_to_index[req_id])
                    dst_idx.extend(range(q_start, q_end))
                    src_idx.extend(range(src_start, src_start + n_sched))
        if dst_idx:
            ids = runner.input_ids_cpu.clone()
            flat_src = _async_source_tokens_cpu(pre_async)
            if flat_src is None:
                return HostEngramInputs(
                    input_ids=runner.input_ids_cpu,
                    positions=runner.positions_cpu,
                    query_start_loc=runner.query_start_loc_cpu,
                    lookback_token_ids=runner.lookback_token_ids.cpu,
                    all_prompt=all_prompt,
                    host_ready=False,
                )
            ids[dst_idx] = flat_src[src_idx].to(dtype=ids.dtype)

    return HostEngramInputs(
        input_ids=ids,
        positions=runner.positions_cpu,
        query_start_loc=runner.query_start_loc_cpu,
        lookback_token_ids=lookback,
        all_prompt=all_prompt,
        host_ready=True,
    )


def gather_host_engram_hashes(
    engram_hashes: torch.Tensor,
    *,
    dp_shared_memory: bool = False,
    gather_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Gather Engram hashes across DP, using ``dp_group.cpu_group`` on CPU."""
    if engram_hashes.device.type == "cpu" and not dp_shared_memory:
        dp_group = get_engram_dp_group()
        if (
            dp_group is not None
            and getattr(dp_group, "world_size", 1) > 1
            and getattr(dp_group, "cpu_group", None) is not None
        ):
            dp_metadata = (
                getattr(get_forward_context(), "dp_metadata", None)
                if is_forward_context_available()
                else None
            )
            slot = (
                engram_gathered_num_tokens()
                if dp_metadata is not None
                else int(engram_hashes.shape[0])
            )
            if engram_hashes.shape[0] > slot:
                raise ValueError("Engram token count exceeds the DP token slot")
            if engram_hashes.shape[0] < slot:
                pad = engram_hashes.new_full(
                    (slot - engram_hashes.shape[0], *engram_hashes.shape[1:]),
                    DEAD_ID,
                )
                engram_hashes = torch.cat((engram_hashes, pad))
            out = engram_hashes.new_empty(
                (dp_group.world_size * slot, *engram_hashes.shape[1:])
            )
            torch.distributed.all_gather_into_tensor(
                out, engram_hashes.contiguous(), group=dp_group.cpu_group
            )
            return out
    if gather_fn is not None:
        return gather_fn(engram_hashes, dp_shared_memory=dp_shared_memory)
    from vllm.models.deepseek_v41.common.engram import gather_engram_hashes

    return gather_engram_hashes(engram_hashes, dp_shared_memory=dp_shared_memory)


def install_host_engram_inputs() -> None:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    original = TPUModelRunner._prepare_lookback_token_ids
    if getattr(original, "_deepseek_v41_host_engram", False):
        return

    def prepare(self, num_reqs: int) -> torch.Tensor:
        ids = original(self, num_reqs)
        if is_forward_context_available():
            get_forward_context().additional_kwargs[HOST_ENGRAM_KEY] = (
                capture_host_engram_inputs(self, num_reqs)
            )
        return ids

    prepare._deepseek_v41_host_engram = True  # type: ignore[attr-defined]
    TPUModelRunner._prepare_lookback_token_ids = prepare  # type: ignore[method-assign]
