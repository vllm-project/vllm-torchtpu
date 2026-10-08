# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise import heads, methods
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import (
    _MEAN_POOL_ACCUMULATION_CHUNK_BYTES,
    CLSPool,
    LastPool,
    MeanPool,
    SequencePoolingMethod,
    SequencePoolingMethodOutput,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


class PromptTokenIdsTensor(torch.Tensor):
    """A 2D CPU torch.Tensor representing batched prompt token IDs.

    Also provides dictionary mapping access `{req_id: prompt_token_ids}`
    so StepPool can inspect token tags (e.g. `<step>` or `\\n\\n`) on host CPU
    while preserving standard tensor indexing and slicing for PoolingMetadata.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, req_id_map: dict[str, list[int]]):
        obj = torch.Tensor._make_subclass(cls, tensor)
        obj.req_id_map = req_id_map
        return obj

    def __init__(self, tensor: torch.Tensor, req_id_map: dict[str, list[int]]):
        self.req_id_map = req_id_map

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.req_id_map[key]
        item = self.as_subclass(torch.Tensor)[key]
        if item.dim() >= 2:
            return PromptTokenIdsTensor(item, self.req_id_map)
        return item

    def get(self, key, default=None):
        if isinstance(key, str):
            return self.req_id_map.get(key, default)
        return default

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.req_id_map
        return False

    def items(self):
        return self.req_id_map.items()

    def keys(self):
        return self.req_id_map.keys()

    def values(self):
        return self.req_id_map.values()


class TPUMeanPool(SequencePoolingMethod):
    """TPU-native MeanPool with online FP32 accumulation across chunked prefill."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        if not pooling_cursor.is_partial_prefill():
            prompt_lens_cpu = pooling_cursor.prompt_lens_cpu
            num_seqs = prompt_lens_cpu.numel()
            hidden_size = hidden_states.shape[-1]

            if num_seqs == 0:
                return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)

            if isinstance(prompt_lens_cpu, torch.Tensor):
                prompt_lens = prompt_lens_cpu.to(
                    device=hidden_states.device, dtype=torch.int64, non_blocking=True
                )
            else:
                prompt_lens = async_tensor_h2d(
                    prompt_lens_cpu, device=hidden_states.device, dtype=torch.int64
                )
            segment_ids = torch.repeat_interleave(
                torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
                prompt_lens,
                output_size=int(prompt_lens_cpu.sum()),
            )
            segment_sums = torch.zeros(
                (num_seqs, hidden_size),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
            chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)

            for start in range(0, hidden_states.shape[0], chunk_size):
                end = min(start + chunk_size, hidden_states.shape[0])
                segment_sums.index_add_(
                    0,
                    segment_ids[start:end],
                    hidden_states[start:end].to(dtype=torch.float32),
                )

            return segment_sums / prompt_lens.unsqueeze(1)

        # Chunked prefill online accumulation on TPU in FP32
        num_scheduled_cpu = pooling_cursor.num_scheduled_tokens_cpu.tolist()
        hidden_states_lst = list(torch.split(hidden_states, num_scheduled_cpu))
        finished_mask = pooling_cursor.is_finished().tolist()
        pooling_states = pooling_metadata.pooling_states

        output_list: list[torch.Tensor | None] = []
        for hs_chunk, p, finished in zip(
            hidden_states_lst, pooling_states, finished_mask
        ):
            # Accumulate on-device in FP32 to prevent numerical cancellation/underflow
            chunk_sum = hs_chunk.to(dtype=torch.float32).sum(dim=0)
            chunk_count = hs_chunk.shape[0]

            if not hasattr(p, "mean_pool_sum") or p.mean_pool_sum is None:
                p.mean_pool_sum = chunk_sum
                p.mean_pool_count = chunk_count
            else:
                p.mean_pool_sum = p.mean_pool_sum + chunk_sum
                p.mean_pool_count = p.mean_pool_count + chunk_count

            if finished:
                # Keep fp32 like the single-pass path and upstream MeanPool;
                # EmbeddingPoolerHead applies head_dtype afterwards.
                mean_emb = p.mean_pool_sum / p.mean_pool_count
                output_list.append(mean_emb)
                # Cleanup state to prevent leaks in CachedRequestState
                p.mean_pool_sum = None
                p.mean_pool_count = 0
                if hasattr(p, "clean"):
                    p.clean()
            else:
                output_list.append(None)

        return output_list


_tpu_pooling_patches_lock = threading.Lock()
_tpu_pooling_patches_installed = False


def install_tpu_pooling_patches() -> None:
    """Install TPU-native pooling patches non-invasively (thread-safe, idempotent)."""
    global _tpu_pooling_patches_installed
    with _tpu_pooling_patches_lock:
        if _tpu_pooling_patches_installed:
            return
        _install_tpu_pooling_patches_locked()
        _tpu_pooling_patches_installed = True


def _install_tpu_pooling_patches_locked() -> None:
    # 1. Patch MeanPool to use TPU online accumulation in Chunked Prefill
    methods.MeanPool.forward = TPUMeanPool.forward

    # 2. Patch EmbeddingPoolerHead to safely handle None elements in Chunked Prefill
    orig_embed_forward = heads.EmbeddingPoolerHead.forward

    def safe_embed_forward(self, pooled_data, pooling_metadata):
        if isinstance(pooled_data, list) and any(d is None for d in pooled_data):
            outputs = []
            for i, (d, p) in enumerate(
                zip(pooled_data, pooling_metadata.pooling_params)
            ):
                if d is None:
                    outputs.append(None)
                else:
                    # d is the pooled [hidden_size] vector, so d.shape[0] is
                    # hidden_size, not the prompt length. Use the real one.
                    sub_meta = PoolingMetadata(
                        prompt_lens=pooling_metadata.prompt_lens[i : i + 1],
                        prompt_token_ids=None,
                        prompt_token_ids_cpu=None,
                        pooling_params=[p],
                        pooling_states=[],
                    )
                    single_out = orig_embed_forward(self, [d], sub_meta)
                    if (
                        isinstance(single_out, torch.Tensor)
                        and single_out.shape[0] == 1
                    ):
                        outputs.append(single_out[0])
                    else:
                        outputs.append(single_out)
            return outputs
        return orig_embed_forward(self, pooled_data, pooling_metadata)

    heads.EmbeddingPoolerHead.forward = safe_embed_forward

    # 3. Patch PoolingMetadata._get_prompt_token_ids to gracefully accept dict mapping
    orig_get_prompt_token_ids = PoolingMetadata._get_prompt_token_ids

    def safe_get_prompt_token_ids(self, prompt_token_ids):
        if prompt_token_ids is None:
            raise ValueError(
                "prompt_token_ids is required but was not set. "
                "Please set `requires_token_ids=True` in `get_pooling_updates`"
            )
        if isinstance(prompt_token_ids, dict):
            return [
                torch.tensor(t, dtype=torch.int32) for t in prompt_token_ids.values()
            ]
        return orig_get_prompt_token_ids(self, prompt_token_ids)

    PoolingMetadata._get_prompt_token_ids = safe_get_prompt_token_ids


def build_pooling_metadata(
    batch, runner_requests: dict | None = None
) -> PoolingMetadata:
    """Build PoolingMetadata with prompt_token_ids_cpu and persistent pooling_states."""
    reqs = getattr(batch, "requests", None)
    if reqs is None:
        if runner_requests is not None and hasattr(batch, "req_ids"):
            num_reqs = getattr(batch, "num_reqs", len(batch.req_ids))
            req_ids = batch.req_ids[:num_reqs]
            reqs = [runner_requests[rid] for rid in req_ids]
        else:
            reqs = []
    else:
        num_reqs = getattr(batch, "num_reqs", len(reqs))
        reqs = reqs[:num_reqs]

    pooling_params = [r.pooling_params for r in reqs]
    pooling_states = []
    for r in reqs:
        p_state = getattr(r, "pooling_states", None)
        if p_state is None:
            p_state = PoolingStates()
            if hasattr(r, "pooling_states"):
                r.pooling_states = p_state
        pooling_states.append(p_state)

    def _extract_prompt_len(r):
        val = getattr(r, "prompt_token_ids_len", None)
        if isinstance(val, int):
            return val
        val = getattr(r, "num_prompt_tokens", None)
        if isinstance(val, int):
            return val
        p_tokens = getattr(r, "prompt_token_ids", None)
        if p_tokens is not None and hasattr(p_tokens, "__len__"):
            return len(p_tokens)
        return 0

    prompt_lens = torch.tensor(
        [_extract_prompt_len(r) for r in reqs], dtype=torch.int32
    )

    prompt_token_ids_map: dict[str, list[int]] = {}
    max_prompt_len = 0
    for r in reqs:
        p_tokens = getattr(r, "prompt_token_ids", None)
        if p_tokens is not None:
            prompt_token_ids_map[r.req_id] = p_tokens
            if len(p_tokens) > max_prompt_len:
                max_prompt_len = len(p_tokens)
        else:
            prompt_token_ids_map[r.req_id] = []

    if max_prompt_len == 0 and len(prompt_lens) > 0:
        max_prompt_len = int(prompt_lens.max().item())

    tensor = torch.zeros((len(reqs), max_prompt_len), dtype=torch.int32)
    for i, r in enumerate(reqs):
        p_tokens = getattr(r, "prompt_token_ids", None)
        if p_tokens:
            tensor[i, : len(p_tokens)] = torch.tensor(p_tokens, dtype=torch.int32)

    prompt_token_ids_cpu = PromptTokenIdsTensor(tensor, prompt_token_ids_map)

    return PoolingMetadata(
        prompt_lens=prompt_lens,
        prompt_token_ids=None,
        prompt_token_ids_cpu=prompt_token_ids_cpu,
        pooling_params=pooling_params,
        pooling_states=pooling_states,
    )


# ---------------------------------------------------------------------------
# Static (AOT-bucketed) pooler
#
# The eager pooler consumes data-dependent shapes ([total_scheduled_tokens, H]
# and [num_reqs] cursors), so on TPU nearly every step lands on a new compile.
# The static pooler instead consumes the padded backbone output [T_pad, H] and
# fixed-size [R_pad] / [T_pad] index tensors built on the host, which collapses
# its shape space to the finite (T_pad, R_pad) bucket grid that
# TPUModelRunner._precompile_pooler_subgraphs compiles ahead of time.
# ---------------------------------------------------------------------------

_STATIC_POOLING_TYPES: dict[type, str] = {
    LastPool: "LAST",
    CLSPool: "CLS",
    MeanPool: "MEAN",
}


@dataclass(frozen=True)
class StaticPoolerSpec:
    """Per-process static configuration of the compiled pooler."""

    # "LAST" | "CLS" | "MEAN".
    pooling_type: str
    # EmbeddingPoolerHead.head_dtype (fp32 by default for pooling models).
    head_dtype: torch.dtype | None
    # Sentence-Transformers projector; None for e.g. Qwen3-Embedding.
    projector: torch.nn.Module | None
    # True when the head activation is PoolerNormalize.
    normalize: bool
    # Width of the pooled output after the projector.
    out_dim: int


def _resolve_head_dtype(head_dtype: Any) -> tuple[torch.dtype | None, bool]:
    if head_dtype is None or isinstance(head_dtype, torch.dtype):
        return head_dtype, True
    if isinstance(head_dtype, str):
        resolved = getattr(torch, head_dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved, True
    return None, False


def _projector_out_dim(projector: torch.nn.Module) -> int | None:
    last_linear = None
    for module in projector.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    return None if last_linear is None else int(last_linear.out_features)


def resolve_static_pooler_spec(
    model: Any, model_config: Any
) -> tuple[StaticPoolerSpec | None, str]:
    """Return ``(spec, reason)`` for the model's embed pooler.

    The spec is None (and ``reason`` explains why) unless the embed pooler is a
    ``SequencePooler`` with a LAST / CLS / MEAN pooling method and a plain
    ``EmbeddingPoolerHead``. Everything else (StepPool, classifier heads,
    custom activations, ...) stays on the eager pooler path.
    """
    pooler = getattr(model, "pooler", None)
    if pooler is None:
        return None, "model has no pooler"
    if isinstance(pooler, DispatchPooler):
        pooler = pooler.poolers_by_task.get("embed")
        if pooler is None:
            return None, "DispatchPooler has no 'embed' sub-pooler"
    if type(pooler) is not SequencePooler:
        return None, f"unsupported pooler type {type(pooler).__name__}"

    pooling_type = _STATIC_POOLING_TYPES.get(type(pooler.pooling))
    if pooling_type is None:
        return None, f"unsupported pooling method {type(pooler.pooling).__name__}"

    head = pooler.head
    if type(head) is not EmbeddingPoolerHead:
        return None, f"unsupported pooler head {type(head).__name__}"

    activation = head.activation
    if activation is None:
        normalize = False
    elif type(activation) is PoolerNormalize:
        normalize = True
    else:
        return None, f"unsupported head activation {type(activation).__name__}"

    projector = head.projector
    if projector is not None and not isinstance(projector, torch.nn.Module):
        return None, "projector is not an nn.Module"

    head_dtype, ok = _resolve_head_dtype(head.head_dtype)
    if not ok:
        return None, f"unsupported head_dtype {head.head_dtype!r}"

    try:
        if projector is None:
            out_dim = int(model_config.get_hidden_size())
        else:
            out_dim = _projector_out_dim(projector) or int(model_config.embedding_size)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"cannot determine embedding size: {exc}"

    return (
        StaticPoolerSpec(
            pooling_type=pooling_type,
            head_dtype=head_dtype,
            projector=projector,
            normalize=normalize,
            out_dim=out_dim,
        ),
        "ok",
    )


def _static_pool_impl(
    hidden_states: torch.Tensor,
    gather_idx: torch.Tensor | None,
    seg_ids: torch.Tensor | None,
    lens: torch.Tensor | None,
    dims: torch.Tensor,
    use_act: torch.Tensor,
    *,
    pooling_type: str,
    head_dtype: torch.dtype | None,
    projector: torch.nn.Module | None,
    normalize: bool,
) -> torch.Tensor:
    """Pool a padded bucket into ``[R_pad, out_dim]``.

    hidden_states: [T_pad, H] padded backbone output.
    gather_idx:    [R_pad] int32 token index (LAST / CLS); None for MEAN.
    seg_ids:       [T_pad] int32 request id per token (MEAN); padded tokens
                   point at the trash row R_pad. None for LAST / CLS.
    lens:          [R_pad] fp32 token counts (MEAN; padded rows = 1.0).
    dims:          [R_pad] int32 matryoshka width (out_dim when unset).
    use_act:       [R_pad] bool per-request normalize flag.

    Only static-shape ops, no host syncs and no data-dependent Python control
    flow, so it traces under ``fullgraph=True, dynamic=False``.
    """
    if pooling_type == "MEAN":
        assert seg_ids is not None and lens is not None
        rows = lens.shape[0]
        num_tokens, hidden_size = hidden_states.shape
        # One extra trash row absorbs padded tokens; index_add never mixes
        # them into real rows (unlike a one-hot matmul, where 0 * NaN
        # contaminates every row).
        sums = hidden_states.new_zeros((rows + 1, hidden_size), dtype=torch.float32)
        step = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // (hidden_size * 4))
        for start in range(0, num_tokens, step):  # static trip count
            end = min(start + step, num_tokens)
            sums = sums.index_add(
                0, seg_ids[start:end], hidden_states[start:end].to(torch.float32)
            )
        # Divide by exact counts, as upstream MeanPool does.
        pooled = sums[:rows] / lens[:, None]
    else:
        assert gather_idx is not None
        pooled = torch.index_select(hidden_states, 0, gather_idx)

    if head_dtype is not None:
        pooled = pooled.to(head_dtype)
    if projector is not None:
        pooled = projector(pooled)

    # Matryoshka: zero the tail columns so the norm matches slice-then-normalize;
    # the host slices [:d] afterwards.
    col = torch.arange(pooled.shape[-1], device=pooled.device, dtype=torch.int32)
    pooled = torch.where(col[None, :] < dims[:, None], pooled, torch.zeros_like(pooled))
    if normalize:
        pooled = torch.where(use_act[:, None], F.normalize(pooled, p=2, dim=-1), pooled)
    return pooled


_static_pool_compiled = torch.compile(
    _static_pool_impl, backend="tpu", fullgraph=True, dynamic=False
)


@dataclass
class StaticPoolInputs:
    """Device inputs of one static-pool call plus host-side bookkeeping."""

    gather_idx: torch.Tensor | None
    seg_ids: torch.Tensor | None
    lens: torch.Tensor | None
    dims: torch.Tensor
    use_act: torch.Tensor
    # Host-only.
    num_reqs: int
    finished: np.ndarray
    out_dims: list[int | None]


def static_pool(
    hidden_states: torch.Tensor,
    inputs: StaticPoolInputs,
    spec: StaticPoolerSpec,
) -> torch.Tensor:
    """Run the static pooler: compiled on TPU, eager elsewhere (CPU tests)."""
    fn = (
        _static_pool_compiled
        if hidden_states.device.type == "tpu"
        else _static_pool_impl
    )
    return fn(
        hidden_states,
        inputs.gather_idx,
        inputs.seg_ids,
        inputs.lens,
        inputs.dims,
        inputs.use_act,
        pooling_type=spec.pooling_type,
        head_dtype=spec.head_dtype,
        projector=spec.projector,
        normalize=spec.normalize,
    )


def _to_device(array: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(array).to(device, non_blocking=True)


def build_static_pool_inputs(
    spec: StaticPoolerSpec,
    pooling_params: Sequence[Any],
    prompt_lens_np: np.ndarray,
    num_computed_np: np.ndarray,
    num_scheduled_np: np.ndarray,
    num_tokens_padded: int,
    padded_num_reqs: int,
    device: torch.device | str,
    last_token_idx: torch.Tensor | None = None,
) -> StaticPoolInputs | None:
    """Build padded static-pool inputs on the host; None if ineligible.

    A chunk is ineligible (and takes the eager path) when a request is not an
    ``embed`` task, the chunk does not fit the padded bucket, or, for CLS /
    MEAN, any request is a partial prefill.

    LAST reuses ``last_token_idx`` (the runner's ``logits_indices``, already on
    device, padded with ``total - 1``) when it is an int32 ``[padded_num_reqs]``
    tensor; otherwise it builds the same values on the host.
    """
    num_reqs = len(pooling_params)
    if num_reqs == 0 or num_reqs > padded_num_reqs:
        return None
    if any(p is None or getattr(p, "task", None) != "embed" for p in pooling_params):
        return None

    prompt_lens_np = np.asarray(prompt_lens_np, dtype=np.int64)
    num_computed_np = np.asarray(num_computed_np, dtype=np.int64)
    num_scheduled_np = np.asarray(num_scheduled_np, dtype=np.int64)
    if not (
        prompt_lens_np.shape[0]
        == num_computed_np.shape[0]
        == num_scheduled_np.shape[0]
        == num_reqs
    ):
        return None
    total = int(num_scheduled_np.sum())
    if total <= 0 or total > num_tokens_padded:
        return None

    is_full_prefill = bool(
        np.all(num_computed_np == 0) and np.all(num_scheduled_np == prompt_lens_np)
    )
    if spec.pooling_type in ("CLS", "MEAN") and not is_full_prefill:
        return None

    finished = (num_computed_np + num_scheduled_np) == prompt_lens_np
    query_start = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(num_scheduled_np, out=query_start[1:])

    gather_idx = seg_ids = lens = None
    if spec.pooling_type == "LAST":
        if (
            last_token_idx is not None
            and tuple(last_token_idx.shape) == (padded_num_reqs,)
            and last_token_idx.dtype == torch.int32
        ):
            gather_idx = last_token_idx
        else:
            idx = np.full(padded_num_reqs, total - 1, dtype=np.int32)
            idx[:num_reqs] = query_start[1:] - 1
            gather_idx = _to_device(idx, device)
    elif spec.pooling_type == "CLS":
        idx = np.zeros(padded_num_reqs, dtype=np.int32)
        idx[:num_reqs] = query_start[:-1]
        gather_idx = _to_device(idx, device)
    else:  # MEAN
        seg = np.full(num_tokens_padded, padded_num_reqs, dtype=np.int32)
        seg[:total] = np.repeat(np.arange(num_reqs, dtype=np.int32), num_scheduled_np)
        seg_ids = _to_device(seg, device)
        lens_np = np.ones(padded_num_reqs, dtype=np.float32)
        lens_np[:num_reqs] = num_scheduled_np
        lens = _to_device(lens_np, device)

    dims_np = np.full(padded_num_reqs, spec.out_dim, dtype=np.int32)
    use_act_np = np.zeros(padded_num_reqs, dtype=np.bool_)
    out_dims: list[int | None] = []
    for i, p in enumerate(pooling_params):
        d = getattr(p, "dimensions", None)
        if d is not None:
            d = min(max(int(d), 1), spec.out_dim)
            dims_np[i] = d
        out_dims.append(d)
        use_act_np[i] = bool(getattr(p, "use_activation", None))

    return StaticPoolInputs(
        gather_idx=gather_idx,
        seg_ids=seg_ids,
        lens=lens,
        dims=_to_device(dims_np, device),
        use_act=_to_device(use_act_np, device),
        num_reqs=num_reqs,
        finished=finished,
        out_dims=out_dims,
    )


def build_dummy_static_pool_inputs(
    spec: StaticPoolerSpec,
    num_tokens: int,
    num_reqs: int,
    device: torch.device | str,
) -> StaticPoolInputs:
    """Precompile inputs with the exact runtime dtypes and shapes.

    Every argument is a distinct tensor: Dynamo specializes on aliasing, so
    reusing one buffer for two arguments would compile a graph that serving
    never hits.
    """
    gather_idx = seg_ids = lens = None
    if spec.pooling_type in ("LAST", "CLS"):
        gather_idx = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    else:
        seg_ids = torch.full((num_tokens,), num_reqs, dtype=torch.int32, device=device)
        lens = torch.ones(num_reqs, dtype=torch.float32, device=device)
    return StaticPoolInputs(
        gather_idx=gather_idx,
        seg_ids=seg_ids,
        lens=lens,
        dims=torch.full((num_reqs,), spec.out_dim, dtype=torch.int32, device=device),
        use_act=torch.ones(num_reqs, dtype=torch.bool, device=device),
        num_reqs=num_reqs,
        finished=np.zeros(num_reqs, dtype=np.bool_),
        out_dims=[None] * num_reqs,
    )


def iter_pooler_precompile_grid(
    num_tokens_paddings: Sequence[int],
    num_reqs_paddings: Sequence[int],
    max_num_reqs: int,
):
    """Yield the reachable ``(num_tokens, num_reqs)`` static-pool buckets.

    Same pruning as the logits precompile: a padded batch never carries more
    requests than its token bucket (or ``max_num_reqs``).
    """
    for num_tokens in num_tokens_paddings:
        for num_reqs in num_reqs_paddings:
            yield num_tokens, num_reqs
            if num_reqs >= min(num_tokens, max_num_reqs):
                break


@dataclass
class StaticPoolerChunkOutput:
    """Deferred static-pool output of one chunk (unpacked in sample_tokens)."""

    # [R_pad, D] on device; None when no request in the chunk finished.
    padded: torch.Tensor | None
    num_reqs: int
    finished: np.ndarray
    out_dims: list[int | None]


def unpack_static_pool_output(
    out: StaticPoolerChunkOutput,
) -> list[torch.Tensor | None]:
    """One D2H per chunk; None for unfinished rows; ``row[:d]`` for matryoshka.

    The full padded tensor is copied and sliced on the host: slicing
    ``[:num_reqs]`` on device would add a per-num_reqs program.
    """
    if out.padded is None:
        return [None] * out.num_reqs
    host = out.padded.cpu()
    width = host.shape[-1]
    rows: list[torch.Tensor | None] = []
    for i in range(out.num_reqs):
        if not out.finished[i]:
            rows.append(None)
            continue
        row = host[i]
        d = out.out_dims[i]
        if d is not None and d < width:
            row = row[:d]
        rows.append(row)
    return rows


def gate_unfinished(pooler_output: Any, finished: np.ndarray) -> Any:
    """Replace rows of unfinished requests with None (eager fallback).

    vLLM's scheduler finishes a pooling request as soon as it receives a
    non-None output, so a LAST-pooled chunked prefill must not emit a row
    until its final chunk (mirrors ``finished_mask`` in vLLM's GPU runner).
    """
    finished = np.asarray(finished, dtype=np.bool_)
    if pooler_output is None or bool(finished.all()):
        return pooler_output
    if isinstance(pooler_output, torch.Tensor):
        rows: list[Any] = list(pooler_output.cpu().unbind(dim=0))
    else:
        rows = list(pooler_output)
    if len(rows) != len(finished):
        raise ValueError(
            f"pooler output has {len(rows)} rows but {len(finished)} requests"
        )
    return [row if done else None for row, done in zip(rows, finished)]


# ---------------------------------------------------------------------------
# Pooling KV cache cap
# ---------------------------------------------------------------------------


def compute_pooling_kv_cache_cap_tokens(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    block_size: int,
    async_scheduling: bool,
    long_prefill_token_threshold: int | None = 0,
    override_tokens: int | None = None,
) -> int:
    """Block-aligned KV token budget for a pure pooling (prefill-only) model.

    Live KV is bounded by the tokens of the in-flight steps (S of them; two
    under async scheduling) plus the earlier chunks of the requests still in
    chunked prefill (P of them), plus per-request tail-block rounding. The
    result is floored at one full ``max_model_len`` request so vLLM's startup
    check still passes.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    floor_blocks = cdiv(max_model_len, block_size)
    if override_tokens is not None and override_tokens > 0:
        blocks = cdiv(override_tokens, block_size)
    else:
        steps = 2 if async_scheduling else 1
        threshold = long_prefill_token_threshold or 0
        partial = (
            min(max_num_seqs, cdiv(max_num_batched_tokens, threshold))
            if threshold > 0
            else 1
        )
        tokens = steps * max_num_batched_tokens + partial * max_model_len
        blocks = cdiv(tokens, block_size) + steps * max_num_seqs
    return max(blocks, floor_blocks) * block_size


def compute_pooling_kv_cache_cap_bytes(
    kv_cache_spec: dict[str, Any] | None,
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    async_scheduling: bool,
    long_prefill_token_threshold: int | None = 0,
    override_tokens: int | None = None,
) -> int | None:
    """Invert vLLM's ``available // page_size // num_layers`` block sizing.

    Returns None (no cap) unless every layer is a full-attention layer with
    one shared page size and block size.
    """
    if not kv_cache_spec:
        return None
    specs = list(kv_cache_spec.values())
    if not all(
        type(s) is FullAttentionSpec
        and getattr(s, "sliding_window", None) is None
        and getattr(s, "attention_chunk_size", None) is None
        for s in specs
    ):
        return None
    page_sizes = {s.page_size_bytes for s in specs}
    block_sizes = {s.block_size for s in specs}
    if len(page_sizes) != 1 or len(block_sizes) != 1:
        return None
    block_size = block_sizes.pop()
    tokens = compute_pooling_kv_cache_cap_tokens(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        async_scheduling=async_scheduling,
        long_prefill_token_threshold=long_prefill_token_threshold,
        override_tokens=override_tokens,
    )
    # +1 block: vLLM's block pool reserves block 0 as the null block.
    num_blocks = tokens // block_size + 1
    return num_blocks * page_sizes.pop() * len(specs)
