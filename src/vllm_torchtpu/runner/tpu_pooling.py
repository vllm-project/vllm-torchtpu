# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import torch
from vllm.model_executor.layers.pooler.seqwise import heads, methods
from vllm.model_executor.layers.pooler.seqwise.methods import (
    _MEAN_POOL_ACCUMULATION_CHUNK_BYTES,
    SequencePoolingMethod,
    SequencePoolingMethodOutput,
)
from vllm.utils.torch_utils import async_tensor_h2d
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
                mean_emb = (p.mean_pool_sum / p.mean_pool_count).to(hidden_states.dtype)
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
