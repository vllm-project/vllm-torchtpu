# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Port vLLM's paged FP8 accuracy test to TPU, retaining its BF16 reference.

Source: tests/kernels/attention/test_deepgemm_attention.py and calc_diff in
vllm/utils/deep_gemm.py at e37e51dd246cf421c554e7d6f53e178e3fd29085.
PyTorch generates inputs and computes the reference on CPU; CUDA RNG samples
are not reproduced. Only cache packing and the kernel interface are TPU-specific.
"""

import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)


def _ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, width):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        (batch_size * next_n, width), -torch.inf, device=q.device, dtype=torch.float32
    )
    for i, context_len in enumerate(context_lens.tolist()):
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range((context_len + block_size - 1) // block_size):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                -torch.inf,
            )
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, -torch.inf)
    return logits


def _pack_fp8_cache(kv_cache, layout):
    # Same quantizer as upstream kv_cache_cast_to_fp8; transpose bytes, not values.
    amax = kv_cache.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scales = amax / 448.0
    values = (kv_cache * (1.0 / scales)).to(torch.float8_e4m3fn)
    words = values.view(torch.int32).reshape(layout.pages, layout.page_size, 32)
    scale_words = scales.view(torch.int32).reshape(layout.pages, 1, layout.page_size)
    packed = torch.cat((words.transpose(1, 2), scale_words), dim=1)
    packed = torch.nn.functional.pad(
        packed.flatten(1),
        (0, layout.page_rows * 128 - packed.shape[1] * layout.page_size),
    )
    return jnp.asarray(packed.reshape(-1, 128).numpy())


@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("reuse_keys", [True, False])
def test_fp8_paged_mqa_logits_vllm_accuracy(
    batch_size, next_n, reuse_keys, record_property
):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires the TPU paged MQA kernel")
    torch.manual_seed(0)
    random.seed(0)
    width, avg_kv, heads, dim = 4096, 2048, 32, 128
    num_blocks, block_size = width * 2, 64
    q = torch.randn((batch_size, next_n, heads, dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, block_size, 1, dim), dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads), dtype=torch.float32)
    context_lens = torch.randint(
        int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,)
    ).to(torch.int32)
    max_block_len = (
        (context_lens.max().item() + block_size - 1) // block_size * block_size
    )
    block_tables = torch.zeros((batch_size, max_block_len), dtype=torch.int32)
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    counter = 0
    for i in range(batch_size):
        for j in range((int(context_lens[i].item()) + block_size - 1) // block_size):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1
    q_fp8 = q.to(torch.float8_e4m3fn)
    layout = PackedFP8Layout(num_blocks, page_size=block_size)
    cache = _pack_fp8_cache(kv_cache, layout)
    context_lens_2d = (
        context_lens.unsqueeze(-1)
        - next_n
        + 1
        + torch.arange(next_n, dtype=torch.int32)
    ).contiguous()
    actual = fp8_paged_mqa_logits(
        jnp.asarray(q_fp8.view(torch.uint8).numpy())
        .view(jnp.float8_e4m3fn)
        .reshape(-1, heads, dim),
        cache,
        jnp.asarray(weights.numpy()),
        jnp.asarray(context_lens_2d.flatten().numpy()),
        jnp.asarray(block_tables.numpy()),
        query_start_loc=jnp.arange(batch_size + 1, dtype=jnp.int32) * next_n,
        max_model_len=width,
        q_scale=jnp.ones(weights.shape, dtype=jnp.float32),
        query_size=min(next_n, 2),
        cache_layout=layout,
        reuse_keys=reuse_keys,
    )
    logits = torch.from_numpy(np.asarray(actual).copy())
    ref_logits = _ref_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, width
    )
    positions = torch.arange(width).unsqueeze(0).expand(batch_size * next_n, -1)
    row_indices = torch.arange(batch_size * next_n) // next_n
    next_n_offset = torch.arange(batch_size * next_n) % next_n
    mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )
    x = logits.masked_fill(~mask, 0).double()
    y = ref_logits.masked_fill(~mask, 0).double()
    denominator = (x * x + y * y).sum()
    diff = 1 - 2 * (x * y).sum() / denominator
    record_property("calc_diff", diff.item())
    assert diff < 1e-3, f"{diff=}"
