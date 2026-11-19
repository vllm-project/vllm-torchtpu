# SPDX-License-Identifier: Apache-2.0
"""Main-KV DCP sparse prefill attends only the rank's page of each KV tile.

A DCP KV tile is one table entry (dcp_world pages of tokens); the rank holds
the page at token offset dcp_rank * page. Merging every rank's (out, lse)
must equal the replicated prefill over the same logical keys. TPU only: the
Pallas interpreter cannot run the kernel's bitcast FP8 cache loads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_prefill,
)

RATIO, PAGE, TOKENS = 4, 512, 512


@pytest.mark.tpu_test
@pytest.mark.parametrize("dcp", [2, 4])
def test_dcp_ranks_merge_to_the_replicated_prefill(dcp):
    rng = np.random.default_rng(dcp)
    entries = 3
    logical_tokens = entries * dcp * PAGE
    query = jnp.asarray(rng.normal(size=(TOKENS, 12, 256)), jnp.bfloat16)
    lengths = np.linspace(1, logical_tokens, TOKENS).astype(np.int32)
    blocks = np.full((TOKENS, 512), -1, np.int32)
    for r, n in enumerate(lengths):
        pick = rng.permutation(-(-n // RATIO))[:512]
        blocks[r, : len(pick)] = pick
    rows = rng.normal(size=(logical_tokens, 1, 4, 256)).astype(np.float32)
    # Replicated: logical page p in physical page p + 1 (page 0 is null).
    replicated = np.zeros((entries * dcp + 1, PAGE, 1, 4, 256), np.float32)
    replicated[1:] = rows.reshape(entries * dcp, PAGE, 1, 4, 256)
    table = np.broadcast_to(np.arange(1, entries * dcp + 1, dtype=np.int32), (TOKENS, entries * dcp))
    common = dict(ratio=RATIO, kv_heads=1, query_block=512, bf16_io=True)
    expect = block_sparse_gqa_prefill(
        query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table.copy()),
        jnp.asarray(replicated, jnp.float8_e4m3fn), kv_tile=PAGE, **common,
    )
    outs, lses = [], []
    for rank in range(dcp):
        # Rank r holds token t with (t // PAGE) % dcp == r at (entry t // (dcp*PAGE), t % PAGE).
        local = np.zeros((entries + 1, PAGE, 1, 4, 256), np.float32)
        for e in range(entries):
            page = e * dcp + rank
            local[e + 1] = rows[page * PAGE : (page + 1) * PAGE]
        local_table = np.broadcast_to(np.arange(1, entries + 1, dtype=np.int32), (TOKENS, entries))
        out, lse = block_sparse_gqa_prefill(
            query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(local_table.copy()),
            jnp.asarray(local, jnp.float8_e4m3fn), kv_tile=dcp * PAGE,
            dcp_rank=jnp.asarray([rank], jnp.int32), dcp_world=dcp, return_lse=True, **common,
        )
        outs.append(np.asarray(out, np.float32))
        lses.append(np.asarray(lse, np.float32))
    lse = np.stack(lses)
    top = np.max(lse, axis=0)
    weight = np.where(np.isneginf(lse), 0.0, np.exp(lse - top))
    merged = np.einsum("rth,rthd->thd", weight, np.stack(outs)) / weight.sum(0)[..., None]
    np.testing.assert_allclose(merged, np.asarray(expect, np.float32), rtol=2e-2, atol=2e-2)
