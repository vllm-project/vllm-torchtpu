# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention


def test_query_heads_are_padded_only_in_vmem() -> None:
    """Four BF16 query heads require padding to the eight-head VMEM tile."""
    num_q_heads = 4
    num_tokens = 1
    lkv_dim = 128
    r_dim = 128
    page_size = 128
    packing = 2

    ql_nope = jnp.ones((num_q_heads, num_tokens, lkv_dim), jnp.bfloat16)
    q_pe = jnp.ones((num_tokens, num_q_heads, r_dim), jnp.bfloat16)
    new_kv_c = jnp.arange(lkv_dim, dtype=jnp.bfloat16)[None, :]
    new_k_pe = jnp.ones((num_tokens, r_dim), jnp.bfloat16)
    cache_kv = jnp.zeros(
        (1, page_size // packing, packing, lkv_dim + r_dim),
        jnp.bfloat16,
    )

    output, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        jnp.asarray([1], jnp.int32),
        jnp.asarray([0], jnp.int32),
        jnp.asarray([0, 1], jnp.int32),
        jnp.asarray([1, 1, 1], jnp.int32),
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
    )
    output.block_until_ready()

    assert output.shape == (num_q_heads, num_tokens, lkv_dim)
    np.testing.assert_array_equal(
        np.asarray(output),
        np.broadcast_to(np.asarray(new_kv_c), output.shape),
    )
