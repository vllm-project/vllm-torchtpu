# SPDX-License-Identifier: Apache-2.0
"""Compile the Qwen3.8 TP8DCP8 HND serving specialization with eight shards.

This is a compile-only regression: abstract inputs preserve the full serving
cache/metadata shapes without allocating the 26.45 GiB KV pool per rank.
The NEW_TOKENS_ONLY pass receives two local Q heads, unlike CACHE_ONLY's
sixteen gathered heads. Both output and LSE must remain live during compile.

With the original VMEM estimate, automatic selection produces the mixed
kernel RPAm-snh-p256-b2-q768-k768 and exceeds TPU7x's 64 MiB VMEM budget.
Compilation is expected to succeed after fixing the estimator; do not catch
the OOM or pin a smaller tile here.

Run from the repository root in its installed Python environment on a
single host with eight TPU7x devices (no model weights or server required)::

    env -u TPU_WORKER_HOSTNAMES -u TPU_PROCESS_ADDRESSES -u TPU_PROCESS_PORT \
      -u TPU_TOPOLOGY -u TPU_CHIPS_PER_HOST_BOUNDS -u TPU_HOST_BOUNDS \
      -u TPU_TOPOLOGY_WRAP -u TPU_WORKER_ID \
      TPU_SKIP_MDS_QUERY=true PYTHONPATH=src \
      python -m pytest -v -s --tb=short --timeout=600 \
      tests/kernels/experimental/batched_rpa_longctx/test_dcp_vmem.py

Baseline: 66.67 MiB required, 63.94 MiB available, 2.73 MiB excess.
Pytest should report FAILED with RESOURCE_EXHAUSTED, not PASSED or XFAIL.
This isolates the new-token RPA compilation, not DCP collective/LSE merging.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)


@pytest.fixture
def mesh():
    if jax.default_backend() != "tpu":
        pytest.skip("LONGCTX compilation requires TPU")
    devices = jax.local_devices()
    if len(devices) < 8:
        pytest.skip(
            "The serving reproduction requires eight local TPU devices")
    tpu_info = pltpu.get_tpu_info()
    if (tpu_info.generation != 7
            or tpu_info.vmem_capacity_bytes != 64 * 1024**2):
        pytest.skip("This reproduction targets TPU7x with 64 MiB VMEM")
    return Mesh(np.asarray(devices[:8]), ("dcp", ))


def test_dcp8_hnd_prefill_auto_blocks_compile_spmd(mesh):
    dcp_size = 8
    num_tokens = 4096
    num_seqs = 8  # Serving pads max_num_seqs=4 to eight metadata slots.
    local_q_heads, kv_heads, head_dim = 2, 1, 256
    # Manager block_size=768 consists of three 256-token kernel pages.
    # These are the actual cache/table shapes from the failed 12-layer run.
    page_size = 256
    local_pages = 216696
    num_page_indices = 4104
    output_specs = (P(None, "dcp"), P("dcp"), P(None, "dcp"))

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("dcp"), P(None, "dcp"), P(), P(), P(), P(), P(), P()),
        out_specs=output_specs,
        check_vma=False,
    )
    def run(cache, query, key, value, seq_lens, page_indices, cu_q_lens,
            distribution):
        return wrapper.ragged_paged_attention(
            queries=query,
            keys=key,
            values=value,
            kv_cache=cache,
            kv_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=head_dim**-0.5,
            k_scale=1.0,
            v_scale=1.0,
            cp_group_size=dcp_size,
            cp_rank=jax.lax.axis_index("dcp").reshape(1),
            attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
            return_lse=True,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        )

    cache_shape = wrapper.get_kv_cache_shape(
        local_pages * dcp_size,
        page_size,
        kv_heads,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    abstract = jax.ShapeDtypeStruct
    args = (
        abstract(cache_shape, jnp.float8_e4m3fn),
        abstract((num_tokens, local_q_heads * dcp_size, head_dim),
                 jnp.bfloat16),
        abstract((num_tokens, kv_heads, head_dim), jnp.float8_e4m3fn),
        abstract((num_tokens, kv_heads, head_dim), jnp.float8_e4m3fn),
        abstract((num_seqs, ), jnp.int32),
        abstract((num_page_indices, ), jnp.int32),
        abstract((num_seqs + 1, ), jnp.int32),
        abstract((3, ), jnp.int32),
    )
    # Donation matches the serving custom op's in-place KV-cache update.
    jax.jit(run, donate_argnums=(0, )).lower(*args).compile()
