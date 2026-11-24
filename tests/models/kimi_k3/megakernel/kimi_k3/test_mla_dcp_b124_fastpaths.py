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

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend import core as jex_core

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import mla_dcp


def _trace_dcp_attention(
    batch: int,
    *,
    cache_dtype=jnp.float8_e4m3fn,
    sequence_rows: int = 1,
    skip: tuple[str, ...] = (),
) -> jex_core.ClosedJaxpr:
    group = 32
    heads = mla_dcp.HEADS_PER_RANK
    page_tokens = 512 if cache_dtype == jnp.float8_e4m3fn else 256
    packing = 4 // jnp.dtype(cache_dtype).itemsize
    plan = mla_dcp.page_plan(batch, heads, sequence_rows, cache_dtype)
    persistent = mla_dcp.persistent_scratch(
        batch,
        group=group,
        heads=heads,
        cache_dtype=cache_dtype,
        sequence_rows=sequence_rows,
    )
    scoped = mla_dcp.scoped_scratch(
        batch,
        group=group,
        heads=heads,
        cache_dtype=cache_dtype,
        page_tokens=page_tokens,
        sequence_rows=sequence_rows,
        **plan,
    )
    pages = 4

    def kernel(
        pos_ref, table_ref, layer_ref, pool_ref, q_ref, nl_ref, out_ref, *scratch
    ):
        cache = mla_dcp.DcpCache(
            (pool_ref,), table_ref, pages, layer=layer_ref[0], group=group
        )
        workspace = mla_dcp.DcpWorkspace(*scratch)
        attention = mla_dcp.DcpAttention(
            cache,
            workspace,
            pos_ref,
            batch,
            sequence_rows=sequence_rows,
            skip=skip,
        )
        attention.prime()
        out_ref[...] = attention.attend(q_ref[...], nl_ref[...], 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, heads, mla_dcp.LATENT), jnp.float32),
        in_specs=[smem, smem, smem, pl.BlockSpec(memory_space=pltpu.HBM), vmem, vmem],
        out_specs=vmem,
        scratch_shapes=(*persistent, *scoped),
    )
    operands = (
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch * pages,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct(
            (batch * pages, page_tokens // packing, packing, mla_dcp.LANES),
            cache_dtype,
        ),
        jax.ShapeDtypeStruct((batch, heads, mla_dcp.LANES), jnp.float32),
        jax.ShapeDtypeStruct((batch, mla_dcp.LANES), jnp.float32),
    )
    return jax.make_jaxpr(call, axis_env=[("tp", group)])(*operands)


def _count_primitive(jaxpr, name: str) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        count += eqn.primitive.name == name
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    count += _count_primitive(inner, name)
    return count


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_should_keep_bf16_query_gather_wire_for_low_batch_fp8_kv_decode(
    batch: int,
) -> None:
    pers = mla_dcp.persistent_scratch(
        batch,
        group=32,
        heads=mla_dcp.HEADS_PER_RANK,
        cache_dtype=jnp.float8_e4m3fn,
        sequence_rows=1,
    )
    assert pers[0].dtype == jnp.bfloat16


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_should_stage_native_pages_and_cap_fp8_span_for_low_batch_plain_decode(
    batch: int,
) -> None:
    assert mla_dcp.native_pages(batch, mla_dcp.HEADS_PER_RANK, 1, jnp.float8_e4m3fn)
    assert (
        mla_dcp.staged_folds(batch, mla_dcp.HEADS_PER_RANK, 1, jnp.float8_e4m3fn) == 1
    )
    scoped = mla_dcp.scoped_scratch(
        batch,
        span=8,
        buffers=4,
        heads=mla_dcp.HEADS_PER_RANK,
        group=32,
        cache_dtype=jnp.float8_e4m3fn,
        page_tokens=512,
        sequence_rows=1,
    )
    pages_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp latent pages")]
    bounds_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup bounds")]
    counts_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup counts")]
    assert pages_spec.shape == (4, 1024, mla_dcp.LANES)
    assert bounds_spec.shape == (batch,)
    assert counts_spec.shape == (batch,)


@pytest.mark.parametrize(
    ("batch", "heads", "sequence_rows", "cache_dtype", "ktile"),
    [
        (1, 3, 1, jnp.bfloat16, 1024),
        (2, 3, 2, jnp.float8_e4m3fn, 1024),
        (4, 6, 1, jnp.float8_e4m3fn, 1024),
        (4, 3, 1, jnp.float8_e4m3fn, 2048),
        (3, 3, 1, jnp.float8_e4m3fn, 1024),
    ],
)
def test_should_disable_low_batch_native_pages_when_preconditions_fail(
    batch: int,
    heads: int,
    sequence_rows: int,
    cache_dtype,
    ktile: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_kwargs = (
        {"mla_decode_ktile": ktile, "attention_dp": 2, "dcp_group": 16}
        if ktile == 2048
        else {"mla_decode_ktile": ktile}
    )
    monkeypatch.setattr(k3_config, "_ACTIVE", k3_config.KernelConfig(**cfg_kwargs))
    assert not mla_dcp.native_pages(batch, heads, sequence_rows, cache_dtype)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_should_trace_split_tail_range_for_fp8_and_pipeline_fallback_for_bf16(
    batch: int,
) -> None:
    fp8_jaxpr = _trace_dcp_attention(batch, cache_dtype=jnp.float8_e4m3fn).jaxpr
    bf16_jaxpr = _trace_dcp_attention(batch, cache_dtype=jnp.bfloat16).jaxpr
    skipped_jaxpr = _trace_dcp_attention(
        batch, cache_dtype=jnp.float8_e4m3fn, skip=("pagecompute",)
    ).jaxpr

    assert _count_primitive(fp8_jaxpr, "dot_general") > _count_primitive(
        bf16_jaxpr, "dot_general"
    )
    assert _count_primitive(bf16_jaxpr, "dot_general") > 0
    assert _count_primitive(skipped_jaxpr, "dot_general") == 0


def test_should_trace_staged_and_chunked_verify_branches_in_attend_range() -> None:
    staged_b32 = _trace_dcp_attention(32, cache_dtype=jnp.float8_e4m3fn).jaxpr
    verify_b32 = _trace_dcp_attention(
        32, cache_dtype=jnp.float8_e4m3fn, sequence_rows=4
    ).jaxpr
    verify_b4 = _trace_dcp_attention(
        4, cache_dtype=jnp.float8_e4m3fn, sequence_rows=2
    ).jaxpr

    assert _count_primitive(staged_b32, "dot_general") > 0
    assert _count_primitive(verify_b32, "dot_general") > 0
    assert _count_primitive(verify_b4, "dot_general") > 0
