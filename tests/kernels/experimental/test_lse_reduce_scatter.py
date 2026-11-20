# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_DCP_MERGE_LSE_RS, no TPU needed: the log-sum-exp reduce-scatter of the
V-space merge over a 16-core mesh.

- Each core's result is the float64 softmax merge, over every core, of the
  bf16 W_UV projections of its head block, within 6 bf16 ulps of the largest
  term (Pallas interpret mode, cores in a shuffled chip order, rows where a
  core owns no key carry NaN partials and log-sum-exp -inf).
- Positive control: the same result is far from an equal-weight merge.
- The shape gate takes the served buckets (1,024 and 2,048 tokens, bf16 or
  fp8 W_UV) and refuses chunks that do not fill whole head-block tiles.

The kernel tests run on 16 devices; CPU virtual devices are sufficient, e.g.
XLA_FLAGS=--xla_force_host_platform_device_count=16."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import lse_reduce_scatter as lrs
from vllm_torchtpu.kernels.experimental import rotated_collectives

RANKS, H, C, V = 16, 64, 128, 256
CHUNK, CHUNKS = 96, 3
O_PARTS = 4
ULPS = 6


@pytest.fixture(scope="module")
def merge():
    """(kernel result [cores, H / 16, t, V], projections [cores, H, t, V],
    log-sum-exps [cores, H, t]) in float64."""
    if jax.device_count() < RANKS:
        pytest.skip("requires 16 devices; CPU virtual devices are sufficient")
    t = CHUNK * CHUNKS
    rng = np.random.default_rng(0)
    o = jnp.asarray(rng.standard_normal((RANKS, t, H, C)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((H, C, V)) * 0.1, jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.random((H, V)) + 0.5, jnp.float32)
    lse = (rng.standard_normal((RANKS, t, H)) * 3 + 20).astype(np.float32)
    lse[rng.random((RANKS, t, H)) < 0.15] = -np.inf
    lse[:, :5] = -np.inf
    o = o.at[:, :5].set(jnp.nan)

    cores = [
        ((x, y, z), k)
        for x in range(2)
        for y in range(2)
        for z in range(2)
        for k in range(2)
    ]
    order = rng.permutation(RANKS)
    table = jnp.asarray(
        rotated_collectives.peer_table([cores[i] for i in order], "reduce_scatter")
    )
    mesh = Mesh(np.asarray(jax.devices()[:RANKS]).reshape(1, RANKS), ("lane", "dcp"))
    spec = P("dcp")

    def body(x, lse_, rank):
        parts = [x[0][c * CHUNK : (c + 1) * CHUNK] for c in range(CHUNKS)]
        parts += parts[-1:] * (O_PARTS - CHUNKS)
        return lrs.lse_matmul_reduce_scatter_chunks(
            tuple(parts),
            w,
            scale,
            lse_[0],
            rank,
            table,
            axis="dcp",
            other_ids={"lane": 0},
            chunk_tokens=CHUNK,
            interpret=pltpu.InterpretParams(),
        )[None]

    def put(a):
        return jax.device_put(a, NamedSharding(mesh, spec))

    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    got = run(
        put(o),
        put(jnp.asarray(lse)),
        put(jnp.arange(RANKS, dtype=jnp.int32).reshape(RANKS, 1)),
    )

    proj = (
        jnp.einsum(
            "nthc,hcv->nhtv",
            o.astype(jnp.float32),
            w.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale[None, :, None, :]
    )
    proj = np.asarray(proj.astype(jnp.bfloat16).astype(jnp.float32), np.float64)
    return (
        np.asarray(got).astype(np.float64),
        proj,
        np.transpose(lse, (0, 2, 1)).astype(np.float64),
    )


def _elements_off(got, proj, weights):
    """Elements of `got` more than ULPS bf16 ulps of the largest live term
    away from the `weights` merge of `proj`, over every core's head block."""
    terms = np.where(weights[..., None] > 0, proj, 0.0)
    want = (terms * weights[..., None]).sum(axis=0)
    hb = H // RANKS
    off = 0
    for core in range(RANKS):
        heads = slice(core * hb, (core + 1) * hb)
        err = np.abs(got[core] - want[heads]) / (
            np.abs(terms[:, heads]).max(axis=0) + 1e-30
        )
        off += int(np.count_nonzero(~(err <= ULPS * 2.0**-8)))
    return off


def _softmax_weights(lse):
    top = lse.max(axis=0)
    w = np.exp(lse - np.where(np.isfinite(top), top, 0.0))
    total = w.sum(axis=0)
    return np.where(total > 0, w / np.where(total > 0, total, 1.0), 0.0)


@pytest.mark.multichip
def test_result_is_the_softmax_merge_of_the_projections(merge):
    got, proj, lse = merge
    assert _elements_off(got, proj, _softmax_weights(lse)) == 0


@pytest.mark.multichip
def test_the_comparison_sees_the_merge_weights(merge):
    got, proj, lse = merge
    equal = np.where(np.isfinite(lse), 1.0, 0.0) / RANKS
    assert _elements_off(got, proj, equal) > 0


@pytest.mark.parametrize("tokens", [1024, 2048])
@pytest.mark.parametrize("w_bytes", [1, 2])
def test_takes_the_served_buckets(tokens, w_bytes):
    assert lrs.chunks_supports(tokens, 64, 512, 256, w_bytes)


@pytest.mark.parametrize(
    "tokens, heads",
    [
        (1000, 64),  # not whole 256-row chunks
        (1024, 48),  # 3 head blocks per 16-head group do not pair up
        (1024, 32),  # 2-head blocks leave 192-row parts, not whole 128-row tiles
    ],
)
def test_refuses_chunks_without_whole_head_block_tiles(tokens, heads):
    assert not lrs.chunks_supports(tokens, heads, 512, 256)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_refuses_o_parts_that_are_not_bf16(dtype):
    """The kernel reads 2 bf16 heads from each 32-bit word of the o parts."""
    o = jnp.zeros((256, 64, 512), dtype)
    with pytest.raises(ValueError, match="takes bf16 o parts"):
        lrs.lse_matmul_reduce_scatter_chunks(
            (o,),
            jnp.zeros((64, 512, 256), jnp.float8_e4m3fn),
            None,
            jnp.zeros((256, 64), jnp.float32),
            jnp.zeros((1,), jnp.int32),
            jnp.zeros((rotated_collectives.CORES,), jnp.int32),
            axis="dcp",
        )
