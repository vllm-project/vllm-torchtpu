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
"""The small all-to-all's merge mode (TPU_DCP_MLA_SEND_SMALL_A2A_MERGE) on 8
cores against the merge it replaces, on the same received words.

One program runs the plain small all-to-all and the merge mode on the same DCP
send blocks (16- and 32-token buckets, 4 heads per rank). The plain call's
receive buffer goes through the incumbent ``_mask_unpack`` and
``_merge_received`` (cp_mla_attention; where that module does not import, a
numpy transcription of the two). Live rows of the merge mode equal the
incumbent's f32 merge within half a bf16 step plus the f32 rounding of the two
summation orders, and at least 99% of them equal its bf16 rounding bit for bit;
rows past the live count are exactly zero. The send blocks hold lse rows of
-inf with NaN partials (a rank owning nothing), a row no rank owns, garbage
before the lse word row and NaN in padding rows.

On TPU it runs compiled (``multichip``). On a host with 8 CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu``) the
same checks run with the kernel in Pallas interpret mode.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_to_all as a2a

from .oracle import AXIS, N, bf16_ulp

V = 512
LOCAL = 4
# kernels.mla.sparse.kernel.SEND_PAD_ROWS: the lse word row is row PAD - 1 of
# a send block, the partial rows start at PAD.
PAD = 16
ROW0 = (PAD - 1) // 8 * 8


@pytest.fixture(scope="module", params=["tpu", "interpret"])
def merge_mesh(request):
    """The 8 TPU cores of one host, compiled; or 8 CPU devices with the
    kernel in interpret mode."""
    if request.param == "tpu":
        if jax.default_backend() != "tpu" or jax.device_count() < N:
            pytest.skip("needs the 8 TPU cores of one host")
    elif jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip("needs 8 CPU devices (--xla_force_host_platform_device_count)")
    return request.param, Mesh(np.array(jax.devices()[:N]), (AXIS,))


def _words(lse):
    """f32 [..., tl] lse -> its int16 word row [..., V]: hi halves at lanes
    [0, tl), lo halves at [tl, 2 tl), zeros after."""
    bits = lse.astype(np.float32).view(np.int32)
    hi = (bits >> 16).astype(np.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(np.int16)
    row = np.zeros(lse.shape[:-1] + (V,), np.int16)
    tl = lse.shape[-1]
    row[..., :tl] = hi
    row[..., tl : 2 * tl] = lo
    return row


def _send_blocks(tokens, live, seed):
    """[sender, destination, PAD + tl, V] int16 DCP send blocks."""
    rng = np.random.default_rng(seed)
    tl = tokens * LOCAL
    lse = (rng.standard_normal((N, N, tl)) * 3).astype(np.float32)
    part = rng.standard_normal((N, N, tl, V)).astype(np.float32)
    owns_nothing = rng.random((N, N, tl)) < 0.2
    lse[owns_nothing] = -np.inf
    part[owns_nothing] = np.nan
    lse[:, :, 1] = -np.inf
    part[:, :, 1] = np.nan
    part = part.astype(jnp.bfloat16).view(np.int16)
    part[:, :, live:] = rng.integers(-32768, 32767, size=part[:, :, live:].shape)
    part[:, :, live:, :2] = np.array(np.nan, jnp.bfloat16).view(np.int16)
    blocks = rng.integers(-32768, 32767, size=(N, N, PAD + tl, V)).astype(np.int16)
    blocks[:, :, PAD - 1] = _words(lse)
    blocks[:, :, PAD:] = part
    return blocks


def _merge_numpy(recv, live, tl):
    """``_mask_unpack`` + ``_merge_received`` in numpy f32 on one core's
    receive buffer [senders, block, V]; also the summed magnitudes
    sum_w |o_w| * weight_w that bound the f32 rounding."""
    words = recv[:, PAD - 1].astype(np.int32)
    bits = (words[:, :tl] << 16) | (words[:, tl : 2 * tl] & 0xFFFF)
    lse = bits.astype(np.int32).view(np.float32)
    o = (recv[:, PAD:].astype(np.int32) << 16).view(np.float32)
    o = np.where(np.arange(tl)[None, :, None] < live, o, np.float32(0))
    finite = np.isfinite(lse)
    shift = np.where(finite, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        weights = np.where(finite, np.exp(lse - shift[None]), 0).astype(np.float32)
    denom = weights.sum(axis=0, dtype=np.float32)
    weights = weights / np.where(denom > 0, denom, 1).astype(np.float32)
    o = np.where(np.isnan(o), np.float32(0), o)
    w_hi = weights.astype(jnp.bfloat16).astype(np.float32)
    w_lo = weights - w_hi
    merged = (o * w_hi[..., None]).sum(axis=0, dtype=np.float32) + (
        o * w_lo[..., None]
    ).sum(axis=0, dtype=np.float32)
    magnitude = (np.abs(o) * weights[..., None]).sum(axis=0, dtype=np.float64)
    return merged, magnitude, (o * w_hi[..., None]).sum(axis=0, dtype=np.float32)


def _merge_incumbent(recv, live, tl):
    """The decode path's own unpack and merge on one core's receive buffer,
    f32 [tl, V]; None where cp_mla_attention does not import."""
    try:
        import torch

        from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp
    except ImportError:
        return None
    assert SEND_PAD_ROWS == PAD
    o, lse = cp._mask_unpack(jnp.asarray(recv), jnp.asarray([1 + live], jnp.int32), tl)
    o = torch.from_numpy(np.asarray(o).view(np.int16)).view(torch.bfloat16)
    lse = torch.from_numpy(np.asarray(lse))
    merged = cp._merge_received(o[:, :, None], lse[:, :, None])
    return merged[:, 0].numpy()


def _run(mode, mesh, blocks, live, ranks=None):
    """Plain and merge-mode calls in one program; per core the plain
    receive buffer [senders, block, V] and the merge [tl, V] bf16."""
    n, _, block, _ = blocks.shape
    tl = block - PAD
    spec = P(AXIS)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.asarray(blocks.reshape(n * n * block, V)), sharding)
    r = jax.device_put(
        jnp.asarray(range(n) if ranks is None else ranks, jnp.int32), sharding
    )
    count = jnp.asarray([live], jnp.int32)

    def body(x, r, count):
        common = dict(num_cores=n, axis=AXIS, row0=ROW0, rank=r)
        plain = a2a.small_all_to_all(x, **common)
        merged = a2a.small_all_to_all(x, **common, merge_live=count, merge_row=PAD)
        return plain, merged

    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, P()),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )
    with pytest.MonkeyPatch.context() as mp:
        if mode == "interpret":
            interpret = functools.partial(
                pl.pallas_call, interpret=pltpu.InterpretParams()
            )
            mp.setattr(a2a.pl, "pallas_call", interpret)
        plain, merged = fn(x, r, count)
    plain = np.asarray(plain).reshape(n, n, block, V)
    merged = np.asarray(merged).reshape(n, tl, V)
    return plain, merged


def _within(got, want, magnitude, live):
    """Live rows of ``got`` (bf16) against an f32 merge: half a bf16 step at
    the result plus 16 f32 roundings of the summed magnitudes (two summation
    orders, the exp and the weight division). Returns the elementwise verdict
    and the share of rows equal to ``want``'s bf16 rounding."""
    got = got[:live].astype(np.float64)
    want = want[:live].astype(np.float64)
    order = 16 * 2.0**-24 * magnitude[:live]
    tolerance = bf16_ulp(np.abs(want) + order) / 2 + order
    rounded = want.astype(np.float32).astype(jnp.bfloat16).astype(np.float64)
    exact = (got == rounded).mean() if live else 1.0
    return np.abs(got - want) <= tolerance, exact


CASES = [(16, 64), (16, 48), (16, 36), (16, 0), (32, 128), (32, 80), (32, 100)]


@pytest.mark.multichip
@pytest.mark.parametrize("tokens,live", CASES)
def test_merge_matches_the_incumbent_merge(merge_mesh, tokens, live):
    mode, mesh = merge_mesh
    tl = tokens * LOCAL
    blocks = _send_blocks(tokens, live, seed=tokens + live)
    plain, merged = _run(mode, mesh, blocks, live)
    np.testing.assert_array_equal(
        plain[:, :, ROW0:],
        a2a.reference(blocks, ROW0).reshape(plain.shape)[:, :, ROW0:],
    )
    for core in range(N):
        want, magnitude, _ = _merge_numpy(plain[core], live, tl)
        incumbent = _merge_incumbent(plain[core], live, tl)
        got = merged[core].astype(np.float32)
        assert not np.isnan(got).any()
        np.testing.assert_array_equal(merged[core][live:].view(np.int16), 0)
        for name, ref in (("numpy", want), ("incumbent", incumbent)):
            if ref is None:
                continue
            ok, exact = _within(got, ref, magnitude, live)
            assert ok.all(), (name, core, np.argwhere(~ok)[:5])
            assert exact >= 0.99, (name, core, exact)


@pytest.mark.multichip
def test_the_comparison_sees_a_dropped_lo_weight(merge_mesh):
    """Positive control for the tolerance: the merge without ``w_lo`` (a
    2^-9 relative error per term) must fail the same comparison."""
    mode, mesh = merge_mesh
    tokens, live = 16, 64
    blocks = _send_blocks(tokens, live, seed=5)
    plain, merged = _run(mode, mesh, blocks, live)
    want, magnitude, hi_only = _merge_numpy(plain[0], live, tokens * LOCAL)
    ok, exact = _within(merged[0].astype(np.float32), want, magnitude, live)
    assert ok.all() and exact >= 0.99
    ok, exact = _within(merged[0].astype(np.float32), hi_only, magnitude, live)
    assert not ok.all() or exact < 0.99


@pytest.mark.multichip
def test_a_wrong_rank_order_fails(merge_mesh):
    """Positive control for the exchange: cores 0 and 1 told each other's
    index send block 0 / 1 to themselves instead of to each other, so core 0
    merges its own block for core 1 and misses core 1's, and the comparison
    has to see it. (Cores 2-7 only get two senders' slots swapped, which the
    merge does not depend on.)"""
    mode, mesh = merge_mesh
    tokens, live = 16, 64
    blocks = _send_blocks(tokens, live, seed=7)
    _, merged = _run(mode, mesh, blocks, live, ranks=[1, 0, *range(2, N)])
    recv = a2a.reference(blocks, ROW0).reshape(N, N, PAD + tokens * LOCAL, V)
    want, magnitude, _ = _merge_numpy(recv[0], live, tokens * LOCAL)
    ok, _ = _within(merged[0].astype(np.float32), want, magnitude, live)
    assert not ok.all()
    want, magnitude, _ = _merge_numpy(recv[2], live, tokens * LOCAL)
    ok, _ = _within(merged[2].astype(np.float32), want, magnitude, live)
    assert ok.all()


def _run_live(mode, mesh, blocks, live):
    """Merge mode with and without ``live_rows`` (as
    TPU_DCP_MLA_SEND_SMALL_A2A_LIVE_ROWS passes it) in one program; per core
    both merges [tl, V] bf16."""
    n, _, block, _ = blocks.shape
    tl = block - PAD
    spec = P(AXIS)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.asarray(blocks.reshape(n * n * block, V)), sharding)
    r = jax.device_put(jnp.asarray(range(n), jnp.int32), sharding)
    count = jnp.asarray([live], jnp.int32)

    def body(x, r, count):
        common = dict(
            num_cores=n, axis=AXIS, row0=ROW0, rank=r, merge_live=count, merge_row=PAD
        )
        whole = a2a.small_all_to_all(x, **common)
        sized = a2a.small_all_to_all(x, **common, live_rows=count, live_row0=PAD)
        return whole, sized

    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, P()),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )
    with pytest.MonkeyPatch.context() as mp:
        if mode == "interpret":
            interpret = functools.partial(
                pl.pallas_call, interpret=pltpu.InterpretParams()
            )
            mp.setattr(a2a.pl, "pallas_call", interpret)
        whole, sized = fn(x, r, count)
    return (
        np.asarray(whole).reshape(n, tl, V),
        np.asarray(sized).reshape(n, tl, V),
    )


@pytest.mark.multichip
@pytest.mark.parametrize(
    "tokens,live",
    sorted(
        set(CASES + [(16, 4), (32, 124)])
        | {(t, p) for t in (16, 32) for p in range(0, t * LOCAL + 1, 8)}
    ),
)
def test_live_rows_merge_equals_the_whole_block_merge(merge_mesh, tokens, live):
    """With ``live_rows`` the copies end at the tile row holding the last live
    partial row and land in the VMEM receive scratch at the same rows; the
    merge reads no row past it, so it is bit for bit the whole-block merge.
    Live counts at every multiple of 8 take every copy size, one tile to the
    whole block."""
    mode, mesh = merge_mesh
    blocks = _send_blocks(tokens, live, seed=3 * tokens + live)
    whole, sized = _run_live(mode, mesh, blocks, live)
    np.testing.assert_array_equal(sized.view(np.int16), whole.view(np.int16))


def test_rejects_shapes_without_a_merge_layout():
    x = jnp.zeros((N * (PAD + 64), V), jnp.int16)
    live = jnp.zeros((1,), jnp.int32)
    common = dict(num_cores=N, axis=AXIS, row0=ROW0, merge_live=live)
    with pytest.raises(ValueError):
        a2a.small_all_to_all(x.astype(jnp.float32), **common, merge_row=PAD)
    with pytest.raises(ValueError):
        a2a.small_all_to_all(x, **common, merge_row=PAD + 4)
    with pytest.raises(ValueError):
        a2a.small_all_to_all(
            jnp.zeros((N * (PAD + 256), V), jnp.int16), **common, merge_row=PAD
        )
