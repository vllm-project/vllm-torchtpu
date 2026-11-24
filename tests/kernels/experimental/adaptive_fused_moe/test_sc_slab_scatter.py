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
"""sc_slab_scatter: CPU emulation (race-checked) against the jnp scatter, the
slab tables the layer builds from it, Mosaic lowering and a deviceless
tpu7x:2x2x1 compile."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import sc_slab_scatter as scs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.host import (
    HIDDEN_LANE_BLOCK,
    act_scale_slab_rows,
    gather_clamp_bounds,
)

LANES = scs.LANES
GARBAGE = -77777


class _NumpyOps:
    xp = np

    def __init__(self):
        self.trace = None

    @staticmethod
    def full(x):
        return np.full((LANES,), x, np.int32)

    @staticmethod
    def load(ref, at):
        at = int(at)
        assert at >= 0 and at + LANES <= ref.size
        return ref[at : at + LANES].copy()

    @staticmethod
    def store(ref, at, x):
        at = int(at)
        assert at >= 0 and at + LANES <= ref.size
        ref[at : at + LANES] = x

    def store_scatter(self, ref, index, x, mask):
        idx = np.asarray(index, np.int32)
        m = np.asarray(mask, bool)
        live = idx[m]
        assert (live >= 0).all() and (live < ref.size).all(), live
        assert len(set(live.tolist())) == len(live), "intra-lane scatter collision"
        if self.trace is not None:
            for a in live.tolist():
                key = (id(ref), a)
                assert key not in self.trace, f"cross-iteration race on {key}"
                self.trace.add(key)
        ref[live] = np.asarray(x, np.int32)[m]

    def loop(self, lower, upper, *, unroll=1):
        def run(fn):
            self.trace = set()
            for i in range(int(lower), int(upper)):
                fn(i)
            self.trace = None

        return run


def _emulate(slab_row, tok_ids, scale_bits, *, shard_stride, scale_out_len, workers=32):
    n = len(slab_row)
    has_scale = scale_bits is not None
    out_len = max(shard_stride, scale_out_len if has_scale else 0)
    slice_len = scs._align_up(out_len, workers * LANES) // workers
    padded_out = slice_len * workers
    chunk = scs.CHUNK if n >= scs.CHUNK else scs._align_up(max(n, LANES), LANES)
    n_pad = scs._align_up(n, chunk)
    pad = n_pad - n
    r_pad = np.pad(slab_row, (0, pad), constant_values=shard_stride).astype(np.int32)
    t_pad = np.pad(tok_ids, (0, pad), constant_values=0).astype(np.int32)
    s_pad = (
        np.pad(scale_bits, (0, pad), constant_values=0).astype(np.int32)
        if has_scale
        else None
    )

    out_tok = np.full((padded_out,), GARBAGE, np.int32)
    out_scl = np.full((padded_out,), GARBAGE, np.int32) if has_scale else None
    ops = _NumpyOps()

    for w in range(workers):
        lo = w * slice_len
        valid_hi = min(lo + slice_len, shard_stride)
        tok_v = np.full((slice_len,), GARBAGE, np.int32)
        scl_v = np.full((slice_len,), GARBAGE, np.int32) if has_scale else None
        scs.zero_buf(ops, tok_v, slice_len // LANES)
        if has_scale:
            scs.zero_buf(ops, scl_v, slice_len // LANES)
        for c in range(0, n_pad, chunk):
            scs.scatter_chunk(
                ops,
                r_pad[c : c + chunk],
                t_pad[c : c + chunk],
                s_pad[c : c + chunk] if has_scale else None,
                tok_v,
                scl_v,
                lo=lo,
                valid_hi=valid_hi,
                num_vec=chunk // LANES,
                has_scale=has_scale,
            )
        out_tok[lo : lo + slice_len] = tok_v
        if has_scale:
            out_scl[lo : lo + slice_len] = scl_v

    return out_tok[:shard_stride], (out_scl[:scale_out_len] if has_scale else None)


@pytest.mark.parametrize(
    "T,K,ep", [(16, 10, 8), (64, 10, 8), (512, 10, 8), (4096, 10, 8)]
)
@pytest.mark.parametrize("has_scale", [True, False])
def test_emulation_matches_jax_reference_bit_exact(T, K, ep, has_scale):
    rng = np.random.default_rng(T * 100 + K * 10 + int(has_scale))
    n = T * K
    shard_stride = scs._align_up(n // ep + 512, 128)
    scale_out_len = scs._align_up(shard_stride, 128) + 128

    # Construct injective local slab rows for ~1/ep pairs and out-of-range for rest.
    is_local = rng.random(n) < (1.0 / ep)
    n_local = int(is_local.sum())
    local_slots = rng.choice(shard_stride, size=n_local, replace=False).astype(np.int32)
    slab_row = rng.integers(shard_stride, shard_stride + 5000, size=n, dtype=np.int32)
    slab_row[is_local] = local_slots

    tok_ids = np.broadcast_to(np.arange(T, dtype=np.int32)[:, None], (T, K)).reshape(-1)
    scale_bits = (
        np.repeat(rng.integers(1, 1 << 30, size=T, dtype=np.int32), K)
        if has_scale
        else None
    )

    got_tok, got_scl = _emulate(
        slab_row,
        tok_ids,
        scale_bits,
        shard_stride=shard_stride,
        scale_out_len=scale_out_len,
    )

    ref_tok = np.asarray(
        jnp.zeros((shard_stride,), jnp.int32).at[slab_row].add(tok_ids, mode="drop")
    )
    np.testing.assert_array_equal(got_tok, ref_tok)

    if has_scale:
        ref_scl = np.asarray(
            jnp.zeros((shard_stride,), jnp.int32)
            .at[slab_row]
            .add(scale_bits, mode="drop")
        )
        # Slab rows [0, shard_stride) are bit-identical; the tail is zeros.
        np.testing.assert_array_equal(got_scl[:shard_stride], ref_scl)
        assert not got_scl[shard_stride:].any()
    else:
        assert got_scl is None


@pytest.fixture
def tpu7x_abstract():
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType, use_abstract_mesh

    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("x",), (AxisType.Explicit,), abstract_device=device)
    with use_abstract_mesh(mesh):
        yield
    jax.clear_caches()


@pytest.mark.parametrize("T,has_scale", [(4096, True), (8192, True), (4096, False)])
def test_sc_slab_scatter_lowers_to_mosaic_tpu7x(tpu7x_abstract, T, has_scale):
    n = T * 10
    shard_stride = scs._align_up(n + 7 * 512 + 128, 128)
    scale_out_len = scs._align_up(shard_stride, 128) + 128
    spec = jax.ShapeDtypeStruct

    if has_scale:
        fn = jax.jit(
            functools.partial(
                scs.sc_fused_slab_scatter,
                shard_stride=shard_stride,
                scale_out_len=scale_out_len,
            )
        )
        lowered = fn.trace(
            spec((n,), jnp.int32),
            spec((n,), jnp.int32),
            spec((n,), jnp.int32),
        ).lower(lowering_platforms=("tpu",))
    else:
        fn = jax.jit(
            lambda r, t: scs.sc_fused_slab_scatter(
                r, t, None, shard_stride=shard_stride, scale_out_len=0
            )[0]
        )
        lowered = fn.trace(
            spec((n,), jnp.int32),
            spec((n,), jnp.int32),
        ).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


def _case(T, K, ep, seed):
    """Injective local slab rows for ~1/ep of the pairs, off-shard (>= stride)
    rows for the rest, as local_slab_rows hands them to the scatter."""
    rng = np.random.default_rng(seed)
    n = T * K
    shard_stride = scs._align_up(n // ep + 512, 128)
    is_local = rng.random(n) < (1.0 / ep)
    slab_row = rng.integers(shard_stride, shard_stride + 5000, size=n, dtype=np.int32)
    slab_row[is_local] = rng.choice(
        shard_stride, size=int(is_local.sum()), replace=False
    )
    scale_bits = rng.integers(1, 1 << 30, size=T, dtype=np.int32)
    return slab_row, scale_bits, shard_stride


@pytest.mark.parametrize("quantized", [True, False])
def test_slab_tables_match_the_tensorcore_path(monkeypatch, quantized):
    """sc_slab_tables gives the token_gather and scale_slab adaptive_fused_moe
    builds with jnp scatters, on every element the MoE kernel reads."""
    T, K, window_rows = 512, 10, 256
    slab_row, scale_bits, shard_stride = _case(T, K, 8, seed=7 + quantized)
    scale_rows = act_scale_slab_rows(shard_stride)

    def emulated(rows, tok, scl, *, shard_stride, scale_out_len):
        got_tok, got_scl = _emulate(
            np.asarray(rows),
            np.asarray(tok),
            None if scl is None else np.asarray(scl),
            shard_stride=shard_stride,
            scale_out_len=scale_out_len,
        )
        scl_out = None if got_scl is None else jnp.asarray(got_scl)
        return jnp.asarray(got_tok), scl_out

    monkeypatch.setattr(scs, "sc_fused_slab_scatter", emulated)
    token_gather, scale_slab = scs.sc_slab_tables(
        jnp.asarray(slab_row),
        jnp.asarray(scale_bits) if quantized else None,
        n_tokens=T,
        topk=K,
        shard_stride=shard_stride,
        window_rows=window_rows,
        scale_rows=scale_rows,
    )

    # The layer's TensorCore path (shard_token_gather + pad, scale scatter).
    tok_of_pair = np.repeat(np.arange(T, dtype=np.int32), K)
    lo, hi = gather_clamp_bounds(T)
    scattered = (
        jnp.zeros((shard_stride,), jnp.int32).at[slab_row].add(tok_of_pair, mode="drop")
    )
    ref_tok = jnp.clip(scattered, lo, hi)
    ref_tok = jnp.pad(ref_tok, (0, window_rows))
    np.testing.assert_array_equal(np.asarray(token_gather), np.asarray(ref_tok))
    if not quantized:
        assert scale_slab is None
        return
    ref_scale = (
        jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK,), jnp.int32)
        .at[slab_row]
        .add(np.repeat(scale_bits, K), mode="drop")
    )
    assert scale_slab.shape == (scale_rows, HIDDEN_LANE_BLOCK)
    assert scale_slab.dtype == jnp.float32
    got_bits = np.asarray(jax.lax.bitcast_convert_type(scale_slab, jnp.int32))
    got_bits = got_bits.reshape(-1)
    # The kernel reads the slab's shard_stride rows; the view's tail is unread.
    np.testing.assert_array_equal(
        got_bits[:shard_stride], np.asarray(ref_scale)[:shard_stride]
    )


@pytest.mark.parametrize("T,has_scale", [(4096, True), (8192, True), (4096, False)])
def test_sc_slab_scatter_compiles_for_tpu7x_2x2x1(T, has_scale):
    """Deviceless Mosaic compile on the served topology (one TensorCore's
    SparseCores), not just lowering."""
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = topologies.get_topology_desc(
        platform="tpu", topology_name="tpu7x:2x2x1"
    ).devices
    mesh = Mesh(np.array(devices[:1]), ("x",))
    rep = NamedSharding(mesh, PartitionSpec())
    n = T * 10
    shard_stride = scs._align_up(n + 7 * 512 + 128, 128)
    scale_out_len = scs._align_up(shard_stride, 128) + 128
    arg = jax.ShapeDtypeStruct((n,), jnp.int32, sharding=rep)

    def fn(rows, tok, *scl):
        return scs.sc_fused_slab_scatter(
            rows,
            tok,
            scl[0] if scl else None,
            shard_stride=shard_stride,
            scale_out_len=scale_out_len if scl else 0,
        )

    args = (arg, arg, arg) if has_scale else (arg, arg)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        compiled = jax.jit(fn).lower(*args).compile()
    assert compiled.as_text()
