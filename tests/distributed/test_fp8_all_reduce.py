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
"""The fp8-wire TP all-reduce (kernels/collectives/fp8_all_reduce.py,
TPU_TP_FP8_ALL_REDUCE): quantization and payload layout, gating, numerics
against the f32 sum on 8 simulated CPU devices (heavy-tailed partials with
outlier rows and channels; SNR per mode and scale granularity next to the
bf16 psum), and its routing inside the Pallas all-reduce op body, where the
switch off leaves the program unchanged."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import fp8_all_reduce as f8

N = 8
H = 6144


def _mesh(n=N):
    devs = jax.devices("cpu")
    if len(devs) < n:
        pytest.skip(f"needs {n} CPU devices (XLA_FLAGS set too late)")
    return Mesh(np.asarray(devs[:n]), axis_names=("d",))


def _partials(t, h, seed=0, n=N):
    """[n, t, h] per-rank partials: Student-t(4) values, log-normal row
    norms per rank, 8 outlier channels (x50), 1/64 of the rows x30, and a
    few zero rows."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(4, (n, t, h)) * np.exp(rng.normal(0.0, 1.0, (n, t, 1)))
    x[:, :, rng.choice(h, 8, replace=False)] *= 50
    x[:, rng.choice(t, max(t // 64, 1), replace=False), :] *= 30
    x[:, :3, :] = 0.0
    return x.astype(np.float32)


def _run(body, x, mesh):
    f = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )
    return np.asarray(f(x).astype(jnp.float32))


def _snr_db(ref, got):
    return float(10 * np.log10((ref**2).sum() / max(((got - ref) ** 2).sum(), 1e-30)))


# ---------------------------------------------------------------------------
# quantization, payload, gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", f8.GRANULARITIES)
def test_roundtrip_exact_on_e4m3_values(granularity):
    row = np.zeros(256, np.float32)
    row[:6] = [448.0, -2.0, 0.5, 0.25, -448.0, 1.0]
    row[128:131] = [448.0, 3.0, -0.75]
    x = jnp.asarray(np.stack([row, row * 0.5, np.zeros(256, np.float32)]), jnp.bfloat16)
    q, s = f8.quantize(x, granularity)
    assert q.dtype == f8.FP8
    assert s.shape == (3, f8.num_groups(256, granularity))
    np.testing.assert_array_equal(
        np.asarray(f8.dequantize(q, s)), np.asarray(x, np.float32)
    )
    p = f8.pack(q, s, granularity)
    assert p.dtype == jnp.int8
    assert p.shape == (3, f8.wire_bytes_per_row(256, granularity))
    q2, s2 = f8.unpack(p, 256, granularity)
    np.testing.assert_array_equal(np.asarray(q2, np.float32), np.asarray(q, np.float32))
    np.testing.assert_array_equal(np.asarray(s2), np.asarray(s))


def test_scale_bytes():
    assert f8.scale_bytes(H, "row") == 128
    assert f8.scale_bytes(H, "g128") == 256  # 48 f32 scales
    assert f8.wire_bytes_per_row(H, "g128") == H + 256
    with pytest.raises(ValueError):
        f8.num_groups(H + 64, "g128")
    with pytest.raises(ValueError):
        f8.num_groups(H, "g64")


@pytest.mark.parametrize(
    "rows,min_tokens,gran,want",
    [
        (8192, 2048, "g128", True),
        (2048, 2048, "row", True),
        (2040, 2048, "g128", False),  # below the minimum
        (2052, 2048, "g128", False),  # not a multiple of n
        (8192, 0, "g128", False),  # 0 = never
        (48, 2048, "row", False),  # decode
    ],
)
def test_applies(rows, min_tokens, gran, want):
    assert f8.applies(rows, H, N, min_tokens, gran) is want


def test_applies_needs_128_columns_for_g128():
    assert not f8.applies(4096, H + 64, N, 2048, "g128")
    assert f8.applies(4096, H + 64, N, 2048, "row")
    assert not f8.applies(4096, H, 1, 2048, "row")


def test_nan_propagates_and_inf_saturates():
    x = jnp.asarray([[1.0, jnp.nan, 2.0, 4.0], [jnp.inf, 1.0, 0.5, 0.0]], jnp.float32)
    q, s = f8.quantize(x, "row")
    back = np.asarray(f8.dequantize(q, s))
    assert np.isnan(back[0, 1]) and np.isfinite(s).all()
    np.testing.assert_allclose(back[0, [0, 2, 3]], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(back[1, 1:], [1.0, 0.5, 0.0])
    assert back[1, 0] == 1.0  # inf is left out of the amax; it saturates at it


# ---------------------------------------------------------------------------
# collectives on 8 simulated devices
# ---------------------------------------------------------------------------

# Measured on _partials(256, 6144) (bf16 psum: ~55 dB): a2a 30.6 / 34.8 dB,
# rs 32.8 / 37.6, a2a16 33.4 / 37.9 (row / g128). Floors leave ~1.5 dB.
_SNR_FLOOR = {
    ("a2a", "row"): 29.0,
    ("a2a", "g128"): 33.0,
    ("rs", "row"): 31.0,
    ("rs", "g128"): 36.0,
    ("a2a16", "row"): 31.5,
    ("a2a16", "g128"): 36.0,
}


@pytest.fixture(scope="module")
def case():
    t = 256
    parts = _partials(t, H)
    x = jnp.asarray(parts.reshape(N * t, H), jnp.bfloat16)
    ref = np.asarray(x.astype(jnp.float32)).reshape(N, t, H).sum(0)
    return t, x, ref


@pytest.mark.parametrize("mode", f8.MODES)
@pytest.mark.parametrize("granularity", f8.GRANULARITIES)
def test_all_reduce_snr_and_same_bits_everywhere(case, mode, granularity):
    t, x, ref = case
    mesh = _mesh()
    out = _run(lambda v: f8.all_reduce(v, "d", N, mode, granularity), x, mesh)
    out = out.reshape(N, t, H)
    for r in range(1, N):
        np.testing.assert_array_equal(out[r], out[0])
    snr = _snr_db(ref, out[0])
    assert snr >= _SNR_FLOOR[(mode, granularity)], snr
    # Row placement and outlier rows: every row's own relative error stays
    # small (a row block in the wrong place would be ~100%).
    nz = (ref**2).mean(1) > 0
    rel = np.sqrt(((out[0] - ref) ** 2).mean(1)[nz] / (ref**2).mean(1)[nz])
    assert rel.max() < 0.08, rel.max()
    np.testing.assert_array_equal(out[0][~nz], 0.0)


def test_psum_reference_is_much_tighter(case):
    t, x, ref = case
    out = _run(lambda v: jax.lax.psum(v, "d"), x, _mesh()).reshape(N, t, H)
    assert _snr_db(ref, out[0]) > 50


def test_two_ranks_and_bf16_rounding_only_differences():
    """n=2, values exact in e4m3 with a per-row scale of 1 (amax 448)."""
    mesh = _mesh(2)
    t, h = 8, 256
    a = np.zeros((2, t, h), np.float32)
    a[:, :, 0] = 448.0
    a[0, :, 1:5] = [1.0, 2.0, -4.0, 0.5]
    a[1, :, 1:5] = [3.0, -2.0, 8.0, 0.5]
    x = jnp.asarray(a.reshape(2 * t, h), jnp.bfloat16)
    out = _run(lambda v: f8.all_reduce(v, "d", 2, "a2a", "row"), x, mesh).reshape(
        2, t, h
    )
    want = a.sum(0)
    np.testing.assert_array_equal(out[0], want)
    np.testing.assert_array_equal(out[1], want)


# ---------------------------------------------------------------------------
# routing in the Pallas all-reduce op body
# ---------------------------------------------------------------------------


def _jaxpr(body, rows, mesh):
    """The traced program (the Pallas kernel branch does not lower on CPU)."""
    f = jax.shard_map(
        body, mesh=mesh, in_specs=(P("d"), P("d")), out_specs=P("d"), check_vma=False
    )
    x = jnp.zeros((N * rows, H), jnp.bfloat16)
    rank = jnp.zeros((N, 1), jnp.int32)
    return str(jax.make_jaxpr(f)(x, rank))


@pytest.fixture
def par():
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.distributed import pallas_all_reduce

    return pallas_all_reduce


def test_body_off_is_unchanged(par):
    mesh = _mesh()
    for rows in (48, 4096):
        base = _jaxpr(par._body(N, 589824, "d"), rows, mesh)
        assert _jaxpr(par._body(N, 589824, "d", None), rows, mesh) == base
    assert "all_to_all" not in _jaxpr(par._body(N, 589824, "d"), 4096, mesh)


def test_body_routes_large_rows_only(par):
    mesh = _mesh()
    fp8 = par.Fp8Config(min_tokens=2048, mode="a2a", granularity="g128")
    off = par._body(N, 589824, "d")
    on = par._body(N, 589824, "d", fp8)
    # decode (kernel) and below-minimum (psum) sizes: same program as off
    for rows in (48, 1024, 2040):
        assert _jaxpr(on, rows, mesh) == _jaxpr(off, rows, mesh)
    text = _jaxpr(on, 2048, mesh)
    assert "all_to_all" in text and "all_gather" in text
    assert "psum" not in text
    assert "psum" in _jaxpr(off, 2048, mesh)


@pytest.mark.parametrize("mode", f8.MODES)
def test_body_fp8_matches_module(par, mode):
    mesh = _mesh()
    t = 256
    fp8 = par.Fp8Config(min_tokens=t, mode=mode, granularity="row")
    x = jnp.asarray(_partials(t, H, seed=3).reshape(N * t, H), jnp.bfloat16)
    rank = jnp.arange(N, dtype=jnp.int32).reshape(N, 1)
    f = jax.jit(
        jax.shard_map(
            par._body(N, 0, "d", fp8),
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    got = np.asarray(f(x, rank).astype(jnp.float32))
    want = _run(lambda v: f8.all_reduce(v, "d", N, mode, "row"), x, mesh)
    np.testing.assert_array_equal(got, want)


def test_fp8_config_from_env(par, monkeypatch):
    for k in (
        "TPU_TP_FP8_ALL_REDUCE",
        "TPU_TP_FP8_ALL_REDUCE_MIN_TOKENS",
        "TPU_TP_FP8_ALL_REDUCE_MODE",
        "TPU_TP_FP8_ALL_REDUCE_SCALE",
    ):
        monkeypatch.delenv(k, raising=False)
    assert par.fp8_config_from_env() is None
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE", "1")
    assert par.fp8_config_from_env() == par.Fp8Config(2048, "a2a", "g128")
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_MIN_TOKENS", "4096")
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_MODE", "rs")
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_SCALE", "row")
    assert par.fp8_config_from_env() == par.Fp8Config(4096, "rs", "row")
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_MIN_TOKENS", "0")
    assert par.fp8_config_from_env() is None
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_MIN_TOKENS", "2048")
    monkeypatch.setenv("TPU_TP_FP8_ALL_REDUCE_MODE", "bogus")
    with pytest.raises(ValueError):
        par.fp8_config_from_env()


def test_fp8_selftest_tolerance_catches_misplaced_blocks(par):
    """The arm-time check accepts the fp8 sum of its inputs and rejects the
    same sum with two row blocks swapped."""
    n, rows, h = N, 256, 512
    x = par._selftest_inputs(n, rows, h)
    xb = jnp.asarray(x.reshape(n * rows, h), jnp.bfloat16)
    got = _run(lambda v: f8.all_reduce(v, "d", n, "a2a", "g128"), xb, _mesh())
    got = got.reshape(n, rows, h)[0]
    assert par.fp8_selftest_ok(x, got)
    blk = rows // n
    swapped = got.copy()
    swapped[:blk], swapped[blk : 2 * blk] = got[blk : 2 * blk], got[:blk]
    assert not par.fp8_selftest_ok(x, swapped)
