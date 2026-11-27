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
"""TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS: the two-level small all-reduce
(interpret mode over 8 CPU devices), its layout table and its dispatch."""

import dataclasses
import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import tp_all_reduce as flat  # noqa: E402
from vllm_torchtpu.kernels.collectives import tp_all_reduce_hier as hier  # noqa: E402

N = 8


@dataclasses.dataclass(frozen=True)
class _Dev:
    id: int
    coords: tuple
    core_on_chip: int


def _devices(order):
    """Mesh entries for mesh order `order` of physical ids 2 * chip + core,
    chips x-major over the 2x2 grid."""
    return [_Dev(i, ((i // 2) % 2, (i // 2) // 2, 0), i % 2) for i in order]


ORDERS = [tuple(range(N)), (3, 6, 0, 5, 1, 7, 2, 4)]


@pytest.mark.parametrize("order", ORDERS)
def test_layout_table(order):
    devs = _devices(order)
    tab = hier.layout_table(devs)
    for i, d in enumerate(devs):
        s = devs[tab[hier.SIB, i]]
        assert s.coords == d.coords and s.core_on_chip != d.core_on_chip
        assert tab[hier.CORE, i] == d.core_on_chip
        assert tab[hier.AT + tab[hier.POS, i], i] == i
        peers = [devs[tab[hier.AT + q, i]] for q in range(4)]
        assert len({p.coords for p in peers}) == 4
        assert all(p.core_on_chip == d.core_on_chip for p in peers)
    with pytest.raises(ValueError):
        hier.layout_table(devs[:4])


@pytest.fixture
def interpret(monkeypatch):
    if len(jax.devices()) < N:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    for m in (hier, flat):
        monkeypatch.setattr(
            m.pl,
            "pallas_call",
            functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
        )
    hier._call.cache_clear()
    flat._call.cache_clear()
    yield
    hier._call.cache_clear()
    flat._call.cache_clear()


def _run(parts, order, direct_rows, mode="ag"):
    mesh = Mesh(np.array(jax.devices()[:N]), ("x",))
    tab = hier.layout_table(_devices(order))
    ranks = np.arange(N, dtype=np.int32).reshape(N, 1)
    rows = parts.shape[1]
    if mode == "flat":
        fn = lambda x, r: flat.all_reduce(x, r, N, direct_rows)  # noqa: E731
    else:
        fn = lambda x, r: hier.all_reduce(x, r, tab, direct_rows, mode)  # noqa: E731
    f = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=P("x"),
        check_vma=False,
    )
    x = jnp.asarray(parts.reshape(N * rows, -1), jnp.bfloat16)
    out = jax.jit(f)(x, ranks)
    return np.asarray(out.astype(jnp.float32)).reshape(N, rows, -1)


@pytest.mark.parametrize("mode", hier.MODES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rows,direct_rows", [(32, 0), (32, 8), (8, 8), (24, 0)])
def test_integer_sums_are_exact(interpret, mode, order, rows, direct_rows):
    """Integer partials sum exactly in any order: every chunk must land in
    its place on every core."""
    hidden = 2048
    d = np.arange(N)[:, None, None]
    r = np.arange(rows)[None, :, None]
    c = np.arange(hidden)[None, None, :] // 128
    parts = (((d + 1) * 3 + r * 5 + c * 7) % 11 - 5).astype(np.float32)
    out = _run(parts, order, direct_rows, mode)
    for i in range(N):
        np.testing.assert_array_equal(out[i], parts.sum(0))


def _random_parts(rows, hidden, seed=7):
    rng = np.random.default_rng(seed)
    return np.asarray(
        jnp.asarray(rng.standard_normal((N, rows, hidden)), jnp.bfloat16).astype(jnp.float32)
    )


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rows,direct_rows", [(32, 8), (24, 0)])
def test_ag_mode_is_bitwise_flat(interpret, order, rows, direct_rows):
    """Mode "ag" keeps the flat reduce-scatter: the same bits as
    tp_all_reduce, with chunks owned by mesh index in any mesh order."""
    parts = _random_parts(rows, 2048, seed=rows)
    got = _run(parts, order, direct_rows, "ag")
    want = _run(parts, order, direct_rows, "flat")
    np.testing.assert_array_equal(got, want)


def test_full_mode_agrees_and_rounds_like_a_tree(interpret):
    """Mode "full": all cores get the same bits, and the error against the
    exact sum is of bf16-rounding size (two roundings)."""
    rows, hidden = 32, 2048
    parts = _random_parts(rows, hidden)
    out = _run(parts, ORDERS[1], 8, "full")
    for i in range(1, N):
        np.testing.assert_array_equal(out[i], out[0])
    exact = parts.astype(np.float64).sum(0)
    rel = np.linalg.norm(out[0] - exact) / np.linalg.norm(exact)
    one_rounding = np.asarray(jnp.asarray(exact, jnp.bfloat16).astype(jnp.float32))
    rel1 = np.linalg.norm(one_rounding - exact) / np.linalg.norm(exact)
    assert rel < 2.5 * rel1, (rel, rel1)


def test_supported():
    assert hier.supported(32, 6144)
    assert hier.supported(16, 12288)
    assert not hier.supported(32, 6144 + 128)
    assert flat.direct_ok(32, 6144, N, 8)


@pytest.mark.parametrize(
    "rows,hier_rows,direct_rows,want",
    [
        (32, 64, 8, "hier"),
        (32, 64, 0, "hier"),  # [16, w] view
        (48, 64, 8, "hier"),  # 589824 B, at the byte cap
        (48, 40, 8, "pallas"),  # above the hier cap
        (32, 0, 8, "pallas"),  # flag off
        (512, 1024, 8, "psum"),  # above the byte cap
        (1024, 1024, 8, "qar"),  # qar first
    ],
)
def test_size_routing(monkeypatch, rows, hier_rows, direct_rows, want):
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    calls = []
    monkeypatch.setattr(par.kernel, "all_reduce", lambda x, r, n, d=0: calls.append("pallas") or x)
    monkeypatch.setattr(par.hier_ar, "all_reduce",
                        lambda x, r, t, d=0, m="ag": calls.append("hier") or x)
    monkeypatch.setattr(par.qar, "all_reduce_v2", lambda x, r, t: calls.append("qar") or x)
    monkeypatch.setattr(par.lax, "psum", lambda x, a: calls.append("psum") or x)
    tab = hier.layout_table(_devices(range(N)))
    body = par._body(N, 589824, "x", direct_rows=direct_rows, qar_min_rows=1024,
                     qar_tab=object(), hier_rows=hier_rows, hier_tab=tab)
    body(np.zeros((rows, 6144), np.float32), None)
    assert calls == [want]


def test_flag_default_off(monkeypatch):
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    monkeypatch.delenv("TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS", raising=False)
    assert par.envs.TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS == 0
    assert par._hier_table(None, 8, 6144) is None
