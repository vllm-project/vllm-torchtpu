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
"""VLLM_TPU_MK_IO_LEAN: every lever's operand prep gives the bytes of the code it replaces.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_inputs as wsi  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_io_lean import (  # noqa: E402
    LEVERS,
    io_lean_levers,
    ring_columns,
)
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import (  # noqa: E402
    scatter_gdn_ckpts_fast,
)


def _bytes(x):
    return np.asarray(x).view(np.uint8)


def test_lever_parsing():
    assert io_lean_levers("") == frozenset()
    assert io_lean_levers("0") == frozenset()
    assert io_lean_levers("1") == frozenset(LEVERS)
    assert io_lean_levers("all") == frozenset(LEVERS)
    assert io_lean_levers("ring, conv") == {"ring", "conv"}
    with pytest.raises(AssertionError):
        io_lean_levers("rign")


def test_flag_is_registered_and_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_IO_LEAN", raising=False)
    assert "VLLM_TPU_MK_IO_LEAN" in envs.environment_variables  # compile-cache key
    assert envs.environment_variables["VLLM_TPU_MK_IO_LEAN"]() == ""
    assert io_lean_levers() == frozenset()


# --- ring ---------------------------------------------------------------------------------

def test_ring_columns_are_column_zero_of_each_flat_table():
    rng = np.random.default_rng(0)
    n, W, L = 16, 32, 3
    tabs = []
    for _ in range(L):
        tabs += [jnp.asarray(rng.integers(-1, 2**31 - 1, size=n * W), jnp.int32),
                 jnp.asarray(rng.integers(0, 9, size=n * 4), jnp.int32),
                 jnp.asarray(rng.integers(0, 9, size=n * 5), jnp.int32)]
    got = jax.jit(lambda *t: ring_columns(t, n))(*tabs)
    for o in range(L):
        want = np.asarray(tabs[3 * o]).reshape(n, W)[:, 0]
        np.testing.assert_array_equal(np.asarray(got[3 * o]), want)
        # The readers' own expression gives the same column on the replaced table.
        np.testing.assert_array_equal(np.asarray(got[3 * o]).reshape(n, -1)[:, 0], want)
        for k in (1, 2):  # other tables untouched
            np.testing.assert_array_equal(np.asarray(got[3 * o + k]), np.asarray(tabs[3 * o + k]))


# --- qsa_step_inputs: ring + cktab + rope ---------------------------------------------------

def _inputs_case(rng, n_req, L, n_cols_comp, big_ids):
    hi = 2**26 - 1 if big_ids else 4000  # 2**26 * 32 pages = 2**31: the int32 bound
    tables = []
    for _ in range(L):
        comp = rng.integers(0, hi, size=(n_req, n_cols_comp)).astype(np.int32)
        comp[0, -1] = -1  # an unallocated column
        comp[-1, 0] = hi  # the largest id
        tables.append((
            jnp.asarray(rng.integers(1, 4000, size=n_req * 32), jnp.int32),
            jnp.asarray(rng.integers(1, 4000, size=n_req * 40), jnp.int32),
            jnp.asarray(comp.reshape(-1)),
        ))
    return tables


@pytest.mark.parametrize("mp", [0, 1, 2, 4])
@pytest.mark.parametrize("layer_major", [False, True])
@pytest.mark.parametrize("big_ids", [False, True])
def test_qsa_step_inputs_lean_is_bitwise(mp, layer_major, big_ids, monkeypatch):
    """All outputs of `qsa_step_inputs` (batched inputs, served 512-token blocks), lean vs
    stock, with each layer's ring table replaced by `ring_columns` in the lean arm, for
    every manager-page width, both table forms, an idle and a fresh request and block ids up
    to the int32 bound of the per-page formula."""
    def fake_read(pool, ids, n, m):  # rows that name the page they were read from
        rows = ids.astype(jnp.float32)[:, None] + jnp.arange(n, dtype=jnp.float32)[None] / n
        return rows.astype(jnp.bfloat16), None

    monkeypatch.setattr(wsi, "read_ring_and_tags", fake_read)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", True, raising=False)
    rng = np.random.default_rng(1 + mp)
    pool = jnp.zeros((64, 512, 1, 4, 256), jnp.float8_e4m3fn)
    n_req, nreq, T, max_groups, L = 6, 4, 4, 8192, 5
    tables = _inputs_case(rng, n_req, L, 20, big_ids)
    slots = [(o % 2 if o % 3 else -1, o % 2) for o in range(L)]
    starts = jnp.asarray([0, 4, 4, 8, 12, 16, 16], jnp.int32)  # request 1 idle
    context = jnp.asarray([3000, 100, 0, 77_777, 200_001, 9], jnp.int32)  # 2 fresh
    trig = jnp.asarray(rng.standard_normal((262_144, 128)), jnp.bfloat16)
    positions = jnp.asarray(rng.integers(0, 262_000, size=nreq * T), jnp.int32)
    kw = dict(M=nreq * T, T=T, pages_per_seq=64, max_groups=max_groups,
              layer_major_tables=layer_major, manager_pages=mp)

    def run(lean: str, tabs):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", lean, raising=False)
        return jax.jit(lambda tabs: wsi.qsa_step_inputs(
            pool, tabs, slots, trig, positions, starts, context, **kw))(tabs)

    want = run("", tables)
    flat = [x for t in tables for x in t]
    lean_tabs = ring_columns(flat, n_req)
    lean_tabs = [tuple(lean_tabs[i:i + 3]) for i in range(0, len(lean_tabs), 3)]
    got = run("ring,cktab,rope", lean_tabs)
    assert want.keys() == got.keys()
    for k in want:
        a, b = np.asarray(want[k]), np.asarray(got[k])
        assert a.shape == b.shape and a.dtype == b.dtype, k
        np.testing.assert_array_equal(_bytes(a), _bytes(b), err_msg=k)


def test_cktab_negative_control(monkeypatch):
    """A wrong slot term in the direct table (cslot * lp instead of cslot * lp // mp) must
    fail the same comparison."""
    real = wsi.manager_page_table_direct

    def wrong(comp, cslots, mp, lp, physical, n_ck):
        return real(comp, [c * mp for c in cslots], mp, lp, physical, n_ck)

    monkeypatch.setattr(wsi, "manager_page_table_direct", wrong)
    with pytest.raises(AssertionError):
        test_qsa_step_inputs_lean_is_bitwise(4, True, False, monkeypatch)


# --- conv: the scatter reads windows from the history -------------------------------------

NL, NS, T = 2, 3, 4
NMB = 1 + NL * NS * T


def _scatter_case(seed, kb, pad):
    shape = (NMB * (512 // kb), kb, 1, 4, 256)
    layout = gs.Qwen38GdnPoolLayout(pool_shape=shape, pool_dtype="float8_e4m3fn")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(
        jnp.bfloat16)
    wb = 1 + np.arange(NL * NS * T, dtype=np.int32).reshape(NL, NS, T)
    if pad:
        wb[:, -1] = 0
    return pool, conv, rec, jnp.asarray(wb), layout


@pytest.mark.parametrize("kb,pad,ring,prefetch", [
    (128, False, 4, 1), (128, True, 4, 1), (512, True, 4, 3), (128, True, 16, 12),
    (512, False, 12, 6),
])
def test_scatter_from_history_is_bitwise(kb, pad, ring, prefetch):
    pool, conv, rec, wb, layout = _scatter_case(kb + ring, kb, pad)
    kw = dict(layout=layout, interpret=True, ring=ring, prefetch=prefetch)
    want = scatter_gdn_ckpts_fast(pool, conv, rec, wb, hist=False, **kw)
    got = scatter_gdn_ckpts_fast(pool, conv, rec, wb, hist=True, **kw)
    np.testing.assert_array_equal(
        _bytes(jax.lax.bitcast_convert_type(got, jnp.uint8)),
        _bytes(jax.lax.bitcast_convert_type(want, jnp.uint8)))


def test_scatter_from_history_follows_the_flag(monkeypatch):
    """Unset `hist` reads VLLM_TPU_MK_IO_LEAN (conv); the result stays the same bytes."""
    pool, conv, rec, wb, layout = _scatter_case(3, 128, True)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "conv", raising=False)
    got = scatter_gdn_ckpts_fast(pool, conv, rec, wb, layout=layout, interpret=True)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "", raising=False)
    want = scatter_gdn_ckpts_fast(pool, conv, rec, wb, layout=layout, interpret=True)
    np.testing.assert_array_equal(
        _bytes(jax.lax.bitcast_convert_type(got, jnp.uint8)),
        _bytes(jax.lax.bitcast_convert_type(want, jnp.uint8)))


def test_scatter_from_history_negative_control():
    """The history shifted by one row (a window off by one) must change the conv rows."""
    pool, conv, rec, wb, layout = _scatter_case(5, 128, False)
    kw = dict(layout=layout, interpret=True, ring=4, prefetch=1)
    want = scatter_gdn_ckpts_fast(pool, conv, rec, wb, hist=False, **kw)
    got = scatter_gdn_ckpts_fast(pool, jnp.roll(conv, 1, axis=2), rec, wb, hist=True, **kw)
    assert not np.array_equal(
        _bytes(jax.lax.bitcast_convert_type(got, jnp.uint8)),
        _bytes(jax.lax.bitcast_convert_type(want, jnp.uint8)))


@pytest.mark.parametrize("cap,pcap", [(48, 32), (32, 64)])
def test_row_capped_levers(monkeypatch, cap, pcap):
    """cktab / conv switch off above VLLM_TPU_MK_IO_LEAN_MAX_ROWS rows (c32 b64 qsa_select
    placement flip), pleconv above VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS (slower stock
    decode PLE conv); the other levers and rows=None are not capped."""
    from vllm_torchtpu.megakernel_runtime.qwen38_io_lean import ROW_CAPPED
    from vllm_torchtpu.megakernel_runtime.qwen38_io_lean import io_lean as lean

    assert ROW_CAPPED == {"cktab", "conv", "pleconv"}
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "all", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN_MAX_ROWS", cap, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS", pcap, raising=False)
    for rows in (4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128):
        for lever in ("cktab", "conv"):
            assert lean(lever, rows=rows) == (rows <= cap), (lever, rows)
        assert lean("pleconv", rows=rows) == (rows <= pcap), rows
        for lever in ("ring", "rope", "ple"):
            assert lean(lever, rows=rows)
    assert lean("cktab") and lean("conv") and lean("pleconv")
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "ring", raising=False)
    assert not lean("cktab", rows=8) and not lean("conv", rows=8)
    assert not lean("pleconv", rows=8)


def test_max_rows_registered_default_48(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_IO_LEAN_MAX_ROWS", raising=False)
    assert envs.environment_variables["VLLM_TPU_MK_IO_LEAN_MAX_ROWS"]() == 48
    monkeypatch.delenv("VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS", raising=False)
    assert envs.environment_variables["VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS"]() == 32
    monkeypatch.setenv("VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS", "64")
    assert envs.environment_variables["VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS"]() == 64


def test_self_check_line_and_once(monkeypatch, caplog):
    import logging

    from vllm_torchtpu.megakernel_runtime import qwen38_io_lean as il

    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN_MAX_ROWS", 48, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS", 32, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", False, raising=False)
    assert il.io_lean_line() is None
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "all", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", True, raising=False)
    assert il.io_lean_line() == (
        "MK_IO_LEAN levers=ring,cktab,rope,conv,ple,pleconv max_rows=48 pleconv_max_rows=32 ple_trim=1"
    )
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "", raising=False)
    assert il.io_lean_line() == "MK_IO_LEAN levers=none max_rows=48 pleconv_max_rows=32 ple_trim=1"
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "rope,ring", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", False, raising=False)
    assert il.io_lean_line() == "MK_IO_LEAN levers=ring,rope max_rows=48 pleconv_max_rows=32 ple_trim=0"
    monkeypatch.setattr(il, "_SELF_CHECK_LOGGED", False)
    with caplog.at_level(logging.INFO):
        il.log_io_lean_once()
        il.log_io_lean_once()
    assert [r.getMessage() for r in caplog.records if "MK_IO_LEAN levers" in r.getMessage()] == [
        "MK_IO_LEAN levers=ring,rope max_rows=48 pleconv_max_rows=32 ple_trim=0"
    ]


def test_self_check_silent_when_off_or_not_rank0(monkeypatch, caplog):
    import logging

    from vllm_torchtpu.megakernel_runtime import qwen38_io_lean as il

    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", False, raising=False)
    monkeypatch.setattr(il, "_SELF_CHECK_LOGGED", False)
    with caplog.at_level(logging.INFO):
        il.log_io_lean_once()
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", "all", raising=False)
    monkeypatch.setattr(il, "_SELF_CHECK_LOGGED", False)
    monkeypatch.setattr(il, "_rank0", lambda: False)
    with caplog.at_level(logging.INFO):
        il.log_io_lean_once()
    assert not [r for r in caplog.records if "MK_IO_LEAN levers" in r.getMessage()]
