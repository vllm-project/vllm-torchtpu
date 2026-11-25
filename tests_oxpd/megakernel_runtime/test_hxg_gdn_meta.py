# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-GLUE GM (VLLM_TPU_HXG_GDN_META, CPU): the whole step's shared GDN metadata program at every bucket (b16 included) and the
GDN-SIDECAR accepted index from one vectorised gather.

Rows:
  G1 side_offsets_vectorised == `_side_xs`'s per-layer list, bitwise (shape, dtype, values) over random ids: short / exact / long id
     arrays, out-of-range ids, 1 / 3 / 33 layers, and the mixed-shape fallback.
  G2 the gate: gdn_meta_shared(16) with DEDUP=1, MIN_ROWS=32 is False with GM off and True with GM on; DEDUP=0 keeps it False.
  G3 _side_xs_gm == _side_xs (every operand bitwise) with GM on; GM off it IS _side_xs's result; no sidecar operand -> {}.
  G4 gm_bind_line: logger.info on every call naming the rank; raises ValueError without the shared program.
  G5 call sites (AST, live calls): call_shared builds the sidecar operands through _side_xs_gm; bind_pool calls gm_bind_line under the
     key; gdn_meta_shared consults _hxg_gm_on before MIN_ROWS.
  G6 the shared block ids equal the per-layer ones at NS = 16 (gdn_blocks_shared vs gdn_blocks, the D07-b1 identity GM now uses at b16).
  G7 registration: registered and hashed.
Run: PYTHONPATH=src pytest tests_oxpd/megakernel_runtime/test_hxg_gdn_meta.py
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import hixglue
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import GDN_META, T_VERIFY, _fit, gdn_blocks, gdn_blocks_shared

KEY = "VLLM_TPU_HXG_GDN_META"


def _per_layer(off, sts, NS, T):
    return jnp.stack([jnp.clip(off[_fit(jnp.asarray(st), NS)], 0, T - 1) for st in sts]).astype(jnp.int32)


def test_g1_vectorised_offsets_bitwise():
    rng = np.random.default_rng(1)
    n = 0
    for NS in (4, 16):
        for L in (NS - 3, NS, NS + 4):
            for NL in (1, 3, 33):
                off = jnp.asarray(rng.integers(-2, 6, size=64), dtype=jnp.int32)
                sts = [jnp.asarray(rng.integers(-1, 70, size=L), dtype=jnp.int32) for _ in range(NL)]
                a, b = _per_layer(off, sts, NS, T_VERIFY), wsc.side_offsets_vectorised(off, sts, NS, T_VERIFY)
                assert a.shape == b.shape and a.dtype == b.dtype and bool(jnp.array_equal(a, b))
                n += 1
    off = jnp.arange(10, dtype=jnp.int32) % 4
    sts = [jnp.arange(3, dtype=jnp.int32), jnp.arange(7, dtype=jnp.int32)]
    assert bool(jnp.array_equal(_per_layer(off, sts, 4, 4), wsc.side_offsets_vectorised(off, sts, 4, 4)))
    assert n == 18


@pytest.mark.parametrize(("dedup", "minr", "gm", "B", "want"), [
    ("1", "32", "0", 16, False), ("1", "32", "1", 16, True), ("1", "32", "0", 32, True), ("0", "0", "1", 16, False),
])
def test_g2_gate(monkeypatch, dedup, minr, gm, B, want):
    monkeypatch.setenv("VLLM_TPU_MK_GDN_META_DEDUP", dedup)
    monkeypatch.setenv("VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS", minr)
    monkeypatch.setenv(KEY, gm)
    assert wsc.gdn_meta_shared(B) is want


class _Fn:
    def __init__(self):
        from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_REC

        self.extra_names = ("gdn_pool", SIDE_REC)


def test_g3_side_xs_gm(monkeypatch):
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_OFF

    rng = np.random.default_rng(2)
    off = jnp.asarray(rng.integers(0, 4, size=40), dtype=jnp.int32)
    sts = [jnp.asarray(rng.integers(0, 40, size=16), dtype=jnp.int32) for _ in range(33)]
    side = (jnp.zeros((2, 3)), jnp.ones((2, 2)), jnp.asarray([1], dtype=jnp.int32))
    monkeypatch.setenv(KEY, "0")
    ref = wsc._side_xs(_Fn(), side, off, sts, 16, T_VERIFY)
    off_mode = wsc._side_xs_gm(_Fn(), side, off, sts, 16, T_VERIFY)
    monkeypatch.setenv(KEY, "1")
    got = wsc._side_xs_gm(_Fn(), side, off, sts, 16, T_VERIFY)
    assert set(ref) == set(got) == set(off_mode)
    for k in ref:
        assert bool(jnp.array_equal(ref[k], got[k])) and bool(jnp.array_equal(ref[k], off_mode[k]))
        assert ref[k].dtype == got[k].dtype
    assert got[SIDE_OFF].shape == (33, 16)
    assert wsc._side_xs_gm(_Fn(), None, off, sts, 16, T_VERIFY) == {}


def test_g4_bind_line(caplog):
    caplog.set_level(logging.INFO, logger=hixglue.logger.name)
    m1 = wsc.gm_bind_line(5, "whole", True, (8, 16))
    m2 = wsc.gm_bind_line(6, "whole", True, (16,))
    assert m1.startswith(hixglue.BIND_PREFIX + KEY + " bound rank 5:") and "(8,16)" in m1
    assert sum(KEY + " bound rank" in r.getMessage() for r in caplog.records) == 2 and "rank 6" in m2
    with pytest.raises(ValueError, match="VLLM_TPU_MK_GDN_META_DEDUP=1"):
        wsc.gm_bind_line(5, "whole", False, (16,))


def _calls(src: str, name: str) -> int:
    tree = ast.parse(textwrap.dedent(src))
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
               and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name))


def test_g5_call_sites():
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as ws

    src = inspect.getsource(wsc.build_wholestep_call)
    i = src.index("def call_shared(")
    body = src[i: src.index("# q38 L1-D07B1 end", i)]  # call_shared up to its end marker
    assert _calls(body, "_side_xs_gm") == 1 and _calls(body, "_side_xs") == 0
    assert _calls(inspect.getsource(ws.Qwen38WholeStep.bind_pool), "gm_bind_line") == 1
    gate = inspect.getsource(wsc.gdn_meta_shared)
    assert gate.index("_hxg_gm_on()") < gate.index("VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS)")


def test_g6_shared_ids_equal_per_layer_at_b16():
    rng = np.random.default_rng(3)
    NS, NL, nb = 16, 33, 200
    off = jnp.asarray(rng.integers(0, 4, size=nb), dtype=jnp.int32)
    qsl = jnp.asarray(np.r_[0, np.cumsum([4, 4, 4, 0] + [0] * (NS - 4))][: NS + 1], dtype=jnp.int32)
    seq = jnp.asarray(rng.integers(0, 9, size=NS), dtype=jnp.int32)
    sts = [jnp.asarray(rng.integers(0, nb, size=NS), dtype=jnp.int32) for _ in range(NL)]
    cks = [jnp.asarray(rng.integers(0, nb, size=(NS, T_VERIFY + 1)), dtype=jnp.int32) for _ in range(NL)]
    meta = []
    for st, ck in zip(sts, cks):
        meta += [st, off, ck, qsl, seq]
    assert len(meta) == GDN_META * NL
    for write in (True, False):
        a = gdn_blocks(meta, NS, write)
        b = gdn_blocks_shared(off, qsl, seq, sts, cks, NS, write)
        for x, y in zip(a, b):
            assert bool(jnp.array_equal(x, y))


def test_g7_registered_and_hashed():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert KEY in envs.environment_variables and KEY not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
