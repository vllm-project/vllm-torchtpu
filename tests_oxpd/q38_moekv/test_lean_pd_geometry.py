# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-MOEKV lever 6: MM's lean fused KV writer at PD's served decode geometry and flags.

The lean writer (q38mm DSTOCK-B, MM; cherry-picked from q38mm-dstockb-pickcheck d0d1e8ab0) sits behind
``VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN=1`` together with ``VLLM_TPU_QSA_FUSED_KV_WRITE=1``. It keeps
``fused_layer_write``'s interface and must write the same bytes from compacted work lists. MM proved it
under MM's c48 flags (DCP-2 main KV, compressed DCP, paired main slots, PAIR_SHIFT) and at a DCP-4 256-row
geometry on ranks 0 and 3. This file holds it to PD's served decode instead: the FR10b / stack decode env
(``SERVED_ENV`` from test_kvf_pd_geometry.py: every VLLM_TPU_QSA_* flag of FR10b's DECODE_EXTRA_SERVER_ENV
plus the recipe's two), MAIN_KV_DCP=4 on every rank, compressed keys replicated (no COMPRESSED_DCP),
whole-row main KV (main_slot -1), packed compressed keys (slots 0-7), 512-row pages, up to 64 requests x 4
verify rows (256 tokens), one-row draft steps (VLLM_TPU_QSA_MTP_KV_DCP=1 routes the draft through the same
DCP writes), and O4's 1664-row pages with no DCP.

Every case runs ``qsa_cache_writes`` three ways on the same inputs: the separate writers (the reference,
FUSED off), the fused writer (FUSED on, LEAN unset) and the lean writer (FUSED on, LEAN=1). The pool bytes
and (req, logical, valid) of fused and lean must equal the reference's. Dispatch is asserted per arm: off
calls neither writer, fused calls ``fused_layer_write`` once and never ``_lean_layer_write``, lean calls
both once. Pallas interpret mode on CPU with TPU7x chip info, as the repo's own tests.
"""

import os
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kvf_pd_geometry import (  # noqa: E402
    K_SCALE,
    RATIO,
    ROTARY,
    V_SCALE,
    interpret,  # noqa: F401 (fixture: pallas interpret, TPU7x info, SERVED_ENV)
    make_step,
    raw,
)

import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa  # noqa: E402

LEAN_ENV = "VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN"
_BASE = {}  # reference (separate writers) result per case, shared by the fused and lean arms


def run_writes(monkeypatch, mode, step, *, rank, dcp, cslot):
    """``mode``: "off" (separate writers), "fused" (FUSED on, LEAN unset) or "lean" (FUSED on, LEAN=1)."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", mode != "off")
    if mode == "lean":
        monkeypatch.setenv(LEAN_ENV, "1")
    else:
        monkeypatch.delenv(LEAN_ENV, raising=False)
    fused_calls, lean_calls = [], []
    original_fused = fused_kv_write.fused_layer_write
    original_lean = fused_kv_write._lean_layer_write

    def fused_recording(*args, **kwargs):
        fused_calls.append(len(kwargs["main_rows"]))
        return original_fused(*args, **kwargs)

    def lean_recording(*args, **kwargs):
        lean_calls.append(1)
        return original_lean(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", fused_recording)
    monkeypatch.setattr(fused_kv_write, "_lean_layer_write", lean_recording)
    jax.clear_caches()
    s = step
    extra = {} if dcp <= 1 else {"dcp": (jnp.asarray([rank], jnp.int32), dcp)}
    out = pooled_qsa.qsa_cache_writes(
        s["pool"], s["k"], s["v"], s["raw"], s["positions"], s["nik"], s["trig"], s["starts"],
        s["context"], s["ring_ids"], s["main_table"], s["compressed_table"],
        compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE,
        compressed_slot=cslot, main_slot=-1, max_query_len=s["max_query_len"], **extra,
    )
    monkeypatch.delenv(LEAN_ENV, raising=False)
    return [raw(out[0])] + [np.asarray(x) for x in out[1:]], fused_calls, lean_calls


def check(monkeypatch, mode, key, step, *, rank, dcp, cslot=0):
    """``mode`` arm against the separate writers; returns the number of (page, row) pool rows written."""
    before = raw(step["pool"])
    if key not in _BASE:
        base, f0, l0 = run_writes(monkeypatch, "off", step, rank=rank, dcp=dcp, cslot=cslot)
        assert f0 == [] and l0 == [], "off arm must use neither fused writer"
        _BASE[key] = base
    base = _BASE[key]
    got, fc, lc = run_writes(monkeypatch, mode, step, rank=rank, dcp=dcp, cslot=cslot)
    assert len(fc) == 1, f"{mode} arm must call fused_layer_write once, got {fc}"
    assert lc == ([1] if mode == "lean" else []), f"{mode} arm: lean writer calls {lc}"
    shape = (before.shape[0], before.shape[1], -1)
    assert not np.array_equal(base[0], before), "nothing written"
    diff = np.argwhere((got[0] != base[0]).reshape(shape).any(-1))
    assert len(diff) == 0, f"{mode}: {len(diff)} pool rows differ; first (page, row) {diff[:3].tolist()}"
    for a, b in zip(got[1:], base[1:]):
        np.testing.assert_array_equal(a, b)
    return int((base[0] != before).reshape(shape).any(-1).sum())


WRITERS = ["fused", "lean"]


# --- O1: TP8 + DCP-4, 512-row pages ------------------------------------------------------------

@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("rank,cslot", [(0, 0), (1, 7), (2, 3), (3, 5)])
def test_o1_b64_256_rows(interpret, monkeypatch, writer, rank, cslot):
    """64 requests x 4 verify rows (O1 B64 and B48's m_256 bucket) on every DCP-4 rank, packed
    compressed-key slots from 0 to 7."""
    key = ("b64", rank, cslot)
    rows = check(monkeypatch, writer, key, make_step(101 + rank, 64, page_rows=512, dcp=4),
                 rank=rank, dcp=4, cslot=cslot)
    assert rows >= 16, f"only {rows} pool rows written"


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("requests,idle,rank,pad_to",
                         [(32, (), 1, 0), (48, (5, 20, 47), 2, 256), (16, (), 3, 0)])
def test_o1_128_192_64_rows_and_idle(interpret, monkeypatch, writer, requests, idle, rank, pad_to):
    """B32 (128 rows); B48 with idle slots padded to the m_256 bucket's 256 tokens; B16 (64 rows)."""
    key = ("mid", requests, rank)
    check(monkeypatch, writer, key, make_step(201 + requests, requests, page_rows=512, dcp=4, idle=idle,
                                              pad_to=pad_to), rank=rank, dcp=4, cslot=3)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("requests", [1, 4])
def test_o1_small_batch(interpret, monkeypatch, writer, requests):
    """B1 / B4."""
    check(monkeypatch, writer, ("small", requests),
          make_step(301 + requests, requests, page_rows=512, dcp=4), rank=0, dcp=4, cslot=5)


@pytest.mark.parametrize("writer", WRITERS)
def test_o1_draft_one_row_per_request(interpret, monkeypatch, writer):
    """MTP draft steps: one row per request (64 rows), DCP-4 (FR10b decode: VLLM_TPU_QSA_MTP_KV_DCP=1)."""
    check(monkeypatch, writer, ("draft",), make_step(401, 64, page_rows=512, dcp=4, rows_per_request=1),
          rank=2, dcp=4, cslot=0)


# --- O4: DP4 x TP2, 1664-row pages, no DCP --------------------------------------------------

@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("requests", [16, 8])
def test_o4_per_engine(interpret, monkeypatch, writer, requests):
    """Host B64 / B32 on O4 = 16 / 8 requests x 4 rows per engine, runs crossing pages."""
    check(monkeypatch, writer, ("o4", requests),
          make_step(501 + requests, requests, page_rows=1664, dcp=1, base_ctx=99840), rank=0, dcp=1, cslot=2)


# --- the switch ----------------------------------------------------------------------------

def test_lean_flag_parser(monkeypatch):
    """Unset / "" / "0" -> off, "1" -> on, anything else raises (fused_kv_write.lean_writer_enabled)."""
    assert fused_kv_write.LEAN_ENV == LEAN_ENV
    monkeypatch.delenv(LEAN_ENV, raising=False)
    assert fused_kv_write.lean_writer_enabled() is False
    for text, want in (("", False), ("0", False), (" 0 ", False), ("1", True), (" 1", True)):
        monkeypatch.setenv(LEAN_ENV, text)
        assert fused_kv_write.lean_writer_enabled() is want, text
    for text in ("2", "true", "on", "-1"):
        monkeypatch.setenv(LEAN_ENV, text)
        with pytest.raises(ValueError):
            fused_kv_write.lean_writer_enabled()
