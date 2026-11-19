# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL MOE: the PLE lookup chunk on v8's stock buckets only (Benliang Wang <benliangw@google.com> serves
VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS=512 on benliangw/q38fh-integ-pin-5b44702aea; the key, the chunked lookup and its op are
already on v8, byte-identical to his; default 8 = 12 sequential gathers at 96 rows). The port adds
VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS: with 80, lookups of fewer than 80 tokens (the D14 megakernel buckets, at most 64 rows,
which call the same lookup from forward_wholestep's apply_ple) keep today's 8-token program.

Harness (CPU): the REAL decode lookup. ``ResidentPLETable._finish_storage`` (models/vllm/qwen4_exp/ple_storage.py, run on a
stand-in table object) reads the env and calls the lookup builder; the arguments it passes then build the op with the real
``layers/adapter/custom_ops/ple_op.build_raw_lookup_op`` (torch_tpu's jax_op replaced by a recorder that keeps the JAX
function, which is what the TPU op runs), and that function runs under jax.jit on CPU. Served per-token structure: 16
n-gram heads of 160 elements in 128-element FP8 storage segments (Qwen3.8-Flash: ple_embed_dim 2560, 2 n-grams x 8
heads), ids as Qwen4ExpPooledPLE._lookup_ids hands them over (int32, -1 on rows past the valid tokens, rows of other TP
shards masked to zero), one shard of 4096 rows at vocab_start 3 * 4096 of 8.

  BW  at 80 / 96 / 112 / 128 / 160 / 192 / 224 / 256 rows (and 512, and the D14 buckets 8 / 16 / 32 / 48 / 64): flag on
      (CHUNK 512, MIN_ROWS 80) equals flag off (today: CHUNK 8) bit for bit, and both equal a numpy reference.
  ENG the number of gathers the traced program runs per lookup (scan trip count x gathers in the body): flag on 1 at every
      stock bucket, T / 8 flag off; plus the trace-time record ``q38 PORT-BL MOE PLE lookup traced for <T> tokens``.
  D14 with the flag on, the traced lookup program at 8..64 rows is the flag-off program (location-normalised text).
  W   _finish_storage passes MIN_ROWS only when set (the default call is today's); op names; env registered and hashed.
  J   the gated op builds through torch_tpu's REAL jax_op (signature check, schema, registration), its own name.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("torch_tpu")
jnp = jax.numpy
import torch  # noqa: E402

from vllm_torchtpu.layers.adapter.custom_ops import ple_op  # noqa: E402

HEAD_DIM, HEADS, SEG = 160, 16, 128
LOCAL_ROWS, SHARDS, RANK = 4096, 8, 3
STOCK_ROWS = (80, 96, 112, 128, 160, 192, 224, 256)
D14_ROWS = (8, 16, 32, 48, 64)
ON = {"VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS": "512", "VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS": "80"}
KEYS = tuple(ON)
_LOC = re.compile(r"[\w./<>-]*\.py:\d+(?::\d+)?")


class _Op:
    def __init__(self, name, fn, **kwargs):
        self.name, self.fn = name, fn

    def register_fake(self, fake):
        self.fake = fake
        return fake


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)


@pytest.fixture
def stub_jax_op(monkeypatch):
    ple_op.build_raw_lookup_op.cache_clear()
    monkeypatch.setattr(ple_op.pallas, "jax_op", _Op)
    log = _Log()
    monkeypatch.setattr(ple_op, "logger", log)
    monkeypatch.setattr(ple_op, "_LOOKUP_RECORDED", set())
    yield log
    ple_op.build_raw_lookup_op.cache_clear()


def _set_env(monkeypatch, on: bool) -> None:
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    if on:
        for name, value in ON.items():
            monkeypatch.setenv(name, value)


def _finish_storage(monkeypatch):
    """Runs the real ResidentPLETable._finish_storage on a stand-in; returns the builder call it made and the op."""
    from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage

    calls = []

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return ple_op.build_raw_lookup_op(*args, **kwargs)

    monkeypatch.setattr(ple_storage, "build_raw_lookup_op", builder)
    table = SimpleNamespace(
        weight=torch.zeros(LOCAL_ROWS, HEAD_DIM, dtype=torch.float8_e4m3fn),
        storage_dtype=torch.float8_e4m3fn,
        table=torch.empty(0, dtype=torch.uint8),
        shard_indices=SimpleNamespace(org_vocab_start_index=RANK * LOCAL_ROWS, num_org_elements=LOCAL_ROWS),
        embedding_dim=HEAD_DIM,
    )
    ple_storage.ResidentPLETable._finish_storage(table)
    assert len(calls) == 1
    return calls[0], table._raw_lookup


def _table():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 256, (LOCAL_ROWS, HEAD_DIM), dtype=np.uint8)
    return rows, jnp.asarray(rows.reshape(-1, SEG))


def _ids(tokens: int):
    """[tokens, 16] int32 as _lookup_ids returns them: hashed rows over every shard, -1 on the 3 rows past the valid ones."""
    rng = np.random.default_rng(10 + tokens)
    ids = rng.integers(0, SHARDS * LOCAL_ROWS, (tokens, HEADS)).astype(np.int32)
    ids[max(tokens - 3, 1) :] = -1
    return ids


def _expected(rows, ids):
    start = RANK * LOCAL_ROWS
    local = ids.astype(np.int64) - start
    owned = (local >= 0) & (local < LOCAL_ROWS)
    out = np.where(owned[..., None], rows[np.where(owned, local, 0)], 0)
    return out.reshape(ids.shape[0], -1).view(np.int8)


def _gathers(fn, args) -> int:
    """Gathers one call of the traced program runs: gather equations weighted by enclosing scan trip counts."""

    def walk(jaxpr, mult):
        n = 0
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "gather":
                n += mult
            inner = mult * int(eqn.params["length"]) if eqn.primitive.name == "scan" else mult
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(sub, "jaxpr") and hasattr(sub, "consts"):  # ClosedJaxpr
                        n += walk(sub.jaxpr, inner)
                    elif hasattr(sub, "eqns"):  # Jaxpr
                        n += walk(sub, inner)
        return n

    return walk(jax.make_jaxpr(fn)(*args).jaxpr, 1)


def _text(fn, args) -> str:
    return _LOC.sub("<loc>", str(jax.make_jaxpr(fn)(*args)))


def _run(monkeypatch, on: bool, tokens: int, table):
    _set_env(monkeypatch, on)
    call, op = _finish_storage(monkeypatch)
    ids = jnp.asarray(_ids(tokens))
    start = jnp.asarray([RANK * LOCAL_ROWS], jnp.int32)
    out = np.asarray(jax.jit(op.fn)(table, ids, start))
    return call, op, out, _gathers(op.fn, (table, ids, start)), _text(op.fn, (table, ids, start))


@pytest.mark.parametrize("tokens", STOCK_ROWS + (512,) + D14_ROWS, ids=lambda t: f"t{t}")
def test_bw_and_eng(monkeypatch, stub_jax_op, tokens):
    rows, table = _table()
    want = _expected(rows, _ids(tokens))
    c0, op0, off, g0, t0 = _run(monkeypatch, False, tokens, table)
    rec_off = list(stub_jax_op.lines)
    c1, op1, on, g1, t1 = _run(monkeypatch, True, tokens, table)
    rec_on = [s for s in stub_jax_op.lines[len(rec_off) :] if f"traced for {tokens} tokens" in s]
    stock = tokens >= 80
    want_on = -(-tokens // 512) if stock else -(-tokens // 8)
    same = bool(np.array_equal(on, off))
    ref = bool(np.array_equal(off, want)) and bool(np.array_equal(on, want))
    print(
        f"PBL_PLE t{tokens} {'stock' if stock else 'd14'} bw_on_eq_off {same} eq_reference {ref} gathers off {g0} "
        f"on {g1} (want {want_on}) op off {op0.name} on {op1.name} program_same {t0 == t1} record_on {rec_on}"
    )
    # BW (and against the reference, so a fault both arms share is caught too)
    assert off.dtype == on.dtype == np.int8 and off.shape == on.shape == (tokens, HEADS * HEAD_DIM)
    assert (off != 0).mean() > 0.05, "too few owned rows to be a test"
    assert same, f"{int((on != off).sum())} bytes differ between flag on and off"
    assert ref, "lookup differs from the numpy reference"
    # ENG: gathers per lookup from the traced program
    assert g0 == -(-tokens // 8), g0
    assert g1 == want_on, (g1, want_on)
    assert rec_off == []
    assert len(rec_on) == 1 and f"{want_on} gather(s)" in rec_on[0]
    # D14 buckets: the flag-on program is today's
    if not stock:
        assert t1 == t0, "flag on changed the lookup program of a D14 bucket"
    else:
        assert t1 != t0
    assert c0 == ((LOCAL_ROWS, HEAD_DIM, 8), {}) and c1 == ((LOCAL_ROWS, HEAD_DIM, 512), {"min_rows": 80})


def test_op_names(monkeypatch, stub_jax_op):
    names = {
        k: ple_op.build_raw_lookup_op(LOCAL_ROWS, HEAD_DIM, *k).name
        for k in ((8,), (8, 80), (512,), (512, 80))
    }
    print(f"PBL_PLE names {names}")
    assert names[(8,)] == names[(8, 80)] == f"qwen4_exp::ple_raw_lookup_{LOCAL_ROWS}_{HEAD_DIM}"
    assert names[(512,)] == f"qwen4_exp::ple_raw_lookup_{LOCAL_ROWS}_{HEAD_DIM}_c512"
    assert names[(512, 80)] == f"qwen4_exp::ple_raw_lookup_{LOCAL_ROWS}_{HEAD_DIM}_c512_m80"


def test_his_env_only_mode_takes_512_everywhere(monkeypatch, stub_jax_op):
    """CHUNK 512 without MIN_ROWS is Benliang's served behaviour: one gather at every bucket, D14's too."""
    _, table = _table()
    _set_env(monkeypatch, False)
    monkeypatch.setenv("VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", "512")
    call, op = _finish_storage(monkeypatch)
    start = jnp.asarray([RANK * LOCAL_ROWS], jnp.int32)
    got = {t: _gathers(op.fn, (table, jnp.asarray(_ids(t)), start)) for t in (16, 64, 96, 256)}
    print(f"PBL_PLE his-env-only call {call} gathers {got}")
    assert call == ((LOCAL_ROWS, HEAD_DIM, 512), {}) and got == {16: 1, 64: 1, 96: 1, 256: 1}


def test_env_keys_registered_and_hashed(monkeypatch):
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.compilation import tpu_compiler

    _set_env(monkeypatch, False)
    assert envs.VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS == 0 and envs.VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS == 8
    _set_env(monkeypatch, True)
    assert envs.VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS == 80 and envs.VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS == 512
    for name in KEYS:
        assert name in envs.environment_variables and name not in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_j_gated_op_builds_through_the_real_jax_op():
    """J: torch_tpu's real jax_op builds the gated op (its signature check would reject a missing annotation)."""
    ple_op.build_raw_lookup_op.cache_clear()
    try:
        rows = LOCAL_ROWS + 8  # a name no other test registers
        default = ple_op.build_raw_lookup_op(rows, HEAD_DIM, 8)
        gated = ple_op.build_raw_lookup_op(rows, HEAD_DIM, 512, 80)
        print(f"PBL_PLE_J built {default} {gated}")
        assert callable(default) and callable(gated) and default is not gated
    finally:
        ple_op.build_raw_lookup_op.cache_clear()
