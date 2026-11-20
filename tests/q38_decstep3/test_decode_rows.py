# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-STEP-3: the decode step above 128 seats (384 / 448 / 480 token rows, k2 verify).

  R1 VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE (port of DECODE-QSA WIDE v2): call width 32 with the key off, 64 above
     one 32-request call's rows with it on; only 64 accepted; covered request slots never read the key.
  R2 WIDE glue: union_decode_attention with a per-sequence stand-in for the RPA kernel (each sequence's rows come out
     as a function of its own rows, kv length and page words only) gives the same output and LSE with the key on as
     off at 384 / 448 / 480 rows, 128 / 149 / 160 requests, the sliced and the chained (VLLM_TPU_QSA_UNION_CHAINED_CALLS)
     paths; calls 4 / 5 / 5 -> 2 / 3 / 3; the last WIDE call takes the remaining requests, no padded ones.
  R3 VLLM_TPU_DECODE_ROWS_GLUE: qsa_rpa.selected_tokens equal (values, dtype) with the key on as off.
  R4 VLLM_TPU_DECODE_ROWS_GLUE: gdn wrapper.pack_gate_inputs equal with the key on as off.
  R5 VLLM_TPU_QSA_DCP_SORT_ROWS: pooled_qsa._sort_rows / compressed_dcp_blocks equal to the one sort; 448 / 480 rows
     at 384 run two row blocks; at or below the limit one sort.
  R6 keys registered, default off, compile-ignored (the compile factors do not change).
  R7 lowering (TPU7x Mosaic on CPU): selected_tokens holds no scatter with the glue on (it holds one per tail off);
     _sort_rows holds 2 sorts at 448 rows with the limit 384 and 1 without it.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper
from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa, qsa_rpa

WIDE = "VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE"
GLUE = "VLLM_TPU_DECODE_ROWS_GLUE"
SORT = "VLLM_TPU_QSA_DCP_SORT_ROWS"
KEYS = (WIDE, GLUE, SORT)
SHAPES = [(384, 128), (448, 149), (480, 160)]  # (token rows, requests), k2: 3 rows a request


@pytest.fixture
def clean(monkeypatch):
    for k in KEYS + ("VLLM_TPU_QSA_UNION_CHAINED_CALLS", "VLLM_TPU_QSA_UNION_LEAN_GLUE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_RPA_UNION", "640")
    qsa_rpa.WIDE_TRACED.clear()
    return monkeypatch


# ---------------- R1


def test_r1_call_width(clean):
    for rows in (96, 288, 384, 448, 480):
        assert qsa_rpa.union_call_requests(rows, 3) == 32
    clean.setenv(WIDE, "64")
    assert qsa_rpa.union_call_requests(96, 3) == 32
    assert [qsa_rpa.union_call_requests(r, 3) for r in (97, 288, 384, 448, 480)] == [64] * 5
    for bad in ("32", "48", "96", "128"):
        clean.setenv(WIDE, bad)
        assert qsa_rpa.union_call_requests(96, 3) == 32
        with pytest.raises(ValueError, match="only 64"):
            qsa_rpa.union_call_requests(448, 3)


@pytest.mark.parametrize("rows, requests", SHAPES + [(288, 96), (160, 53)])
def test_r1_covered_slots_never_read_the_key(clean, rows, requests):
    off = qsa_rpa.union_request_slots(rows, requests, 3)
    clean.setenv(WIDE, "64")
    assert qsa_rpa.union_request_slots(rows, requests, 3) == off


# ---------------- R2


class _FakeRPA:
    """Per-sequence stand-in for wrapper.ragged_paged_attention: sequence i's rows (cu_q_lens[i]..cu_q_lens[i+1] of the
    call's query) come out as query * (1 + kv_len % 5) + (page-word sum % 7) and LSE kv_len + row-in-sequence; other
    rows are left as the call found them (zeros, or the carried buffers)."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, cache, kv_lens, pages, cu, dist, *, carry=None, return_carry=False, **kw):
        q = np.asarray(q); kv_lens = np.asarray(kv_lens); cu = np.asarray(cu); pages = np.asarray(pages)
        n = len(kv_lens)
        per = pages.reshape(n, -1)
        self.calls.append(dict(n=n, rows=q.shape[0], dist=np.asarray(dist).tolist(), sel=np.asarray(kw["selection_bits"]).shape[0]))
        if carry is not None:
            out, lse = (np.array(x, np.float32) for x in carry)
        else:
            out = np.zeros(q.shape, np.float32)
            lse = np.full(q.shape[:2], -np.inf, np.float32)
        for i in range(n):
            a, b = int(cu[i]), int(cu[i + 1])
            if b <= a:
                continue
            out[a:b] = q[a:b] * (1 + int(kv_lens[i]) % 5) + float(int(per[i].sum()) % 7)
            lse[a:b] = (int(kv_lens[i]) + np.arange(b - a))[:, None].astype(np.float32)
        return jnp.asarray(out), cache, jnp.asarray(lse)


def _union_case(rows, requests, seed):
    rng = np.random.default_rng(seed)
    slots = qsa_rpa.union_request_slots(rows, requests, 3)
    q_rows = rng.integers(1, 4, size=requests)  # 1..3 rows a request (verify rows)
    starts = np.concatenate([[0], np.cumsum(q_rows)]).astype(np.int32)
    starts = np.minimum(starts, rows)
    starts = np.concatenate([starts, np.full(max(0, slots - requests), starts[-1], np.int32)])[: slots + 1]
    q = rng.standard_normal((rows, 12, 256)).astype(np.float32)
    pages = rng.integers(0, 1000, size=(slots, 2080)).astype(np.int32)
    sel = rng.integers(0, 2**31, size=(slots, 2080)).astype(np.int32)
    meta = np.stack([np.zeros(slots), rng.integers(1, 9000, size=slots)], axis=1).astype(np.int32)
    meta = np.pad(meta, ((0, 0), (0, 14)))
    live = np.arange(rows) < starts[-1]
    return q, (jnp.asarray(starts), jnp.asarray(pages), jnp.asarray(sel), jnp.asarray(meta), jnp.asarray(live))


def _run_union(monkeypatch, rows, requests, chained, wide, seed=0):
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

    fake = _FakeRPA()
    monkeypatch.setattr(rpa_wrapper, "ragged_paged_attention", fake)
    monkeypatch.setattr(rpa_wrapper, "finish_carry", lambda carry, shape, n, dtype: carry)
    monkeypatch.setenv("VLLM_TPU_QSA_UNION_CHAINED_CALLS", "1" if chained else "0")
    if wide:
        monkeypatch.setenv(WIDE, "64")
    else:
        monkeypatch.delenv(WIDE, raising=False)
    q, meta = _union_case(rows, requests, seed)
    cache = jnp.zeros((4, 512, 1, 4, 256), jnp.float32)
    out, lse, _ = qsa_rpa.union_decode_attention(
        jnp.asarray(q), None, None, None, cache, None, ratio=4, k_scale=1.0, v_scale=1.0, kv_head_offset=1,
        max_query_len=3, dcp=None, read_only=True, metadata=meta,
    )
    return np.asarray(out), np.asarray(lse), fake.calls


@pytest.mark.parametrize("rows, requests", SHAPES)
@pytest.mark.parametrize("chained", [False, True])
def test_r2_wide_glue_same_output(clean, rows, requests, chained):
    out0, lse0, calls0 = _run_union(clean, rows, requests, chained, wide=False, seed=rows)
    out1, lse1, calls1 = _run_union(clean, rows, requests, chained, wide=True, seed=rows)
    assert np.array_equal(out0, out1) and np.array_equal(lse0, lse1)
    slots = qsa_rpa.union_request_slots(rows, requests, 3)
    assert len(calls0) == -(-slots // 32) and len(calls1) == -(-slots // 64)
    assert [c["n"] for c in calls1] == [min(64, slots - 64 * i) for i in range(len(calls1))]
    assert all(c["dist"] == [c["n"]] * 3 and c["sel"] == c["n"] for c in calls1)
    print(f"R2 rows {rows} requests {requests} slots {slots} chained {chained}: calls {len(calls0)} -> {len(calls1)} "
          f"{[c['n'] for c in calls1]}")


def test_r2_wide_marker_once_per_program(clean, caplog):
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

    clean.setattr(rpa_wrapper, "ragged_paged_attention", _FakeRPA())
    clean.setenv(WIDE, "64")
    q, meta = _union_case(448, 149, 1)
    cache = jnp.zeros((4, 512, 1, 4, 256), jnp.float32)
    for _ in range(2):
        qsa_rpa.union_decode_attention(jnp.asarray(q), None, None, None, cache, None, ratio=4, k_scale=1.0,
                                       v_scale=1.0, kv_head_offset=1, max_query_len=3, dcp=None, read_only=True,
                                       metadata=meta)
    assert qsa_rpa.WIDE_TRACED == {(448, 160): 3}


# ---------------- R3


@pytest.mark.parametrize("rows", [384, 448, 480])
def test_r3_selected_tokens_glue(clean, rows):
    rng = np.random.default_rng(rows)
    blocks = rng.integers(-1, 30000, size=(rows, 512)).astype(np.int32)
    blocks[: rows // 4, 200:] = -1
    lengths = rng.integers(0, 4 * 512 + 9, size=rows).astype(np.int32)
    lengths[:8] = [0, 1, 2, 3, 4, 2048, 2049, 2051]
    off = np.asarray(qsa_rpa.selected_tokens(jnp.asarray(blocks), jnp.asarray(lengths), 2048, 4))
    clean.setenv(GLUE, "1")
    on = np.asarray(qsa_rpa.selected_tokens(jnp.asarray(blocks), jnp.asarray(lengths), 2048, 4))
    assert on.dtype == off.dtype and on.shape == off.shape == (rows, 2051)
    assert np.array_equal(on, off)


# ---------------- R4


@pytest.mark.parametrize("rows, pad", [(384, 0), (448, 0), (480, 0), (447, 1)])
def test_r4_gate_inputs_glue(clean, rows, pad):
    rng = np.random.default_rng(rows)
    b = jnp.asarray(rng.standard_normal((rows, 32)).astype(np.float32))
    a = jnp.asarray(rng.standard_normal((rows, 32)).astype(np.float32))
    off = np.asarray(gdn_wrapper.pack_gate_inputs(b, a, pad, 96, rows + pad))
    clean.setenv(GLUE, "1")
    on = np.asarray(gdn_wrapper.pack_gate_inputs(b, a, pad, 96, rows + pad))
    assert on.shape == off.shape == (rows + pad, 256) and on.dtype == off.dtype
    assert np.array_equal(on, off)


# ---------------- R5


def _count_prims(jaxpr, name):
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == name
        for v in eqn.params.values():
            for sub in v if isinstance(v, (list, tuple)) else (v,):
                if hasattr(sub, "eqns"):
                    n += _count_prims(sub, name)
                elif hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    n += _count_prims(sub.jaxpr, name)
    return n


@pytest.mark.parametrize("rows, limit, parts", [(384, 384, 1), (448, 384, 2), (480, 384, 2), (448, 0, 1), (448, 160, 3)])
def test_r5_sort_rows(clean, rows, limit, parts):
    rng = np.random.default_rng(rows + limit)
    x = rng.integers(0, 2**20, size=(rows, 1024)).astype(np.int32)
    x[rng.random(x.shape) < 0.3] = dcp_select.EMPTY
    want = np.sort(x, axis=1)
    if limit:
        clean.setenv(SORT, str(limit))
    got = np.asarray(pooled_qsa._sort_rows(jnp.asarray(x)))
    assert np.array_equal(got, want)
    jax.clear_caches()
    jaxpr = jax.make_jaxpr(lambda y: pooled_qsa._sort_rows(y))(jax.ShapeDtypeStruct(x.shape, jnp.int32)).jaxpr
    jax.clear_caches()
    assert _count_prims(jaxpr, "sort") == parts, _count_prims(jaxpr, "sort")


@pytest.mark.parametrize("rows", [384, 448, 480])
def test_r5_compressed_dcp_blocks(clean, rows):
    """The real compressed_dcp_blocks body with its two TPU-only selection kernels stood in (the same stand-ins both
    ways): the row-block sort gives the same global selection."""
    rng = np.random.default_rng(rows)
    vals = jnp.asarray(rng.standard_normal((2, rows, 512)).astype(np.float32))
    ids = rng.permuted(np.tile(np.arange(4096), (rows, 1)), axis=1)[:, :512]
    pos = jnp.asarray(np.stack([ids * 2 + r for r in (0, 1)]).astype(np.int32))
    clean.setattr(dcp_select, "merge_threshold", lambda v, p, k, topk=None, from_scores=False: (
        jnp.median(v, axis=(0, 2)), jnp.zeros((v.shape[1],), jnp.int32)))
    clean.setattr(dcp_select, "keep_mask", lambda v, p, tv, tp: v >= tv[:, None])
    off = np.asarray(pooled_qsa.compressed_dcp_blocks(vals, pos, 512))
    clean.setenv(SORT, "384")
    on = np.asarray(pooled_qsa.compressed_dcp_blocks(vals, pos, 512))
    assert on.dtype == off.dtype and on.shape == off.shape == (rows, 512)
    assert np.array_equal(on, off) and (off >= 0).any()


# ---------------- R6


def test_r6_keys_registered_default_off_compile_hashed(clean):
    """Registered, default off, and compile-HASHED: the keys are read inside the model forward, behind custom-op
    boundaries the FX graph fingerprint does not see, so the env factors (and the compile cache directory) must move
    with each of them, or a warm cache would restore the other programs."""
    assert envs.VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE == 0 and envs.VLLM_TPU_DECODE_ROWS_GLUE is False
    assert envs.VLLM_TPU_QSA_DCP_SORT_ROWS == 0
    for k in KEYS:
        assert k in envs.environment_variables and k not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    off = tpu_compiler._tpu_compile_env_factors()
    seen = [off]
    for k, v in ((WIDE, "64"), (GLUE, "1"), (SORT, "384")):
        clean.setenv(k, v)
        f = tpu_compiler._tpu_compile_env_factors()
        assert all(f != g for g in seen), k
        seen.append(f)
        clean.delenv(k)


# ---------------- R7


def _lower_tpu(fn, *specs):
    from jax._src.mesh import AbstractDevice, AbstractMesh

    jax.clear_caches()
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        text = jax.jit(fn).trace(*specs).lower(lowering_platforms=("tpu",)).as_text()
    jax.clear_caches()
    return text


@pytest.mark.parametrize("rows", [384, 448])
def test_r7_selected_tokens_lowering(clean, rows):
    specs = (jax.ShapeDtypeStruct((rows, 512), jnp.int32), jax.ShapeDtypeStruct((rows,), jnp.int32))
    fn = lambda b, l: qsa_rpa.selected_tokens(b, l, 2048, 4)  # noqa: E731
    off = _lower_tpu(fn, *specs)
    clean.setenv(GLUE, "1")
    on = _lower_tpu(fn, *specs)
    print(f"R7 rows {rows}: scatters off {off.count('stablehlo.scatter')} on {on.count('stablehlo.scatter')}")
    assert off.count("stablehlo.scatter") > 0 and on.count("stablehlo.scatter") == 0


# ---------------- R8 engagement markers (traced once per shape; none with the keys off)


def test_r8_markers(clean):
    import vllm_torchtpu.kernels.gdn.v3.wrapper as gw

    qsa_rpa.ROWS_GLUE_TRACED.clear(); pooled_qsa.SORT_ROWS_TRACED.clear(); gw._GATE_GLUE_TRACED.clear()
    blocks = jnp.zeros((448, 512), jnp.int32); lengths = jnp.full((448,), 100, jnp.int32)
    x = jnp.zeros((448, 1024), jnp.int32); b = jnp.zeros((448, 32), jnp.float32)
    qsa_rpa.selected_tokens(blocks, lengths, 2048, 4); pooled_qsa._sort_rows(x); gw.pack_gate_inputs(b, b, 0, 96, 448)
    assert not qsa_rpa.ROWS_GLUE_TRACED and not pooled_qsa.SORT_ROWS_TRACED and not gw._GATE_GLUE_TRACED
    clean.setenv(GLUE, "1"); clean.setenv(SORT, "384")
    for _ in range(2):
        qsa_rpa.selected_tokens(blocks, lengths, 2048, 4); pooled_qsa._sort_rows(x); gw.pack_gate_inputs(b, b, 0, 96, 448)
    assert qsa_rpa.ROWS_GLUE_TRACED == {("selected_tokens", 448)}
    assert pooled_qsa.SORT_ROWS_TRACED == {448} and gw._GATE_GLUE_TRACED == {448}
