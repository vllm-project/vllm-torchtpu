# SPDX-License-Identifier: Apache-2.0
# q38 PORT-BL UNION: adapted from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
# benliangw/q38fh-integ-pin-5b44702aea (tests/kernels/test_qwen4_exp_union_route.py) to the v8 port: the key is
# VLLM_TPU_QSA_DECODE_RPA_UNION_O1S, main-KV DCP is v8's (rank, size) decode layout (no interleave, no own-list
# form), the route also needs 512 selected groups a row, and it is entered only through qsa_attention(o1s_target).
"""Host-side parts of the QSA union decode path: page slots, covered request slots, when the path applies, the
metadata call's form, and that only an O1s target call with the key on reaches the route."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa, qsa_union_meta

KEY = "VLLM_TPU_QSA_DECODE_RPA_UNION_O1S"
ENV = {
    KEY: "640",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
}


@pytest.fixture
def union_env(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def fp8_cache(page=512):
    # Unpaired FP8 main-KV pages (QWEN4_EXP_PACK_MAIN_KV=0): [pages, page, 1 packed sublane, 4 slots, 256].
    return np.zeros((2, page, 1, 4, 256), np.uint8)


def blocks(budget=512):
    return np.zeros((8, budget), np.int32)


def test_page_slots_cover_four_rows_in_whole_tiles():
    slots = qsa_rpa.union_page_slots(512, 4, 640)
    assert slots == 2080 == qsa_union_meta.pages_per_seq(640)
    assert slots * 4 % 640 == 0 and slots >= 4 * (512 + 1)


@pytest.mark.parametrize(
    "rows, requests, covered",
    [(80, 64, 32), (96, 64, 32), (128, 64, 32), (160, 64, 64), (192, 64, 64), (256, 64, 64), (256, 128, 64),
     (512, 128, 128), (32, 8, 32), (64, 16, 32)],
)
def test_covered_request_slots(rows, requests, covered):
    # Calls of 32 requests, as many as rows / 4 fill, at most the program's.
    assert qsa_rpa.union_request_slots(rows, requests, 4) == covered


def test_applies_only_to_the_served_verify_shape(union_env):
    cache = fp8_cache()
    dcp = (np.zeros((1,), np.int32), 4)
    kw = dict(kv_heads=1, ratio=4, kv_head_offset=0, max_query_len=4, dcp=dcp)
    assert qsa_rpa.union_decode_applies(cache, blocks(), **kw)
    assert qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "max_query_len": 2})
    # The draft / non-spec programs (one row a request) never take it, nor longer requests.
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "max_query_len": 1})
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "max_query_len": 5})
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "dcp": None})
    # A third element (an interleave / own-list form) is not v8's decode layout.
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "dcp": (*dcp, 128)})
    assert not qsa_rpa.union_decode_applies(fp8_cache(256), blocks(), **kw)
    assert not qsa_rpa.union_decode_applies(cache, blocks(256), **kw)
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "ratio": 8})
    # Two KV heads: no slot read.
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **{**kw, "kv_heads": 2})
    union_env.setenv("VLLM_TPU_QSA_DECODE_SLOT_READ", "0")
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **kw)
    union_env.setenv("VLLM_TPU_QSA_DECODE_SLOT_READ", "1")
    union_env.setenv("VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", "0")
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **kw)
    union_env.setenv("VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", "1")
    union_env.setenv(KEY, "0")
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **kw)
    union_env.delenv(KEY)
    assert not qsa_rpa.union_decode_applies(cache, blocks(), **kw)
    union_env.setenv(KEY, "600")
    with pytest.raises(ValueError, match="multiple of 128"):
        qsa_rpa.union_decode_applies(cache, blocks(), **kw)


def test_metadata_call_form_and_covered_slots(union_env):
    seen = {}

    def fake_meta(blocks, lengths, starts, table, rank, **kw):
        seen.update(starts=np.asarray(starts), rank=rank, **kw)
        reqs = len(starts) - 1
        return (np.zeros((reqs, 2080), np.int32), np.zeros((reqs, 8320), np.int32), np.zeros((reqs, 16), np.int32))

    union_env.setattr(qsa_union_meta, "union_sc_metadata", fake_meta)
    union_env.setattr(qsa_union_meta, "union_row_live", lambda meta, starts, rows: None)
    rows, requests = 96, 64
    starts = np.minimum(np.arange(requests + 1) * 4, rows).astype(np.int32)
    dcp = (np.zeros((1,), np.int32), 4)
    out = qsa_rpa.union_metadata(
        np.zeros((rows, 512), np.int32), np.zeros((rows,), np.int32), np.zeros((rows, 8), np.int32), starts,
        rows=rows, ratio=4, max_query_len=4, dcp=dcp,
    )
    assert seen["own_list"] is False
    assert seen["pages_per_seq"] == 2080 and seen["world"] == 4
    assert seen["emit_sel"] and seen["write"] == "tile"
    # Only the 32 request slots the step's 96 rows fill reach the kernel.
    assert len(seen["starts"]) == 33 and len(out[0]) == 33


def test_qsa_attention_reaches_the_route_only_for_o1s_target_calls(union_env):
    """qsa_attention enters _union_route only with o1s_target and the key on; every other call (the MTP draft, the
    other DCP attends, prefill) keeps today's decode call."""
    calls = []

    def fake_route(*args, **kwargs):
        calls.append("union")
        q = args[0]
        return jnp.zeros(q.shape, jnp.float32), jnp.zeros(q.shape[:2], jnp.float32), args[5]

    def fake_decode(query, gate, blocks, lengths, table, cache, **kwargs):
        calls.append("rows")
        return jnp.zeros(query.shape, jnp.float32), jnp.zeros(query.shape[:2], jnp.float32), cache

    union_env.setattr(qsa_rpa, "_union_route", fake_route)
    union_env.setattr(qsa_rpa, "decode_attention", fake_decode)
    cache = jnp.zeros((2, 512, 1, 4, 256), jnp.float8_e4m3fn)
    q = jnp.zeros((16, 12, 256), jnp.bfloat16)
    starts = jnp.minimum(jnp.arange(9) * 4, 16).astype(jnp.int32)
    args = (q, q, jnp.zeros((16, 512), jnp.int32), jnp.full((16,), 100, jnp.int32), jnp.zeros((16, 8), jnp.int32),
            cache, starts)
    dcp = (jnp.zeros((1,), jnp.int32), 4)
    kw = dict(kv_heads=1, ratio=4, max_query_len=4, dcp=dcp)
    qsa_rpa.qsa_attention(*args, **kw, o1s_target=True)
    qsa_rpa.qsa_attention(*args, **kw)
    union_env.setenv(KEY, "0")
    qsa_rpa.qsa_attention(*args, **kw, o1s_target=True)
    assert calls == ["union", "rows", "rows"], calls
