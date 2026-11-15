# SPDX-License-Identifier: Apache-2.0
"""Host-side parts of the QSA union decode path (VLLM_TPU_QSA_DECODE_RPA_UNION):
page slots, covered request slots, when the path applies, and the metadata
call's list form. The kernels themselves run on TPU only (bench script modes
f15, f15b, f15_wire_own, f15_bounds in scripts/bench_qsa_rpa_decode.py)."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa, qsa_union_meta

ENV = {
    "VLLM_TPU_QSA_DECODE_RPA_UNION": "640",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
}


@pytest.fixture
def union_env(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def paired_cache(page=512):
    # FP8 paired main-KV pages: [pages, page, 1 packed sublane, 4 slots, 256].
    return np.zeros((2, page, 1, 4, 256), np.uint8)


def test_page_slots_cover_four_rows_in_whole_tiles():
    slots = qsa_rpa.union_page_slots(512, 4, 640)
    assert slots == 2080 == qsa_union_meta.pages_per_seq(640)
    assert slots * 4 % 640 == 0 and slots >= 4 * (512 + 1)


@pytest.mark.parametrize(
    "rows, requests, covered",
    [(96, 128, 32), (128, 128, 32), (192, 128, 64), (256, 128, 64), (384, 128, 96),
     (512, 128, 128), (32, 8, 32), (64, 16, 32), (256, 64, 64)],
)
def test_covered_request_slots(rows, requests, covered):
    # Calls of 32 requests, as many as rows / 4 fill, at most the program's.
    assert qsa_rpa.union_request_slots(rows, requests, 4) == covered


def test_applies_only_to_the_served_verify_shape(union_env):
    cache = paired_cache()
    dcp = (np.zeros((1,), np.int32), 4)
    kw = dict(kv_heads=1, ratio=4, kv_head_offset=1, max_query_len=4, dcp=dcp)
    assert qsa_rpa.union_decode_applies(cache, **kw)
    assert qsa_rpa.union_decode_applies(cache, **{**kw, "dcp": (*dcp, 0, True)})
    assert not qsa_rpa.union_decode_applies(cache, **{**kw, "max_query_len": 1})
    assert not qsa_rpa.union_decode_applies(cache, **{**kw, "max_query_len": 5})
    assert not qsa_rpa.union_decode_applies(cache, **{**kw, "dcp": None})
    # Interleaved ownership is the PCP prefill's layout.
    assert not qsa_rpa.union_decode_applies(cache, **{**kw, "dcp": (*dcp, 128)})
    assert not qsa_rpa.union_decode_applies(paired_cache(256), **kw)
    union_env.setenv("VLLM_TPU_QSA_DECODE_SLOT_READ", "0")
    assert not qsa_rpa.union_decode_applies(cache, **kw)
    union_env.setenv("VLLM_TPU_QSA_DECODE_RPA_UNION", "0")
    assert not qsa_rpa.union_decode_applies(cache, **kw)
    union_env.setenv("VLLM_TPU_QSA_DECODE_RPA_UNION", "600")
    with pytest.raises(ValueError, match="multiple of 128"):
        qsa_rpa.union_decode_applies(cache, **kw)


@pytest.mark.parametrize("own", [False, True])
def test_metadata_reads_the_list_form_and_covered_slots(union_env, own):
    seen = {}

    def fake_meta(blocks, lengths, starts, table, rank, **kw):
        seen.update(starts=np.asarray(starts), **kw)
        reqs = len(starts) - 1
        return (np.zeros((reqs, 2080), np.int32),) * 2 + (np.zeros((reqs, 16), np.int32),)

    union_env.setattr(qsa_union_meta, "union_sc_metadata", fake_meta)
    union_env.setattr(qsa_union_meta, "union_row_live", lambda meta, starts, rows: None)
    rows, requests = 96, 128
    starts = np.minimum(np.arange(requests + 1) * 4, rows).astype(np.int32)
    dcp = (np.zeros((1,), np.int32), 4) + ((0, True) if own else ())
    out = qsa_rpa.union_metadata(
        np.zeros((rows, 512), np.int32), np.zeros((rows,), np.int32),
        np.zeros((rows, 8), np.int32), starts, rows=rows, ratio=4, max_query_len=4,
        dcp=dcp,
    )
    assert seen["own_list"] is own
    assert seen["pages_per_seq"] == 2080 and seen["world"] == 4
    assert seen["emit_sel"] and seen["write"] == "tile"
    # Only the 32 request slots the step's 96 rows fill reach the kernel.
    assert len(seen["starts"]) == 33 and len(out[0]) == 33
