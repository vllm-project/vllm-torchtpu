# SPDX-License-Identifier: Apache-2.0
"""The startup MK option line reads the built kernel spec (`log_mk_options`)."""

from __future__ import annotations

import logging

from vllm_torchtpu.megakernel.exec_spec import Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

G = qwen38.TINY


def _spec(**indexed):
    return wholestep_spec(
        G,
        2,
        L=G.att_every,
        first="qsa",
        qsa=dict(budget=64),
        step=True,
        indexed=dict(max_groups=2048, pages_per_seq=64, **indexed),
    )


def test_line_reads_block_and_tiles():
    spec = _spec(
        thr_fast="te", per_request_scores=True, shard_index=True, union_attn=True,
        loop_requests=True,
        req_async=1, xmerge=True, skip_idle=True,
    )
    line = wc.mk_options_line(16, spec, Tiles(hier=1, rel_fetch=True))
    assert line.startswith("qwen38 MK options b16: ")
    kv = dict(p.split("=", 1) for p in line.split(": ", 1)[1].split())
    assert kv["thr_fast"] == "'te'"
    assert kv["req_async"] == "1" and kv["xmerge"] == "1" and kv["skip_idle"] == "1"
    assert kv["select_nvis"] == "0"
    assert kv["key_slots"] == "2"
    assert kv["hier"] == "1" and kv["rel_fetch"] == "1"
    # Off unless VLLM_TPU_MK_UATTN_LITE (absent on kernels without the field).
    assert kv.get("uattn_lite", "0") == "0"


def test_missing_attributes_are_skipped():
    class Block:  # an older kernel: only some of the fields
        key_slots, req_async = 4, True

    class Spec:
        layer_type = type("LT", (), {"slots": ()})()

        def all_blocks(self):
            return [Block()]

    line = wc.mk_options_line(32, Spec(), Tiles(hier=2))
    assert line == "qwen38 MK options b32: key_slots=4 req_async=1 " + " ".join(
        f"{k}={wc._fmt(getattr(Tiles(hier=2), k))}" for k in wc._MK_TILE_FIELDS
    )


def test_key_lanes_logged_and_auto_slots_doubled(monkeypatch):
    """wholekeylanes=128: the block's `key_lanes` shows in the line; auto key-ring slots
    double (same VMEM: half-width slots)."""
    off = wc.mk_options_line(8, _spec(carrier="fp8"), Tiles())
    assert "key_lanes=256" in off
    monkeypatch.setattr(wc, "KEY_LANES", 128)
    monkeypatch.setattr(wc, "KEY_SLOTS", wc.KEY_SLOTS_AUTO)
    assert wc.indexed_options(16)["ck_lanes"] == 128
    assert (wc.key_slots_for(16), wc.key_slots_for(48), wc.key_slots_for(64)) == (8, 8, 6)
    on = wc.mk_options_line(8, _spec(carrier="fp8", ck_lanes=128), Tiles())
    assert "key_lanes=128" in on
    monkeypatch.setattr(wc, "KEY_LANES", 0)
    assert (wc.key_slots_for(16), wc.key_slots_for(64)) == (4, 3)
    assert "ck_lanes" not in wc.indexed_options(16)


def test_thr_fast_one_resolves_to_ter2_and_defaults_off():
    on = wc.mk_options_line(8, _spec(thr_fast="1"), Tiles())
    assert "thr_fast='ter2'" in on
    off = wc.mk_options_line(8, _spec(), Tiles())
    assert "thr_fast=''" in off and "req_async=0" in off and "xmerge=0" in off


def test_logged_once_per_bucket(caplog, monkeypatch):
    monkeypatch.setattr(wc, "_LOGGED_MK_OPTIONS", set())
    spec = _spec()
    with caplog.at_level(logging.INFO):
        wc.log_mk_options(4, spec, Tiles())
        wc.log_mk_options(4, spec, Tiles())
        wc.log_mk_options(8, spec, Tiles())
    lines = [r.getMessage() for r in caplog.records if "MK options" in r.getMessage()]
    assert [ln.split(":")[0] for ln in lines] == ["qwen38 MK options b4", "qwen38 MK options b8"]


def test_non_rank0_silent(caplog, monkeypatch):
    monkeypatch.setattr(wc, "_LOGGED_MK_OPTIONS", set())
    monkeypatch.setattr(wc, "_is_rank0", lambda: False)
    with caplog.at_level(logging.INFO):
        wc.log_mk_options(4, _spec(), Tiles())
    assert not [r for r in caplog.records if "MK options" in r.getMessage()]


def test_draft_label_logged_separately(caplog, monkeypatch):
    from vllm_torchtpu.megakernel_runtime.qwen38_draft import draft_spec

    monkeypatch.setattr(wc, "_LOGGED_MK_OPTIONS", set())
    dspec = draft_spec(G, 2, budget=64, max_groups=2048, vocab_shard=1024)
    with caplog.at_level(logging.INFO):
        wc.log_mk_options(4, _spec(), Tiles())
        wc.log_mk_options(4, dspec, Tiles(), label="MK draft")
        wc.log_mk_options(4, dspec, Tiles(), label="MK draft")
    lines = [r.getMessage() for r in caplog.records if "options b" in r.getMessage()]
    assert [ln.split(":")[0] for ln in lines] == [
        "qwen38 MK options b4", "qwen38 MK draft options b4"
    ]
    # The draft keeps the 2-slot key ring and no manager pages.
    assert "key_slots=2" in lines[1] and "keymp=0" in lines[1]


def test_draft_key_slots_option(monkeypatch):
    """draftkeyslots=N|auto: the draft kernel's key ring (default 2); auto takes the
    target's per-bucket rule, doubled under wholekeylanes=128; operand packing (no
    bucket) keeps 2."""
    from vllm_torchtpu.megakernel_runtime import qwen38_draft as qd

    def slots(B):
        spec = qd.draft_spec(G, 2, budget=64, max_groups=2048, vocab_shard=1024, B=B)
        return wc._find_qsa_block(spec).key_slots

    assert qd.KEY_SLOTS == 2 and slots(16) == 2
    monkeypatch.setattr(qd, "KEY_SLOTS", 4)
    assert slots(16) == 4 and slots(None) == 4
    monkeypatch.setattr(qd, "KEY_SLOTS", wc.KEY_SLOTS_AUTO)
    assert (slots(16), slots(64), slots(None)) == (4, 3, 2)
    monkeypatch.setattr(wc, "KEY_LANES", 128)
    assert (slots(16), slots(48), slots(64)) == (8, 8, 6)
    # The target's own slots do not move the draft's.
    monkeypatch.setattr(wc, "KEY_SLOTS", 3)
    assert slots(16) == 8


def test_driver_parses_draft_key_slots(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_draft as qd
    from vllm_torchtpu.megakernel_runtime.qwen38_driver import _apply_draft_key_slots

    monkeypatch.setattr(qd, "KEY_SLOTS", 2)
    _apply_draft_key_slots("whole=1,wholekeyslots=auto")
    assert qd.KEY_SLOTS == 2
    _apply_draft_key_slots("whole=1,draftkeyslots=4")
    assert qd.KEY_SLOTS == 4
    _apply_draft_key_slots("whole=1,wholekeyslots=auto,draftkeyslots=auto")
    assert qd.KEY_SLOTS == wc.KEY_SLOTS_AUTO


def test_union_lanepack_logged():
    class Block:
        key_slots, union_lanepack = 8, True

    class Spec:
        layer_type = type("LT", (), {"slots": ()})()

        def all_blocks(self):
            return [Block()]

    line = wc.mk_options_line(16, Spec(), Tiles(hier=2))
    assert "union_lanepack=1" in line
