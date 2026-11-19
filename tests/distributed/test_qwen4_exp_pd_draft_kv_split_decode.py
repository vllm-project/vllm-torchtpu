# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS (q38 BLSTACK-R) on the served Qwen3.8
decode emulation (CPU): q38 F2's draft KV landed on a staged TP8 / DCP4
decode whose compressed keys are split over the DCP ranks
(VLLM_TPU_QSA_COMPRESSED_DCP=1), as Benliang Wang's served decode runs them.

* The real StagedConsumer, built on all 8 ranks from the real pooled layout
  with his decode's layout flags (DCP4, VLLM_TPU_QSA_MTP_KV_DCP,
  VLLM_TPU_QSA_COMPRESSED_DCP, packed compressed keys, paired main KV off and
  on, routed staging QWEN4_EXP_PD_ROUTED_DST off and on): refuses F2 with the key off (test_qwen4_exp_decode_prefix_cache.py's
  test_draft_kv_refuses_compressed_dcp, kept), builds with it on (that test
  inverted), stages the draft's three owners and logs the engagement line.
* The decode emulation of test_qwen4_exp_decode_prefix_cache.py, its reads
  extended to split compressed keys (block k of a split group on DCP rank
  k % D, at row table[t // entry] * B + slot + (t % B) // 8), runs his F2
  scenarios under compressed DCP: byte identity of the draft from producer to
  every rank, multi-turn prefix hits, partial eviction, the producer-hit gap
  and the forced gap. A draft slot landed one row off must be caught.
"""

import math
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed.test_qwen4_exp_decode_prefix_cache import (  # noqa: F401 (fixtures)
    _DRAFT,
    DCP,
    PAGE,
    TP,
    decode_config,
    draft_kv,
    engine_patches,
    scheduler_plan,
    served_decode_env,
)

KEY = "QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS"
REFUSAL = "refuses VLLM_TPU_QSA_COMPRESSED_DCP=1"


@pytest.fixture
def compressed_dcp(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP", "1")


@pytest.fixture
def split_keys(monkeypatch, compressed_dcp):
    monkeypatch.setenv(KEY, "1")


@pytest.fixture(params=["pm0", "pm1"])
def pack_main(request, monkeypatch):
    # His served decode runs QWEN4_EXP_PACK_MAIN_KV=1; the emulation module pins 0.
    monkeypatch.setenv("QWEN4_EXP_PACK_MAIN_KV", "1" if request.param == "pm1" else "0")
    return request.param


@pytest.fixture(params=["route_off", "route_his"])
def routed(request, monkeypatch):
    # His served legs run QWEN4_EXP_PD_ROUTED_DST=8:4:512 (TP8, DCP4, 512-token pages).
    if request.param == "route_his":
        monkeypatch.setenv("QWEN4_EXP_PD_ROUTED_DST", f"{TP}:{DCP}:{PAGE}")
    else:
        monkeypatch.delenv("QWEN4_EXP_PD_ROUTED_DST", raising=False)
    return request.param


def _consumers(ranks=range(TP)):
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups

    cfg = decode_config()
    layout, native, _ = scheduler_plan(cfg)
    out = []
    with patch.object(pd_staging, "make_repack", lambda **kw: None):
        for rank in ranks:
            out.append(
                pd_staging.StagedConsumer(
                    cfg,
                    layout,
                    build_transfer_groups(native.kv_cache_groups),
                    torch.zeros(1, dtype=torch.float8_e4m3fn),
                    pd_staging.RankSlice(
                        tp_size=TP,
                        tp_rank=rank,
                        num_kv_heads=int(cfg.model_config.hf_text_config.num_key_value_heads),
                        main_dcp=DCP,
                    ),
                    staging_bytes=2 * layout.page_bytes * 8,
                )
            )
    return layout, out


@pytest.mark.usefixtures("engine_patches", "draft_kv", "compressed_dcp")
def test_key_off_refuses_on_his_decode_layout(monkeypatch, pack_main, routed):
    monkeypatch.delenv(KEY, raising=False)
    layout = scheduler_plan(decode_config())[0]
    assert _DRAFT + ".compressed_k" in layout.main_dcp_owners
    with pytest.raises(ValueError, match=REFUSAL):
        _consumers(ranks=[0])


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys")
def test_key_on_builds_every_rank_and_logs_the_engagement(monkeypatch, pack_main, routed):
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging

    lines = []
    monkeypatch.setattr(
        pd_staging,
        "logger",
        NS(info=lambda msg, *a: lines.append(msg % a if a else msg),
           warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    layout, consumers = _consumers()
    draft_ck = _DRAFT + ".compressed_k"
    assert draft_ck in layout.main_dcp_owners and _DRAFT in layout.main_dcp_owners
    for rank, c in enumerate(consumers):
        assert c.draft_kv
        assert {_DRAFT, draft_ck, _DRAFT + ".raw_ring"} <= set(c._src_owner)
        assert draft_ck in c._dst_compressed and draft_ck in c._dcp_owners
        assert c._split_rank(draft_ck) is c.rank
        # Routed staging engaged exactly when his route key is set.
        assert (c.route is not None) == (routed == "route_his")
        if c.route is not None:
            assert (c.route.tp, c.route.dcp, c.route.dst_block) == (TP, DCP, PAGE)
    engaged = [m for m in lines if m.startswith("Qwen PD F2 draft KV split keys: rank ")]
    assert len(engaged) == TP and all(draft_ck in m and "DCP 4" in m for m in engaged)


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys")
def test_key_on_inverts_his_full_consumer_refusal():
    # test_qwen4_exp_decode_prefix_cache.py's test_draft_kv_refuses_compressed_dcp, inverted.
    layout = scheduler_plan(decode_config())[0]
    assert any(name.endswith(".compressed_k") for name in layout.main_dcp_owners)
    _consumers(ranks=[0])


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys")
def test_key_on_keeps_the_mtp_kv_dcp_refusal(monkeypatch):
    # Without VLLM_TPU_QSA_MTP_KV_DCP the layout itself refuses compressed DCP.
    monkeypatch.delenv("VLLM_TPU_QSA_MTP_KV_DCP")
    with pytest.raises(ValueError, match="VLLM_TPU_QSA_MTP_KV_DCP"):
        _consumers(ranks=[0])


# --- the emulation, reads extended to split compressed keys ----------------------


class SplitDecode(dpc.Decode):
    """dpc.Decode with every compressed read following its group's split:
    a group of entry E = D x B tokens (pool_layout: a DCP-split owner's
    block_size) stores block k = t // B on DCP rank k % D, at row
    table[t // E] * B + slot offset + (t % B) // 8. With E = B (unsplit)
    this is the module's own rule."""

    def _comp_row(self, staging, gi, table, offset, t):
        entry = self.kvc.kv_cache_groups[gi].kv_cache_spec.block_size
        split = entry // PAGE
        if split > 1 and (t // PAGE) % split != staging.rank.dcp_rank:
            return None
        return table[t // entry] * PAGE + offset + (t % PAGE) // 8

    def read(self, rank, conv, prompt, meta, full_tables):
        staging = self.ranks[rank]
        n = meta.num_tokens
        pool = self.pools[rank]
        bad = []
        for name, gi in staging._dst_owner.items():
            kind = name.rsplit(".", 1)[-1]
            table = full_tables[gi]
            if self.draft_kv and name.startswith("mtp."):
                continue
            if name.endswith(".self_attn"):
                for t in range(n):
                    if (t // PAGE) % DCP != staging.rank.dcp_rank:
                        continue
                    row = table[t // dpc.ENTRY] * PAGE + t % PAGE
                    if pool[row] != dpc.main_value(conv, name, t):
                        bad.append((name, t))
            elif kind == "raw_ring":
                if pool[table[0] * PAGE] != dpc.state_value(conv, name, n, 0):
                    bad.append((name, "ring"))
        checked = 0
        for name, (gi, offset) in staging._dst_compressed.items():
            if self.draft_kv and name.startswith("mtp."):
                continue
            table = full_tables[gi]
            for p in range(math.ceil(math.ceil(n / 4) / 2)):
                row = self._comp_row(staging, gi, table, offset, 8 * p)
                if row is None:
                    continue
                checked += 1
                if pool[row] != dpc.pair_value(conv, name, p, n):
                    bad.append((name, p))
        self.comp_checked = getattr(self, "comp_checked", 0) + checked
        for gi, ids in enumerate(meta.group_block_ids):
            group = self.kvc.kv_cache_groups[gi]
            names = [x for x in group.layer_names if x in staging._dst_owner]
            if not names or not (
                names[0].endswith(".linear_attn") or names[0].endswith(".ple")
            ):
                continue
            k = getattr(group.kv_cache_spec, "num_speculative_blocks", 0)
            if ids != [full_tables[gi][-(k + 1)]]:
                bad.append((names[0], "entry"))
            if pool[ids[0] * PAGE] != dpc.state_value(conv, names[0], n, 0):
                bad.append((names[0], "state"))
        if self.draft_kv:
            bad.extend(self.read_draft(rank, conv, meta, full_tables))
        return bad

    def read_draft(self, rank, conv, meta, full_tables):
        from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import owner_kind

        staging = self.ranks[rank]
        n = meta.num_tokens
        pool = self.pools[rank]
        lo, hi = self.draft_gap(meta)
        bad = []
        for name, (gi, offset) in sorted(self.draft_owners().items()):
            table = full_tables[gi]
            kind = owner_kind(name)
            if kind == "main":
                entry = self.kvc.kv_cache_groups[gi].kv_cache_spec.block_size
                split = entry // PAGE
                for t in range(n):
                    if split > 1 and (t // PAGE) % split != staging.rank.dcp_rank:
                        continue
                    row = table[t // entry] * PAGE + t % PAGE
                    want = 0 if lo <= t < hi else dpc.main_value(conv, name, t)
                    if pool[row] != want:
                        bad.append((name, t))
            elif kind == "compressed":
                for p in range(math.ceil(math.ceil(n / 4) / 2)):
                    t = 8 * p
                    row = self._comp_row(staging, gi, table, offset, t)
                    if row is None:
                        continue
                    self.draft_comp_checked = getattr(self, "draft_comp_checked", 0) + 1
                    want = 0 if lo <= t < hi else dpc.pair_value(conv, name, p, n)
                    if pool[row] != want:
                        bad.append((name, p))
            else:
                assert kind == "ring", name
                for r in range(PAGE):
                    if pool[table[0] * PAGE + r] != dpc.state_value(conv, name, n, r):
                        bad.append((name, "ring", r))
        return bad


@pytest.fixture
def split_emulation(monkeypatch):
    # His scenario functions build `Decode(...)` by module global name.
    monkeypatch.setattr(dpc, "Decode", SplitDecode)


def _split_engaged(decode):
    draft_ck = _DRAFT + ".compressed_k"
    assert decode.layout.main_dcp_owners and draft_ck in decode.layout.main_dcp_owners
    assert all(draft_ck in s._dcp_owners and draft_ck in s._dst_compressed for s in decode.ranks)
    entry = {
        g.kv_cache_spec.block_size
        for g in decode.kvc.kv_cache_groups
        if any(n.endswith(".compressed_k") for n in g.layer_names)
    }
    assert entry == {DCP * PAGE}, entry  # the split groups' D x B entries


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys", "split_emulation")
def test_split_draft_ships_and_every_rank_reads_the_producer_rows():
    dpc.test_draft_kv_ships_the_draft_and_every_rank_reads_the_producer_rows()
    decode = SplitDecode(decode_config())
    _split_engaged(decode)
    conv, prompt = "conv-split", dpc.tokens(20000, 31)
    req, meta, hit, ext, tables = dpc.serve(decode, conv, prompt)
    assert dpc.logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP
    # Every rank checked its own share of the draft's split pairs: each pair on
    # the TP / D ranks of its DCP rank.
    pairs = math.ceil(math.ceil((len(prompt) - 1) / 4) / 2)
    assert decode.draft_comp_checked == (TP // DCP) * pairs
    assert decode.comp_checked > 0


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys", "split_emulation")
def test_split_draft_multi_turn_hits_land_only_the_suffix_and_read_identically():
    dpc.test_multi_turn_decode_hits_land_only_the_suffix_and_read_identically()


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys", "split_emulation")
@pytest.mark.parametrize("kind", ["main", "compressed", "draft"])
def test_split_draft_hit_after_partial_eviction_reads_no_unwritten_rows(kind):
    dpc.test_hit_is_the_shortest_token_group_hit_after_partial_eviction(kind)


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys", "split_emulation")
@pytest.mark.parametrize("case", ["above", "below"])
def test_split_draft_producer_hit_gap_zero_fills_the_owned_rows(case):
    dpc.test_draft_kv_producer_hit_above_the_decode_hit_zero_fills_the_gap(case)


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys", "split_emulation")
def test_split_draft_forced_gap_zero_fills_exactly_the_requested_span(monkeypatch):
    dpc.test_draft_kv_forced_gap_zero_fills_exactly_the_requested_span(monkeypatch)


@pytest.mark.usefixtures("engine_patches", "draft_kv", "split_keys")
def test_the_emulation_catches_a_draft_slot_landed_one_row_off():
    """Control: the extended reads are not vacuous. The draft's split slot
    landed one row down must read wrong on some rank (and its last row lands
    in the next member's slot)."""
    decode = SplitDecode(decode_config())
    draft_ck = _DRAFT + ".compressed_k"
    for s in decode.ranks:
        gi, offset = s._dst_compressed[draft_ck]
        s._dst_compressed[draft_ck] = (gi, offset + 1)
    conv, prompt = "conv-off-by-one", dpc.tokens(20000, 32)
    req, meta, hit, ext, tables = dpc.serve(decode, conv, prompt)
    bad = [decode.read_draft(r, conv, meta, tables) for r in range(TP)]
    assert sum(len(b) for b in bad) > 0
    assert all(name == draft_ck for b in bad for name, *_ in b)
