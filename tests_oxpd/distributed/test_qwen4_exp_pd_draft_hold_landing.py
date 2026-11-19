# SPDX-License-Identifier: Apache-2.0
"""q38 F2-HOLD: the decode lands the same bytes with the producer's draft hold on.

``QWEN4_EXP_PD_DRAFT_HOLD=1`` makes an F2 producer hold MTP draft blocks only
above each request's prefix hit Hp: its draft main-KV and compressed-key
tables open with ``floor(Hp / 3200)`` null blocks (core/producer_draft_hold,
tested on the real KVCacheManager in test_qwen4_exp_pd_draft_hold.py). Its
send registers only the live pages and reads the null positions' rows from a
live page (``pd_route.held_ordinal``); the decode is unchanged and zero-fills
every draft row below the published hit.

This file runs XFER-2's routed S12 + F2 prefix-hit harness
(test_qwen4_exp_pd_routed_prefix_hits_f2.py: DECPC's served TP8 / DCP4
decode, the real KVCacheManager and connector scheduler half, all 8 ranks'
real reserve / destination / skip_bytes / draft_zero_end / indices) with the
producer's spans from the REAL ``Qwen4ExpRaidenWorker._stage3_grouped_spans``
(broadcast and routed) instead of an emulation, over four decodes side by
side: broadcast and routed staging, each with the hold off and on. The
producer's tables are the ones the hold builds (``floor(Hp / 3200)`` leading
nulls in the held groups, from the real ``draft_hold_group_indices``). After
every load, on every rank:

- every logical row holds its content (target rows; draft rows of tokens in
  [Hd, max(Hd, Hp)) are 0 and above that the producer's; rings, PLE, GDN);
- the hold-on pool equals the hold-off pool row for row, under broadcast and
  under routed staging, and routed equals broadcast;
- no write lands in an adopted prefix page;
- the hold-on producer never registers a null page, and registers exactly
  ``floor(Hp / 3200)`` fewer pages per held draft group.

A producer page the hold never allocated is not readable at all here (the
emulated transport refuses block 0), so a span that reached one fails.
"""

import math

import numpy as np
import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed import test_qwen4_exp_pd_routed_prefix_hits_f2 as x2
from tests.distributed.test_qwen4_exp_decode_prefix_cache import (  # noqa: F401
    PAGE,
    draft_kv,
    engine_patches,
    served_decode_env,
)

RB = x2.RB
STALE = x2.STALE


@pytest.fixture(autouse=True)
def served_quarantine(monkeypatch):
    """The served staging quarantine (QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS, default 3), whatever the caller's
    environment: XFER-2's and F2-HOLD's VM wrappers export 0, the served-image CPU job does not."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", "3")


@pytest.fixture
def handoff_align(monkeypatch):
    # The served producer (HANDOFF_ALIGN_TOKENS=1) declares token pages in the
    # request-global destination space (pd_connector._stage3_grouped_spans).
    monkeypatch.setenv("TPU_RAIDEN_HANDOFF_ALIGN_TOKENS", "1")


def held_groups(cfg, monkeypatch):
    """The F2 producer plan's held groups, from the real helper, as the
    producer computes them (both flags on)."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.config import ensure_kv_group_size_rule
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    ensure_kv_group_size_rule()  # the producer's grouping (the platform installs it)
    layout_cfg, tp, dcp, pack, cacheable = pd_staging.producer_geometry(cfg)
    _, specs = pooled_cache_specs(
        layout_cfg, tp=tp, main_dcp=dcp, pack_compressed=pack, draft_prefix_cacheable=cacheable
    )
    with monkeypatch.context() as mp:
        mp.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
        return pd_staging.draft_hold_group_indices(get_kv_cache_groups(cfg, dict(specs)))


class Producer:
    """The F2 producer's send for one load: its group tables (with the hold's
    null blocks when on) and the REAL ``_stage3_grouped_spans``."""

    def __init__(self, staging, held, routed, hold):
        from types import SimpleNamespace as NS

        from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
            Qwen4ExpRaidenWorker,
            QwenPagePlan,
        )

        import dataclasses

        self.staging, self.hold = staging, hold
        self.held = held if hold else frozenset()
        # The decode's model of the producer's pools (the tags, strides and live
        # regions both legs fingerprint), each in a carrier of its own as on the
        # producer (one slot tensor per layer), so a group's layers may share IDs,
        # and as many pages as the producer's pool (the harness's staging has 16).
        manifest = dataclasses.replace(
            staging.manifest,
            pools=[
                dataclasses.replace(p, storage_index=i, num_blocks=1 << 20)
                for i, p in enumerate(staging.manifest.pools)
            ],
        )
        plan = QwenPagePlan(staging.src_groups, manifest, self.held)
        w = object.__new__(Qwen4ExpRaidenWorker)
        w._page_plan = lambda: plan  # the admitted plan (rank checks are not under test)
        w.runner = NS(
            qwen4_stage_page_map=None,
            model_adapter=NS(cache=NS(layout=staging.src_layout)),
        )
        w._qwen_route = staging.route if routed else None
        w._qwen_shared = None
        w.vllm_config = NS(parallel_config=NS(pipeline_parallel_size=8))
        w.tp_rank = 0
        self.worker = w

    def tables(self, n, hp):
        """Distinct block IDs per producer group; a held group's first
        floor(hp / page) entries are the null block when the hold is on."""
        tables, next_id = [], 1
        self.heads = []
        for g in self.staging.src_groups:
            count = g.page_count(n)
            head = hp // g.block_size if g.group_index in self.held else 0
            assert head < count, "the base refuses such a send (its table outgrows the extent)"
            tables.append([0] * head + list(range(next_id, next_id + count - head)))
            next_id += count - head
            self.heads.append(head)
        return tables

    def send(self, n, hp):
        from types import SimpleNamespace as NS

        tables = self.tables(n, hp)
        meta = NS(group_block_ids=tables, local_block_ids=list(tables[0]))
        pages, regs = self.worker._stage3_grouped_spans(meta, n)
        # The registration log carries the held pages next to the live ones.
        logged = self.worker._stage3_held_heads
        assert (list(logged) if logged else []) == ([*self.heads] if any(self.heads) else []), logged
        by_name = {}
        for reg, (pool, gi) in zip(regs, self.worker._page_plan().pool_groups()):
            assert reg.tag == pool.tag
            assert 0 not in reg.block_ids, f"{pool.layer_name} registered the null block"
            full = tables[gi]
            index = {int(b): i for i, b in enumerate(full) if b}
            by_name[pool.layer_name] = (reg, [index[int(b)] for b in reg.block_ids], gi)
        return tables, by_name


class HeldServed(x2.Served):
    """XFER-2's decode, fed by the real producer spans (hold on or off)."""

    def __init__(self, routed, hold, held):
        super().__init__(routed)
        self.hold = hold
        self.producer = Producer(self.d.ranks[0], held, routed, hold)
        self.registered = []
        if hold:
            # The producer publishes its draft skip (floor(Hp / page) pages) next
            # to its hit (pd_connector._publish_draft_skip); the decode's
            # connector counts it and lands with max(hit, skip).
            from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
                PRODUCER_DRAFT_SKIP_PARAM,
                PRODUCER_HIT_PARAM,
            )

            original = self.d.sched.on_new_request
            page = self.src_block

            def on_new_request(r, original=original):
                hp = int(r.kv_transfer_params.get(PRODUCER_HIT_PARAM, 0) or 0)
                if hp // page:
                    r.kv_transfer_params.setdefault(PRODUCER_DRAFT_SKIP_PARAM, hp // page * page)
                return original(r)

            self.d.sched.on_new_request = on_new_request

    def land(self, content, req, meta, table_hp=None):
        """``table_hp``: the hit the producer's tables were built from (the
        published one unless a test makes them disagree)."""
        from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging

        d, n = self.d, meta.num_tokens
        skip = int(meta.skip_tokens or 0)
        hp = content.hp if table_hp is None else table_hp
        tables, regs = self.producer.send(n, hp)
        self.registered.append(
            {name: len(reg.block_ids) for name, (reg, _, _) in regs.items()}
        )
        adopted = sorted(d.adopted_pages(meta))
        record = {"n": n, "skip": skip, "adopted": len(adopted), "written_adopted": 0,
                  "sentinel_changed": 0, "heads": tuple(self.producer.heads)}
        for rank, staging in enumerate(d.ranks):
            pool = d.pools[rank]
            sentinel = {p: pool[p * PAGE:(p + 1) * PAGE].copy() for p in adopted}
            assert staging.reserve(
                meta.uuid, n, skip_tokens=skip,
                draft_zero_until=int(getattr(meta, "src_cached_tokens", 0) or 0),
            )
            tags, blocks, counts = staging.destination(meta.uuid)
            clips = staging.skip_bytes(meta.uuid)
            pages = staging.reservations[meta.uuid]
            _, first = staging.skips[meta.uuid]
            rows = self.geo.stage_rows if self.routed else self.src_block
            stage = np.full(staging.pages * rows, STALE, dtype=np.int64)
            stage[:rows] = 0  # page 0: the never-reserved null page
            cursor = 0
            for pool_entry, tag, count, clip in zip(staging.manifest.pools, tags, counts, clips):
                assert pool_entry.tag == tag
                name = pool_entry.layer_name
                kind = pd_staging.owner_kind(name)
                dest = blocks[cursor:cursor + count]
                cursor += count
                reg, abs_pages, _ = regs[name]
                spans = [
                    (s.src_block_ordinal, s.src_offset_bytes, s.dst_offset_bytes, s.size_bytes,
                     s.dst_unit_ordinal)
                    for s in reg.spans
                ]

                def rows_of(src, r, kind=kind, name=name, abs_pages=abs_pages):
                    return content.producer_rows(kind, name, self.src_block, abs_pages[src], r)

                if self.routed or reg.dst_space_version == 1:
                    live = stride = rows if self.routed else pd_staging._live_page_bytes(pool_entry) // RB
                    if not self.routed:
                        stride = self.src_block
                    x2._apply(stage, spans, rank, clip, dest, live, stride, rows_of)
                else:  # request-state pages: page-indexed and whole
                    assert clip == 0 and count == 1 and len(spans) == 1
                    src, so, do, nb, unit = spans[0]
                    assert (src, so, do, unit) == (0, 0, 0, None) and nb == self.src_block * RB
                    r = np.arange(self.src_block, dtype=np.int64)
                    stage[dest[0] * self.src_block + r] = rows_of(0, r)
            zero_end = staging.draft_zero_end(meta.uuid, n)
            record["zero_end"] = zero_end
            main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = staging.indices(
                pages, meta.group_block_ids, n, skip, first, draft_zero_until=zero_end
            )
            record["draft_rows"] = [a + b for a, b in zip(record.get("draft_rows", [0, 0]),
                                                           staging.last_draft_rows)]
            written = [main_d, copy_d]
            pool[main_d] = stage[main_s]
            pool[copy_d] = stage[copy_s]
            j = np.arange(self.geo.gdn_rows, dtype=np.int64)
            for src, dst in zip(gdn_s, gdn_d):
                if self.routed:
                    src = np.asarray(src, dtype=np.int64).reshape(-1)
                    srow = src[j // self.geo.stage_rows] * self.geo.stage_rows + j % self.geo.stage_rows
                else:
                    srow = int(src) * self.src_block + self.gdn_row(rank, j)
                pool[int(dst) * PAGE + j] = stage[srow]
                written.append(int(dst) * PAGE + j)
            touched = set((np.concatenate(written) // PAGE).tolist())
            record["written_adopted"] += len(touched.intersection(adopted))
            record["sentinel_changed"] += sum(
                1 for p, rows_before in sentinel.items()
                if not np.array_equal(pool[p * PAGE:(p + 1) * PAGE], rows_before)
            )
            staging.release(meta.uuid)
            # The engine's next steps: start_load_kv advances the staging
            # quarantine once per step (pd_connector), returning released pages
            # after QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS steps. The harness has
            # no engine steps, so it runs them here; without them the 2,048
            # staging pages are gone after 10 loads (reserve refuses load 11).
            for _ in range(pd_staging.quarantine_steps()):
                staging.advance_quarantine()
            assert staging.quarantined_pages() == 0
        d.kvm.cache_blocks(req, req.num_computed_tokens)
        self.records.append(record)
        return record


# XFER-2's families, plus conversations that make the hold engage at Hd = Hp
# (4,096 and 8,192 are not producer pages: a 1-page hold below each),
# Hd > Hp with a 2-page hold, and two seeds (Hd = 0) whose producer hit 1 and 2 pages.
FAMILIES = [
    ("fam-6k", 6400, [(1000, 0), (719, 6400), (2512, 3200), (1881, 7000)]),
    ("fam-2k", 2300, [(900, 1600), (1500, 2300), (3100, 0)]),
    ("fam-4k", 4500, [(2000, 0), (650, 4500), (3700, 5000), (1200, 4096)]),
    ("fam-8k", 8400, [(1300, 0), (2900, 8400), (1700, 8192), (2100, 6400)]),
    ("fam-seed", 6400, [(1500, 6400), (900, 6400)]),
    ("fam-seed2", 3300, [(5000, 3200)]),
]


def _requests():
    for fi, (family, shared, convs) in enumerate(FAMILIES):
        preamble = dpc.tokens(shared, 1000 + fi)
        for ci, (extra, hp) in enumerate(convs):
            conv = f"{family}-c{ci}"
            yield family, conv, shared, hp, preamble + dpc.tokens(extra + 1, 2000 + 10 * fi + ci)


ARMS = (("broadcast", False, False), ("broadcast-hold", False, True),
        ("routed", True, False), ("routed-hold", True, True))


@pytest.mark.usefixtures("engine_patches", "draft_kv", "handoff_align")
def test_draft_hold_lands_the_same_bytes_under_broadcast_and_routed_staging(monkeypatch):
    old_blocks = dpc.NUM_BLOCKS
    dpc.NUM_BLOCKS = 6000
    try:
        cfg = dpc.decode_config()
        held = held_groups(cfg, monkeypatch)
        assert len(held) == 2, held  # the draft's main KV and compressed keys, not its ring
        arms = {name: HeldServed(routed, hold, held) for name, routed, hold in ARMS}
        names = {g.group_index: g.layer_names for g in arms["broadcast"].d.ranks[0].src_groups}
        assert sorted(names[i][0] for i in held) == [
            "mtp.layers.48.self_attn", "mtp.layers.48.self_attn.compressed_k"
        ], {i: names[i] for i in held}
        loads, failures = [], []
        for family, conv, shared, hp, prompt in _requests():
            out = {}
            for name, arm in arms.items():
                req, meta, hit, ext = arm.d.admit(conv, prompt, producer_hit=hp)
                assert int(getattr(meta, "src_cached_tokens", 0) or 0) == hp
                tables = dpc.full_tables(arm.d, req.request_id)
                content = x2.Content(family, conv, shared, meta.num_tokens, hp=hp)
                rec = arm.land(content, req, meta)
                bad = arm.mismatches(content, meta, tables)
                out[name] = (meta, hit, ext, tables, rec, bad, arm.registered[-1])
                arm.d.decode(req, steps=3)
                arm.d.finish(req)
            base = out["broadcast"]
            for name in arms:
                m, h, e, t = out[name][:4]
                assert (h, e, m.skip_tokens, t) == (base[1], base[2], base[0].skip_tokens, base[3]), (
                    f"{name} diverged from broadcast"
                )
            pools = {name: arm.d.pools for name, arm in arms.items()}

            def equal(a, b):
                return all(np.array_equal(x, y) for x, y in zip(pools[a], pools[b]))

            n = base[0].num_tokens
            heads = out["broadcast-hold"][4]["heads"]
            row = {
                "conv": conv, "n": n, "skip": int(base[0].skip_tokens or 0), "hp": hp,
                "held_pages": max(heads), "zero_end": base[4]["zero_end"],
                "draft_rows": {k: tuple(v[4]["draft_rows"]) for k, v in out.items()},
                "bad": {k: v[5][:2] for k, v in out.items() if v[5]},
                "hold_equals_off_broadcast": equal("broadcast-hold", "broadcast"),
                "hold_equals_off_routed": equal("routed-hold", "routed"),
                "routed_equals_broadcast": equal("routed", "broadcast"),
                "written_adopted": sum(v[4]["written_adopted"] for v in out.values()),
                "sentinel_changed": sum(v[4]["sentinel_changed"] for v in out.values()),
            }
            # Registered pages per held draft owner: floor(hp / page) fewer with the hold.
            reg_on, reg_off = out["broadcast-hold"][6], out["broadcast"][6]
            row["registered_draft_pages"] = {
                k: (reg_off[k], reg_on[k]) for k in reg_on if k.startswith("mtp.")
            }
            print("Q38_F2HOLD_LOAD", row, flush=True)
            loads.append(row)
            if (row["bad"] or not row["hold_equals_off_broadcast"] or not row["hold_equals_off_routed"]
                    or not row["routed_equals_broadcast"] or row["written_adopted"]
                    or row["sentinel_changed"]):
                failures.append(row)
            page = arms["broadcast"].src_block
            for owner, (off, on) in row["registered_draft_pages"].items():
                want = off - (hp // page if not owner.endswith(".raw_ring") else 0)
                assert on == want, (conv, owner, off, on, hp)
        summary = {
            "loads": len(loads),
            "held_loads": sum(1 for r in loads if r["held_pages"]),
            "held_pages_per_group": sum(r["held_pages"] for r in loads),
            "cases": sorted({("Hd<Hp" if r["skip"] < r["hp"] else "Hd=Hp" if r["skip"] == r["hp"]
                              else "Hd>Hp") + ("/held" if r["held_pages"] else "") for r in loads}),
            "seed_held": sum(1 for r in loads if r["skip"] == 0 and r["held_pages"]),
            "draft_rows_landed_zeroed": {
                k: [sum(r["draft_rows"][k][0] for r in loads), sum(r["draft_rows"][k][1] for r in loads)]
                for k in arms
            },
            "failures": len(failures),
        }
        print("Q38_F2HOLD_SUMMARY", summary, flush=True)
        # Coverage the lead asked for: hit and no-hit loads, Hd < Hp, Hd = Hp
        # and Hd > Hp with the hold engaged, and a held seed load.
        assert {"Hd<Hp/held", "Hd=Hp/held", "Hd>Hp/held", "Hd>Hp"} <= set(summary["cases"]), summary
        assert summary["seed_held"] >= 2 and summary["held_loads"] >= 10, summary
        assert any(r["hp"] == 0 for r in loads)
        rows = summary["draft_rows_landed_zeroed"]
        assert len({tuple(v) for v in rows.values()}) == 1, rows  # every arm landed / zeroed the same rows
        # The hold arms' decodes saw the producer's skip on every held load, never above the hit.
        for name in ("broadcast-hold", "routed-hold"):
            c = arms[name].d.conn._hold_decode
            assert c["skip_loads"] == summary["held_loads"] and c["skip_above_hit"] == 0, (name, c)
        assert arms["broadcast"].d.conn._hold_decode["skip_loads"] == 0
        assert not failures, failures
    finally:
        dpc.NUM_BLOCKS = old_blocks


@pytest.mark.usefixtures("engine_patches", "draft_kv", "handoff_align")
def test_a_draft_skip_above_the_published_hit_never_lands_the_filled_rows(monkeypatch):
    """The guard behind the producer's published draft skip: if the hold ever
    skipped more than the hit it published (it cannot by construction), the
    decode zero-fills the draft rows up to the skip (pd_staging.
    published_hit_tokens) instead of landing the live page's rows the send
    filled those positions with."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import PRODUCER_DRAFT_SKIP_PARAM

    old_blocks = dpc.NUM_BLOCKS
    dpc.NUM_BLOCKS = 6000
    try:
        cfg = dpc.decode_config()
        held = held_groups(cfg, monkeypatch)
        for routed in (False, True):
            arm = HeldServed(routed, True, held)
            names = {g.group_index: g.layer_names for g in arm.d.ranks[0].src_groups}
            assert all(names[i][0].startswith("mtp.") for i in held)
            prompt = dpc.tokens(9000, 77)
            published, table_hp = 3200, 6400  # a skip of 2 pages behind a published hit of 1
            # The skip rides the params next to the hit, as the producer publishes it.
            original = arm.d.sched.on_new_request

            def on_new_request(r, original=original):
                r.kv_transfer_params[PRODUCER_DRAFT_SKIP_PARAM] = table_hp
                return original(r)

            monkeypatch.setattr(arm.d.sched, "on_new_request", on_new_request)
            req, meta, hit, ext = arm.d.admit("bad-skip", prompt, producer_hit=published)
            assert int(meta.skip_tokens or 0) == 0
            assert meta.src_cached_tokens == table_hp  # carried by the connector (published_hit_tokens)
            tables = dpc.full_tables(arm.d, req.request_id)
            rec = arm.land(x2.Content("bad", "bad-skip", 0, meta.num_tokens, hp=table_hp), req, meta,
                           table_hp=table_hp)
            assert rec["zero_end"] == table_hp
            bad = arm.mismatches(x2.Content("bad", "bad-skip", 0, meta.num_tokens, hp=table_hp),
                                 meta, tables)
            assert not bad, bad
    finally:
        dpc.NUM_BLOCKS = old_blocks



def draft_rows(arm, meta, tables, lo, hi):
    """Per rank, the pool rows of the draft's main KV and compressed pairs of
    tokens in [lo, hi) (as XFER-2's ``mismatches`` addresses them)."""
    out = []
    for rank, staging in enumerate(arm.d.ranks):
        rows = []
        for name, gi in staging._dst_owner.items():
            if name.startswith("mtp.") and name.endswith(".self_attn"):
                entry = arm.d.kvc.kv_cache_groups[gi].kv_cache_spec.block_size
                split = entry // PAGE
                t = np.arange(lo, hi, dtype=np.int64)
                if split > 1:
                    t = t[(t // PAGE) % split == staging.rank.dcp_rank]
                rows.append(np.asarray(tables[gi], dtype=np.int64)[t // entry] * PAGE + t % PAGE)
        for name, (gi, offset) in staging._dst_compressed.items():
            if name.startswith("mtp."):
                p = np.arange(-(-lo // 8), -(-hi // 8), dtype=np.int64)
                t = 8 * p
                rows.append(np.asarray(tables[gi], dtype=np.int64)[t // PAGE] * PAGE + offset + (t % PAGE) // 8)
        out.append(np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64))
    return out


@pytest.mark.usefixtures("engine_patches", "draft_kv", "handoff_align")
def test_disabling_the_eagle_block_drop_moves_the_zero_fill_up_by_exactly_one_block(monkeypatch):
    """Arm HD against arm H: without vLLM's EAGLE/MTP block drop the producer's
    hit Hp is one 3,200-token page higher (test_qwen4_exp_pd_draft_hold.py,
    on the real KVCacheManager). Its draft computed rows only from that Hp, so
    the decode zero-fills one more page of draft rows, [max(Hd, Hp - 3200), Hp),
    that the drop's producer computed and shipped. Every other row, the
    draft's included, is identical; under broadcast and routed staging."""
    old_blocks = dpc.NUM_BLOCKS
    dpc.NUM_BLOCKS = 6000
    try:
        cfg = dpc.decode_config()
        held = held_groups(cfg, monkeypatch)
        P = None
        loads = (  # (family, shared preamble, suffix, Hp without the drop)
            ("e-6k", 6400, 1000, 0), ("e-6k", 6400, 719, 6400), ("e-6k", 6400, 2512, 6400),
            ("e-10k", 9700, 1500, 9600), ("e-10k", 9700, 900, 9600),
        )
        for routed in (False, True):
            arms = {"HD": HeldServed(routed, True, held), "H": HeldServed(routed, True, held)}
            P = arms["H"].src_block
            for i, (family, shared, extra, hp) in enumerate(loads):
                preamble = dpc.tokens(shared, 300 + len(family))
                prompt = preamble + dpc.tokens(extra + 1, 700 + i)
                conv = f"{family}-{i}"
                hp_by = {"HD": hp, "H": max(0, hp - P) if hp else 0}
                res = {}
                before = {name: [p.copy() for p in arm.d.pools] for name, arm in arms.items()}
                for name, arm in arms.items():
                    req, meta, hit, ext = arm.d.admit(conv, prompt, producer_hit=hp_by[name])
                    tables = dpc.full_tables(arm.d, req.request_id)
                    content = x2.Content(family, conv, shared, meta.num_tokens, hp=hp_by[name])
                    rec = arm.land(content, req, meta)
                    assert not arm.mismatches(content, meta, tables), (name, conv)
                    res[name] = (meta, tables, rec, int(meta.skip_tokens or 0))
                    arm.d.decode(req, steps=2)
                    arm.d.finish(req)
                (m, t, r_hd, skip), (_, t_h, r_h, skip_h) = res["HD"], res["H"]
                assert t == t_h and skip == skip_h
                assert r_hd["zero_end"] - r_h["zero_end"] == (P if hp else 0), (r_hd["zero_end"], r_h["zero_end"])
                lo, hi = max(skip, hp - P), hp
                want = draft_rows(arms["HD"], m, t, lo, hi) if hp and hi > lo else None
                for rank in range(len(arms["HD"].d.pools)):
                    a, b = arms["HD"].d.pools[rank], arms["H"].d.pools[rank]
                    # Rows this load wrote in either decode (earlier loads' differences stay put).
                    written = (a != before["HD"][rank]) | (b != before["H"][rank])
                    diff = np.flatnonzero((a != b) & written)
                    expect = np.unique(want[rank]) if want is not None else np.zeros(0, dtype=np.int64)
                    # Exactly the extra page of draft rows differs: zeros in HD, the producer's in H.
                    assert np.array_equal(diff, expect[a[expect] != b[expect]]), (routed, conv, rank)
                    assert set(diff.tolist()) <= set(expect.tolist())
                    assert (a[expect] == 0).all() and (len(expect) == 0 or (b[expect] != 0).any())
                print("Q38_F2HOLD_EAGLE_LANDING", {"routed": routed, "conv": conv, "skip": skip, "hp_HD": hp,
                                                  "hp_H": hp_by["H"], "zero_end": (r_hd["zero_end"], r_h["zero_end"]),
                                                  "extra_zero_rows_rank0": int(len(want[0])) if want is not None else 0},
                      flush=True)
    finally:
        dpc.NUM_BLOCKS = old_blocks
