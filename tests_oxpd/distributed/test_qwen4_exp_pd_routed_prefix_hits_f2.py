# SPDX-License-Identifier: Apache-2.0
"""S12 routed staging + F2 draft KV on decode prefix-cache hits (CPU, served plan).

The S12 + F2 combination for FR7 (q38-int-3fbxd): the producer ships the MTP
draft's KV (QWEN4_EXP_PD_DRAFT_KV=1) and this TP8 / DCP4 decode splits the
draft's main KV like a target layer's (VLLM_TPU_QSA_MTP_KV_DCP=1). XFER-2's
routed prefix-hit check (``test_qwen4_exp_pd_routed_prefix_hits`` on
int-3fbx) extended to the draft's owners and F2 option (a)'s gap: the
producer published its prefix hit Hp, computed no draft rows below it (the
emulated producer stages ``UNWRITTEN`` there), and the decode's repack
zero-fills the draft rows in [Hd, Hp) (``draft_zero_end``) when Hp > Hd.

A broadcast decode and a routed decode run the same requests side by side on
DECPC's served plan (the real KVCacheManager with the decode's engine-core
patches, the real connector scheduler half carrying Hp, all 8 ranks' real
``StagedConsumer`` reserve / destination / skip_bytes / draft_zero_end /
indices, the producer's real routed spans for every pool, the draft's too).
After every load, on every rank:

- target rows hold their tokens' content; draft rows of tokens in
  [Hd, max(Hd, Hp)) are 0 and above that hold the producer's content; rings
  (the draft's too), PLE and the committed GDN entry hold the state;
- the routed pool equals the broadcast pool, row for row;
- no write lands in an adopted prefix page (the draft's included), and those
  pages are row-identical before and after the load.

The transport follows tpu-sync's ``SplitPlainGlobalSpan`` (drop below the
clip, trim the straddler, re-base); staging is stale-filled before each load.
"""

import math

import numpy as np
import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed.test_qwen4_exp_decode_prefix_cache import (  # noqa: F401
    DCP,
    ENTRY,
    K,
    PAGE,
    TP,
    UNWRITTEN,
    draft_kv,
    engine_patches,
    served_decode_env,
)

ROUTED = "8:4:512"
RB = 1024
STALE = -7


# -- content: KV of token t depends on tokens <= t ------------------------------------


class Content:
    """Producer row values of one load. Tokens before ``shared`` belong to the
    family (identical KV for every conversation sharing the preamble); later
    tokens, every partial compression pair and all request state belong to
    the conversation."""

    def __init__(self, family, conv, shared, n, hp=0):
        self.family, self.conv, self.shared, self.n = family, conv, shared, n
        self.hp = hp  # F2: the producer computed no draft rows below its hit
        self._main, self._pair, self._state = {}, {}, {}

    def main(self, owner, t):
        t = np.asarray(t, dtype=np.int64)
        out = np.zeros(t.shape, dtype=np.int64)
        for i, x in np.ndenumerate(t):
            x = int(x)
            if x >= self.n:
                continue
            key = (owner, x)
            v = self._main.get(key)
            if v is None:
                who = self.family if x < self.shared else self.conv
                v = self._main[key] = dpc.main_value(who, owner, x)
            out[i] = v
        return out

    def pair(self, owner, p):
        p = np.asarray(p, dtype=np.int64)
        out = np.zeros(p.shape, dtype=np.int64)
        for i, x in np.ndenumerate(p):
            x = int(x)
            if 8 * x >= self.n:
                continue
            key = (owner, x)
            v = self._pair.get(key)
            if v is None:
                full = 8 * x + 8 <= self.n
                who = self.family if full and 8 * x + 8 <= self.shared else self.conv
                v = self._pair[key] = dpc.pair_value(who, owner, x, self.n)
            out[i] = v
        return out

    def state(self, owner, rows):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros(rows.shape, dtype=np.int64)
        for i, r in np.ndenumerate(rows):
            key = (owner, int(r))
            v = self._state.get(key)
            if v is None:
                v = self._state[key] = dpc.state_value(self.conv, owner, self.n, int(r))
            out[i] = v
        return out

    def producer_rows(self, kind, owner, src_block, page, rows):
        """Rows ``rows`` of the producer's page ordinal ``page`` of ``owner``."""
        rows = np.asarray(rows, dtype=np.int64)
        unwritten = self.hp if owner.startswith("mtp.") else 0
        if kind == "main":
            t = page * src_block + rows
            return np.where(t < unwritten, UNWRITTEN, self.main(owner, t))
        if kind == "compressed":
            live = src_block // 8
            p = page * live + rows
            vals = np.where(8 * p < unwritten, UNWRITTEN, self.pair(owner, p))
            return np.where(rows < live, vals, 0)
        if page != 0:
            raise AssertionError(f"request-state owner {owner} has one page")
        return self.state(owner, rows)


# -- the transport and the repack, emulated from the production plans -------------------


def _apply(stage, spans, rank, clip_bytes, dest, live_rows, stride_rows, rows_of):
    """Raiden's landing of one tag on one destination unit.

    ``spans``: (src page, src byte offset, dst byte offset, bytes, unit or
    None), dst offsets in the request-global space of live destination
    bytes; ``dest``: the tag's destination (staging) pages, each
    ``live_rows`` live rows at a ``stride_rows`` row stride; the clip drops
    the first ``clip_bytes`` of that space (a whole number of pages)."""
    assert clip_bytes % (live_rows * RB) == 0, "dst_skip_bytes is not whole destination pages"
    skip = clip_bytes // RB
    cover = []
    for src, so, do, nb, unit in spans:
        if unit is not None and unit != rank:
            continue
        assert so % RB == 0 and do % RB == 0 and nb % RB == 0 and nb > 0
        s0, d0, nr = so // RB, do // RB, nb // RB
        if d0 + nr <= skip:
            continue
        if d0 < skip:  # the straddling span is trimmed (SplitPlainGlobalSpan)
            cut = skip - d0
            s0, d0, nr = s0 + cut, skip, nr - cut
        g0 = d0 - skip
        cover.append((g0, g0 + nr))
        r = np.arange(nr, dtype=np.int64)
        gp, ip = np.divmod(g0 + r, live_rows)
        assert gp.max() < len(dest), "span past the transfer's destination pages"
        stage[np.asarray(dest, dtype=np.int64)[gp] * stride_rows + ip] = rows_of(src, s0 + r)
    cover.sort()
    end = 0
    for a, b in cover:
        assert a == end, f"coverage gap or overlap at row {end} (span at {a})"
        end = b
    assert (len(dest) - 1) * live_rows < end <= len(dest) * live_rows, (
        "coverage must fill every destination page; only the final one may be partial"
    )


def _broadcast_spans(kind, num_pages, live_rows, src_block):
    """``Qwen4ExpRaidenWorker._stage3_grouped_spans`` (pipeline producer,
    TPU_RAIDEN_HANDOFF_ALIGN_TOKENS > 0): token pages in the global space,
    request-state pages page-indexed (never clipped)."""
    if kind in ("main", "compressed"):
        return [(i, 0, i * live_rows * RB, live_rows * RB, None) for i in range(num_pages)], True
    return [(0, 0, 0, src_block * RB, None)], False


class Served:
    """One decode (broadcast or routed staging) on DECPC's served plan."""

    def __init__(self, routed: bool):
        from vllm_torchtpu.models.vllm.qwen4_exp import pd_route

        with pytest.MonkeyPatch.context() as mp:
            if routed:
                mp.setenv("QWEN4_EXP_PD_ROUTED_DST", ROUTED)
            else:
                mp.delenv("QWEN4_EXP_PD_ROUTED_DST", raising=False)
            self.d = dpc.Decode(dpc.decode_config())
        self.routed = routed
        r0 = self.d.ranks[0]
        assert (r0.route is not None) == routed
        self.src_block = r0.src_block
        if routed:
            self.geo = r0.route
        else:
            # The served numbers, for the GDN head map of the broadcast repack.
            from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import gdn_geometry

            s = gdn_geometry(self.d.cfg, 1, r0.src_layout.token_bytes)
            t = gdn_geometry(self.d.cfg, TP, self.d.layout.token_bytes)
            self.geo = pd_route.RouteGeometry(
                tp=TP, dcp=DCP, dst_block=PAGE, src_block=self.src_block, ratio=r0.ratio,
                ssm_rows_src=s.ssm_tokens, conv_rows_src=s.conv_tokens, ssm_rows_dst=t.ssm_tokens,
            )
        self.records = []

    def gdn_row(self, rank, j):
        """Producer row of a rank's GDN destination row ``j`` (its value heads'
        SSM rows, then the conv rows; channel selection stays inside rows)."""
        g = self.geo
        j = np.asarray(j, dtype=np.int64)
        return np.where(j < g.ssm_rows_dst, rank * g.ssm_rows_dst + j,
                        g.ssm_rows_src + (j - g.ssm_rows_dst))

    def land(self, content, req, meta):
        from vllm_torchtpu.models.vllm.qwen4_exp import pd_route, pd_staging

        d, n = self.d, meta.num_tokens
        skip = int(meta.skip_tokens or 0)
        adopted = sorted(d.adopted_pages(meta))
        record = {"n": n, "skip": skip, "adopted": len(adopted), "written_adopted": 0,
                  "sentinel_changed": 0, "straddles": 0}
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
            stage[:rows] = 0  # page 0: the never-reserved null page, zero (the gap's source row)
            cursor = 0
            for pool_entry, tag, count, clip in zip(staging.manifest.pools, tags, counts, clips):
                assert pool_entry.tag == tag
                name = pool_entry.layer_name
                kind = pd_staging.owner_kind(name)
                dest = blocks[cursor:cursor + count]
                cursor += count
                assert dest == pages[name]
                src_pages = staging.src_groups[staging._src_owner[name]].page_count(n)

                def rows_of(src, r, kind=kind, name=name):
                    return content.producer_rows(kind, name, self.src_block, src, r)

                if self.routed:
                    spans = pd_route.producer_spans(kind, n, src_pages, self.geo)
                    live = stride = self.geo.stage_rows
                    if clip:
                        skip_r = clip // RB
                        record["straddles"] += sum(
                            1 for _, _, do, nb, u in spans
                            if u in (None, rank) and do // RB < skip_r < (do + nb) // RB
                        )
                    _apply(stage, spans, rank, clip, dest, live, stride, rows_of)
                else:
                    live = pd_staging._live_page_bytes(pool_entry) // RB
                    spans, global_space = _broadcast_spans(kind, src_pages, live, self.src_block)
                    if global_space:
                        _apply(stage, spans, rank, clip, dest, live, self.src_block, rows_of)
                    else:
                        assert clip == 0 and count == 1
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
        d.kvm.cache_blocks(req, req.num_computed_tokens)
        self.records.append(record)
        return record

    def mismatches(self, content, meta, tables):
        """(rank, owner, what) of every logical row that does not hold its content."""
        d, n = self.d, meta.num_tokens
        skip = int(meta.skip_tokens or 0)
        gap_hi = max(skip, min(content.hp, n))  # F2: draft rows in [skip, gap_hi) are 0
        bad = []
        for rank, staging in enumerate(d.ranks):
            pool = d.pools[rank]
            for name, gi in staging._dst_owner.items():
                table = tables[gi]
                draft = name.startswith("mtp.")
                if name.endswith(".self_attn"):
                    entry = d.kvc.kv_cache_groups[gi].kv_cache_spec.block_size
                    split = entry // PAGE  # DCP ranks sharing one table entry
                    # A draft's prefix rows are its own cache's (sentinel-checked).
                    t = np.arange(skip if draft else 0, n, dtype=np.int64)
                    if split > 1:
                        t = t[(t // PAGE) % split == staging.rank.dcp_rank]
                    rows = np.asarray(table, dtype=np.int64)[t // entry] * PAGE + t % PAGE
                    want = content.main(name, t)
                    if draft:
                        want = np.where(t < gap_hi, 0, want)
                    wrong = t[pool[rows] != want]
                    if len(wrong):
                        bad.append((rank, name, "main", int(wrong[0]), len(wrong)))
                elif name.endswith(".raw_ring"):
                    r = np.arange(self.geo.prefix_rows, dtype=np.int64)
                    if not np.array_equal(pool[table[0] * PAGE + r], content.state(name, r)):
                        bad.append((rank, name, "ring"))
            for name, (gi, offset) in staging._dst_compressed.items():
                table = np.asarray(tables[gi], dtype=np.int64)
                draft = name.startswith("mtp.")
                p = np.arange((skip // 8) if draft else 0, math.ceil(math.ceil(n / 4) / 2), dtype=np.int64)
                t = 8 * p
                rows = table[t // PAGE] * PAGE + offset + (t % PAGE) // 8
                want = content.pair(name, p)
                if draft:
                    want = np.where(t < gap_hi, 0, want)
                wrong = p[pool[rows] != want]
                if len(wrong):
                    bad.append((rank, name, "pairs", int(wrong[0]), int(wrong[-1]), len(wrong)))
            for gi, ids in enumerate(meta.group_block_ids):
                group = d.kvc.kv_cache_groups[gi]
                names = [x for x in group.layer_names if x in staging._dst_owner]
                if not names or not (names[0].endswith(".linear_attn") or names[0].endswith(".ple")):
                    continue
                k = getattr(group.kv_cache_spec, "num_speculative_blocks", 0)
                if list(ids) != [tables[gi][-(k + 1)]]:
                    bad.append((rank, names[0], "entry"))
                    continue
                j = np.arange(self.geo.gdn_rows if names[0].endswith(".linear_attn")
                              else self.geo.prefix_rows, dtype=np.int64)
                want_rows = self.gdn_row(rank, j) if names[0].endswith(".linear_attn") else j
                for name in names:
                    if not np.array_equal(pool[ids[0] * PAGE + j], content.state(name, want_rows)):
                        bad.append((rank, name, "state"))
        return bad


# -- the g1p-shaped workload -------------------------------------------------------------

# (family, preamble tokens, conversations, suffix tokens of each). Hits land
# at floor(preamble / 2048) * 2048: 2,048, 4,096, 6,144 (g1p: 1,319 of 1,320
# loads, num_tokens 7,119..8,912) and 8,192. At 6,144 the compressed clip
# (768 compact rows) cuts the producer's second 400-row page span.
# Each conversation: (suffix tokens, the producer's prefix hit Hp). Gaps
# (Hp > Hd) on hit loads at every skip, aligned (6,400 = two producer pages)
# and not, one crossing a 512-token page; Hp < Hd and Hp = 0 (no gap); and a
# seed (Hd = 0) whose producer had a hit, so its gap starts at token 0.
FAMILIES = [
    ("fam-6k", 6400, [(1000, 0), (719, 6400), (2512, 3200), (1881, 7000)]),
    ("fam-2k", 2300, [(900, 1600), (1500, 2300), (3100, 0)]),
    ("fam-4k", 4500, [(2000, 0), (650, 4500), (3700, 5000)]),
    ("fam-8k", 8400, [(1300, 0), (2900, 8400)]),
]


def _requests():
    for fi, (family, shared, convs) in enumerate(FAMILIES):
        preamble = dpc.tokens(shared, 1000 + fi)
        for ci, (extra, hp) in enumerate(convs):
            conv = f"{family}-c{ci}"
            yield family, conv, shared, hp, preamble + dpc.tokens(extra + 1, 2000 + 10 * fi + ci)


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_s12_f2_routed_prefix_hits_land_like_broadcast_with_the_draft_gap_zeroed():
    old_blocks = dpc.NUM_BLOCKS
    dpc.NUM_BLOCKS = 6000  # every family's preamble stays cached for its turns
    try:
        arms = {"broadcast": Served(False), "routed": Served(True)}
        hits, failures = [], []
        for family, conv, shared, hp, prompt in _requests():
            out = {}
            for arm_name, arm in arms.items():
                req, meta, hit, ext = arm.d.admit(conv, prompt, producer_hit=hp)
                assert int(getattr(meta, "src_cached_tokens", 0) or 0) == hp, "Hp did not reach the load"
                tables = dpc.full_tables(arm.d, req.request_id)
                content = Content(family, conv, shared, meta.num_tokens, hp=hp)
                rec = arm.land(content, req, meta)
                bad = arm.mismatches(content, meta, tables)
                out[arm_name] = (meta, hit, ext, tables, rec, bad)
                arm.d.decode(req, steps=3)
                arm.d.finish(req)
            (bm, bh, be, bt, brec, bbad), (rm, rh, re_, rt, rrec, rbad) = (
                out["broadcast"], out["routed"]
            )
            assert (bh, be, bm.skip_tokens, bt) == (rh, re_, rm.skip_tokens, rt), "the arms diverged"
            pools_equal = all(
                np.array_equal(a, b)
                for a, b in zip(arms["broadcast"].d.pools, arms["routed"].d.pools)
            )
            hits.append((conv, rm.num_tokens, int(rm.skip_tokens or 0), rrec["straddles"],
                         hp, rrec.get("zero_end", 0), tuple(rrec.get("draft_rows", (0, 0))),
                         tuple(brec.get("draft_rows", (0, 0)))))
            row = {
                "conv": conv, "n": rm.num_tokens, "skip": int(rm.skip_tokens or 0), "hp": hp,
                "zero_end": rrec.get("zero_end"), "draft_rows_routed": rrec.get("draft_rows"),
                "draft_rows_broadcast": brec.get("draft_rows"),
                "adopted": rrec["adopted"], "straddling_spans": rrec["straddles"],
                "broadcast_bad": bbad[:3], "routed_bad": rbad[:3],
                "routed_bad_count": len(rbad), "pools_equal": pools_equal,
                "routed_written_adopted": rrec["written_adopted"],
                "routed_sentinel_changed": rrec["sentinel_changed"],
                "broadcast_written_adopted": brec["written_adopted"],
                "broadcast_sentinel_changed": brec["sentinel_changed"],
            }
            print("Q38_XFER2_LOAD", row, flush=True)
            if bbad or rbad or not pools_equal or rrec["written_adopted"] or brec["written_adopted"] \
                    or rrec["sentinel_changed"] or brec["sentinel_changed"]:
                failures.append(row)
        skips = sorted({h[2] for h in hits})
        gap_loads = [h for h in hits if h[4] > h[2]]
        print("Q38_XFER2_SUMMARY", {"loads": len(hits), "skips": skips,
                                    "hit_loads": sum(1 for h in hits if h[2]),
                                    "gap_loads": len(gap_loads),
                                    "gap_hit_loads": sum(1 for h in gap_loads if h[2]),
                                    "draft_rows_landed_zeroed_routed": [sum(h[6][0] for h in hits), sum(h[6][1] for h in hits)],
                                    "draft_rows_landed_zeroed_broadcast": [sum(h[7][0] for h in hits), sum(h[7][1] for h in hits)],
                                    "failures": len(failures)}, flush=True)
        # The workload must exercise g1p's skip (6,144) and the other
        # entry-aligned skips, gaps on hit loads (Hp > Hd > 0) at each of
        # them, and draft rows both landed and zero-filled on both paths.
        assert {2048, 4096, 6144, 8192} <= set(skips), skips
        assert {h[2] for h in gap_loads if h[2]} == {2048, 4096, 6144, 8192}, gap_loads
        assert all(h[5] == min(max(h[4], 0), h[1]) for h in hits)  # draft_zero_end == min(Hp, n)
        assert all(h[6][1] > 0 and h[7][1] > 0 for h in gap_loads)
        assert all(h[6][1] == 0 for h in hits if h[4] <= h[2])
        assert not failures, failures
    finally:
        dpc.NUM_BLOCKS = old_blocks
