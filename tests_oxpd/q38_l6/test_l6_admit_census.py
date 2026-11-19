# SPDX-License-Identifier: Apache-2.0
"""q38 L6 admission census (host_levers_h2.md section 4.1 row 5): the runner's
~am_stage_bt splits into ~am_pack / ~am_h2d / ~am_walk inside
layers/core/attention_metadata.py (log-only host_phases sub-marks).

Real staging code on a virtual clock: the geometry walk (10 us per group), the
row pack (20 us per BlockTable read), the staged upload (300 us) and the
walk-products issue (150 us) land in exactly their sub-marks on the full pack,
the incremental pack and the reuse path; with the census off nothing is
recorded; the staged bytes are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.layers.core import attention_metadata as am
from vllm_torchtpu.runner import host_phases

WIDTHS = [6, 3, 6, 4]
NREQ = 4


class _Clk:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


class _BT:
    """A BlockTable stand-in whose CPU-tensor read costs 20 us of virtual time."""

    def __init__(self, clk, width, seed):
        self.clk = clk
        self.t = torch.arange(seed, seed + 8 * width, dtype=torch.int32).view(8, width)

    def get_cpu_tensor(self):
        self.clk.tick(20)
        return self.t


class _Builder:
    def __init__(self, clk, gid, bt, width):
        self.clk, self.kv_cache_group_id, self.bt, self.width = clk, gid, bt, width

    def _block_table_geometry(self, ctx):
        self.clk.tick(10)
        return self.bt, self.width


@pytest.fixture
def env(monkeypatch):
    def setup(subs):
        log = []
        for name, v in (("_EVERY", 1), ("_SLOW_NS", 0), ("_SUBS", subs), ("_GC", False), ("_last", None),
                        ("_steps", 0), ("_samples", {}), ("_cur", {}), ("_ctx", []), ("_sub_last", None),
                        ("_sub_acc", {}), ("_sub_samples", {}), ("_rank", 0)):
            monkeypatch.setattr(host_phases, name, v)
        monkeypatch.setattr(host_phases, "logger", NS(info=lambda fmt, *a: log.append(fmt % a)))
        clk = _Clk()
        monkeypatch.setattr(host_phases.time, "perf_counter_ns", clk)
        monkeypatch.setenv("VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", "1")
        monkeypatch.setenv("VLLM_TPU_STAGE_ENTRIES_CACHE", "0")
        monkeypatch.setattr(am, "_STAGE_INC", False)
        monkeypatch.setattr(am, "_STAGE_INC_CHECK", [0])
        monkeypatch.setattr(am, "_STAGE_INC_STATS", {"inc": 0, "full": 0, "rows": 0, "logged": False})

        def upload(src, device):
            clk.tick(300)
            return src.clone()
        monkeypatch.setattr(am, "_inc_upload", upload)
        walks = []

        def walk(runner, ctx, entries, builders, nrp, geom, staged, reused_key=None):
            clk.tick(150)
            walks.append(staged.clone())
        monkeypatch.setattr(am, "_issue_walk_products", walk)
        bts = [_BT(clk, w, 100 * i) for i, w in enumerate(WIDTHS)]
        builders = [_Builder(clk, i, bt, w) for i, (bt, w) in enumerate(zip(bts, WIDTHS))]
        runner = NS(attn_groups=[[NS(metadata_builders=[b])] for b in builders], input_batch=object(),
                    _block_table_stage_cpu=None, device=torch.device("cpu"))
        ctx = NS(use_max_model_len=False, mamba_ckpt_window=None, start_index=0, num_reqs=3,
                 position_ids_override=None)
        return NS(log=log, clk=clk, bts=bts, runner=runner, ctx=ctx, walks=walks)
    return setup


def _expected(bts, ctx):
    parts = []
    for bt, w in zip(bts, WIDTHS):
        rows = torch.zeros((NREQ, w), dtype=torch.int32)
        rows[: ctx.num_reqs] = bt.t[ctx.start_index: ctx.start_index + ctx.num_reqs, :w]
        parts.append(rows.flatten())
    return torch.cat(parts)


def _one_step(e, fn):
    host_phases.begin_step()
    host_phases.mark("before")
    fn()
    host_phases.sub("am_stage_bt")
    host_phases.mark("prep_attention_metadata")
    host_phases.begin_step()
    got = [ln for ln in e.log if ln.startswith("HOSTSUBS") and "~am_" in ln]
    return got


def _subs(line):
    return line.split("(p50/p90/mean-per-step/n-steps) ", 1)[1]


@pytest.mark.parametrize("subs", [True, False])
def test_full_pack_split(env, subs):
    e = env(subs)
    got = _one_step(e, lambda: am.stage_block_table_uploads(e.runner, e.ctx, NREQ))
    assert len(e.walks) == 1 and torch.equal(e.walks[0], _expected(e.bts, e.ctx))
    ph = [ln for ln in e.log if ln.startswith("HOSTPHASES") and "prep_attention_metadata=" in ln]
    assert len(ph) == 1 and "prep_attention_metadata=570/570/570" in ph[-1], e.log
    if not subs:
        assert got == [] and host_phases._sub_acc == {}
        return
    # geometry 4 x 10 + pack 4 x 20 = 120; the upload 300; the walk 150; nothing left for ~am_stage_bt
    assert len(got) == 1 and _subs(got[0]) == (
        "~am_pack=120/120/120/1 ~am_h2d=300/300/300/1 ~am_walk=150/150/150/1 ~am_stage_bt=0/0/0/1"), got


def test_reuse_path_has_no_h2d(env):
    e = env(True)
    _one_step(e, lambda: am.stage_block_table_uploads(e.runner, e.ctx, NREQ))  # first step: full pack + upload
    e.log.clear()
    got = _one_step(e, lambda: am.stage_block_table_uploads(e.runner, e.ctx, NREQ))
    assert len(e.walks) == 2 and torch.equal(e.walks[1], _expected(e.bts, e.ctx))
    # unchanged version and geometry: the previous upload is reused -> geometry walk 40, no pack, no upload
    assert len(got) == 1 and _subs(got[0]) == (
        "~am_pack=40/40/40/1 ~am_walk=150/150/150/1 ~am_stage_bt=0/0/0/1"), got


def test_incremental_pack_split(env, monkeypatch):
    e = env(True)
    monkeypatch.setattr(am, "_STAGE_INC", True)
    entries = [(i, bt, w) for i, (bt, w) in enumerate(zip(e.bts, WIDTHS))]
    geom, off = [], 0
    for _, _, w in entries:
        geom.append((off, NREQ, w))
        off += NREQ * w

    def pack(ver):
        key = (ver, e.ctx.start_index, e.ctx.num_reqs, tuple(geom))
        return am._pack_and_upload(e.runner, e.ctx, entries, NREQ, geom, key)

    _one_step(e, lambda: pack(1))  # full pack: keys the incremental state
    assert am._STAGE_INC_STATS["full"] == 1
    e.bts[2].t[1, :] = 9999
    e.bts[2]._q38_dirty = {1}
    e.log.clear()
    out = []
    got = _one_step(e, lambda: out.append(pack(2)))
    assert am._STAGE_INC_STATS["inc"] == 1 and torch.equal(out[0], _expected(e.bts, e.ctx))
    # one dirty row copied (one 20 us table read), then the upload
    assert len(got) == 1 and _subs(got[0]) == "~am_pack=20/20/20/1 ~am_h2d=300/300/300/1 ~am_stage_bt=0/0/0/1", got
