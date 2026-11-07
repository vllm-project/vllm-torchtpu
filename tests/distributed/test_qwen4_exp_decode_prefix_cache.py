# SPDX-License-Identifier: Apache-2.0
"""Decode-side prefix cache with MTP on the served Qwen3.8 decode (CPU).

The served decode is a staged TP8 consumer of PP8 x TP1 prefill stages, with
main-KV DCP4 (2048-token main-KV entries on 512-token pages), packed
compressed keys and an MTP draft (K=3), so its GDN/PLE state tables carry
three speculative blocks. These tests drive the real vLLM ``KVCacheManager``
(with the engine-core patches the decode installs), the real
``Qwen4ExpRaidenConnector`` scheduler half and the real staging repack row
plan (``StagedConsumer.indices``) for all eight decode ranks, over a
multi-turn conversation, and compare every logical cache row the decode
would read (main KV per DCP rank, packed compressed keys, ring, PLE and the
committed GDN state page) between a decode with prefix caching and one
without. Page contents are emulated: each producer row holds a value
determined by (conversation, cache owner, token or pair, extent), so a row
landed in the wrong page, adopted from the wrong conversation or never
written shows up as a mismatch.
"""

import math
import os
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
TP, DCP, K = 8, 4, 3
PAGE = 512  # decode manager tokens (FP32 SSM at TP8)
ENTRY = DCP * PAGE  # main-KV DCP entry and scheduler block
NUM_BLOCKS = 3000


@pytest.fixture(autouse=True)
def served_decode_env(monkeypatch):
    for name, value in {
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
        "VLLM_TPU_QSA_MAIN_KV_DCP": str(DCP),
        "QWEN4_EXP_PACK_COMPRESSED_K": "1",
        "QWEN4_EXP_PACK_MAIN_KV": "0",
        "TPU_KV_RESHARD_TRANSPORT": "raiden",
        "TPU_RAIDEN_CONTROLLER_ADDRESS": "decode:27000",
        "TPU_RAIDEN_ENGINE_ID": "decode-engine",
        "TPU_RAIDEN_TRANSFER_PARALLELISM": "8",
        "TPU_RAIDEN_POOL_TAGS_PER_LAYER": "1",
        "TPU_RAIDEN_PREFIX_AWARE_LOAD": "1",
        "TPU_RAIDEN_QWEN35_ADMISSION": "0",
        "TPU_RAIDEN_KIMIK3_ADMISSION": "0",
        "TPU_RAIDEN_GLM_ADMISSION": "0",
        "TPU_RAIDEN_DSV4_ADMISSION": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("QWEN4_EXP_PD_DIVERGENT_HITS", raising=False)
    # q38 F2 stays off unless a test asks for it (the draft_kv fixture).
    for name in (
        "QWEN4_EXP_PD_DRAFT_KV",
        "VLLM_TPU_QSA_MTP_KV_DCP",
        "QWEN4_EXP_PD_ZERO_DRAFT_KV",
        "QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP",
        "VLLM_TPU_QSA_COMPRESSED_DCP",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def draft_kv(monkeypatch):
    """q38 F2: the producer ships the MTP draft's KV (QWEN4_EXP_PD_DRAFT_KV=1)
    and this DCP4 decode splits the draft's main KV like a target layer's
    (VLLM_TPU_QSA_MTP_KV_DCP=1)."""
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_MTP_KV_DCP", "1")


@pytest.fixture
def engine_patches():
    """Install the decode engine-core patches; restore vLLM afterwards."""
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    from vllm_torchtpu.core.divergent_hybrid_hits import patch_divergent_hybrid_hits
    from vllm_torchtpu.core.mamba_checkpoint_retention import (
        patch_mamba_checkpoint_retention,
    )

    saved = [
        (cls, name, cls.__dict__[name])
        for cls, names in (
            (
                MambaManager,
                (
                    "cache_blocks",
                    "_remove_blocks_in_range",
                    "remove_skipped_blocks",
                    "pop_blocks_for_free",
                ),
            ),
            (KVCacheCoordinator, ("free", "pop_blocks_for_free")),
            (KVCacheManager, ("get_computed_blocks_for_connector",)),
        )
        for name in names
    ]
    patch_mamba_checkpoint_retention()
    patch_divergent_hybrid_hits()
    yield
    for cls, name, value in saved:
        setattr(cls, name, value)


# -- the served decode's cache plan ---------------------------------------------


def decode_config(mtp=K, prefix_caching=True):
    from tests.models.vllm.test_qwen4_exp_mtp import spec_config
    from tests.runner.test_qwen4_exp_pool import pool_config

    cfg = pool_config(tp=TP, kv="fp8_e4m3", ssm="float32")
    cfg.model_config.max_model_len = 262144
    cfg.cache_config.enable_prefix_caching = prefix_caching
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1
    if mtp:
        cfg.speculative_config = spec_config(mtp)
    return cfg


def scheduler_plan(cfg):
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    from vllm_torchtpu.models.vllm.qwen4_exp.config import ensure_kv_group_size_rule
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    ensure_kv_group_size_rule()  # the platform installs it before planning
    layout, specs = pooled_cache_specs(cfg)
    groups = get_kv_cache_groups(cfg, specs)
    native = get_kv_cache_config_from_groups(cfg, groups, NUM_BLOCKS * layout.page_bytes)
    return layout, native, generate_scheduler_kv_cache_config([native])


def block_sizes(kvc):
    sizes = [g.kv_cache_spec.block_size for g in kvc.kv_cache_groups]
    hashing = [
        g.kv_cache_spec.block_size
        for g in kvc.kv_cache_groups
        if g.kv_cache_spec.prefix_cacheable
    ]
    return math.lcm(*sizes), math.gcd(*hashing)


class Decode:
    """One decode engine: scheduler, connector and 8 ranks' emulated pools."""

    def __init__(self, cfg, *, divergent=True):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        from tests.distributed.test_tpu_connector import _make_vllm_config
        from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
        from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups
        from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
            Qwen4ExpRaidenConnector,
        )

        self.cfg = cfg
        self.layout, native, self.kvc = scheduler_plan(cfg)
        assert self.layout.block_size == PAGE
        sched_block, hash_block = block_sizes(self.kvc)
        if cfg.cache_config.enable_prefix_caching:
            assert (sched_block, hash_block) == (ENTRY, PAGE)
        else:  # vLLM hashes at the scheduler block without align-mode state
            hash_block = sched_block
        self.kvm = KVCacheManager(
            self.kvc,
            max_model_len=262144,
            scheduler_block_size=sched_block,
            hash_block_size=hash_block,
            enable_caching=bool(cfg.cache_config.enable_prefix_caching),
            use_eagle=False,  # VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 (served)
        )
        vcfg = _make_vllm_config(tp_size=TP, is_producer=False)
        vcfg.cache_config.enable_prefix_caching = cfg.cache_config.enable_prefix_caching
        vcfg.speculative_config = cfg.speculative_config
        env = {} if divergent else {"QWEN4_EXP_PD_DIVERGENT_HITS": "0"}
        with (
            patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
            patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
            patch(f"{_MOD}._prefix_aware_load_supported", return_value=True),
            patch.dict("os.environ", env),
        ):
            self.conn = Qwen4ExpRaidenConnector(
                vcfg, KVConnectorRole.SCHEDULER, self.kvc
            )
        self.sched = self.conn.connector_scheduler
        groups = build_transfer_groups(native.kv_cache_groups)
        import torch

        page_bytes = self.layout.page_bytes
        with patch.object(pd_staging, "make_repack", lambda **kw: None):
            self.ranks = []
            for rank in range(TP):
                staging = pd_staging.StagedConsumer(
                    cfg,
                    self.layout,
                    groups,
                    torch.zeros(1, dtype=torch.float8_e4m3fn),
                    pd_staging.RankSlice(
                        tp_size=TP,
                        tp_rank=rank,
                        num_kv_heads=int(cfg.model_config.hf_text_config.num_key_value_heads),
                        main_dcp=DCP,
                    ),
                    staging_bytes=2 * page_bytes * 8,
                )
                staging.pages = 2048
                staging.free = torch.ones(staging.pages, dtype=torch.int32)
                staging.free[0] = 0
                self.ranks.append(staging)
        assert page_bytes == PAGE * self.layout.token_bytes
        self.src_block = self.ranks[0].src_block
        # q38 F2: the producer ships the draft's caches, so read() also
        # checks them (read_draft), from this decode's own plan.
        self.draft_kv = pd_staging.draft_kv_enabled()
        # uuid -> the producer's prefix hit Hp (q38 F2 option (a)): it computed
        # no draft rows below it (_produce stages garbage there).
        self.hp: dict[int, int] = {}
        self.pools = [np.zeros(NUM_BLOCKS * PAGE, dtype=np.int64) for _ in range(TP)]
        self.landed_rows = 0
        self.serial = 0

    # -- the request lifecycle the scheduler drives --

    def admit(self, conv, prompt, producer_hit=0):
        """Admit a turn whose producer published ``len(prompt) - 1`` tokens
        (and, with ``producer_hit``, its prefix hit, as an F2 producer does)."""
        from tests.runner.test_qwen4_exp_offload import _request

        self.serial += 1
        rid = f"{conv}-{self.serial}"
        req = _request(rid, list(prompt))
        n = len(prompt) - 1
        req.kv_transfer_params = {
            "req_id": f"src-{rid}",
            "uuid": 1000 + self.serial,
            "num_tokens": n,
            "src_controller_address": "prefill:27000",
            "src_job_name": "prefill",
            "src_engine_id": "prefill-engine",
            "src_data_replica_idx": 0,
            "src_parallelism": 8,
            "dst_tp_size": TP,
            "dst_dcp_size": 1,
        }
        if producer_hit:
            from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import PRODUCER_HIT_PARAM

            req.kv_transfer_params[PRODUCER_HIT_PARAM] = producer_hit
        self.hp[1000 + self.serial] = producer_hit
        self.sched.on_new_request(req)
        if self.conn.supports_divergent_local_hybrid_hits:
            blocks, hit, _, diverged = self.kvm.get_computed_blocks_for_connector(req)
        else:
            blocks, hit, _ = self.kvm.get_computed_blocks(req)
            diverged = False
        ext, load_async = self.sched.get_num_new_matched_tokens(req, hit)
        if diverged and ext == 0:
            blocks, hit, _ = self.kvm.get_computed_blocks(req)
        assert load_async and ext > 0
        new = self.kvm.allocate_slots(
            req,
            0,
            num_new_computed_tokens=hit,
            new_computed_blocks=blocks,
            num_external_computed_tokens=ext,
            delay_cache_blocks=True,
        )
        assert new is not None
        # Through the connector, as vLLM calls it (it carries the producer's hit).
        self.conn.update_state_after_alloc(req, self.kvm.get_blocks(rid), ext)
        req.num_computed_tokens = hit + ext
        meta = self.sched.reqs_to_load.pop(rid)
        return req, meta, hit, ext

    def land(self, conv, prompt, req, meta):
        """Every rank stages and repacks the load (``StagedConsumer``)."""
        n = meta.num_tokens
        skip = int(meta.skip_tokens or 0)
        for rank, staging in enumerate(self.ranks):
            assert staging.reserve(
                meta.uuid, n, skip_tokens=skip,
                draft_zero_until=int(getattr(meta, "src_cached_tokens", 0) or 0),
            )
            staging.destination(meta.uuid)  # alignment checks
            stage = self._produce(conv, prompt, n, staging, meta.uuid)
            pages = staging.reservations[meta.uuid]
            _, first = staging.skips[meta.uuid]
            if self.draft_kv:
                # The plan the worker runs (it counts the draft rows, draft_stats).
                staging.plan_repack(meta.uuid, meta.group_block_ids, n)
            main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = staging.indices(
                pages, meta.group_block_ids, n, skip, first,
                draft_zero_until=staging.draft_zero_end(meta.uuid, n),
            )
            pool = self.pools[rank]
            # Shared prefix pages adopted from the cache are never written.
            shared = self.adopted_pages(meta)
            written = np.concatenate([main_d, copy_d, gdn_d * PAGE]) // PAGE
            assert not shared.intersection(written.tolist())
            pool[main_d] = stage[main_s]
            pool[copy_d] = stage[copy_s]
            for src, dst in zip(gdn_s, gdn_d):
                # A GDN page carries the rank's head slice of the state.
                pool[dst * PAGE] = stage[src * self.src_block]
            self.landed_rows += len(set(main_d.tolist()))
            staging.release(meta.uuid)
        self.kvm.cache_blocks(req, req.num_computed_tokens)

    def adopted_pages(self, meta):
        skip = int(meta.skip_tokens or 0)
        pages = set()
        for gi, ids in enumerate(meta.group_block_ids):
            spec = self.kvc.kv_cache_groups[gi].kv_cache_spec
            if skip and len(ids) > 1 and spec.prefix_cacheable:
                pages.update(ids[: skip // self.kvm.coordinator.single_type_managers[gi].block_size])
        return pages

    def decode(self, req, steps, accepted=2):
        """MTP verify steps: K drafts each, ``accepted`` kept."""
        if req.num_computed_tokens == req.num_tokens:
            req.num_computed_tokens -= 1
        k = K if self.cfg.speculative_config is not None else 0
        for step in range(steps):
            scheduled = req.num_tokens - req.num_computed_tokens + k
            assert self.kvm.allocate_slots(req, scheduled, num_lookahead_tokens=k)
            keep = min(accepted, k) + 1
            req.num_computed_tokens += keep
            req.append_output_token_ids([900000 + step * 8 + i for i in range(keep)])
            self.kvm.cache_blocks(req, req.num_computed_tokens)

    def finish(self, req):
        self.kvm.free(req)

    # -- emulated contents --

    def _produce(self, conv, prompt, n, staging, uuid):
        """The producer's staged rows for this load (zeros past ``n``)."""
        from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import owner_kind

        sb = self.src_block
        stage = np.zeros(staging.pages * sb, dtype=np.int64)
        pages = staging.reservations[uuid]
        _, first = staging.skips[uuid]
        # F2 option (a): the producer's draft blocks below its hit were never
        # written (not prefix-cacheable there): they hold unrelated bytes.
        hp = self.hp.get(uuid, 0)
        for name, owner_pages in pages.items():
            kind = owner_kind(name)
            f = first[staging._src_owner[name]]
            unwritten = hp if name.startswith("mtp.") else 0
            for i, page in enumerate(owner_pages):
                base = page * sb
                if kind == "main":
                    t = (f + i) * sb + np.arange(sb)
                    vals = np.array([main_value(conv, name, x) for x in t])
                    vals = np.where(t < unwritten, UNWRITTEN, vals)
                    stage[base : base + sb] = np.where(t < n, vals, 0)
                elif kind == "compressed":
                    pairs = (f + i) * sb // 8 + np.arange(sb // 8)
                    stage[base : base + sb // 8] = [
                        (UNWRITTEN if 8 * p < unwritten else pair_value(conv, name, p, n))
                        if 8 * p < n else 0
                        for p in pairs
                    ]
                else:  # ring, PLE, GDN: one state page at the extent
                    stage[base : base + PAGE] = [
                        state_value(conv, name, n, r) for r in range(PAGE)
                    ]
        return stage

    def read(self, rank, conv, prompt, meta, full_tables):
        """Mismatched (owner, position) of everything this rank would read."""
        staging = self.ranks[rank]
        n = meta.num_tokens
        pool = self.pools[rank]
        bad = []
        for name, gi in staging._dst_owner.items():
            kind = name.rsplit(".", 1)[-1]
            table = full_tables[gi]
            if self.draft_kv and name.startswith("mtp."):
                continue  # read_draft, with option (a)'s zero-filled gap
            if name.endswith(".self_attn"):
                for t in range(n):
                    if (t // PAGE) % DCP != staging.rank.dcp_rank:
                        continue
                    row = table[t // ENTRY] * PAGE + t % PAGE
                    if pool[row] != main_value(conv, name, t):
                        bad.append((name, t))
            elif kind == "raw_ring":
                row = table[0] * PAGE
                if pool[row] != state_value(conv, name, n, 0):
                    bad.append((name, "ring"))
        for name, (gi, offset) in staging._dst_compressed.items():
            if self.draft_kv and name.startswith("mtp."):
                continue
            table = full_tables[gi]
            for p in range(math.ceil(math.ceil(n / 4) / 2)):
                t = 8 * p
                row = table[t // PAGE] * PAGE + offset + (t % PAGE) // 8
                if pool[row] != pair_value(conv, name, p, n):
                    bad.append((name, p))
        for gi, ids in enumerate(meta.group_block_ids):
            group = self.kvc.kv_cache_groups[gi]
            names = [x for x in group.layer_names if x in staging._dst_owner]
            if not names or not (
                names[0].endswith(".linear_attn") or names[0].endswith(".ple")
            ):
                continue
            # The landed state is the committed entry, -(K + 1)th of the table.
            k = getattr(group.kv_cache_spec, "num_speculative_blocks", 0)
            if ids != [full_tables[gi][-(k + 1)]]:
                bad.append((names[0], "entry"))
            if pool[ids[0] * PAGE] != state_value(conv, names[0], n, 0):
                bad.append((names[0], "state"))
        if self.draft_kv:
            bad.extend(self.read_draft(rank, conv, meta, full_tables))
        return bad

    def draft_owners(self):
        """The MTP draft's cache owners in this decode's own plan (its
        scheduler groups and packed compressed slots, not the staging's
        view of them): name -> (group, row offset in the group's page)."""
        from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import owner_kind

        group_of = {
            name: gi
            for gi, group in enumerate(self.kvc.kv_cache_groups)
            for name in group.layer_names
        }
        owners = {
            name: (gi, 0)
            for name, gi in group_of.items()
            if name.startswith("mtp.") and owner_kind(name) != "compressed"
        }
        if self.layout.compressed_owners:
            member_rows = PAGE // 8  # pair rows of one member (ratio 4)
            for prefix, owner, slot in self.layout.compressed_owners:
                if prefix.startswith("mtp."):
                    owners[prefix + ".compressed_k"] = (group_of[owner], slot * member_rows)
        else:
            owners.update(
                (name, (gi, 0))
                for name, gi in group_of.items()
                if name.startswith("mtp.") and owner_kind(name) == "compressed"
            )
        return owners

    def draft_gap(self, meta):
        """[lo, hi): the draft tokens this load must read as zero (q38 F2
        option (a)), derived here from the load alone: above the decode's
        local hit, below the producer's hit raised to the forced gap
        (QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP of the tokens, down to a pair)."""
        n = meta.num_tokens
        lo = int(meta.skip_tokens or 0)
        fraction = float(os.environ.get("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP") or 0)
        hi = min(max(self.hp.get(meta.uuid, 0), int(fraction * n) // 8 * 8), n)
        return (lo, max(lo, hi)) if self.draft_kv else (lo, lo)

    def read_draft(self, rank, conv, meta, full_tables):
        """Mismatched draft rows this rank would read (q38 F2): main KV per
        DCP rank at its group's entry size, the packed compressed slot, and
        every ring row the repack copies, against the producer's values."""
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
                split = entry // PAGE  # DCP ranks sharing one table entry
                for t in range(n):
                    if split > 1 and (t // PAGE) % split != staging.rank.dcp_rank:
                        continue
                    row = table[t // entry] * PAGE + t % PAGE
                    want = 0 if lo <= t < hi else main_value(conv, name, t)
                    if pool[row] != want:
                        bad.append((name, t))
            elif kind == "compressed":
                for p in range(math.ceil(math.ceil(n / 4) / 2)):
                    t = 8 * p
                    row = table[t // PAGE] * PAGE + offset + (t % PAGE) // 8
                    want = 0 if lo <= t < hi else pair_value(conv, name, p, n)
                    if pool[row] != want:
                        bad.append((name, p))
            else:
                assert kind == "ring", name
                for r in range(PAGE):
                    if pool[table[0] * PAGE + r] != state_value(conv, name, n, r):
                        bad.append((name, "ring", r))
        return bad


# What an F2 producer's never-written draft rows hold in this emulation.
UNWRITTEN = -0x5EED


def main_value(conv, owner, t):
    return hash((conv, owner, "kv", int(t))) & 0x7FFFFFFFFFFF


def pair_value(conv, owner, pair, n):
    # A partial compression group differs from the complete one.
    return hash((conv, owner, "ck", int(pair), min(8, n - 8 * int(pair)))) & 0x7FFFFFFFFFFF


def state_value(conv, owner, n, row):
    return hash((conv, owner, "state", int(n), int(row))) & 0x7FFFFFFFFFFF


def tokens(count, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 200000, size=count).tolist()


def full_tables(decode, rid):
    return [list(ids) for ids in decode.kvm.get_block_ids(rid)]


def serve(decode, conv, prompt, steps=6, producer_hit=0):
    req, meta, hit, ext = decode.admit(conv, prompt, producer_hit)
    tables = full_tables(decode, req.request_id)
    decode.land(conv, prompt, req, meta)
    return req, meta, hit, ext, tables


def logical_mismatches(decode, conv, prompt, meta, tables):
    return [decode.read(rank, conv, prompt, meta, tables) for rank in range(TP)]


# -- tests ----------------------------------------------------------------------


def test_served_decode_plan_has_speculative_align_state_tables():
    from vllm.v1.kv_cache_interface import MambaSpec

    _, _, kvc = scheduler_plan(decode_config())
    state = [g for g in kvc.kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)]
    assert len(state) == 37  # 36 GDN + 1 PLE
    assert {g.kv_cache_spec.mamba_cache_mode for g in state} == {"align"}
    assert {g.kv_cache_spec.num_speculative_blocks for g in state} == {K}
    assert block_sizes(kvc) == (ENTRY, PAGE)


@pytest.mark.usefixtures("engine_patches")
def test_multi_turn_decode_hits_land_only_the_suffix_and_read_identically():
    """Cache on vs off: identical logical rows, hits at 2048 boundaries."""
    on = Decode(decode_config())
    off = Decode(decode_config(prefix_caching=False))
    conv = "conv-a"
    prompt = tokens(20000, 0)
    hits = []
    for turn in range(3):
        results = {}
        for name, decode in (("on", on), ("off", off)):
            req, meta, hit, ext, tables = serve(decode, conv, prompt)
            results[name] = (meta, hit, ext, tables)
            assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP
            decode.decode(req, steps=12)
            decode.finish(req)
        meta, hit, ext, tables = results["on"]
        off_meta, off_hit, off_ext, _ = results["off"]
        n = len(prompt) - 1
        assert (off_hit, off_ext, off_meta.skip_tokens) == (0, n, 0)
        hits.append(hit)
        if turn:
            # The previous prompt's whole 2048-token entries are reused; the
            # entry holding its last token mixes in this turn's new tokens.
            assert hit == previous // ENTRY * ENTRY
            assert meta.skip_tokens == hit and ext == n - hit
        previous = len(prompt)
        # The next turn: the trace's assistant reply and user message, which
        # differ from the tokens this decode generated.
        prompt = prompt + tokens(3000 + 700 * turn, 10 + turn)
    assert hits[0] == 0 and all(h > 0 for h in hits[1:])
    assert on.landed_rows < off.landed_rows


def victim_group(kvc, kind):
    """A token group of one kind: a target main-KV group (2048-token DCP
    entries), a packed compressed-key group, or the MTP draft's main KV."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

    for gid, group in enumerate(kvc.kv_cache_groups):
        spec, name = group.kv_cache_spec, group.layer_names[0]
        if kind == "compressed" and isinstance(spec, MLAAttentionSpec):
            return gid
        if isinstance(spec, MLAAttentionSpec) or not isinstance(
            spec, FullAttentionSpec
        ):
            continue
        if kind == "main" and spec.block_size == ENTRY:
            return gid
        if kind == "draft" and name.startswith("mtp."):
            return gid
    raise AssertionError(kind)


@pytest.mark.usefixtures("engine_patches")
@pytest.mark.parametrize("kind", ["main", "compressed", "draft"])
def test_hit_is_the_shortest_token_group_hit_after_partial_eviction(kind):
    """One token group lost a tail block: no group reads unwritten rows."""
    decode = Decode(decode_config())
    conv = "conv-b"
    prompt = tokens(20000, 1)
    req, meta, _, _, _ = serve(decode, conv, prompt)
    decode.decode(req, steps=4)
    victim = victim_group(decode.kvc, kind)
    manager = decode.kvm.coordinator.single_type_managers[victim]
    # The block holding the last token of the second-to-last cached entry.
    lost = ((len(prompt) - 1) // ENTRY - 1) * ENTRY
    tail = manager.req_to_blocks[req.request_id][
        (lost + ENTRY - 1) // manager.block_size
    ]
    decode.finish(req)
    assert decode.kvm.block_pool._maybe_evict_cached_block(tail)
    prompt = prompt + tokens(4000, 2)
    req, meta, hit, ext, tables = serve(decode, conv, prompt)
    assert hit == lost
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP


def pd_consumer_mtp_config(prefix_caching, mode, role="kv_consumer"):
    from tests.runner.test_qwen4_exp_offload import offload_config
    from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig

    cfg = offload_config()  # TP8/EP8, FP8 KV, MTP K=3
    cfg.cache_config.enable_prefix_caching = prefix_caching
    cfg.cache_config.mamba_cache_mode = mode
    cfg.cache_config.mamba_ssm_cache_dtype = "float32"
    cfg.kv_transfer_config = NS(
        kv_connector=qconfig.PD_CONNECTOR,
        kv_connector_module_path=None,
        kv_role=role,
        is_kv_producer=role == "kv_producer",
    )
    return cfg


@pytest.mark.parametrize(
    "prefix_caching,mode", [(True, "align"), (False, "align"), (False, "none")]
)
def test_admits_mtp_on_a_staged_decode_with_or_without_prefix_caching(
    prefix_caching, mode
):
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    validate_config(pd_consumer_mtp_config(prefix_caching, mode))


def test_still_rejects_mtp_on_a_producer_and_caching_without_align():
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    with pytest.raises(ValueError, match="staged TP>1 decode"):
        validate_config(pd_consumer_mtp_config(True, "align", role="kv_producer"))
    with pytest.raises(ValueError, match="mamba_cache_mode='align'"):
        validate_config(pd_consumer_mtp_config(True, "none"))


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 511, 512, 513, 19999])
def test_speculative_state_tables_transfer_only_the_committed_entry(
    positional, num_tokens
):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
        _Stage3TransferGroup,
    )

    group = _Stage3TransferGroup(
        cache_group_index=0,
        page_tokens=PAGE,
        request_state=True,
        positional_state=positional,
        speculative_blocks=K,
    )
    columns = (num_tokens + PAGE - 1) // PAGE if positional else 1
    table = [0] * (columns - 1) + [101, 102, 103, 104]
    assert group.transfer_pages(table, num_tokens) == [101]
    # A table that is not that prefix (a column short or long) is refused.
    assert group.transfer_pages(table[1:] if positional else table[:-1], num_tokens) is None
    assert group.transfer_pages([0] + table, num_tokens) is None


@pytest.mark.usefixtures("engine_patches")
def test_a_local_hit_never_covers_the_whole_transfer():
    """At least one token always comes from the producer with the state."""
    decode = Decode(decode_config())
    conv = "conv-c"
    prompt = tokens(4 * ENTRY + 1, 3)  # publishes exactly 4 entries
    req, _, _, _, _ = serve(decode, conv, prompt)
    decode.decode(req, steps=2)
    decode.finish(req)
    req, meta, hit, ext, tables = serve(decode, conv, list(prompt))
    assert (hit, ext) == (3 * ENTRY, ENTRY)
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP


@pytest.mark.usefixtures("engine_patches")
def test_other_conversations_never_hit_and_the_off_switch_restores_state_hits():
    decode = Decode(decode_config(), divergent=False)
    assert not decode.conn.supports_divergent_local_hybrid_hits
    prompt = tokens(20000, 4)
    req, *_ = serve(decode, "conv-d", prompt)
    decode.decode(req, steps=4)
    decode.finish(req)
    # Without a GDN/PLE state saved at a 2048 boundary (the landed state sits
    # at prompt - 1), a state-requiring hit never happens.
    _, _, hit, _, _ = serve(decode, "conv-d", prompt + tokens(3000, 5))
    assert hit == 0
    divergent = Decode(decode_config())
    req, *_ = serve(divergent, "conv-e", prompt)
    divergent.finish(req)
    _, meta, hit, _, tables = serve(divergent, "conv-f", tokens(23000, 6))
    assert hit == 0 and meta.skip_tokens == 0


@pytest.mark.usefixtures("engine_patches")
@pytest.mark.parametrize("zero", [False, True])
def test_zero_draft_kv_option_clears_only_the_landed_draft_rows(monkeypatch, zero):
    """QWEN4_EXP_PD_ZERO_DRAFT_KV=1: the draft's landed rows read zero."""
    if zero:
        monkeypatch.setenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", "1")
    decode = Decode(decode_config())
    stale = -7
    for pool in decode.pools:
        pool[:] = stale
    conv = "conv-z"
    prompt = tokens(9000, 7)
    req, _, _, _, _ = serve(decode, conv, prompt)
    decode.decode(req, steps=2)
    decode.finish(req)
    prompt = prompt + tokens(5000, 8)
    req, meta, hit, ext, tables = serve(decode, conv, prompt)
    assert hit == (9000 - 1) // ENTRY * ENTRY
    draft = victim_group(decode.kvc, "draft")
    n = meta.num_tokens
    for rank in range(TP):
        pool = decode.pools[rank]
        rows = [tables[draft][t // PAGE] * PAGE + t % PAGE for t in range(n)]
        values = pool[rows]
        # Rows the first turn landed (adopted now) and this turn's suffix.
        assert (values == (0 if zero else stale)).all()
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP


@pytest.mark.usefixtures("engine_patches")
def test_staged_clips_count_destination_pages_by_live_bytes():
    """dst_skip_bytes must be a whole multiple of the destination page's live
    bytes (Raiden controller): a trimmed compressed-key page counts 409,600."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import compressed_live_bytes
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import owner_kind

    decode = Decode(decode_config())
    staging = decode.ranks[0]
    live = compressed_live_bytes(staging.src_layout)
    assert live == 409_600 and staging.src_layout.page_bytes == 3_276_800
    skip = 5 * ENTRY  # 10,240 tokens: 3 whole 3,200-token source pages
    assert staging.reserve(77, 40_000, skip_tokens=skip)
    clips = staging.skip_bytes(77)
    for pool, clip in zip(staging.manifest.pools, clips):
        kind = owner_kind(pool.layer_name)
        if kind == "compressed":
            assert clip == 3 * live
        elif kind == "main":
            assert clip == 3 * pool.block_stride_bytes
        else:
            assert clip == 0


# -- q38 F2: the producer ships the MTP draft's KV (QWEN4_EXP_PD_DRAFT_KV) -------
#
# The producer's last stage runs prefill-only MTP at K = 1 and ships the draft
# layer's main KV, compressed keys and ring like any QSA layer's. The staged
# decode rebuilds that plan (pd_staging.producer_geometry), stages and repacks
# the draft's pages with the target's, and its draft reads the producer's rows
# instead of unwritten ones. _produce emits every owner of the rebuilt plan;
# read() then also checks the draft from the decode's own plan (read_draft).

_DRAFT = "mtp.layers.48.self_attn"


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_ships_the_draft_and_every_rank_reads_the_producer_rows():
    """Byte identity of the draft's pages from producer to decode rows: main
    KV per DCP rank (split like a target layer's), its packed compressed slot
    and its whole ring, on all 8 ranks."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import owner_kind
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    decode = Decode(decode_config())
    assert decode.draft_kv
    # The decode splits the draft's main KV (VLLM_TPU_QSA_MTP_KV_DCP=1).
    assert _DRAFT in decode.layout.main_dcp_owners
    # The staging source plan is a K=1 MTP producer's (TP1 stages, no DCP,
    # unpacked compressed keys), built here from its own config.
    _, producer = pooled_cache_specs(
        decode_config(mtp=1), tp=1, main_dcp=1, pack_compressed=False
    )
    assert {_DRAFT, _DRAFT + ".compressed_k", _DRAFT + ".raw_ring"} <= set(producer)
    for staging in decode.ranks:
        assert set(staging._src_owner) == set(producer)
        assert {pool.layer_name for pool in staging.manifest.pools} == set(producer)
    owners = decode.draft_owners()
    assert sorted(owner_kind(name) for name in owners) == ["compressed", "main", "ring"]
    conv, prompt = "conv-f2", tokens(20000, 11)
    req, meta, hit, ext, tables = serve(decode, conv, prompt)
    assert (hit, meta.skip_tokens) == (0, 0)
    assert [decode.read_draft(r, conv, meta, tables) for r in range(TP)] == [[]] * TP
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_multi_turn_hits_land_only_the_suffix_and_read_identically():
    """The multi-turn prefix-caching run with the draft's KV shipped: every
    turn's logical rows, the draft's included, are right on a caching and a
    non-caching decode, and hits still land only the suffix."""
    test_multi_turn_decode_hits_land_only_the_suffix_and_read_identically()


@pytest.mark.usefixtures("engine_patches", "draft_kv")
@pytest.mark.parametrize("kind", ["main", "compressed", "draft"])
def test_draft_kv_hit_after_partial_eviction_reads_no_unwritten_draft_rows(kind):
    """With the draft's rows landed, a token group (the draft's main KV among
    them) that lost a tail block still bounds the hit, and every rank reads
    the producer's draft rows."""
    test_hit_is_the_shortest_token_group_hit_after_partial_eviction(kind)


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_refuses_zero_draft_kv(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_ZERO_DRAFT_KV", "1")
    with pytest.raises(ValueError, match="QWEN4_EXP_PD_ZERO_DRAFT_KV=1 would zero"):
        Decode(decode_config())


@pytest.mark.usefixtures("engine_patches")
def test_draft_kv_refuses_a_dcp_decode_without_mtp_kv_dcp(monkeypatch):
    """Without VLLM_TPU_QSA_MTP_KV_DCP the draft's main KV is not DCP-split on
    this DCP4 decode, and the repack would slice it as if it were."""
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    with pytest.raises(ValueError, match="needs VLLM_TPU_QSA_MTP_KV_DCP=1") as err:
        Decode(decode_config())
    assert _DRAFT in str(err.value)


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_refuses_a_producer_plan_without_draft_owners(monkeypatch):
    """A producer plan with no draft pages (as without MTP on the producer)
    may not leave the decode's draft owners unlanded."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging

    monkeypatch.setattr(
        pd_staging,
        "producer_geometry",
        lambda cfg: (pd_staging.without_speculation(cfg), 1, 1, False, True),
    )
    with pytest.raises(ValueError, match="no pages for") as err:
        Decode(decode_config())
    assert _DRAFT in str(err.value) and _DRAFT + ".raw_ring" in str(err.value)


@pytest.mark.usefixtures("draft_kv")
def test_draft_kv_refuses_a_decode_without_mtp():
    """No draft on this engine: nothing would land the producer's draft pages."""
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups

    cfg = decode_config()
    layout, native, _ = scheduler_plan(cfg)
    cfg.speculative_config = None
    with pytest.raises(ValueError, match="MTP speculative config"):
        pd_staging.StagedConsumer(
            cfg,
            layout,
            build_transfer_groups(native.kv_cache_groups),
            torch.zeros(1, dtype=torch.float8_e4m3fn),
            pd_staging.RankSlice(
                tp_size=TP,
                tp_rank=0,
                num_kv_heads=int(cfg.model_config.hf_text_config.num_key_value_heads),
                main_dcp=DCP,
            ),
            staging_bytes=2 * layout.page_bytes * 8,
        )


@pytest.mark.usefixtures("engine_patches")
def test_flag_off_producer_plan_has_no_draft_owners():
    """q38 F2 off: the staging source plan (the producer's) has no mtp. owner,
    and the decode lands none (its draft's prompt KV starts empty)."""
    decode = Decode(decode_config())
    assert not decode.draft_kv
    for staging in decode.ranks:
        assert not [n for n in staging._src_owner if n.startswith("mtp.")]
        assert not [p.layer_name for p in staging.manifest.pools if p.layer_name.startswith("mtp.")]
        assert not [n for n in staging._dst_owner if n.startswith("mtp.")]
        assert not [n for n in staging._dst_compressed if n.startswith("mtp.")]


# -- q38 F2 option (a): the producer never caches draft KV ------------------------
#
# The producer's draft groups are not prefix-cacheable, so it computes draft rows
# only from its own prefix hit Hp and publishes Hp
# (pd_staging.PRODUCER_HIT_PARAM). A decode with local hit Hd keeps its own draft
# rows in [0, Hd), lands the producer's from max(Hd, Hp), and zero-fills [Hd, Hp)
# when Hp > Hd. _produce stages UNWRITTEN below Hp; read_draft expects zeros in
# Decode.draft_gap, which it derives from the load alone.


@pytest.mark.usefixtures("engine_patches", "draft_kv")
@pytest.mark.parametrize("case", ["above", "below"])
def test_draft_kv_producer_hit_above_the_decode_hit_zero_fills_the_gap(case):
    """Draft rows in [Hd, Hp) read zero, [Hp, N) the producer's values and
    [0, Hd) the decode's own cached rows; a producer hit below the decode's
    zero-fills nothing."""
    decode = Decode(decode_config())
    conv = "conv-hp"
    prompt = tokens(20000, 21)
    req, meta, hit, _, _ = serve(decode, conv, prompt)
    assert (hit, meta.src_cached_tokens) == (0, 0)
    decode.decode(req, steps=4)
    decode.finish(req)
    hd = len(prompt) // ENTRY * ENTRY  # the decode's own hit: 18,432
    prompt = prompt + tokens(3000, 22)
    n = len(prompt) - 1
    # A producer hit on its 3200-token page boundary above Hd, or below Hd.
    hp = n // 3200 * 3200 if case == "above" else hd - ENTRY
    req, meta, hit, _, tables = serve(decode, conv, prompt, producer_hit=hp)
    assert hit == meta.skip_tokens == hd
    assert meta.src_cached_tokens == hp  # carried from the transfer params
    gap = (hd, hp) if case == "above" else (hd, hd)
    assert decode.draft_gap(meta) == gap and (case == "below" or hp - hd == 3968)
    assert [decode.read_draft(r, conv, meta, tables) for r in range(TP)] == [[]] * TP
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP
    stats = decode.ranks[0].draft_stats()
    assert stats["loads"] == 2 and stats["hp_minus_hd"] == hp - hd
    assert (stats["gap_loads"], stats["gap_tokens"]) == ((1, hp - hd) if case == "above" else (0, 0))
    assert (stats["rows_zeroed"] > 0) == (case == "above") and stats["rows_landed"] > 0


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_forced_gap_zero_fills_exactly_the_requested_span(monkeypatch):
    """QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP=0.3 with no producer hit: the draft rows
    of the first 5,992 of 19,999 tokens (int(0.3 n) down to a pair) read zero,
    the rest the producer's values."""
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", "0.3")
    decode = Decode(decode_config())
    conv, prompt = "conv-gap", tokens(20000, 23)
    req, meta, hit, _, tables = serve(decode, conv, prompt)
    assert (hit, meta.num_tokens) == (0, 19999)
    assert decode.draft_gap(meta) == (0, 5992)
    assert [decode.read_draft(r, conv, meta, tables) for r in range(TP)] == [[]] * TP
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP
    stats = decode.ranks[0].draft_stats()
    assert (stats["gap_loads"], stats["gap_tokens"]) == (1, 5992) and stats["rows_zeroed"] > 0


@pytest.mark.usefixtures("engine_patches")
def test_flag_off_ignores_a_published_producer_hit():
    """Without QWEN4_EXP_PD_DRAFT_KV a published hit changes nothing: it is not
    carried, nothing is zero-filled, every target row reads as before."""
    decode = Decode(decode_config())
    conv, prompt = "conv-off", tokens(20000, 24)
    req, meta, hit, _, tables = serve(decode, conv, prompt, producer_hit=12800)
    assert meta.src_cached_tokens == 0
    assert decode.ranks[0].draft_stats() == {}
    assert logical_mismatches(decode, conv, prompt, meta, tables) == [[]] * TP


@pytest.mark.usefixtures("engine_patches", "draft_kv")
def test_draft_kv_warns_when_the_decode_prefix_cache_is_off(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging

    seen = []
    monkeypatch.setattr(
        pd_staging,
        "logger",
        NS(warning=lambda msg, *a: seen.append(msg % a if a else msg), info=lambda *a, **k: None),
    )
    want = "zero-fills draft KV on every producer prefix hit when the decode prefix cache is off"
    Decode(decode_config(prefix_caching=False))
    assert sum(want in m for m in seen) == TP  # one per rank; a warning, not a refusal
    seen.clear()
    Decode(decode_config())
    assert not any(want in m for m in seen)


@pytest.mark.usefixtures("draft_kv")
def test_draft_kv_refuses_compressed_dcp(monkeypatch):
    """VLLM_TPU_QSA_COMPRESSED_DCP splits compressed keys over DCP ranks; the
    staging repack lands them unsplit."""
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups

    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP", "1")
    cfg = decode_config()
    layout, native, _ = scheduler_plan(cfg)
    assert any(name.endswith(".compressed_k") for name in layout.main_dcp_owners)
    with (
        patch.object(pd_staging, "make_repack", lambda **kw: None),
        pytest.raises(ValueError, match="VLLM_TPU_QSA_COMPRESSED_DCP=1"),
    ):
        pd_staging.StagedConsumer(
            cfg,
            layout,
            build_transfer_groups(native.kv_cache_groups),
            torch.zeros(1, dtype=torch.float8_e4m3fn),
            pd_staging.RankSlice(
                tp_size=TP,
                tp_rank=0,
                num_kv_heads=int(cfg.model_config.hf_text_config.num_key_value_heads),
                main_dcp=DCP,
            ),
            staging_bytes=2 * layout.page_bytes * 8,
        )
