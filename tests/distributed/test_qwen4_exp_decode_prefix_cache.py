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
        self.pools = [np.zeros(NUM_BLOCKS * PAGE, dtype=np.int64) for _ in range(TP)]
        self.landed_rows = 0
        self.serial = 0

    # -- the request lifecycle the scheduler drives --

    def admit(self, conv, prompt):
        """Admit a turn whose producer published ``len(prompt) - 1`` tokens."""
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
        self.sched.update_state_after_alloc(req, self.kvm.get_blocks(rid), ext)
        req.num_computed_tokens = hit + ext
        meta = self.sched.reqs_to_load.pop(rid)
        return req, meta, hit, ext

    def land(self, conv, prompt, req, meta):
        """Every rank stages and repacks the load (``StagedConsumer``)."""
        n = meta.num_tokens
        skip = int(meta.skip_tokens or 0)
        for rank, staging in enumerate(self.ranks):
            assert staging.reserve(meta.uuid, n, skip_tokens=skip)
            staging.destination(meta.uuid)  # alignment checks
            stage = self._produce(conv, prompt, n, staging, meta.uuid)
            pages = staging.reservations[meta.uuid]
            _, first = staging.skips[meta.uuid]
            main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = staging.indices(
                pages, meta.group_block_ids, n, skip, first
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
        for name, owner_pages in pages.items():
            kind = owner_kind(name)
            f = first[staging._src_owner[name]]
            for i, page in enumerate(owner_pages):
                base = page * sb
                if kind == "main":
                    t = (f + i) * sb + np.arange(sb)
                    vals = np.array([main_value(conv, name, x) for x in t])
                    stage[base : base + sb] = np.where(t < n, vals, 0)
                elif kind == "compressed":
                    pairs = (f + i) * sb // 8 + np.arange(sb // 8)
                    stage[base : base + sb // 8] = [
                        pair_value(conv, name, p, n) if 8 * p < n else 0 for p in pairs
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
        return bad


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


def serve(decode, conv, prompt, steps=6):
    req, meta, hit, ext = decode.admit(conv, prompt)
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
