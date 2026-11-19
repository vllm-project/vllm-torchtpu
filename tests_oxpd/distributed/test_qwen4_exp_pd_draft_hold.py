# SPDX-License-Identifier: Apache-2.0
"""q38 F2-HOLD (QWEN4_EXP_PD_DRAFT_HOLD): the F2 producer's draft hold, on the
REAL vLLM KVCacheManager over the served producer plan (PP8 x TP1 stages, the
MTP draft's main KV, compressed keys and ring in groups of their own, not
prefix-cacheable) and the REAL Qwen PD connector scheduler half.

- Draft blocks: with the hold, the draft's main-KV and compressed-key tables
  open with floor(Hp / 3200) null blocks and allocate only above; without it
  they allocate the whole prompt. The ring keeps its block. The pool's free
  count differs by exactly 2 * floor(Hp / 3200) per running request and
  returns to its baseline when the requests are freed.
- The admission gate (full-sequence must fit) counts the same.
- Preemption and resume: the resumed request skips its resumed hit, which is
  the hit the producer publishes.
- The producer's finish accepts the held tables, keeps the null blocks in the
  send's tables, and publishes the draft skip next to the hit.

The decode side (bytes landed with the hold on equal the hold off, broadcast
and routed) is test_qwen4_exp_pd_draft_hold_landing.py.
"""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed.test_qwen4_exp_decode_prefix_cache import (  # noqa: F401
    draft_kv,
    engine_patches,
    served_decode_env,
)

P = 3200  # the producer's page tokens (TP1 stage)
NUM_BLOCKS = 4000
_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


@pytest.fixture
def hold(monkeypatch):
    """QWEN4_EXP_PD_DRAFT_HOLD=1 with the manager registered for the test's
    duration (the engine-core patch, core/producer_draft_hold)."""
    from vllm_torchtpu.core import producer_draft_hold as pdh

    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    assert pdh.register_producer_draft_hold()
    yield
    pdh.unregister_producer_draft_hold()


def producer_plan(cfg):
    """The F2 producer's scheduler plan, as its decode rebuilds it
    (pd_staging.producer_geometry): whole-model groups, draft not cacheable."""
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.config import ensure_kv_group_size_rule
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    ensure_kv_group_size_rule()
    layout_cfg, tp, dcp, pack, cacheable = pd_staging.producer_geometry(cfg)
    layout, specs = pooled_cache_specs(
        layout_cfg, tp=tp, main_dcp=dcp, pack_compressed=pack, draft_prefix_cacheable=cacheable
    )
    assert layout.block_size == P
    groups = get_kv_cache_groups(cfg, dict(specs))
    native = get_kv_cache_config_from_groups(cfg, groups, NUM_BLOCKS * layout.page_bytes)
    return generate_scheduler_kv_cache_config([native])


class Producer:
    """The F2 producer's scheduler: the real KVCacheManager, chunked prefill."""

    def __init__(self, cfg, eagle_drop=False):
        """``eagle_drop``: vLLM's EAGLE/MTP block drop on (the F2 producer
        without VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1: ``use_eagle=True``); off,
        the manager is built as core/eagle_block_drop builds it (use_eagle=False,
        every drop flag cleared)."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        from vllm_torchtpu.core.eagle_block_drop import clear_eagle_block_drop

        self.kvc = producer_plan(cfg)
        sched_block, hash_block = dpc.block_sizes(self.kvc)
        assert (sched_block, hash_block) == (P, P)
        self.kvm = KVCacheManager(
            self.kvc,
            max_model_len=262144,
            scheduler_block_size=sched_block,
            hash_block_size=hash_block,
            enable_caching=True,
            use_eagle=eagle_drop,
        )
        if not eagle_drop:
            clear_eagle_block_drop(self.kvm)
        groups = self.kvc.kv_cache_groups
        self.draft = {
            g.layer_names[0]: i for i, g in enumerate(groups) if g.layer_names[0].startswith("mtp.")
        }
        self.main = self.draft["mtp.layers.48.self_attn"]
        self.comp = self.draft["mtp.layers.48.self_attn.compressed_k"]
        self.ring = self.draft["mtp.layers.48.self_attn.raw_ring"]
        self.serial = 0

    def request(self, prompt):
        from vllm import SamplingParams
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.request import Request

        init_none_hash(sha256)
        self.serial += 1
        return Request(
            f"p{self.serial}",
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(P, sha256),
        )

    def admit(self, req, first_chunk=P):
        """The first step: the local hit and the first chunk (None if it does
        not fit)."""
        self.kvm.new_step_starts()  # one scheduler step per call, as vLLM's schedule()
        blocks, hit, _ = self.kvm.get_computed_blocks(req)
        step = min(first_chunk, req.num_tokens - hit)
        new = self.kvm.allocate_slots(
            req, step, num_new_computed_tokens=hit, new_computed_blocks=blocks
        )
        if new is None:
            return None
        req.num_computed_tokens = hit + step
        return hit

    def run(self, req, chunk=P, until=None):
        until = req.num_tokens if until is None else until
        while req.num_computed_tokens < until:
            step = min(chunk, until - req.num_computed_tokens)
            self.kvm.new_step_starts()
            assert self.kvm.allocate_slots(req, step) is not None
            req.num_computed_tokens += step

    def prefill(self, req):
        hit = self.admit(req)
        assert hit is not None
        self.run(req)
        return hit

    def tables(self, req):
        return [list(ids) for ids in self.kvm.get_block_ids(req.request_id)]

    def free(self):
        return self.kvm.block_pool.get_num_free_blocks()


def _nulls(ids):
    n = 0
    for b in ids:
        if b:
            break
        n += 1
    return n


def _shared_prompt(pages, extra, seed):
    base = dpc.tokens(pages * P, 500 + seed)
    return base, base + dpc.tokens(extra, 900 + seed)


@pytest.mark.usefixtures("draft_kv")
@pytest.mark.parametrize("on", [False, True], ids=["hold_off", "hold_on"])
def test_draft_blocks_are_allocated_above_the_hit_only_and_the_pool_returns(on, monkeypatch, request):
    from vllm_torchtpu.core import producer_draft_hold as pdh

    if on:
        request.getfixturevalue("hold")
    cfg = dpc.decode_config()
    prod = Producer(cfg)
    managers = prod.kvm.coordinator.single_type_managers
    held_cls = pdh.producer_draft_hold_manager()
    assert (type(managers[prod.main]) is held_cls) == on
    assert (type(managers[prod.comp]) is held_cls) == on
    assert type(managers[prod.ring]) is not held_cls  # the ring keeps its one block
    assert all(type(m) is not held_cls for i, m in enumerate(managers) if i not in (prod.main, prod.comp))

    baseline = prod.free()
    # A seed of 3 whole pages plus a tail: no hit, every draft page allocated.
    _, seed_prompt = _shared_prompt(3, 700, 1)
    seed = prod.request(seed_prompt)
    assert prod.prefill(seed) == 0
    t = prod.tables(seed)
    assert [len(t[prod.main]), _nulls(t[prod.main])] == [4, 0]
    assert [len(t[prod.comp]), _nulls(t[prod.comp])] == [4, 0]
    prod.kvm.free(seed)  # its target pages stay cached

    # Same 3-page prefix, a new suffix: a 3-page hit.
    rows, running = [], []
    for extra, seedno in ((2000, 1), (4500, 1)):
        base, prompt = _shared_prompt(3, extra, seedno)
        req = prod.request(prompt)
        running.append(req)
        before = prod.free()
        hit = prod.prefill(req)
        used = before - prod.free()
        t = prod.tables(req)
        n_pages = -(-len(prompt) // P)
        held = hit // P if on else 0
        for g in (prod.main, prod.comp):
            assert len(t[g]) == n_pages
            assert _nulls(t[g]) == held
            assert all(t[g][held:]), "only the leading entries may be null"
        assert len(t[prod.ring]) == 1 and t[prod.ring][0]
        # The draft's live blocks belong to it alone.
        others = {b for i, ids in enumerate(t) if i not in (prod.main, prod.comp) for b in ids if b}
        assert not others.intersection(t[prod.main][held:] + t[prod.comp][held:])
        rows.append((hit, used, n_pages, held))
    print("Q38_F2HOLD_ALLOC", {"on": on, "rows": rows}, flush=True)
    assert all(hit == 3 * P for hit, *_ in rows), rows  # the prefix hit the test relies on
    for req in running:
        prod.kvm.free(req)
    assert prod.free() == baseline, "the pool's free count returns to its baseline"


@pytest.mark.usefixtures("draft_kv")
def test_the_hold_saves_exactly_two_blocks_per_held_page(monkeypatch, request):
    """Hold on against hold off, same requests: the pool difference while the
    requests run is 2 * floor(Hp / 3200) per request (draft main KV and
    compressed keys), and the admission gate counts the same."""
    from vllm_torchtpu.core import producer_draft_hold as pdh

    cfg = dpc.decode_config()
    used = {}
    gate = {}
    for on in (False, True):
        if on:
            monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
            pdh.register_producer_draft_hold()
        try:
            prod = Producer(cfg)
            _, seed_prompt = _shared_prompt(4, 100, 3)
            seed = prod.request(seed_prompt)
            prod.prefill(seed)
            prod.kvm.free(seed)
            before = prod.free()
            hits = []
            for extra in (1000, 2500, 6100):
                _, prompt = _shared_prompt(4, extra, 3)
                req = prod.request(prompt)
                hits.append(prod.prefill(req))
            used[on] = (before - prod.free(), hits)
            # The full-sequence admission gate's count for one more such request.
            _, prompt = _shared_prompt(4, 3000, 3)
            req = prod.request(prompt)
            prod.kvm.new_step_starts()
            blocks, hit, _ = prod.kvm.get_computed_blocks(req)
            gate[on] = (
                prod.kvm.coordinator.get_num_blocks_to_allocate(
                    request_id=req.request_id, num_tokens=req.num_tokens,
                    new_computed_blocks=blocks.blocks, num_encoder_tokens=0,
                    total_computed_tokens=hit, num_local_computed_tokens=hit,
                    num_tokens_main_model=req.num_tokens, apply_admission_cap=True,
                ),
                hit,
            )
        finally:
            if on:
                pdh.unregister_producer_draft_hold()
    print("Q38_F2HOLD_SAVING", {"used": used, "gate": gate}, flush=True)
    (off_used, off_hits), (on_used, on_hits) = used[False], used[True]
    assert off_hits == on_hits == [4 * P] * 3
    assert off_used - on_used == sum(2 * (h // P) for h in on_hits) == 24
    assert gate[False][1] == gate[True][1] == 4 * P
    assert gate[False][0] - gate[True][0] == 2 * 4


def _producer_connector(cfg_kvc):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from tests.distributed.test_tpu_connector import _make_vllm_config
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    vcfg = _make_vllm_config(tp_size=1, is_producer=True, pp_size=8)
    vcfg.cache_config.enable_prefix_caching = True
    vcfg.speculative_config = NS(method="mtp", num_speculative_tokens=1)
    with (
        patch(f"{_MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{_MOD}.dist_utils.get_kv_ports", return_value=9100),
    ):
        return Qwen4ExpRaidenConnector(vcfg, KVConnectorRole.SCHEDULER, cfg_kvc)


def _finish(conn, req):
    from vllm.v1.request import RequestStatus

    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.kv_transfer_params = {"_p_side_truncated": True}
    return conn.request_finished_all_groups(req, tuple(conn_tables(req)))


def conn_tables(req):
    return req._q38_tables


def _schedule(conn, new=(), cached=(), resumed=()):
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    output = NS(
        scheduled_new_reqs=[NS(req_id=r, num_computed_tokens=c) for r, c in new],
        scheduled_cached_reqs=NS(
            req_ids=[r for r, _ in cached],
            resumed_req_ids=set(resumed),
            num_computed_tokens=[c for _, c in cached],
        ),
    )
    with patch.object(TPURaidenConnector, "build_connector_meta", lambda self, so: NS()):
        conn.build_connector_meta(output)


@pytest.mark.usefixtures("draft_kv")
@pytest.mark.parametrize("on", [False, True], ids=["hold_off", "hold_on"])
def test_preempted_producer_resumes_holding_its_new_hit_and_publishes_it(on, monkeypatch, request):
    """A request starts cold, computes two pages and is preempted (vLLM frees
    its blocks and recomputes from its next hit); it resumes on a 2-page hit
    another request cached meanwhile. The hold skips the resumed hit; the
    producer publishes that hit (not the first) and the matching skip, keeps
    the null blocks in the send's tables, and the send is not refused."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM,
        PRODUCER_HIT_PARAM,
        published_hit_tokens,
    )

    monkeypatch.setenv("TPU_RAIDEN_PREFIX_AWARE_LOAD", "0")
    if on:
        request.getfixturevalue("hold")
    cfg = dpc.decode_config()
    prod = Producer(cfg)
    conn = _producer_connector(prod.kvc)
    held = [g.cache_group_index for g in conn._stage3_transfer_groups if g.leading_null_pages_ok]
    assert held == ([prod.main, prod.comp] if on else [])

    # The request starts cold (no hit) and computes two pages, then is
    # preempted: vLLM frees its blocks and resets it. Meanwhile a seed with the
    # same 2-page prefix completes, so the request resumes on a 2-page hit.
    _, prompt = _shared_prompt(2, 3 * P + 300, 7)
    req = prod.request(prompt)
    h1 = prod.admit(req)
    assert h1 == 0
    _schedule(conn, new=[(req.request_id, h1)])
    prod.run(req, until=2 * P)
    assert _nulls(prod.tables(req)[prod.main]) == 0
    prod.kvm.free(req)
    req.num_computed_tokens = 0
    _, seed_prompt = _shared_prompt(2, 700, 7)
    seed = prod.request(seed_prompt)
    assert prod.prefill(seed) == 0
    prod.kvm.free(seed)
    h2 = prod.admit(req)
    assert h2 == 2 * P, h2
    _schedule(conn, cached=[(req.request_id, h2)], resumed=[req.request_id])
    prod.run(req)
    t2 = prod.tables(req)
    for g in (prod.main, prod.comp):
        assert _nulls(t2[g]) == (h2 // P if on else 0)
    req._q38_tables = t2
    delay, params = _finish(conn, req)
    assert delay and params.get("num_tokens") == len(prompt), params
    assert params[PRODUCER_HIT_PARAM] == h2
    if on:
        assert params[PRODUCER_DRAFT_SKIP_PARAM] == h2
    else:
        assert PRODUCER_DRAFT_SKIP_PARAM not in params
    assert published_hit_tokens(params) == h2
    send = conn.connector_scheduler.build_connector_meta().reqs_to_send[req.request_id]
    assert send.group_block_ids[prod.main] == t2[prod.main]  # the null blocks ride the table
    c = conn._hold_counters
    assert (c["held_sends"], c["held_pages"], c["skip_above_hit"], c["unheld_hits"]) == (
        (1, h2 // P, 0, 0) if on else (0, 0, 0, 0)
    )
    print("Q38_F2HOLD_PREEMPT", {"on": on, "h1": h1, "h2": h2, "nulls": _nulls(t2[prod.main]),
                                 "params": {k: v for k, v in params.items() if "qwen" in k}}, flush=True)


@pytest.mark.usefixtures("draft_kv", "hold")
def test_a_skip_above_the_hit_is_counted_and_published():
    """The producer's guard: its published skip is the table's, even above the
    hit it publishes (it cannot be by construction), so the decode zero-fills
    up to it (pd_staging.published_hit_tokens; the landing test checks the
    bytes)."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM,
        PRODUCER_HIT_PARAM,
        published_hit_tokens,
    )

    cfg = dpc.decode_config()
    prod = Producer(cfg)
    conn = _producer_connector(prod.kvc)
    _, seed_prompt = _shared_prompt(2, 50, 9)
    seed = prod.request(seed_prompt)
    prod.prefill(seed)
    prod.kvm.free(seed)
    _, prompt = _shared_prompt(2, 2000, 9)
    req = prod.request(prompt)
    assert prod.prefill(req) == 2 * P
    _schedule(conn, new=[(req.request_id, P)])  # a stale record: 1 page
    req._q38_tables = prod.tables(req)
    delay, params = _finish(conn, req)
    assert params[PRODUCER_HIT_PARAM] == P and params[PRODUCER_DRAFT_SKIP_PARAM] == 2 * P
    assert published_hit_tokens(params) == 2 * P
    assert conn._hold_counters["skip_above_hit"] == 1


def test_published_hit_takes_the_larger_of_the_hit_and_the_skip():
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM as S,
        PRODUCER_HIT_PARAM as H,
        published_hit_tokens,
    )

    assert published_hit_tokens({H: 6400, S: 3200}) == 6400
    assert published_hit_tokens({H: 3200, S: 6400}) == 6400
    assert published_hit_tokens({S: 6400}) == 6400
    assert published_hit_tokens({H: 6400}) == 6400
    assert published_hit_tokens({H: "6400", S: -1}) == 0
    assert published_hit_tokens(None) == 0


def test_held_ordinals_keep_every_destination_and_read_only_live_pages():
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_route

    assert [pd_route.held_ordinal(p, 2) for p in range(5)] == [0, 0, 0, 1, 2]
    assert [pd_route.held_ordinal(p, 0) for p in range(3)] == [0, 1, 2]
    spans = [(0, 0, 0, 8, None), (1, 16, 8, 8, 3), (2, 0, 16, 8, None), (3, 0, 24, 8, 1)]
    held = pd_route.held_spans(spans, 2)
    assert [s[1:] for s in held] == [s[1:] for s in spans]  # offsets, sizes, units
    assert [s[0] for s in held] == [0, 0, 0, 1]
    assert pd_route.held_spans(spans, 0) == spans


@pytest.mark.usefixtures("draft_kv")
def test_hold_without_draft_kv_is_refused(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV", raising=False)
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    from tests.distributed.test_qwen4_exp_pd_draft_kv import build_connector, connector_config

    cfg = connector_config()
    cfg.speculative_config = None
    with pytest.raises(ValueError, match="needs QWEN4_EXP_PD_DRAFT_KV=1"):
        build_connector(cfg)


def test_flag_is_off_by_default_out_of_the_compile_key_and_registers_nothing(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
    from vllm_torchtpu.core import producer_draft_hold as pdh

    monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_HOLD", raising=False)
    assert envs.environment_variables["QWEN4_EXP_PD_DRAFT_HOLD"]() is False
    assert "QWEN4_EXP_PD_DRAFT_HOLD" in _TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    assert pdh.register_producer_draft_hold() is False
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    for spec_cls in pdh.held_spec_types():
        assert KVCacheSpecRegistry.get_manager_class(object.__new__(spec_cls)) is not (
            pdh.producer_draft_hold_manager()
        )


def _semantics(monkeypatch, hold_on, *, tp_size):
    from tests.distributed import test_qwen4_exp_pd_draft_kv as dk

    if hold_on:
        monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    else:
        monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_HOLD", raising=False)
    return dk.semantics(monkeypatch, True, tp_size=tp_size)


def test_a_one_sided_hold_flag_fails_the_peer_handshake(monkeypatch):
    from tests.distributed.test_qwen4_exp_pd_draft_kv import peer_fingerprint

    on = _semantics(monkeypatch, True, tp_size=1)
    assert on["ranks"][0]["state_flags"]["mtp_draft_hold"] is True
    off = _semantics(monkeypatch, False, tp_size=1)
    assert "mtp_draft_hold" not in off["ranks"][0]["state_flags"]
    p_on, p_off = peer_fingerprint(on), peer_fingerprint(off)
    d_on = peer_fingerprint(_semantics(monkeypatch, True, tp_size=8))
    d_off = peer_fingerprint(_semantics(monkeypatch, False, tp_size=8))
    assert p_on == d_on and p_off == d_off
    assert p_on != d_off and p_off != d_on


@pytest.mark.usefixtures("draft_kv", "hold")
def test_disabling_the_eagle_block_drop_raises_the_hit_and_the_hold_by_exactly_one_block():
    """q38 F2-HOLD, arm HD: with VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 on the F2
    producer its prefix hit is one 3,200-token block longer than with vLLM's
    EAGLE/MTP block drop (F2-4c G1P: producer hit 38.6% with F2, 77.2% without),
    so the hold skips exactly one more draft block per hit request and
    publishes a hit and a skip one block higher. Nothing else changes: the
    target's tables above the hit and the draft's live blocks above it are
    allocated alike."""
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM,
        PRODUCER_HIT_PARAM,
    )

    cfg = dpc.decode_config()
    out = {}
    for drop in (True, False):
        prod = Producer(cfg, eagle_drop=drop)
        conn = _producer_connector(prod.kvc)
        _, seed_prompt = _shared_prompt(3, 700, 11)
        seed = prod.request(seed_prompt)
        assert prod.prefill(seed) == 0
        prod.kvm.free(seed)
        rows = []
        for extra in (900, 4000):
            _, prompt = _shared_prompt(3, extra, 11)
            req = prod.request(prompt)
            hit = prod.admit(req)
            _schedule(conn, new=[(req.request_id, hit)])
            prod.run(req)
            t = prod.tables(req)
            req._q38_tables = t
            _, params = _finish(conn, req)
            rows.append((hit, _nulls(t[prod.main]), _nulls(t[prod.comp]), params.get(PRODUCER_HIT_PARAM),
                         params.get(PRODUCER_DRAFT_SKIP_PARAM), len(t[prod.main]),
                         [len(ids) for ids in t]))
        out[drop] = rows
    print("Q38_F2HOLD_EAGLE", out, flush=True)
    for (h_on, n_on, c_on, ph_on, ps_on, len_on, lens_on), (h_off, n_off, c_off, ph_off, ps_off, len_off, lens_off) in zip(
        out[True], out[False]
    ):
        assert h_off == h_on + P and h_off == 3 * P, (h_on, h_off)
        assert (n_off, c_off) == (n_on + 1, c_on + 1) == (3, 3)
        assert (ph_off, ps_off) == (h_off, h_off)
        assert ph_on == h_on and (ps_on or 0) == h_on
        assert len_on == len_off and lens_on == lens_off  # the same tables, one more null block in each draft group


@pytest.mark.usefixtures("draft_kv", "hold")
def test_held_groups_survive_the_pipeline_stage_projection():
    """F2-HOLD gate H/HD (2026-11-08, j-d7417a2d / j-6621454c): on the PP8 producer vLLM's scheduler KV-cache config is
    stage 0's (generate_scheduler_kv_cache_config deep-copies worker 0's config), and
    _project_kv_cache_groups_to_worker empties the layer names of every group with no layer on that stage, the
    draft's included (it lives on the last stage). Held groups chosen by layer name were then empty: the scheduler
    refused every held table ("state-bearing source requires complete valid pages … group=7 pages=(0, 30, 19)",
    2,638 times) and the decode recomputed every prompt. The held groups must come from the draft spec itself, on
    the whole plan, on every stage's projection and on the scheduler config, and the producer connector built on
    that scheduler config must accept a held table and publish its skip."""
    import copy

    from vllm.v1.core.kv_cache_utils import (
        _project_kv_cache_groups_to_worker,
        generate_scheduler_kv_cache_config,
    )

    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM,
        PRODUCER_HIT_PARAM,
        draft_hold_group_indices,
    )

    cfg = dpc.decode_config()
    prod = Producer(cfg)
    kvc = prod.kvc
    want = {prod.main, prod.comp}
    assert draft_hold_group_indices(kvc.kv_cache_groups) == want
    layer_specs = {n: g.kv_cache_spec for g in kvc.kv_cache_groups for n in g.layer_names}
    # Stage 0 (no draft layer) and the draft's own stage (the draft plus a few target layers).
    stages = {
        "stage0": {n: s for n, s in layer_specs.items() if not n.startswith("mtp.")},
        "draft stage": {n: s for n, s in layer_specs.items() if n.startswith("mtp.") or n.endswith(".47.self_attn")},
    }
    for name, worker_spec in stages.items():
        projected = _project_kv_cache_groups_to_worker(kvc.kv_cache_groups, worker_spec)
        assert len(projected) == len(kvc.kv_cache_groups)
        if name == "stage0":
            assert all(not projected[i].layer_names for i in want), "stage 0 holds no draft layer"
        assert draft_hold_group_indices(projected) == want, name
        stage_cfg = copy.deepcopy(kvc)
        stage_cfg.kv_cache_groups = projected
        sched = generate_scheduler_kv_cache_config([stage_cfg])
        assert draft_hold_group_indices(sched.kv_cache_groups) == want, name
        if name != "stage0":
            continue
        # The producer's scheduler connector as vLLM builds it on a pipeline: on stage 0's config.
        conn = _producer_connector(sched)
        assert {g.cache_group_index for g in conn._stage3_transfer_groups if g.leading_null_pages_ok} == want
        _, seed_prompt = _shared_prompt(2, 50, 13)
        seed = prod.request(seed_prompt)
        prod.prefill(seed)
        prod.kvm.free(seed)
        _, prompt = _shared_prompt(2, 1900, 13)
        req = prod.request(prompt)
        hit = prod.prefill(req)
        assert hit == 2 * P
        _schedule(conn, new=[(req.request_id, hit)])
        req._q38_tables = prod.tables(req)
        assert _nulls(req._q38_tables[prod.main]) == 2
        delay, params = _finish(conn, req)
        assert delay and params[PRODUCER_HIT_PARAM] == hit and params[PRODUCER_DRAFT_SKIP_PARAM] == hit, params
