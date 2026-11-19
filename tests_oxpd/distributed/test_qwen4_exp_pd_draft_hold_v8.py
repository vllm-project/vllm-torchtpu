# SPDX-License-Identifier: Apache-2.0
"""q38 HIGHC-CAPACITY: F2-HOLD (QWEN4_EXP_PD_DRAFT_HOLD) on the v8 tree, against the failure of its only served run.

F2-HOLD gate arm HD (j-6621454c, 2026-11-08) ran with the hold engaged in the manager and NO send: its PP8 producer
logged 'Stage-3 state-bearing source requires complete valid pages: ... group=7 pages=(0, 0, 28)' 2,638 times, twice for
each of its 1,319 requests (final attempt pd0-0-0-fr75b prefill log, md5 91275ee0), and 0 F2-HOLD producer lines, so the
decode recomputed every prompt. Cause (F2-HOLD 13:53Z read; fix 8b2cef5c1, ported here): vLLM builds the PP scheduler's
KV-cache config from stage 0's projection, which empties the layer names of the MTP draft groups (the draft lives on the
last stage); the held groups were chosen by layer name, so the scheduler never marked them, while the KV-cache manager
(chosen by spec type) still opened their tables with null blocks.

These tests rebuild that exact setting on the served v8 prefill plan: the whole-model F2 producer plan, its projection to
each of the 8 stages of VLLM_PP_LAYER_PARTITION=7,5,7,5,7,5,7,5 (the T94u render), vLLM's own
generate_scheduler_kv_cache_config over those 8 configs, the REAL KVCacheManager built on that scheduler config, the REAL
Qwen PD connector scheduler half built on it, and every stage worker's page plan (build_transfer_groups / held_heads /
handoff_block_ids on its own projection). Each test runs with vLLM's EAGLE/MTP block drop (the v8 prefill today) and with
it disabled (S0's F-a, VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1, which lifts each hit by one 3,200-token block).

Assertion messages start with the names the CDK CPU proof's mutants match: HD-SEL (held groups on every projection),
HD-MGR (the held manager on the scheduler config), HD-FA (the hit with / without the drop), HD-HOLD (null blocks only
below the hit), HD-SEND (the send goes out and is not refused), HD-SKIP (the published skip), HD-WORKER (every stage's
worker publishes the held table), HOLD-LOG (the registration line reaches vLLM's log handler).
"""

import copy
import logging
import os

import pytest

from tests.distributed import test_qwen4_exp_decode_prefix_cache as dpc
from tests.distributed import test_qwen4_exp_pd_draft_hold as H
# served_decode_env is autouse: the served Raiden / pool-tag env (TPU_RAIDEN_POOL_TAGS_PER_LAYER=1, ...) that the
# hold's other test files get the same way; without it pd_prefill_mtp_producer is False and the connector refuses
# the producer (hcap-dev1 j-1b3c12ea, 2026-11-16: the first version of this file omitted it).
from tests.distributed.test_qwen4_exp_decode_prefix_cache import draft_kv, served_decode_env  # noqa: F401
from tests.distributed.test_qwen4_exp_pd_draft_hold import hold  # noqa: F401

P = H.P
PARTITION = (7, 5, 7, 5, 7, 5, 7, 5)  # VLLM_PP_LAYER_PARTITION of the v8 T94u prefill (j-64613705 recipe)
REFUSAL = "state-bearing source requires complete valid pages"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self._seen = set()

    def emit(self, record):
        if id(record) in self._seen:  # one record can reach both loggers (vLLM's logging left unconfigured)
            return
        self._seen.add(id(record))
        try:
            self.records.append((record.name, record.levelname, record.getMessage()))
        except Exception as e:  # noqa: BLE001
            self.records.append((record.name, record.levelname, f"<format error {e}>"))


@pytest.fixture
def vllm_log():
    """Every record that reaches vLLM's 'vllm' logger (propagate False, so pytest's caplog does not see them) or the
    root logger, at INFO and above."""
    cap = _Capture()
    loggers = [logging.getLogger("vllm"), logging.getLogger()]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(cap)
        if lg.level == 0 or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    yield cap
    for lg, lv in zip(loggers, levels):
        lg.removeHandler(cap)
        lg.setLevel(lv)


def _stage_of(name, n_layers):
    if name.startswith("mtp."):
        return len(PARTITION) - 1
    index = int(name.split(".")[2])  # model.layers.<i>.<owner>
    assert sum(PARTITION) == n_layers, (PARTITION, n_layers)
    edge = 0
    for stage, count in enumerate(PARTITION):
        edge += count
        if index < edge:
            return stage
    raise AssertionError(name)


def _stage_configs(cfg, kvc):
    """The 8 PP stages' KV-cache configs as vLLM builds them: the whole-model groups projected onto each stage's
    layers (kv_cache_utils._project_kv_cache_groups_to_worker)."""
    from vllm.v1.core.kv_cache_utils import _project_kv_cache_groups_to_worker

    n_layers = len(cfg.model_config.hf_text_config.layer_types)
    layer_specs = {n: g.kv_cache_spec for g in kvc.kv_cache_groups for n in g.layer_names}
    out = []
    for stage in range(len(PARTITION)):
        worker = {n: s for n, s in layer_specs.items() if _stage_of(n, n_layers) == stage}
        assert worker, stage
        c = copy.deepcopy(kvc)
        c.kv_cache_groups = _project_kv_cache_groups_to_worker(kvc.kv_cache_groups, worker)
        assert len(c.kv_cache_groups) == len(kvc.kv_cache_groups)
        out.append(c)
    return out


FA_WARNING = "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: EAGLE/MTP trailing prefix-cache block drop is disabled"


def _served_scheduler(kvc, fa, monkeypatch):
    """The served prefill's scheduler (TpuPipelineChunkScheduler, a vLLM Scheduler) built on ``kvc`` with the
    producer's MTP k=1 (use_eagle=True), as HIGHC-S0TEST's real-scheduler bench builds it (j-300d077d). With ``fa``,
    VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 and core/eagle_block_drop's REAL patch are installed first, so the KV-cache
    manager the test drives is the one the patched scheduler builds (an inert patch leaves vLLM's drop on and the hit
    one 3,200-token block lower). The Scheduler class is restored at teardown."""
    import vllm.v1.core.sched.scheduler as scheduler_module
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_torchtpu.core import eagle_block_drop
    from vllm_torchtpu.core.pp_chunk_scheduler import TpuPipelineChunkScheduler

    sched_cls = scheduler_module.Scheduler
    monkeypatch.setattr(sched_cls, "__init__", sched_cls.__init__)  # restored at teardown
    monkeypatch.setattr(sched_cls, "_mamba_block_aligned_split", sched_cls._mamba_block_aligned_split)
    assert not getattr(sched_cls.__init__, "_tpu_disable_eagle_block_drop_patch", False), "another test left F-a on"
    if fa:
        monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
        eagle_block_drop.patch_disable_eagle_block_drop()
    else:
        monkeypatch.delenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", raising=False)
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")  # the tiny model config at the served 262,144 (as S0)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tiny = os.path.join(root, "tests", "q38_dstash", "tiny_opt")
    model = ModelConfig(model=tiny, tokenizer=tiny, dtype="float16", seed=42, max_model_len=262144,
                        skip_tokenizer_init=True)
    cache = CacheConfig(block_size=P, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=True,
                        mamba_cache_mode="align")
    vc = VllmConfig(
        scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=8192, max_model_len=262144,
                                         enable_chunked_prefill=True, is_encoder_decoder=False),
        model_config=model, cache_config=cache, parallel_config=ParallelConfig(),
        speculative_config=SpeculativeConfig(model="ngram", num_speculative_tokens=1, prompt_lookup_max=3,
                                             prompt_lookup_min=1),
        device_config=DeviceConfig(device="cpu"),
    )
    # The served prefill: --speculative-config '{"method":"mtp","num_speculative_tokens":1}' (no draft model on CPU).
    object.__setattr__(vc.speculative_config, "method", "mtp")
    vc.cache_config.num_gpu_blocks = kvc.num_blocks
    register_all_kvcache_specs(vc)
    sbs, hbs = kv_cache_utils.resolve_kv_cache_block_sizes(kvc, vc)
    assert (sbs, hbs) == (P, P), (sbs, hbs)
    sched = TpuPipelineChunkScheduler(vllm_config=vc, kv_cache_config=kvc,
                                      structured_output_manager=StructuredOutputManager(vc),
                                      block_size=sbs, hash_block_size=hbs, log_stats=True)
    assert sched.use_eagle, "HD-FA: the producer's MTP must make the scheduler an EAGLE scheduler"
    return sched


class StageProducer(H.Producer):
    """H.Producer driving the KV-cache manager of the served scheduler (``_served_scheduler``) built on stage 0's
    projection, as EngineCore builds it on the PP8 producer."""

    def __init__(self, kvm, groups):  # noqa: D107 (no super().__init__: it builds its own plan)
        self.kvm = kvm
        self.kvc = None
        self.main, self.comp, self.ring = groups
        self.serial = 0


def _producer_connector(kvc):
    """H._producer_connector, with the F2-producer admission predicate checked first and its inputs reported."""
    from types import SimpleNamespace as NS
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from tests.distributed.test_tpu_connector import _make_vllm_config
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp.config import pd_prefill_mtp_producer
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import draft_kv_enabled

    vcfg = _make_vllm_config(tp_size=1, is_producer=True, pp_size=8)
    vcfg.cache_config.enable_prefix_caching = True
    vcfg.speculative_config = NS(method="mtp", num_speculative_tokens=1)
    t, p = vcfg.kv_transfer_config, vcfg.parallel_config
    inputs = {
        "draft_kv": draft_kv_enabled(), "is_kv_producer": getattr(t, "is_kv_producer", None),
        "pp": p.pipeline_parallel_size, "tp": p.tensor_parallel_size,
        "TPU_RAIDEN_POOL_TAGS_PER_LAYER": envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER,
        "spec": (vcfg.speculative_config.method, vcfg.speculative_config.num_speculative_tokens),
    }
    assert pd_prefill_mtp_producer(vcfg), f"HD-CFG: the test config is not the served F2 producer: {inputs}"
    with (
        patch(f"{H._MOD}.dist_utils.get_kv_ips", return_value="127.0.0.1"),
        patch(f"{H._MOD}.dist_utils.get_kv_ports", return_value=9100),
    ):
        return Qwen4ExpRaidenConnector(vcfg, KVConnectorRole.SCHEDULER, kvc)


def _draft_groups(kvc):
    """(main KV, compressed keys, ring) of the MTP draft, by the whole-model plan's layer names."""
    by_name = {g.layer_names[0]: i for i, g in enumerate(kvc.kv_cache_groups) if g.layer_names}
    return (by_name["mtp.layers.48.self_attn"], by_name["mtp.layers.48.self_attn.compressed_k"],
            by_name["mtp.layers.48.self_attn.raw_ring"])


@pytest.mark.usefixtures("draft_kv", "hold")
@pytest.mark.parametrize("fa", [False, True], ids=["eagle_drop", "fa"])
def test_hd_held_sends_go_out_on_the_pp8_stage0_scheduler_and_every_stage_worker(fa, vllm_log, monkeypatch):
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    from vllm_torchtpu.core import producer_draft_hold as pdh
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import build_transfer_groups, handoff_block_ids, held_heads
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import (
        PRODUCER_DRAFT_SKIP_PARAM,
        PRODUCER_HIT_PARAM,
        draft_hold_group_indices,
    )

    cfg = dpc.decode_config()
    whole = H.producer_plan(cfg)
    main, comp, ring = _draft_groups(whole)
    want = {main, comp}
    stages = _stage_configs(cfg, whole)
    sched = generate_scheduler_kv_cache_config(stages)

    # The HD condition itself: stage 0 (the scheduler's config) names no draft layer.
    assert all(not sched.kv_cache_groups[i].layer_names for i in (main, comp, ring))
    for stage, c in enumerate(stages):
        got = draft_hold_group_indices(c.kv_cache_groups)
        assert got == want, f"HD-SEL: stage {stage}'s projection holds {sorted(got)}, want {sorted(want)}"
    got = draft_hold_group_indices(sched.kv_cache_groups)
    assert got == want, f"HD-SEL: the scheduler config holds {sorted(got)}, want {sorted(want)}"

    scheduler = _served_scheduler(sched, fa, monkeypatch)
    fa_lines = [r for r in vllm_log.records if FA_WARNING in r[2]]
    assert len(fa_lines) == (1 if fa else 0), f"HD-FA: F-a engagement lines {fa_lines} (fa={fa})"
    prod = StageProducer(scheduler.kv_cache_manager, (main, comp, ring))
    managers = prod.kvm.coordinator.single_type_managers
    held_cls = pdh.producer_draft_hold_manager()
    assert {i for i, m in enumerate(managers) if type(m) is held_cls} == want, (
        f"HD-MGR: held managers {[type(m).__name__ for m in managers]}"
    )
    conn = _producer_connector(sched)
    marked = {g.cache_group_index for g in conn._stage3_transfer_groups if g.leading_null_pages_ok}
    assert marked == want, f"HD-SEL: the producer connector marks {sorted(marked)}, want {sorted(want)}"

    # A 3-page seed, then two requests on the same 3-page prefix (the hit: 2 pages with the drop, 3 with F-a).
    _, seed_prompt = H._shared_prompt(3, 700, 21)
    seed = prod.request(seed_prompt)
    assert prod.prefill(seed) == 0
    prod.kvm.free(seed)
    want_hit = 3 * P if fa else 2 * P
    sends = []
    for extra in (900, 4400):
        _, prompt = H._shared_prompt(3, extra, 21)
        req = prod.request(prompt)
        hit = prod.admit(req)
        assert hit == want_hit, f"HD-FA: hit {hit}, want {want_hit} (fa={fa})"
        H._schedule(conn, new=[(req.request_id, hit)])
        prod.run(req)
        tables = prod.tables(req)
        n_pages = -(-len(prompt) // P)
        for g in (main, comp):
            assert len(tables[g]) == n_pages and H._nulls(tables[g]) == hit // P, (
                f"HD-HOLD: group {g} table opens with {H._nulls(tables[g])} null blocks of {len(tables[g])}, "
                f"want {hit // P} of {n_pages}"
            )
            assert all(tables[g][hit // P:]), f"HD-HOLD: group {g} has a null block above the hit: {tables[g]}"
        assert len(tables[ring]) == 1 and tables[ring][0], f"HD-HOLD: the ring keeps its block, got {tables[ring]}"
        req._q38_tables = tables
        delay, params = H._finish(conn, req)
        assert delay and isinstance(params, dict) and params.get("num_tokens") == len(prompt), (
            f"HD-SEND: the producer refused the held send (delay={delay}, params={params})"
        )
        assert params.get(PRODUCER_HIT_PARAM) == hit, f"HD-SEND: published hit {params.get(PRODUCER_HIT_PARAM)}"
        assert params.get(PRODUCER_DRAFT_SKIP_PARAM) == hit, (
            f"HD-SKIP: published draft skip {params.get(PRODUCER_DRAFT_SKIP_PARAM)}, want {hit}"
        )
        sends.append((req, tables, len(prompt), hit))

    refusals = [r for r in vllm_log.records if REFUSAL in r[2]]
    assert not refusals, f"HD-SEND: {len(refusals)} refusal lines, first {refusals[0]}"
    c = conn._hold_counters
    assert (c["sends"], c["held_sends"], c["held_pages"], c["skip_above_hit"], c["unheld_hits"]) == (
        2, 2, 2 * (want_hit // P), 0, 0
    ), f"HD-SEND: hold counters {c}"
    meta = conn.connector_scheduler.build_connector_meta().reqs_to_send

    # Every stage worker publishes the send's tables (stages 0-6 have no draft layer; the pre-fix name rule held
    # nothing there and refused the null entries as invalid IDs).
    for req, tables, n, hit in sends:
        sent = [list(ids) for ids in meta[req.request_id].group_block_ids]
        assert sent[main] == tables[main], "HD-SEND: the null blocks ride the send's table"
        for stage, sc in enumerate(stages):
            groups = build_transfer_groups(sc.kv_cache_groups)
            held = draft_hold_group_indices(sc.kv_cache_groups)
            heads = held_heads(groups, sent, held)
            try:
                pages = handoff_block_ids(groups, sent, committed_tokens=n, computed_tokens=n, heads=heads)
            except ValueError as e:
                pytest.fail(f"HD-WORKER: stage {stage} refused the held table: {e}")
            assert heads[main] == heads[comp] == hit // P, f"HD-WORKER: stage {stage} heads {heads}"
            assert len(pages[main]) == len(tables[main]) - hit // P and 0 not in pages[main], (
                f"HD-WORKER: stage {stage} pages {pages[main]}"
            )
    print("Q38_HCAP_HD", {"fa": fa, "hit": want_hit, "held_pages": c["held_pages"], "refusals": len(refusals),
                          "stages": len(stages)}, flush=True)


@pytest.mark.usefixtures("draft_kv")
def test_hold_registration_line_reaches_the_vllm_log_handler(monkeypatch, vllm_log):
    from vllm_torchtpu.core import producer_draft_hold as pdh

    name = getattr(pdh.logger, "_name", None) or getattr(pdh.logger, "name", "")
    assert name.startswith("vllm."), (
        f"HOLD-LOG: the hold's logger {name!r} is outside vLLM's 'vllm' tree; its INFO lines never print"
    )
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    pdh.unregister_producer_draft_hold()
    try:
        assert pdh.register_producer_draft_hold()
    finally:
        pdh.unregister_producer_draft_hold()
    lines = [r for r in vllm_log.records if "q38 F2-HOLD: the producer's MTP draft holds blocks" in r[2]]
    assert len(lines) == 1 and lines[0][0].startswith("vllm."), f"HOLD-LOG: registration lines {lines}"
