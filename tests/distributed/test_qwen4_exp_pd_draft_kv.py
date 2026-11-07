# SPDX-License-Identifier: Apache-2.0
"""q38 F2 (QWEN4_EXP_PD_DRAFT_KV): admission of the prefill-only MTP producer,
and the flag in the PD identity.

The producer is the served prefill shape, PP8 x TP1 stages, running the MTP
draft over its prefill chunks at num_speculative_tokens=1 so it can ship the
draft layer's caches. One predicate, ``config.pd_prefill_mtp_producer``,
admits it in ``validate_config``, in ``Qwen4ExpRaidenConnector.__init__`` and
in the worker's ``_validate_stage3_transfer_parallelism``; every other
producer MTP stays refused, and the staged decode's decode-only MTP rule is
unchanged. A one-sided flag must fail the peers' handshake: the flag is part
of the loaded-state identity both legs fingerprint.

The decode side (byte identity, refusals, multi-turn prefix caching) is in
test_qwen4_exp_decode_prefix_cache.py; the M1 quarantine with draft owners in
tests/runner/test_qwen4_exp_pd_staging_reuse.py.
"""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def producer_env(monkeypatch):
    for name, value in {
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
        "QWEN4_EXP_PACK_COMPRESSED_K": "0",
        "QWEN4_EXP_PACK_MAIN_KV": "0",
        "TPU_KV_RESHARD_TRANSPORT": "raiden",
        "TPU_RAIDEN_POOL_TAGS_PER_LAYER": "1",
        "TPU_RAIDEN_QWEN35_ADMISSION": "0",
        "TPU_RAIDEN_KIMIK3_ADMISSION": "0",
        "TPU_RAIDEN_GLM_ADMISSION": "0",
        "TPU_RAIDEN_DSV4_ADMISSION": "0",
    }.items():
        monkeypatch.setenv(name, value)
    for name in (
        "QWEN4_EXP_PD_DRAFT_KV",
        "QWEN4_EXP_PD_ZERO_DRAFT_KV",
        "VLLM_TPU_QSA_MAIN_KV_DCP",
        "VLLM_TPU_QSA_MTP_KV_DCP",
        "QWEN4_EXP_PD_SYNC_REPACK",
        "QWEN4_EXP_PD_SYNC_REPACK_AB_S",
        "QWEN4_EXP_PD_SHARED_COMPRESSED",
        "QWEN4_EXP_PD_ROUTED_DST",
        "QWEN4_EXP_PD_DIVERGENT_HITS",
        "QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def flag(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")


def set_flag(monkeypatch, on):
    if on:
        monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    else:
        monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_KV", raising=False)


# -- configs ----------------------------------------------------------------------


def producer_config(k=1, tp=1, pp=8, prefix_caching=False, role="kv_producer"):
    """A Qwen PD producer of the test model with an MTP draft of ``k`` tokens:
    by default the served prefill shape, PP8 x TP1 (TP x DP x PP = 8)."""
    from tests.models.vllm.test_qwen4_exp_mtp import spec_config
    from tests.runner.test_qwen4_exp_offload import offload_config
    from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig

    cfg = offload_config()  # TP8/EP8, FP8 KV, MTP K=3
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.pipeline_parallel_size = pp
    cfg.cache_config.enable_prefix_caching = prefix_caching
    cfg.cache_config.mamba_cache_mode = "align" if prefix_caching else "none"
    cfg.cache_config.mamba_ssm_cache_dtype = "float32"
    # The layout tests build real cache specs (gdn_cache_spec reads it); vLLM's CacheConfig default.
    if not hasattr(cfg.cache_config, "mamba_cache_dtype"):
        cfg.cache_config.mamba_cache_dtype = "auto"
    cfg.speculative_config = spec_config(k)
    cfg.kv_transfer_config = NS(
        kv_connector=qconfig.PD_CONNECTOR,
        kv_connector_module_path=None,
        kv_role=role,
        is_kv_producer=role == "kv_producer",
    )
    return cfg


def connector_config(k=1, tp=1, pp=8, producer=True, method="mtp"):
    """The connector's view: the TPU connector tests' engine config with an MTP
    speculative config (by default the PP8 x TP1 producer)."""
    from tests.distributed.test_tpu_connector import _make_vllm_config

    cfg = _make_vllm_config(is_producer=producer, tp_size=tp, pp_size=pp)
    cfg.speculative_config = NS(method=method, num_speculative_tokens=k)
    return cfg


# -- the one predicate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,env,want",
    [
        pytest.param({}, {}, True, id="served"),
        pytest.param({}, {"QWEN4_EXP_PD_DRAFT_KV": None}, False, id="no_flag"),
        pytest.param(dict(k=3), {}, False, id="k3"),
        pytest.param(dict(k=2), {}, False, id="k2"),
        pytest.param(dict(tp=2, pp=4), {}, False, id="tp2"),
        pytest.param(dict(pp=1), {}, False, id="pp1"),
        pytest.param(dict(producer=False), {}, False, id="consumer"),
        pytest.param(dict(method="eagle3"), {}, False, id="eagle3"),
        pytest.param({}, {"TPU_RAIDEN_POOL_TAGS_PER_LAYER": "0"}, False, id="shared_tags"),
    ],
)
def test_one_predicate_admits_only_the_flagged_pp_tp1_k1_mtp_producer(
    monkeypatch, kwargs, env, want
):
    from vllm_torchtpu.models.vllm.qwen4_exp.config import pd_prefill_mtp_producer

    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert pd_prefill_mtp_producer(connector_config(**kwargs)) is want


# -- validate_config -------------------------------------------------------------------


@pytest.mark.usefixtures("flag")
@pytest.mark.parametrize(
    "prefix_caching", [pytest.param(False, id="none"), pytest.param(True, id="align")]
)
def test_config_admits_the_flagged_pp_tp1_k1_producer(prefix_caching):
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    validate_config(producer_config(prefix_caching=prefix_caching))


@pytest.mark.parametrize(
    "kwargs,flag_on",
    [
        pytest.param(dict(k=3), True, id="k3"),
        pytest.param({}, False, id="no_flag"),
        pytest.param(dict(tp=2, pp=4), True, id="tp2"),
    ],
)
def test_config_refuses_every_other_pipeline_mtp(monkeypatch, kwargs, flag_on):
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    set_flag(monkeypatch, flag_on)
    with pytest.raises(ValueError, match="pipeline parallelism requires TP1 stages"):
        validate_config(producer_config(**kwargs))


@pytest.mark.usefixtures("flag")
def test_config_flag_does_not_admit_mtp_on_a_non_pipeline_producer():
    """A TP8 producer stays refused with the flag too (without it:
    test_still_rejects_mtp_on_a_producer_and_caching_without_align)."""
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    with pytest.raises(ValueError, match="staged TP>1 decode"):
        validate_config(producer_config(k=1, tp=8, pp=1))


# -- Qwen4ExpRaidenConnector ---------------------------------------------------------------


def build_connector(cfg):
    """The model connector's own admission; the shared Stage-3 lifecycle
    (TPURaidenConnector.__init__) is not under test and is skipped."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenConnector

    with patch.object(TPURaidenConnector, "__init__", lambda self, *a, **k: None):
        return Qwen4ExpRaidenConnector(cfg, KVConnectorRole.SCHEDULER, None)


@pytest.mark.usefixtures("flag")
def test_connector_admits_the_flagged_pp_tp1_k1_producer():
    conn = build_connector(connector_config())
    assert not conn._sync_repack  # a producer never runs the decode's L1 repack


@pytest.mark.parametrize(
    "kwargs,flag_on",
    [
        pytest.param(dict(k=3), True, id="k3"),
        pytest.param({}, False, id="no_flag"),
        pytest.param(dict(method="eagle3"), True, id="eagle3"),
        pytest.param(dict(tp=2, pp=4), True, id="tp2"),
    ],
)
def test_connector_refuses_every_other_producer_mtp(monkeypatch, kwargs, flag_on):
    set_flag(monkeypatch, flag_on)
    with pytest.raises(NotImplementedError, match="MTP only on a staged TP>1 decode"):
        build_connector(connector_config(**kwargs))


@pytest.mark.usefixtures("flag")
def test_connector_keeps_the_decode_only_mtp_rule():
    build_connector(connector_config(k=3, tp=8, pp=1, producer=False))  # staged decode
    with pytest.raises(NotImplementedError, match="MTP only on a staged TP>1 decode"):
        build_connector(connector_config(k=3, tp=1, pp=1, producer=False))


def raiden_worker(cfg, *, producer=True, tp_size=1):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    worker = object.__new__(Qwen4ExpRaidenWorker)
    worker.vllm_config, worker.is_producer, worker.tp_size = cfg, producer, tp_size
    return worker


@pytest.mark.usefixtures("flag")
def test_worker_admits_the_flagged_producer_stages():
    raiden_worker(connector_config())._validate_stage3_transfer_parallelism(8)


@pytest.mark.parametrize(
    "kwargs,flag_on",
    [pytest.param(dict(k=3), True, id="k3"), pytest.param({}, False, id="no_flag")],
)
def test_worker_refuses_every_other_producer_mtp(monkeypatch, kwargs, flag_on):
    set_flag(monkeypatch, flag_on)
    with pytest.raises(ValueError, match="CP1 and MTP off"):
        raiden_worker(connector_config(**kwargs))._validate_stage3_transfer_parallelism(8)


# -- the PD identity ---------------------------------------------------------------------


def semantics(monkeypatch, flag_on, *, tp_size):
    """``_collect_pd_semantics`` of a one-rank engine of ``tp_size`` (a TP1
    pipeline stage, or one rank of a TP8 decode) with the flag set or not."""
    import torch

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pc

    set_flag(monkeypatch, flag_on)
    worker = object.__new__(pc.Qwen4ExpRaidenWorker)
    worker.vllm_config = NS(
        parallel_config=NS(enable_expert_parallel=False),
        cache_config=NS(mamba_ssm_cache_dtype="float32"),
    )
    worker.tp_rank, worker.tp_size = 0, tp_size
    model = NS(
        pd_gdn_qk_pair_layout=envs.TPU_GDN_CONV_QK_PAIR_LAYOUT,
        model=NS(stage=range(0), layers=[]),
        config=NS(to_dict=lambda: {"hidden_size": 16, "_name_or_path": "/ckpt"}),
        pd_loaded_weight_identity="a" * 64,
        pd_loaded_ple_identity="b" * 64,
    )
    group = NS(rank_in_group=0, world_size=1, cpu_group=None)
    monkeypatch.setattr(pc, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj, group=None: out.__setitem__(0, obj),
    )
    return worker._collect_pd_semantics(NS(model_adapter=NS(cache=NS(model=model))))


def peer_fingerprint(sem):
    """The semantics half of what a per-layer-tag peer fingerprints
    (``_measure_stage3_layout``), over one fixed layout."""
    from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
        canonical_layout_fingerprint,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
        _owner_invariant_semantics,
    )

    return canonical_layout_fingerprint(
        dict(layout={"schema": "fixed"}, semantics=_owner_invariant_semantics(sem))
    )


def test_draft_kv_flag_is_part_of_the_loaded_state_identity(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
        _owner_invariant_semantics,
    )

    on = semantics(monkeypatch, True, tp_size=1)
    off = semantics(monkeypatch, False, tp_size=1)
    assert on["ranks"][0]["state_flags"]["mtp_draft_kv"] is True
    assert _owner_invariant_semantics(on)["state_flags"]["mtp_draft_kv"] is True
    # Flag off: today's identity, without the key.
    assert "mtp_draft_kv" not in off["ranks"][0]["state_flags"]


def test_a_one_sided_draft_kv_flag_fails_the_peer_handshake(monkeypatch):
    producer_on = peer_fingerprint(semantics(monkeypatch, True, tp_size=1))
    producer_off = peer_fingerprint(semantics(monkeypatch, False, tp_size=1))
    decode_on = peer_fingerprint(semantics(monkeypatch, True, tp_size=8))
    decode_off = peer_fingerprint(semantics(monkeypatch, False, tp_size=8))
    # The peers agree when both set the flag or neither does ...
    assert producer_on == decode_on and producer_off == decode_off
    # ... and fail closed when only one does.
    assert producer_on != decode_off and producer_off != decode_on


def test_flag_is_off_by_default_and_out_of_the_compile_key():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert envs.environment_variables["QWEN4_EXP_PD_DRAFT_KV"]() is False
    assert "QWEN4_EXP_PD_DRAFT_KV" in _TPU_COMPILE_ENV_IGNORED


def test_force_gap_is_off_by_default_and_out_of_the_compile_key():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert envs.environment_variables["QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP"]() == 0.0
    assert "QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP" in _TPU_COMPILE_ENV_IGNORED


# -- q38 F2 option (a): the producer never caches draft KV -----------------------------


def _schedule(conn, new=(), cached=(), resumed=()):
    """One scheduler step's ``build_connector_meta`` on ``conn`` (the shared
    Stage-3 half stubbed): ``new`` / ``cached`` are (req id, computed tokens)."""
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    output = NS(
        scheduled_new_reqs=[NS(req_id=r, num_computed_tokens=c) for r, c in new],
        scheduled_cached_reqs=NS(
            req_ids=[r for r, _ in cached],
            resumed_req_ids=set(resumed),
            num_computed_tokens=[c for _, c in cached],
        ),
    )
    meta = NS()
    with patch.object(TPURaidenConnector, "build_connector_meta", lambda self, so: meta):
        assert conn.build_connector_meta(output) is meta


def _finish(conn, req_id, num_cached_tokens=-1, num_tokens=12799):
    """``request_finished_all_groups`` with the shared half publishing a send
    of ``num_tokens``; returns the consumer-facing params."""
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    published = {"uuid": 5, "num_tokens": num_tokens}
    with patch.object(
        TPURaidenConnector, "request_finished_all_groups", lambda self, r, ids: (True, dict(published))
    ):
        delay, params = conn.request_finished_all_groups(
            NS(request_id=req_id, num_cached_tokens=num_cached_tokens), ([],)
        )
    assert delay
    return params


@pytest.mark.usefixtures("flag")
def test_producer_publishes_the_prefix_hit_it_was_last_scheduled_with():
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import PRODUCER_HIT_PARAM

    conn = build_connector(connector_config())
    _schedule(conn, new=[("r1", 6400), ("r2", 0), ("r5", 3200)])
    # r1 is preempted and resumes with a longer hit; r2 just continues.
    _schedule(conn, cached=[("r1", 9600), ("r2", 3000)], resumed=["r1"])
    assert _finish(conn, "r1")[PRODUCER_HIT_PARAM] == 9600
    assert PRODUCER_HIT_PARAM not in _finish(conn, "r2")  # no hit
    assert _finish(conn, "r3", num_cached_tokens=3200)[PRODUCER_HIT_PARAM] == 3200  # never seen
    assert _finish(conn, "r4", num_cached_tokens=99999)[PRODUCER_HIT_PARAM] == 12799  # capped
    assert PRODUCER_HIT_PARAM not in _finish(conn, "r6")  # vLLM's count unset (-1)
    assert conn._producer_hits == {"r5": 3200}


def test_only_the_f2_producer_publishes_a_hit(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import PRODUCER_HIT_PARAM

    cfg = connector_config()
    cfg.speculative_config = None  # a producer without the draft
    for on in (False, True):
        set_flag(monkeypatch, on)
        conn = build_connector(cfg)
        _schedule(conn, new=[("r1", 6400)])
        assert conn._producer_hits is None
        assert PRODUCER_HIT_PARAM not in _finish(conn, "r1", num_cached_tokens=6400)


def test_worker_hands_the_carried_hit_to_the_staging_and_keeps_it_in_the_snapshot():
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    calls = []

    class Staging:
        pages = 4
        free = None
        reservations, skips, tokens = {}, {}, {}

        def reserve(self, uuid, num_tokens, tp_group=None, **kwargs):
            calls.append((uuid, kwargs.get("draft_zero_until")))
            return True

    worker = object.__new__(Qwen4ExpRaidenWorker)
    worker._staging = Staging()
    meta = NS(uuid=7, num_tokens=20000, skip_tokens=2048, src_cached_tokens=9600)
    assert worker._stage3_reserve_destination("d7", meta, None)
    assert worker._stage3_reserve_destination_reduced(meta, reduced_free=None)
    assert calls == [(7, 9600), (7, 9600)]
    # The batched admission's snapshot restores the per-load hits too.
    import torch

    staging = worker._staging
    staging.free = torch.ones(4, dtype=torch.int32)
    staging.draft_zero = {7: 9600}
    saved = worker._stage3_admission_snapshot()
    staging.draft_zero[8] = 3200
    worker._stage3_admission_restore(saved)
    assert staging.draft_zero == {7: 9600}


def test_producer_draft_specs_are_non_prefix_cacheable_on_both_legs(monkeypatch):
    """The producer engine (native_cache_specs on pd_prefill_mtp_producer) and
    a decode rebuilding its plan (producer_geometry) both give the draft's
    owners non-prefix-cacheable specs, in groups of their own; the targets'
    and the decode's own draft stay cacheable. PENDING the vLLM probe: this is
    the served image's verdict on config.non_prefix_cacheable."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging
    from vllm_torchtpu.models.vllm.qwen4_exp.config import (
        ensure_kv_group_size_rule,
        pd_prefill_mtp_producer,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    ensure_kv_group_size_rule()
    producer = producer_config()
    decode = producer_config(k=3, tp=8, pp=1, role="kv_consumer")
    assert pd_prefill_mtp_producer(producer) and not pd_prefill_mtp_producer(decode)
    _, own = pooled_cache_specs(producer)
    layout_cfg, tp, dcp, pack, cacheable = pd_staging.producer_geometry(decode)
    _, rebuilt = pooled_cache_specs(
        layout_cfg, tp=tp, main_dcp=dcp, pack_compressed=pack, draft_prefix_cacheable=cacheable
    )
    draft = "mtp.layers.48.self_attn"
    owners = {draft, draft + ".compressed_k", draft + ".raw_ring"}
    assert {n for n in own if n.startswith("mtp.")} == owners
    assert {n for n in rebuilt if n.startswith("mtp.")} == owners
    for specs in (own, rebuilt):
        assert not any(specs[n].prefix_cacheable for n in owners)
        targets = [n for n in specs if n.endswith((".self_attn", ".compressed_k")) and n not in owners]
        assert targets and all(specs[n].prefix_cacheable for n in targets)
    assert {n: type(own[n]) for n in owners} == {n: type(rebuilt[n]) for n in owners}
    for cfg, specs in ((producer, own), (decode, rebuilt)):
        assert pd_staging.draft_group_problems(get_kv_cache_groups(cfg, dict(specs))) == []
    # The decode's own draft keeps its prefix cache.
    _, decode_own = pooled_cache_specs(decode)
    assert decode_own[draft].prefix_cacheable and decode_own[draft + ".compressed_k"].prefix_cacheable


def test_the_mtp_draft_satisfies_supports_pp():
    """F2's pipeline producer runs Qwen4ExpMTP on its last stage. vLLM requires every model under a
    pipeline to satisfy `SupportsPP` (q38 F2-1 CPU job j-ea7903ad probe: `supports_pp(Qwen4ExpMTP)`
    was False); the draft declares it by attribute and never makes intermediate tensors."""
    from vllm.model_executor.models.interfaces import supports_pp

    from vllm_torchtpu.models.vllm.qwen4_exp.mtp import Qwen4ExpMTP

    assert supports_pp(Qwen4ExpMTP)
    with pytest.raises(NotImplementedError, match="runs whole on the last pipeline stage"):
        Qwen4ExpMTP.make_empty_intermediate_tensors(object(), 1, None, None)
