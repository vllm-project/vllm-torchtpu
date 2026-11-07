# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_STAGING_LEASES: bounded host staging on a Qwen PD producer.

The q38 producer registered its pools with no staging leases, so tpu_sync kept
a full host mirror of every pool (about 8 x QWEN4_EXP_MAX_POOL_BYTES per
prefill host). With N > 0 the producer passes ``staging_leases=N`` and one
uniform hint per pool (pd_manifest.qwen_staging_hints). The arena itself is
native and device-backed only, so these tests cover the hints and the call.
"""

import logging
from types import SimpleNamespace

import pytest

from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import qwen_staging_hints

MAX_LEN = 262_144  # the served --max-model-len
BLOCK = 3_200  # the TP1 producer's page tokens

# FR2 (j-bef2cd63) producer pools per PP stage, from its 128,080
# raiden_stage3_grouped_request_blocks_registered events: (tag, layer, storage),
# the storage being the layer's index within its group (layer_to_pool_index),
# and the most source pages any one transfer touched on each storage (at up to
# 256,183 tokens).
FR2_STAGES = {
    0: {
        "pools": [
            ('qwen4.g0.model.layers.0.linear_attn', 'model.layers.0.linear_attn', 0),
            ('qwen4.g0.model.layers.4.linear_attn', 'model.layers.4.linear_attn', 1),
            ('qwen4.g1.model.layers.1.linear_attn', 'model.layers.1.linear_attn', 0),
            ('qwen4.g1.model.layers.5.linear_attn', 'model.layers.5.linear_attn', 1),
            ('qwen4.g2.model.layers.2.linear_attn', 'model.layers.2.linear_attn', 0),
            ('qwen4.g3.model.layers.1.ple', 'model.layers.1.ple', 0),
            ('qwen4.g4.model.layers.3.self_attn', 'model.layers.3.self_attn', 0),
            ('qwen4.g5.model.layers.3.self_attn.compressed_k', 'model.layers.3.self_attn.compressed_k', 0),
            ('qwen4.g6.model.layers.3.self_attn.raw_ring', 'model.layers.3.self_attn.raw_ring', 0),
        ],
        "max_slots": {0: 167, 1: 2},
    },
    1: {
        "pools": [
            ('qwen4.g0.model.layers.8.linear_attn', 'model.layers.8.linear_attn', 0),
            ('qwen4.g1.model.layers.9.linear_attn', 'model.layers.9.linear_attn', 0),
            ('qwen4.g2.model.layers.10.linear_attn', 'model.layers.10.linear_attn', 1),
            ('qwen4.g2.model.layers.6.linear_attn', 'model.layers.6.linear_attn', 0),
            ('qwen4.g4.model.layers.11.self_attn', 'model.layers.11.self_attn', 1),
            ('qwen4.g4.model.layers.7.self_attn', 'model.layers.7.self_attn', 0),
            ('qwen4.g5.model.layers.11.self_attn.compressed_k', 'model.layers.11.self_attn.compressed_k', 1),
            ('qwen4.g5.model.layers.7.self_attn.compressed_k', 'model.layers.7.self_attn.compressed_k', 0),
            ('qwen4.g6.model.layers.11.self_attn.raw_ring', 'model.layers.11.self_attn.raw_ring', 1),
            ('qwen4.g6.model.layers.7.self_attn.raw_ring', 'model.layers.7.self_attn.raw_ring', 0),
        ],
        "max_slots": {0: 166, 1: 164},
    },
    2: {
        "pools": [
            ('qwen4.g0.model.layers.12.linear_attn', 'model.layers.12.linear_attn', 0),
            ('qwen4.g0.model.layers.16.linear_attn', 'model.layers.16.linear_attn', 1),
            ('qwen4.g1.model.layers.13.linear_attn', 'model.layers.13.linear_attn', 0),
            ('qwen4.g1.model.layers.17.linear_attn', 'model.layers.17.linear_attn', 1),
            ('qwen4.g2.model.layers.14.linear_attn', 'model.layers.14.linear_attn', 0),
            ('qwen4.g4.model.layers.15.self_attn', 'model.layers.15.self_attn', 0),
            ('qwen4.g5.model.layers.15.self_attn.compressed_k', 'model.layers.15.self_attn.compressed_k', 0),
            ('qwen4.g6.model.layers.15.self_attn.raw_ring', 'model.layers.15.self_attn.raw_ring', 0),
        ],
        "max_slots": {0: 166, 1: 2},
    },
    3: {
        "pools": [
            ('qwen4.g0.model.layers.20.linear_attn', 'model.layers.20.linear_attn', 0),
            ('qwen4.g1.model.layers.21.linear_attn', 'model.layers.21.linear_attn', 0),
            ('qwen4.g2.model.layers.18.linear_attn', 'model.layers.18.linear_attn', 0),
            ('qwen4.g2.model.layers.22.linear_attn', 'model.layers.22.linear_attn', 1),
            ('qwen4.g4.model.layers.19.self_attn', 'model.layers.19.self_attn', 0),
            ('qwen4.g4.model.layers.23.self_attn', 'model.layers.23.self_attn', 1),
            ('qwen4.g5.model.layers.19.self_attn.compressed_k', 'model.layers.19.self_attn.compressed_k', 0),
            ('qwen4.g5.model.layers.23.self_attn.compressed_k', 'model.layers.23.self_attn.compressed_k', 1),
            ('qwen4.g6.model.layers.19.self_attn.raw_ring', 'model.layers.19.self_attn.raw_ring', 0),
            ('qwen4.g6.model.layers.23.self_attn.raw_ring', 'model.layers.23.self_attn.raw_ring', 1),
        ],
        "max_slots": {0: 166, 1: 164},
    },
    4: {
        "pools": [
            ('qwen4.g0.model.layers.24.linear_attn', 'model.layers.24.linear_attn', 0),
            ('qwen4.g0.model.layers.28.linear_attn', 'model.layers.28.linear_attn', 1),
            ('qwen4.g1.model.layers.25.linear_attn', 'model.layers.25.linear_attn', 0),
            ('qwen4.g1.model.layers.29.linear_attn', 'model.layers.29.linear_attn', 1),
            ('qwen4.g2.model.layers.26.linear_attn', 'model.layers.26.linear_attn', 0),
            ('qwen4.g4.model.layers.27.self_attn', 'model.layers.27.self_attn', 0),
            ('qwen4.g5.model.layers.27.self_attn.compressed_k', 'model.layers.27.self_attn.compressed_k', 0),
            ('qwen4.g6.model.layers.27.self_attn.raw_ring', 'model.layers.27.self_attn.raw_ring', 0),
        ],
        "max_slots": {0: 166, 1: 2},
    },
    5: {
        "pools": [
            ('qwen4.g0.model.layers.32.linear_attn', 'model.layers.32.linear_attn', 0),
            ('qwen4.g1.model.layers.33.linear_attn', 'model.layers.33.linear_attn', 0),
            ('qwen4.g2.model.layers.30.linear_attn', 'model.layers.30.linear_attn', 0),
            ('qwen4.g2.model.layers.34.linear_attn', 'model.layers.34.linear_attn', 1),
            ('qwen4.g4.model.layers.31.self_attn', 'model.layers.31.self_attn', 0),
            ('qwen4.g4.model.layers.35.self_attn', 'model.layers.35.self_attn', 1),
            ('qwen4.g5.model.layers.31.self_attn.compressed_k', 'model.layers.31.self_attn.compressed_k', 0),
            ('qwen4.g5.model.layers.35.self_attn.compressed_k', 'model.layers.35.self_attn.compressed_k', 1),
            ('qwen4.g6.model.layers.31.self_attn.raw_ring', 'model.layers.31.self_attn.raw_ring', 0),
            ('qwen4.g6.model.layers.35.self_attn.raw_ring', 'model.layers.35.self_attn.raw_ring', 1),
        ],
        "max_slots": {0: 166, 1: 164},
    },
    6: {
        "pools": [
            ('qwen4.g0.model.layers.36.linear_attn', 'model.layers.36.linear_attn', 0),
            ('qwen4.g0.model.layers.40.linear_attn', 'model.layers.40.linear_attn', 1),
            ('qwen4.g1.model.layers.37.linear_attn', 'model.layers.37.linear_attn', 0),
            ('qwen4.g1.model.layers.41.linear_attn', 'model.layers.41.linear_attn', 1),
            ('qwen4.g2.model.layers.38.linear_attn', 'model.layers.38.linear_attn', 0),
            ('qwen4.g4.model.layers.39.self_attn', 'model.layers.39.self_attn', 0),
            ('qwen4.g5.model.layers.39.self_attn.compressed_k', 'model.layers.39.self_attn.compressed_k', 0),
            ('qwen4.g6.model.layers.39.self_attn.raw_ring', 'model.layers.39.self_attn.raw_ring', 0),
        ],
        "max_slots": {0: 166, 1: 2},
    },
    7: {
        "pools": [
            ('qwen4.g0.model.layers.44.linear_attn', 'model.layers.44.linear_attn', 0),
            ('qwen4.g1.model.layers.45.linear_attn', 'model.layers.45.linear_attn', 0),
            ('qwen4.g2.model.layers.42.linear_attn', 'model.layers.42.linear_attn', 0),
            ('qwen4.g2.model.layers.46.linear_attn', 'model.layers.46.linear_attn', 1),
            ('qwen4.g4.model.layers.43.self_attn', 'model.layers.43.self_attn', 0),
            ('qwen4.g4.model.layers.47.self_attn', 'model.layers.47.self_attn', 1),
            ('qwen4.g5.model.layers.43.self_attn.compressed_k', 'model.layers.43.self_attn.compressed_k', 0),
            ('qwen4.g5.model.layers.47.self_attn.compressed_k', 'model.layers.47.self_attn.compressed_k', 1),
            ('qwen4.g6.model.layers.43.self_attn.raw_ring', 'model.layers.43.self_attn.raw_ring', 0),
            ('qwen4.g6.model.layers.47.self_attn.raw_ring', 'model.layers.47.self_attn.raw_ring', 1),
        ],
        "max_slots": {0: 166, 1: 164},
    },
}


def pools_of(stage):
    return [SimpleNamespace(tag=t, layer_name=n, storage_index=s) for t, n, s in FR2_STAGES[stage]["pools"]]


# --- C1: the hints on the served producer ------------------------------------


@pytest.mark.parametrize("stage", range(8))
def test_served_hints_are_uniform_per_stage(stage):
    """Stage 0 (PLE) 169, every other stage 168, on BOTH storages: the even
    stages' second storage holds GDN layers only and would otherwise get 2."""
    hints = qwen_staging_hints(pools_of(stage), BLOCK, MAX_LEN)
    want = 169 if stage == 0 else 168
    assert hints == [want] * len(FR2_STAGES[stage]["pools"])


def test_unknown_kind_keeps_the_full_mirror():
    pools = pools_of(1) + [SimpleNamespace(tag="qwen4.g9.model.layers.9.mystery", layer_name="model.layers.9.mystery", storage_index=0)]
    assert qwen_staging_hints(pools, BLOCK, MAX_LEN) == [0] * len(pools)


# --- C7: no FR2 transfer needs more slots on a storage than the hint ----------


@pytest.mark.parametrize("stage", range(8))
def test_no_fr2_transfer_exceeds_its_lease(stage):
    """A transfer needing more slots than N x hint fails at once in tpu_sync
    ("the arena holds only ..."); one lease must hold any max-length load."""
    hint = qwen_staging_hints(pools_of(stage), BLOCK, MAX_LEN)[0]
    for storage, slots in FR2_STAGES[stage]["max_slots"].items():
        assert slots <= hint, (stage, storage, slots, hint)


# --- C8: compressed-key sub-pages ---------------------------------------------


def test_subpages_count_physical_pages():
    """8 compressed blocks per physical page: ceil(82 / 8) = 11 pages, so the
    stage-0 hint is 1+1+1+1+82+11+1 = 98 instead of 169."""
    assert qwen_staging_hints(pools_of(0), BLOCK, MAX_LEN, compressed_slots=8)[0] == 98
    assert qwen_staging_hints(pools_of(1), BLOCK, MAX_LEN, compressed_slots=8)[0] == 97


# --- C2-C6: the registration call ---------------------------------------------


class FakeEngine:
    def __init__(self, keyword_support=True, summary=None):
        self.calls = []
        self.keyword_support = keyword_support
        self.summary = summary or {}

    def register_pools(self, pools, **kwargs):
        self.calls.append((pools, kwargs))
        if kwargs and not self.keyword_support:
            raise TypeError("register_pools() got an unexpected keyword argument 'staging_leases'")
        return dict(self.summary)


def worker(producer=True, staged=False):
    return SimpleNamespace(
        is_producer=producer,
        _staging=object() if staged else None,
        tp_rank=0,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=MAX_LEN)),
    )


def manifest(stage=1):
    pools = pools_of(stage)
    dicts = [{"tag": p.tag} for p in pools]
    return SimpleNamespace(pools=pools, pool_dicts=lambda: dicts)


def register(w, engine, m):
    pc = pytest.importorskip("vllm_torchtpu.models.vllm.qwen4_exp.pd_connector")
    return pc.Qwen4ExpRaidenWorker._register_qwen_pools(w, engine, m, BLOCK)


def test_off_by_default_registers_exactly_as_before(monkeypatch):
    """C2 and C6: unset (and the generic TPU_RAIDEN_POOL_STAGING_LEASES=8 must
    not turn it on): one call, the pool dicts only, no keywords."""
    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_LEASES", raising=False)
    monkeypatch.setenv("TPU_RAIDEN_POOL_STAGING_LEASES", "8")
    engine, m = FakeEngine(), manifest()
    register(worker(), engine, m)
    assert engine.calls == [(m.pool_dicts(), {})]


def test_leases_pass_n_and_one_hint_per_pool(monkeypatch):
    """C3: N and the uniform hint for every pool, in manifest order."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_LEASES", "16")
    engine, m = FakeEngine(), manifest(stage=0)
    register(worker(), engine, m)
    assert engine.calls == [
        (m.pool_dicts(), {"staging_leases": 16, "staging_blocks_per_pool": [169] * len(m.pools)})
    ]


def test_consumer_is_never_bounded(monkeypatch):
    """C3: producer only; a staged consumer and a non-producer keep the full mirror."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_LEASES", "16")
    for w in (worker(producer=False), worker(staged=True)):
        engine, m = FakeEngine(), manifest()
        register(w, engine, m)
        assert engine.calls == [(m.pool_dicts(), {})]


def test_old_tpu_sync_falls_back_to_the_full_mirror(monkeypatch, caplog):
    """C4: a tpu_sync without the keywords: warn, then register as before."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_LEASES", "16")
    engine, m = FakeEngine(keyword_support=False), manifest()
    with caplog.at_level(logging.WARNING):
        register(worker(), engine, m)
    assert [kw for _, kw in engine.calls] == [
        {"staging_leases": 16, "staging_blocks_per_pool": [168] * len(m.pools)},
        {},
    ]
    assert "bounded host staging unavailable" in caplog.text


def test_summary_line_reports_the_arena(monkeypatch, caplog):
    """C5: the engagement line a gate reads (mode, leases, hint, bytes, storages)."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_LEASES", "16")
    summary = {"host_staging": {"mode": "bounded", "leases": 16, "bounded_storage_bytes_per_shard": 17_716_838_400,
                                "full_storage_bytes_per_shard": 0, "storages": [{}, {}]}}
    engine, m = FakeEngine(summary=summary), manifest()
    with caplog.at_level(logging.INFO):
        out = register(worker(), engine, m)
    assert out["host_staging"]["mode"] == "bounded"
    assert ("Qwen PD producer host staging rank0: mode=bounded leases=16 hint=168 "
            "bounded_bytes_per_shard=17716838400 full_mirror_bytes_per_shard=0 storages=2") in caplog.text


def test_env_registered_and_outside_the_compile_key(monkeypatch):
    """C6: default 0, and host-only (in the compile-key ignore set)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_LEASES", raising=False)
    assert envs.QWEN4_EXP_PD_STAGING_LEASES == 0
    assert "QWEN4_EXP_PD_STAGING_LEASES" in envs.environment_variables
    assert "QWEN4_EXP_PD_STAGING_LEASES" in _TPU_COMPILE_ENV_IGNORED
