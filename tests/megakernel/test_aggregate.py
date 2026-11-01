# SPDX-License-Identifier: Apache-2.0
"""The megakernel aggregate role: selection, row split, loader sharing, caching."""

import json
import types

import numpy as np
import pytest

from vllm_torchtpu.megakernel import backend
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3 import program as k3
from vllm_torchtpu.megakernel.kimi_k3.aggregate import (
    KimiAggregate,
    role_config,
    split_rows,
)
from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader
from vllm_torchtpu.megakernel.program import RequestInput

pytestmark = pytest.mark.cpu_test


def _vllm_config(kv_role=None, extra=None):
    kv = None
    if kv_role is not None:
        kv = types.SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return types.SimpleNamespace(
        load_config=types.SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config=extra or {"presharded": "/p"},
        ),
        kv_transfer_config=kv,
        model_config=None,
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "kv_role, expected",
    [(None, "aggregate"), ("kv_producer", "prefill"), ("kv_consumer", "decode")],
)
def test_role_selection(kv_role, expected):
    assert backend.role(_vllm_config(kv_role)) == expected


def test_role_rejects_kv_both_and_unknown_aggregate_keys():
    with pytest.raises(ValueError, match="one role of a P/D pair"):
        backend.role(_vllm_config("kv_both"))
    with pytest.raises(ValueError, match="aggregate role"):
        backend.role(_vllm_config(None, {"presharded": "/p", "draft": "/d"}))
    config = _vllm_config(
        None, {"presharded": "/p", "synthetic_acceptance_lengths": {"7": 3.78}}
    )
    assert backend.synthetic_acceptance_lengths(config) == {7: 3.78}
    assert backend.attention_dp(config) == 1
    assert backend.draft(config) is None


def test_backend_not_selected():
    config = _vllm_config()
    config.load_config.load_format = "auto"
    assert backend.role(config) is None


def _engine_config(**overrides):
    values = dict(
        role="aggregate",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=4,
        max_model_len=1 << 20,
        max_num_batched_tokens=8256,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=((1, 1, 7), (2, 4, 3)),
    )
    values.update(overrides)
    return MegakernelEngineConfig(**values)


def test_aggregate_capabilities_and_role_configs():
    config = _engine_config()
    caps = k3.capabilities(config)
    assert caps.roles == frozenset({"prefill", "decode"})
    assert caps.speculation == "dspark" and caps.num_speculative_tokens == 7
    assert caps.page_aligned_chunks and caps.block_copies
    assert caps.max_step_tokens == 8256
    prefill = role_config(config, "prefill")
    assert prefill.role == "prefill" and prefill.max_num_batched_tokens == 8256
    assert prefill.num_speculative_tokens == 0
    k3.check(prefill)
    decode = role_config(config, "decode")
    assert decode.role == "decode" and decode.num_speculative_tokens == 7
    k3.check(decode)
    with pytest.raises(NotImplementedError, match="decode-only"):
        k3.check(_engine_config(mamba_ssm_cache_dtype="bfloat16"))
    with pytest.raises(NotImplementedError, match="async"):
        k3.check(_engine_config(async_scheduling=False))


def _row(req_id, computed, scheduled, num_tokens, prompt):
    return RequestInput(
        req_id=req_id,
        computed=computed,
        scheduled=scheduled,
        token_ids=np.arange(num_tokens + 8, dtype=np.int32),
        num_tokens=num_tokens,
        num_prompt_tokens=prompt,
        block_tables=(np.zeros(4, np.int32),),
    )


def _key(row):
    return (row.computed, row.scheduled, row.num_tokens, row.num_prompt_tokens)


def test_split_rows():
    rows = [
        _row("chunk", 0, 4096, 6000, 6000),  # a prompt chunk
        _row("last", 4096, 1904, 6000, 6000),  # the prompt's last chunk
        _row("decoding", 7000, 8, 7001, 6000),  # verify of 7 drafts
        _row("idle", 7000, 0, 7001, 6000),  # not scheduled
        _row("hit", 5999, 1, 6000, 6000),  # cached up to the last token
        _row("restarted", 0, 6500, 6500, 6000),  # preempted, recomputing
        _row("to_last", 0, 5999, 6000, 6000),  # a chunk ending at the last token
    ]
    prefill, decode = split_rows(rows, {"decoding", "idle", "restarted"})
    assert [_key(r) for r in prefill] == [
        (0, 4096, 5999, 5999),
        (4096, 1903, 5999, 5999),
        (7000, 0, 7001, 6000),
        (7000, 0, 7001, 6000),
        (5999, 0, 5999, 5999),
        (0, 6499, 6499, 6499),
        (0, 5999, 5999, 5999),
    ]
    assert [_key(r) for r in decode] == [
        (0, 0, 6000, 6000),
        (5999, 1, 6000, 6000),
        (7000, 8, 7001, 6000),
        (7000, 0, 7001, 6000),
        (5999, 1, 6000, 6000),
        (6499, 1, 6500, 6000),
        (0, 0, 6000, 6000),
    ]
    assert [r.req_id for r in prefill] == [r.req_id for r in rows]


def _write_preshard(root, rank):
    path = root / f"rank{rank:02d}"
    path.mkdir(parents=True)
    families, blob, offset = {}, [], 0
    dense = {
        name: (2, 3) for name in rank_loader.PREFILL_IO + rank_loader.PREFILL_COMMON
    }
    dense.update(
        {name: (2, 3) for name in rank_loader.PREFILL_KDA + rank_loader.PREFILL_MLA}
    )
    dense.update({name: (2, 3) for name in rank_loader.PREFILL_MOE_DENSE})
    dense.update(k_o=(2, 768, 3), m_o=(2, 768, 3))
    for name, shape in dense.items():
        value = np.full(shape, rank, np.float32)
        families[name] = {
            "file": "dense.bin",
            "offset": offset,
            "shape": list(shape),
            "dtype": "float32",
        }
        blob.append(value.tobytes())
        offset += value.nbytes
    (path / "dense.bin").write_bytes(b"".join(blob))
    for name in rank_loader.EXPERT_FAMILIES:
        value = np.full((2, 3, 4), rank, np.uint8)
        (path / f"{name}.bin").write_bytes(value.tobytes())
        families[name] = {
            "files": [f"{name}.bin"],
            "layer_ranges": [[0, 2]],
            "shape": [2, 3, 4],
            "dtype": "uint8",
        }
    (path / "index.json").write_text(
        json.dumps({"rank": rank, "layers": 93, "families": families})
    )


def _write_cache(root):
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    families = {}
    for name in (
        "latent_down",
        "latent_up",
        "shared_gu_0",
        "shared_gu_1",
        "shared_down_0",
        "shared_down_1",
    ):
        np.zeros((2, 2), np.float32).tofile(cache / f"{name}.bin")
        families[name] = {"file": f"{name}.bin", "shape": [2, 2], "dtype": "float32"}
    (cache / "layout.json").write_text(json.dumps({"layers": 93, "families": families}))
    return cache


@pytest.fixture
def preshard(tmp_path):
    for rank in (0, 1):
        _write_preshard(tmp_path, rank)
    return tmp_path, _write_cache(tmp_path)


def test_prefill_loader_skips_shared_families(monkeypatch, preshard):
    root, cache = preshard
    monkeypatch.setattr(
        rank_loader, "_config", lambda: KernelConfig(prefill_expert_parallel=8)
    )
    full = {
        name
        for _, name, _ in rank_loader.iter_prefill_rank(
            root, cache, 0, split_experts=True
        )
    }
    skip = frozenset(rank_loader.PREFILL_SHARED)
    left = [
        (g, n)
        for g, n, _ in rank_loader.iter_prefill_rank(
            root, cache, 0, split_experts=True, skip=skip
        )
    ]
    assert sorted(n for _, n in left) == sorted(
        ["k_o", "m_o", "latent_down", "latent_up", "shared_gu", "shared_down"]
    )
    assert ("kda", "k_o") in left and ("mla", "m_o") in left
    assert {n.partition("@")[0] for n in full} == set(rank_loader.PREFILL_SHARED) | {
        n for _, n in left
    }


def test_prefill_loader_shares_experts_only_at_ep8(monkeypatch, preshard):
    root, cache = preshard
    monkeypatch.setattr(
        rank_loader, "_config", lambda: KernelConfig(prefill_expert_parallel=16)
    )
    with pytest.raises(ValueError, match="EP8"):
        list(
            rank_loader.iter_prefill_rank(
                root, cache, 0, skip=frozenset(rank_loader.EXPERT_FAMILIES)
            )
        )
    monkeypatch.setattr(
        rank_loader, "_config", lambda: KernelConfig(prefill_expert_parallel=8)
    )
    with pytest.raises(ValueError, match="EP8"):
        list(
            rank_loader.iter_prefill_rank(root, cache, 0, skip=frozenset({"expert_gu"}))
        )
    with pytest.raises(ValueError, match="cannot be shared"):
        list(rank_loader.iter_prefill_rank(root, cache, 0, skip=frozenset({"k_o"})))


def test_aggregate_hands_the_decoders_tensors_to_the_prefill():
    weights = {
        name: object() for name in rank_loader.PREFILL_SHARED + ("k_o", "latent_down")
    }
    calls = []
    aggregate = object.__new__(KimiAggregate)
    aggregate.decoder = types.SimpleNamespace(
        load_weights=lambda: calls.append("decode"),
        ops=types.SimpleNamespace(weights=weights),
    )
    aggregate.prefiller = types.SimpleNamespace(
        load_weights=lambda shared: calls.append(shared)
    )
    aggregate.rank, aggregate._loaded = 1, False
    aggregate.load_weights()
    assert calls[0] == "decode"
    assert calls[1] == {n: weights[n] for n in rank_loader.PREFILL_SHARED}
    aggregate.load_weights()
    assert len(calls) == 2


def test_aggregate_aligns_prefiller_pool_order_to_decoder_mesh_rank(
    monkeypatch, tmp_path
):
    from vllm_torchtpu.megakernel.kimi_k3 import aggregate as agg_mod
    from vllm_torchtpu.megakernel.kimi_k3 import prefill as prefill_mod
    from vllm_torchtpu.megakernel.kimi_k3 import program as prog_mod

    seen = {}

    def fake_build_decoder(cfg, presharded, draft, *, tp_rank):
        seen["decoder_tp_rank"] = tp_rank
        return types.SimpleNamespace(rank=8)

    def fake_prefiller(cfg, presharded, draft, *, tp_rank, compute_logits=True):
        seen["prefiller_tp_rank"] = tp_rank
        seen["compute_logits"] = compute_logits
        return types.SimpleNamespace(rank=8)

    monkeypatch.setattr(
        agg_mod, "active", lambda: KernelConfig(prefill_expert_parallel=8)
    )
    monkeypatch.setattr(prog_mod, "build_decoder", fake_build_decoder)
    monkeypatch.setattr(prefill_mod, "KimiPrefiller", fake_prefiller)
    engine = KimiAggregate(_engine_config(), tmp_path, tp_rank=0)
    assert engine.rank == 8
    assert seen == {
        "decoder_tp_rank": 0,
        "prefiller_tp_rank": 8,
        "compute_logits": False,
    }


def test_aggregate_caches_prompt_tokens_only():
    from vllm_torchtpu import _cache_only_prefill_tokens

    seen = []
    coordinator = types.SimpleNamespace(cache_blocks=lambda request, n: seen.append(n))
    scheduler = types.SimpleNamespace(
        vllm_config=_vllm_config(None),
        kv_cache_manager=types.SimpleNamespace(coordinator=coordinator),
    )
    assert _cache_only_prefill_tokens(scheduler)
    request = types.SimpleNamespace(num_prompt_tokens=6000)
    coordinator.cache_blocks(request, 4096)
    coordinator.cache_blocks(request, 7000)
    assert seen == [4096, 5999]
    pair = types.SimpleNamespace(
        vllm_config=_vllm_config("kv_consumer"),
        kv_cache_manager=types.SimpleNamespace(coordinator=types.SimpleNamespace()),
    )
    assert not _cache_only_prefill_tokens(pair)
