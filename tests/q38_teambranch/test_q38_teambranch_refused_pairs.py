# SPDX-License-Identifier: Apache-2.0
"""q38 TEAMBRANCH: the two key pairs the curve integration refuses (shriharim/q38-curve-1123-int).

1. TPU_QWEN4_HC_PREFILL_COMPARE (Benliang's cmp prefill HC kernel) with q38 PSTEP-HC's TPU_QWEN4_HC_PREFILL_FOLD or
   TPU_QWEN4_HC_PREFILL_DOWN_ASM: GatedResidual.__init__ (model build) and the prefill mix builders raise ValueError.
2. VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND (q38 PSTEP-IDX) with VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS (Benliang's
   compare-core counters, with VLLM_TPU_QSA_PREFILL_COMPARE_CORE): Qwen4ExpQSA.bind_pool (KV-cache bind at startup) raises.

Each member alone does not raise. Every check runs on a CPU host: op builders are replaced by stand-ins.
"""
import importlib
import sys
import types
from functools import partial

import pytest

HC_KEYS = ("TPU_QWEN4_HC_PREFILL_COMPARE", "TPU_QWEN4_HC_PREFILL_FOLD", "TPU_QWEN4_HC_PREFILL_DOWN_ASM",
           "TPU_QWEN4_HC_PREFILL_FUSED_MIX")
QSA_KEYS = ("VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND", "VLLM_TPU_QSA_PREFILL_COMPARE_CORE",
            "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS", "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "VLLM_TPU_MEGAKERNEL",
            "VLLM_TPU_QSA_INDEXER_SHARD")


def _clear(monkeypatch, keys):
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def _gated_residual():
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual, HyperConnectionConfig

    def build():
        return GatedResidual(HyperConnectionConfig(hc_count=4, hidden_size=256, params_dtype=torch.bfloat16, hc_lowrank=16,
                                                   rms_norm_eps=1e-6, hc_per_branch_norm=True))
    return build


@pytest.mark.parametrize("pair", ["TPU_QWEN4_HC_PREFILL_FOLD", "TPU_QWEN4_HC_PREFILL_DOWN_ASM"])
def test_hc_prefill_compare_pair_refused_at_model_build(monkeypatch, pair):
    build = _gated_residual()
    _clear(monkeypatch, HC_KEYS)
    monkeypatch.setenv("TPU_QWEN4_HC_PREFILL_COMPARE", "1")
    monkeypatch.setenv(pair, "1")
    with pytest.raises(ValueError, match="TPU_QWEN4_HC_PREFILL_COMPARE"):
        build()


@pytest.mark.parametrize("alone", ["TPU_QWEN4_HC_PREFILL_COMPARE", "TPU_QWEN4_HC_PREFILL_FOLD", "TPU_QWEN4_HC_PREFILL_DOWN_ASM", None])
def test_hc_prefill_each_member_alone_builds(monkeypatch, alone):
    build = _gated_residual()
    _clear(monkeypatch, HC_KEYS)
    if alone:
        monkeypatch.setenv(alone, "1")
    build()


def _adapter(monkeypatch):
    va = importlib.import_module("vllm_torchtpu.kernels.qwen4_exp.vllm_adapter")
    made = []

    class _Op:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn

        def register_fake(self, f):
            return f

    monkeypatch.setattr(va, "pallas", types.SimpleNamespace(jax_op=lambda name, fn, **kw: made.append(name) or _Op(name, fn)))
    return va, made


def test_hc_prefill_builders_refuse_the_pair_and_name_each_member(monkeypatch):
    va, made = _adapter(monkeypatch)
    inj, comb = va.build_prefill_inj_mix_op.__wrapped__, va.build_prefill_combine_mix_op.__wrapped__
    with pytest.raises(ValueError, match="TPU_QWEN4_HC_PREFILL_DOWN_ASM"):
        inj(4, 256, 1e-6, 256, False, "vpu", True, True)
    for kw in ({"reread": True}, {"down_asm": True}, {"reread": True, "down_asm": True}):
        with pytest.raises(ValueError, match="TPU_QWEN4_HC_PREFILL_COMPARE"):
            comb(4, 256, 1e-6, 256, False, "vpu", True, **kw)
    assert not made
    inj(4, 256, 1e-6, 256, False, "vpu", True)
    inj(4, 256, 1e-6, 256, False, "vpu", False, True)
    inj(4, 256, 1e-6, 256, False, "vpu")
    comb(4, 256, 1e-6, 256, False, "vpu", True)
    comb(4, 256, 1e-6, 256, False, "vpu", False, True, True)
    comb(4, 256, 1e-6, 256, False, "vpu")
    tails = [n.split("_e", 1)[1].split("_", 1)[1] if "_" in n.split("_e", 1)[1] else "" for n in made]
    assert tails == ["cmp_v1", "da_v1", "v1", "cmp_v1", "rr_da_v1", "v1"], made


def _qsa(monkeypatch):
    try:
        importlib.import_module("torch_tpu._internal.pallas")
    except Exception:  # noqa: BLE001
        for name in ("torch_tpu", "torch_tpu._internal", "torch_tpu._internal.pallas"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["torch_tpu._internal"].pallas = sys.modules["torch_tpu._internal.pallas"]
    qsa_op = importlib.import_module("vllm_torchtpu.layers.adapter.custom_ops.qsa_op")
    qsa_model = importlib.import_module("vllm_torchtpu.models.vllm.qwen4_exp.qsa")
    stats = importlib.import_module("vllm_torchtpu.models.vllm.qwen4_exp.compare_core_stats")
    sconfig = importlib.import_module("vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config")

    def op(tag):
        def f(*args, **kwargs):
            return tag
        f.tag = tag
        return f

    monkeypatch.setattr(qsa_op, "build_pooled_qsa_op", lambda: op("off"))
    monkeypatch.setattr(qsa_op, "build_pooled_qsa_nocond_op", lambda: op("nocond"))
    monkeypatch.setattr(qsa_op, "build_pooled_qsa_rows_op", lambda: op("rows"))
    monkeypatch.setattr(stats, "register", lambda layer, device: "stats-buffer")
    if hasattr(sconfig, "_layer_scheduling_group_bases"):
        monkeypatch.setattr(sconfig, "_layer_scheduling_group_bases", {})
    first = {"compress_ratio": 4, "token_topk": 2048, "norm_eps": 1e-6, "rotary_dim": 64}

    def bind():
        layer = types.SimpleNamespace(prefix="model.layers.6.self_attn", _op=partial(op(None), **first), max_query_len=1,
                                      pcp=1, main_kv_dcp=1, tp_size=1, tp_rank=0, num_kv_heads=2, single_graph=False)
        qsa_model.Qwen4ExpQSA.bind_pool(layer, types.SimpleNamespace(device="cpu"),
                                        compressed_owner=("model.layers.6.compressed_k", 2), main_owner=("model.layers.6", 1))
        return layer
    return bind


def test_nocond_with_compare_core_stats_refused_at_bind(monkeypatch):
    bind = _qsa(monkeypatch)
    _clear(monkeypatch, QSA_KEYS)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS", "30")
    with pytest.raises(ValueError, match="VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND"):
        bind()


@pytest.mark.parametrize("alone", ["nocond", "stats", "none"])
def test_nocond_and_compare_core_stats_each_alone_bind(monkeypatch, alone):
    bind = _qsa(monkeypatch)
    _clear(monkeypatch, QSA_KEYS)
    if alone == "nocond":
        monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND", "1")
    elif alone == "stats":
        monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1")
        monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS", "30")
    layer = bind()
    want = {"nocond": "nocond", "stats": "off", "none": "off"}[alone]
    fn = layer._op
    while isinstance(fn, partial) and not hasattr(fn.func, "tag"):
        fn = fn.func
    assert fn.func.tag == want, fn
    if alone == "stats":
        assert layer._op.keywords.get("cmp_stats") == "stats-buffer"
    else:
        assert "cmp_stats" not in layer._op.keywords
