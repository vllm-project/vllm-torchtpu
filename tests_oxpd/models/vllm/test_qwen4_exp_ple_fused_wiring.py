# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_FUSED / VLLM_TPU_PLE_CONV_ONEHOT wiring in Qwen4ExpPooledPLE
(CPU; device ops are recording doubles). The kernels' numerics are tested in
tests/kernels/test_qwen4_exp_ple_fused.py and on TPU."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.vllm.test_qwen4_exp import (
    _PLE,
    HC,
    HIDDEN,
    _randomize_ple,
    _set_tp,
    _text_config,
)
from vllm_torchtpu.models.vllm.qwen4_exp.ple import PLEInputs


class _Group:
    rank_in_group = 0
    world_size = 1

    def all_reduce(self, rows):
        return rows


def _adapter(monkeypatch, env, flipped=None):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage, pooled_ple
    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel

    for name in (
        "VLLM_TPU_PLE_FUSED",
        "VLLM_TPU_PLE_FUSED_TILE",
        "VLLM_TPU_PLE_FUSED_TAP_MODE",
        "VLLM_TPU_PLE_CONV_ONEHOT",
        "Q38_PROBE_PLE_ONEHOT_MUTANT",
        "VLLM_TPU_PLE_FUSED_NORM",
        "VLLM_TPU_PLE_FUSED_PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _set_tp(monkeypatch, 0, 1)
    group = _Group()

    def raw_lookup(rows, width, chunk_tokens=8):
        def run(table, ids, vocab_start):
            result = table.reshape(rows, width)[ids.clamp(0, rows - 1)].clone()
            result[(ids < 0) | (ids >= rows)] = 0
            return result.flatten(-2)

        return run

    monkeypatch.setattr(ple_storage, "build_raw_lookup_op", raw_lookup)
    monkeypatch.setattr(
        ple_storage, "build_decode_op", lambda: lambda raw, scale: raw.view(torch.bfloat16)
    )
    log = SimpleNamespace(conv=[], spec=0, fused=[], built=[])

    def conv_builder(dilation, state_len, past_matmul=False):
        log.built.append(("conv", dilation, state_len, past_matmul))

        def run(x, weight, pool, starts, ids, initial):
            log.conv.append(x.shape[0])
            return torch.zeros_like(x)

        return run

    def spec_builder(dilation, state_len):
        def run(x, *args):
            log.spec += 1
            return torch.zeros_like(x)

        return run

    def fused_builder(streams, hidden, eps, dilation, state_len, tile, tap_mode, norm_mode="sum",
                      precision="torch", weights_kn=False):
        log.precision = precision
        log.weights_kn = weights_kn
        log.built.append(("fused", streams, hidden, eps, dilation, state_len, tile, tap_mode))
        log.norm = norm_mode

        def run(h, e, key_w, value_w, wk, wq, wc, conv_w, pool, starts, ids, initial, valid):   # key first (fix 4)
            log.fused.append(
                dict(h=h, e=e, key_w=key_w, value_w=value_w, wq=wq, wk=wk, wc=wc,
                     conv_w=conv_w, pool=pool, starts=starts, ids=ids, initial=initial,
                     valid=valid)
            )
            return torch.full_like(h, 7)

        return run

    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", conv_builder)
    monkeypatch.setattr(pooled_ple, "build_pooled_speculative_short_conv_op", spec_builder)
    monkeypatch.setattr(pooled_ple, "build_pooled_fused_ple_op", fused_builder)
    with set_default_torch_dtype(torch.bfloat16):
        reference = pooled_ple.Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, "ple")
        _randomize_ple(reference)
        adapted = pooled_ple.Qwen4ExpPooledPLE(
            _text_config(**_PLE), 0, None, "ple", parallel=PLEParallel(group, group,
            embedding_across_dp=False), table_group=group,
        )
    weights = dict(reference.state_dict())
    table = weights.pop("ple_embedding.ngram_embedding.weight")
    for i, rows in enumerate(table.split(22)):
        weights[f"ple_embedding.ngram_embedding.shard_{i}.weight"] = rows
    adapted.load_checkpoint(iter(weights.items()))
    adapted.finalize_weights()
    pool = torch.zeros(5, 16, dtype=torch.bfloat16)
    if flipped is not None:   # (key, value) as the TPU linear method marks them after loading
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        setattr(adapted.key_proj, WEIGHT_FLIPPED_ATTR, flipped[0])
        setattr(adapted.value_proj, WEIGHT_FLIPPED_ATTR, flipped[1])
    adapted.bind_pool(pool)
    return adapted, pool, log


def _step(adapted, tokens, speculative=None):
    ids = torch.arange(tokens, dtype=torch.int64) % 7 + 4
    starts = torch.tensor([0, tokens // 2, tokens, tokens], dtype=torch.int32)
    slots = torch.tensor([2, 3, -1], dtype=torch.int32)
    initial = torch.tensor([True, False, False])
    contexts = torch.full((3, 2), _PLE["eos_token_id"], dtype=torch.int64)
    hidden = torch.randn(tokens, HC * HIDDEN).to(torch.bfloat16)
    valid = starts[-1:]
    with torch.no_grad():
        out = adapted.forward_with_history(
            hidden, PLEInputs(ids, starts, contexts), slots, initial, valid, None, speculative
        )
    return out, dict(hidden=hidden, starts=starts, slots=slots, initial=initial, valid=valid)


def test_fused_flag_routes_whole_tiles_and_passes_the_layer_tensors(monkeypatch):
    adapted, pool, log = _adapter(
        monkeypatch,
        {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "8",
         "VLLM_TPU_PLE_FUSED_TAP_MODE": "mxu"},
    )
    built = [b for b in log.built if b[0] == "fused"]
    assert built == [("fused", HC, HIDDEN, float(_PLE["rms_norm_eps"]), _PLE["ngram_size"],
                      (_PLE["ple_conv_kernel_size"] - 1) * _PLE["ngram_size"], 8, "mxu")]
    out, step = _step(adapted, 16)
    assert len(log.fused) == 1 and log.conv == []
    assert torch.all(out == 7)
    got = log.fused[0]
    assert got["h"] is step["hidden"]
    assert got["key_w"] is adapted.key_proj.weight
    assert got["value_w"] is adapted.value_proj.weight
    assert got["wq"] is adapted.norm_query.weight
    assert got["wk"] is adapted.norm_key.weight
    assert got["wc"] is adapted.norm_conv.weight
    assert got["conv_w"] is adapted.conv1d.weight   # whole [C, 1, taps]; sliced inside the op
    assert got["pool"] is pool
    assert got["starts"] is step["starts"] and got["ids"] is step["slots"]
    assert got["initial"] is step["initial"] and got["valid"] is step["valid"]
    assert got["e"].shape == (16, _PLE["ple_embed_dim"]) and got["e"].dtype == torch.bfloat16


@pytest.mark.parametrize("tokens", [12, 4])
def test_fused_flag_keeps_partial_tiles_on_the_served_path(monkeypatch, tokens):
    adapted, _, log = _adapter(
        monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "8"}
    )
    _step(adapted, tokens)
    assert log.fused == [] and log.conv == [tokens]


def test_fused_flag_keeps_speculative_steps_on_the_served_path(monkeypatch):
    adapted, _, log = _adapter(
        monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "8"}
    )
    marker = (torch.zeros(3, dtype=torch.int32),) * 3
    _step(adapted, 16, speculative=marker)
    assert log.fused == [] and log.spec == 1


def test_flags_off_build_nothing_new(monkeypatch):
    adapted, _, log = _adapter(monkeypatch, {})
    assert [b for b in log.built if b[0] == "fused"] == []
    assert [b for b in log.built if b[0] == "conv"] == [
        ("conv", _PLE["ngram_size"], (_PLE["ple_conv_kernel_size"] - 1) * _PLE["ngram_size"], False)
    ]
    _step(adapted, 16)
    assert log.fused == [] and log.conv == [16]


@pytest.mark.parametrize("mode", ["single", "single_mxu"])
def test_single_pass_modes_reach_the_fused_builder(monkeypatch, mode):
    _, _, log = _adapter(
        monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "8",
                      "VLLM_TPU_PLE_FUSED_TAP_MODE": mode},
    )
    assert [b[-1] for b in log.built if b[0] == "fused"] == [mode]


def test_onehot_flag_reaches_the_pooled_conv_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_CONV_ONEHOT": "1"})
    assert [b[3] for b in log.built if b[0] == "conv"] == [True]


def test_probe_mutant_flag_reaches_the_pooled_conv_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"Q38_PROBE_PLE_ONEHOT_MUTANT": "1"})
    assert [b[3] for b in log.built if b[0] == "conv"] == ["mutant"]


def test_norm_mode_reaches_the_fused_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_NORM": "mean"})
    assert log.norm == "mean"


def test_precision_reaches_the_fused_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_FUSED": "1"})
    assert log.precision == "torch"
    _, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_PRECISION": "xla"})
    assert log.precision == "xla"


def test_pooled_fused_ple_op_builds_through_jax_op():
    """build_pooled_fused_ple_op wraps pooled_fused_ple with torch_tpu's pallas.jax_op, which needs every
    parameter annotated (srv1-F j-4dfa116b failed engine init without them). Build the real op, unmocked."""
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    ple_op.build_pooled_fused_ple_op.cache_clear() if hasattr(ple_op.build_pooled_fused_ple_op, "cache_clear") else None
    run = ple_op.build_pooled_fused_ple_op(4, 2560, 1e-6, 3, 9, 128, "single", "sum", "xla")
    assert callable(run)


def _flipped_like_serving(weight):
    """Run a dense weight through the TPU linear method's post-load hook, as serving does: it is stored
    [in, out] afterwards and the layer carries WEIGHT_FLIPPED_ATTR."""
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedLinearMethod

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False) and tuple(layer.weight.shape) == tuple(weight.shape[::-1])
    return layer


@pytest.mark.parametrize("single", [False, True])
def test_fused_ple_matches_reference_through_the_tpu_linear_load_path(single):
    """srv1-F re-run (j-a8ea9be6) failed because the served key/value weights are [in, out] after loading.
    Weights flipped by the real TPU linear method + weights_kn read from the layer give the same bits as the
    torch-layout weights with weights_kn False. The square value_proj is the case a wrong axis would not crash on."""
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu  # noqa: F401

    from vllm_torchtpu.kernels.qwen4_exp import ple_fused
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    S, H, E, T, R, SL = 4, 128, 128, 256, 8, 9
    C = S * H
    g = torch.Generator().manual_seed(11)
    rnd = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16)  # noqa: E731
    key_t, value_t = rnd(C, E, sc=E ** -0.5), rnd(H, E, sc=E ** -0.5)   # torch Linear layout [out, in]
    key_l, value_l = _flipped_like_serving(key_t), _flipped_like_serving(value_t)
    kn = bool(getattr(key_l, WEIGHT_FLIPPED_ATTR)) and bool(getattr(value_l, WEIGHT_FLIPPED_ATTR))
    j = lambda t: jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)  # noqa: E731
    h, e = j(rnd(T, C)), j(rnd(T, E))
    one = lambda: (1.0 + jnp.asarray(rnd(C, sc=0.1).float().numpy())).reshape(1, C)  # noqa: E731
    wq, wk, wc = one(), one(), one()
    conv_w, history = j(rnd(C, 4, sc=0.5)), j(rnd(R, SL, C))
    starts = jnp.asarray([0, 80, 256] + [256] * (R - 2), jnp.int32)
    ids = jnp.asarray([5, 9] + [-1] * (R - 2), jnp.int32)
    init = jnp.asarray([True, False] + [False] * (R - 2))
    valid = jnp.asarray(256, jnp.int32)
    fn = ple_fused.ple_single_after_lookup if single else ple_fused.ple_fused_after_lookup
    kw = dict(streams=S, hidden=H, eps=1e-6, dilation=3, token_tile=128, interpret=True)
    ref_out, ref_win = fn(h, e, j(key_t), j(value_t), wq, wk, wc, conv_w, history, starts, ids, init, valid, **kw)
    out, win = fn(h, e, j(key_l.weight), j(value_l.weight), wq, wk, wc, conv_w, history, starts, ids, init, valid,
                  **kw, weights_kn=kn)
    np.testing.assert_array_equal(np.asarray(out).view(np.uint16), np.asarray(ref_out).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(win).view(np.uint16), np.asarray(ref_win).view(np.uint16))


@pytest.mark.parametrize("key_flipped,value_flipped,built,kn", [
    (False, False, True, False), (True, True, True, True), (True, False, False, None)])
def test_bind_pool_reads_the_weight_layout(monkeypatch, key_flipped, value_flipped, built, kn):
    """bind_pool passes weights_kn from WEIGHT_FLIPPED_ATTR; a key/value mismatch keeps the served ops."""
    adapted, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_FUSED": "1"}, flipped=(key_flipped, value_flipped))
    assert (adapted._fused is not None) == built
    if built:
        assert log.weights_kn is kn


def _first_use_order(adapted, tokens):
    """The order in which one forward first hands each PLE layer parameter to a dispatched op (aten or custom).

    Dynamo lifts a module parameter into a graph placeholder at its first use by a traced op, so this order is the
    order those parameters take in the step's subgraph inputs (I). Under shape-variant compilation a token bucket can
    move between the fused trace and the served trace only if their subgraph inputs match in order (srv1-F3
    j-afef8265 failed engine init with ShapeSpecializationError because the fused trace lifted conv1d.weight first)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    params = {
        "key": adapted.key_proj.weight, "value": adapted.value_proj.weight, "norm_query": adapted.norm_query.weight,
        "norm_key": adapted.norm_key.weight, "norm_conv": adapted.norm_conv.weight, "conv": adapted.conv1d.weight,
    }
    ptrs = {p.data_ptr(): name for name, p in params.items()}
    seen = []

    class _Order(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            for a in list(args) + list((kwargs or {}).values()):
                for t in (a if isinstance(a, (list, tuple)) else [a]):
                    if isinstance(t, torch.Tensor) and t.data_ptr() in ptrs and ptrs[t.data_ptr()] not in seen:
                        seen.append(ptrs[t.data_ptr()])
            return func(*args, **(kwargs or {}))

    with _Order():
        _step(adapted, tokens)
    return seen, [(name, str(params[name].dtype), tuple(params[name].shape)) for name in seen]


def _op_doubles(monkeypatch):
    """Fused and conv doubles as real torch custom ops, so each call is one dispatched op taking every argument in
    order, as the served pallas.jax_op custom ops are."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_ple

    @torch.library.custom_op("q38test_ple::fused", mutates_args=())
    def fused_op(h: torch.Tensor, e: torch.Tensor, key_w: torch.Tensor, value_w: torch.Tensor, wk: torch.Tensor,
                 wq: torch.Tensor, wc: torch.Tensor, conv_w: torch.Tensor, pool: torch.Tensor, starts: torch.Tensor,
                 ids: torch.Tensor, initial: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.full_like(h, 7)

    @torch.library.custom_op("q38test_ple::conv", mutates_args=())
    def conv_op(x: torch.Tensor, weight: torch.Tensor, pool: torch.Tensor, starts: torch.Tensor, ids: torch.Tensor,
                initial: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    monkeypatch.setattr(pooled_ple, "build_pooled_fused_ple_op", lambda *a, **k: fused_op)
    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", lambda *a, **k: conv_op)


def test_fused_and_served_steps_use_the_layer_parameters_in_the_same_order(monkeypatch):
    adapted, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_FUSED": "1", "VLLM_TPU_PLE_FUSED_TILE": "8"})
    _op_doubles(monkeypatch)
    adapted.bind_pool(adapted.cache_owners[0])   # rebuild the ops through the custom-op doubles
    fused, fused_sig = _first_use_order(adapted, 16)     # whole tiles: the fused op
    served, served_sig = _first_use_order(adapted, 12)   # a partial tile: the served ops
    print("FIRST_USE fused", fused_sig, "served", served_sig)
    assert set(fused) == set(served) == {"key", "value", "norm_query", "norm_key", "norm_conv", "conv"}, (fused, served)
    # Fix 4: the NAME order must match, not only the (dtype, shape) order. shape_variants hands a compiled bucket
    # between the two traces when their subgraph signatures match by (dtype, shape) in order and calls it
    # positionally (compilation/shape_variants.py BucketExecutables.get), so two equal-shaped parameters (the [C]
    # norm weights) trading places would be silently swapped in every handed bucket.
    assert fused == served, (fused, served)
    assert [x[1:] for x in fused_sig] == [x[1:] for x in served_sig], (fused_sig, served_sig)
