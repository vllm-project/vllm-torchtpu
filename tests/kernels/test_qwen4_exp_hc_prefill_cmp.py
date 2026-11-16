# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_PREFILL_COMPARE on CPU (interpret mode): hc_prefill_cmp's
numerics, its combine prologue against the separate combine, and the prefill
fold of each layer's mlp combine into the next layer's attn mix."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from jax.experimental.pallas import tpu as pltpu

from tests.compilation.dynamo_input_order import parameters_by_first_use
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_cmp import hc_prefill_mix_cmp
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

# Flash's low rank (320: the inject rows sit in lanes 64-67 of the down
# projection's third column tile) at a narrower hidden size.
BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 256, 320, 1e-6, 256
WIDTH = BRANCHES * HIDDEN
INTERPRET = pltpu.InterpretParams()


def _inputs(seed, tokens):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = (
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    )
    return residual, block, injection.astype(jnp.float32), weights


def _np(a):
    return np.asarray(jnp.asarray(a, jnp.float32), np.float64)


def _bf16(a):
    return a.astype(ml_dtypes.bfloat16).astype(np.float64)


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def _reference(residual, block, injection, weights, prologue):
    """float64 math with the cmp kernel's bf16 rounding points."""
    w, down, up, inject = (_np(a) for a in weights)
    h = _np(residual)
    if prologue:
        blk, inj = _np(block), _np(injection)
        h = np.concatenate(
            [_bf16(h[:, b * HIDDEN : (b + 1) * HIDDEN] + blk * inj[:, b : b + 1]) for b in range(BRANCHES)],
            axis=1,
        )
    norm = []
    for b in range(BRANCHES):
        v = h[:, b * HIDDEN : (b + 1) * HIDDEN]
        r = 1.0 / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + EPS)
        norm.append(_bf16(v * r * (1.0 + w[b * HIDDEN : (b + 1) * HIDDEN])))
    norm = np.concatenate(norm, axis=1)
    z = _bf16(_bf16(norm @ down.T) / BRANCHES)
    gate = _bf16(z * _sigmoid(z))
    acc = 0.0
    for b in range(BRANCHES):
        acc = acc + _sigmoid(_bf16(gate @ up[b * HIDDEN : (b + 1) * HIDDEN].T)) * norm[:, b * HIDDEN : (b + 1) * HIDDEN]
    mixed = _bf16(acc / BRANCHES)
    inj_out = 2.0 * _sigmoid(_bf16(norm @ inject.T) / BRANCHES)
    return mixed, inj_out, h


def _metrics(actual, expected):
    d = np.abs(actual - expected)
    return dict(
        max_rel=float(d.max() / max(np.abs(expected).max(), 1e-30)),
        l2=float(np.linalg.norm(d) / max(np.linalg.norm(expected), 1e-30)),
    )


cmp = functools.partial(
    hc_prefill_mix_cmp, branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, interpret=INTERPRET
)
served = functools.partial(
    hc_prefill_mix, branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, interpret=INTERPRET,
    norm_impl="bf16ep64",
)


@pytest.mark.parametrize("prologue", [True, False])
@pytest.mark.parametrize(
    "tokens,swap",
    [(2 * TILE, True), (3 * TILE, True), (2 * TILE + 48, True), (2 * TILE, False)],
)
def test_cmp_matches_reference_and_served(tokens, swap, prologue):
    """Within bf16 rounding of a float64 reference at the kernel's rounding
    points, no less accurate than the served kernel, and within 1% of max of
    the served kernel (the served one rounds the norm twice). The combined
    streams are the served kernel's bit for bit."""
    residual, block, injection, weights = _inputs(5 * tokens + int(prologue), tokens)
    partial = bool(tokens % TILE)
    extra = dict(block=block, prev_injection=injection) if prologue else {}
    got = cmp(residual, *weights, **extra, partial=partial, swap=swap)
    want = served(residual, *weights, **extra, partial=partial)
    ref = _reference(residual, block, injection, weights, prologue)
    assert len(got) == len(want) == (3 if prologue else 2)
    for name, g, s, r in zip(("mixed", "injection", "combined"), got, want, ref):
        g, s = _np(g), _np(s)
        assert np.isfinite(g).all(), name
        if name == "combined":
            np.testing.assert_array_equal(g, s)
            continue
        vs, cr, sr = _metrics(g, s), _metrics(g, r), _metrics(s, r)
        assert vs["max_rel"] <= 1e-2 and vs["l2"] <= 1e-2, (name, vs)
        assert cr["max_rel"] <= 2.0**-7 and cr["l2"] <= 1e-3, (name, cr)
        assert cr["l2"] <= 2.0 * sr["l2"] + 1e-5, (name, cr, sr)


@pytest.mark.parametrize("tokens", [2 * TILE, 2 * TILE + 48])
def test_prologue_is_the_separate_combine(tokens):
    """The fold's premise: the combine prologue then the mix gives the
    gated_residual combine followed by the mix without prologue, bit for bit."""
    residual, block, injection, weights = _inputs(17, tokens)
    partial = bool(tokens % TILE)
    mixed, inj, combined = cmp(residual, *weights, block=block, prev_injection=injection, partial=partial)
    sep = gated_residual_fn(BRANCHES, HIDDEN, TILE, partial, INTERPRET)(residual, block, injection)
    mixed2, inj2 = cmp(sep, *weights, partial=partial)
    for a, b in ((combined, sep), (mixed, mixed2), (inj, inj2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_rejects_unsupported_rows():
    residual, block, injection, weights = _inputs(3, 2 * TILE)
    with pytest.raises(ValueError, match="cmp mix"):
        jax.eval_shape(cmp, residual[: 2 * TILE - 16], *weights)
    with pytest.raises(ValueError, match="cmp mix"):
        jax.eval_shape(functools.partial(cmp, partial=True), residual[: 2 * TILE - 8], *weights)


# ---- the GatedResidual path: the decoder layers' mix -> combine_mix -> (folded) combine


def _module(seed):
    torch.manual_seed(seed)
    module = GatedResidual(
        HyperConnectionConfig(
            hc_count=BRANCHES,
            hidden_size=HIDDEN,
            params_dtype=torch.bfloat16,
            hc_lowrank=LOWRANK,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        module.hc_norm.weight.normal_(0.0, 0.3)
        module.input_mix_weight_down.weight.normal_(0.0, 0.02)
        module.input_mix_weight_up.weight.normal_(0.0, 0.05)
        module.block_inject_weight.weight.normal_(0.0, 0.02)
    return module


def _to_jax(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy(), jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a):
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def _prefill_tiles(module, partial):
    module.prefill_full_tile = TILE
    module._prefill_tile_step = TILE
    module._prefill_partial = partial
    return module


def _install_cmp(module, calls, partial=False):
    """Interpret-mode stand-ins for the registered prefill ops under the flag
    (the torch op boundary needs torch_tpu); the combine is gated_residual."""
    _prefill_tiles(module, partial)

    def inj_mix(x, *weights):
        calls.append("mix")
        return tuple(map(_to_torch, cmp(*map(_to_jax, (x, *weights)), partial=partial)))

    def combine_mix(residual, block, gate, *weights):
        calls.append("combine_mix")
        outs = cmp(
            *map(_to_jax, (residual, *weights)),
            block=_to_jax(block),
            prev_injection=_to_jax(gate),
            partial=partial,
        )
        return tuple(map(_to_torch, outs))

    def combine(residual, block, gates):
        calls.append("combine")
        fn = gated_residual_fn(BRANCHES, HIDDEN, TILE, partial, INTERPRET)
        return _to_torch(fn(_to_jax(residual), _to_jax(block), _to_jax(gates)))

    module._prefill_inj_mix_op = inj_mix
    module._prefill_combine_mix_op = combine_mix
    module._prefill_combine_op = combine
    return module


def _run_stage(layers, hyper, fold):
    """A pipeline stage's layers as Qwen4ExpBackbone runs them: with ``fold``
    each mlp combine but the last is deferred into the next attn mix through
    ``combine_mix(prefill_fused=True)``. ``layers`` are (attn, mlp, attention,
    moe) with callable blocks. Returns (stage output, every mixed input)."""
    blocks = []
    pending = None
    for index, (attn, mlp, attention, moe) in enumerate(layers):
        if pending is None:
            block, residual = attn.mix(hyper)
        else:
            prev, output, prev_residual = pending
            block, residual = prev.combine_mix(output, prev_residual, attn, prefill_fused=True)
        blocks.append(block)
        block, residual = attn.combine_mix(attention(block), residual, mlp)
        blocks.append(block)
        if fold and index + 1 < len(layers):
            pending = (mlp, moe(block), residual)
        else:
            pending = None
            hyper = mlp.combine(moe(block), residual)
    return hyper, blocks


@pytest.mark.parametrize("tokens", [2 * TILE, 2 * TILE + 32])
def test_stage_fold_matches_unfolded_stage(tokens):
    """Three layers of a stage: folded (one mix, five combine+mix, one
    combine) against unfolded (three mixes, three combine+mixes, three
    combines), bit for bit."""
    partial = bool(tokens % TILE)
    torch.manual_seed(21)
    hyper = torch.randn(tokens, WIDTH).to(torch.bfloat16)
    outs = [torch.randn(tokens, HIDDEN).to(torch.bfloat16) for _ in range(6)]

    def stage(calls):
        layers = []
        for i in range(3):
            attention_out, moe_out = outs[2 * i], outs[2 * i + 1]
            layers.append(
                (
                    _install_cmp(_module(30 + 2 * i), calls, partial),
                    _install_cmp(_module(31 + 2 * i), calls, partial),
                    lambda block, out=attention_out: out + 0.25 * block,
                    lambda block, out=moe_out: out - 0.25 * block,
                )
            )
        return layers

    ref_calls, calls = [], []
    with torch.no_grad():
        ref_out, ref_blocks = _run_stage(stage(ref_calls), hyper, fold=False)
        out, blocks = _run_stage(stage(calls), hyper, fold=True)
    assert ref_calls == ["mix", "combine_mix", "combine"] * 3, ref_calls
    assert calls == ["mix"] + ["combine_mix"] * 5 + ["combine"], calls
    for i, (block, ref) in enumerate(zip(blocks, ref_blocks)):
        np.testing.assert_array_equal(block.view(torch.int16).numpy(), ref.view(torch.int16).numpy(), f"block {i}")
    np.testing.assert_array_equal(out.view(torch.int16).numpy(), ref_out.view(torch.int16).numpy())


def _stub_ops(module):
    """Traceable torch stand-ins for the registered prefill ops, reading the
    same weights in the same order."""
    _prefill_tiles(module, False)

    def mix_op(x, norm, down, up, inject):
        normed = x * (1.0 + norm)
        mixed = F.linear(F.silu(F.linear(normed, down)), up)[:, : module.hidden_size]
        return mixed, F.linear(normed, inject).float()

    def combine_mix_op(residual, block, gate, *weights):
        combined = residual + block.repeat(1, BRANCHES) * gate.repeat_interleave(
            module.hidden_size, dim=-1
        ).to(block.dtype)
        mixed, injection = mix_op(combined, *weights)
        return mixed, injection, combined

    def combine_op(residual, block, gate):
        return residual + block.repeat(1, BRANCHES) * gate.repeat_interleave(
            module.hidden_size, dim=-1
        ).to(block.dtype)

    module._prefill_inj_mix_op = mix_op
    module._prefill_combine_mix_op = combine_mix_op
    module._prefill_combine_op = combine_op
    return module


class _Stage(torch.nn.Module):
    """Two folded layers of a stage; the blocks own weights, as attention and
    MoE do."""

    def __init__(self, fused):
        super().__init__()
        hcs = [_module(seed) for seed in range(4)]
        if fused:
            hcs = [_stub_ops(m) for m in hcs]
        self.hcs = torch.nn.ModuleList(hcs)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(HIDDEN, HIDDEN, bias=False).to(torch.bfloat16) for _ in range(4)
        )

    def forward(self, hyper):
        hcs, blocks = self.hcs, self.blocks
        layers = [(hcs[2 * i], hcs[2 * i + 1], blocks[2 * i], blocks[2 * i + 1]) for i in range(2)]
        return _run_stage(layers, hyper, fold=True)[0]


def test_folded_prefill_traces_use_weights_in_one_order():
    """With the prefill fold, a fused prefill bucket and an unfused bucket
    first use every weight in one order (the shape-variant hand-over refuses
    a bucket whose subgraph inputs moved)."""
    big = (torch.zeros(2 * TILE, WIDTH, dtype=torch.bfloat16),)
    small = (torch.zeros(16, WIDTH, dtype=torch.bfloat16),)
    fused_model = _Stage(True)
    assert fused_model.hcs[0]._prefill_fused_mix(big[0])
    assert not fused_model.hcs[0]._prefill_fused_mix(small[0])
    fused = parameters_by_first_use(fused_model, *big)
    assert len(fused) == 4 * 4 + 4
    assert fused == parameters_by_first_use(_Stage(True), *small)


@pytest.fixture
def fake_adapter(monkeypatch):
    """Stand-in for kernels/qwen4_exp/vllm_adapter (it needs torch_tpu): every
    builder records its arguments and returns a named sentinel."""
    import sys
    import types

    built = {}
    fake = types.ModuleType("vllm_torchtpu.kernels.qwen4_exp.vllm_adapter")

    def builder(name):
        def build(*args):
            built.setdefault(name, []).append(args)
            return lambda *a: name

        return build

    for name in ("build_gated_residual_op", "build_full_mix_op", "build_norm_op",
                 "build_up_epilogue_op", "build_decode_mix_op", "build_decode_combine_mix_op",
                 "build_prefill_inj_mix_op", "build_prefill_combine_mix_op"):
        setattr(fake, name, builder(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    for name in ("TPU_QWEN4_HC_PREFILL_ODD_TILES", "TPU_QWEN4_HC_PREFILL_FUSED_MIX",
                 "TPU_QWEN4_HC_DECODE_FUSED", "TPU_QWEN4_HC_PREFILL_PARTIAL_TILE",
                 "TPU_QWEN4_HC_PREFILL_COMPARE"):
        monkeypatch.setattr(envs, name, False)
    return built


@pytest.mark.parametrize("fused_mix", [False, True])
def test_flag_builds_the_cmp_ops(monkeypatch, fake_adapter, fused_mix):
    """The flag builds both fused prefill mixes with ``cmp`` (with or without
    TPU_QWEN4_HC_PREFILL_FUSED_MIX) and takes odd tile counts; off, they are
    built only under FUSED_MIX and without ``cmp``."""
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_FUSED_MIX", fused_mix)
    off = GatedResidual(_module(1).config, prefill_full_tile=256)
    assert (off._prefill_inj_mix_op is not None) == fused_mix
    if fused_mix:
        assert fake_adapter["build_prefill_inj_mix_op"][-1][-1] is False
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_COMPARE", True)
    on = GatedResidual(_module(1).config, prefill_full_tile=256)
    assert on._prefill_inj_mix_op is not None and on._prefill_combine_mix_op is not None
    assert fake_adapter["build_prefill_inj_mix_op"][-1][-1] is True
    assert fake_adapter["build_prefill_combine_mix_op"][-1][-1] is True
    assert on._prefill_full_rows(3 * 256) and not on._prefill_full_rows(256)
    cfg = _module(2).config
    assert GatedResidual(cfg, prefill_full_tile=256, use_combine=False)._prefill_inj_mix_op is None
