# SPDX-License-Identifier: Apache-2.0
"""Fused decode HC mix (interpret mode) against the GatedResidual torch path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from jax.experimental.pallas import tpu as pltpu

from tests.compilation.dynamo_input_order import parameters_by_first_use
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import (
    decode_kernel_tile,
    decode_token_tile,
    hc_decode_mix,
)
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

# Qwen3.8-Flash: 4 streams of hidden 2560, low rank 320; HC is TP-replicated,
# so the per-rank shapes are the full shapes.
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6


def _module(seed, hidden=HIDDEN, lowrank=LOWRANK):
    torch.manual_seed(seed)
    module = GatedResidual(
        HyperConnectionConfig(
            hc_count=BRANCHES,
            hidden_size=hidden,
            params_dtype=torch.bfloat16,
            hc_lowrank=lowrank,
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


def _mixer(seed, hidden=HIDDEN, lowrank=LOWRANK):
    """The final mixer: a hyper-connection without a combine."""
    torch.manual_seed(seed)
    module = GatedResidual(
        HyperConnectionConfig(
            hc_count=BRANCHES,
            hidden_size=hidden,
            params_dtype=torch.bfloat16,
            hc_lowrank=lowrank,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        ),
        use_combine=False,
    )
    with torch.no_grad():
        module.hc_norm.weight.normal_(0.0, 0.3)
        module.input_mix_weight_down.weight.normal_(0.0, 0.02)
        module.input_mix_weight_up.weight.normal_(0.0, 0.05)
    return module


def _jax(t):
    return jnp.asarray(t.detach().float().numpy(), jnp.bfloat16)


def _weights(module):
    return (
        _jax(module.hc_norm.weight),
        _jax(module.input_mix_weight_down.weight),
        _jax(module.input_mix_weight_up.weight),
        _jax(module.block_inject_weight.weight),
    )


def _check_bf16(actual, expected, name):
    actual = np.asarray(jnp.asarray(actual, jnp.float32))
    expected = expected.float().numpy()
    assert np.isfinite(actual).all(), name
    diff = np.abs(actual - expected)
    # Only the matmul accumulation order (and FMA contraction) differ from
    # the torch path, so a BF16 rounding boundary can move by one ulp on a
    # small fraction of elements and propagate through the gate. The mixed
    # output is a sum of branch terms, so a one-ulp term change is bounded
    # by the tensor's scale rather than by the (possibly cancelled) result.
    scale = np.abs(expected).max()
    assert diff.max() <= 2.0**-7 * scale, (name, diff.max(), scale)
    assert np.linalg.norm(diff) <= 1e-3 * np.linalg.norm(expected), name
    assert (diff > 0).mean() <= 0.02, (name, (diff > 0).mean())


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128])
def test_tile_choice(tokens):
    assert decode_token_tile(tokens) == tokens


def test_tile_choice_large():
    assert decode_token_tile(256) == 128
    assert decode_token_tile(512) == 128
    assert decode_token_tile(192) == 64
    assert decode_token_tile(136) is None
    assert decode_token_tile(1024) is None
    assert decode_token_tile(0) is None


@pytest.mark.parametrize(
    "max_tile, expected",
    [
        (0, {96: 96, 128: 128, 192: 64, 256: 128, 320: 64, 384: 128, 448: 64}),
        (
            256,
            {96: 96, 112: 112, 128: 128, 144: 144, 192: 192, 208: 208, 256: 256,
             320: 160, 384: 192, 448: 224, 512: 256},
        ),
        (224, {192: 192, 256: 128, 320: 160, 384: 192, 448: 224}),
        (512, {192: 192, 256: 256, 320: 320, 448: 448}),
    ],
)
def test_kernel_tile_rows(max_tile, expected):
    """TPU_QWEN4_HC_DECODE_TILE_ROWS: the fewest equal tiles of at most
    max_tile rows (multiples of 16), never fewer counts than the torch-side
    gate decode_token_tile accepts."""
    for tokens, tile in expected.items():
        assert decode_kernel_tile(tokens, max_tile) == tile, tokens
    for tokens in range(1, 1025):
        tile = decode_kernel_tile(tokens, max_tile)
        assert (tile is None) == (decode_token_tile(tokens) is None), tokens
        if tile is not None:
            assert tokens % tile == 0 and tile >= decode_token_tile(tokens)


@pytest.mark.parametrize("tokens", [144, 192])
def test_decode_mix_tile_rows_matches_128_tiling(tokens):
    """One 144- or 192-row tile (a 16-row tail after the 32-row norm loop at
    144) against the 16- and 64-row tilings: the rows are independent."""
    module = _module(tokens)
    torch.manual_seed(300 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    gate = 2.0 * torch.sigmoid(torch.randn(tokens, BRANCHES))
    weight, down, up, inject = _weights(module)
    outs = [
        hc_decode_mix(
            _jax(hyper),
            weight,
            down,
            up,
            inject,
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
            block=_jax(block),
            prev_injection=jnp.asarray(gate.numpy()),
            max_tile=max_tile,
            interpret=pltpu.InterpretParams(),
        )
        for max_tile in (0, 256)
    ]
    for ref, out in zip(*outs):
        np.testing.assert_array_equal(
            np.asarray(jnp.asarray(out, jnp.float32)),
            np.asarray(jnp.asarray(ref, jnp.float32)),
        )


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128, 256])
def test_decode_mix_matches_torch_path(tokens):
    module = _module(tokens)
    torch.manual_seed(100 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected_mixed, residual = module.mix(hyper)
        block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
        expected_out = module.combine(block, residual)
        # mix returns the combine's injection gate as the second residual.
        injection = residual[1]

    weight, down, up, inject = _weights(module)
    mixed, inj = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    _check_bf16(mixed, expected_mixed, "mixed")
    np.testing.assert_allclose(np.asarray(inj), injection.numpy(), rtol=2e-2, atol=1e-6)
    # The combine consuming the kernel's injection gate.
    with torch.no_grad():
        out = module.combine(block, (hyper, torch.from_numpy(np.asarray(inj))))
    _check_bf16(_jax(out), expected_out, "combine")


@pytest.mark.parametrize("tokens", [4, 32, 128])
def test_decode_combine_mix_matches_torch_path(tokens):
    attn, mlp = _module(tokens), _module(1000 + tokens)
    torch.manual_seed(200 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        _, residual = attn.mix(hyper)
        injection = residual[1]
        expected_h = attn.combine(block, residual)
        expected_mixed, _ = mlp.mix(expected_h)

    weight, down, up, inject = _weights(mlp)
    mixed, inj, h = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        block=_jax(block),
        prev_injection=jnp.asarray(injection.numpy()),
        interpret=pltpu.InterpretParams(),
    )
    # The combine is elementwise FP32 with one BF16 rounding; only FMA
    # contraction may move an element by one ulp.
    _check_bf16(h, expected_h, "h")
    _check_bf16(mixed, expected_mixed, "mixed")
    assert inj.shape == (tokens, BRANCHES)


def test_decode_mix_without_injection():
    module = _module(7)
    hyper = torch.randn(8, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected_mixed, _ = module.mix(hyper)
    weight, down, up, _ = _weights(module)
    mixed, inj = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    assert inj is None
    _check_bf16(mixed, expected_mixed, "mixed")


def test_decode_mix_rejects_unsupported_shapes():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    args = (
        shape((136, width), jnp.bfloat16),
        shape((width,), jnp.bfloat16),
        shape((LOWRANK, width), jnp.bfloat16),
        shape((width, LOWRANK), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="decode-sized"):
        jax.eval_shape(
            lambda *a: hc_decode_mix(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS),
            *args,
        )


def _interpret_ops(module):
    """Stand-ins for the registered torch ops that run the kernel in
    interpret mode on CPU tensors."""

    def to_torch(a):
        if a.dtype == jnp.bfloat16:
            return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(
                torch.bfloat16
            )
        return torch.from_numpy(np.array(a))

    def to_jax(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            return _jax(t)
        return jnp.asarray(t.numpy())

    def mix_op(x, *weights):
        mixed, inj = hc_decode_mix(
            *map(to_jax, (x, *weights)),
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
            interpret=pltpu.InterpretParams(),
        )
        return to_torch(mixed) if inj is None else (to_torch(mixed), to_torch(inj))

    def combine_mix_op(residual, block, gate, *weights):
        mixed, inj, combined = hc_decode_mix(
            *map(to_jax, (residual, *weights)),
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
            block=to_jax(block),
            prev_injection=to_jax(gate),
            interpret=pltpu.InterpretParams(),
        )
        if inj is None:
            return to_torch(mixed), to_torch(combined)
        return to_torch(mixed), to_torch(inj), to_torch(combined)

    module._decode_mix_op = mix_op
    module._decode_combine_mix_op = combine_mix_op
    return module


def test_layer_plumbing_matches_unfused_layer():
    """attn mix -> combine_mix -> mlp combine, as Qwen4ExpDecoderLayer runs."""
    tokens = 8
    attn, mlp = _module(11), _module(12)
    fused_attn, fused_mlp = _interpret_ops(_module(11)), _interpret_ops(_module(12))
    torch.manual_seed(13)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    attn_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    mlp_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        outs = []
        for a, m in ((attn, mlp), (fused_attn, fused_mlp)):
            block, residual = a.mix(hyper)
            block2, residual = a.combine_mix(attn_out, residual, m)
            outs.append((block, block2, m.combine(mlp_out, residual)))
    (ref_block, ref_block2, ref_out), (block, block2, out) = outs
    assert fused_attn._decode_fused(hyper)
    _check_bf16(_jax(block), ref_block, "attn mixed")
    _check_bf16(_jax(block2), ref_block2, "mlp mixed")
    _check_bf16(_jax(out), ref_out, "layer output")
    # Unfused modules never take the decode path.
    assert not attn._decode_fused(hyper)


def _stub_ops(module):
    """Traceable torch stand-ins for the registered ops, reading the same
    weights in the same order."""

    def mix_op(x, norm, down, up, inject=None):
        normed = x * (1.0 + norm)
        mixed = F.linear(F.silu(F.linear(normed, down)), up)[:, : module.hidden_size]
        if inject is None:
            return mixed
        return mixed, F.linear(normed, inject).float()

    def combine_mix_op(residual, block, gate, *weights):
        combined = residual + block.repeat(1, BRANCHES) * gate.repeat_interleave(
            module.hidden_size, dim=-1
        ).to(block.dtype)
        if len(weights) == 3:
            return mix_op(combined, *weights), combined
        mixed, injection = mix_op(combined, *weights)
        return mixed, injection, combined

    module._decode_mix_op = mix_op
    module._decode_combine_mix_op = combine_mix_op
    return module


class _Layer(torch.nn.Module):
    """attn mix -> attention -> combine_mix -> mlp -> combine, as
    Qwen4ExpDecoderLayer runs; the blocks own weights, as attention and MoE do."""

    def __init__(self, attn, mlp):
        super().__init__()
        self.attn, self.mlp = attn, mlp
        hidden = attn.hidden_size
        self.attention = torch.nn.Linear(hidden, hidden, bias=False).to(torch.bfloat16)
        self.moe = torch.nn.Linear(hidden, hidden, bias=False).to(torch.bfloat16)

    def forward(self, hyper):
        block, residual = self.attn.mix(hyper)
        block, residual = self.attn.combine_mix(
            self.attention(block), residual, self.mlp
        )
        return self.mlp.combine(self.moe(block), residual)


def test_fused_and_unfused_traces_use_weights_in_one_order():
    """A fused decode bucket and an unfused bucket of the same model must first
    use the HC weights in one order (h20 j-dfd6839e, h21 j-b1210ac2: the
    shape-variant hand-over refused the bucket because the unfused path first
    used the attention HC's block_inject_weight in combine, after the attention
    weights)."""
    hidden, lowrank, tokens = 128, 32, 8
    args = (torch.zeros(tokens, BRANCHES * hidden, dtype=torch.bfloat16),)

    def layer(fused):
        modules = [_module(seed, hidden=hidden, lowrank=lowrank) for seed in (1, 2)]
        if fused:
            modules = [_stub_ops(m) for m in modules]
        return _Layer(*modules)

    fused = parameters_by_first_use(layer(True), *args)
    assert layer(True).attn._decode_fused(args[0])
    assert len(fused) == 10
    assert fused == parameters_by_first_use(layer(False), *args)


def _run_folded(layers, mixer, hyper, fold):
    """Layers and the final mixer as Qwen4ExpBackbone runs them: with ``fold``
    each mlp combine is deferred into the next attn mix (and the last into the
    final mixer) through ``combine_mix``. ``layers`` are (attn, mlp, attention,
    moe) with callable blocks. Returns (mixed, final hyper streams, attn and
    mlp inputs of every layer)."""
    blocks = []
    pending = None
    for attn, mlp, attention, moe in layers:
        if pending is None:
            block, residual = attn.mix(hyper)
        else:
            prev, output, prev_residual = pending
            block, residual = prev.combine_mix(
                output, prev_residual, attn, prefill_fused=False
            )
        blocks.append(block)
        block, residual = attn.combine_mix(attention(block), residual, mlp)
        blocks.append(block)
        if fold:
            pending = (mlp, moe(block), residual)
        else:
            hyper = mlp.combine(moe(block), residual)
    if pending is not None:
        prev, output, prev_residual = pending
        mixed, (hyper, _) = prev.combine_mix(
            output, prev_residual, mixer, prefill_fused=False
        )
    else:
        mixed = mixer.mix(hyper)[0]
    return mixed, hyper, blocks


def test_fold_matches_unfolded_layers():
    """TPU_QWEN4_HC_DECODE_FOLD: the mlp combine as the next mix's prologue
    (and the final mixer's) against the separate combine and fused mix."""
    tokens = 8
    torch.manual_seed(21)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    outs = [torch.randn(tokens, HIDDEN).to(torch.bfloat16) for _ in range(4)]

    def model():
        layers = []
        for i in range(2):
            attention_out, moe_out = outs[2 * i], outs[2 * i + 1]
            layers.append(
                (
                    _interpret_ops(_module(30 + 2 * i)),
                    _interpret_ops(_module(31 + 2 * i)),
                    lambda block, out=attention_out: out + 0.25 * block,
                    lambda block, out=moe_out: out - 0.25 * block,
                )
            )
        return layers, _interpret_ops(_mixer(40))

    with torch.no_grad():
        ref_mixed, ref_hyper, ref_blocks = _run_folded(*model(), hyper, fold=False)
        mixed, folded_hyper, blocks = _run_folded(*model(), hyper, fold=True)
    for i, (block, ref) in enumerate(zip(blocks, ref_blocks)):
        _check_bf16(_jax(block), ref, f"block {i}")
    _check_bf16(_jax(folded_hyper), ref_hyper, "final streams")
    _check_bf16(_jax(mixed), ref_mixed, "mixed")


class _Folded(torch.nn.Module):
    """Two layers and the final mixer with TPU_QWEN4_HC_DECODE_FOLD."""

    def __init__(self, fused, hidden, lowrank):
        super().__init__()
        hcs = [_module(seed, hidden=hidden, lowrank=lowrank) for seed in range(4)]
        mixer = _mixer(4, hidden=hidden, lowrank=lowrank)
        if fused:
            hcs, mixer = [_stub_ops(m) for m in hcs], _stub_ops(mixer)
        self.hcs, self.mixer = torch.nn.ModuleList(hcs), mixer
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden, bias=False).to(torch.bfloat16)
            for _ in range(4)
        )

    def forward(self, hyper):
        hcs, blocks = self.hcs, self.blocks
        layers = [
            (hcs[2 * i], hcs[2 * i + 1], blocks[2 * i], blocks[2 * i + 1])
            for i in range(2)
        ]
        mixed, hyper, _ = _run_folded(layers, self.mixer, hyper, fold=True)
        return mixed, hyper


def test_folded_traces_use_weights_in_one_order():
    """With the fold, a fused decode bucket and an unfused bucket still first
    use every weight in one order (see
    test_fused_and_unfused_traces_use_weights_in_one_order)."""
    hidden, lowrank, tokens = 128, 32, 8
    args = (torch.zeros(tokens, BRANCHES * hidden, dtype=torch.bfloat16),)
    fused_model = _Folded(True, hidden, lowrank)
    assert fused_model.mixer._decode_fused(args[0])
    fused = parameters_by_first_use(fused_model, *args)
    # 4 HCs x 4 weights, the mixer's 3, 4 blocks.
    assert len(fused) == 23
    assert fused == parameters_by_first_use(_Folded(False, hidden, lowrank), *args)


def test_disable_decode_fused_keeps_the_unfused_mix():
    """The MTP draft (traced once for every token bucket) disables the fused
    decode mix, whose choice depends on the token count while tracing."""
    module = _stub_ops(_module(3, hidden=128, lowrank=32))
    hyper = torch.zeros(8, BRANCHES * 128, dtype=torch.bfloat16)
    assert module._decode_fused(hyper)
    module.disable_decode_fused()
    assert not module._decode_fused(hyper)
    with torch.no_grad():
        mixed, (residual, injection) = module.mix(hyper)
    assert mixed.shape == (8, 128) and injection.shape == (8, BRANCHES)


@pytest.mark.parametrize("bucket", [128, 512])
def test_fused_gate_keeps_the_bucket_dynamic(bucket):
    """A trace at a bucket keeps its token count symbolic next to a
    ``tokens >= bucket`` branch of the same step (j-037e02fd: the trace at 512
    took the fused mix, its token count was refined to the constant 512, its
    graph lost the token-count SymInt inputs, and the shape-variant hand-over
    refused the buckets the earlier traces had built)."""
    hidden, lowrank = 128, 32
    module = _stub_ops(_module(4, hidden=hidden, lowrank=lowrank))
    symints = []

    def backend(gm, example_inputs):
        symints.extend(
            node
            for node in gm.graph.nodes
            if node.op == "placeholder"
            and isinstance(node.meta.get("example_value"), torch.SymInt)
        )
        return gm

    def step(hyper):
        mixed, _ = module.mix(hyper)
        if hyper.shape[0] >= bucket:
            mixed = mixed + 1
        return mixed

    hyper = torch.zeros(bucket, BRANCHES * hidden, dtype=torch.bfloat16)
    assert module._decode_fused(hyper) == (bucket < 512)
    torch._dynamo.reset()
    torch._dynamo.mark_dynamic(hyper, 0)
    with torch.no_grad():
        torch.compile(step, backend=backend, fullgraph=True)(hyper)
    torch._dynamo.reset()
    assert symints
