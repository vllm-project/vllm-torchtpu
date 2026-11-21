# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-HC L3: TPU_QWEN4_HC_PREFILL_FOLD through the real backbone forward.

The backbone and decoder-layer forward methods are real; attention, GDN and MoE
are deterministic bf16 stand-ins. The HC modules run the fused prefill kernels
in interpret mode (the torch op boundary needs torch_tpu), and the separate mlp
combine runs the gated_residual kernel. With the fold, each layer's mlp combine
is the reread prologue of the next layer's attn mix at whole-tile steps; the
stage output must equal the unfolded stage's bit for bit, and the call counts
show the engagement (one gated_residual left: the last layer's, ahead of the
final mixer, which has no injection gate).

Rule 23: excess precision is pinned off before jax is imported; the one-ulp
positive control lives in tests/kernels/test_qwen4_exp_hc_prefill_fold_asm.py.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import functools  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from torch import nn  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (  # noqa: E402
    HyperConnectionConfig,
)

HC, HIDDEN, LOWRANK, EPS, TILE, VOCAB, LAYERS = 4, 256, 64, 1e-6, 128, 64, 4
INTERPRET = pltpu.InterpretParams()
fused = functools.partial(
    hc_prefill_mix,
    branches=HC,
    hidden=HIDDEN,
    eps=EPS,
    token_tile=TILE,
    interpret=INTERPRET,
    partial=True,
    norm_impl="bf16ep64",
)
combine_kernel = gated_residual_fn(HC, HIDDEN, TILE, partial=True, interpret=INTERPRET)


def _to_jax(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy(), jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a):
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a))


class _Sublayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden, *args):
        return torch.tanh(self.proj(hidden))

    def forward_with_indices(self, hidden, *args, **kwargs):
        return self.forward(hidden), None


def _hc(calls, *, use_combine=True, fold=False):
    module = qwen4_model.GatedResidual(
        HyperConnectionConfig(
            hc_count=HC,
            hidden_size=HIDDEN,
            hc_lowrank=LOWRANK,
            params_dtype=torch.bfloat16,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        ),
        use_combine=use_combine,
    )
    if not use_combine:
        return module
    module.prefill_full_tile = TILE
    module._prefill_tile_step = TILE
    module._prefill_partial = True

    def inj_mix(x, *w):
        calls.append("mix")
        return tuple(map(_to_torch, fused(*map(_to_jax, (x, *w)))))

    def prologue(tag, reread):
        def run(residual, block, gate, *w):
            calls.append(tag)
            outs = fused(
                *map(_to_jax, (residual, *w)),
                block=_to_jax(block),
                prev_injection=_to_jax(gate),
                reread=reread,
            )
            return tuple(map(_to_torch, outs))

        return run

    def combine(residual, block, gates):
        calls.append("combine")
        return _to_torch(combine_kernel(*map(_to_jax, (residual, block, gates))))

    module._prefill_inj_mix_op = inj_mix
    module._prefill_combine_mix_op = prologue("combine_mix", False)
    module._prefill_fold_mix_op = prologue("fold", True) if fold else None
    module._prefill_combine_op = combine
    return module


def _backbone(calls, fold):
    torch.manual_seed(0)
    with torch.device("cpu"):
        result = qwen4_model.Qwen4ExpBackbone.__new__(qwen4_model.Qwen4ExpBackbone)
        nn.Module.__init__(result)
        result.hc_count = HC
        result.host_ple_index = None
        result.emit_mtp_hidden = False
        result.sp_pad_rows = 0
        result.stage = range(LAYERS)
        result.first_stage = result.last_stage = True
        result.embed_tokens = nn.Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16)
        result.layers = nn.ModuleList()
        for index in range(LAYERS):
            kind = "full_attention" if index % 2 else "linear_attention"
            layer = qwen4_model.Qwen4ExpDecoderLayer.__new__(qwen4_model.Qwen4ExpDecoderLayer)
            nn.Module.__init__(layer)
            layer.ple = None
            layer.layer_type = kind
            setattr(layer, "self_attn" if index % 2 else "linear_attn", _Sublayer())
            layer.mlp = _Sublayer()
            layer.attn_hyper_connection = _hc(calls, fold=fold)
            layer.mlp_hyper_connection = _hc(calls, fold=fold)
            result.layers.append(layer)
        result.hyper_connection_mixer = _hc(calls, use_combine=False)
        generator = torch.Generator(device="cpu").manual_seed(1)
        scales = {"hc_norm": 0.3, "input_mix_weight_down": 0.05, "input_mix_weight_up": 0.1,
                  "block_inject_weight": 0.05}
        with torch.no_grad():
            for name, param in result.named_parameters():
                scale = next((v for k, v in scales.items() if k in name), 0.05)
                param.copy_(torch.randn(param.shape, generator=generator) * scale)
    return result


def _metadata(tokens):
    return dict(
        query_start_loc=torch.tensor([0, tokens]),
        ngram_context=torch.zeros((1, 2), dtype=torch.int64),
        state_slots=torch.tensor([1]),
        context_lens=torch.tensor([0]),
        valid_tokens=torch.ones(tokens, dtype=torch.bool),
        main_block_table=torch.tensor([[1]]),
        compressed_block_table=torch.tensor([[2]]),
    )


def _run(monkeypatch, fold, tokens):
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_FOLD", False)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_FOLD", fold)
    calls = []
    backbone = _backbone(calls, fold)
    input_ids = torch.arange(tokens) % VOCAB
    with torch.no_grad():
        out = backbone.forward(input_ids, torch.arange(tokens), **_metadata(tokens))
    return out, calls


def test_excess_precision_pinned_in_this_file():
    assert _STRICT in os.environ["XLA_FLAGS"]


@pytest.mark.parametrize("tokens", [4 * TILE, 3 * TILE + 64])
def test_prefill_fold_stage_is_bitwise_and_engaged(monkeypatch, tokens):
    base, base_calls = _run(monkeypatch, False, tokens)
    folded, fold_calls = _run(monkeypatch, True, tokens)
    assert base.dtype == torch.bfloat16 and base.shape == (tokens, HIDDEN)
    np.testing.assert_array_equal(
        folded.view(torch.int16).numpy(), base.view(torch.int16).numpy()
    )
    # Unfolded: every layer runs its attn mix, its attn combine + mlp mix and
    # a separate mlp combine.
    assert base_calls.count("mix") == LAYERS
    assert base_calls.count("combine_mix") == LAYERS
    assert base_calls.count("combine") == LAYERS
    assert "fold" not in base_calls
    # Folded: layers 1..3 take their attn mix as the reread prologue of the
    # previous mlp combine; the last mlp combine stays separate (final mixer).
    assert fold_calls.count("mix") == 1
    assert fold_calls.count("fold") == LAYERS - 1
    assert fold_calls.count("combine_mix") == LAYERS
    assert fold_calls.count("combine") == 1


def test_prefill_fold_below_fused_rows_takes_the_separate_ops(monkeypatch):
    """Below two tiles the fused ops do not apply: both arms run the same XLA
    combine and mix, so the outputs are identical and nothing is folded."""
    tokens = 64
    base, base_calls = _run(monkeypatch, False, tokens)
    folded, fold_calls = _run(monkeypatch, True, tokens)
    np.testing.assert_array_equal(
        folded.view(torch.int16).numpy(), base.view(torch.int16).numpy()
    )
    assert base_calls == fold_calls == []


def test_fold_flag_unset_keeps_the_fold_op_unbuilt(monkeypatch):
    """GatedResidual builds the reread op only under the flag (stand-in adapter)."""
    import sys
    import types

    built = []
    fake = types.ModuleType("vllm_torchtpu.kernels.qwen4_exp.vllm_adapter")

    def builder(name):
        def build(*args, **kwargs):
            built.append((name, tuple(sorted(kwargs.items()))))
            return lambda *a: name

        return build

    for name in ("build_gated_residual_op", "build_full_mix_op", "build_norm_op",
                 "build_up_epilogue_op", "build_decode_mix_op", "build_decode_combine_mix_op",
                 "build_prefill_inj_mix_op", "build_prefill_combine_mix_op"):
        setattr(fake, name, builder(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_FUSED_MIX", True)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_FUSED", False)
    config = HyperConnectionConfig(
        hc_count=HC, hidden_size=HIDDEN, hc_lowrank=LOWRANK, params_dtype=torch.bfloat16,
        rms_norm_eps=EPS, hc_per_branch_norm=True,
    )
    for fold, asm in ((False, False), (True, False), (False, True), (True, True)):
        monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_FOLD", fold)
        monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DOWN_ASM", asm)
        built.clear()
        module = qwen4_model.GatedResidual(config, prefill_full_tile=TILE)
        kw = (("down_asm", True),) if asm else ()
        prefill = [b for b in built if b[0].startswith("build_prefill")]
        want = [("build_prefill_inj_mix_op", kw), ("build_prefill_combine_mix_op", kw)]
        if fold:
            want.append(
                ("build_prefill_combine_mix_op", tuple(sorted(kw + (("reread", True),))))
            )
        assert prefill == want, (fold, asm, prefill)
        assert (module._prefill_fold_mix_op is not None) == fold
