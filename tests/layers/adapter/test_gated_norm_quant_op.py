# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU_GDN_FUSED_NORM_QUANT on a real GDN layer: what install accepts and
refuses, and that the fused output projection matches RMSNormGated followed by
the fp8 linear. The torch<->JAX op bridge runs the JAX functions on the host."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from einops import rearrange

from vllm_torchtpu.kernels.experimental import gated_norm_quant as gnq
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor
from vllm_torchtpu.layers.adapter import gated_norm_quant_op, linear_common

pytestmark = pytest.mark.cpu_test

_TO_JAX = {
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    torch.float8_e4m3fn: jnp.float8_e4m3fn,
}
_TO_TORCH = {jnp.dtype(v): k for k, v in _TO_JAX.items()}


class _HostOp:
    """`pallas.jax_op` stand-in that runs the JAX function on CPU arrays."""

    def __init__(self, name, fn):
        self.fn = fn

    def register_fake(self, fake):
        pass

    def __call__(self, *args):
        arrays = [
            jnp.asarray(a.detach().float().numpy(), dtype=_TO_JAX[a.dtype])
            for a in args
        ]
        return jax.tree.map(
            lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(
                _TO_TORCH[a.dtype]
            ),
            self.fn(*arrays),
        )


@pytest.fixture
def host_ops(monkeypatch):
    """Call after constructing the layer, whose own ops stay mocked."""
    monkeypatch.setattr(gated_norm_quant_op, "_norm_ops", {})
    monkeypatch.setattr(linear_common, "_prequant_matmul_op", None)
    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op", None)
    monkeypatch.setattr(
        gnq,
        "gated_norm_quant",
        functools.partial(gnq.gated_norm_quant, interpret=True),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        yield lambda: monkeypatch.setattr(gated_norm_quant_op.pallas, "jax_op", _HostOp)


def _loaded_gdn(make_gdn_attention, host_ops, quantization="fp8_channel", tp_size=1):
    attention, _ = make_gdn_attention(tp_size, 0, quantization)
    host_ops()
    out_proj = attention.out_proj
    generator = torch.Generator().manual_seed(0)
    weight = torch.randint(-16, 17, out_proj.weight.shape, generator=generator)
    out_proj.weight.data.copy_((weight / 16).to(out_proj.weight.dtype))
    if quantization != "bf16":
        out_proj.weight_scale.data.fill_(0.02)
    out_proj.quant_method.process_weights_after_loading(out_proj)
    attention.norm.weight.data.copy_(
        1 + 0.1 * torch.randn(attention.norm.weight.shape, generator=generator)
    )
    return attention


def _gdn_outputs(attention, tokens=16, seed=1):
    generator = torch.Generator().manual_seed(seed)
    shape = (tokens, attention.num_v_heads // attention.tp_size, attention.head_v_dim)
    core = (3 * torch.randn(shape, generator=generator)).to(torch.bfloat16)
    z = torch.randn(shape, generator=generator).to(torch.bfloat16)
    return core, z


# A block-quantized checkpoint is requantized per channel at load.
@pytest.mark.parametrize("quantization", ["fp8_channel", "fp8_block"])
def test_install_marks_every_gdn_layer(make_gdn_attention, host_ops, quantization):
    attention = _loaded_gdn(make_gdn_attention, host_ops, quantization)
    assert not attention.fused_norm_quant
    assert gated_norm_quant_op.install(torch.nn.ModuleList([attention])) == 1
    assert attention.fused_norm_quant


def _block_scales(attention):
    out_proj = attention.out_proj
    out_proj.weight_scale = torch.nn.Parameter(
        torch.ones(1, out_proj.weight.shape[0] // 128, 1, out_proj.weight.shape[1]),
        requires_grad=False,
    )


def _sigmoid_gate(attention):
    attention.norm.activation = "sigmoid"


def _bias(attention):
    attention.out_proj.bias = torch.nn.Parameter(
        torch.zeros(attention.out_proj.weight.shape[1]), requires_grad=False
    )


@pytest.mark.parametrize(
    "quantization, edit, reason",
    [
        ("bf16", None, "not float8_e4m3fn"),
        ("fp8_channel", _block_scales, "not per-channel"),
        ("fp8_channel", _sigmoid_gate, "gates with sigmoid"),
        ("fp8_channel", _bias, "has a bias"),
    ],
)
def test_install_refuses_an_unsupported_layer(
    make_gdn_attention, host_ops, quantization, edit, reason
):
    attention = _loaded_gdn(make_gdn_attention, host_ops, quantization)
    if edit is not None:
        edit(attention)
    with pytest.raises(NotImplementedError, match=reason):
        gated_norm_quant_op.install(torch.nn.ModuleList([attention]))
    assert not attention.fused_norm_quant


def test_install_refuses_a_model_without_gdn(host_ops):
    with pytest.raises(NotImplementedError, match="no gated-delta-net layer"):
        gated_norm_quant_op.install(torch.nn.Linear(2, 2))


def test_fused_output_projection_matches_norm_then_linear(make_gdn_attention, host_ops):
    attention = _loaded_gdn(make_gdn_attention, host_ops)
    core, z = _gdn_outputs(attention)
    want = attention._apply_output_projection(core, z, core.shape[0])
    gated_norm_quant_op.install(torch.nn.ModuleList([attention]))
    got = attention._apply_output_projection(core, z, core.shape[0])
    assert got.dtype == want.dtype
    # The kernel rounds the normed rows to bf16 before quantizing, as
    # RMSNormGated returns them, so the fp8 rows and the result are the same.
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fused_output_projection_all_reduces_under_tp(
    make_gdn_attention, host_ops, monkeypatch
):
    attention = _loaded_gdn(make_gdn_attention, host_ops, tp_size=2)
    gated_norm_quant_op.install(torch.nn.ModuleList([attention]))
    reduced = []

    def all_reduce(x):
        reduced.append(x)
        return 2 * x

    monkeypatch.setattr(
        gated_norm_quant_op, "tensor_model_parallel_all_reduce", all_reduce
    )
    core, z = _gdn_outputs(attention)
    out = attention._apply_output_projection(core, z, core.shape[0])
    assert len(reduced) == 1
    torch.testing.assert_close(out, 2 * reduced[0])


def test_norm_then_linear_is_unchanged_without_install(make_gdn_attention, host_ops):
    attention = _loaded_gdn(make_gdn_attention, host_ops)
    core, z = _gdn_outputs(attention)
    normed = rearrange(attention.norm(core, z), "... h d -> ... (h d)")
    want, _ = attention.out_proj(normed)
    torch.testing.assert_close(
        attention._apply_output_projection(core, z, core.shape[0]), want
    )


def test_prequant_matmul_equals_the_quantizing_matmul():
    generator = torch.Generator().manual_seed(3)
    x = jnp.asarray(torch.randn(16, 256, generator=generator).numpy(), jnp.bfloat16)
    w_q = jnp.asarray(
        (0.05 * torch.randn(256, 96, generator=generator)).numpy(), jnp.float8_e4m3fn
    )
    w_s = jnp.full((96,), 0.02, jnp.float32)
    x_q, x_s = quantize_tensor(x, jnp.float8_e4m3fn)
    np.testing.assert_array_equal(
        np.asarray(linear_common._prequant_matmul_jax(x_q, x_s, w_q, w_s), np.float32),
        np.asarray(linear_common._quantized_matmul_jax(x, w_q, w_s), np.float32),
    )


def test_build_ops_registers_with_the_real_op_bridge(monkeypatch):
    """`pallas.jax_op` validates the JAX function's signature, including its
    return annotation, at registration; the host stand-in above does not."""
    monkeypatch.setattr(gated_norm_quant_op, "_norm_ops", {})
    monkeypatch.setattr(linear_common, "_prequant_matmul_op", None)
    gated_norm_quant_op._build_ops(1e-6)
    assert list(gated_norm_quant_op._norm_ops) == [1e-6]
