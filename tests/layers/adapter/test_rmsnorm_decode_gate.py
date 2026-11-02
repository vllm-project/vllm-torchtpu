# SPDX-License-Identifier: Apache-2.0
"""TPU_FUSED_ADD_RMSNORM_DECODE_MAX_TOKENS: which op a decode bucket's norm
runs. The kernels are mocked (they need a TPU); the gate and the pairing
choices are the real ones."""

from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402

pytestmark = pytest.mark.cpu_test
HIDDEN = 6144


@pytest.fixture
def norm(monkeypatch, vllm_config_context):
    from vllm.model_executor.layers.layernorm import RMSNorm

    monkeypatch.setattr(envs, "TPU_FUSED_ADD_RMSNORM_MIN_TOKENS", 16384)
    monkeypatch.setattr(envs, "TPU_FUSED_ADD_RMSNORM_DECODE_MAX_TOKENS", 64)
    n = RMSNorm(HIDDEN, eps=1e-5).to(torch.bfloat16)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    monkeypatch.setattr(n, "_use_pallas", lambda x, residual: True)
    n._op_res = Mock(return_value=("y", "res"))
    return n


def _rows(n):
    return torch.zeros(n, HIDDEN, dtype=torch.bfloat16)


def test_decode_rows_skip_the_moe_quant_pairing(norm):
    """The stage-A MoE quant serves prefill-sized rows only: a decode bucket's
    paired MoE norm runs the plain op and leaves the slot empty."""
    norm._prequant_moe_slot = Mock(value=None)
    norm._op_quant_moe = Mock()
    assert norm.forward(_rows(4), _rows(4)) == ("y", "res")
    norm._op_quant_moe.assert_not_called()
    assert norm._prequant_moe_slot.value is None


@pytest.mark.parametrize("decode_ok", [True, False])
def test_decode_rows_pair_only_a_linear_off_the_skinny_kernel(norm, decode_ok):
    """qkv_a (XLA quantized matmul) takes the norm's fp8 rows at decode; a
    skinny-kernel linear (a dense gate_up) keeps its own quant there."""
    slot = Mock(value=None)
    norm._prequant_slot = slot
    norm._prequant_decode = decode_ok
    norm._op_quant = Mock(return_value=("y", "y_q", "scale", "res"))
    out = norm.forward(_rows(4), _rows(4))
    if decode_ok:
        assert out == ("y", "res") and slot.value == ("y_q", "scale")
        norm._op_res.assert_not_called()
    else:
        assert out == ("y", "res") and slot.value is None
        norm._op_quant.assert_not_called()


@pytest.mark.parametrize(
    "min_tokens,cap",
    [
        (512, 512),  # above the prefill gate's edge (256)
        (16384, 512),  # more rows than one block
        (16384, 48),  # not a power of 2
    ],
)
def test_bad_caps_are_refused(monkeypatch, vllm_config_context, min_tokens, cap):
    from vllm.model_executor.layers.layernorm import RMSNorm

    monkeypatch.setattr(envs, "TPU_FUSED_ADD_RMSNORM_MIN_TOKENS", min_tokens)
    monkeypatch.setattr(envs, "TPU_FUSED_ADD_RMSNORM_DECODE_MAX_TOKENS", cap)
    with pytest.raises(ValueError, match="DECODE_MAX_TOKENS"):
        RMSNorm(HIDDEN, eps=1e-5)
