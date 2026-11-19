# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS wiring in Qwen4ExpPooledPLE.bind_pool (q38 L1-GLUEFIX; CPU).

bind_pool builds the speculative conv ops with exactly the D07-b2 arguments when the min rows are unset (or the bounded
flag is unset), and passes the min rows to the bounded op's builder when both are set. The builders are recorders, so
this checks the call arguments only; the gated computation itself is tests/kernels/test_qwen4_exp_ple_decode_bounded_min_rows.py.
"""

from types import SimpleNamespace

import pytest
import torch

BND, MINR = "VLLM_TPU_PLE_DECODE_BOUNDED", "VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS"


@pytest.mark.parametrize("env,want", [
    ({}, [(3, 9)]),
    ({MINR: "32"}, [(3, 9)]),
    ({BND: "1"}, [(3, 9), (3, 9, 1)]),
    ({BND: "2", MINR: "0"}, [(3, 9), (3, 9, 2)]),
    ({BND: "1", MINR: "32"}, [(3, 9), (3, 9, 1, 32)]),
    ({BND: "2", MINR: "48"}, [(3, 9), (3, 9, 2, 48)]),
])
def test_bind_pool_passes_the_min_rows_to_the_bounded_op(monkeypatch, env, want):
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_ple

    for k in (BND, MINR, "VLLM_TPU_PLE_FUSED", "VLLM_TPU_PLE_CONV_ONEHOT", "Q38_PROBE_PLE_ONEHOT_MUTANT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    built = []
    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", lambda *a: ("conv", a))
    monkeypatch.setattr(pooled_ple, "build_pooled_speculative_short_conv_op", lambda *a: built.append(a) or ("spec", a))
    monkeypatch.setattr(pooled_ple.pdense_engagement, "register_ple", lambda *a, **k: None)
    layer = SimpleNamespace(conv_dilation=3, conv_state_len=9, key_proj=SimpleNamespace(weight=torch.zeros(1)),
                            value_proj=SimpleNamespace(weight=torch.zeros(1)))
    pooled_ple.Qwen4ExpPooledPLE.bind_pool(layer, torch.zeros(1))
    assert built == want, built
    assert (layer._speculative_conv_bounded is None) == (BND not in env)
