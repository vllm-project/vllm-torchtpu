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
"""The shared expert computed inside fused_moe_func (TPU_MOE_STREAM_SHARED_EXPERT)."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    fused_moe_func_shared,
    shared_expert_xla,
)


def _shared_weights(key, hidden, fs):
    k = jax.random.split(key, 4)
    sw1 = (jax.random.normal(k[0], (hidden, 2 * fs)) * 8).astype(jnp.float8_e4m3fn)
    sw2 = (jax.random.normal(k[1], (fs, hidden)) * 8).astype(jnp.float8_e4m3fn)
    s1 = (2.0 ** jnp.round(jax.random.uniform(k[2], (2 * fs,), minval=-9, maxval=-6)))
    s2 = (2.0 ** jnp.round(jax.random.uniform(k[3], (hidden,), minval=-9, maxval=-6)))
    return sw1, s1.astype(jnp.bfloat16), sw2, s2.astype(jnp.bfloat16)


def _torch_shared_mlp(x, sw1, s1, sw2, s2):
    """MiniMaxM3MLP on MiniMaxM3Mxfp8LinearMethod's e4m3 weights, with
    SiluAndMulWithClamp.forward_native's (limit 7, alpha 1.702, beta 1) ops."""
    t = lambda a: torch.from_numpy(np.array(a.astype(jnp.float32))).to(torch.bfloat16)
    xt = t(x)
    gu = torch.matmul(xt, t(sw1)) * t(s1)
    d = gu.shape[-1] // 2
    gate = torch.clamp(gu[..., :d], max=7.0)
    up = torch.clamp(gu[..., d:], min=-7.0, max=7.0)
    h = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    return (torch.matmul(h, t(sw2)) * t(s2)).float().numpy()


def test_shared_expert_xla_matches_torch_mlp():
    hidden, fs, n = 256, 128, 8
    sw1, s1, sw2, s2 = _shared_weights(jax.random.key(0), hidden, fs)
    x = jax.random.uniform(jax.random.key(1), (n, hidden), minval=-1, maxval=1).astype(
        jnp.bfloat16
    )
    ref = _torch_shared_mlp(x, sw1, s1, sw2, s2)
    got = np.asarray(
        shared_expert_xla(
            x, sw1, s1.reshape(1, -1).astype(jnp.float32), sw2,
            s2.reshape(1, -1).astype(jnp.float32),
        ).astype(jnp.float32)
    )
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= 1e-2 * scale
    assert np.abs(got - ref).mean() <= 1e-3 * scale


def test_fused_moe_func_adds_shared_in_bf16(monkeypatch):
    """Without the streamed kernel the shared slice comes from shared_expert_xla
    and is added to the routed output in bf16; with it, from the kernel."""
    from vllm_torchtpu.layers.core import fused_moe_gmm

    num_tokens, hidden, fs = 8, 256, 128
    k = jax.random.split(jax.random.key(2), 3)
    x = jax.random.uniform(k[0], (num_tokens, hidden), minval=-1, maxval=1).astype(
        jnp.bfloat16
    )
    routed = jax.random.normal(k[1], (num_tokens, hidden)).astype(jnp.bfloat16)
    sw1, s1, sw2, s2 = _shared_weights(k[2], hidden, fs)
    seen = {}

    def fake_routed(*args, stream_shared=None, **kwargs):
        if stream_shared is not None:
            seen["shapes"] = [a.shape for a in stream_shared["args"]]
        return routed

    monkeypatch.setattr(fused_moe_gmm, "_fused_moe_routed", fake_routed)
    args = (x, None, None, None, None, None, None, None, None, None)
    # The traced functions: jit would cache the first patched trace.
    fused = fused_moe_func.__wrapped__
    fused_shared = fused_moe_func_shared.__wrapped__
    got = fused_shared(*args, sw1, s1, sw2, s2)
    assert seen["shapes"] == [
        (16, hidden), (hidden, 2 * fs), (1, 2 * fs), (fs, hidden), (1, hidden)
    ]
    sh = shared_expert_xla(
        x, sw1, s1.reshape(1, -1).astype(jnp.float32), sw2,
        s2.reshape(1, -1).astype(jnp.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)),
        np.asarray((sh + routed).astype(jnp.float32)),
    )
    # The kernel's slice (rows past num_tokens dropped) replaces the XLA one.
    kernel_sh = jnp.ones((16, hidden), jnp.bfloat16)

    def fake_routed_kernel(*args, stream_shared=None, **kwargs):
        stream_shared["out"] = kernel_sh
        return routed

    monkeypatch.setattr(fused_moe_gmm, "_fused_moe_routed", fake_routed_kernel)
    got = fused_shared(*args, sw1, s1, sw2, s2)
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)),
        np.asarray((kernel_sh[:num_tokens] + routed).astype(jnp.float32)),
    )
    # No shared: the routed output unchanged.
    monkeypatch.setattr(fused_moe_gmm, "_fused_moe_routed", fake_routed)
    assert fused(*args) is routed


def test_custom_op_signatures_are_valid():
    """torch_tpu's jax_op refuses parameters it cannot type (a tuple did)."""
    import functools
    import inspect

    from torch_tpu._internal.pallas import pallas

    for fn in (fused_moe_func, fused_moe_func_shared):
        bound = functools.partial(
            fn, topk=4, activation="swigluoai", use_ep=True, use_sparse_core=True,
            onehot_moe_permute_threshold=0, rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )
        sig = inspect.signature(inspect.unwrap(bound), eval_str=True)
        pallas._verify_signature(sig, bound_names=pallas._get_bound_names(bound))


def test_fused_moe_custom_ops_build():
    from vllm_torchtpu.layers.adapter import fused_moe

    for with_shared in (False, True):
        fused_moe._build_fused_moe_custom_op(
            topk=4, activation="swigluoai", use_ep=True, skip_padded_tokens=True,
            with_shared=with_shared,
        )
