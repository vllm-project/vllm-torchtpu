# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU parity and routing edge cases for the opt-in dense-expert path."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import moe_dense_expert as kernel
from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper


@pytest.fixture(scope='module')
def tensors():
    if jax.default_backend() != 'tpu':
        pytest.skip('requires TPU')
    m, h, i, e = 16, 512, 512, 4
    rng = np.random.default_rng(27)
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    w1 = jnp.asarray(rng.integers(-8, 9, size=(e, h, 2 * i)),
                     jnp.float8_e4m3fn)
    w2 = jnp.asarray(rng.integers(-8, 9, size=(e, i, h)), jnp.float8_e4m3fn)
    s1 = jnp.asarray(rng.uniform(.006, .012, size=(e, 1, 1, 2 * i)),
                     jnp.float32)
    s2 = jnp.asarray(rng.uniform(.006, .012, size=(e, 1, 1, h)), jnp.float32)
    return x, w1, w2, s1, s2


@pytest.mark.parametrize('case',
                         ['mixed', 'single', 'empty', 'zero', 'duplicate'])
def test_same_entrypoint_and_ep_mask(tensors, monkeypatch, case):
    x, w1, w2, s1, s2 = tensors
    ids = np.tile(np.array([[64, 65, 63, 68]], np.int32), (16, 1))
    ids[8:, :2] = [66, 67]
    weights = np.tile(np.array([[.125, .25, .375, .25]], np.float32), (16, 1))
    if case == 'single':
        ids[:, :] = [67, 0, 1, 2]
    elif case == 'empty':
        ids[:, :] = [0, 1, 2, 3]
    elif case == 'zero':
        weights[:] = 0
    elif case == 'duplicate':
        ids[:, :] = [64, 64, 65, 65]
    args = (x, w1, w2, s1, s2, None, None, jnp.asarray(weights, jnp.bfloat16),
            jnp.asarray(ids))
    kwargs = dict(experts_start=jnp.array(64, jnp.int32),
                  topk=4,
                  use_ep=True,
                  onehot_moe_permute_threshold=32768)
    monkeypatch.setattr(wrapper, '_DECODE_IMPL', 'standard')
    wrapper.fused_moe_func.clear_cache()
    expected = np.asarray(wrapper.fused_moe_func(*args,
                                                 **kwargs)).astype('float32')
    monkeypatch.setattr(wrapper, '_DECODE_IMPL', 'dense_expert')
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = np.asarray(wrapper.fused_moe_func(*args,
                                                   **kwargs)).astype('float32')
        hlo = wrapper.fused_moe_func.lower(*args, **kwargs).as_text()
        assert 'dense_expert_moe-e_4' in hlo
        assert 'blockwise_onehot_unpermute' not in hlo
        assert np.isfinite(actual).all()
        if case in ('empty', 'zero'):
            np.testing.assert_array_equal(actual, 0)
        else:
            rel = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert rel < .008, rel
    finally:
        wrapper.fused_moe_func.clear_cache()


@pytest.mark.parametrize(
    'unsupported',
    ['prefill', 'bias', 'activation', 'scale', 'padding', 'bf16_weights'])
def test_unsupported_inputs_keep_standard_path(tensors, monkeypatch,
                                               unsupported):
    x, w1, w2, s1, s2 = tensors
    b1, activation, padded = None, 'silu', False
    if unsupported == 'prefill':
        x = jnp.tile(x, (17, 1))
    elif unsupported == 'bias':
        b1 = jnp.zeros((4, 1, 1024), jnp.float32)
    elif unsupported == 'activation':
        activation = 'gelu'
    elif unsupported == 'scale':
        s1 = None
    elif unsupported == 'padding':
        padded = True
    else:
        w1 = w1.astype(jnp.bfloat16)
    assert not kernel.can_use_dense_expert(x,
                                           w1,
                                           w2,
                                           s1,
                                           s2,
                                           b1,
                                           None,
                                           activation=activation,
                                           rhs_quant_dtype=None,
                                           skip_padded_tokens=padded)


def test_prefill_entrypoint_falls_back(tensors, monkeypatch):
    x, w1, w2, s1, s2 = tensors
    x = jnp.tile(x, (17, 1))  # 272 tokens exceeds the decode kernel cap.
    args = (x, w1, w2, s1, s2, None, None, jnp.ones(
        (272, 1), jnp.bfloat16), jnp.zeros((272, 1), jnp.int32))
    monkeypatch.setattr(wrapper, '_DECODE_IMPL', 'standard')
    wrapper.fused_moe_func.clear_cache()
    expected = wrapper.fused_moe_func(*args)
    monkeypatch.setattr(wrapper, '_DECODE_IMPL', 'dense_expert')
    wrapper.fused_moe_func.clear_cache()
    try:
        actual = wrapper.fused_moe_func(*args)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert 'dense_expert_moe-e_' not in wrapper.fused_moe_func.lower(
            *args).as_text()
    finally:
        wrapper.fused_moe_func.clear_cache()


def test_small_vmem_is_ineligible(tensors, monkeypatch):
    monkeypatch.setattr(
        kernel.pltpu, 'get_tpu_info',
        lambda: SimpleNamespace(vmem_capacity_bytes=16 * 1024 * 1024))
    x, w1, w2, s1, s2 = tensors
    assert not kernel.can_use_dense_expert(x,
                                           w1,
                                           w2,
                                           s1,
                                           s2,
                                           None,
                                           None,
                                           activation='silu',
                                           rhs_quant_dtype=None,
                                           skip_padded_tokens=False)
