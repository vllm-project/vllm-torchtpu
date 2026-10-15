# SPDX-License-Identifier: Apache-2.0
"""Compact GDN inputs preserve the existing cast and reshape contract."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput


@pytest.mark.parametrize("prepared", [False, True])
def test_prepared_input_preserves_output_dtype_and_default_conversion(
    monkeypatch, prepared
):
    from types import SimpleNamespace

    class InputsPrepared(Exception):
        pass

    def stop_before_kernel(**kwargs):
        raise InputsPrepared

    original_pad = jnp.pad
    pad_dtypes = []

    def record_pad(value, *args, **kwargs):
        pad_dtypes.append(value.dtype)
        return original_pad(value, *args, **kwargs)

    monkeypatch.setattr(wrapper.jnp, "pad", record_pad)
    monkeypatch.setattr(
        wrapper.pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128)
    )
    monkeypatch.setattr(wrapper.memory_ref, "ConvWeightsRef", stop_before_kernel)
    qkv = jnp.zeros((128, 384), jnp.bfloat16)
    if prepared:
        qkv = Conv1DInput(qkv.astype(jnp.float32)[:, None, :], qkv.dtype)
    with pytest.raises(InputsPrepared):
        wrapper.fused_conv1d_gdn(
            qkv,
            jnp.zeros((128, 1), jnp.bfloat16),
            jnp.zeros((128, 1), jnp.bfloat16),
            jnp.zeros((2, 3, 384), jnp.float32),
            jnp.zeros((2, 1, 128, 128), jnp.bfloat16),
            jnp.zeros((384, 1, 4), jnp.bfloat16),
            None,
            jnp.zeros((1,), jnp.float32),
            jnp.zeros((1,), jnp.float32),
            jnp.array([0, 128], jnp.int32),
            jnp.array([1], jnp.int32),
            jnp.array([0, 1, 1], jnp.int32),
            jnp.array([128], jnp.int32),
            n_kq=1,
            n_v=1,
            d_k=128,
            d_v=128,
            kernel_size=4,
        )
    assert pad_dtypes[0] == jnp.float32
