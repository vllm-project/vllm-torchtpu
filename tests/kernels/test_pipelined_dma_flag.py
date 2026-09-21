# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
import sys
from unittest.mock import MagicMock, patch

# Provide lightweight stubs for optional heavy runtime dependencies if not present
for mod_name in ("torch", "vllm", "vllm.utils"):
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.ragged_paged_attention.v2 import \
    ragged_kv_cache_update  # noqa: E402
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import \
    kernel  # noqa: E402


def test_env_var_default():
    """Verify that both flags default to False when unset."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_env_var_toggle():
    """Verify that setting either flag toggles both flags."""
    # Test setting USE_RPA_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"},
                    clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test setting USE_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_PIPELINED_DMA_STAGING": "1"},
                    clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test boolean string true
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "true"},
                    clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test explicit disable "0"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "0"},
                    clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_ragged_kv_cache_update_signature_and_wiring():
    """Verify ragged_kv_cache_update parameter defaults and env resolution."""
    sig = inspect.signature(ragged_kv_cache_update.kv_cache_update)
    assert "pipelined_dma" in sig.parameters
    assert sig.parameters["pipelined_dma"].default is None

    mock_kv = MagicMock()
    mock_new_kv = MagicMock()
    mock_new_kv.shape = (10, 2, 128)

    # Verify when env var is False
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(ragged_kv_cache_update,
                          "_kv_cache_update") as mock_update:
            ragged_kv_cache_update.kv_cache_update.__wrapped__(
                mock_new_kv,
                MagicMock(),
                mock_kv,
                MagicMock(),
                num_slices_per_block=4)
            mock_update.assert_called_once()
            _, kwargs = mock_update.call_args
            assert kwargs["pipelined_dma"] is False

    # Verify when env var is True
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"},
                    clear=True), patch.object(
                        ragged_kv_cache_update,
                        "_kv_cache_update") as mock_update:
        ragged_kv_cache_update.kv_cache_update.__wrapped__(
            mock_new_kv,
            MagicMock(),
            mock_kv,
            MagicMock(),
            num_slices_per_block=4)
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is True

    # Verify explicit override pipelined_dma=False when env var is True
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"},
                    clear=True), patch.object(
                        ragged_kv_cache_update,
                        "_kv_cache_update") as mock_update:
        ragged_kv_cache_update.kv_cache_update.__wrapped__(
            mock_new_kv,
            MagicMock(),
            mock_kv,
            MagicMock(),
            num_slices_per_block=4,
            pipelined_dma=False)
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is False


def test_kernel_v3_prepare_inputs_outputs_wiring():
    """Verify kernel.py prepare_inputs/prepare_outputs wiring to env flag."""
    sig_in = inspect.signature(kernel.prepare_inputs)
    assert "fuse_non_tiling_axis_swap" in sig_in.parameters
    assert sig_in.parameters["fuse_non_tiling_axis_swap"].default is None

    sig_out = inspect.signature(kernel.prepare_outputs)
    assert "fuse_non_tiling_axis_swap" in sig_out.parameters
    assert sig_out.parameters["fuse_non_tiling_axis_swap"].default is None


def test_kernel_v3_prepare_inputs_outputs_execution():
    """Verify prepare_inputs and prepare_outputs shape behavior under env toggle."""
    import jax.numpy as jnp

    max_tokens = 16
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    q = jnp.zeros((max_tokens, q_heads, head_dim))
    k = jnp.zeros((max_tokens, kv_heads, head_dim))
    v = jnp.zeros((max_tokens, kv_heads, head_dim))

    # Default / False: returns [kv_heads, max_tokens, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Flag enabled: returns [max_tokens, kv_heads, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"},
                    clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (16, 2, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Explicit override False when flag is True
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"},
                    clear=True):
        q_out, _ = kernel.prepare_inputs(q,
                                         k,
                                         v,
                                         fuse_non_tiling_axis_swap=False)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out,
                                     4,
                                     128,
                                     fuse_non_tiling_axis_swap=False)
        assert out.shape == (16, 8, 128)
