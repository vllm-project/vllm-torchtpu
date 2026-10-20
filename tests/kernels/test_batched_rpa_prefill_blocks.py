# SPDX-License-Identifier: Apache-2.0
"""TPU_RPA_PREFILL_BLOCKS overrides the batched RPA prefill tiles (and only those)."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _tpu_info():
    m = MagicMock()
    m.generation = 7
    m.chip_version = "v7x"
    m.num_lanes = 128
    m.mxu_column_size = 256
    m.vmem_capacity_bytes = 64 * 1024 * 1024
    m.fp8_ops_per_second = 2
    m.bf16_ops_per_second = 1
    m.num_sublanes = 8
    return m


def _blocks(total_q_tokens=4352):
    # The PP=8 stage geometry: 32 q heads, 2 KV heads, head_dim 256, fp8 KV, page 128.
    model = configs.ModelConfigs(num_q_heads=32, num_kv_heads=2, head_dim=256, mask_value=-1e30)
    serve = configs.ServingConfigs(
        num_seqs=32, page_size=128, total_q_tokens=total_q_tokens, num_page_indices=2048,
        dtype_q=jnp.bfloat16, dtype_kv=jnp.float8_e4m3fn, dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=_tpu_info()):
        return wrapper.calculate_block_sizes(model, serve, 64 * 1024 * 1024, decode_query_size=4)


def test_unset_keeps_heuristic(monkeypatch):
    monkeypatch.delenv("TPU_RPA_PREFILL_BLOCKS", raising=False)
    dec, pre = _blocks()
    monkeypatch.setenv("TPU_RPA_PREFILL_BLOCKS", "")
    dec2, pre2 = _blocks()
    assert (dec, pre) == (dec2, pre2)


def test_override_replaces_prefill_only(monkeypatch):
    monkeypatch.delenv("TPU_RPA_PREFILL_BLOCKS", raising=False)
    dec, pre = _blocks()
    monkeypatch.setenv("TPU_RPA_PREFILL_BLOCKS", "256,32,1024,1")
    dec2, pre2 = _blocks()
    assert dec2 == dec
    assert pre2 == configs.BlockSizes(bq_sz=256, bq_c_sz=32, bkv_sz=1024, batch_size=1)
    assert pre2 != pre


def test_override_clamps_bq_to_tokens(monkeypatch):
    monkeypatch.setenv("TPU_RPA_PREFILL_BLOCKS", "512,128,1024,1")
    _, pre = _blocks(total_q_tokens=100)
    assert pre.bq_sz == 256 and 256 % pre.bq_c_sz == 0 and pre.bkv_sz == 1024


@pytest.mark.parametrize("bad", ["256,32,1024", "a,b,c,d", "256,32,1000,1", "0,32,1024,1"])
def test_override_rejects_bad_values(monkeypatch, bad):
    monkeypatch.setenv("TPU_RPA_PREFILL_BLOCKS", bad)
    with pytest.raises(ValueError):
        _blocks()


def test_registered_in_envs():
    # The compile-cache hash covers the registry; an unregistered knob could
    # load kernels compiled with other tiles.
    assert "TPU_RPA_PREFILL_BLOCKS" in envs.environment_variables
