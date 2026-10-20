
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

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


    # The PP=8 stage geometry: 32 q heads, 2 KV heads, head_dim 256, fp8 KV, page 128.
    serve = configs.ServingConfigs(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=_tpu_info()):




    dec2, pre2 = _blocks()
    assert dec2 == dec
    assert pre2 != pre


    assert pre.bq_sz == 256 and 256 % pre.bq_c_sz == 0 and pre.bkv_sz == 1024


        _blocks()


