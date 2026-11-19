# SPDX-License-Identifier: Apache-2.0
"""q38 L2 X-IDX-4-RESIZE: L11 (VLLM_TPU_QSA_INDEXER_KEY_LANES, 128-lane index key copies) on top of D24
(VLLM_TPU_QSA_INDEXER_STRIDED_KEYS, word-strided key reads), both default off.

The decode select at the served FP8-carrier shape lowers for TPU7x in all four flag combinations and every StreamIndex
kernel it emits carries exactly the arm's name: "-sk" iff D24, "-l128" iff L11, D24's suffix first. The CPU interpreter
cannot load through a ref-level bitcast, so the D24 reader's bits under -l128 are checked on device (the KB car), not
here; L11's own interpret-mode tests (test_qwen4_exp_indexer_key_lanes.py) and D24's (test_qwen4_exp_d24_strided_keys.py)
cover each lever alone on this tree.
"""

import functools
import re

import jax
import jax.numpy as jnp
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import indexer_components, pooled_qsa

SERVED = (("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4"), ("VLLM_TPU_QSA_DECODE_KV_BLOCK", "3328"),
          ("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", "1024"), ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "1"),
          ("VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", "1"), ("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1"))


def _lower_select(rows):
    S = jax.ShapeDtypeStruct
    args = (S((rows, 4, 128), jnp.bfloat16), S((512, 1664, 1, 4, 256), jnp.float8_e4m3fn), S((rows,), jnp.int32),
            S((64, 158), jnp.int32), S((rows,), jnp.int32), S((65,), jnp.int32), S((64,), jnp.int32))
    fn = jax.jit(functools.partial(pooled_qsa.select_blocks, ratio=4, block_topk=512, max_query_len=4))
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


@pytest.mark.parametrize("sk,lanes", [("0", "0"), ("1", "0"), ("1", "128"), ("0", "128")])
def test_both_levers_lower_with_the_arm_name(monkeypatch, sk, lanes):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for k, v in SERVED + (("VLLM_TPU_QSA_INDEXER_STRIDED_KEYS", sk), ("VLLM_TPU_QSA_INDEXER_KEY_LANES", lanes)):
        monkeypatch.setenv(k, v)
    jax.clear_caches()
    names = set(re.findall(r'kernel_name = "(StreamIdxTC-[^"]*)"', _lower_select(64)))
    want = "StreamIdxTC-m-bq_4-bkvp_104" + ("-sk" if sk == "1" else "") + ("-l128" if lanes == "128" else "")
    assert names == {want}, (sk, lanes, names)
    print(f"Q38_L2_X4_NAME sk={sk} lanes={lanes} kernels={sorted(names)}")


def test_qsa_indexing_carries_both():
    c = indexer_components.qsa_indexing(strided_keys=True, key_lanes=128)
    assert c.key_lanes == 128 and getattr(c.keys, "strided", False) is True
    c = indexer_components.qsa_indexing(strided_keys=False)
    assert c.key_lanes == 0 and getattr(c.keys, "strided", True) is False
