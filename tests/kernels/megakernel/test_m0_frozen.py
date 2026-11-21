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
"""M0 unit tests for frozen reference megakernels (`_frozen.qwen35` and `_frozen.glm5_mk_dcp`)."""

from __future__ import annotations

import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=8 --xla_llvm_disable_expensive_passes=true",
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megakernel._frozen.glm5_mk_dcp import topk2
from vllm_torchtpu.kernels.megakernel._frozen.qwen35 import megakernel as qwen_k
from vllm_torchtpu.kernels.megakernel._frozen.qwen35 import pool_io as qwen_pool
from vllm_torchtpu.kernels.megakernel.bench.bench_glm5_dcp import compile_glm5_dcp_aot
from vllm_torchtpu.kernels.megakernel.bench.bench_qwen35 import (
    compile_qwen35_aot,
    run_qwen35_interpret,
)

INT_MIN = np.int32(-(2**31))


def test_qwen35_frozen_shapes_and_hlo_guard() -> None:
    """Verify frozen Qwen3.5 MKConfig, Tiles, smalls_layout, and HLO guard."""
    lay, s_width = qwen_k.smalls_layout(qwen_k.FULL_EP)
    assert "in_norm" in lay and "post_norm" in lay and "qn" in lay and "kn" in lay
    assert s_width % 128 == 0

    c72 = qwen_k.MKConfig(moe_tp=False, stack_e=False, e_slots=72, paged_kv=True)
    lay72, _ = qwen_k.smalls_layout(c72)
    assert "ecopy" in lay72

    good_hlo = """
    HloModule fused_mk
    ENTRY main {
      %p0 = f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)} parameter(0)
      %call = bf16[16,4096]{1,0:T(8,128)S(1)} custom-call(%p0), custom_call_target="tpu_custom_call"
      ROOT %tup = (bf16[16,4096]{1,0:T(8,128)}, f8e4m3fn[4096,2,64,4,128]{4,3,2,1,0:T(8,128)}) tuple(%call, %p0)
    }
    """
    ok, reason = qwen_pool.check_fused_pool_hlo(good_hlo, page_size=128)
    assert ok is True and reason == "ok"


def test_qwen35_frozen_interpret() -> None:
    """Verify frozen Qwen3.5 unpooled and pooled steps on CPU interpret (T=1, B=2)."""
    results = run_qwen35_interpret(configs=[(1, 2)], moe_tp=False)
    assert len(results) == 2
    for row in results:
        assert row["out_l2_norm"] > 0.0
        if "cosine_vs_unpooled" in row:
            assert row["cosine_vs_unpooled"] > 0.99, row


def test_qwen35_frozen_aot_t1_and_t4() -> None:
    """Verify TPU7x 8-rank AOT compilation of frozen Qwen3.5 FULL_EP72 for T=1 and T=4."""
    results = compile_qwen35_aot(configs=[(1, 4), (4, 8)], moe_tp=False, include_pooled=True)
    assert len(results) == 4
    for row in results:
        assert row["vmem_scratch_mib"] < 62.0, row
        if row["step_kind"] == "make_step":
            assert row["custom_calls"] == 1
        else:
            # gather_gdn_states + qwen3_5_decode_megakernel + scatter_pool_states
            assert row["custom_calls"] == 3


def _np_topk(keys: np.ndarray, n_valid: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(keys.size)
    k = np.where(pos < n_valid, keys.reshape(-1), INT_MIN).astype(np.int64)
    valid = pos < n_valid
    order = sorted(np.nonzero(valid)[0], key=lambda p: (-k[p], p))
    win = np.sort(np.array(order[:K], np.int64))
    idx = np.full(K, -1, np.int32)
    kv = np.full(K, INT_MIN, np.int32)
    idx[: len(win)] = win
    kv[: len(win)] = k[win]
    return idx, kv


def test_glm5_dcp_frozen_topk2() -> None:
    """Verify frozen GLM-5.2 DCP two-level exact top-k (`topk2.local_topk`) on CPU interpret."""
    rng = np.random.default_rng(0)
    K = 256
    ROWS = 128
    keys = ((rng.integers(-2**31 + 1, 2**31 - 1, size=(ROWS, 128), dtype=np.int64) % 7) - 1).astype(
        np.int32
    )
    n_valid = 2000

    def kern(nv_ref, keys_in, idx_o, key_o, keys_s, pre, acc, cc):
        keys_s[...] = keys_in[...]
        topk2.local_topk(keys_s, nv_ref[0], K, idx_o, key_o, pre, acc, cc)

    f = pl.pallas_call(
        kern,
        out_shape=[jax.ShapeDtypeStruct((K // 128, 128), jnp.int32)] * 2,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        scratch_shapes=[
            pltpu.VMEM((ROWS, 128), jnp.int32),
            pltpu.VMEM((ROWS, 128), jnp.int32),
            pltpu.VMEM((256, 768), jnp.float32),
            pltpu.VMEM((2, 128, 128), jnp.float32),
        ],
        interpret=pltpu.InterpretParams(),
    )
    i_out, k_out = f(jnp.array([n_valid], jnp.int32), jnp.asarray(keys))
    i_exp, k_exp = _np_topk(keys, n_valid, K)
    assert np.array_equal(np.asarray(i_out).reshape(-1), i_exp)
    assert np.array_equal(np.asarray(k_out).reshape(-1), k_exp)


def test_glm5_dcp_frozen_aot() -> None:
    """Verify TPU7x 16-rank AOT compilation of frozen GLM-5.2 DCP FULL."""
    results = compile_glm5_dcp_aot()
    assert len(results) == 1
    assert results[0]["vmem_scratch_mib"] < 64.0
