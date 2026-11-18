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
"""TPU component tests for MiniMax-H3 reference KV cache."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

REF_KV_MAX_REL_L2 = {"none": 2.0e-2, "int8": 6.0e-2}
NUM_LAYERS = 2
REF_ROWS = 4
TARGET_ROWS = 4


def _run_two_steps(
    state: object,
    q_target: torch.Tensor,
    k_full_layers: list[torch.Tensor],
    v_full_layers: list[torch.Tensor],
    k_target_layers: list[torch.Tensor],
    v_target_layers: list[torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_mask = torch.tensor([True] * REF_ROWS + [False] * TARGET_ROWS, dtype=torch.bool)
    state.set_global_reference_mask(ref_mask)

    state.begin_step(0)
    for layer_idx in range(NUM_LAYERS):
        state.process_post_parallel_layer(
            layer_idx,
            k_full_layers[layer_idx].to(device=device, dtype=dtype),
            v_full_layers[layer_idx].to(device=device, dtype=dtype),
            compact=False,
            parallel_strategy="none",
        )
    state.end_step()

    state.begin_step(1)
    k_out = torch.empty(0)
    v_out = torch.empty(0)
    attn_out = torch.empty(0)
    q_dev = q_target.to(device=device, dtype=dtype).transpose(1, 2)
    for layer_idx in range(NUM_LAYERS):
        state.record_projection_skip(
            layer_idx, REF_ROWS + TARGET_ROWS, projected_rows=TARGET_ROWS
        )
        k_out, v_out = state.process_post_parallel_layer(
            layer_idx,
            k_target_layers[layer_idx].to(device=device, dtype=dtype),
            v_target_layers[layer_idx].to(device=device, dtype=dtype),
            compact=True,
            parallel_strategy="none",
        )
        attn_out = F.scaled_dot_product_attention(
            q_dev, k_out.transpose(1, 2), v_out.transpose(1, 2), is_causal=False
        ).transpose(1, 2)
    state.end_step()
    return k_out, v_out, attn_out


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason=(
        "vllm_omni.diffusion.kv_offload_config and minimax_h3.reference_kv_cache "
        "are not in vllm-omni 0.32.0rc1"
    ),
)
@pytest.mark.parametrize("kv_host_quantization", ["none", "int8"])
def test_minimax_h3_reference_kv_cache(kv_host_quantization: str) -> None:
    """MiniMax-H3 Reference KV Cache

    - TPU path: Import parse_reference_kv_config and
      MiniMaxH3ReferenceKVTier2State inside the test body, build Tier-2 state
      on "tpu" (bfloat16) over kv_host_quantization in ("none", "int8"), and
      run step 0 refresh + step 1 layerwise H2D prefetch/reuse.
    - Setup check: Assert state.stats.captured_layers == 2, h2d_steps == 1,
      projection_skipped_layers == 2, and substituted_layers == 2.
    - Outcome check: Assert reuse-step concatenated (k_full, v_full) and
      attention output on "tpu" match CPU float32 reference within relative-L2
      tolerance.
    """
    from vllm_omni.diffusion.kv_offload_config import parse_reference_kv_config
    from vllm_omni.diffusion.models.minimax_h3.reference_kv_cache import (
        MiniMaxH3ReferenceKVTier2State,
    )

    cfg = parse_reference_kv_config(
        {
            "mode": "on",
            "kv_refresh_interval": 2,
            "kv_host_quantization": kv_host_quantization,
        }
    )
    tpu_state = MiniMaxH3ReferenceKVTier2State.from_config(
        config=cfg,
        num_layers=NUM_LAYERS,
        global_reference_rows=REF_ROWS,
        device=torch.device("tpu"),
    )
    cpu_state = MiniMaxH3ReferenceKVTier2State.from_config(
        config=cfg,
        num_layers=NUM_LAYERS,
        global_reference_rows=REF_ROWS,
        device=torch.device("cpu"),
    )
    assert tpu_state is not None, (
        "Expected TPU MiniMaxH3ReferenceKVTier2State from from_config, got None"
    )
    assert cpu_state is not None, (
        "Expected CPU MiniMaxH3ReferenceKVTier2State from from_config, got None"
    )
    cpu_state.pin_memory = False

    torch.manual_seed(0)
    q_target = torch.randn(1, TARGET_ROWS, 2, 8)
    k_full_layers = [
        torch.randn(1, REF_ROWS + TARGET_ROWS, 2, 8) for _ in range(NUM_LAYERS)
    ]
    v_full_layers = [
        torch.randn(1, REF_ROWS + TARGET_ROWS, 2, 8) for _ in range(NUM_LAYERS)
    ]
    k_target_layers = [torch.randn(1, TARGET_ROWS, 2, 8) for _ in range(NUM_LAYERS)]
    v_target_layers = [torch.randn(1, TARGET_ROWS, 2, 8) for _ in range(NUM_LAYERS)]

    with torch.no_grad():
        tpu_k, tpu_v, tpu_attn = _run_two_steps(
            tpu_state,
            q_target,
            k_full_layers,
            v_full_layers,
            k_target_layers,
            v_target_layers,
            "tpu",
            torch.bfloat16,
        )
        cpu_k, cpu_v, cpu_attn = _run_two_steps(
            cpu_state,
            q_target,
            k_full_layers,
            v_full_layers,
            k_target_layers,
            v_target_layers,
            "cpu",
            torch.float32,
        )

    assert tpu_state.stats.captured_layers == NUM_LAYERS, (
        f"Expected captured_layers == {NUM_LAYERS}, "
        f"got {tpu_state.stats.captured_layers}"
    )
    assert tpu_state.stats.h2d_steps == 1, (
        f"Expected h2d_steps == 1, got {tpu_state.stats.h2d_steps}"
    )
    assert tpu_state.stats.projection_skipped_layers == NUM_LAYERS, (
        f"Expected projection_skipped_layers == {NUM_LAYERS}, "
        f"got {tpu_state.stats.projection_skipped_layers}"
    )
    assert tpu_state.stats.substituted_layers == NUM_LAYERS, (
        f"Expected substituted_layers == {NUM_LAYERS}, "
        f"got {tpu_state.stats.substituted_layers}"
    )

    max_rel_l2 = REF_KV_MAX_REL_L2[kv_host_quantization]
    assert_parity(
        tpu_k.cpu(),
        cpu_k,
        name=f"MiniMaxH3 ref KV ({kv_host_quantization}) k_full",
        max_rel_l2=max_rel_l2,
    )
    assert_parity(
        tpu_v.cpu(),
        cpu_v,
        name=f"MiniMaxH3 ref KV ({kv_host_quantization}) v_full",
        max_rel_l2=max_rel_l2,
    )
    assert_parity(
        tpu_attn.cpu(),
        cpu_attn,
        name=f"MiniMaxH3 ref KV ({kv_host_quantization}) attn_out",
        max_rel_l2=max_rel_l2,
    )
