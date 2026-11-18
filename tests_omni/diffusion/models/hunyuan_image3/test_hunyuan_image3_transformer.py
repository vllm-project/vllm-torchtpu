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
"""Multi-rank TPU tests for diffusion expert parallelism on HunYuanImage-3.0."""

from __future__ import annotations

import os
from typing import Any

import portpicker
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch_tpu._internal.utils import hardware
from transformers import PretrainedConfig
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import set_forward_context as set_vllm_forward_context
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunYuanSparseMoeBlock,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

EP_WORLD_SIZE = 2
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 128
MOE_INTERMEDIATE_SIZE = 256
BATCH_SIZE = 2
SEQ_LEN = 8
MOE_SEED = 73
BF16_ERR_RATIO = 2.0
MOE_REL_L2_FLOOR = 2.0e-2
MOE_ABS_FLOOR = 3.0e-2

DIFFUSION_EP_XFAIL_REASON = (
    "create_diffusion_vllm_config leaves quant_config=None instead of "
    "VllmUnquantizedConfig, so HunYuanSparseMoeBlock falls back to upstream "
    "UnquantizedFusedMoEMethod -> FusedTopKRouter -> "
    "torch.ops._moe_C.topk_softmax (missing on TPU)."
)


def _sample_moe_weights() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    gen = torch.Generator(device="cpu").manual_seed(MOE_SEED)
    gate_w = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), generator=gen, dtype=torch.float32)
    w1 = (
        torch.randn(
            (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            generator=gen,
            dtype=torch.float32,
        )
        * 0.1
    )
    w3 = (
        torch.randn(
            (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            generator=gen,
            dtype=torch.float32,
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            generator=gen,
            dtype=torch.float32,
        )
        * 0.1
    )
    return gate_w, w1, w3, w2


def _hunyuan_sparse_moe_reference(
    hidden_states: torch.Tensor,
    gate_w: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    tokens = hidden_states.to(dtype=dtype).view(-1, HIDDEN_SIZE)
    gate = gate_w.to(dtype=dtype)
    w1_cast = w1.to(dtype=dtype)
    w3_cast = w3.to(dtype=dtype)
    w2_cast = w2.to(dtype=dtype)

    logits = (tokens @ gate.T).float()
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k=TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(tokens, dtype=torch.float32)
    for token_idx in range(tokens.shape[0]):
        h_tok = tokens[token_idx : token_idx + 1]
        for slot in range(TOP_K):
            expert_id = int(topk_ids[token_idx, slot].item())
            weight = float(topk_weights[token_idx, slot].item())
            gated = F.silu(h_tok @ w1_cast[expert_id].T) * (
                h_tok @ w3_cast[expert_id].T
            )
            proj = gated @ w2_cast[expert_id].T
            out[token_idx : token_idx + 1] += weight * proj.float()
    return out.view(orig_shape)


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    diff = (actual.double() - reference.double()).flatten()
    ref_flat = reference.double().flatten()
    return float(diff.norm() / ref_flat.norm()), float(diff.abs().max())


def _load_hunyuan_moe_weights(
    moe: HunYuanSparseMoeBlock,
    gate_w: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    device: torch.device,
) -> None:
    moe.gate.weight_loader(
        moe.gate.weight, gate_w.to(device=device, dtype=torch.bfloat16)
    )
    routed = moe.experts.routed_experts
    weights = []
    for expert_id in range(NUM_EXPERTS):
        weights.append(
            (
                f"{expert_id}.gate_proj.weight",
                w1[expert_id].to(device=device, dtype=torch.bfloat16),
            )
        )
        weights.append(
            (
                f"{expert_id}.up_proj.weight",
                w3[expert_id].to(device=device, dtype=torch.bfloat16),
            )
        )
        weights.append(
            (
                f"{expert_id}.down_proj.weight",
                w2[expert_id].to(device=device, dtype=torch.bfloat16),
            )
        )
    routed.load_weights(weights)
    assert routed.quant_method is not None, (
        "HunYuanSparseMoeBlock routed_experts.quant_method is None after load_weights"
    )
    routed.quant_method.process_weights_after_loading(routed)


def _ep_worker(
    rank: int,
    master_port: int,
    x_cpu: torch.Tensor,
    gate_w: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    result_queue: Any,
) -> None:
    setup_info: tuple[int, bool, int] | None = None
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(EP_WORLD_SIZE)

        device = torch.device("tpu")
        OmniTpuPlatform.set_device(device)

        od_config = OmniDiffusionConfig(
            model=None,
            dtype=torch.bfloat16,
            tf_model_config=TransformerConfig.from_dict({"num_experts": NUM_EXPERTS}),
            parallel_config=DiffusionParallelConfig(
                tensor_parallel_size=EP_WORLD_SIZE,
                enable_expert_parallel=True,
            ),
            enforce_eager=True,
        )
        vllm_config = create_diffusion_vllm_config(device, od_config)
        os.environ["WORLD_SIZE"] = str(EP_WORLD_SIZE)
        OmniTpuPlatform.init_diffusion_worker_vllm_config(vllm_config)

        with (
            set_forward_context(
                vllm_config=vllm_config, omni_diffusion_config=od_config
            ),
            set_vllm_forward_context(None, vllm_config),
            set_current_vllm_config(vllm_config),
        ):
            init_distributed_environment(
                world_size=EP_WORLD_SIZE,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{master_port}",
                local_rank=rank,
            )
            initialize_model_parallel(
                tensor_parallel_size=EP_WORLD_SIZE,
                enable_expert_parallel=True,
            )

            hf_cfg = PretrainedConfig(
                hidden_size=HIDDEN_SIZE,
                intermediate_size=MOE_INTERMEDIATE_SIZE,
                moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
                num_experts=NUM_EXPERTS,
                moe_topk=TOP_K,
                use_mixed_mlp_moe=0,
                num_shared_expert=0,
                hidden_act="silu",
            )
            with torch.device(device):
                moe = HunYuanSparseMoeBlock(
                    config=hf_cfg,
                    quant_config=None,
                    layer_id=0,
                    prefix="layers.0.mlp",
                ).to(dtype=torch.bfloat16)
            _load_hunyuan_moe_weights(moe, gate_w, w1, w3, w2, device)

            setup_info = (
                get_ep_group().world_size,
                moe.experts.moe_config.moe_parallel_config.use_ep,
                moe.experts.routed_experts.local_num_experts,
            )

            x_tpu = x_cpu.to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                out = moe(x_tpu)

            torch.accelerator.synchronize()
            out_cpu = out.detach().cpu().float().numpy() if rank == 0 else None
            result_queue.put(("ok", rank, setup_info, out_cpu))
    except Exception as exc:
        result_queue.put(("error", rank, setup_info, exc))
    finally:
        destroy_distributed_env()


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=DIFFUSION_EP_XFAIL_REASON,
)
def test_diffusion_expert_parallelism() -> None:
    """Diffusion Expert Parallelism: 2-rank HunYuanSparseMoeBlock on TPU.

    - TPU path: run a 4-expert bf16 ``HunYuanSparseMoeBlock`` across 2 spawned TPU
      ranks with ``tensor_parallel_size=2`` and ``enable_expert_parallel=True``.
    - Setup check: assert ``get_ep_group().world_size == 2``, ``use_ep is True``,
      and ``local_num_experts == 2`` on both ranks.
    - Outcome check: compare rank-0 EP=2 output against single-device CPU fp32
      4-expert SwiGLU MoE reference via ``assert_parity``.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= EP_WORLD_SIZE, (
        f"Test requires {EP_WORLD_SIZE} TPU devices, found {device_count}"
    )

    gen = torch.Generator(device="cpu").manual_seed(89)
    x_cpu = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), generator=gen, dtype=torch.float32
    )
    gate_w, w1, w3, w2 = _sample_moe_weights()

    ref_fp32 = _hunyuan_sparse_moe_reference(x_cpu, gate_w, w1, w3, w2, torch.float32)
    ref_bf16 = _hunyuan_sparse_moe_reference(x_cpu, gate_w, w1, w3, w2, torch.bfloat16)
    cpu_bf16_rel_l2, cpu_bf16_max_abs = _error_metrics(ref_bf16, ref_fp32)

    master_port = portpicker.pick_unused_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    procs = [
        ctx.Process(
            target=_ep_worker,
            args=(rank, master_port, x_cpu, gate_w, w1, w3, w2, result_queue),
        )
        for rank in range(EP_WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    # Read before join: a worker's queued tensor storage dies with the worker.
    records = [result_queue.get() for _ in range(EP_WORLD_SIZE)]
    for proc in procs:
        proc.join()
    wait_for_tpu_release()

    records.sort(key=lambda item: item[1])

    expected_setup = (EP_WORLD_SIZE, True, NUM_EXPERTS // EP_WORLD_SIZE)
    for _status, rank, setup_info, payload in records:
        assert setup_info == expected_setup, (
            f"EP rank {rank}: setup {setup_info} != {expected_setup} "
            f"(payload={payload!r})"
        )
    for status, _rank, _setup, payload in records:
        if status == "error":
            raise payload

    rank0_out = records[0][3]
    assert rank0_out is not None, "Rank 0 did not return HunYuanSparseMoeBlock output"
    assert_parity(
        torch.from_numpy(rank0_out),
        ref_fp32,
        name="hunyuan_moe_ep2",
        max_rel_l2=max(BF16_ERR_RATIO * cpu_bf16_rel_l2, MOE_REL_L2_FLOOR),
        max_abs=max(BF16_ERR_RATIO * cpu_bf16_max_abs, MOE_ABS_FLOOR),
    )
