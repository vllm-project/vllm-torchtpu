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
"""TPU tests for MiniMax-H3 Qwen3-VL diffusion text encoder tensor parallelism."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch
from torch_tpu._internal.utils import hardware
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    init_world_group,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

TP_SIZE = 2
VOCAB_SIZE = 256
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 1024
NUM_LAYERS = 2
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 128
REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    diff = actual.detach().cpu().double() - reference.detach().cpu().double()
    return float(diff.norm() / reference.detach().cpu().double().norm())


def _assert_bf16_parity(
    actual: torch.Tensor,
    ref_fp32: torch.Tensor,
    ref_bf16: torch.Tensor,
    *,
    name: str,
) -> None:
    assert actual.shape == ref_fp32.shape, (
        f"{name}: shape {tuple(actual.shape)} != expected {tuple(ref_fp32.shape)}"
    )
    assert bool(torch.isfinite(actual).all()), f"{name}: output has non-finite values"
    cpu_bf16_err = _rel_l2(ref_bf16, ref_fp32)
    limit = max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR)
    tpu_err = _rel_l2(actual, ref_fp32)
    assert tpu_err <= limit, (
        f"{name}: relative L2 error {tpu_err:.4e} exceeds limit {limit:.4e} "
        f"(cpu_bf16_err={cpu_bf16_err:.4e})"
    )


def _build_tiny_qwen3vl_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "mrope_section": [24, 20, 20],
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": HIDDEN_SIZE,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=250,
        video_token_id=251,
    )


def _encoder_tp_worker(
    rank: int, world_size: int, master_port: int, work_dir: str
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        dtype=torch.bfloat16,
        num_gpus=world_size,
        parallel_config=DiffusionParallelConfig(text_encoder_tp_size=world_size),
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    current_omni_platform.init_diffusion_worker_vllm_config(vllm_config)

    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_diffusion_config(od_config),
    ):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=rank,
        )
        try:
            initialize_model_parallel(tensor_parallel_size=1)
            encoder_group = init_world_group(
                ranks=list(range(world_size)),
                local_rank=rank,
                backend=current_omni_platform.dist_backend,
            )
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            encoder = MiniMaxH3Qwen3VLEncoder(
                str(Path(work_dir) / "model_cfg"),
                device=device,
                load_model=True,
                encoder_group=encoder_group,
            )
            encoder.load_weights(payload["weights"].items())
            encoder.load_to_device()
            out = encoder.encode_ids(payload["input_ids"])

            layer0 = encoder.text_model.layers[0]
            torch.save(
                {
                    "tp_size": encoder.tp_size,
                    "embed_shape": tuple(encoder.text_model.embed_tokens.weight.shape),
                    "qkv_rows": int(layer0.self_attn.qkv_proj.weight.shape[0]),
                    "o_cols": int(layer0.self_attn.o_proj.weight.shape[1]),
                    "down_cols": int(layer0.mlp.down_proj.weight.shape[1]),
                    "output": out.detach().cpu().float(),
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


def test_diffusion_text_encoder_tensor_parallelism(tmp_path: Path) -> None:
    """Diffusion Text Encoder Tensor Parallelism: TP=2 Qwen3-VL encoder parity.

    - TPU path: spawns 2 workers on tpu_dist, creates an encoder GroupCoordinator
      over [0, 1], and runs a 2-layer MiniMaxH3Qwen3VLEncoder (vocab, QKV, merged
      gate/up, and fp32-all-reduced row-parallel linears + MRoPE + causal SDPA).
    - Setup check: encoder.tp_size == 2 and embed_tokens, qkv_proj, o_proj, and
      down_proj weights have half-sharded partition shapes on both ranks.
    - Outcome check: rank-0 and rank-1 embeddings match bitwise and stay within
      max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR) of CPU fp32 Qwen3-VL.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= TP_SIZE, (
        f"Text encoder TP test requires {TP_SIZE} TPU devices, found {device_count}"
    )

    cfg = _build_tiny_qwen3vl_config()
    model_dir = tmp_path / "model_cfg"
    cfg.save_pretrained(model_dir)

    # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
    torch.default_generator.manual_seed(0)
    hf_model = Qwen3VLForConditionalGeneration(cfg).eval()
    bf16_weights = {
        name: tensor.detach().to(torch.bfloat16)
        for name, tensor in hf_model.state_dict().items()
    }
    hf_model.load_state_dict(
        {name: tensor.float() for name, tensor in bf16_weights.items()}
    )
    # MiniMaxH3Qwen3VLEncoder returns unnormalized decoder hidden states.
    hf_model.model.language_model.norm = torch.nn.Identity()
    hf_bf16_lm = copy.deepcopy(hf_model.model.language_model).to(torch.bfloat16).eval()

    # Span both [0, 128) and [128, 256) vocab partitions across the two TP ranks.
    input_ids = torch.tensor([3, 19, 140, 211, 64, 128, 245, 42], dtype=torch.long)
    position_ids = (
        torch.arange(input_ids.numel(), dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1)
    )

    with torch.no_grad():
        ref_fp32 = (
            hf_model.model.language_model(
                input_ids=input_ids[None], position_ids=position_ids
            )
            .last_hidden_state[0]
            .float()
        )
        ref_bf16 = (
            hf_bf16_lm(input_ids=input_ids[None], position_ids=position_ids)
            .last_hidden_state[0]
            .float()
        )

    torch.save(
        {"weights": bf16_weights, "input_ids": input_ids},
        tmp_path / "payload.pt",
    )

    master_port = get_open_port()
    try:
        torch.multiprocessing.spawn(
            _encoder_tp_worker,
            args=(TP_SIZE, master_port, str(tmp_path)),
            nprocs=TP_SIZE,
        )
    finally:
        wait_for_tpu_release()

    rank0 = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank_1.pt", weights_only=True)

    expected_embed_shape = (VOCAB_SIZE // TP_SIZE, HIDDEN_SIZE)
    expected_qkv_rows = ((NUM_HEADS + 2 * NUM_KV_HEADS) // TP_SIZE) * HEAD_DIM
    expected_o_cols = (NUM_HEADS // TP_SIZE) * HEAD_DIM
    expected_down_cols = INTERMEDIATE_SIZE // TP_SIZE
    for rank_idx, res in enumerate((rank0, rank1)):
        assert res["tp_size"] == TP_SIZE, (
            f"rank {rank_idx}: expected encoder.tp_size={TP_SIZE}, got {res['tp_size']}"
        )
        assert res["embed_shape"] == expected_embed_shape, (
            f"rank {rank_idx}: expected embed_tokens shape {expected_embed_shape}, "
            f"got {res['embed_shape']}"
        )
        assert res["qkv_rows"] == expected_qkv_rows, (
            f"rank {rank_idx}: expected qkv_proj rows {expected_qkv_rows}, "
            f"got {res['qkv_rows']}"
        )
        assert res["o_cols"] == expected_o_cols, (
            f"rank {rank_idx}: expected o_proj cols {expected_o_cols}, "
            f"got {res['o_cols']}"
        )
        assert res["down_cols"] == expected_down_cols, (
            f"rank {rank_idx}: expected down_proj cols {expected_down_cols}, "
            f"got {res['down_cols']}"
        )

    assert torch.equal(rank0["output"], rank1["output"]), (
        "Encoder TP=2 rank-0 and rank-1 hidden states differed after all-reduce"
    )
    _assert_bf16_parity(
        rank0["output"], ref_fp32, ref_bf16, name="MiniMax-H3 Qwen3-VL TP=2"
    )
