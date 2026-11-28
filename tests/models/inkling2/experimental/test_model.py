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
"""CPU end-to-end parity tests for the TPU-portable Inkling 2 Loop model."""

from __future__ import annotations

import torch

from vllm_torchtpu.models.inkling2.experimental.config import Inkling2LoopConfig
from vllm_torchtpu.models.inkling2.experimental.model import (
    Inkling2LoopCache,
    Inkling2LoopForCausalLM,
)


def _tiny_loop_config(*, num_passes: int = 3) -> Inkling2LoopConfig:
    return Inkling2LoopConfig(
        vocab_size=64,
        unpadded_vocab_size=60,
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=4,
        rel_extent=8,
        log_scaling_n_floor=8,
        local_layer_ids=(0, 2, 4, 6),
        sconv_kernel_size=4,
        sliding_window_size=8,
        n_layers_per_attn_res_block=2,
        n_routed_experts=6,
        num_experts_per_tok=2,
        intermediate_size=16,
        core_start=2,
        core_end=6,
        num_passes=num_passes,
        gate_rank=8,
    )


def test_config_visits_and_hf_parsing() -> None:
    mini = Inkling2LoopConfig()
    assert mini.num_logical_layers == 48
    assert mini.intermediate_size == 1152
    assert mini.visits[0] == (0, 0)
    assert mini.visits[6] == (6, 0)
    assert mini.visits[18] == (6, 1)
    assert mini.visits[30] == (6, 2)
    assert mini.visits[-1] == (23, 0)

    parsed = Inkling2LoopConfig.from_hf_config(
        {
            "text_config": {
                "vocab_size": 64,
                "unpadded_vocab_size": 60,
                "hidden_size": 32,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "d_rel": 4,
                "rel_extent": 8,
                "qk_l2_clip_threshold": 22.6274,
                "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": [0, 2, 4, 6],
                "n_routed_experts": 6,
                "num_experts_per_tok": 2,
                "intermediate_size": 16,
                "loop_config": {
                    "core_start": 2,
                    "core_end": 6,
                    "num_passes": 3,
                    "gate_rank": 8,
                },
            }
        }
    )
    assert parsed.num_logical_layers == 16


def test_loop_model_split_tpu_matches_dense_and_incremental_decode() -> None:
    torch.manual_seed(123)
    cfg = _tiny_loop_config(num_passes=3)
    model = Inkling2LoopForCausalLM(cfg).eval()
    seq_len = 14  # > rel_extent=8 so full-attention layers exercise far + near merge
    input_ids = torch.randint(0, cfg.unpadded_vocab_size, (seq_len,))
    positions = torch.arange(seq_len, dtype=torch.long)

    with torch.no_grad():
        logits_dense = model(input_ids, positions, attn_impl="dense")
        logits_split = model(input_ids, positions, attn_impl="split_tpu")

        cache = Inkling2LoopCache()
        prefill_len = 9
        step_logits = [
            model(
                input_ids[:prefill_len],
                positions[:prefill_len],
                cache,
                attn_impl="split_tpu",
            )
        ]
        for idx in range(prefill_len, seq_len):
            step_logits.append(
                model(
                    input_ids[idx : idx + 1],
                    positions[idx : idx + 1],
                    cache,
                    attn_impl="split_tpu",
                )
            )
        logits_inc = torch.cat(step_logits, dim=0)

    assert logits_split.shape == (seq_len, cfg.unpadded_vocab_size)
    assert len(cache.visits) == cfg.num_logical_layers
    torch.testing.assert_close(logits_split, logits_dense, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits_inc, logits_split, atol=1e-4, rtol=1e-4)


def test_non_looped_inkling2_schedule_runs_cleanly() -> None:
    torch.manual_seed(456)
    cfg = _tiny_loop_config(num_passes=1)
    assert cfg.num_logical_layers == cfg.num_hidden_layers
    model = Inkling2LoopForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.unpadded_vocab_size, (10,))
    positions = torch.arange(10, dtype=torch.long)
    with torch.no_grad():
        logits_dense = model(input_ids, positions, attn_impl="dense")
        logits_split = model(input_ids, positions, attn_impl="split_tpu")
    torch.testing.assert_close(logits_split, logits_dense, atol=1e-4, rtol=1e-4)
