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
"""Tests for runner-side internal prefill checkpoint offset computation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes
from vllm_torchtpu.gdn_pool_layout import PrefillCheckpointUnits
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_PCP_LAYOUT_RANK = "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank"
_PCP_LAYOUT_WORLD_SIZE = (
    "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size"
)

PCP8_B = 384 * 8


def _pcp8_units(drop_eagle_block):
    return PrefillCheckpointUnits(
        hash_block_size=PCP8_B,
        mamba_block_size=PCP8_B,
        alignment=64,
        drop_eagle_block=drop_eagle_block,
    )


def _fill(units, seq_lens, query_lens, out_len=None):
    out = np.full(out_len or len(seq_lens), 99, dtype=np.int32)
    units.fill_offsets(
        out,
        np.asarray(seq_lens, dtype=np.int32),
        np.asarray(query_lens, dtype=np.int32),
    )
    return out.tolist()


@pytest.mark.parametrize("drop_eagle_block, expected", [(True, 3072), (False, 6144)])
def test_pcp8_fresh_8191_token_prompt(drop_eagle_block, expected):
    assert _fill(_pcp8_units(drop_eagle_block), [8191], [8191]) == [expected]


def test_pcp8_second_chunk_of_a_19456_token_prompt():
    # 15360 + 4096: the checkpoint at 18432 is 3072 into the second chunk.
    assert _fill(_pcp8_units(False), [15360, 19456], [15360, 4096]) == [12288, 3072]


def test_pcp8_mixed_batch_zeroes_rows_without_a_checkpoint_and_the_tail():
    seq_lens = [9000, 9002, 8191, 3000, 19456]
    query_lens = [1, 2, 8191, 3000, 4096]
    # Decode and verify rows, and queries of at most one hash block, never
    # checkpoint; the padded tail (stale 99s) is zeroed.
    assert _fill(_pcp8_units(False), seq_lens, query_lens, out_len=8) == [
        0,
        0,
        6144,
        0,
        3072,
        0,
        0,
        0,
    ]


def _pcp8_vllm_config(
    *, prefix_match_unit=None, speculative=True, disable_eagle_block_drop=False
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=384,
            mamba_block_size=384,
            enable_prefix_caching=True,
            prefix_match_unit=prefix_match_unit,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=8,
        ),
        model_config=SimpleNamespace(architecture="Qwen3_5MoeForCausalLM"),
        speculative_config=(
            SimpleNamespace(use_eagle_block_drop=lambda: not disable_eagle_block_drop)
            if speculative
            else None
        ),
        kv_transfer_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
    )


def _pcp8_kv_cache_config(*, declares=True):
    attention = FullAttentionSpec(
        block_size=384, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    groups = [KVCacheGroupSpec(["attn.0"], attention)]
    for gid in range(3):
        groups.append(
            KVCacheGroupSpec(
                [f"gdn.{gid}"],
                MambaSpec(
                    block_size=384,
                    shapes=((3, 768), (4, 128, 128)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_cache_mode="align",
                    num_speculative_blocks=1,
                    num_prefill_checkpoint_blocks=1 if declares else 0,
                    prefill_checkpoint_alignment=64 if declares else None,
                ),
            )
        )
    return KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)


@pytest.fixture
def pcp8_runner(monkeypatch):
    # Install the PCP resolver for this test only. Another test can leave the
    # patch marker set while its monkeypatch restores the upstream resolver,
    # which would turn the call below into a no-op.
    monkeypatch.setattr(
        kv_cache_utils, "_tpu_hybrid_pcp_block_sizes_patch", False, raising=False
    )
    monkeypatch.setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        kv_cache_utils.resolve_kv_cache_block_sizes,
    )
    _patch_vllm_hybrid_pcp_block_sizes()
    for name in (
        "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP",
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        runner_mod, "get_pcp_group", lambda: SimpleNamespace(world_size=8)
    )
    runner = SimpleNamespace(max_num_reqs=16, vllm_config=_pcp8_vllm_config())
    return runner, monkeypatch


def _init(runner, kv_cache_config):
    TPUModelRunner._init_prefill_checkpoint_units(runner, kv_cache_config)
    return runner._prefill_ckpt_units


def test_pcp8_units_match_the_engine_core(pcp8_runner):
    runner, _ = pcp8_runner
    units = _init(runner, _pcp8_kv_cache_config())

    assert units == PrefillCheckpointUnits(
        hash_block_size=3072,
        mamba_block_size=3072,
        alignment=64,
        drop_eagle_block=True,
    )
    assert runner._prefill_ckpt_offsets_cpu.shape == (16,)
    assert runner._prefill_ckpt_offsets_cpu.dtype == torch.int32


def test_prefix_match_unit_sets_the_hash_block(pcp8_runner):
    runner, _ = pcp8_runner
    runner.vllm_config = _pcp8_vllm_config(prefix_match_unit=3072)
    units = _init(runner, _pcp8_kv_cache_config())
    assert units.hash_block_size == 3072
    assert units.mamba_block_size == 3072


def test_finer_prefix_match_unit_leaves_no_units(pcp8_runner):
    runner, _ = pcp8_runner
    runner.vllm_config = _pcp8_vllm_config(prefix_match_unit=1024)
    assert _init(runner, _pcp8_kv_cache_config()) is None


def test_eagle_override_reaches_the_worker_in_the_config(pcp8_runner):
    runner, _ = pcp8_runner
    runner.vllm_config = _pcp8_vllm_config(disable_eagle_block_drop=True)
    assert _init(runner, _pcp8_kv_cache_config()).drop_eagle_block is False


def test_eagle_env_in_the_worker_does_not_override_the_config(pcp8_runner):
    runner, monkeypatch = pcp8_runner
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    assert _init(runner, _pcp8_kv_cache_config()).drop_eagle_block is True


def test_no_mtp_has_no_eagle_drop(pcp8_runner):
    runner, _ = pcp8_runner
    runner.vllm_config = _pcp8_vllm_config(speculative=False)
    assert _init(runner, _pcp8_kv_cache_config()).drop_eagle_block is False


def test_non_align_mode_leaves_no_units(pcp8_runner):
    runner, _ = pcp8_runner
    runner.vllm_config.cache_config.mamba_cache_mode = "all"
    assert _init(runner, _pcp8_kv_cache_config()) is None
    assert runner._prefill_ckpt_offsets_cpu is None
    assert runner._prefill_ckpt_zero_offsets == {}


def test_dummy_run_zero_offsets_are_uploaded_once_per_bucket(pcp8_runner):
    runner, _ = pcp8_runner
    runner.device = "cpu"
    _init(runner, _pcp8_kv_cache_config())

    first = TPUModelRunner._dummy_prefill_ckpt_offsets(runner, 8)
    assert first.tolist() == [0] * 8 and first.dtype == torch.int32
    assert TPUModelRunner._dummy_prefill_ckpt_offsets(runner, 8) is first
    assert TPUModelRunner._dummy_prefill_ckpt_offsets(runner, 16).shape == (16,)
    assert set(runner._prefill_ckpt_zero_offsets) == {8, 16}


def test_undeclared_mamba_groups_leave_no_units(pcp8_runner):
    runner, _ = pcp8_runner
    assert _init(runner, _pcp8_kv_cache_config(declares=False)) is None


def test_prepare_inputs_uploads_logical_offsets_under_pcp(monkeypatch):
    # PCP2 with 16-token rank-local blocks: logical B = H = 32. The 80-token
    # request (40 rows per rank) checkpoints at 64, its logical position.
    scheduled = [80, 20]
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
        pcp_size=2,
    )
    runner._prefill_ckpt_units = PrefillCheckpointUnits(
        hash_block_size=32,
        mamba_block_size=32,
        alignment=16,
        drop_eagle_block=False,
    )
    # Stale values from an earlier chunk must not leak into padded rows.
    runner._prefill_ckpt_offsets_cpu = torch.full(
        (runner.max_num_reqs,), 7, dtype=torch.int32
    )
    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    TPUModelRunner._prepare_inputs(runner, _scheduler_output(scheduled), 0, 0)

    offsets = runner._attn_metadata_builder_ctx.mamba_prefill_ckpt_offsets
    assert offsets.tolist() == [64] + [0] * (runner.num_reqs_max_model_len - 1)


def test_prepare_inputs_without_units_sets_no_offsets(monkeypatch):
    scheduled = [80]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
        pcp_size=2,
    )
    runner._prefill_ckpt_units = None
    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    TPUModelRunner._prepare_inputs(runner, _scheduler_output(scheduled), 0, 0)

    assert runner._attn_metadata_builder_ctx.mamba_prefill_ckpt_offsets is None
