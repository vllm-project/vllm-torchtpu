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
"""CPU unit tests for standalone request metadata bucketing (M1)."""

from __future__ import annotations

import threading
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

from tests.runner.runner_input_test_utils import (
    _BlockTable,
    _fake_tpu_events,
    _make_runner,
    _scheduler_output,
)
from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import PrefillCheckpointUnits
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.runner.metadata_req_bucketing import (
    build_warmup_req_groups,
    get_active_metadata_req_bucket,
    get_metadata_req_bucketing_stats,
    metadata_req_bucket_scope,
    record_metadata_step,
    reset_metadata_req_bucketing_stats,
    resolve_req_buckets,
    select_target_num_reqs,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

DEFAULT_BUCKETS = [4, 8, 16, 32, 64, 128, 256]
DEFAULT_TOKEN_PADDINGS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture(autouse=True)
def _reset_env_and_stats(monkeypatch):
    monkeypatch.delenv("TPU_METADATA_REQ_BUCKETING", raising=False)
    monkeypatch.delenv("TPU_METADATA_REQ_BUCKETS", raising=False)
    monkeypatch.delenv("TPU_PRECOMPILE_TRACE_FIRST", raising=False)
    reset_metadata_req_bucketing_stats()
    yield
    reset_metadata_req_bucketing_stats()


# ---------------------------------------------------------------------------
# 1. Environment variable parsing & fail-closed validation
# ---------------------------------------------------------------------------


def test_env_defaults():
    assert envs.TPU_METADATA_REQ_BUCKETING is False
    assert envs.TPU_METADATA_REQ_BUCKETS == [4, 8, 16, 256]


@pytest.mark.parametrize("blank", ["", "   "])
def test_env_blank_reads_as_default(monkeypatch, blank):
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", blank)
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETS", blank)
    assert envs.TPU_METADATA_REQ_BUCKETING is False
    assert envs.TPU_METADATA_REQ_BUCKETS == [4, 8, 16, 256]


def test_env_custom_buckets_parsing(monkeypatch):
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "1")
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETS", " 4 , 16 , 64 , 256 ")
    assert envs.TPU_METADATA_REQ_BUCKETING is True
    assert envs.TPU_METADATA_REQ_BUCKETS == [4, 16, 64, 256]


@pytest.mark.parametrize(
    "bad_buckets",
    ["0,8,16", "-4,8", "4,,8", ",4,8", "4,8,", "4,abc", "4,8.5", "none"],
)
def test_env_buckets_fail_closed_on_invalid_input(monkeypatch, bad_buckets):
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETS", bad_buckets)
    with pytest.raises(ValueError, match="TPU_METADATA_REQ_BUCKETS"):
        _ = envs.TPU_METADATA_REQ_BUCKETS


def test_env_bucketing_bool_fails_closed_on_invalid_input(monkeypatch):
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "enabled")
    with pytest.raises(ValueError, match="TPU_METADATA_REQ_BUCKETING"):
        _ = envs.TPU_METADATA_REQ_BUCKETING


# ---------------------------------------------------------------------------
# 2. metadata_req_bucket_scope: nesting, exception safety, thread isolation,
#    compilation hash isolation, and fail-closed inputs
# ---------------------------------------------------------------------------


def test_metadata_req_bucket_scope_nesting_and_exception_cleanup():
    assert get_active_metadata_req_bucket() is None
    with metadata_req_bucket_scope(16):
        assert get_active_metadata_req_bucket() == 16
        with metadata_req_bucket_scope(4):
            assert get_active_metadata_req_bucket() == 4
            with metadata_req_bucket_scope(None):
                assert get_active_metadata_req_bucket() is None
            assert get_active_metadata_req_bucket() == 4
        assert get_active_metadata_req_bucket() == 16
        with (
            pytest.raises(RuntimeError, match="boom"),
            metadata_req_bucket_scope(64),
        ):
            assert get_active_metadata_req_bucket() == 64
            raise RuntimeError("boom")
        assert get_active_metadata_req_bucket() == 16
    assert get_active_metadata_req_bucket() is None


def test_metadata_req_bucket_scope_is_thread_local():
    barrier = threading.Barrier(2)
    observed: dict[str, list[int | None]] = {"t1": [], "t2": []}

    def worker(name: str, bucket: int) -> None:
        observed[name].append(get_active_metadata_req_bucket())
        with metadata_req_bucket_scope(bucket):
            barrier.wait()
            observed[name].append(get_active_metadata_req_bucket())
            barrier.wait()
        observed[name].append(get_active_metadata_req_bucket())

    t1 = threading.Thread(target=worker, args=("t1", 4))
    t2 = threading.Thread(target=worker, args=("t2", 128))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert observed["t1"] == [None, 4, None]
    assert observed["t2"] == [None, 128, None]


@pytest.mark.parametrize("bad_scope", [0, -8, True, False, "16", 4.0])
def test_metadata_req_bucket_scope_fails_closed_on_bad_values(bad_scope):
    with (
        pytest.raises(ValueError, match="num_reqs"),
        metadata_req_bucket_scope(bad_scope),  # type: ignore[arg-type]
    ):
        pass


def test_compilation_hash_distinguishes_metadata_req_bucket_scopes():
    from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.7,
            num_gpu_blocks=9573,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=True,
            max_num_batched_tokens=4096,
            max_num_seqs=256,
        ),
        model_config=SimpleNamespace(is_hybrid=True, max_model_len=4352),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            nnodes=1,
        ),
    )

    with (
        patch(
            "vllm_torchtpu.tpu_info.get_libtpu_cache_key", return_value="fake_libtpu"
        ),
        patch(
            "vllm_torchtpu.compilation.tpu_compiler.prefill_checkpoint_enabled",
            return_value=False,
        ),
    ):
        h_none = compute_tpu_compilation_hash(vllm_cfg)
        with metadata_req_bucket_scope(4):
            h_4 = compute_tpu_compilation_hash(vllm_cfg)
        with metadata_req_bucket_scope(8):
            h_8 = compute_tpu_compilation_hash(vllm_cfg)
        with metadata_req_bucket_scope(256):
            h_256 = compute_tpu_compilation_hash(vllm_cfg)

    assert len({h_none, h_4, h_8, h_256}) == 4


# ---------------------------------------------------------------------------
# 3. Bucket resolution, target selection, warmup grouping & coverage theorem
# ---------------------------------------------------------------------------


def test_resolve_req_buckets_sorts_deduplicates_and_caps_at_max_num_seqs():
    assert resolve_req_buckets([256, 16, 4, 8, 16, 512, 64, 32, 128], 256) == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    # Non-power-of-two max_num_seqs is always preserved as the exact top cap.
    assert resolve_req_buckets(DEFAULT_BUCKETS, 96) == [4, 8, 16, 32, 64, 96]
    assert resolve_req_buckets([64, 128], 16) == [16]
    assert resolve_req_buckets([], 256) == [256]


@pytest.mark.parametrize(
    "raw,cap",
    [
        ([0, 8], 256),
        ([-4, 8], 256),
        ([True, 8], 256),
        ([4, 8], 0),
        ([4, 8], -16),
        ([4, 8], False),
    ],
)
def test_resolve_req_buckets_fails_closed(raw, cap):
    with pytest.raises(ValueError):
        resolve_req_buckets(raw, cap)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "active_reqs,num_tokens_padded,is_decode,expected",
    [
        (1, 16, True, 4),
        (3, 16, True, 4),
        (4, 16, True, 4),
        (5, 16, True, 8),
        (8, 16, True, 8),
        (9, 16, True, 16),
        (15, 16, True, 16),
        (16, 16, True, 16),
        (17, 32, True, 32),
        (32, 32, True, 32),
        (33, 64, True, 64),
        (65, 128, True, 128),
        (129, 256, True, 256),
        (256, 256, True, 256),
        # Prefill / mixed steps in min token bucket use token_req_cap (=16)
        (1, 16, False, 16),
        (4, 16, False, 16),
        # Larger prefill / mixed chunks cap at max_reqs_cap=256
        (1, 512, False, 256),
        (16, 2048, False, 256),
        (64, 4096, False, 256),
        (256, 4096, False, 256),
    ],
)
def test_select_target_num_reqs_boundaries(
    active_reqs, num_tokens_padded, is_decode, expected
):
    got = select_target_num_reqs(
        num_tokens_padded=num_tokens_padded,
        min_token_bucket=16,
        active_num_reqs=active_reqs,
        is_decode_only=is_decode,
        req_buckets=DEFAULT_BUCKETS,
        max_reqs_cap=256,
    )
    assert got == expected
    assert active_reqs <= got <= 256


def test_warmup_groups_cover_every_reachable_step_shape():
    groups = build_warmup_req_groups(DEFAULT_TOKEN_PADDINGS, DEFAULT_BUCKETS, 256)
    assert groups == [
        (4, [16]),
        (8, [16]),
        (16, [16]),
        (32, [32]),
        (64, [64]),
        (128, [128]),
        (256, [256, 512, 1024, 2048, 4096]),
    ]
    warmed_pairs = {(r, tok) for r, toks in groups for tok in toks}
    # Total compiled (req_cap, token_bucket) pairs is 11 (vs 9 without M1).
    assert len(warmed_pairs) == 11

    for tok in DEFAULT_TOKEN_PADDINGS:
        max_active = min(tok, 256)
        for active in range(1, max_active + 1):
            for is_dec in (True, False):
                if is_dec and tok > DEFAULT_TOKEN_PADDINGS[0] and active <= tok // 2:
                    # Still valid if DP/scheduler chose a larger token bucket
                    pass
                r = select_target_num_reqs(
                    num_tokens_padded=tok,
                    min_token_bucket=DEFAULT_TOKEN_PADDINGS[0],
                    active_num_reqs=active,
                    is_decode_only=is_dec,
                    req_buckets=DEFAULT_BUCKETS,
                    max_reqs_cap=256,
                )
                assert (r, tok) in warmed_pairs, (
                    f"Unwarmed pair {(r, tok)} for active={active}"
                )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            num_tokens_padded=16,
            min_token_bucket=16,
            active_num_reqs=-1,
            is_decode_only=True,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=256,
        ),
        dict(
            num_tokens_padded=512,
            min_token_bucket=16,
            active_num_reqs=257,
            is_decode_only=False,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=256,
        ),
        dict(
            num_tokens_padded=16,
            min_token_bucket=16,
            active_num_reqs=17,
            is_decode_only=True,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=256,
        ),
        dict(
            num_tokens_padded=0,
            min_token_bucket=16,
            active_num_reqs=0,
            is_decode_only=True,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=256,
        ),
        dict(
            num_tokens_padded=16,
            min_token_bucket=0,
            active_num_reqs=1,
            is_decode_only=True,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=256,
        ),
        dict(
            num_tokens_padded=16,
            min_token_bucket=16,
            active_num_reqs=1,
            is_decode_only=True,
            req_buckets=DEFAULT_BUCKETS,
            max_reqs_cap=0,
        ),
    ],
)
def test_select_target_num_reqs_fails_closed_on_impossible_bounds(kwargs):
    with pytest.raises(ValueError):
        select_target_num_reqs(**kwargs)


def test_stats_recording_and_reset():
    record_metadata_step(
        num_tokens_padded=16,
        target_num_reqs=4,
        is_decode_only=True,
        max_num_seqs_cap=256,
    )
    record_metadata_step(
        num_tokens_padded=16,
        target_num_reqs=4,
        is_decode_only=True,
        max_num_seqs_cap=256,
    )
    record_metadata_step(
        num_tokens_padded=512,
        target_num_reqs=256,
        is_decode_only=False,
        max_num_seqs_cap=256,
    )
    stats = get_metadata_req_bucketing_stats()
    assert stats == {
        "step_counts": {"tok16_req4_dec": 2, "tok512_req256_mix": 1},
        "max_num_seqs_cap": 256,
    }
    reset_metadata_req_bucketing_stats()
    assert get_metadata_req_bucketing_stats() == {
        "step_counts": {},
        "max_num_seqs_cap": None,
    }


# ---------------------------------------------------------------------------
# 4. Wrapper capture/activation & forward_model fail-closed dispatch
# ---------------------------------------------------------------------------


def test_forward_model_switches_wrappers_and_fails_closed_on_uncompiled_bucket():
    calls: list[str] = []

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tag = "init"

        def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
            calls.append(self.tag)
            return torch.zeros((input_ids.shape[0], 4))

    model = DummyModel()
    runner = SimpleNamespace(
        model=model,
        _req_bucket_states={
            4: [(model, {"tag": "req4"})],
            256: [(model, {"tag": "req256"})],
        },
        _current_target_num_reqs=4,
        _compiling_req_bucket=False,
    )
    TPUModelRunner.forward_model(
        runner, torch.zeros(16, dtype=torch.int32), torch.zeros(16, dtype=torch.int32)
    )
    runner._current_target_num_reqs = 256
    TPUModelRunner.forward_model(
        runner, torch.zeros(256, dtype=torch.int32), torch.zeros(256, dtype=torch.int32)
    )
    assert calls == ["req4", "req256"]

    runner._current_target_num_reqs = 32
    with pytest.raises(RuntimeError, match="Uncompiled metadata request bucket 32"):
        TPUModelRunner.forward_model(
            runner,
            torch.zeros(32, dtype=torch.int32),
            torch.zeros(32, dtype=torch.int32),
        )


# ---------------------------------------------------------------------------
# 5. TPUModelRunner._prepare_inputs end-to-end CPU padding invariants
#    (seq_lens, query_start_loc, block_tables, mamba_state_indices,
#     prefill_ckpt_offsets, decode cache key)
# ---------------------------------------------------------------------------


def _attach_real_attn_and_mamba_groups(runner: Any, max_num_reqs: int = 256) -> None:
    """Configure AttentionMetadataBuilder + Mamba slot pool + prefill ckpts on CPU."""
    runner.max_num_reqs = max_num_reqs
    runner.num_reqs_max_model_len = max_num_reqs
    runner.enforce_eager = False
    runner.requests = {
        f"req{i}": SimpleNamespace() for i in range(runner.input_batch.num_reqs)
    }
    runner._req_bucket_states = {}
    runner._current_target_num_reqs = None
    runner._compiling_req_bucket = False
    runner._metadata_req_bucketing_enabled = types.MethodType(
        TPUModelRunner._metadata_req_bucketing_enabled, runner
    )
    runner._resolved_metadata_req_buckets = types.MethodType(
        TPUModelRunner._resolved_metadata_req_buckets, runner
    )

    # Expand CPU staging buffers for max_num_reqs=256
    max_toks = max(runner.max_num_tokens, max_num_reqs + 16)
    runner.query_start_loc_cpu = torch.zeros(max_toks + 1, dtype=torch.int32)
    runner.query_start_loc_np = runner.query_start_loc_cpu.numpy()
    runner.seq_lens_cpu = torch.zeros(max_toks, dtype=torch.int32)
    runner.seq_lens_np = runner.seq_lens_cpu.numpy()

    # Compact Mamba slot pool (slot 0 = null sentinel, slots 1..300 usable)
    runner._mamba_slot_stride = 1
    TPUModelRunner._init_mamba_slot_pool(runner, 300)
    runner._build_mamba_state_indices = types.MethodType(
        TPUModelRunner._build_mamba_state_indices, runner
    )

    # Prefill checkpoint offsets buffer
    runner._prefill_ckpt_units = PrefillCheckpointUnits(
        hash_block_size=16,
        cache_hit_alignment_tokens=16,
        mamba_block_size=64,
        alignment=16,
        drop_eagle_block=False,
        num_hash_lookahead_tokens=0,
    )
    runner._prefill_ckpt_offsets_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)

    # Real FullAttention + Mamba KV cache groups & builders
    fa_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=16 * 128 * 2,
    )
    mamba_spec = MambaSpec(
        shapes=((16, 128), (3, 128)),
        dtypes=(torch.float32, torch.float32),
        block_size=64,
    )
    fa_bt = torch.zeros((max_num_reqs, 8), dtype=torch.int32)
    mamba_bt = torch.zeros((max_num_reqs, 4), dtype=torch.int32)
    num_live = runner.input_batch.num_reqs
    for r in range(num_live):
        fa_bt[r] = torch.arange(10 + r * 8, 10 + (r + 1) * 8, dtype=torch.int32)
        mamba_bt[r] = torch.arange(100 + r * 4, 100 + (r + 1) * 4, dtype=torch.int32)
    runner.input_batch.block_table = [_BlockTable(fa_bt), _BlockTable(mamba_bt)]

    fa_builder = AttentionMetadataBuilder(
        kv_cache_spec=fa_spec,
        layer_names=["layers.3.attn"],
        vllm_config=runner.vllm_config,
        device=torch.device("cpu"),
        runner=runner,
        kv_cache_group_id=0,
    )

    mamba_builder = AttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layers.0.gdn"],
        vllm_config=runner.vllm_config,
        device=torch.device("cpu"),
        runner=runner,
        kv_cache_group_id=1,
    )
    # Test prefill ckpt offset propagation on mamba group without unified pool assertion
    mamba_builder.has_prefill_ckpt_blocks = False

    runner.kv_cache_config = SimpleNamespace(
        num_blocks=512,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layers.3.attn"], kv_cache_spec=fa_spec),
            KVCacheGroupSpec(layer_names=["layers.0.gdn"], kv_cache_spec=mamba_spec),
        ],
        has_mamba_layers=True,
    )
    runner.attn_groups = [
        [
            SimpleNamespace(
                layer_names=["layers.3.attn"], metadata_builders=[fa_builder]
            ),
            SimpleNamespace(
                layer_names=["layers.0.gdn"], metadata_builders=[mamba_builder]
            ),
        ]
    ]
    runner._build_attention_metadata = types.MethodType(
        TPUModelRunner._build_attention_metadata, runner
    )


def test_prepare_inputs_decode_padding_invariants_and_stale_buffer_scrubbing(
    monkeypatch,
):
    """3 active decode requests with max_num_seqs=256 must slice all metadata to R=4
    and scrub stale non-zero tail entries to 0."""
    _fake_tpu_events(monkeypatch)
    runner = _make_runner(
        num_computed_tokens=[20, 35, 50],
        prompt_tokens=[16, 32, 48],
        scheduled_tokens=[1, 1, 1],
        token_paddings=[16, 32, 64, 128, 256],
        pcp_size=1,
        max_num_reqs=256,
    )
    sched_out = _scheduler_output([1, 1, 1])
    _attach_real_attn_and_mamba_groups(runner, max_num_reqs=256)

    # Poison stale tail positions from a hypothetical prior larger batch!
    runner.seq_lens_np[:] = 999
    runner.query_start_loc_np[:] = 777
    assert runner._prefill_ckpt_offsets_cpu is not None
    runner._prefill_ckpt_offsets_cpu[:] = 555

    # First verify stock default (TPU_METADATA_REQ_BUCKETING=0) keeps full 256
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "0")
    out_ctrl = TPUModelRunner._prepare_inputs(
        runner, sched_out, start_index=0, num_decode_reqs=3, num_windowed_reqs=3
    )
    attn_ctrl = out_ctrl[0]
    assert tuple(attn_ctrl["layers.3.attn"].seq_lens.shape) == (256,)
    assert tuple(attn_ctrl["layers.3.attn"].query_start_loc.shape) == (257,)
    assert tuple(attn_ctrl["layers.3.attn"].block_tables.shape) == (256 * 8,)
    assert tuple(attn_ctrl["layers.0.gdn"].mamba_state_indices.shape) == (256,)
    assert runner._current_target_num_reqs is None
    assert runner.max_num_reqs == 256

    # Re-poison tail and enable M1 (TPU_METADATA_REQ_BUCKETING=1)
    runner.seq_lens_np[:] = 999
    runner.query_start_loc_np[:] = 777
    runner._prefill_ckpt_offsets_cpu[:] = 555
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "1")

    out_m1 = TPUModelRunner._prepare_inputs(
        runner, sched_out, start_index=0, num_decode_reqs=3, num_windowed_reqs=3
    )
    attn_m1 = out_m1[0]
    fa_md = attn_m1["layers.3.attn"]
    gdn_md = attn_m1["layers.0.gdn"]

    # Global admission capacity is unchanged (256), while step metadata is R=4!
    assert runner.max_num_reqs == 256
    assert runner.num_reqs_max_model_len == 256
    assert runner._current_target_num_reqs == 4

    # 1. seq_lens: active [21, 36, 51], padded slot [3] scrubbed to 0
    assert fa_md.seq_lens.tolist() == [21, 36, 51, 0]
    assert gdn_md.seq_lens.tolist() == [21, 36, 51, 0]

    # 2. query_start_loc: length R+1 = 5, padded q_len = 0 (3 - 3 = 0)
    assert fa_md.query_start_loc.tolist() == [0, 1, 2, 3, 3]

    # 3. block_tables: FA (4 * 8 = 32) and Mamba (4 * 4 = 16), padded row 3 all zeros
    fa_bt_2d = fa_md.block_tables.view(4, 8)
    assert torch.equal(
        fa_bt_2d[:3], runner.input_batch.block_table[0].get_cpu_tensor()[:3]
    )
    assert torch.all(fa_bt_2d[3] == 0)

    mamba_bt_2d = gdn_md.block_tables.view(4, 4)
    assert torch.equal(
        mamba_bt_2d[:3], runner.input_batch.block_table[1].get_cpu_tensor()[:3]
    )
    assert torch.all(mamba_bt_2d[3] == 0)

    # 4. mamba_state_indices: length 4, unique active slots, null slot 0 at tail
    sids = gdn_md.mamba_state_indices.tolist()
    assert len(sids) == 4
    assert all(s > 0 for s in sids[:3])
    assert len(set(sids[:3])) == 3
    assert sids[3] == 0

    # 5. prefill_ckpt_offsets in builder context: length 4, scrubbed to 0
    ctx_ckpt = runner._attn_metadata_builder_ctx.mamba_prefill_ckpt_offsets
    assert ctx_ckpt is not None
    assert ctx_ckpt.tolist() == [0, 0, 0, 0]

    # 6. Decode cache key distinguishes target_num_reqs=4 vs 256
    assert runner._decode_device_cache_key[2] == 4


def test_prepare_inputs_mixed_prefill_uses_token_bucket_cap(monkeypatch):
    """Mixed batch (1 decode + 1 prefill of 20 tokens => 21 tokens => tok=32):
    - under default [4, 8, 16, 256], tok=32 promotes to R=256 (or tok=16 selects R=16);
    - under 7-bucket [4, 8, 16, 32, 64, 128, 256], tok=32 selects target_num_reqs=32
      and preserves zero-padded tails across rows 2..31."""
    _fake_tpu_events(monkeypatch)
    runner = _make_runner(
        num_computed_tokens=[24, 0],
        prompt_tokens=[20, 20],
        scheduled_tokens=[1, 12],
        token_paddings=[16, 32, 64, 128, 256],
        pcp_size=1,
        max_num_reqs=256,
    )
    sched_out_16 = _scheduler_output([1, 12])
    _attach_real_attn_and_mamba_groups(runner, max_num_reqs=256)
    runner.seq_lens_np[:] = 888
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "1")

    out_16 = TPUModelRunner._prepare_inputs(
        runner, sched_out_16, start_index=0, num_decode_reqs=1, num_windowed_reqs=1
    )
    assert runner._current_target_num_reqs == 16
    assert tuple(out_16[0]["layers.3.attn"].seq_lens.shape) == (16,)

    runner2 = _make_runner(
        num_computed_tokens=[24, 0],
        prompt_tokens=[20, 20],
        scheduled_tokens=[1, 20],
        token_paddings=[16, 32, 64, 128, 256],
        pcp_size=1,
        max_num_reqs=256,
    )
    sched_out = _scheduler_output([1, 20])
    _attach_real_attn_and_mamba_groups(runner2, max_num_reqs=256)
    runner2.seq_lens_np[:] = 888
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETS", "4,8,16,32,64,128,256")

    out = TPUModelRunner._prepare_inputs(
        runner2, sched_out, start_index=0, num_decode_reqs=1, num_windowed_reqs=1
    )
    fa_md = out[0]["layers.3.attn"]
    gdn_md = out[0]["layers.0.gdn"]

    assert runner2._current_target_num_reqs == 32
    assert tuple(fa_md.seq_lens.shape) == (32,)
    assert fa_md.seq_lens[:2].tolist() == [25, 20]
    assert fa_md.seq_lens[2:].tolist() == [0] * 30
    assert tuple(fa_md.query_start_loc.shape) == (33,)
    assert fa_md.query_start_loc[:3].tolist() == [0, 1, 21]
    assert fa_md.query_start_loc[3:].tolist() == [21] * 30
    assert gdn_md.mamba_state_indices[2:].tolist() == [0] * 30


# ---------------------------------------------------------------------------
# 6. Fail-closed gate checks on unsupported runner configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutator",
    [
        lambda r, mp: setattr(r, "pcp_manager", object()),
        lambda r, mp: setattr(r, "speculative_config", object()),
        lambda r, mp: setattr(r, "most_model_len", 2048),
        lambda r, mp: setattr(r, "lora_config", object()),
        lambda r, mp: setattr(r, "enforce_eager", True),
        lambda r, mp: setattr(r.parallel_config, "pipeline_parallel_size", 2),
        lambda r, mp: setattr(r.parallel_config, "data_parallel_size", 2),
        lambda r, mp: mp.setenv("TPU_PRECOMPILE_TRACE_FIRST", "1"),
    ],
)
def test_runner_gate_fails_closed_on_unsupported_modes(monkeypatch, mutator):
    _fake_tpu_events(monkeypatch)
    runner = _make_runner(
        num_computed_tokens=[10],
        prompt_tokens=[10],
        scheduled_tokens=[1],
        token_paddings=[16, 32, 64],
        pcp_size=1,
        max_num_reqs=256,
    )
    sched_out = _scheduler_output([1])
    _attach_real_attn_and_mamba_groups(runner, max_num_reqs=256)
    runner.parallel_config.data_parallel_size = 1
    runner.parallel_config.nnodes = 1
    monkeypatch.setenv("TPU_METADATA_REQ_BUCKETING", "1")
    assert runner._metadata_req_bucketing_enabled() is True

    mutator(runner, monkeypatch)
    assert runner._metadata_req_bucketing_enabled() is False

    # And _prepare_inputs cleanly falls back to full num_reqs_max_model_len=256
    if (
        getattr(runner, "pcp_manager", None) is None
        and getattr(runner, "most_model_len", None) is None
        and getattr(runner, "lora_config", None) is None
        and getattr(runner, "speculative_config", None) is None
    ):
        out = TPUModelRunner._prepare_inputs(
            runner, sched_out, start_index=0, num_decode_reqs=1, num_windowed_reqs=1
        )
        assert tuple(out[0]["layers.3.attn"].seq_lens.shape) == (256,)
        assert runner._current_target_num_reqs is None
