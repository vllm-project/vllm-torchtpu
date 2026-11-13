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
"""The r6 opt-in variants read `--additional-config` keys
(`vllm_torchtpu.opt_in_config`), not env vars: each key sets what its env
var used to set, and an absent key keeps the default path."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.opt_in_config import DEFAULTS, option

SRC = Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu"

# key -> the env var it replaces
REPLACED_ENV = {
    "minimax_split_min_tokens": "TPU_MINIMAX_M3_SPLIT_MIN_TOKENS",
    "minimax_split_attention_first": "TPU_MINIMAX_M3_SPLIT_ATTENTION_FIRST",
    "minimax_dcp_heads_attend": "TPU_MINIMAX_M3_DCP_HEADS_ATTEND",
    "minimax_msa_unstable_sort": "TPU_MINIMAX_M3_MSA_UNSTABLE_SORT",
    "minimax_dcp_fine_scan_widths": "TPU_MINIMAX_M3_DCP_FINE_SCAN_WIDTHS",
    "minimax_msa_onehot_lookup": "TPU_MINIMAX_M3_MSA_ONEHOT_LOOKUP",
    "minimax_indexer_dma_ring3": "TPU_MINIMAX_M3_INDEXER_DMA_RING3",
    "minimax_indexer_fp8_transposed": "TPU_MINIMAX_M3_INDEXER_FP8_TRANSPOSED",
    "minimax_msa_lse_r6": "TPU_MINIMAX_M3_MSA_LSE_R6",
    "minimax_dspark_split_fc": "TPU_MINIMAX_M3_DSPARK_SPLIT_FC",
    "raiden_host_order_dma": "TPU_RAIDEN_HOST_ORDER_DMA",
    "moe_gmm_fp8_decode_tiles": "TPU_GMM_FP8_DECODE_TILES",
    "spec_synthetic_one_row_verify": "TPU_SPEC_SYNTHETIC_ONE_ROW_VERIFY",
    "spec_warmup_sweep_token_buckets": "TPU_SPEC_WARMUP_SWEEP_TOKEN_BUCKETS",
    "minimax_indexer_large_pages_per_step": "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP",
    "gmm_v2_prefill_tiles": "TPU_GMM_V2_PREFILL_TILES",
    "gmm_v2_decode_w13": "TPU_GMM_V2_DECODE_W13",
    "minimax_msa_prefill_width12": "TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12",
    "minimax_msa_prefill_widths": "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS",
    "minimax_msa_prefill_page_cache": "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE",
    "minimax_msa_prefill_page_cache_tile_pages": (
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES"
    ),
    "minimax_msa_prefill_page_cache_bf16": "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16",
    "minimax_msa_prefill_row_split": "TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT",
    "minimax_msa_prefill_block_q": "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q",
    "minimax_msa_prefill_pages_per_step": "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP",
    "minimax_msa_decode_work_list_kernel": "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL",
    "minimax_msa_decode_one_call": "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL",
    "moe_tc_combine_min_tokens": "TPU_MOE_TC_COMBINE_MIN_TOKENS",
    "moe_v3_payload_sort": "TPU_MOE_V3_PAYLOAD_SORT",
    "moe_flat_routing_min_tokens": "TPU_MOE_FLAT_ROUTING_MIN_TOKENS",
    "tight_vmem": "TPU_TIGHT_VMEM",
    "tight_vmem_candidates_mib": "TPU_TIGHT_VMEM_CANDIDATES_MIB",
}
# Served values of the value-bearing keys (the rest are booleans).
SAMPLE = {
    "minimax_split_min_tokens": 4096,
    "minimax_indexer_large_pages_per_step": 64,
    "gmm_v2_prefill_tiles": "256:128/256@12288",
    "gmm_v2_decode_w13": "3072:2:2",
    "minimax_msa_prefill_widths": "4/8/12",
    "minimax_msa_prefill_page_cache": 512,
    "minimax_msa_prefill_page_cache_tile_pages": 64,
    "minimax_msa_prefill_row_split": 2,
    "minimax_msa_prefill_block_q": 8,
    "minimax_msa_prefill_pages_per_step": 8,
    "moe_tc_combine_min_tokens": 4096,
    "moe_flat_routing_min_tokens": 1024,
    "tight_vmem_candidates_mib": "32,40,48",
}
# Keys added after the env-var migration: they never had an env var.
NEW_KEYS = {
    "raiden_offload_refresh_lru",
    "minimax_indexer_hoist_metadata",
    "minimax_indexer_row_gather",
}
# Dropped with no key: hid the offload connector from draft forwards, which
# v3's skip_draft_kv unbinding already covers in every served config.
DROPPED_ENV = ("TPU_DRAFT_HIDE_NOOP_KV_CONNECTOR",)


def test_every_key_replaces_an_env_var():
    assert set(DEFAULTS) == set(REPLACED_ENV) | NEW_KEYS
    for name in (*REPLACED_ENV.values(), *DROPPED_ENV):
        assert name not in envs.environment_variables, name
        for path in SRC.rglob("*.py"):
            assert name not in path.read_text(), (name, path)


@pytest.mark.parametrize("key", sorted(DEFAULTS))
def test_defaults_match_the_old_env_defaults(key):
    want = {"minimax_split_attention_first": True}
    for config in (None, {}, {"other": 1}):
        assert option(config, key) == DEFAULTS[key]
    # Every key but this one is off by default (False, 0 or "").
    assert DEFAULTS[key] == want.get(key, type(DEFAULTS[key])())


@pytest.mark.parametrize("key", sorted(DEFAULTS))
def test_values_round_trip_through_json(key):
    """`--additional-config` is parsed JSON: true/false, ints and strings."""
    value = SAMPLE.get(key, not DEFAULTS[key])
    config = json.loads(json.dumps({key: value}))
    assert option(config, key) == value


def test_wrong_type_is_an_error():
    with pytest.raises(ValueError, match="minimax_msa_lse_r6"):
        option({"minimax_msa_lse_r6": 1}, "minimax_msa_lse_r6")
    with pytest.raises(ValueError, match="minimax_split_min_tokens"):
        option({"minimax_split_min_tokens": True}, "minimax_split_min_tokens")


def _toggles():
    from vllm_torchtpu.models.minimax_m3 import kernel_toggles
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    return kernel_toggles, [
        *kernel_toggles.TOGGLES,
        ("minimax_dcp_heads_attend", msa_ops, "DCP_HEADS_ATTEND"),
    ]


def test_indexer_large_pages_per_step_key():
    from vllm_torchtpu.models.minimax_m3 import kernel_toggles
    from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_indexer

    key = "minimax_indexer_large_pages_per_step"
    try:
        kernel_toggles.apply({key: 64})
        assert msa_indexer.LARGE_PAGES_PER_STEP == 64
        assert msa.block_score_options(1)["pages_per_step"] == 64
        for bad in (-32, 48):
            with pytest.raises(ValueError, match=key):
                kernel_toggles.apply({key: bad})
    finally:
        kernel_toggles.apply({})
    assert msa_indexer.LARGE_PAGES_PER_STEP == msa_indexer.DEFAULT_PAGES_PER_STEP == 32


def test_r6_prefill_and_decode_keys():
    from vllm_torchtpu.models.minimax_m3 import kernel_toggles
    from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse_r6 as r6

    try:
        kernel_toggles.apply(
            {
                "minimax_msa_lse_r6": True,
                "minimax_msa_prefill_page_cache": 512,
                "minimax_msa_prefill_page_cache_tile_pages": 64,
                "minimax_msa_prefill_widths": "4/8/12",
                "minimax_msa_decode_one_call": True,
            }
        )
        assert r6.PREFILL_PAGE_CACHE_SLOTS == 512
        assert r6.PAGE_CACHE_TILE_PAGES == 64
        assert r6.PREFILL_STEP_WIDTHS == (4, 8, 12)
        assert r6.DECODE_ONE_CALL is True
        assert r6.prefill_options(True) == {
            "step_widths": (4, 8, 12),
            "page_cache_slots": 512,
        }
        kernel_toggles.apply({"minimax_msa_prefill_width12": True})
        assert r6.PREFILL_STEP_WIDTHS == (4, 8, 12)
        kernel_toggles.apply({"minimax_msa_prefill_widths": "2,4,6,8,12"})
        assert r6.PREFILL_STEP_WIDTHS == (2, 4, 6, 8, 12)
        with pytest.raises(ValueError, match="minimax_msa_prefill_page_cache"):
            kernel_toggles.apply({"minimax_msa_prefill_page_cache": -1})
    finally:
        kernel_toggles.apply({})
    assert r6.prefill_options(True) == {}
    assert (r6.PREFILL_ROW_SPLIT, r6.PAGE_CACHE_TILE_PAGES) == (1, 32)
    assert r6.DECODE_ONE_CALL is False and r6.DECODE_WORK_LIST_KERNEL is False


def test_kernel_toggles_set_the_module_flags_their_env_vars_set():
    kernel_toggles, toggles = _toggles()
    try:
        for key, module, attr in toggles:
            kernel_toggles.apply({key: True})
            for other_key, other_module, other_attr in toggles:
                assert getattr(other_module, other_attr) is (other_key == key), (
                    key,
                    other_key,
                )
        kernel_toggles.apply({key: True for key, _, _ in toggles})
        assert all(getattr(m, a) is True for _, m, a in toggles)
    finally:
        kernel_toggles.apply({})
    assert all(getattr(m, a) is False for _, m, a in toggles)


def test_dspark_split_fc_key_sets_the_flag():
    from vllm_torchtpu.models.minimax_m3 import dspark

    try:
        dspark.set_split_fc({"minimax_dspark_split_fc": True})
        assert dspark.SPLIT_FC is True
    finally:
        dspark.set_split_fc({})
    assert dspark.SPLIT_FC is False


def test_raiden_manager_host_order_dma():
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager

    assert make_manager(FakeKVCacheStore())._host_order_dma is False
    assert make_manager(FakeKVCacheStore(), host_order_dma=True)._host_order_dma


def test_raiden_manager_refresh_lru():
    from tests.offload.test_raiden_store_offload import FakeKVCacheStore, make_manager

    assert make_manager(FakeKVCacheStore()).refresh_lru is False
    assert make_manager(FakeKVCacheStore(), refresh_lru=True).refresh_lru


def test_synthetic_one_row_verify_follows_the_runner_attribute():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    def runner(on):
        return SimpleNamespace(
            _synthetic_one_row_verify=on,
            rejection_sampler=SimpleNamespace(synthetic_mode=True),
            input_batch=SimpleNamespace(max_num_logprobs=None, req_id_to_index={}),
            _min_tokens=SimpleNamespace(any_in=lambda _: False),
        )

    step = SimpleNamespace(has_structured_output_requests=False)
    check = TPUModelRunner._synthetic_verify_this_step
    assert check(runner(True), step) is True
    assert not check(runner(False), step)


def test_pallas_all_reduce_ordered_op_is_opt_in():
    from vllm_torchtpu.distributed import pallas_all_reduce

    assert inspect.signature(pallas_all_reduce.arm).parameters["ordered"].default is False


@pytest.mark.parametrize(
    "key,path",
    [
        ("minimax_split_min_tokens", "models/minimax_m3/model.py"),
        ("minimax_split_attention_first", "models/minimax_m3/model.py"),
        ("minimax_split_min_tokens", "runner/tpu_runner.py"),
        ("spec_synthetic_one_row_verify", "runner/tpu_runner.py"),
        ("spec_warmup_sweep_token_buckets", "runner/tpu_runner.py"),
        ("moe_gmm_fp8_decode_tiles", "runner/tpu_runner.py"),
        ("raiden_host_order_dma", "offload/raiden_store.py"),
        ("raiden_offload_refresh_lru", "offload/raiden_store.py"),
        ("minimax_dspark_split_fc", "models/minimax_m3/dspark.py"),
    ],
)
def test_runtime_keys_are_read_where_the_env_var_was(key, path):
    """Readers that need a built engine: the key is read through `option` at
    the site that read the env var."""
    text = (SRC / path).read_text()
    assert f'"{key}"' in text and "option(" in text


def test_split_guard_passes_with_split_keys_off():
    from vllm_torchtpu.opt_in_config import check_split_disabled

    check_split_disabled(None)
    check_split_disabled({})
    check_split_disabled({"minimax_split_min_tokens": 0, "moe_gmm_tuned_tiles": True})


@pytest.mark.parametrize(
    "config",
    [
        {"minimax_split_min_tokens": 4096},
        {"minimax_split_attention_first": True},
        {"minimax_split_attention_first": False},
        {"minimax_split_min_tokens": 4096, "minimax_split_attention_first": False},
    ],
)
def test_split_guard_refuses_split_keys(config):
    from vllm_torchtpu.opt_in_config import check_split_disabled

    with pytest.raises(ValueError, match="leak data across requests"):
        check_split_disabled(config)


def test_tight_vmem_keys():
    from vllm_torchtpu.kernels import tight_vmem

    assert tight_vmem.ENABLED is False
    try:
        tight_vmem.set_options(True, "48,32")
        assert tight_vmem.ENABLED is True
        assert tight_vmem._CANDIDATES == (32 << 20, 48 << 20)
        with pytest.raises(ValueError, match="tight_vmem_candidates_mib"):
            tight_vmem.set_options(True, "0")
    finally:
        tight_vmem.set_options(False)
    assert tight_vmem._CANDIDATES == tuple(
        v << 20 for v in tight_vmem.DEFAULT_CANDIDATES_MIB
    )
