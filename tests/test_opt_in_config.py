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
}
# Dropped with no key: hid the offload connector from draft forwards, which
# v3's skip_draft_kv unbinding already covers in every served config.
DROPPED_ENV = ("TPU_DRAFT_HIDE_NOOP_KV_CONNECTOR",)


def test_every_key_replaces_an_env_var():
    assert set(DEFAULTS) == set(REPLACED_ENV)
    for name in (*REPLACED_ENV.values(), *DROPPED_ENV):
        assert name not in envs.environment_variables, name
        for path in SRC.rglob("*.py"):
            assert name not in path.read_text(), (name, path)


@pytest.mark.parametrize("key", sorted(DEFAULTS))
def test_defaults_match_the_old_env_defaults(key):
    want = {"minimax_split_min_tokens": 0, "minimax_split_attention_first": True}
    assert option(None, key) == want.get(key, False)
    assert option({}, key) == want.get(key, False)
    assert option({"other": 1}, key) == want.get(key, False)


@pytest.mark.parametrize("key", sorted(DEFAULTS))
def test_values_round_trip_through_json(key):
    """`--additional-config` is parsed JSON: true/false and ints."""
    value = 4096 if key == "minimax_split_min_tokens" else not DEFAULTS[key]
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
