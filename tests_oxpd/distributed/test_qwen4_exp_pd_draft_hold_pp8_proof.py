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
"""P-PIPE-3's CPU proof additions for D-MK's F2-HOLD (8c273bfc4 / int-6 port a7e34c5b1, projection fix 8b2cef5c1) on the
PP8 tree 74938cd66: the hold engages on a minimal producer plan, the producer's send path takes a held table whose leading
entries are the null block (page 0) without refusing it, and the flag changes no compile-env factor (host-only, measured).
Assertion messages start with the names the CDK CPU proof's mutants are matched against (HOLD-ENGAGE, HOLD-SUFFIX,
HOLD-SEND, HOLD-KEY)."""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig

BS = 16


@pytest.fixture
def hold_on(monkeypatch):
    from vllm_torchtpu.core import producer_draft_hold as pdh

    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    registered = pdh.register_producer_draft_hold()
    yield registered
    pdh.unregister_producer_draft_hold()


def _manager():
    spec = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    groups = [KVCacheGroupSpec(["model.layers.3.self_attn"], spec),
              KVCacheGroupSpec(["mtp.layers.48.self_attn"], qconfig.non_prefix_cacheable(spec))]
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)
    m = KVCacheManager(KVCacheConfig(num_blocks=400, kv_cache_tensors=[], kv_cache_groups=groups), max_model_len=8192,
                       scheduler_block_size=BS, hash_block_size=BS, enable_caching=True, use_eagle=False,
                       dcp_world_size=1, pcp_world_size=1)
    return m, get_request_block_hasher(BS, fn)


def _hit(m, hasher, prefix=48, total=96):
    rnd = random.Random(5)
    a = [rnd.randrange(1, 50000) for _ in range(total)]
    ra = Request("a", a, SamplingParams(max_tokens=1), None, block_hasher=hasher)
    blocks, hit, *_ = m.get_computed_blocks(ra)
    assert m.allocate_slots(ra, total, hit, blocks) is not None
    ra.num_computed_tokens = total
    m.free(ra)
    rb = Request("b", a[:prefix] + [rnd.randrange(1, 50000) for _ in range(total - prefix)], SamplingParams(max_tokens=1),
                 None, block_hasher=hasher)
    blocks, hit, *_ = m.get_computed_blocks(rb)
    assert m.allocate_slots(rb, total - hit, hit, blocks) is not None
    return hit, [list(g) for g in m.get_blocks("b").get_block_ids()]


def test_the_hold_engages_on_the_pp8_tree(hold_on):
    assert hold_on, "HOLD-ENGAGE: the hold must register with QWEN4_EXP_PD_DRAFT_HOLD=1 and QWEN4_EXP_PD_DRAFT_KV=1"
    m, hasher = _manager()
    hit, ids = _hit(m, hasher)
    assert hit == 48
    draft = ids[1]
    assert draft[: hit // BS] == [0] * (hit // BS), f"HOLD-ENGAGE: the draft's hit blocks must be the null block, got {draft}"
    assert all(b != 0 for b in draft[hit // BS:]) and len(draft) == 96 // BS, f"HOLD-SUFFIX: computed span real, got {draft}"
    assert all(b != 0 for b in ids[0]), f"HOLD-SUFFIX: target blocks are real, got {ids[0]}"


def test_the_send_path_takes_a_held_table_with_null_heads(monkeypatch):
    """The page-0 case: a held draft group whose table opens with the null block (page 0) is published with its
    leading nulls stripped (held_heads / handoff_block_ids heads), never refused as a null page or an overlap."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pdc
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import TransferExtent, TransferGroup

    groups = (TransferGroup(0, ("model.layers.3.self_attn",), BS, TransferExtent.TOKEN_PAGES),
              TransferGroup(1, ("mtp.layers.48.self_attn",), BS, TransferExtent.TOKEN_PAGES))
    pools = [SimpleNamespace(layer_name="model.layers.3.self_attn", tag="t3", num_blocks=100, storage_index=0),
             SimpleNamespace(layer_name="mtp.layers.48.self_attn", tag="t48", num_blocks=100, storage_index=1)]
    plan = pdc.QwenPagePlan(groups=groups, manifest=SimpleNamespace(pools=pools), held=frozenset({1}))
    try:
        pages, heads = plan.request_pages_and_heads([[11, 12, 13, 14], [0, 0, 0, 21]], 4 * BS)
    except ValueError as e:
        pytest.fail(f"HOLD-SEND: a held table with null heads must publish, raised {e}")
    assert pages == ((11, 12, 13, 14), (21,)) and heads == (0, 3), f"HOLD-SEND: pages {pages} heads {heads}"
    with pytest.raises(ValueError):
        plan.request_pages_and_heads([[11, 0, 13, 14], [0, 0, 0, 21]], 4 * BS)   # a null in an unheld group still refused


def test_the_flag_changes_no_compile_env_factor(monkeypatch):
    """Host-only, measured: the TorchTPU compile-env factors (the env part of compute_tpu_compilation_hash) are
    identical with the hold on and off; a control flag that does enter the key changes them."""
    from vllm_torchtpu.compilation import tpu_compiler as tc

    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_KV", "1")
    monkeypatch.delenv("QWEN4_EXP_PD_DRAFT_HOLD", raising=False)
    off = tc._tpu_compile_env_factors()
    monkeypatch.setenv("QWEN4_EXP_PD_DRAFT_HOLD", "1")
    on = tc._tpu_compile_env_factors()
    diff = sorted(k for k in set(on) | set(off) if on.get(k) != off.get(k))
    assert on == off, f"HOLD-KEY: the hold flag must not change the compile key (differing: {diff})"
    keyed = next(n for n in tc.envs.environment_variables if n not in tc._TPU_COMPILE_ENV_IGNORED
                 and n.startswith("VLLM_TPU_") and isinstance(tc.envs.environment_variables[n](), bool))
    monkeypatch.setenv(keyed, "0" if tc.envs.environment_variables[keyed]() else "1")
    assert tc._tpu_compile_env_factors() != off, f"HOLD-KEY: control flag {keyed} must change the factors"
