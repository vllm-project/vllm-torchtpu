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
"""Tests for DFlash/DSpark sliding-window extra_retained_tokens patch."""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_torchtpu import (
    _patch_dflash_sliding_window_retained_tokens,
    patch_registry,
)


def _base_vllm_config_kwargs() -> dict[str, object]:
    cache_cfg = CacheConfig(enable_prefix_caching=True)
    cache_cfg.kv_cache_layout = "LBHNC"
    return {
        "cache_config": cache_cfg,
        "model_config": SimpleNamespace(
            max_model_len=2048,
            original_max_model_len=2048,
            hf_config=SimpleNamespace(),
            supports_kv_cache_grouping=False,
        ),
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        "scheduler_config": SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_seqs=4
        ),
        "attention_config": SimpleNamespace(hisparse_config=None),
        "speculative_config": None,
        "num_lookahead_tokens": 0,
        "num_prefill_lookahead_tokens": 0,
        "max_in_flight_tokens": 16,
    }


def _apply_only_swa_retained_patch(monkeypatch: pytest.MonkeyPatch):
    """Install only the SWA retained-tokens patch without leaking other patches."""
    target = "vllm_torchtpu:_patch_dflash_sliding_window_retained_tokens"
    matches = [p for p in patch_registry.PATCHES if p.target == target]
    assert len(matches) == 1
    patch_entry = matches[0]
    assert "engine_core" in patch_entry.stages
    assert "platform_activation" in patch_entry.stages
    assert "worker_init" in patch_entry.stages
    assert patch_entry.refresh == target

    orig_groups = inspect.unwrap(kv_cache_planning.get_kv_cache_groups)
    monkeypatch.setattr(kv_cache_planning, "get_kv_cache_groups", orig_groups)
    monkeypatch.setattr(patch_registry, "PATCHES", (patch_entry,))
    monkeypatch.setattr(
        patch_registry,
        "_applied",
        set(patch_registry._applied) - {target},  # noqa: SLF001 -- isolated test registry state
    )
    kv_mgr = sys.modules.get("vllm_torchtpu.runner.kv_cache_manager")
    if kv_mgr is not None:
        monkeypatch.setattr(kv_mgr, "get_kv_cache_groups", orig_groups)

    patch_registry.apply("engine_core")
    return orig_groups


def test_patch_is_idempotent_and_refreshes_by_name_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_kv_mgr = ModuleType("vllm_torchtpu.runner.kv_cache_manager")
    orig_before = inspect.unwrap(kv_cache_planning.get_kv_cache_groups)
    fake_kv_mgr.get_kv_cache_groups = orig_before
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.runner.kv_cache_manager", fake_kv_mgr
    )

    orig_groups = _apply_only_swa_retained_patch(monkeypatch)
    patched_groups = kv_cache_planning.get_kv_cache_groups
    assert patched_groups is not orig_groups
    assert fake_kv_mgr.get_kv_cache_groups is patched_groups

    # Direct second call must stay idempotent rather than stacking wrappers.
    _patch_dflash_sliding_window_retained_tokens()
    assert kv_cache_planning.get_kv_cache_groups is patched_groups

    # Simulate a late-imported module binding the unwrapped function before a
    # subsequent stage triggers patch_registry refresh.
    fake_kv_mgr.get_kv_cache_groups = orig_groups
    patch_registry.apply("worker_init")
    assert kv_cache_planning.get_kv_cache_groups is patched_groups
    assert fake_kv_mgr.get_kv_cache_groups is patched_groups


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_retains_swa_tokens_for_dflash_and_dspark_end_to_end(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_only_swa_retained_patch(monkeypatch)

    base_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    mla_swa = SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    high_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        extra_retained_tokens=10,
    )
    full_spec = FullAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    base_vcfg = _base_vllm_config_kwargs()

    # Non-speculative config leaves extra_retained_tokens unchanged.
    groups_plain = kv_cache_planning.get_kv_cache_groups(
        SimpleNamespace(**base_vcfg), {"attn.0": base_swa}
    )
    assert groups_plain[0].kv_cache_spec.extra_retained_tokens == 0

    # Zero lookahead leaves extra_retained_tokens unchanged.
    spec_cfg = SimpleNamespace(
        method=method,
        use_eagle=lambda: False,
        use_eagle_block_drop=lambda: False,
    )
    vcfg_zero_la = SimpleNamespace(
        **{**base_vcfg, "speculative_config": spec_cfg, "num_lookahead_tokens": 0}
    )
    groups_zero_la = kv_cache_planning.get_kv_cache_groups(
        vcfg_zero_la, {"attn.0": base_swa}
    )
    assert groups_zero_la[0].kv_cache_spec.extra_retained_tokens == 0

    # Active lookahead raises low SWA specs to lookahead + 1 while preserving
    # higher SWA retention and leaving FullAttentionSpec untouched.
    vcfg_spec = SimpleNamespace(
        **{**base_vcfg, "speculative_config": spec_cfg, "num_lookahead_tokens": 5}
    )
    groups_mixed = kv_cache_planning.get_kv_cache_groups(
        vcfg_spec,
        {"swa.0": base_swa, "swa.1": high_swa, "full.0": full_spec},
    )
    by_layer = {g.layer_names[0]: g.kv_cache_spec for g in groups_mixed}
    assert by_layer["swa.0"].extra_retained_tokens == 6
    assert by_layer["swa.1"].extra_retained_tokens == 10
    assert isinstance(by_layer["full.0"], FullAttentionSpec)

    # End-to-end: upstream get_global_kv_cache_groups resets extra_retained_tokens
    # to 0 before calling get_kv_cache_groups, yet the patched path retains 6 tokens.
    global_groups = kv_cache_planning.get_global_kv_cache_groups(
        vcfg_spec, [{"attn.0": mla_swa}]
    )
    configs = kv_cache_planning.get_kv_cache_configs(
        vcfg_spec, [{"attn.0": mla_swa}], [64 * 1024 * 1024], global_groups
    )
    retained_spec = configs[0].kv_cache_groups[0].kv_cache_spec
    assert retained_spec.extra_retained_tokens == 6

    mgr_stub = SimpleNamespace(
        sliding_window=retained_spec.sliding_window,
        extra_retained_tokens=retained_spec.extra_retained_tokens,
    )
    skipped = SlidingWindowManager.get_num_skipped_tokens(mgr_stub, 895)
    assert skipped == 762 and skipped // 32 == 23
