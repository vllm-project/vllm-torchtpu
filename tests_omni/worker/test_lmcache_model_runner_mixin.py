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
"""TPU test for KV Cache Offload (LMCache)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import _resolve_worker_cls

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

_CHUNK_TOKENS = 16
_HIDDEN_DIM = 8


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason=(
        "vllm_omni.worker.lmcache_model_runner_mixin is not in vllm-omni 0.32.0rc1 "
        "(added in vllm-omni PR #1330), and the lmcache package is not installed"
    ),
)
def test_kv_cache_offload_lmcache(tmp_path: Path) -> None:
    """KV Cache Offload (LMCache)

    - TPU path: import ``LMCacheHiddenStateMixin`` and ``lmcache``, configure
      ``omni_kv_config.kv_store_config.lmcache_config`` with
      ``enable_prefix_caching=False``, and store/restore ``"tpu"`` hidden states.
    - Setup check: ``LMCacheConnectorV1`` is configured, TPU AR runner subclasses
      ``LMCacheHiddenStateMixin``, and ``_setup_lmcache_hidden_state_offload()`` sets
      ``runner._has_lmcache is True``.
    - Outcome check: restored hidden states for matching prefix tokens match the
      original ``"tpu"`` prefix hidden states copied to CPU.
    """
    import lmcache  # noqa: F401
    from vllm_omni.worker.lmcache_model_runner_mixin import LMCacheHiddenStateMixin

    cfg_file = tmp_path / "lmcache.yaml"
    cfg_file.write_text("chunk_size: 16\nlocal_cpu: true\nmax_local_cpu_size: 0.1\n")

    worker_cls = _resolve_worker_cls(current_omni_platform.get_omni_ar_worker_cls())
    assert issubclass(worker_cls.model_runner_cls, LMCacheHiddenStateMixin), (
        f"{worker_cls.__name__}.model_runner_cls does not subclass "
        "LMCacheHiddenStateMixin"
    )

    runner = object.__new__(worker_cls.model_runner_cls)
    runner.model_config = SimpleNamespace(
        omni_kv_config={
            "kv_store_config": {
                "lmcache_config": {
                    "config_file": str(cfg_file),
                    "max_local_cpu_size": 0.1,
                }
            }
        }
    )
    runner.cache_config = SimpleNamespace(enable_prefix_caching=False)
    runner.model = SimpleNamespace(talker_config=None)
    runner._setup_lmcache_hidden_state_offload()
    assert runner._has_lmcache is True, "runner._has_lmcache was not enabled"

    torch.manual_seed(0)
    prefix_cpu = torch.randn(_CHUNK_TOKENS, _HIDDEN_DIM, dtype=torch.float32)
    prefix_tpu = prefix_cpu.to(device=torch.device("tpu"))
    runner.input_batch = SimpleNamespace(
        req_ids=["req-1", "req-2"],
        token_ids_cpu=torch.arange(_CHUNK_TOKENS * 2, dtype=torch.int32).view(2, -1),
    )
    runner._maybe_store_hs_to_lmcache("req-1", {"hidden": prefix_tpu})
    runner._maybe_restore_hs_from_lmcache("req-2")
    restored = runner._take_restored_mm("req-2")
    assert restored is not None, "LMCache did not restore hidden states"
    assert_parity(
        restored["hidden"],
        prefix_cpu,
        name="lmcache_restored_hidden",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
