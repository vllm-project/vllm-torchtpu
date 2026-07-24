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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.worker import tpu_rank_binding as binding


@pytest.fixture(autouse=True)
def clear_pcp_remaps():
    binding.clear_pcp_local_rank_remaps()
    yield
    binding.clear_pcp_local_rank_remaps()


def _parallel_config(**kwargs):
    values = {
        "world_size": 8,
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "data_parallel_index": None,
        "prefill_context_parallel_size": 8,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_compute_pcp_local_rank_remap_matches_jax_device_axis_order():
    remap = binding.compute_pcp_local_rank_remap(
        local_rank_to_device_id=(0, 1, 4, 5, 6, 7, 2, 3),
        jax_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    )

    assert remap == (0, 1, 6, 7, 2, 3, 4, 5)


def test_compute_pcp_local_rank_remap_uses_runtime_order_not_sorted_order():
    remap = binding.compute_pcp_local_rank_remap(
        local_rank_to_device_id=(10, 20, 30, 40),
        jax_device_ids=(30, 10, 40, 20),
    )

    assert remap == (2, 0, 3, 1)


@pytest.mark.parametrize(
    ("local_rank_to_device_id", "jax_device_ids", "message"),
    [
        ((0, 1, 1), (0, 1, 2), "duplicate"),
        ((0, 1, 2), (0, 1, 1), "duplicate"),
        ((0, 1, 2), (0, 1, 3), "missing"),
    ],
)
def test_compute_pcp_local_rank_remap_rejects_invalid_probe_results(
        local_rank_to_device_id, jax_device_ids, message):
    with pytest.raises(RuntimeError, match=message):
        binding.compute_pcp_local_rank_remap(local_rank_to_device_id,
                                             jax_device_ids)


def test_set_pcp_local_rank_remap_rejects_invalid_permutation():
    with pytest.raises(ValueError):
        binding.set_pcp_local_rank_remap(4, (0, 1, 1, 3), source="test")


def test_ensure_pcp_local_rank_remap_registers_probe_result():
    with patch.object(binding,
                      "probe_pcp_local_rank_remap",
                      return_value=(0, 1, 6, 7, 2, 3, 4, 5)):
        binding.ensure_pcp_local_rank_remap(8,
                                            get_topology=lambda _world: "2,2")

    remap, source = binding.get_pcp_native_rank_local_rank_remap(8)
    assert remap == (0, 1, 6, 7, 2, 3, 4, 5)
    assert "dynamic probe" in source


def test_probe_pcp_local_rank_remap_timeout_reports_all_rank_logs(monkeypatch):

    class HungProbe:

        def __init__(self, _args, *, stdout, stderr, env):
            self.rank = int(env["RANK"])
            self.returncode = None
            stdout.write(f"hung rank {self.rank}\n")
            stdout.flush()

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            raise binding.subprocess.TimeoutExpired("probe", timeout)

    monkeypatch.setattr(binding, "_PCP_REMAP_PROBE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(binding.subprocess, "Popen", HungProbe)

    with pytest.raises(RuntimeError, match="timed out") as exc_info:
        binding.probe_pcp_local_rank_remap(2, get_topology=lambda _world: "2")

    message = str(exc_info.value)
    assert "hung rank 0" in message
    assert "hung rank 1" in message


def test_probe_pcp_local_rank_remap_failure_reports_all_rank_logs(monkeypatch):

    class FailedProbe:

        def __init__(self, _args, *, stdout, stderr, env):
            self.rank = int(env["RANK"])
            self.returncode = 1
            stdout.write(f"failed rank {self.rank}\n")
            stdout.flush()

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(binding.subprocess, "Popen", FailedProbe)

    with pytest.raises(RuntimeError, match="probe failed") as exc_info:
        binding.probe_pcp_local_rank_remap(2, get_topology=lambda _world: "2")

    message = str(exc_info.value)
    assert "failed rank 0" in message
    assert "failed rank 1" in message


def test_registered_pcp_local_rank_remap_drives_worker_spawn_binding():
    binding.set_pcp_local_rank_remap(8, (0, 1, 6, 7, 2, 3, 4, 5),
                                     source="test probe")

    b = binding.get_tpu_worker_binding(_parallel_config(),
                                       rank=2,
                                       local_rank=2,
                                       env={})

    assert b.as_env() == {
        "RANK": "2",
        "LOCAL_RANK": "6",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    assert b.init_local_rank == 6
    assert b.native_local_rank == 2
    assert b.pcp_local_rank_remap == (0, 1, 6, 7, 2, 3, 4, 5)
    assert b.pcp_remap_source == "test probe"


def test_parent_spawn_binding_lazily_probes_missing_pcp_remap():
    with patch.object(binding,
                      "probe_pcp_local_rank_remap",
                      return_value=(0, 1, 6, 7, 2, 3, 4, 5)) as mock_probe:
        b = binding.get_tpu_worker_binding(_parallel_config(),
                                           rank=2,
                                           local_rank=2,
                                           env={})

    mock_probe.assert_called_once()
    assert b.as_env() == {
        "RANK": "2",
        "LOCAL_RANK": "6",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    assert b.pcp_local_rank_remap == (0, 1, 6, 7, 2, 3, 4, 5)
    assert "dynamic probe" in b.pcp_remap_source


def test_spawned_worker_binding_uses_inherited_local_rank_without_probe():
    b = binding.get_tpu_worker_binding(
        _parallel_config(),
        rank=2,
        local_rank=2,
        env={
            "LOCAL_RANK": "6",
            "LOCAL_WORLD_SIZE": "8",
        },
        use_spawned_pcp_local_rank=True,
    )

    assert b.as_env() == {
        "RANK": "2",
        "LOCAL_RANK": "6",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    assert b.init_local_rank == 6
    assert b.pcp_local_rank_remap is None
    assert b.pcp_remap_source == "spawn LOCAL_RANK env"


def test_pcp_disabled_does_not_remap_without_probe():
    pc = _parallel_config(prefill_context_parallel_size=1)

    b = binding.get_tpu_worker_binding(pc, rank=6, local_rank=6, env={})

    assert b.as_env()["LOCAL_RANK"] == "6"
    assert b.init_local_rank == 6
    assert b.pcp_local_rank_remap is None


def test_dp_binding_uses_registered_probe_result_and_applies_offset():
    binding.set_pcp_local_rank_remap(8, tuple(range(8)), source="test probe")
    env = {
        "TORCH_TPU_DP_SIZE": "2",
        "TPU_LOCAL_RANK_OFFSET": "1",
    }
    pc = _parallel_config(world_size=4,
                          data_parallel_size=1,
                          data_parallel_rank=0,
                          data_parallel_index=1,
                          prefill_context_parallel_size=4)

    b = binding.get_tpu_worker_binding(pc, rank=2, local_rank=2, env=env)

    assert b.dp_rank == 1
    assert b.as_env() == {
        "RANK": "6",
        "LOCAL_RANK": "7",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "9",
    }
    assert b.init_rank == 2
    assert b.init_world_size == 4
    assert b.init_local_rank == 6


def test_dp_binding_rejects_unresolved_data_parallel_index():
    # data_parallel_index is resolved by ParallelConfig.__post_init__; an
    # unresolved index under DP must fail loudly instead of being defaulted.
    pc = _parallel_config(data_parallel_size=2, data_parallel_index=None)

    with pytest.raises(AssertionError, match="data_parallel_index"):
        binding.get_tpu_worker_binding(pc, rank=0, local_rank=0, env={})
