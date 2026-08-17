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
"""Tests for rank -> TPU device binding.

The property under test is that ``LOCAL_RANK`` is passed through untouched.
It selects the physical chip -- torch_tpu copies it into
``TPU_VISIBLE_CHIPS`` -- so permuting it silently moves a rank onto a
different chip. These tests keep topology concerns out of this layer: ring
ordering belongs to the PCP group's rank order, which builds the mesh axis the
ring kernel walks.
"""

from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig

from vllm_torchtpu.worker import tpu_rank_binding as binding


def _parallel_config(**kwargs):
    # Cannot use real ParallelConfig instance directly because world_size is
    # derived (pp*tp*pcp), not independently settable, and several tests here
    # deliberately hold it independent from prefill_context_parallel_size to
    # test get_tpu_worker_binding's own remap math across combinations the real
    # class wouldn't itself construct.
    values = vars(ParallelConfig(tensor_parallel_size=8)).copy()
    values["prefill_context_parallel_size"] = 8
    values.update(kwargs)
    # Single-host tests: local chip count equals the full TP world size
    # unless a test explicitly overrides it (e.g. to simulate --nnodes > 1).
    values.setdefault("local_world_size", values["world_size"])
    return SimpleNamespace(**values)


@pytest.mark.parametrize("rank", range(8))
def test_local_rank_is_identity_under_pcp(rank):
    """LOCAL_RANK must equal the native local rank for every rank.

    Regression guard for the removed remap: any permutation here rebinds ranks
    onto different chips. The whole range is checked rather than one rank,
    because a permutation can leave some points fixed.
    """
    b = binding.get_tpu_worker_binding(_parallel_config(),
                                       rank=rank,
                                       local_rank=rank,
                                       env={})

    assert b.as_env()["LOCAL_RANK"] == str(rank)
    assert b.init_local_rank == rank
    assert b.native_local_rank == rank


def test_binding_does_not_depend_on_pcp_size():
    """PCP size must not influence chip binding.

    Data sharding is keyed on pcp_rank; device binding is not. Making the two
    covary is exactly the coupling this change removed.
    """
    bindings = {}
    for pcp in (1, 2, 4, 8):
        bindings[pcp] = binding.get_tpu_worker_binding(
            _parallel_config(prefill_context_parallel_size=pcp),
            rank=6,
            local_rank=6,
            env={}).as_env()

    assert len({tuple(sorted(b.items())) for b in bindings.values()}) == 1
    assert bindings[8]["LOCAL_RANK"] == "6"


def test_spawned_worker_binding_uses_inherited_local_rank():
    """A spawned PCP worker adopts the LOCAL_RANK its parent handed it."""
    b = binding.get_tpu_worker_binding(
        _parallel_config(enable_expert_parallel=True),
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


def test_spawned_worker_binding_rejects_out_of_range_local_rank():
    with pytest.raises(ValueError, match="out of range"):
        binding.get_tpu_worker_binding(
            _parallel_config(),
            rank=0,
            local_rank=0,
            env={"LOCAL_RANK": "99"},
            use_spawned_pcp_local_rank=True,
        )


def test_dp_binding_applies_offset_without_remapping():
    """Under DP the rank shifts by the DP block and the chip offset.

    Both are positional bookkeeping, not topology: local_rank stays
    dp_rank*world + local_rank, plus TPU_LOCAL_RANK_OFFSET.
    """
    env = {
        "TORCH_TPU_DP_SIZE": "2",
        "TPU_LOCAL_RANK_OFFSET": "1",
    }
    pc = _parallel_config(
        world_size=4,
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_index=1,
        # Single-host DP: run_engine_core sets this equal to
        # the global DP rank, since every replica is local.
        data_parallel_rank_local=1,
        prefill_context_parallel_size=4,
        enable_expert_parallel=True)

    b = binding.get_tpu_worker_binding(pc, rank=2, local_rank=2, env=env)

    assert b.dp_rank == 1
    assert b.as_env() == {
        "RANK": "6",
        "LOCAL_RANK": "7",  # offset 1 + native 6
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "9",
    }
    assert b.init_rank == 2
    assert b.init_world_size == 4
    assert b.init_local_rank == 6
    assert b.native_local_rank == 6


@pytest.mark.skip(reason="Pending #409 submission")
def test_dense_dp_binding_keeps_native_local_rank_dp_aware_but_stays_local():
    # Dense (non-EP) DP replicas have no cross-rank collective, so each
    # bootstraps its own, replica-local torch_tpu slice: rank/world_size
    # must stay local to the replica (unlike the flattened EP case above).
    # But native_local_rank -- the physical chip-selection ordinal
    # TPUWorker uses for TPU_VISIBLE_CHIPS/TPU_VISIBLE_DEVICES -- must still
    # be offset by dp_rank, or two independent replicas' same-tp-rank
    # workers resolve to the same physical chip and collide. Regression
    # test: this exact bug was introduced twice by well-intentioned
    # simplifications of get_tpu_worker_binding.
    bindings = {}
    for dp_rank in (0, 1):
        pc = _parallel_config(world_size=2,
                              data_parallel_size=2,
                              data_parallel_index=dp_rank,
                              enable_expert_parallel=False,
                              prefill_context_parallel_size=1)
        for tp_rank in (0, 1):
            bindings[(dp_rank, tp_rank)] = binding.get_tpu_worker_binding(
                pc, rank=tp_rank, local_rank=tp_rank, env={})

    native_local_ranks = [b.native_local_rank for b in bindings.values()]
    assert len(set(native_local_ranks)) == 4, (
        f"expected 4 distinct chip-selection ordinals, got {bindings}")
    assert bindings[(0, 0)].native_local_rank == 0
    assert bindings[(0, 1)].native_local_rank == 1
    assert bindings[(1, 0)].native_local_rank == 2
    assert bindings[(1, 1)].native_local_rank == 3

    for (dp_rank, tp_rank), b in bindings.items():
        assert b.dp_rank == dp_rank
        assert b.rank == tp_rank
        assert b.world_size == 2, "non-EP DP must stay replica-local"


def test_dp_binding_rejects_unresolved_data_parallel_index():
    # data_parallel_index is resolved by ParallelConfig.__post_init__; an
    # unresolved index under DP must fail loudly instead of being defaulted.
    pc = _parallel_config(data_parallel_size=2,
                          data_parallel_index=None,
                          enable_expert_parallel=True)

    with pytest.raises(AssertionError, match="data_parallel_index"):
        binding.get_tpu_worker_binding(pc, rank=0, local_rank=0, env={})


def test_no_remap_surface_remains():
    """The remap API is gone, not merely unused.

    A dormant remap left importable invites a caller to resurrect the
    device-binding-layer fix this change removed.
    """
    for name in (
            "compute_pcp_local_rank_remap",
            "set_pcp_local_rank_remap",
            "ensure_pcp_local_rank_remap",
            "probe_pcp_local_rank_remap",
            "get_pcp_worker_local_rank_env",
    ):
        assert not hasattr(binding, name), f"{name} should have been removed"


def test_executor_slice_binding_overrides_single_host_dp_layout():
    # Multihost DP: the slice-global rank of a worker is not
    # dp_rank * world_size + rank, because the DP shards are spread across
    # hosts. The executor hands the worker its real placement instead.
    env = {
        "TORCH_TPU_DP_SIZE":
        "4",
        **binding.slice_binding_env(rank=9,
                                    local_rank=1,
                                    world_size=16,
                                    local_world_size=8),
    }
    pc = _parallel_config(world_size=4,
                          data_parallel_size=4,
                          data_parallel_index=2,
                          prefill_context_parallel_size=1,
                          enable_expert_parallel=True)

    b = binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)

    assert b.as_env() == {
        "RANK": "9",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
    }
    # vLLM's own distributed init stays per-engine.
    assert b.init_rank == 1
    assert b.init_world_size == 4
    assert b.init_local_rank == 1
    assert b.dp_rank == 2
    assert b.dp_size == 4


def test_executor_slice_binding_applies_local_rank_offset():
    env = {
        "TORCH_TPU_DP_SIZE":
        "2",
        "TPU_LOCAL_RANK_OFFSET":
        "2",
        **binding.slice_binding_env(rank=3,
                                    local_rank=1,
                                    world_size=4,
                                    local_world_size=2),
    }
    pc = _parallel_config(world_size=2,
                          data_parallel_size=2,
                          data_parallel_index=1,
                          prefill_context_parallel_size=1,
                          enable_expert_parallel=True)

    b = binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)

    assert b.local_rank == 3
    assert b.local_world_size == 4
    assert b.local_rank_offset == 2


def test_executor_slice_binding_ignored_without_dp():
    env = binding.slice_binding_env(rank=9,
                                    local_rank=1,
                                    world_size=16,
                                    local_world_size=8)
    pc = _parallel_config(world_size=4,
                          data_parallel_size=1,
                          prefill_context_parallel_size=1,
                          enable_expert_parallel=True)

    b = binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)

    assert b.rank == 1
    assert b.world_size == 4


def test_partial_executor_slice_binding_is_rejected():
    env = {
        "TORCH_TPU_DP_SIZE": "2",
        binding.SLICE_RANK_ENV: "3",
        binding.SLICE_WORLD_SIZE_ENV: "4",
    }
    pc = _parallel_config(world_size=2,
                          data_parallel_size=2,
                          data_parallel_index=1,
                          prefill_context_parallel_size=1,
                          enable_expert_parallel=True)

    with pytest.raises(ValueError, match="incomplete"):
        binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({
            "world_size": 8
        }, "does not match"),
        ({
            "rank": 4
        }, "out of range"),
        ({
            "local_rank": 2
        }, "out of range"),
    ],
)
def test_executor_slice_binding_rejects_inconsistent_geometry(
        overrides, message):
    values = {
        "rank": 3,
        "local_rank": 1,
        "world_size": 4,
        "local_world_size": 2,
    }
    values.update(overrides)
    env = {"TORCH_TPU_DP_SIZE": "2", **binding.slice_binding_env(**values)}
    pc = _parallel_config(world_size=2,
                          data_parallel_size=2,
                          data_parallel_index=1,
                          prefill_context_parallel_size=1,
                          enable_expert_parallel=True)

    with pytest.raises(ValueError, match=message):
        binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)


def test_executor_slice_binding_rejects_pcp():
    env = {
        "TORCH_TPU_DP_SIZE":
        "2",
        **binding.slice_binding_env(rank=3,
                                    local_rank=1,
                                    world_size=8,
                                    local_world_size=4),
    }
    pc = _parallel_config(world_size=4,
                          data_parallel_size=2,
                          data_parallel_index=1,
                          prefill_context_parallel_size=2,
                          enable_expert_parallel=True)

    with pytest.raises(NotImplementedError, match="context parallelism"):
        binding.get_tpu_worker_binding(pc, rank=1, local_rank=1, env=env)
