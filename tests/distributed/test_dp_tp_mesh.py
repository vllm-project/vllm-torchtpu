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

import numpy as np
import pytest
import torch.tpu.pallas as pallas_mod

from vllm_torchtpu.compilation.tpu_compiler import _RUNTIME_CACHE_KEY_PATHS
from vllm_torchtpu.distributed import dp_tp_mesh


def _groups(rank: int, dp: int, tp: int) -> tuple[list[int], list[int]]:
    d, t = divmod(rank, tp)
    return [x * tp + t for x in range(dp)], list(range(d * tp, (d + 1) * tp))


def _fake_2d_mesh(ids, shape, axis_names=("dp", "tp")) -> SimpleNamespace:
    devs = np.array([SimpleNamespace(id=d) for d in ids], dtype=object).reshape(shape)
    return SimpleNamespace(axis_names=axis_names, devices=devs)


def _patch_dist_and_groups(
    monkeypatch: pytest.MonkeyPatch,
    *,
    global_rank: int,
    world_size: int,
    dp_ranks: list[int],
    tp_ranks: list[int],
) -> None:
    monkeypatch.setattr(dp_tp_mesh, "_MESH_2D_CACHE", {})
    monkeypatch.setattr(dp_tp_mesh.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_rank", lambda: global_rank)
    monkeypatch.setattr(dp_tp_mesh.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        dp_tp_mesh.parallel_state, "model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        dp_tp_mesh.parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(ranks=dp_ranks),
    )
    monkeypatch.setattr(
        dp_tp_mesh.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=tp_ranks),
    )


@pytest.mark.parametrize(("dp", "tp"), [(2, 4), (1, 8), (8, 1), (8, 16)])
def test_validate_accepts_rank_ordered_grid(
    monkeypatch: pytest.MonkeyPatch, dp: int, tp: int
) -> None:
    world = dp * tp
    ranks = {0, 5 % world, world // 2, world - 1}
    if world == 128:
        ranks.add(37)
    for rank in sorted(ranks):
        dp_ranks, tp_ranks = _groups(rank, dp, tp)
        _patch_dist_and_groups(
            monkeypatch,
            global_rank=rank,
            world_size=world,
            dp_ranks=dp_ranks,
            tp_ranks=tp_ranks,
        )
        assert dp_tp_mesh._validate_dp_tp_group_ranks(
            dp, tp, dp_ranks, tp_ranks
        ) == divmod(rank, tp)


@pytest.mark.parametrize(
    ("dp_ranks", "tp_ranks"),
    [
        ([4, 5], [4, 5, 6, 7]),  # dp column scrambled
        ([1, 5], [5, 4, 6, 7]),  # tp row permuted
        ([1, 5], [4, 5, 6]),  # tp row short
    ],
)
def test_build_rejects_mismatched_groups(
    monkeypatch: pytest.MonkeyPatch,
    dp_ranks: list[int],
    tp_ranks: list[int],
) -> None:
    _patch_dist_and_groups(
        monkeypatch,
        global_rank=5,
        world_size=8,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )
    with pytest.raises(ValueError):
        dp_tp_mesh.build_dp_tp_mesh()


def test_build_rejects_world_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    dp_ranks, tp_ranks = _groups(5, 2, 4)
    _patch_dist_and_groups(
        monkeypatch,
        global_rank=5,
        world_size=16,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )
    with pytest.raises(ValueError, match="world_size == dp_size \\* tp_size"):
        dp_tp_mesh.build_dp_tp_mesh()


def test_build_requires_initialized_parallel_state_and_dp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dp_tp_mesh, "_MESH_2D_CACHE", {})
    monkeypatch.setattr(
        dp_tp_mesh.parallel_state, "model_parallel_is_initialized", lambda: False
    )
    with pytest.raises(RuntimeError, match="parallel_state is not initialized"):
        dp_tp_mesh.build_dp_tp_mesh()

    monkeypatch.setattr(
        dp_tp_mesh.parallel_state, "model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        dp_tp_mesh.parallel_state,
        "get_dp_group",
        lambda: (_ for _ in ()).throw(AssertionError("dp group is not initialized")),
    )
    with pytest.raises(RuntimeError, match="DP/TP process groups are not initialized"):
        dp_tp_mesh.build_dp_tp_mesh()


def test_build_dp_tp_mesh_validates_groups_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_ranks, tp_ranks = _groups(1, 2, 2)
    _patch_dist_and_groups(
        monkeypatch,
        global_rank=1,
        world_size=4,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )
    rank_ids = (40, 20, 10, 30)
    calls = []

    def fake_get_pallas_mesh(*, axis_names, mesh_shape):
        calls.append((axis_names, mesh_shape))
        return _fake_2d_mesh(rank_ids, mesh_shape, axis_names)

    monkeypatch.setattr(pallas_mod, "get_pallas_mesh", fake_get_pallas_mesh)

    mesh = dp_tp_mesh.build_dp_tp_mesh()
    assert calls == [(("dp", "tp"), (2, 2))]
    assert dp_tp_mesh.build_dp_tp_mesh() is mesh
    assert len(calls) == 1


def test_build_dp_tp_mesh_inside_compile_requires_cached_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_ranks, tp_ranks = _groups(0, 2, 2)
    _patch_dist_and_groups(
        monkeypatch,
        global_rank=0,
        world_size=4,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )
    monkeypatch.setattr(dp_tp_mesh.torch.compiler, "is_compiling", lambda: True)

    with pytest.raises(RuntimeError, match="before torch.compile tracing"):
        dp_tp_mesh.build_dp_tp_mesh()
    cached = _fake_2d_mesh((0, 1, 2, 3), (2, 2))
    dp_tp_mesh._MESH_2D_CACHE[(2, 2)] = cached
    assert dp_tp_mesh.build_dp_tp_mesh() is cached


def test_dp_tp_mesh_is_covered_by_runtime_cache_key_paths() -> None:
    assert "vllm_torchtpu/distributed/dp_tp_mesh.py" in _RUNTIME_CACHE_KEY_PATHS
