# SPDX-License-Identifier: Apache-2.0
"""The (stage, group) mesh a pipeline stage's context-parallel kernels run
on, and the stage tag those kernels carry."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import pytest

from vllm_torchtpu.distributed import mesh_utils, sharded_jax_op

pytestmark = pytest.mark.cpu_test

# Four stages of two ranks on the binding rank -> device
# (0, 1, 4, 5, 6, 7, 2, 3): stage s holds ranks 2s and 2s + 1, whatever
# devices they sit on.
TABLE = mesh_utils.StageDeviceTable(
    stages=(0, 0, 1, 1, 2, 2, 3, 3),
    columns=(0, 1, 0, 1, 0, 1, 0, 1),
    device_ids=(0, 1, 4, 5, 6, 7, 2, 3),
    stage=3,
    column=1,
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_MESH_CACHE", {})
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", None)


def test_rows_are_stage_major_ranks_in_group_order():
    assert TABLE.rows() == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert TABLE.stage_device_ids(1) == (4, 5)
    assert TABLE.num_stages == 4 and TABLE.num_columns == 2


def _world(monkeypatch, ranks):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state, "get_world_group",
                        lambda: SimpleNamespace(ranks=list(ranks)))


def test_stage_groups_put_each_stage_on_a_rank_block(monkeypatch):
    _world(monkeypatch, (3, 1, 0, 2, 7, 5, 4, 6))
    assert mesh_utils.stage_group_ranks(4, 2) == {
        "pcp": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "pp": [[0, 2, 4, 6], [1, 3, 5, 7]],
        "ep": [[0, 1], [2, 3], [4, 5], [6, 7]],
    }
    assert mesh_utils.stage_group_ranks(2, 4)["pcp"] == [[0, 1, 2, 3],
                                                         [4, 5, 6, 7]]


def test_stage_groups_need_the_whole_rank_range(monkeypatch):
    _world(monkeypatch, (0, 1, 2, 3))
    with pytest.raises(RuntimeError, match="ranks 0..7"):
        mesh_utils.stage_group_ranks(2, 4)


def test_stage_mesh_rows_follow_the_table(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", TABLE)
    devices = [SimpleNamespace(id=i) for i in range(8)]
    monkeypatch.setattr(jax, "devices", Mock(return_value=devices))
    mesh = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)
    first = mesh_utils.get_or_create_cp_mesh("pcp", group=None)
    rows = [[d.id for d in row] for row in mesh.call_args.args[0]]
    assert rows == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert mesh.call_args.kwargs == {"axis_names": ("stage", "pcp")}
    assert mesh_utils.get_or_create_stage_mesh("pcp") is first
    assert mesh_utils.get_or_create_stage_mesh("d") is not first


def test_stage_mesh_rejects_stages_off_their_rank_blocks(monkeypatch):
    """Stages laid out on device blocks of the (0, 1, 4, 5, 6, 7, 2, 3)
    binding: a kernel's collective would name mesh positions the stage tag
    does not hold, so the layout is refused before any program is built."""
    table = mesh_utils.StageDeviceTable(stages=(0, 0, 2, 2, 3, 3, 1, 1),
                                        columns=(0, 1, 0, 1, 0, 1, 0, 1),
                                        device_ids=(0, 1, 4, 5, 6, 7, 2, 3),
                                        stage=1,
                                        column=0)
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", table)
    with pytest.raises(RuntimeError, match="consecutive ranks"):
        mesh_utils.get_or_create_stage_mesh("pcp")


def test_stage_tag_is_this_stage_in_group_order(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", TABLE)
    assert mesh_utils.stage_collective_devices() == [6, 7]
    mesh = SimpleNamespace(axis_names=("stage", "pcp"))
    assert sharded_jax_op.stage_tag_for_mesh(mesh) == [6, 7]
    assert sharded_jax_op.stage_tag_for_mesh(
        SimpleNamespace(axis_names=("pp", ))) is None


def test_no_tag_outside_pipeline_parallelism_without_reading_the_mesh(
        monkeypatch):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state, "get_pp_group",
                        lambda: SimpleNamespace(world_size=1))
    assert sharded_jax_op.stage_tag_for_mesh(Mock()) is None


def test_no_table_outside_pipeline_parallelism(monkeypatch):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state, "get_pp_group",
                        lambda: SimpleNamespace(world_size=1))
    assert mesh_utils.get_stage_device_table() is None
    assert mesh_utils.stage_collective_devices() is None


def test_an_ungathered_table_is_an_error(monkeypatch):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state, "get_pp_group",
                        lambda: SimpleNamespace(world_size=4))
    with pytest.raises(RuntimeError, match="not gathered"):
        mesh_utils.get_stage_device_table()


def test_tagged_callable_needs_a_torch_tpu_that_takes_the_tag():

    class Old:

        def __init__(self, name):
            self.kwargs = {"name": name}

    class New(Old):

        def __init__(self, collective_devices, name):
            super().__init__(name)
            self.collective_devices = collective_devices

    assert sharded_jax_op.make_jax_callable(Old,
                                            collective_devices=None,
                                            name="a").kwargs == {
                                                "name": "a"
                                            }
    made = sharded_jax_op.make_jax_callable(New,
                                            collective_devices=[2, 3],
                                            name="a")
    assert made.collective_devices == [2, 3]
    with pytest.raises(NotImplementedError, match="collective_devices"):
        sharded_jax_op.make_jax_callable(Old,
                                         collective_devices=[2, 3],
                                         name="a")


def test_stage_mesh_1d_under_4059_mesh_device_assignment(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", TABLE)
    monkeypatch.setattr(mesh_utils, "_uses_mesh_device_assignment",
                        lambda: True)
    monkeypatch.setattr(sharded_jax_op, "_uses_mesh_device_assignment",
                        lambda: True)
    devices = [SimpleNamespace(id=i) for i in range(8)]
    monkeypatch.setattr(jax, "devices", Mock(return_value=devices))
    mesh = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(axis_names=
                                                                    ("pcp", )))
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)

    first = mesh_utils.get_or_create_cp_mesh("pcp", group=None)
    dev_ids = [d.id for d in mesh.call_args.args[0]]
    # Stage 3 has world ranks (6, 7) -> physical PJRT device_ids (2, 3).
    assert dev_ids == [2, 3]
    assert mesh.call_args.kwargs == {"axis_names": ("pcp", )}
    assert mesh_utils.get_or_create_stage_mesh("pcp") is first


def test_stage_groups_with_tp_and_pcp_and_device_order(monkeypatch):
    _world(monkeypatch, range(8))
    device_ids = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 2, 7: 3}
    groups = mesh_utils.stage_group_ranks(2, 2, tp_size=2,
                                          device_ids=device_ids)
    # Stage 0 ranks [0, 1, 2, 3] (dev 0, 1, 4, 5); Stage 1 sorted by dev id
    # -> [6, 7, 4, 5] (dev 2, 3, 6, 7).
    assert groups == {
        "pcp": [[0, 1], [2, 3], [6, 7], [4, 5]],
        "tp": [[0, 2], [1, 3], [6, 4], [7, 5]],
        "pp": [[0, 6], [1, 7], [2, 4], [3, 5]],
        "ep": [[0, 1, 2, 3], [6, 7, 4, 5]],
    }


def test_stage_mesh_2d_with_tp_lanes_under_4059(monkeypatch):
    table_2x4 = mesh_utils.StageDeviceTable(
        stages=(0, 0, 0, 0, 1, 1, 1, 1),
        columns=(0, 1, 2, 3, 2, 3, 0, 1),
        device_ids=(0, 1, 4, 5, 6, 7, 2, 3),
        stage=1,
        column=0,
    )
    monkeypatch.setattr(mesh_utils, "_STAGE_TABLE", table_2x4)
    monkeypatch.setattr(mesh_utils, "_uses_mesh_device_assignment",
                        lambda: True)
    devices = [SimpleNamespace(id=i) for i in range(8)]
    monkeypatch.setattr(jax, "devices", Mock(return_value=devices))
    mesh = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(
        axis_names=("stage_tp", "pcp")))
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)

    group = SimpleNamespace(world_size=2)
    mesh_utils.get_or_create_cp_mesh("pcp", group=group)
    grid = [[d.id for d in row] for row in mesh.call_args.args[0]]
    assert grid == [[2, 3], [6, 7]]
    assert mesh.call_args.kwargs == {"axis_names": ("stage_tp", "pcp")}

