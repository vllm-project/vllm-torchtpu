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
"""Which calls the small-collective bridge hands to a kernel.

The kernels themselves are tested on TPU in
tests/kernels/experimental/small_collectives; the warmup hand-over between the
two sides of the row fork in tests/compilation/test_shape_variants.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives import common as sc_common
from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

pytestmark = pytest.mark.cpu_test

# The device-info reader itself, before prebuild_layout replaces it.
DEVICES_PER_CHIP = sc_common.devices_per_chip

CORES = 8
MESH = tuple(range(CORES))
HIDDEN = 4096


@pytest.fixture
def armed(monkeypatch):
    """Both ops replaced by mocks, inside a warmup trace."""
    ar, rs = Mock(name="all_reduce_op"), Mock(name="reduce_scatter_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    return ar, rs


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize("rows", [1, 8, 64, 96, 128])
def test_small_block_takes_the_all_reduce_kernel(armed, rows):
    ar, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, _fallback) is ar.return_value
    (block, rank), _ = ar.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize("rows", [1, 16, 40])
def test_small_combine_takes_the_reduce_scatter_kernel(armed, rows):
    _, rs = armed
    x = _block(rows * CORES)
    assert op.dp_combine_or(x, _fallback) is rs.return_value
    (block, rank), _ = rs.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize(
    "call,rows",
    [
        (lambda x: op.all_reduce_or(x, MESH, _fallback), op.MAX_ROWS + 1),
        (lambda x: op.dp_combine_or(x, _fallback), CORES * op.COMBINE_MAX_ROWS + CORES),
    ],
    ids=["all_reduce", "dp_combine"],
)
def test_large_block_falls_back_and_reads_the_rank(armed, call, rows):
    ar, rs = armed
    x = _block(rows)
    out = call(x)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    ar.assert_not_called()
    rs.assert_not_called()


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(40, True), (41, False)])
def test_combine_cap_is_40_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, rs = armed
    op.dp_combine_or(_block(rows_per_engine * CORES), _fallback)
    assert rs.called is armed_expected


@pytest.mark.parametrize("rows,armed_expected", [(128, True), (129, False)])
def test_all_reduce_cap_is_128_rows(armed, rows, armed_expected):
    ar, _ = armed
    op.all_reduce_or(_block(rows), MESH, _fallback)
    assert ar.called is armed_expected


def test_large_block_fallback_depends_on_the_rank_buffer(armed, monkeypatch):
    """The retain is a live read: a NaN rank buffer poisons the result, which a
    read the compiler could drop would not."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    out = op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, _fallback)
    assert torch.isnan(out).all()


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), MESH),  # dtype
        (_block(8, hidden=HIDDEN + 128), MESH),  # width does not split over cores
        (_block(8).reshape(2, 4, HIDDEN), MESH),  # not 2-D
        (_block(8), (0, 1, 2, 3)),  # another group than the mesh
    ],
    ids=["fp32", "width", "3d", "group"],
)
def test_ineligible_all_reduce_falls_back_untouched(armed, x, ranks):
    ar, _ = armed
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    fallback.assert_called_once_with(x)
    ar.assert_not_called()


@pytest.mark.parametrize("outside", ["not_compiling", "no_warmup"])
def test_traces_outside_the_backbone_warmup_keep_the_collective(
    armed, monkeypatch, outside
):
    """The MTP drafter is compiled outside the warmup, which could not fork it."""
    ar, rs = armed
    if outside == "not_compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    ar.assert_not_called()
    rs.assert_not_called()


def test_a_forced_region_takes_the_all_reduce_kernel_outside_the_warmup(
    armed, monkeypatch
):
    """The bounded MTP loop graph forces the kernel: any row count, no warmup
    fork. The combine is never forced, and leaving the region restores the
    collective."""
    ar, rs = armed
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    x = _block(2 * op.MAX_ROWS)
    fallback = Mock(return_value=object())
    with op.force_small_all_reduce():
        assert op.all_reduce_or(x, MESH, fallback) is ar.return_value
        assert op.dp_combine_or(_block(8 * CORES), fallback) is fallback.return_value
        with op.force_small_all_reduce(enabled=False):
            assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
        assert op.all_reduce_or(x, MESH, fallback) is ar.return_value
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    rs.assert_not_called()


def test_combine_needs_the_dp_group_to_be_the_mesh(armed, monkeypatch):
    _, rs = armed
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    x = _block(8 * CORES)
    fallback = Mock(return_value=object())
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    rs.assert_not_called()


def test_unarmed_is_the_fallback(monkeypatch):
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value


def _config(additional_config):
    return SimpleNamespace(additional_config=additional_config)


ON = _config({"tpu_collectives": {"small_all_reduce": True}})


@pytest.mark.parametrize(
    "additional_config",
    [
        {},
        {"tpu_collectives": {}},
        {"tpu_collectives": {"small_all_reduce": False}},
        {"other": {"small_all_reduce": True}},
    ],
    ids=["empty", "no_key", "off", "other_subsystem"],
)
def test_prebuild_is_a_no_op_unless_the_setting_is_on(monkeypatch, additional_config):
    """Off by default: main's collectives stay untouched for every model."""
    build = Mock()
    monkeypatch.setattr(op, "build_ep_mesh", build)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild(_config(additional_config))
    build.assert_not_called()
    assert op._AR_OP is None


def test_prebuild_refuses_a_deployment_without_an_ep_group(monkeypatch):
    monkeypatch.setattr(op, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(NotImplementedError, match="small_all_reduce"):
        op.prebuild(ON)


def test_prebuild_builds_the_ops_when_the_setting_is_on(monkeypatch):
    """The setting reaches the mesh build (the 'on' control for the no-op test)."""
    monkeypatch.setattr(op, "build_ep_mesh", Mock(side_effect=RuntimeError("built")))
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(RuntimeError, match="built"):
        op.prebuild(ON)


@pytest.fixture
def prebuild_on_a_host(monkeypatch):
    """prebuild over a stand-in 8-core EP mesh whose DP and TP groups are the
    mesh; returns the block order the reduce-scatter op was built with."""
    from vllm.distributed import parallel_state

    group = SimpleNamespace(ranks=list(MESH))
    built = []
    monkeypatch.setattr(
        op, "build_ep_mesh", lambda: SimpleNamespace(shape={op.EP_AXIS_NAME: CORES})
    )
    monkeypatch.setattr(op, "get_ep_group", lambda: group)
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 2)
    monkeypatch.setattr(op, "_rank_tensor", lambda i: torch.tensor([i]))
    monkeypatch.setattr(op, "_build_all_reduce", lambda mesh: Mock())
    monkeypatch.setattr(
        op, "_build_reduce_scatter", lambda mesh, order: built.append(order) or Mock()
    )
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    for name, value in [
        ("_AR_OP", None),
        ("_RS_OP", None),
        ("_RANK", None),
        ("_CORES", 0),
        ("_MESH_RANKS", ()),
        ("_DP_COMBINE", False),
    ]:
        monkeypatch.setattr(op, name, value)
    return built


@pytest.mark.parametrize(
    "mesh_order,block_order",
    [
        # A TPU7x host's mesh in device-id order (the default): local ranks
        # 0..7 sit on devices [0, 1, 4, 5, 6, 7, 2, 3].
        ((0, 1, 6, 7, 2, 3, 4, 5), (0, 1, 6, 7, 2, 3, 4, 5)),
        # A rank-ordered mesh (PCP, DCP or the sorted gather on).
        (MESH, None),
    ],
)
def test_the_combine_sends_each_dp_ranks_rows_to_its_mesh_core(
    prebuild_on_a_host, monkeypatch, mesh_order, block_order
):
    """The combine's block r is DP rank r's rows, and the mesh core at index i
    serves EP (and DP) rank mesh_order[i], so core i must get block
    mesh_order[i]; block r to core r hands engines each other's rows."""
    monkeypatch.setattr(op, "ep_rank_order", lambda: mesh_order)
    op.prebuild(ON)
    assert prebuild_on_a_host == [block_order]
    assert op._DP_COMBINE is True
    assert op._RANK.tolist() == [2]


def test_retain_is_the_identity_when_unarmed_or_outside_the_warmup(monkeypatch):
    x = _block(8)
    monkeypatch.setattr(op, "_RANK", None)
    assert op.retain(x) is x
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.retain(x) is x


# The TP-pair all-reduce: a TP group of two cores inside the mesh (DP x TP2).

PAIR = (2, 3)


@pytest.fixture
def pair_armed(armed, monkeypatch):
    """The mesh ops and the TP-pair op as mocks; this core is TP rank 1."""
    tp_op = Mock(name="tp_all_reduce_op")
    monkeypatch.setattr(op, "_AR_TP_OP", tp_op)
    monkeypatch.setattr(op, "_TP_RANKS", PAIR)
    monkeypatch.setattr(op, "_TP_RANK", torch.tensor([1], dtype=torch.int32))
    return tp_op


@pytest.mark.parametrize("rows", [1, 8, 64, 128])
def test_the_tp_pair_takes_the_tp_pair_kernel(armed, pair_armed, rows):
    ar, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, PAIR, _fallback) is pair_armed.return_value
    (block, rank), _ = pair_armed.call_args
    assert block is x and rank is op._TP_RANK
    ar.assert_not_called()


def test_the_mesh_group_still_takes_the_mesh_kernel(armed, pair_armed):
    ar, _ = armed
    assert op.all_reduce_or(_block(8), MESH, _fallback) is ar.return_value
    pair_armed.assert_not_called()


def test_a_large_tp_pair_block_falls_back_and_reads_the_pair_rank(
    armed, pair_armed, monkeypatch
):
    x = _block(op.MAX_ROWS + 1)
    torch.testing.assert_close(
        op.all_reduce_or(x, PAIR, _fallback), _fallback(x), rtol=0, atol=0
    )
    pair_armed.assert_not_called()
    monkeypatch.setattr(op, "_TP_RANK", torch.tensor([float("nan")]))
    assert torch.isnan(op.all_reduce_or(x, PAIR, _fallback)).all()


def test_a_forced_region_takes_the_tp_pair_kernel_above_the_cap(armed, pair_armed):
    x = _block(op.MAX_ROWS * 2)
    with op.force_small_all_reduce():
        assert op.all_reduce_or(x, PAIR, _fallback) is pair_armed.return_value


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), PAIR),
        (_block(8, hidden=HIDDEN + 128), PAIR),
        (_block(8), (4, 5)),
    ],
    ids=["fp32", "width", "other_pair"],
)
def test_ineligible_tp_pair_calls_fall_back(armed, pair_armed, x, ranks):
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    pair_armed.assert_not_called()


@pytest.fixture
def prebuild_layout(monkeypatch):
    """prebuild's collaborators on an 8-core EP mesh, with the kernels'
    builders and the pairs check (a collective) as mocks. Call the returned
    function with the TP group's ranks; the DP group is the cores with the
    same TP index."""
    import vllm.distributed.parallel_state as ps

    builders = SimpleNamespace(
        ar=Mock(name="_build_all_reduce"),
        rs=Mock(name="_build_reduce_scatter"),
        tp=Mock(name="_build_tp_all_reduce"),
        pairs=Mock(name="groups_match_mesh", return_value=True),
    )
    mesh = SimpleNamespace(shape={op.EP_AXIS_NAME: CORES})
    monkeypatch.setattr(op, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(op, "get_ep_group", lambda: SimpleNamespace(ranks=MESH))
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 3)
    monkeypatch.setattr(op, "_build_all_reduce", builders.ar)
    monkeypatch.setattr(op, "_build_reduce_scatter", builders.rs)
    monkeypatch.setattr(op, "_build_tp_all_reduce", builders.tp)
    monkeypatch.setattr(op, "groups_match_mesh", builders.pairs)
    # TPU7x: each device is one of a chip's two cores.
    monkeypatch.setattr(sc_common, "devices_per_chip", lambda: 2)
    for name, value in (
        ("_AR_OP", None),
        ("_RS_OP", None),
        ("_RANK", None),
        ("_AR_TP_OP", None),
        ("_TP_RANK", None),
        ("_TP_RANKS", ()),
        ("_DP_COMBINE", False),
    ):
        monkeypatch.setattr(op, name, value)
    host_tensor = torch.tensor

    def tensor(*args, device=None, **kwargs):
        return host_tensor(*args, **kwargs)

    monkeypatch.setattr(op.torch, "tensor", tensor)

    def layout(tp_ranks):
        tp = len(tp_ranks)
        dp_ranks = tuple(r for r in MESH if r % tp == 3 % tp)
        monkeypatch.setattr(
            ps,
            "get_tp_group",
            lambda: SimpleNamespace(ranks=tp_ranks, world_size=tp),
        )
        monkeypatch.setattr(
            ps,
            "get_dp_group",
            lambda: SimpleNamespace(ranks=dp_ranks, world_size=len(dp_ranks)),
        )
        return builders, mesh

    return layout


def test_prebuild_arms_the_tp_pair_on_dp4_tp2(prebuild_layout):
    builders, mesh = prebuild_layout((2, 3))
    op.prebuild(ON)
    assert op._AR_TP_OP is builders.tp.return_value
    assert op._TP_RANKS == (2, 3)
    assert op._TP_RANK.tolist() == [1]  # mesh index 3 is TP rank 1
    builders.tp.assert_called_once_with(mesh, 2)
    builders.pairs.assert_called_once_with(3, 2)


@pytest.mark.parametrize(
    "tp_ranks", [MESH, (0, 1, 2, 3)], ids=["tp8_is_the_mesh", "tp4"]
)
def test_prebuild_leaves_other_tp_groups_to_the_mesh_kernel_or_xla(
    prebuild_layout, tp_ranks
):
    """TP8 (the mesh) takes the mesh kernel; TP4 was never measured with the
    TP-pair kernel, so it keeps XLA's all-reduce. Neither runs the pairs
    collective."""
    builders, _ = prebuild_layout(tp_ranks)
    op.prebuild(ON)
    assert op._AR_OP is builders.ar.return_value
    assert op._AR_TP_OP is None
    builders.tp.assert_not_called()
    builders.pairs.assert_not_called()


def test_prebuild_refuses_tp_pairs_that_are_not_mesh_blocks(prebuild_layout):
    builders, _ = prebuild_layout((2, 3))
    builders.pairs.return_value = False
    op.prebuild(ON)
    assert op._AR_TP_OP is None
    builders.tp.assert_not_called()


def test_prebuild_without_the_setting_builds_no_tp_pair(prebuild_layout):
    builders, _ = prebuild_layout((2, 3))
    op.prebuild(_config({}))
    assert op._AR_TP_OP is None
    builders.pairs.assert_not_called()


def _device_info(monkeypatch, chip, cores_per_device=1):
    """The real devices_per_chip over a faked pltpu.get_tpu_info, and the
    module logger as a mock."""
    info = pltpu.get_tpu_info_for_chip(chip, cores_per_device)
    monkeypatch.setattr(sc_common, "devices_per_chip", DEVICES_PER_CHIP)
    monkeypatch.setattr(sc_common.pltpu, "get_tpu_info", lambda: info)
    logger = Mock(name="logger")
    monkeypatch.setattr(op, "logger", logger)
    return logger


def test_prebuild_builds_no_tp_pair_across_chips_and_says_so(
    prebuild_layout, monkeypatch
):
    """A TP2 group is one chip's cores only where a chip holds two devices.
    With one device per chip (the device info's report) the measured DP x TP2
    layout keeps XLA's all-reduce, runs no collective, and warns once with the
    reported value."""
    builders, _ = prebuild_layout((2, 3))
    logger = _device_info(monkeypatch, pltpu.ChipVersion.TPU_V6E)
    op.prebuild(ON)
    assert op._AR_TP_OP is None
    builders.tp.assert_not_called()
    builders.pairs.assert_not_called()
    logger.warning_once.assert_called_once()
    message, what, reported, tp, _ = logger.warning_once.call_args.args
    assert "device(s) per chip" in message
    assert (what, reported, tp) == ("The TP-pair small all-reduce", 1, 2)


def test_prebuild_arms_the_tp_pair_on_tpu7x_device_info_without_a_warning(
    prebuild_layout, monkeypatch
):
    builders, mesh = prebuild_layout((2, 3))
    logger = _device_info(monkeypatch, pltpu.ChipVersion.TPU_7X)
    op.prebuild(ON)
    assert op._AR_TP_OP is builders.tp.return_value
    builders.tp.assert_called_once_with(mesh, 2)
    logger.warning_once.assert_not_called()


def test_an_unmeasured_tp_size_off_the_chip_is_not_warned_about(
    prebuild_layout, monkeypatch
):
    """TP4 was never served by the TP-pair kernel, so a chip that does not
    hold four devices is no news."""
    builders, _ = prebuild_layout((0, 1, 2, 3))
    logger = _device_info(monkeypatch, pltpu.ChipVersion.TPU_7X)
    op.prebuild(ON)
    assert op._AR_TP_OP is None
    logger.warning_once.assert_not_called()


@pytest.mark.parametrize(
    "chip,cores_per_device,want",
    [
        (pltpu.ChipVersion.TPU_7X, 1, 2),
        (pltpu.ChipVersion.TPU_V6E, 1, 1),
        (pltpu.ChipVersion.TPU_V5P, 1, 2),
        (pltpu.ChipVersion.TPU_V5P, 2, 1),
    ],
    ids=["tpu7x", "v6e", "v5p_split", "v5p_megacore"],
)
def test_devices_per_chip_comes_from_the_device_info(
    monkeypatch, chip, cores_per_device, want
):
    info = pltpu.get_tpu_info_for_chip(chip, cores_per_device)
    monkeypatch.setattr(sc_common.pltpu, "get_tpu_info", lambda: info)
    assert sc_common.devices_per_chip() == want


def test_tp_pair_form_is_oneshot():
    assert op.TP_PAIR_FORM == "oneshot"


# The layout predicate behind groups_match_mesh, on an 8-core mesh whose mesh
# index i is rank ORDER[i].

ORDER = (0, 1, 2, 3, 4, 5, 6, 7)


@pytest.mark.parametrize(
    "mesh_index,tp_ranks,dp_ranks,want",
    [
        (3, (2, 3), None, True),
        (3, (2, 3), (1, 3, 5, 7), True),
        (0, (0, 1), (0, 2, 4, 6), True),
        (3, (3, 4), None, False),
        (3, (1, 3), None, False),
        (3, (2, 3), (0, 2, 4, 6), False),
        (3, (2, 3), (1, 3, 5), False),
        (3, (2, 3, 4), None, False),
    ],
    ids=[
        "tp_pair_block",
        "tp_pair_and_dp_column",
        "first_core",
        "pair_straddles_blocks",
        "pair_not_consecutive",
        "dp_group_other_tp_index",
        "dp_group_short",
        "tp_group_not_tp",
    ],
)
def test_groups_on_mesh_blocks(mesh_index, tp_ranks, dp_ranks, want):
    got = op._groups_on_mesh_blocks(
        ORDER, mesh_index, tp_ranks, 2, dp_ranks, 4 if dp_ranks is not None else 0
    )
    assert got is want


def test_groups_on_mesh_blocks_follows_the_rank_order():
    """Mesh index, not rank: with ranks 2 and 3 swapped on the mesh, the pair
    at mesh block 2..3 is still ranks {2, 3}, and DP column 1 holds rank 2."""
    order = (0, 1, 3, 2, 4, 5, 6, 7)
    # Mesh index 3 is rank 2: TP index 1 of mesh block 2..3.
    assert op._groups_on_mesh_blocks(order, 3, (2, 3), 2, (1, 2, 5, 7), 4) is True
    assert op._groups_on_mesh_blocks(order, 3, (2, 3), 2, (1, 3, 5, 7), 4) is False


def test_groups_on_mesh_blocks_without_a_rank_order():
    assert op._groups_on_mesh_blocks(None, 3, (2, 3), 2) is False
