# SPDX-License-Identifier: Apache-2.0
"""CPU tests for TPU_MOE_EXPERT_PLACEMENT (expert map at load, ids after
routing) and TPU_MOE_ROUTING_COUNTS_DIR (routing counts and their file)."""

import time
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import expert_placement, moe_routing, token_padding
from vllm_torchtpu.layers.core import expert_placement as core

pytestmark = pytest.mark.cpu_test

E, EP, NAME = 64, 8, "model.layers.3.mlp.experts"


@pytest.fixture
def slots(monkeypatch):
    counts = np.random.default_rng(0).gamma(2.0, 10.0, size=E)
    slots = core.balance(counts, EP)
    monkeypatch.setattr(expert_placement, "_PLACEMENT", {NAME: slots})
    monkeypatch.setattr(expert_placement, "_EP_SIZE", EP)
    return slots


def _layer(ep_rank, name=NAME, strategy="linear", eplb=False):
    em = pytest.importorskip("vllm.model_executor.layers.fused_moe.expert_map_manager")
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu import _patch_expert_map_host_lookup

    _patch_expert_map_host_lookup()
    parallel = SimpleNamespace(
        use_ep=True,
        ep_size=EP,
        ep_rank=ep_rank,
        tp_size=1,
        tp_rank=0,
        dp_size=EP,
        dp_rank=0,
        pcp_size=1,
        enable_eplb=eplb,
        use_all2all_kernels=False,
        needs_round_robin_routing_tables=False,
    )
    manager = em.ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=2,
        global_num_experts=E,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy=strategy,
        enable_eplb=False,
    )
    layer = torch.nn.Module()
    layer.layer_name = name
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel)
    layer.expert_map_manager = manager
    layer.global_num_experts = E
    layer.update_expert_map_info = types.MethodType(
        RoutedExperts.update_expert_map_info, layer
    )
    layer.update_expert_map_info()
    return layer


def test_expert_map_sends_each_expert_to_its_slot(slots):
    local = E // EP
    owner = {}
    for rank in range(EP):
        layer = _layer(rank)
        manager = layer.expert_map_manager
        before = [manager.map_global_to_local(e) for e in range(E)]
        assert before == [
            e - rank * local if e // local == rank else -1 for e in range(E)
        ]
        expert_placement.place_experts(layer)
        np.testing.assert_array_equal(expert_placement.host_slots(layer), slots)
        after = [manager.map_global_to_local(e) for e in range(E)]
        assert layer._expert_map.tolist() == after
        for e, index in enumerate(after):
            if index >= 0:
                assert slots[e] == rank * local + index
                owner[e] = rank
    assert sorted(owner) == list(range(E))
    np.testing.assert_array_equal([owner[e] for e in range(E)], slots // local)


def test_layers_outside_the_file_keep_linear_placement(slots):
    layer = _layer(0, name="mtp.layers.0.mlp.experts")
    expert_placement.place_experts(layer)
    assert expert_placement.host_slots(layer) is None
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    assert layer._expert_slots is None
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    assert expert_placement.to_slots(layer, ids) is ids


def test_unplaced_named_layer_and_bad_layouts_raise(slots, monkeypatch):
    layer = _layer(0)
    with pytest.raises(NotImplementedError, match="NVFP4"):
        expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="EPLB"):
        expert_placement.place_experts(_layer(0, eplb=True))
    monkeypatch.setattr(expert_placement, "_EP_SIZE", 4)
    with pytest.raises(ValueError, match="EP 4"):
        expert_placement.place_experts(_layer(0))


def test_to_slots_maps_ids_and_keeps_negative_ones(slots):
    layer = _layer(2)
    expert_placement.place_experts(layer)
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    ids = torch.randint(0, E, (12, 10), dtype=torch.int32)
    ids[0, 3] = -1
    got = expert_placement.to_slots(layer, ids)
    assert got.dtype == torch.int32
    want = np.where(ids.numpy() >= 0, slots[ids.numpy().clip(0)], -1)
    np.testing.assert_array_equal(got.numpy(), want)


def test_local_experts_without_redundant_slots_are_the_primaries(slots):
    local = E // EP
    for rank in (0, 6):
        layer = _with_expert_tensors(_layer(rank), local)
        expert_placement.place_experts(layer)
        assert expert_placement.redundant_slots_per_rank(layer) == 0
        assert layer.local_num_experts == local
        np.testing.assert_array_equal(
            expert_placement.local_experts(layer),
            np.argsort(slots)[rank * local : (rank + 1) * local],
        )
        expert_placement.zero_empty_redundant(layer)
        assert all(float(p.min()) == 1.0 for p in layer.parameters(recurse=False))
        expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
        np.testing.assert_array_equal(layer._expert_slots.numpy(), slots)
        assert layer._expert_replicas is None
        expert_placement.check_replicas(layer)  # nothing to check, no collective


@pytest.fixture
def redundant(slots, monkeypatch):
    """One redundant slot per rank: rank r replicates the first expert of rank
    r + 1; rank 3's redundant slot is empty."""
    local = E // EP
    logical = np.argsort(slots)
    redundant = logical[((np.arange(EP) + 1) % EP) * local].astype(np.int32)
    redundant[3] = -1
    core.validate(slots[None], [NAME], EP, redundant[None])
    monkeypatch.setattr(expert_placement, "_REDUNDANT", {NAME: redundant})
    monkeypatch.setattr(expert_placement, "_REDUNDANT_PER_RANK", 1)
    monkeypatch.setattr(envs, "TPU_MOE_RAGGED_A2A", True, raising=False)
    return redundant


def _with_expert_tensors(layer, slots_per_rank):
    """Prepared-weight stand-ins: one row of each per-expert tensor per slot."""
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        param = torch.nn.Parameter(
            torch.ones(slots_per_rank, 4, 2), requires_grad=False
        )
        layer.register_parameter(name, param)
    return layer


def test_expert_map_puts_replicas_after_the_primaries(slots, redundant):
    local = E // EP
    for rank in range(EP):
        layer = _layer(rank)
        manager = layer.expert_map_manager
        assert expert_placement.redundant_slots_per_rank(layer) == 1
        expert_placement.place_experts(layer)
        assert manager.local_num_experts == layer.local_num_experts == local + 1
        after = [manager.map_global_to_local(e) for e in range(E)]
        assert layer._expert_map.tolist() == after
        for e, index in enumerate(after):
            if index < local:
                assert (index >= 0) == (slots[e] // local == rank)
                if index >= 0:
                    assert slots[e] == rank * local + index
            else:
                assert index == local and redundant[rank] == e
        assert (redundant[rank] >= 0) == (local in after)
        want = list(np.argsort(slots)[rank * local : (rank + 1) * local])
        np.testing.assert_array_equal(
            expert_placement.local_experts(layer), want + [redundant[rank]]
        )


def test_replicas_need_the_ragged_exchange(slots, redundant, monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_RAGGED_A2A", False, raising=False)
    with pytest.raises(NotImplementedError, match="TPU_MOE_RAGGED_A2A"):
        expert_placement.place_experts(_layer(0))


@pytest.mark.parametrize(
    "name,redundant_slots", [(NAME, 1), ("mtp.layers.0.mlp.experts", 0)]
)
def test_experts_start_counts_the_redundant_slots(
    slots, redundant, name, redundant_slots
):
    for rank in (0, 5):
        layer = SimpleNamespace(
            layer_name=name,
            global_num_experts=E,
            moe_config=SimpleNamespace(
                moe_parallel_config=SimpleNamespace(use_ep=True),
                ep_size=EP,
                ep_rank=rank,
            ),
        )
        assert expert_placement.redundant_slots_per_rank(layer) == redundant_slots
        assert moe_routing.get_experts_start(layer) == rank * (
            E // EP + redundant_slots
        )


def test_buffers_with_replicas_hold_physical_slots(slots, redundant):
    layer = _with_expert_tensors(_layer(2), E // EP + 1)
    expert_placement.place_experts(layer)
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    primary = core.physical_slots(slots, EP, 1)
    np.testing.assert_array_equal(layer._expert_slots.numpy(), primary)
    assert layer._expert_slots.dtype == torch.int32
    table = core.replica_table(slots, redundant, EP)
    np.testing.assert_array_equal(layer._expert_replicas.numpy(), table)
    ids = torch.randint(0, E, (12, 10), dtype=torch.int32)
    ids[0, 3] = -1
    got = expert_placement.to_slots(layer, ids).numpy()
    np.testing.assert_array_equal(got, np.where(ids >= 0, primary[ids.clip(0)], -1))
    # A layer outside the file has no slots and no replica table.
    plain = _layer(0, name="mtp.layers.0.mlp.experts")
    expert_placement.place_experts(plain)
    expert_placement.register_slots_buffer(plain, device=torch.device("cpu"))
    assert plain._expert_slots is None and plain._expert_replicas is None


def test_empty_redundant_slots_are_zeroed(slots, redundant):
    for rank in (3, 4):  # rank 3's redundant slot is empty, rank 4's is not
        layer = _with_expert_tensors(_layer(rank), E // EP + 1)
        expert_placement.place_experts(layer)
        expert_placement.zero_empty_redundant(layer)
        for param in layer.parameters(recurse=False):
            assert float(param[: E // EP].min()) == 1.0
            assert float(param[E // EP].abs().max()) == float(rank != 3)


def test_boot_check_finds_a_replica_without_its_primary_bytes(
    slots, redundant, monkeypatch
):
    """Every rank's slot sums gathered over EP: a replica whose sums differ
    from its primary's stops the boot; an empty redundant slot is not
    checked."""
    from vllm.distributed import parallel_state

    def sums_of(rank, tamper):
        layer = _layer(rank)
        expert_placement.place_experts(layer)
        experts = expert_placement.local_experts(layer)
        sums = torch.from_numpy(experts[:, None] * 8 + np.arange(8)).to(torch.int32)
        if tamper is not None and tamper[0] == rank:
            sums[tamper[1], 0] += 1
        return sums

    monkeypatch.setattr(
        parallel_state,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=EP, cpu_group=None),
    )
    local = E // EP
    # (rank, slot) whose sums change: none; rank 3's empty redundant slot; rank
    # 5's replica; rank 5's first primary, which rank 4 replicates.
    cases = [
        (None, None),
        ((3, local), None),
        ((5, local), r"replica of expert \d+ on EP rank 5 "),
        ((5, 0), r"replica of expert \d+ on EP rank 4 "),
    ]
    for tamper, error in cases:
        everyone = [sums_of(rank, tamper) for rank in range(EP)]

        def all_gather(out, tensor, group=None, everyone=everyone):
            for o, t in zip(out, everyone):
                o.copy_(t)

        monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
        monkeypatch.setattr(
            expert_placement, "_slot_sums", lambda *t, mine=everyone[1]: mine
        )
        layer = _with_expert_tensors(_layer(1), local + 1)
        expert_placement.place_experts(layer)
        if error is None:
            expert_placement.check_replicas(layer)
        else:
            with pytest.raises(RuntimeError, match=error):
                expert_placement.check_replicas(layer)


@pytest.fixture
def counting(monkeypatch, tmp_path):
    """TPU_MOE_ROUTING_COUNTS_DIR set to a fresh directory; returns it."""
    counts_dir = tmp_path / "counts"
    monkeypatch.setattr(expert_placement, "_COUNTS_DIR", str(counts_dir))
    monkeypatch.setattr(expert_placement, "_counted_layers", {})
    monkeypatch.setattr(expert_placement, "_num_counted_experts", 0)
    monkeypatch.setattr(expert_placement, "_writer", None)
    expert_placement._probe_counts_dir.cache_clear()
    yield counts_dir
    expert_placement._probe_counts_dir.cache_clear()


class _Log:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg, *args):
        self.infos.append(msg % args)

    def warning(self, msg, *args):
        self.warnings.append(msg % args)


class _Clock:
    """Stands in for the module's ``time``: a settable ``time()``."""

    strftime = staticmethod(time.strftime)
    gmtime = staticmethod(time.gmtime)

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def _padding(monkeypatch, rows, valid):
    state = token_padding.TokenPaddingState.create(rows, torch.device("cpu"))
    state.update(valid, rows)
    monkeypatch.setattr(token_padding, "_padding_state", state)


def _ranks(monkeypatch, tp, dp=1):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: tp)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(rank_in_group=dp)
    )
    monkeypatch.setattr(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )


def _counted_layer(name=NAME, experts=E, dp_size=1, dp_rank=0, pcp_size=1):
    """A MoE layer stub through place_experts and register_slots_buffer."""
    layer = torch.nn.Module()
    layer.layer_name = name
    layer.global_num_experts = experts
    parallel = SimpleNamespace(dp_size=dp_size, dp_rank=dp_rank, pcp_size=pcp_size)
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel)
    layer.quant_method = SimpleNamespace(supports_internal_mk=dp_size == 1)
    expert_placement.place_experts(layer)
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    return layer


def _read(path):
    with np.load(path) as data:
        assert set(data.files) == {"counts", "layer_names"}
        return data["counts"], [str(n) for n in data["layer_names"]]


def test_counting_is_off_without_the_switch(slots, monkeypatch, tmp_path):
    monkeypatch.setattr(expert_placement, "_COUNTS_DIR", "")
    expert_placement._probe_counts_dir.cache_clear()
    layer = _layer(2)
    expert_placement.place_experts(layer)
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    assert not hasattr(layer, "_tpu_count_routes")
    assert expert_placement.routing_counts(layer) is None
    # Returns before any tensor op.
    assert expert_placement.count_routes(layer, object()) is None
    assert not expert_placement.counting_routes()
    assert expert_placement._probe_counts_dir.cache_info().misses == 0
    expert_placement.start_writing_counts()
    assert expert_placement._writer is None
    expert_placement.write_counts(final=True)
    assert list(tmp_path.iterdir()) == []


def test_route_counts_logical_ids_of_real_rows(slots, counting, monkeypatch):
    layer = _layer(2)
    layer.quant_method = SimpleNamespace(supports_internal_mk=True)
    expert_placement.place_experts(layer)
    expert_placement.register_slots_buffer(layer, device=torch.device("cpu"))
    assert expert_placement.counting_routes()
    _padding(monkeypatch, rows=8, valid=5)
    ids = torch.randint(0, E, (8, 10), dtype=torch.int32)
    weights = torch.rand(8, 10)
    monkeypatch.setattr(moe_routing, "_route_logical", lambda *args: (weights, ids))
    got_weights, got_ids = moe_routing.route(layer, None, None)
    assert got_weights is weights
    np.testing.assert_array_equal(got_ids.numpy(), slots[ids.numpy()])
    want = np.bincount(ids[:5].numpy().ravel(), minlength=E)
    counts = expert_placement.routing_counts(layer)
    assert counts.dtype == torch.int32
    np.testing.assert_array_equal(counts.numpy(), want)
    moe_routing.route(layer, None, None)
    np.testing.assert_array_equal(counts.numpy(), 2 * want)


def test_gathered_rows_count_only_this_engine(counting, monkeypatch):
    layer = _counted_layer(dp_size=4, dp_rank=2)
    assert not layer.quant_method.supports_internal_mk
    _padding(monkeypatch, rows=8, valid=5)
    ids = torch.randint(0, E, (32, 10), dtype=torch.int32)
    expert_placement.count_routes(layer, ids)
    np.testing.assert_array_equal(
        expert_placement.routing_counts(layer).numpy(),
        np.bincount(ids[16:21].numpy().ravel(), minlength=E),
    )


def test_counting_refuses_pcp_mixed_expert_counts_and_bad_dirs(
    counting, monkeypatch, tmp_path
):
    with pytest.raises(NotImplementedError, match="PCP"):
        _counted_layer(pcp_size=2)
    _counted_layer("model.layers.0.mlp.experts", experts=64)
    with pytest.raises(NotImplementedError, match="experts"):
        _counted_layer("model.layers.1.mlp.experts", experts=32)
    (tmp_path / "file").write_text("")
    monkeypatch.setattr(expert_placement, "_COUNTS_DIR", str(tmp_path / "file" / "sub"))
    expert_placement._probe_counts_dir.cache_clear()
    with pytest.raises(OSError):
        _counted_layer("model.layers.2.mlp.experts")


def test_counts_file_baseline_minute_wrap_and_final(counting, monkeypatch):
    clock = _Clock(1000.5)
    monkeypatch.setattr(expert_placement, "time", clock)
    _ranks(monkeypatch, tp=0, dp=1)
    first = _counted_layer("model.layers.0.mlp.experts")
    second = _counted_layer("model.layers.1.mlp.experts")
    expert_placement.start_writing_counts()
    (path,) = counting.glob("routing-counts-*")
    assert path.name.startswith("routing-counts-dp1-pp0-") and path.suffix == ".npz"
    counts, names = _read(path)
    assert names == ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]
    assert counts.dtype == np.int64 and counts.shape == (2, E) and not counts.any()

    # Counted before the first step (profiling, warm-up): not in the totals.
    expert_placement.routing_counts(first).add_(5)
    expert_placement.write_counts()
    assert expert_placement._writer.due == 1020
    assert not _read(path)[0].any()
    clock.now = 1019.9
    expert_placement.routing_counts(first).add_(7)
    expert_placement.routing_counts(second).add_(1)
    expert_placement.write_counts()
    assert not _read(path)[0].any()
    clock.now = 1020.0
    expert_placement.write_counts()
    assert expert_placement._writer.due == 1080
    np.testing.assert_array_equal(_read(path)[0], [[7] * E, [1] * E])

    # Totals stay exact across the int32 wrap of the device counts.
    expert_placement.routing_counts(first).fill_(2**31 - 1)
    expert_placement.write_counts(final=True)
    before = _read(path)[0][0, 0]
    assert before == 7 + 2**31 - 1 - 12
    expert_placement.routing_counts(first).fill_(-(2**31) + 9)
    expert_placement.write_counts(final=True)
    assert _read(path)[0][0, 0] == before + 10
    assert list(counting.glob("*.tmp")) == []


def test_counts_file_only_on_the_first_tp_rank_and_write_errors_warn(
    counting, monkeypatch
):
    clock = _Clock(1000.5)
    monkeypatch.setattr(expert_placement, "time", clock)
    log = _Log()
    monkeypatch.setattr(expert_placement, "logger", log)
    layer = _counted_layer()
    _ranks(monkeypatch, tp=1)
    expert_placement.start_writing_counts()
    assert expert_placement._writer is None
    assert list(counting.iterdir()) == []

    _ranks(monkeypatch, tp=0)
    expert_placement.start_writing_counts()
    (path,) = counting.iterdir()
    expert_placement.write_counts()
    expert_placement.routing_counts(layer).add_(3)
    save_counts = core.save_counts

    def full_disk(*args):
        raise OSError("no space left on device")

    monkeypatch.setattr(core, "save_counts", full_disk)
    clock.now = 1020.0
    expert_placement.write_counts()
    assert len(log.warnings) == 1 and "not written" in log.warnings[0]
    assert not _read(path)[0].any()
    monkeypatch.setattr(core, "save_counts", save_counts)
    expert_placement.routing_counts(layer).add_(4)
    clock.now = 1080.0
    expert_placement.write_counts()
    np.testing.assert_array_equal(_read(path)[0], [[7] * E])


def test_final_write_failure_warns_and_periodic_failure_raises(counting, monkeypatch):
    clock = _Clock(1000.5)
    monkeypatch.setattr(expert_placement, "time", clock)
    log = _Log()
    monkeypatch.setattr(expert_placement, "logger", log)
    _ranks(monkeypatch, tp=0)
    _counted_layer()
    expert_placement.start_writing_counts()
    expert_placement.write_counts()

    def device_fault():
        raise RuntimeError("device halted")

    monkeypatch.setattr(expert_placement._writer, "read", device_fault)
    expert_placement.write_counts(final=True)
    assert len(log.warnings) == 1 and "not saved" in log.warnings[0]
    clock.now = 1100.0
    with pytest.raises(RuntimeError, match="device halted"):
        expert_placement.write_counts()
