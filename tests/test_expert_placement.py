"""The static expert placement (expert_placement.py, TPU_EXPERT_PLACEMENT) on
synthetic tables and on the shipped Qwen3.5 tables: the table checks, the slot
maps, the router-row permutation, the identity without a table, the backbone
anchoring that leaves the MTP drafter's layer on the checkpoint's contiguous
placement, the loader's expert filter, the expert-map patch on vLLM's manager,
the primary-interval bound of the grouped / dense / few-token ids paths, and
the moe_lowtok_scores entry behavior with and without copies.
"""

from __future__ import annotations

import functools
import json
import pathlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu import model_loader_patches as mlp
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores as lowtok_scores
from vllm_torchtpu.layers.core import fused_moe_gmm as gmm

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "scripts/vllm/benchmark/inferencex/qwen3.5"
PLAIN = DATA / "expert_placement_dp8_ep8.json"
COPIES = DATA / "expert_placement_dp8_ep8_opt_copies128.json"

BACKBONE_PREFIX = "language_model.model.layers.{n}.mlp.experts"
BACKBONE_NAME = "model.language_model.layers.{n}.mlp.experts.{e}.down_proj.weight"
BACKBONE_GATE = "model.language_model.layers.{n}.mlp.gate.weight"
MTP_PREFIX = "mtp.layers.0.mlp.experts"
MTP_NAME = "mtp.layers.0.mlp.experts.{e}.down_proj.weight"
MTP_GATE = "mtp.layers.0.mlp.gate.weight"


@pytest.fixture(autouse=True)
def _fresh_table(monkeypatch):
    """Every test starts without a table and with the module's cache empty."""
    monkeypatch.setattr(envs, "TPU_EXPERT_PLACEMENT", "")
    monkeypatch.setattr(ep, "_DATA", None)
    monkeypatch.setattr(ep, "_DATA_PATH", "")
    monkeypatch.setattr(ep, "_LOGGED", set())
    ep._local_ids.cache_clear()
    yield
    ep._local_ids.cache_clear()


def _use(monkeypatch, path):
    monkeypatch.setattr(envs, "TPU_EXPERT_PLACEMENT", str(path))


def _table(tmp_path, dup=False):
    order = [3, 3, 2, 1] if dup else [3, 0, 2, 1]  # rank 0: slots 0, 1; rank 1
    table = {
        "shards": 2,
        "experts": 4,
        "slots_per_core": 2,
        "layers": {str(n): {"cores": [order[:2], order[2:]]} for n in (0, 1)},
    }
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(table))
    return str(path), order


def _copy_table(tmp_path):
    table = {
        "shards": 2,
        "experts": 4,
        "owned_per_core": 2,
        "slots_per_core": 3,
        "copies": True,
        "layers": {
            "0": {
                "cores": [[0, 1, 2], [2, 3, 0]],
                "copies": {"2": 0, "0": 1},
            }
        },
    }
    path = tmp_path / "copies.json"
    path.write_text(json.dumps(table))
    return path, table


def test_maps_and_router_rows(tmp_path, monkeypatch):
    path, order = _table(tmp_path)
    _use(monkeypatch, path)
    assert ep.enabled()
    assert (ep.shards(), ep.slots(), ep.global_count()) == (2, 2, 4)
    assert ep.virtual_order(0) == order
    for rank in range(2):
        m = ep.expert_map(0, rank)
        for j, e in enumerate(ep.cores(0)[rank]):
            assert m[e] == j
            assert ep.virtual_order(0)[rank * 2 + j] == e
        assert ep.local_ids(0, rank) == frozenset(ep.cores(0)[rank])
        assert m.count(-1) == 2
    w = torch.arange(4.0).reshape(4, 1)
    out = ep.permute_router_rows(BACKBONE_GATE.format(n=0), w)
    assert out[:, 0].tolist() == [3.0, 0.0, 2.0, 1.0]
    assert (
        ep.permute_router_rows("model.language_model.layers.0.mlp.up_proj.weight", w)
        is w
    )
    with pytest.raises(ValueError):
        ep.permute_router_rows(BACKBONE_GATE.format(n=1), torch.zeros(5, 1))


def test_a_duplicated_expert_is_refused(tmp_path, monkeypatch):
    path, _ = _table(tmp_path, dup=True)
    _use(monkeypatch, path)
    with pytest.raises(ValueError):
        ep.data()


def test_unset_is_the_identity():
    assert not ep.enabled()
    assert not ep.has_copies()
    assert not ep.placed(BACKBONE_PREFIX.format(n=0))
    w = torch.zeros(4, 1)
    assert ep.permute_router_rows(BACKBONE_GATE.format(n=0), w) is w


def test_copy_tables_preserve_primary_order(tmp_path, monkeypatch):
    path, _ = _copy_table(tmp_path)
    _use(monkeypatch, path)
    assert ep.virtual_order(0) == [0, 1, 2, 3]
    assert ep.hot_order(0) == [2, 0]
    assert ep.holder_tables(0, 0) == ([0, 1, 2], [0, 0, 1, 1], [1, -1, 0, -1])
    assert ep.holder_tables(0, 1) == ([2, 3, 0], [0, 0, 1, 1], [1, -1, 0, -1])
    assert ep.expert_map(0, 1) == [2, -1, 0, 1]


@pytest.mark.parametrize("case", ["invalid_id", "duplicate_copy", "wrong_order"])
def test_invalid_copy_tables_are_rejected(tmp_path, monkeypatch, case):
    path, table = _copy_table(tmp_path)
    if case == "invalid_id":
        table["layers"]["0"]["cores"][0][-1] = 4
    elif case == "duplicate_copy":
        table["slots_per_core"] = 4
        table["layers"]["0"]["cores"] = [[0, 1, 2, 2], [2, 3, 0, 1]]
    else:
        table["layers"]["0"]["copies"] = {"2": 1, "0": 0}
    path.write_text(json.dumps(table))
    _use(monkeypatch, path)
    with pytest.raises(ValueError):
        ep.data()


@pytest.mark.parametrize(
    "name, expected",
    [
        (BACKBONE_PREFIX.format(n=12), 12),
        (BACKBONE_NAME.format(n=3, e=511), 3),
        (BACKBONE_GATE.format(n=59), 59),
        ("model.language_model.layers.7.mlp.experts.0.gate_up_proj.weight_scale", 7),
        (MTP_PREFIX, None),
        (MTP_NAME.format(e=5), None),
        (MTP_GATE, None),
        ("mtp.layers.0.mlp.experts.0.gate_up_proj.weight", None),
        ("model.layers.0.mlp.gate.weight", None),
        ("language_model.model.norm.weight", None),
        ("mtp.fc.weight", None),
    ],
)
def test_layer_index_is_anchored_to_the_backbone(name, expected):
    assert ep.layer_index(name) == expected


def test_the_mtp_router_and_experts_are_left_alone(tmp_path, monkeypatch):
    path, _ = _table(tmp_path)
    _use(monkeypatch, path)
    w = torch.arange(4.0).reshape(4, 1)
    assert ep.permute_router_rows(MTP_GATE, w) is w
    assert not ep.placed(MTP_PREFIX)
    assert ep.placed(BACKBONE_PREFIX.format(n=1))
    assert not ep.placed(BACKBONE_PREFIX.format(n=2))


def test_loader_filter_keeps_the_placement_experts_and_the_mtp_contiguous_set(
    tmp_path, monkeypatch
):
    path, order = _table(tmp_path)
    _use(monkeypatch, path)
    monkeypatch.setattr(mlp, "_ep_rank", lambda: 1)
    contiguous = {2, 3}
    kept = {
        e
        for e in range(4)
        if not mlp._should_skip(BACKBONE_NAME.format(n=0, e=e), contiguous)
    }
    assert kept == set(order[2:]) == {2, 1}
    kept_mtp = {
        e for e in range(4) if not mlp._should_skip(MTP_NAME.format(e=e), contiguous)
    }
    assert kept_mtp == contiguous
    assert not mlp._should_skip(BACKBONE_GATE.format(n=0), contiguous)
    assert not mlp._should_skip(
        "model.language_model.layers.0.mlp.experts.0.input_scale", contiguous
    )


def test_loader_filter_is_the_contiguous_set_without_a_table(monkeypatch):
    monkeypatch.setattr(mlp, "_ep_rank", lambda: pytest.fail("rank not needed"))
    contiguous = {2, 3}
    kept = {
        e
        for e in range(4)
        if not mlp._should_skip(BACKBONE_NAME.format(n=0, e=e), contiguous)
    }
    assert kept == contiguous


def test_loader_filter_with_copies_keeps_the_copy_slots_experts(tmp_path, monkeypatch):
    path, _ = _copy_table(tmp_path)
    _use(monkeypatch, path)
    monkeypatch.setattr(mlp, "_ep_rank", lambda: 1)
    kept = {
        e
        for e in range(4)
        if not mlp._should_skip(BACKBONE_NAME.format(n=0, e=e), {2, 3})
    }
    assert kept == {2, 3, 0}


class _Manager:
    def __init__(self, rank, ranks=2, experts=4, use_ep=True):
        self.moe_parallel_config = SimpleNamespace(use_ep=use_ep)
        self.ep_size = ranks
        self.ep_rank = rank
        self.global_num_experts = experts
        self.num_fused_shared_experts = 0
        n = experts // ranks
        m = [-1] * experts
        for j in range(n):
            m[rank * n + j] = j
        self._local_num_experts = n
        self._expert_map = torch.tensor(m, dtype=torch.int32)
        self._expert_mask = None
        self._tpu_expert_map_source = self._expert_map
        self._tpu_expert_map_version = self._expert_map._version
        self._tpu_expert_map_host = list(m)


def test_apply_expert_map_gives_a_placed_layer_its_slots(tmp_path, monkeypatch):
    path, order = _table(tmp_path)
    _use(monkeypatch, path)
    for rank in range(2):
        mgr = _Manager(rank)
        assert ep.apply_expert_map(mgr, BACKBONE_PREFIX.format(n=0))
        assert mgr._local_num_experts == 2
        assert mgr._expert_map.tolist() == ep.expert_map(0, rank)
        assert mgr._expert_map.dtype == torch.int32
        assert mgr._tpu_expert_map_host == ep.expert_map(0, rank)
        assert mgr._tpu_expert_map_source is mgr._expert_map
        for j, e in enumerate(order[2 * rank : 2 * rank + 2]):
            assert mgr._expert_map[e] == j


def test_apply_expert_map_leaves_the_mtp_layer_on_the_contiguous_map(
    tmp_path, monkeypatch
):
    path, _ = _table(tmp_path)
    _use(monkeypatch, path)
    mgr = _Manager(1)
    before = mgr._expert_map.clone()
    assert not ep.apply_expert_map(mgr, MTP_PREFIX)
    assert mgr._local_num_experts == 2
    assert torch.equal(mgr._expert_map, before)
    assert mgr._tpu_expert_map_host == [-1, -1, 0, 1]


def test_apply_expert_map_with_copies_counts_every_slot(tmp_path, monkeypatch):
    path, _ = _copy_table(tmp_path)
    _use(monkeypatch, path)
    mgr = _Manager(1)
    assert ep.apply_expert_map(mgr, BACKBONE_PREFIX.format(n=0))
    assert mgr._local_num_experts == 3
    assert mgr._expert_map.tolist() == [2, -1, 0, 1]


def test_apply_expert_map_refuses_an_unknown_layer_or_a_wrong_table(
    tmp_path, monkeypatch
):
    path, _ = _table(tmp_path)
    _use(monkeypatch, path)
    with pytest.raises(RuntimeError, match="no layer prefix"):
        ep.apply_expert_map(_Manager(0), "")
    with pytest.raises(ValueError, match="2 ranks x 4 experts"):
        ep.apply_expert_map(
            _Manager(0, ranks=4, experts=8), BACKBONE_PREFIX.format(n=0)
        )
    assert not ep.apply_expert_map(
        _Manager(0, use_ep=False), BACKBONE_PREFIX.format(n=0)
    )


def test_install_model_patches_is_a_no_op_without_a_table():
    from vllm.model_executor.layers.fused_moe import expert_map_manager as emm
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    before = (
        fused_layer.FusedMoEFactory,
        emm.ExpertMapManager._calculate_expert_maps,
    )
    ep.install_model_patches()
    assert (
        fused_layer.FusedMoEFactory,
        emm.ExpertMapManager._calculate_expert_maps,
    ) == before
    assert not getattr(fused_layer, "_tpu_placement_patch", False)


@pytest.mark.parametrize("path", [PLAIN, COPIES])
def test_the_shipped_tables_load_and_cover_the_backbone(path, monkeypatch):
    _use(monkeypatch, path)
    assert (ep.shards(), ep.global_count(), ep.owned()) == (8, 512, 64)
    assert sorted(int(k) for k in ep.data()["layers"]) == list(range(60))
    if path is COPIES:
        assert ep.has_copies() and ep.slots() == 80
    else:
        assert not ep.has_copies() and ep.slots() == 64
    for layer in (0, 31, 59):
        order = ep.virtual_order(layer)
        assert sorted(order) == list(range(512))
        for rank in range(8):
            m = ep.expert_map(layer, rank)
            held = [e for e in range(512) if m[e] >= 0]
            assert len(held) == ep.slots()
            assert set(held) == ep.local_ids(layer, rank)
            for j in range(64):
                assert m[order[rank * 64 + j]] == j
            if path is COPIES:
                gid, prim, copy = ep.holder_tables(layer, rank)
                assert len(gid) == 80 and gid[:64] == list(
                    range(rank * 64, rank * 64 + 64)
                )
                assert all(prim[v] == v // 64 for v in range(512))
                assert all(copy[v] != rank for v in gid[:64])
                assert all(copy[v] == rank for v in gid[64:])
        if path is COPIES:
            hot = ep.hot_order(layer)
            assert len(hot) == 128 and len(set(hot)) == 128
    assert not ep.placed(MTP_PREFIX)
    assert ep.placed(BACKBONE_PREFIX.format(n=59))


def _fp8_operands(tokens, experts, hidden=512, inter=256, topk=4):
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(
        rng.standard_normal((tokens, hidden)).astype(np.float32), jnp.bfloat16
    )
    w1 = jnp.zeros((experts, hidden, 2 * inter), jnp.float8_e4m3fn)
    w2 = jnp.zeros((experts, inter, hidden), jnp.float8_e4m3fn)
    s1 = jnp.ones((experts, 1, 1, 2 * inter), jnp.float32)
    s2 = jnp.ones((experts, 1, 1, hidden), jnp.float32)
    ids = jnp.asarray(np.tile(np.arange(24, dtype=np.int32), (tokens, 1)), jnp.int32)
    w = jnp.ones(ids.shape, jnp.bfloat16)
    return x, w1, w2, s1, s2, w, ids


def _ids_seen_by(monkeypatch, *, slots, primaries, cap, dense):
    seen = {}

    def route_probe(topk_ids, topk_weights, num_local_experts):
        seen["lowtok"] = (topk_ids, topk_weights)
        return jnp.zeros((topk_ids.shape[0], num_local_experts), topk_weights.dtype)

    def lowtok_probe(x, route, *a, **k):
        return x

    def dense_probe(x, w1, w2, s1, s2, w, ids):
        seen["dense"] = (ids, w)
        return x

    def grouped_probe(hidden_states, topk_ids, topk_weights, **k):
        seen["grouped"] = (topk_ids, topk_weights)
        rows = hidden_states.shape[0] * topk_ids.shape[1]
        return (
            jnp.zeros((rows, hidden_states.shape[1]), hidden_states.dtype),
            jnp.zeros((k["local_num_experts"],), jnp.int32),
            jnp.zeros((rows,), jnp.int32),
            topk_weights.reshape(-1),
            jnp.ones((rows,), bool),
            jnp.zeros((rows,), jnp.int32),
        )

    monkeypatch.setattr(gmm, "_MOE_LOWTOK", cap > 0)
    monkeypatch.setattr(gmm, "_MOE_LOWTOK_IDS", cap > 0)
    monkeypatch.setattr(gmm, "_MOE_LOWTOK_MAX_TOKENS", cap)
    monkeypatch.setattr(gmm, "_DENSE_EXPERT_THRESHOLD", dense)
    monkeypatch.setattr(gmm, "moe_lowtok_routing_table", route_probe)
    monkeypatch.setattr(gmm, "moe_lowtok", lowtok_probe)
    monkeypatch.setattr(gmm, "can_use_dense_expert", lambda *a, **k: True)
    monkeypatch.setattr(gmm, "dense_expert_moe", dense_probe)
    monkeypatch.setattr(gmm, "prepare_routed_gmm_inputs", grouped_probe)
    monkeypatch.setattr(
        gmm, "moe_gmm", lambda x, *a, **k: jnp.zeros_like(x[: k["num_tokens"]])
    )
    x, w1, w2, s1, s2, w, ids = _fp8_operands(16, slots)
    jax.clear_caches()
    with jax.disable_jit():
        gmm.fused_moe_func(
            x,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            w,
            ids,
            experts_start=jnp.int32(8),
            topk=ids.shape[1],
            activation="silu",
            use_ep=True,
            use_sparse_core=False,
            onehot_moe_permute_threshold=0,
            num_primary_experts=primaries,
        )
    assert len(seen) == 1, seen.keys()
    ((path, (local_ids, weights)),) = seen.items()
    return path, np.asarray(local_ids)[0].tolist(), np.asarray(weights)[0].tolist()


@pytest.mark.parametrize(
    "cap, dense, path", [(256, 64, "lowtok"), (0, 64, "dense"), (0, 0, "grouped")]
)
def test_every_routed_path_sees_only_the_primary_interval(
    monkeypatch, cap, dense, path
):
    taken, ids, w = _ids_seen_by(
        monkeypatch, slots=12, primaries=8, cap=cap, dense=dense
    )
    assert taken == path
    assert ids == [-1] * 8 + list(range(8)) + [-1] * 8
    assert w == [0.0] * 8 + [1.0] * 8 + [0.0] * 8


@pytest.mark.parametrize("cap, dense, path", [(256, 64, "lowtok"), (0, 64, "dense")])
def test_without_a_primary_count_every_slot_is_local(monkeypatch, cap, dense, path):
    taken, ids, w = _ids_seen_by(
        monkeypatch, slots=8, primaries=None, cap=cap, dense=dense
    )
    assert taken == path
    assert ids == [-1] * 8 + list(range(8)) + [-1] * 8


def _stub_grouped_path(monkeypatch):
    def grouped(hidden_states, topk_ids, topk_weights, **k):
        rows = hidden_states.shape[0] * topk_ids.shape[1]
        return (
            jnp.zeros((rows, hidden_states.shape[1]), hidden_states.dtype),
            jnp.zeros((k["local_num_experts"],), jnp.int32),
            jnp.zeros((rows,), jnp.int32),
            topk_weights.reshape(-1),
            jnp.ones((rows,), bool),
            jnp.zeros((rows,), jnp.int32),
        )

    monkeypatch.setattr(gmm, "prepare_routed_gmm_inputs", grouped)

    def moe_gmm(x, w1, s1, b1, w2, s2, b2, group_sizes, revert, weights_flat, *a, **k):
        per_token = weights_flat.reshape(k["num_tokens"], k["topk"]).sum(
            axis=1, keepdims=True
        )
        return jnp.pad(per_token.astype(x.dtype), ((0, 0), (0, x.shape[1] - 1)))

    monkeypatch.setattr(gmm, "moe_gmm", moe_gmm)


@pytest.mark.parametrize("primaries", [8, None])
def test_fused_moe_func_traces_under_jit_with_a_primary_count(monkeypatch, primaries):
    _stub_grouped_path(monkeypatch)
    monkeypatch.setattr(gmm, "_MOE_LOWTOK", False)
    monkeypatch.setattr(gmm, "_DENSE_EXPERT_THRESHOLD", 0)
    x, w1, w2, s1, s2, w, ids = _fp8_operands(16, 12)
    kwargs = dict(
        experts_start=jnp.int32(8),
        topk=ids.shape[1],
        activation="silu",
        use_ep=True,
        use_sparse_core=False,
        onehot_moe_permute_threshold=0,
        num_primary_experts=primaries,
    )
    jax.clear_caches()
    jitted = gmm.fused_moe_func(x, w1, w2, s1, s2, None, None, w, ids, **kwargs)
    with jax.disable_jit():
        eager = gmm.fused_moe_func(x, w1, w2, s1, s2, None, None, w, ids, **kwargs)
    assert jitted.shape == x.shape and jitted.dtype == x.dtype
    assert jnp.array_equal(jitted, eager)
    assert float(jitted[0, 0]) == (8.0 if primaries == 8 else 12.0)
    from jax import export

    fn = jax.jit(functools.partial(gmm.fused_moe_func, **kwargs))
    exported = export.export(fn)(x, w1, w2, s1, s2, None, None, w, ids)
    assert exported.out_avals[0].shape == x.shape


def test_a_primary_count_outside_the_slots_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="num_primary_experts"):
        _ids_seen_by(monkeypatch, slots=8, primaries=9, cap=0, dense=0)


def test_the_custom_op_cache_refuses_copies_without_a_primary_count(
    monkeypatch, tmp_path
):
    from vllm_torchtpu.layers.adapter import fused_moe as adapter

    path, _ = _copy_table(tmp_path)
    _use(monkeypatch, path)
    monkeypatch.setattr(adapter, "_fused_moe_kernel_cache", {})
    with pytest.raises(ValueError, match="num_primary_experts"):
        adapter._get_fused_moe_custom_op(topk=4, activation="silu", use_ep=True)
    adapter._get_fused_moe_custom_op(topk=4, activation="silu", use_ep=False)
    adapter._get_fused_moe_custom_op(
        topk=4, activation="silu", use_ep=True, num_primary_experts=2
    )


def test_the_custom_op_cache_key_without_a_placement_is_the_same_op(monkeypatch):
    from vllm_torchtpu.layers.adapter import fused_moe as adapter

    monkeypatch.setattr(adapter, "_fused_moe_kernel_cache", {})
    a = adapter._get_fused_moe_custom_op(topk=4, activation="silu", use_ep=True)
    b = adapter._get_fused_moe_custom_op(
        topk=4, activation="silu", use_ep=True, num_primary_experts=None
    )
    assert a is b


def test_moe_lowtok_scores_arms_under_plain_placement_and_declines_copies(monkeypatch):
    from vllm.compilation import backends

    monkeypatch.setenv("TPU_MOE_LOWTOK", "1")
    monkeypatch.setattr(backends, "model_tag", "backbone")
    monkeypatch.setattr(lowtok_scores, "_LAYERS", [])

    def make_layer(slots):
        return SimpleNamespace(
            renormalize=True,
            scoring_func="softmax",
            custom_routing_function=None,
            use_grouped_topk=False,
            e_score_correction_bias=None,
            routed_scaling_factor=1.0,
            apply_router_weight_on_input=False,
            w13_bias=None,
            w2_bias=None,
            layer_name=BACKBONE_PREFIX.format(n=0),
            global_num_experts=512,
            moe_config=SimpleNamespace(
                experts_per_token=10,
                ep_size=8,
                ep_rank=0,
                in_dtype=torch.bfloat16,
                moe_parallel_config=SimpleNamespace(use_ep=True),
            ),
            hidden_size=4096,
            w13_weight=torch.empty(slots, 4096, 2048, dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty(slots, 1024, 4096, dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.empty(slots, 1, 1, 2048),
            w2_weight_scale_inv=torch.empty(slots, 1, 1, 4096),
        )

    method = SimpleNamespace(_use_adaptive_fused_moe=False)
    op_plain = lowtok_scores.prebuild(method, make_layer(64), "silu")
    assert op_plain is not None
    op_copies = lowtok_scores.prebuild(method, make_layer(80), "silu")
    assert op_copies is None
    assert lowtok_scores._LAYERS[-1].entry == "ids"
    assert "copy slots" in lowtok_scores._LAYERS[-1].reason


def test_auto_table_rule_and_files_exist():
    """The Qwen3.5 auto-default (tpu_platform) picks the copy count by serving
    shape; every table it can pick ships and validates."""
    from vllm_torchtpu import expert_placement

    pick = expert_placement.auto_table
    assert pick(1, 8).endswith("_opt_copies128.json")
    assert pick(4, 32).endswith("_opt_copies32.json")
    assert pick(2, 16).endswith("_opt_copies32.json")
    assert pick(1, 12).endswith("_opt_copies64.json")
    assert pick(1, 16).endswith("_opt_copies64.json")
    assert pick(1, 24).endswith("_opt_copies32.json")
    for args in ((1, 8), (1, 16), (1, 24), (4, 32)):
        path = pick(*args)
        with open(path) as f:
            table = json.load(f)
        assert table["slots_per_core"] == 64 + table["extra_slots"]
        assert len(table["layers"]) == 60
