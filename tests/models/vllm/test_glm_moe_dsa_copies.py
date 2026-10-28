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
"""GLM-5.2 expert copies on the grouped-matmul path (glm_moe_dsa_patch.py):
the slot tables, the per-pair holder choice, the model gate and refusals, and
the FP8 method's load and forward hooks."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu.models.vllm import glm_moe_dsa_patch as glm

LAYER = "model.layers.40.mlp.experts"
EXPERTS, RANKS, TOP_K = 256, 16, 8
OWNED = EXPERTS // RANKS
SLOTS = OWNED + 1


def _small_table(shares):
    return {
        "shards": 2,
        "experts": 4,
        "layers": {"40": [[3, 0, 2], [2, 1, 0]]},
        "copy_share": {"40": shares},
    }


def _glm_table(seed=0, min_tokens=None):
    """16 ranks x (16 primaries + 1 copy): a random permutation, and a copy of
    each of 16 experts on a rank that does not own it, with a random share."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(EXPERTS, generator=generator).tolist()
    cores = [perm[r * OWNED : (r + 1) * OWNED] for r in range(RANKS)]
    copied = [cores[(r + 1) % RANKS][0] for r in range(RANKS)]
    for r in range(RANKS):
        cores[r] = cores[r] + [copied[r]]
    shares = torch.rand(RANKS, generator=generator).tolist()
    table = {
        "shards": RANKS,
        "experts": EXPERTS,
        "layers": {"40": cores},
        "copy_share": {"40": {str(e): s for e, s in zip(copied, shares)}},
    }
    if min_tokens is not None:
        table["copy_min_tokens"] = min_tokens
    return table


@pytest.fixture
def use_table(tmp_path, monkeypatch):
    def use(table):
        path = tmp_path / "placement.json"
        path.write_text(json.dumps(table))
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
        ep._load.cache_clear()

    monkeypatch.setattr(glm, "_glm_moe_dsa", True)
    monkeypatch.setattr(ep, "_LOCAL_IDS", {})
    monkeypatch.setattr(ep, "_ROUTED", set())
    monkeypatch.setattr(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_FP8_ALL_GATHER", False, raising=False)
    yield use
    ep._load.cache_clear()


class _Experts(torch.nn.Module):
    """What install_copy_routing reads from a RoutedExperts layer."""

    def __init__(self, rank=0, dp_size=RANKS, use_ep=True, pcp_size=1):
        super().__init__()
        self.layer_name = LAYER
        self.w13_weight = torch.zeros(1)
        parallel = SimpleNamespace(
            use_ep=use_ep, ep_rank=rank, dp_size=dp_size, pcp_size=pcp_size
        )
        self.moe_config = SimpleNamespace(moe_parallel_config=parallel)


def _installed(rank=0, **kwargs):
    layer = _Experts(rank, **kwargs)
    assert glm.install_copy_routing(layer)
    return layer


def test_copy_tables_in_slot_ids(use_table):
    use_table(_small_table({"2": 0.5, "0": 0.75}))
    table = ep.placement()
    # Router rows: [3, 0, 2, 1]. Slot ids: rank * 3 + slot.
    rows, copy, share = glm.copy_tables(table, 40, table.shares[40])
    # Expert 0 (row 1) is copied to rank 1 slot 2, expert 2 (row 2) to rank 0.
    assert (rows, copy, share) == ([1, 2], [5, 2], [24576, 16384])


def test_copy_routing_needs_expert_parallelism_and_a_table(use_table, monkeypatch):
    use_table(_small_table({"2": 0.5, "0": 0.5}))
    assert not glm.install_copy_routing(_Experts(use_ep=False))
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT")
    assert not glm.install_copy_routing(_Experts())


def test_shares_are_refused_off_glm(use_table, monkeypatch):
    use_table(_small_table({"2": 0.5, "0": 0.5}))
    monkeypatch.setattr(glm, "_glm_moe_dsa", False)
    with pytest.raises(NotImplementedError, match="GLM-5.2"):
        glm.install_copy_routing(_Experts())


@pytest.mark.parametrize(
    ("name", "value"),
    [("TPU_MOE_COLLECTION_CHUNK_SIZE", 256), ("TPU_MOE_FP8_ALL_GATHER", True)],
)
def test_shares_are_refused_when_the_moe_path_owns_its_collectives(
    use_table, monkeypatch, name, value
):
    """Chunk pipelining and the FP8 all-gather route each DP rank's own rows."""
    use_table(_small_table({"2": 0.5, "0": 0.5}))
    monkeypatch.setattr(envs, name, value, raising=False)
    with pytest.raises(NotImplementedError, match="all-gathered rows"):
        glm.install_copy_routing(_Experts())


def test_shares_need_copy_slots(use_table):
    table = _small_table({"2": 0.5, "0": 0.5})
    table["layers"]["40"] = [[3, 0], [2, 1]]
    use_table(table)
    with pytest.raises(ValueError, match="no copy slots"):
        ep.placement()


@pytest.mark.parametrize(("dp", "pcp"), [(RANKS, 1), (8, 2)])
def test_the_gate_counts_the_rows_gathered_over_dp_and_pcp(use_table, dp, pcp):
    """The routed rows are gathered over the DP and PCP ranks."""
    use_table(_glm_table(min_tokens=4))
    assert _installed(dp_size=dp, pcp_size=pcp)._copy_min_rows == 4 * dp * pcp


def test_the_model_gate(monkeypatch):
    monkeypatch.setattr(glm, "_glm_moe_dsa", False)

    def config(model_type):
        return SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))

    assert glm.maybe_patch_glm_moe_dsa(None) is False
    assert glm.maybe_patch_glm_moe_dsa(config("deepseek_v3")) is False
    assert not glm._glm_moe_dsa
    assert glm.maybe_patch_glm_moe_dsa(config("glm_moe_dsa")) is None
    assert glm._glm_moe_dsa


def _tables(layer=40):
    """Per router row: its primary's slot id, its copy's (the primary's if it
    has none) and its share, from copy_tables."""
    table = ep.placement()
    v = torch.arange(table.experts, dtype=torch.int32)
    primary = (v // table.owned) * table.slots + v % table.owned
    copy, share = primary.clone(), torch.zeros_like(primary)
    rows, slots, shares = glm.copy_tables(table, layer, table.shares[layer])
    copy[rows] = torch.tensor(slots, dtype=torch.int32)
    share[rows] = torch.tensor(shares, dtype=torch.int32)
    return primary, copy, share


def _hash(rows):
    position = torch.arange(rows * TOP_K, dtype=torch.int64)
    return ((position % 32768 * 0x4F1B + position // 32768 * 0x1B3) % 32768).view(
        rows, TOP_K
    )


def test_route_to_copies_equals_the_table_lookup(use_table):
    """The comparison form gives exactly the per-router-row table lookup, at
    more positions than one 15-bit hash block."""
    rows = 16384
    use_table(_glm_table())
    layer = _installed()
    primary, copy, share = _tables()
    torch.manual_seed(rows)
    ids = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32)
    expected = torch.where(_hash(rows) < share[ids], copy[ids], primary[ids])
    assert torch.equal(glm.route_to_copies(layer, ids), expected)


def test_each_pair_has_one_holder_and_shares_hold(use_table):
    use_table(_glm_table())
    layer = _installed()
    primary_t, copy_t, share_t = _tables()
    torch.manual_seed(0)
    rows = 16384
    ids = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32)
    slot_ids = glm.route_to_copies(layer, ids)
    # The grouped path's remap on every rank: exactly one rank serves a pair.
    holders = sum(
        ((slot_ids - r * SLOTS >= 0) & (slot_ids - r * SLOTS < SLOTS)).int()
        for r in range(RANKS)
    )
    assert torch.all(holders == 1)
    share = share_t.double() / glm.SHARE_ONE
    for row in torch.nonzero(copy_t != primary_t).flatten():
        mask = ids == row
        n = int(mask.sum())
        served = float((slot_ids[mask] == copy_t[row]).double().mean())
        s = float(share[row])
        assert abs(served - s) <= 4 * (s * (1 - s) / n) ** 0.5 + 1 / n


def test_small_buckets_keep_the_primaries(use_table):
    use_table(_glm_table(min_tokens=64))
    layer = _installed()
    primary, _, _ = _tables()
    torch.manual_seed(0)
    ids = torch.randint(0, EXPERTS, (64 * RANKS - 1, TOP_K), dtype=torch.int32)
    assert torch.equal(glm.route_to_copies(layer, ids), primary[ids])
    ids = torch.randint(0, EXPERTS, (64 * RANKS, TOP_K), dtype=torch.int32)
    assert not torch.equal(glm.route_to_copies(layer, ids), primary[ids])


def test_route_to_copies_compiles_on_both_sides_of_the_gate(use_table):
    use_table(_glm_table(min_tokens=4))  # the gate is at 4 * 16 = 64 rows
    layer = _installed()
    compiled = torch.compile(
        glm.route_to_copies, backend="eager", dynamic=True, fullgraph=True
    )
    torch.manual_seed(0)
    for rows in (32, 128):
        ids = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32)
        assert torch.equal(compiled(layer, ids), glm.route_to_copies(layer, ids))


def test_the_fp8_method_routes_to_copies(use_table, monkeypatch):
    """apply_monolithic maps the routed ids and passes the slot start."""
    from vllm_torchtpu.layers.adapter.quantization import fp8

    use_table(_glm_table())
    layer = _installed(rank=3)
    layer.moe_config.experts_per_token = TOP_K
    layer._experts_start = torch.tensor(3 * OWNED, dtype=torch.int32)
    for name in ("w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
        setattr(layer, name, torch.zeros(1))
    rows = 4 * RANKS
    torch.manual_seed(0)
    ids = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32)
    weights = torch.rand(rows, TOP_K)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False, raising=False)
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(fp8.moe_routing, "route", Mock(return_value=(weights, ids)))
    gmm = Mock(return_value="gmm")
    monkeypatch.setattr(fp8, "fused_moe_gmm", gmm)
    method = SimpleNamespace(
        _tpu_activation_str="silu",
        _use_adaptive_fused_moe=False,
        _serve_copies=True,
    )
    x = torch.zeros(rows, 8)
    assert fp8.VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, x) == "gmm"
    kwargs = gmm.call_args.kwargs
    assert kwargs["experts_start"] is layer._copy_slots_start
    assert torch.equal(kwargs["topk_ids"], glm.route_to_copies(layer, ids))
    assert kwargs["topk_weights"] is weights


def _fp8_layer(rank=0):
    """A placed layer as RoutedExperts leaves it for the FP8 method's load
    hook: slots x tiny FP8 weights, EP16 at DP 16."""
    layer = torch.nn.Module()
    layer.layer_name = LAYER
    layer.activation = "silu"
    layer.renormalize = True
    layer.global_num_experts = EXPERTS
    layer.expert_placement_strategy = "linear"
    parallel = SimpleNamespace(use_ep=True, ep_rank=rank, dp_size=RANKS, pcp_size=1)
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=parallel,
        experts_per_token=TOP_K,
        ep_size=RANKS,
        ep_rank=rank,
    )
    for name, shape in (
        ("w13_weight", (SLOTS, 128, 256)),
        ("w2_weight", (SLOTS, 128, 128)),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.float8_e4m3fn), requires_grad=False
            ),
        )
    return layer


@pytest.mark.parametrize("case", ["unset", "served"])
def test_the_fp8_method_serves_copies_at_load(use_table, monkeypatch, case):
    """A layer with copy slots is routed to every slot. Without
    TPU_EXPERT_PLACEMENT nothing changes."""
    from vllm_torchtpu.layers.adapter.quantization import fp8

    use_table(_glm_table())
    if case == "unset":
        monkeypatch.delenv("TPU_EXPERT_PLACEMENT")
    else:
        ep._LOCAL_IDS[40] = frozenset(ep.placement().cores[40][3])
    layer = _fp8_layer(rank=3)
    monkeypatch.setattr(fp8, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", None, raising=False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False, raising=False)
    monkeypatch.setattr(
        fp8,
        "_process_fp8_moe_weights",
        lambda layer, **kw: (
            layer.w13_weight.data,
            torch.ones(SLOTS, 1, 1, 256),
            layer.w2_weight.data,
            torch.ones(SLOTS, 1, 1, 128),
            "float8_e4m3fn",
            None,
        ),
    )
    prebuilt = Mock()
    monkeypatch.setattr(fp8, "prebuild_fused_moe_kernel", prebuilt)
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = None
    method.weight_scale_name = "weight_scale"

    method.process_weights_after_loading(layer)

    assert method._serve_copies == (case == "served")
    prebuilt.assert_called_once()
    assert layer._experts_start.item() == 3 * OWNED
    assert hasattr(layer, "_copy_slots_start") == (case == "served")
    if case == "served":
        assert layer._copy_slots_start.item() == 3 * SLOTS
