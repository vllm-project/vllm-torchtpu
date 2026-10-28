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
"""Static expert placement keeps a GLM-5.2-style EP16 MoE layer's output on the
paths moe_tp_decode serves under expert parallelism: its fused kernel's
routing (fast and exact) and per-rank expert compaction (T <= 64), computed
with dense per-expert FFNs in place of the kernel's matmuls, and the psum
path, i.e.
_route plus fused_moe_func per rank with experts_start = 16 * rank (one-hot
and SparseCore ragged branches). The router rows go through
expert_placement.permute_router_rows and each rank's experts through
expert_placement.expert_map, as at load. Pallas kernels that cannot run on CPU
(gmm_v2 and the SparseCore gather / gather-reduce) are replaced by jnp
references in both arms. Negative controls must change the output.

A table with expert copies (16 primaries + 1 copy slot per rank) keeps the
output through fused_moe_func per rank too, with GLM-5.2's copy routing
(route_to_copies, slot ids from experts_start = 17 * rank)."""

import functools
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu import expert_placement as ep
from vllm_torchtpu.kernels import router_topk
from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as mtk
from vllm_torchtpu.layers.adapter import moe_tp_decode as mtd
from vllm_torchtpu.layers.core import fused_moe_gmm as fmg
from vllm_torchtpu.models.vllm import glm_moe_dsa_patch as glm

E, RANKS, TOPK, SCALE = 256, 16, 8, 2.5
SLOTS = E // RANKS
HIDDEN, INTER = 128, 64
THRESH = 2048  # ONEHOT_MOE_PERMUTE_THRESHOLD
LAYER, OTHER_LAYER = 40, 41


def _ref_gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, **kwargs):
    """Grouped matmul over expert-sorted rows; rows past sum(group_sizes) -> 0."""
    m = lhs.shape[0]
    seg = jnp.searchsorted(jnp.cumsum(group_sizes), jnp.arange(m), side="right")
    out = jnp.zeros((m, rhs.shape[-1]), jnp.float32)
    for e in range(rhs.shape[0]):
        prod = lhs.astype(jnp.float32) @ rhs[e].astype(jnp.float32)
        out = out + jnp.where((seg == e)[:, None], prod, 0.0)
    if kwargs.get("fuse_act") == "silu":
        n = out.shape[-1] // 2
        out = jax.nn.silu(out[:, :n]) * out[:, n:]
    return out.astype(kwargs.get("preferred_element_type") or lhs.dtype)


def _ref_ragged_gather(x, indices, starts, counts, *args, **kwargs):
    rows = jnp.arange(indices.shape[0])
    valid = (rows >= starts[0]) & (rows < starts[0] + counts[0])
    return jnp.where(valid[:, None], x[indices], 0).astype(x.dtype)


def _ref_gather_reduce(x, indices, weights, valid, reduce_group_size, *a, **k):
    rows = x[indices].astype(jnp.float32)
    rows = rows * (weights * valid)[:, None].astype(jnp.float32)
    return rows.reshape(-1, reduce_group_size, x.shape[-1]).sum(1)


@pytest.fixture
def refs(monkeypatch):
    """jnp references for the TPU-only kernels; counts ragged-gather calls."""
    calls = {"ragged_gather": 0}

    def gather(*args, **kwargs):
        calls["ragged_gather"] += 1
        return _ref_ragged_gather(*args, **kwargs)

    monkeypatch.setattr(fmg, "gmm_wrapper", _ref_gmm)
    monkeypatch.setattr(fmg, "ragged_gather_v2", gather)
    monkeypatch.setattr(fmg, "ragged_gather_reduce_v5", _ref_gather_reduce)
    monkeypatch.setattr(fmg, "ragged_gather_reduce_v6", _ref_gather_reduce)
    monkeypatch.setattr(fmg, "_use_sc_combine_v5", lambda x, idx, topk: True)
    monkeypatch.setattr(
        mtd,
        "router_select_bias",
        functools.partial(router_topk.select_bias, interpret=True),
    )
    return calls


@pytest.fixture
def table(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    layers = {}
    for layer in (LAYER, OTHER_LAYER):
        perm = rng.permutation(E)
        layers[str(layer)] = [
            perm[r * SLOTS : (r + 1) * SLOTS].tolist() for r in range(RANKS)
        ]
    path = tmp_path / "placement.json"
    path.write_text(json.dumps({"shards": RANKS, "experts": E, "layers": layers}))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
    ep._load.cache_clear()
    table = ep.placement()
    # Both layers' experts placed in this process, as building them does.
    monkeypatch.setattr(
        ep,
        "_LOCAL_IDS",
        {layer: frozenset(table.cores[layer][0]) for layer in table.cores},
    )
    monkeypatch.setattr(ep, "_ROUTED", set())
    yield table
    ep._load.cache_clear()


_STATICS = mtd._OpStatics(
    topk=TOPK,
    routed_scale=SCALE,
    activation="silu",
    use_sparse_core=True,
    onehot_threshold=THRESH,
    use_ep=True,
    router_bias=True,
)


def _psum_path(x, router_w, bias, w13, w2, is_pad):
    """moe_tp_decode's psum path: _route, fused_moe_func per rank, summed."""
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ router_w.T)
    w, ids = mtd._route(x, scores, bias, is_pad, _STATICS)
    return _grouped(x, w, ids, w13, w2, lambda r: r * SLOTS), ids


def _grouped(x, w, ids, w13, w2, start):
    """fused_moe_func per rank with experts_start = start(rank), summed."""
    total = jnp.zeros(x.shape, jnp.float32)
    for r in range(RANKS):
        part = fmg.fused_moe_func(
            x,
            w13[r],
            w2[r],
            None,
            None,
            None,
            None,
            w,
            ids,
            jnp.asarray(start(r), jnp.int32),
            topk=TOPK,
            activation="silu",
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=THRESH,
            skip_padded_tokens=True,
            onehot_sortfree=True,
        )
        total = total + part.astype(jnp.float32)
    return total * SCALE


def _kernel_path(x, router_w, bias, w13, w2, is_pad):
    """The fused kernel's math: its fast routing (kernel._route_fast), or
    kernel._route when the fast one flags a pick outside its window, as the
    kernel falls back; then per rank the vector compaction
    (kernel._compact_vector) of the experts picked in the rank's window
    16r .. 16r + 15, each weighted by routing column 16r + e."""
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ router_w.T)
    n_valid = x.shape[0] - int(np.asarray(is_pad).sum())
    weights, flag = mtk._route_fast(scores, bias[None, :], n_valid, topk=TOPK)
    exact = mtk._route(scores, bias[None, :], n_valid, topk=TOPK)
    if float(np.asarray(flag).max()) == 0.0:
        np.testing.assert_array_equal(np.asarray(weights), np.asarray(exact))
    else:
        weights = exact
    xf = x.astype(jnp.float32)
    total = jnp.zeros(x.shape, jnp.float32)
    for r in range(RANKS):
        ids, count = mtk._compact_vector(weights, r * SLOTS, SLOTS)
        active = np.asarray(ids)[0, : int(np.asarray(count)[0, 0])].astype(int)
        for e in active:
            gate_up = xf @ w13[r, e].astype(jnp.float32)
            act = jax.nn.silu(gate_up[:, :INTER]) * gate_up[:, INTER:]
            down = act @ w2[r, e].astype(jnp.float32)
            total = total + weights[:, r * SLOTS + e][:, None] * down
    return total * SCALE, jnp.argsort(-weights, axis=1)[:, :TOPK]


def _placed_experts(table, layer, w_all):
    """Each rank's slots filled through the expert map, as the loader does."""
    ranks = []
    for r in range(RANKS):
        slot_expert = [0] * table.slots
        for expert, slot in enumerate(ep.expert_map(table, layer, r)):
            if slot >= 0:
                slot_expert[slot] = expert
        ranks.append(w_all[jnp.asarray(slot_expert)])
    return jnp.stack(ranks)


def _router(layer, w, b):
    gate = ep.permute_router_rows(
        f"model.layers.{layer}.mlp.gate.weight", torch.from_numpy(np.array(w))
    )
    bias = ep.permute_router_rows(
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
        torch.from_numpy(np.array(b)),
    )
    return jnp.asarray(gate.numpy()), jnp.asarray(bias.numpy())


def _same(a, b, valid):
    a = np.asarray(a, np.float64)[valid].ravel()
    b = np.asarray(b, np.float64)[valid].ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos >= 0.9999 and np.max(np.abs(a - b)) <= 0.02 * np.max(np.abs(b))


@pytest.mark.parametrize(
    ("tokens", "path", "bias_scale"),
    [
        (16, "kernel", 0.01),  # biased scores close together: fast routing
        (64, "kernel", 0.5),  # a pick outside the fast window: exact routing
        (64, "psum", 0.5),  # one-hot sort-free
        (256, "psum", 0.5),  # one-hot single sort
        (1024, "psum", 0.5),  # SparseCore ragged dispatch and combine
    ],
)
def test_placed_layer_matches_the_unplaced_layer(table, refs, tokens, path, bias_scale):
    rng = np.random.default_rng(tokens)
    router_w = jnp.asarray(rng.standard_normal((E, HIDDEN)) * 0.05, jnp.float32)
    bias = jnp.asarray(rng.standard_normal(E) * bias_scale, jnp.float32)
    w13 = jnp.asarray(rng.standard_normal((E, HIDDEN, 2 * INTER)) * 0.05, jnp.bfloat16)
    w2 = jnp.asarray(rng.standard_normal((E, INTER, HIDDEN)) * 0.05, jnp.bfloat16)
    x = jnp.asarray(rng.standard_normal((tokens, HIDDEN)), jnp.bfloat16)
    is_pad = jnp.asarray(np.arange(tokens) >= tokens - (3 if tokens <= 64 else 8))
    valid = ~np.asarray(is_pad)
    run = _kernel_path if path == "kernel" else _psum_path
    if path == "kernel":
        scores = jax.nn.sigmoid(x.astype(jnp.float32) @ router_w.T)
        _, flag = mtk._route_fast(scores, bias[None, :], int(valid.sum()), topk=TOPK)
        assert (float(np.asarray(flag).max()) == 0.0) == (bias_scale < 0.1)

    linear13 = w13.reshape(RANKS, SLOTS, HIDDEN, 2 * INTER)
    linear2 = w2.reshape(RANKS, SLOTS, INTER, HIDDEN)
    placed13 = _placed_experts(table, LAYER, w13)
    placed2 = _placed_experts(table, LAYER, w2)
    gate, gate_bias = _router(LAYER, router_w, bias)
    order = ep.virtual_order(table, LAYER)
    np.testing.assert_array_equal(np.asarray(gate), np.asarray(router_w)[order])

    base, base_ids = run(x, router_w, bias, linear13, linear2, is_pad)
    placed, placed_ids = run(x, gate, gate_bias, placed13, placed2, is_pad)
    assert _same(placed, base, valid)
    assert np.all(np.asarray(placed)[~valid] == 0)
    # The same experts per token, once the placed ids are mapped back.
    np.testing.assert_array_equal(
        np.sort(np.asarray(base_ids), 1)[valid],
        np.sort(np.asarray(order)[np.asarray(placed_ids)], 1)[valid],
    )
    if path == "psum" and tokens * TOPK > THRESH:
        assert refs["ragged_gather"] > 0

    # Negative controls: each must change the output.
    wrong_gate, wrong_bias = _router(OTHER_LAYER, router_w, bias)
    controls = {
        "another layer's router order": (wrong_gate, wrong_bias, placed13, placed2),
        "router placed, experts linear": (gate, gate_bias, linear13, linear2),
    }
    for name, (g, b, a13, a2) in controls.items():
        out, _ = run(x, g, b, a13, a2, is_pad)
        assert not _same(out, base, valid), name


COPY_SLOTS = SLOTS + 1


@pytest.fixture
def copies_table(tmp_path, monkeypatch):
    """Returns use(hot): LAYER on 16 ranks x (16 primaries + 1 copy), rank r's
    first primary hot[r]; rank r copies hot[r + 1], and the copy takes half of
    that expert's pairs, as a solver would copy the hottest experts."""

    def use(hot):
        rest = [e for e in np.random.default_rng(1).permutation(E) if e not in hot]
        step = SLOTS - 1
        cores = [[hot[r], *rest[r * step : (r + 1) * step]] for r in range(RANKS)]
        cores = [[int(e) for e in c] for c in cores]
        copied = [cores[(r + 1) % RANKS][0] for r in range(RANKS)]
        path = tmp_path / "copies.json"
        path.write_text(
            json.dumps(
                {
                    "shards": RANKS,
                    "experts": E,
                    "layers": {
                        str(LAYER): [c + [copied[r]] for r, c in enumerate(cores)]
                    },
                    "copy_share": {str(LAYER): {str(e): 0.5 for e in copied}},
                }
            )
        )
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
        ep._load.cache_clear()
        table = ep.placement()
        # LAYER placed in this process, as building it does.
        monkeypatch.setattr(ep, "_LOCAL_IDS", {LAYER: frozenset(table.cores[LAYER][0])})
        monkeypatch.setattr(ep, "_ROUTED", set())
        return table

    monkeypatch.setattr(glm, "_glm_moe_dsa", True)
    monkeypatch.setattr(envs, "TPU_MOE_COLLECTION_CHUNK_SIZE", 0, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_FP8_ALL_GATHER", False, raising=False)
    yield use
    ep._load.cache_clear()


def _copy_routed_layer():
    """The routing state install_copy_routing registers (the same on every
    rank but for the slot start, which the arms below pass themselves)."""
    layer = torch.nn.Module()
    layer.layer_name = f"model.layers.{LAYER}.mlp.experts"
    layer.w13_weight = torch.zeros(1)
    parallel = SimpleNamespace(use_ep=True, ep_rank=0, dp_size=1, pcp_size=1)
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel)
    assert glm.install_copy_routing(layer)
    return layer


@pytest.mark.parametrize("tokens", [64, 256, 1024])
def test_copy_slots_keep_the_unplaced_output(copies_table, refs, tokens):
    rng = np.random.default_rng(tokens)
    router_w = jnp.asarray(rng.standard_normal((E, HIDDEN)) * 0.05, jnp.float32)
    bias = jnp.asarray(rng.standard_normal(E) * 0.5, jnp.float32)
    table = copies_table(np.argsort(-np.asarray(bias))[:RANKS].tolist())
    w13 = jnp.asarray(rng.standard_normal((E, HIDDEN, 2 * INTER)) * 0.05, jnp.bfloat16)
    w2 = jnp.asarray(rng.standard_normal((E, INTER, HIDDEN)) * 0.05, jnp.bfloat16)
    x = jnp.asarray(rng.standard_normal((tokens, HIDDEN)), jnp.bfloat16)
    is_pad = jnp.asarray(np.arange(tokens) >= tokens - 8)
    valid = ~np.asarray(is_pad)

    def route(gate, gate_bias):
        scores = jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)
        return mtd._route(x, scores, gate_bias, is_pad, _STATICS)

    w, ids = route(router_w, bias)
    linear13 = w13.reshape(RANKS, SLOTS, HIDDEN, 2 * INTER)
    linear2 = w2.reshape(RANKS, SLOTS, INTER, HIDDEN)
    base = _grouped(x, w, ids, linear13, linear2, lambda r: r * SLOTS)

    placed13 = _placed_experts(table, LAYER, w13)
    placed2 = _placed_experts(table, LAYER, w2)
    assert placed13.shape[1] == COPY_SLOTS
    pw, rows = route(*_router(LAYER, router_w, bias))
    layer = _copy_routed_layer()
    slot_ids = jnp.asarray(
        glm.route_to_copies(layer, torch.from_numpy(np.array(rows))).numpy()
    )
    primary = (rows // SLOTS) * COPY_SLOTS + rows % SLOTS
    # Some pairs, not all of the copied experts', go to the copies.
    moved = np.asarray(slot_ids != primary)[valid]
    copied_rows = np.isin(np.asarray(rows), np.asarray(layer._copy_rows))[valid]
    assert 0 < moved.sum() < copied_rows.sum()

    def run(ids_, start):
        return _grouped(x, pw, ids_, placed13, placed2, start)

    # Slot ids from each rank's first slot, over every slot.
    served = run(slot_ids, lambda r: r * COPY_SLOTS)
    assert _same(served, base, valid)

    # Negative controls: each must change the output.
    controls = {
        # Rank r's copy slot also serves rank r + 1's first primary.
        "router-row ids": run(rows, lambda r: r * SLOTS),
        "slot ids from the primaries' start": run(slot_ids, lambda r: r * SLOTS),
    }
    for name, out in controls.items():
        assert not _same(out, base, valid), name
