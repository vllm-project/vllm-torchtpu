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
"""CPU tests for the DeepSeek-V4 megakernel serving adapter.

No TPU and no runner: the kernel is a trivial JAX function that echoes its
operands into the hidden output, and the torch_tpu export is replaced by a
CPU call, so eligibility, bucket choice, the DP all-or-nothing decision, A/B
alternation, cache binding and operand assembly are checked end to end.
"""

from __future__ import annotations

import dataclasses
import itertools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA

from .runner_binding import runner_bound_model

NS = types.SimpleNamespace
_op_ids = itertools.count()


# ------------------------------------------------------------ pure policy


def test_buckets_parse_select_and_reachable_rows():
    b = VA.parse_row_buckets("32, 16,64,16", q=7, reqs=4)
    assert [x.rows for x in b] == [16, 32, 64]
    assert {x.reqs for x in b} == {4} and {x.own_rows for x in b} == {28}
    # buckets are global rows over the DP group
    assert VA.select_bucket(7, b).rows == 16
    assert VA.select_bucket(16, b).rows == 16
    assert VA.select_bucket(21, b).rows == 32
    assert VA.select_bucket(63, b).rows == 64
    assert VA.select_bucket(70, b) is None
    with pytest.raises(VA.Unsupported):
        VA.parse_row_buckets("4,8", q=7, reqs=4)
    with pytest.raises(VA.Unsupported):
        VA.parse_row_buckets(" ", q=1, reqs=4)
    # this rank's 1..4 requests of 7 rows -> 7/14/21/28 rows padded by the runner
    assert VA.reachable_out_rows(b[1], [8, 16, 32, 64]) == [8, 16, 32]
    assert VA.reachable_out_rows(b[0], [16, 64]) == [16, 64]


def _classify(**kw):
    args = dict(
        num_reqs=3,
        num_decode=0,
        num_windowed=3,
        num_static_verify=3,
        num_tokens=21,
        num_chunks=1,
        has_encoder_inputs=False,
        q=7,
        buckets=VA.parse_row_buckets("16,32,64", 7, reqs=4),
        max_reqs=4,
    )
    args.update(kw)
    return VA.classify_local_step(**args)


def test_classify_local_step():
    # 3 requests of 7 rows occupy 3 global row blocks of 8
    assert _classify() == VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 24)
    assert _classify(num_reqs=0, num_tokens=0).kind == VA.IDLE
    assert _classify(num_windowed=2).reason == "stock_prefill"
    assert _classify(has_encoder_inputs=True).reason == "stock_encoder"
    assert _classify(num_chunks=2).reason == "stock_chunks"
    # a 1-token decode among verify windows, or a short verify window
    assert _classify(num_decode=1, num_static_verify=2, num_tokens=15).reason == (
        "stock_mixed_qlen"
    )
    assert _classify(num_static_verify=2, num_tokens=19).reason == "stock_mixed_qlen"
    # more requests than this rank's slots
    five = dict(num_reqs=5, num_windowed=5, num_static_verify=5, num_tokens=35)
    assert _classify(**five).reason == "stock_own_reqs"
    assert _classify(**five, max_reqs=5).kind == VA.ELIGIBLE
    assert _classify(
        num_reqs=10, num_windowed=10, num_static_verify=10, num_tokens=70, max_reqs=10
    ).reason == ("stock_rows80")
    # no speculative decoding: all 1-token decodes
    plain = dict(
        q=1, buckets=VA.parse_row_buckets("8,16,32", 1, reqs=4), num_static_verify=0
    )
    assert _classify(num_decode=3, num_tokens=3, **plain).rows == 24
    assert _classify(num_decode=2, num_tokens=4, **plain).reason == "stock_mixed_qlen"
    with pytest.raises(AssertionError):
        _classify(num_tokens=20)


def test_classify_context_guard():
    """A request longer than every bucket's max context goes stock."""
    b128 = VA.parse_row_buckets("16,32,64", 7, reqs=4, max_ctx=131072)
    assert _classify(max_seq_len=131072, buckets=b128).kind == VA.ELIGIBLE
    assert _classify(max_seq_len=131072, buckets=b128).ctx == 131072
    assert _classify(max_seq_len=131073, buckets=b128).reason == "stock_ctx"
    assert _classify(max_seq_len=180000, buckets=b128).reason == "stock_ctx"
    # the earlier reasons still win (prefill steps are never eligible)
    assert _classify(num_windowed=2, max_seq_len=180000, buckets=b128).reason == (
        "stock_prefill"
    )
    # per-bucket contexts: 21 rows at 200k fit T=32 (1M), not T=16/64 (128k)
    mixed = VA.parse_row_buckets("16:131072,32,64:131072", 7, reqs=4, max_ctx=1 << 20)
    assert [b.max_ctx for b in mixed] == [131072, 1 << 20, 131072]
    assert _classify(max_seq_len=200000, buckets=mixed).kind == VA.ELIGIBLE
    five = dict(num_reqs=5, num_windowed=5, num_static_verify=5, num_tokens=35)
    assert _classify(**five, max_reqs=5, max_seq_len=200000, buckets=mixed).reason == (
        "stock_rows40"
    )


def test_bucket_ctx_selects_the_smallest_serving_bucket():
    b = VA.parse_row_buckets("16,32,64:131072", 7, reqs=4, max_ctx=1 << 20)
    assert VA.select_bucket(40, b, ctx=100000).rows == 64
    assert VA.select_bucket(40, b, ctx=200000) is None
    assert VA.select_bucket(20, b, ctx=200000).rows == 32
    ok = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 21, 200000)
    # the group's 21 + 21 rows at 200k: T=64 serves only 128k
    long_ctx = _reduced([0, 200000, 21, 21])
    assert VA.decide_step(ok, long_ctx, b) == ("stock_ctx_rows42", None)
    assert VA.decide_step(ok, _reduced([0, 100000, 21, 21]), b)[1] == b[2]


def _reduced(entries, max_tokens=None, max_chunks=1, any_prefill=False):
    """entries as dp_entries: veto, longest context, rows per rank."""
    rows = entries[2:]
    return VA.DPReduced(
        max_tokens=max(rows) if max_tokens is None else max_tokens,
        max_chunks=max_chunks,
        any_prefill=any_prefill,
        any_veto=bool(entries[0]),
        total_rows=sum(rows),
        max_ctx=entries[1],
    )


def test_decide_step_tp_and_dp():
    buckets = VA.parse_row_buckets("16,32,64", 7, reqs=4)
    ok = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 21)
    idle = VA.LocalStep(VA.IDLE, VA.IDLE)
    pre = VA.LocalStep(VA.STOCK, "stock_prefill")
    assert VA.decide_step(ok, None, buckets) == ("mk_q7_t32", buckets[1])
    assert VA.decide_step(pre, None, buckets) == ("stock_prefill", None)
    # 21 + 14 + 7 rows on three ranks, one idle: T holds the sum (42 -> 64)
    clean = _reduced([0, 0, 21, 14, 0, 7])
    assert VA.decide_step(ok, clean, buckets)[1] == buckets[2]
    assert VA.decide_step(idle, clean, buckets)[1] == buckets[2]
    assert VA.decide_step(ok, _reduced([0, 0, 7, 7]), buckets)[1] == buckets[0]
    # every rank eligible on its own, but the group's sum fits no bucket
    big = _reduced([0, 0, 28, 28, 21])
    assert VA.decide_step(ok, big, buckets) == ("stock_total_rows77", None)
    veto = _reduced([1, 0, 21, 0], max_tokens=900, any_prefill=True)
    assert VA.decide_step(ok, veto, buckets) == ("stock_dp_prefill", None)
    assert VA.decide_step(pre, veto, buckets) == ("stock_prefill", None)
    all_idle = _reduced([0, 0, 0, 0], max_chunks=0)
    assert VA.decide_step(idle, all_idle, buckets) == ("stock_dp_idle", None)


def test_dp_entries_carry_each_ranks_rows_in_its_slot():
    ok = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 14, 5000)
    assert VA.dp_entries(ok, 2, 4) == [0, 5000, 0, 0, 14, 0]
    assert VA.dp_entries(None, 1, 3) == [0, 0, 0, 0, 0]
    assert VA.dp_entries(VA.LocalStep(VA.STOCK, "stock_prefill"), 0, 2) == [
        1, 0, 0, 0,
    ]  # fmt: skip
    # the element-wise MAX over ranks is every rank's rows, the longest
    # context, plus any veto
    longer = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 14, 9000)
    votes = [
        VA.dp_entries(ok, 0, 3),
        VA.dp_entries(None, 1, 3),
        VA.dp_entries(longer, 2, 3),
    ]
    reduced = [max(col) for col in zip(*votes)]
    assert reduced == [0, 9000, 14, 0, 14]


def test_ab_calibrator_alternates_per_key_and_logs_quantiles():
    ab = VA.MKABCalibrator(enabled=True, period_s=10.0)
    picks = [ab.choose(k, True) for k in ("mk_a", "mk_a", "mk_b", "mk_a", "mk_b")]
    assert picks == [
        (True, "mk_a"),
        (False, "ab_stock_a"),
        (True, "mk_b"),
        (True, "mk_a"),
        (False, "ab_stock_b"),
    ]
    assert ab.choose("stock_prefill", False) == (False, "stock_prefill")
    assert ab.counts == {"mk_a": 3, "mk_b": 2}
    ab.reset()
    t0 = ab._t0
    for i in range(20):
        ab.choose("mk_a", True)
        assert ab.record(float(i), now=t0 + 1.0) is None
    elapsed, win, cum = ab.record(100.0, now=t0 + 11.0)
    assert elapsed == pytest.approx(11.0)
    import json

    cum = json.loads(cum)["mk_a"]
    # even samples 0, 2, .., 18 on the megakernel, odd ones and 100.0 on stock
    assert cum["mk"] == {"n": 10, "mean": 9.0, "p10": 1.8, "p50": 9.0, "p90": 16.2}
    assert cum["stock"]["n"] == 11 and cum["stock"]["p50"] == 11.0
    assert json.loads(win) == {"mk_a": cum}
    assert ab.summary_dict(window=True) == {}
    off = VA.MKABCalibrator(enabled=False, period_s=1.0)
    assert off.choose("mk_a", True) == (True, "mk_a")
    assert off.record(1.0) is None


# ------------------------------------------------- fake model and caches

PAGES = 4


def make_fake_layers(device="cpu"):
    """Four decoder layers (HCA, CSA, HCA, CSA) with the runner's overlays:
    SWA caches and HCA states on CSA NoPE arrays, CSA states bound to their
    layer's (nope, rope) pair, indexer states on the indexer array."""

    def _arr(shape, dtype, fill):
        return torch.full(shape, fill, dtype=dtype).to(device)

    csa_nope = [_arr((PAGES, 8, 128), torch.int32, 10 + i) for i in range(2)]
    csa_rope = [_arr((PAGES, 2, 128), torch.int32, 20 + i) for i in range(2)]
    idx_k = [_arr((PAGES, 2, 4, 256), torch.uint8, 30 + i) for i in range(2)]
    hca = [_arr((PAGES, 4, 4, 128), torch.uint8, 40 + i) for i in range(2)]
    groups: dict[str, int] = {}

    def holder(name, tensor, gid):
        groups[name] = gid
        return NS(prefix=name, kv_cache=tensor)

    layers = []
    for i, ratio in enumerate((128, 4, 128, 4)):
        p = f"model.layers.{i}.attn"
        swa = holder(f"{p}.swa_cache", csa_nope[i % 2], 3 + i // 2)
        if ratio == 4:
            c = i // 2
            groups[p] = 0
            pair = (csa_nope[c], csa_rope[c])
            attn = NS(
                compress_ratio=4,
                custom_prefix=p,
                kv_cache=pair,
                swa_cache_layer=swa,
                # the runner binds a CSA state to its layer's own entry
                compressor=NS(
                    state_cache=holder(f"{p}.compressor.state_cache", pair, 5)
                ),
                indexer=NS(
                    k_cache=holder(f"{p}.indexer.k_cache", idx_k[c], 2),
                    compressor=NS(
                        state_cache=holder(
                            f"{p}.indexer.compressor.state_cache", idx_k[c], 7
                        )
                    ),
                ),
            )
        else:
            h = i // 2
            groups[p] = 1
            attn = NS(
                compress_ratio=128,
                custom_prefix=p,
                kv_cache=hca[h],
                swa_cache_layer=swa,
                compressor=NS(
                    state_cache=holder(f"{p}.compressor.state_cache", csa_nope[h], 6)
                ),
                indexer=None,
            )
        layers.append(NS(attn=attn))
    distinct = csa_nope + csa_rope + idx_k + hca
    return layers, groups, distinct


def test_bind_layer_caches_dedups_overlays_and_maps_groups():
    layers, groups, distinct = make_fake_layers()
    bound = VA.bind_layer_caches(layers, groups)
    assert len(bound.arrays) == len(distinct) == 8
    assert {id(t) for t in bound.arrays} == {id(t) for t in distinct}
    assert bound.group_ids == list(range(8))
    arr = {id(t): i for i, t in enumerate(bound.arrays)}
    l0, l1 = bound.layers[0], bound.layers[1]
    assert (l0.kind, l1.kind) == (KI.HCA, KI.CSA)
    assert l0.kv_rope is None and l0.indexer_k is None
    # HCA layer 0: SWA and compressor state both on CSA NoPE array 0
    nope0 = layers[1].attn.kv_cache[0]
    assert l0.swa.array == l0.state.array == arr[id(nope0)]
    assert l0.kv.array == arr[id(layers[0].attn.kv_cache)]
    # CSA layer 1: its state shares its NoPE array, its indexer state the indexer
    assert l1.state.array == l1.kv.array == arr[id(nope0)]
    assert l1.indexer_state.array == l1.indexer_k.array
    assert l1.kv_rope.group == l1.kv.group == 0
    assert l1.indexer_state.group == 7 and l0.state.group == 6
    assert bound.group_layer_names[1] == "model.layers.2.attn"
    with pytest.raises(VA.Unsupported, match="in no KV group"):
        VA.bind_layer_caches(
            layers, {k: v for k, v in groups.items() if "swa" not in k}
        )
    layers[1].attn.kv_cache = layers[1].attn.kv_cache[0]
    with pytest.raises(VA.Unsupported, match="pair"):
        VA.bind_layer_caches(layers, groups)


def runner_for(groups, widths=None):
    """The runner attributes `bind_caches` and `warmup` read."""
    widths = widths or [3 + g % 3 for g in range(len(groups))]
    return NS(
        kv_cache_config=NS(kv_cache_groups=groups),
        input_batch=NS(block_table=[NS(max_num_blocks_per_req=w) for w in widths]),
        num_tokens_paddings=[8, 16, 32],
        num_reqs_max_model_len=4,
    )


def test_runner_binding_structure_binds_hca_csa_and_skips_draft():
    targets, drafts, groups, kv = runner_bound_model()
    csa = targets[1].attn
    # the binding as the runner makes it
    assert isinstance(csa.kv_cache, tuple)
    assert csa.compressor.state_cache.kv_cache is csa.kv_cache
    assert csa.indexer.compressor.state_cache.kv_cache is csa.indexer.k_cache.kv_cache
    assert targets[0].attn.compressor.state_cache.kv_cache is csa.kv_cache[0]
    group_of = {n: g for g, grp in enumerate(groups) for n in grp.layer_names}
    bound = VA.bind_layer_caches(targets, group_of)
    arr = {id(t): i for i, t in enumerate(bound.arrays)}
    l0, l1 = bound.layers
    nope, rope = csa.kv_cache
    assert l1.state.array == l1.kv.array == arr[id(nope)]
    assert l1.kv_rope.array == arr[id(rope)]
    assert l1.indexer_state.array == l1.indexer_k.array
    assert (
        l0.state.array == arr[id(nope)]
        and l0.kv.array == arr[id(kv["model.layers.0.attn"])]
    )
    # exactly the target layers' arrays, each once; the draft's own SWA
    # array (an overflow host here) is not bound
    reach = {id(nope), id(rope), id(kv["model.layers.0.attn"])}
    reach |= {id(csa.indexer.k_cache.kv_cache)}
    reach |= {id(t.attn.swa_cache_layer.kv_cache) for t in targets}
    assert {id(t) for t in bound.arrays} == reach
    assert id(drafts[0].attn.swa_cache_layer.kv_cache) not in reach
    assert len(bound.arrays) == len(reach)


def test_runner_binding_rejects_a_state_off_its_layer():
    targets, _, groups, _ = runner_bound_model()
    group_of = {n: g for g, grp in enumerate(groups) for n in grp.layer_names}
    csa = targets[1].attn
    csa.compressor.state_cache.kv_cache = csa.kv_cache[0]
    with pytest.raises(VA.Unsupported, match="state: expected the layer's"):
        VA.bind_layer_caches(targets, group_of)


# ------------------------------------------------- adapter, end to end

HIDDEN = 128
AUX = (59, 60, 61)
N_WEIGHTS = 70  # with the caches, past the dispatcher's 64-argument cap


class EchoKernel:
    """hidden[r] = (ids, pos, seq_len, block-table sum, rank, weight sum, 0..);
    aux[a] = (a + 2) * hidden; every cache + 1 unless all requests are idle.
    `cache_inplace`: no caches returned (a JAX fn cannot write its inputs)."""

    def __init__(self, cache_inplace=False):
        self.built: list[KI.Bucket] = []
        self.cache_inplace = cache_inplace

    def collect_weights(self, model, spec):
        return [
            torch.full((8,), float(i), dtype=torch.float32) for i in range(N_WEIGHTS)
        ]

    def build_step(self, cfg, bucket):
        self.built.append(bucket)
        M, q = bucket.own_rows, bucket.q
        n_bt = len(cfg.group_widths)
        n_aux = len(cfg.model.aux_layers)

        def fn(meta, ids, pos, seq, *rest):
            bts, weights = rest[:n_bt], rest[n_bt : n_bt + N_WEIGHTS]
            caches = rest[n_bt + N_WEIGHTS :]
            assert ids.shape == (M,) and seq.shape == (bucket.reqs,)
            assert [bt.shape for bt in bts] == [
                (bucket.reqs, w) for w in cfg.group_widths
            ]
            req = jnp.minimum(jnp.arange(M) // q, bucket.reqs - 1)
            in_req = jnp.arange(M) < bucket.reqs * q
            bt_sum = sum(bt.sum(axis=1) for bt in bts)[req] * in_req
            wsum = sum(w.sum() for w in weights)
            cols = [ids, pos, seq[req] * in_req, bt_sum]
            cols += [jnp.full((M,), meta[KI.META_RANK]), jnp.full((M,), wsum)]
            h = jnp.zeros((M, HIDDEN), jnp.float32)
            h = h.at[:, : len(cols)].set(jnp.stack(cols, 1).astype(jnp.float32))
            busy = jnp.any(seq > 0)
            outs = [h] + [h * (a + 2) for a in range(n_aux)]
            if self.cache_inplace:
                return outs
            return outs + [c + busy.astype(c.dtype) for c in caches]

        return fn


def cpu_sharded_jax_callable(
    name,
    fn,
    *,
    mesh,
    input_partition_specs,
    output_partition_specs,
    donate_argnums=None,
):
    """A one-device stand-in for the torch_tpu export."""
    assert len(input_partition_specs) > 64
    assert not donate_argnums or donate_argnums[-1] == len(input_partition_specs) - 1
    cpu_sharded_jax_callable.calls.append(
        (name, donate_argnums, output_partition_specs)
    )
    jitted = jax.jit(fn)

    def call(*tensors):
        assert len(tensors) == len(input_partition_specs)
        from .test_weight_map import j2t, t2j

        outs = jitted(*[t2j(t) for t in tensors])
        return [j2t(o).clone() for o in outs]

    return call


cpu_sharded_jax_callable.calls = []


def build_adapter(
    rank=0,
    mode=VA.MODE_TP,
    ab=False,
    aux=AUX,
    device="cpu",
    compile_fn=lambda f: f,
    cache_inplace=False,
    expert_block=None,
):
    layers, groups, _ = make_fake_layers(device)
    spec = KI.ModelSpec(
        hidden_size=HIDDEN,
        num_layers=4,
        compress_ratios=(128, 4, 128, 4),
        dtype="float32",
        q=3,
        mode=mode,
        axis_name="ep",
        num_ranks=1,
        aux_layers=aux,
    )
    kernel = EchoKernel(cache_inplace)
    mk = VA.DSv4Megakernel(
        spec=spec,
        buckets=VA.parse_row_buckets("16,32", 3, reqs=4),
        kernel=kernel,
        mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("ep",)),
        rank=rank,
        expert_block=rank if expert_block is None else expert_block,
        weights=[w.to(device) for w in kernel.collect_weights(None, spec)],
        decoder_layers=layers,
        device=device,
        ab_enabled=ab,
        compile_fn=compile_fn,
        op_namespace=f"dsv4_mk_test{next(_op_ids)}",
    )
    bound = VA.bind_layer_caches(layers, groups)
    widths = [3 if g % 2 else 5 for g in bound.group_ids]
    mk.bind(bound, widths)
    return mk, layers, kernel


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(VA, "sharded_jax_callable", cpu_sharded_jax_callable)
    monkeypatch.setattr(VA.DSv4Megakernel, "direct_dispatch", False)
    syncs = []
    monkeypatch.setattr(VA, "_sync", lambda t: syncs.append(t))
    build_adapter.syncs = syncs
    return build_adapter


def fake_metadata(mk, num_reqs, padded_reqs, bt_cols, device="cpu"):
    """Per-layer metadata as the runner builds it: flattened block tables of
    padded_reqs x width, seq_lens padded to padded_reqs with STALE values past
    num_reqs, request_distribution [decode_end, verify_end, num_reqs]."""
    seq = torch.full((padded_reqs,), 999, dtype=torch.int32)
    seq[:num_reqs] = torch.arange(num_reqs, dtype=torch.int32) * 10 + 100
    md = {}
    for g, name in enumerate(mk.caches.group_layer_names):
        w = bt_cols[g]
        bt = (
            torch.arange(padded_reqs * w, dtype=torch.int32).reshape(padded_reqs, w) + 1
        )
        md[name] = NS(
            block_tables=bt.reshape(-1).to(device),
            seq_lens=seq.to(device),
            request_distribution=torch.tensor(
                [0, num_reqs, num_reqs], dtype=torch.int32
            ).to(device),
            sequence_layout_kind="all",
        )
    return md


def _never(*a):
    raise AssertionError("stock forward called on a megakernel step")


def test_megakernel_step_assembles_operands_and_updates_caches(adapter, monkeypatch):
    mk, layers, kernel = adapter()
    assert [b.rows for b in kernel.built] == [16, 32]
    widths = mk.cfg.group_widths
    # runner-side widths narrower than the bound max for some groups
    bt_cols = [w - 1 if g == 0 else w for g, w in enumerate(widths)]
    md = fake_metadata(mk, num_reqs=3, padded_reqs=4, bt_cols=bt_cols)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    swa0 = layers[0].attn.swa_cache_layer.kv_cache
    before = [t.clone() for t in mk.caches.arrays]

    local = VA.classify_local_step(
        num_reqs=3,
        num_decode=0,
        num_windowed=3,
        num_static_verify=3,
        num_tokens=9,
        num_chunks=1,
        has_encoder_inputs=False,
        q=3,
        buckets=mk.buckets,
        max_reqs=4,
    )
    d = mk.decide(local)
    # 3 requests, one global block of 8 rows each -> T = 32
    assert d.run and d.bucket.rows == 32 and not d.idle and mk.owns_step()
    padded = 16
    ids = torch.arange(padded, dtype=torch.int32) + 1000
    pos = torch.arange(padded, dtype=torch.int32) + 50
    hidden, aux = mk.forward(_never, ids, pos, None, None)
    assert not mk.owns_step()

    assert hidden.shape == (padded, HIDDEN) and len(aux) == len(AUX)
    h = hidden.numpy()
    rows = 9
    np.testing.assert_array_equal(h[:rows, 0], ids[:rows].numpy())
    np.testing.assert_array_equal(h[:rows, 1], pos[:rows].numpy())
    np.testing.assert_array_equal(h[:rows, 2], np.repeat([100, 110, 120], 3))
    expect_bt = np.zeros(3)
    for g, name in enumerate(mk.caches.group_layer_names):
        bt = md[name].block_tables.reshape(4, -1)[:3].numpy()
        expect_bt += bt.sum(axis=1)
    np.testing.assert_array_equal(h[:rows, 3], np.repeat(expect_bt, 3))
    # inactive rows (incl. the stale request 3): no ids, positions, lengths, pages
    np.testing.assert_array_equal(h[rows:, :4], 0)
    own = 12  # 4 slots x 3 rows; the runner's 16 rows past them are padding
    np.testing.assert_array_equal(h[:own, 5], sum(8.0 * i for i in range(N_WEIGHTS)))
    np.testing.assert_array_equal(h[own:], 0)
    for a, t in enumerate(aux):
        np.testing.assert_array_equal(t.numpy(), h * (a + 2))
    # donated caches copied back into the very tensors the layers hold
    for t, b in zip(mk.caches.arrays, before):
        assert torch.equal(t, b + 1)
    assert layers[0].attn.swa_cache_layer.kv_cache is swa0
    assert torch.equal(swa0, before[0] + 1)

    # the runner padded to more rows than the kernel bucket
    mk.decide(local)
    hidden, aux = mk.forward(
        _never,
        torch.zeros(32, dtype=torch.int32),
        torch.zeros(32, dtype=torch.int32),
        None,
        None,
    )
    assert hidden.shape == (32, HIDDEN) and aux[0].shape == (32, HIDDEN)


def test_inplace_kernel_gets_the_bound_caches_undonated(adapter, monkeypatch):
    """`cache_inplace`: the caches go to the export as plain inputs (no
    donation, no cache outputs), and the step is not wrapped in a torch op or
    compiled graph that would functionalize them."""
    cpu_sharded_jax_callable.calls.clear()
    compiled = []
    mk, _, kernel = adapter(
        cache_inplace=True,
        compile_fn=lambda f: compiled.append(f.__code__.co_name) or f,
    )
    n_head = 1 + len(AUX)
    assert [(d, len(o)) for _, d, o in cpu_sharded_jax_callable.calls] == [
        ((), n_head)
    ] * 2
    assert not [n for n in compiled if "step" in n]
    md = fake_metadata(mk, num_reqs=3, padded_reqs=4, bt_cols=mk.cfg.group_widths)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    before = [t.clone() for t in mk.caches.arrays]
    mk.decide(VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 9))
    ids = torch.arange(16, dtype=torch.int32) + 1000
    hidden, aux = mk.forward(_never, ids, ids, None, None)
    np.testing.assert_array_equal(hidden[:9, 0].numpy(), ids[:9].numpy())
    assert len(aux) == len(AUX)
    for t, b in zip(mk.caches.arrays, before):
        assert torch.equal(t, b)


def test_inplace_refuses_view_caches(adapter):
    mk, layers, _ = adapter(cache_inplace=True)
    caches = mk.caches
    view = caches.arrays[0][1:]
    with pytest.raises(VA.Unsupported, match="is a view"):
        mk.bind(
            dataclasses.replace(caches, arrays=[view, *caches.arrays[1:]]),
            mk.cfg.group_widths,
        )


def test_bind_caches_warmup_and_step_on_the_runner_binding(monkeypatch):
    """`bind_caches` -> `warmup` -> one step, from caches bound as the runner
    binds them (`runner_bound_model`), with the target layers HCA, CSA and a
    draft layer present in the KV groups."""
    monkeypatch.setattr(VA, "sharded_jax_callable", cpu_sharded_jax_callable)
    monkeypatch.setattr(VA.DSv4Megakernel, "direct_dispatch", False)
    synced = []
    monkeypatch.setattr(VA, "_sync", lambda t: synced.append(t))
    targets, _, groups, _ = runner_bound_model()
    spec = KI.ModelSpec(hidden_size=HIDDEN, num_layers=2, compress_ratios=(128, 4),
                        dtype="float32", q=3, mode=VA.MODE_TP, axis_name="ep",
                        num_ranks=1, aux_layers=(1, 2))  # fmt: skip
    kernel = EchoKernel()
    mk = VA.DSv4Megakernel(
        spec=spec,
        buckets=VA.parse_row_buckets("16,32", 3, reqs=4),
        kernel=kernel,
        mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("ep",)),
        rank=0,
        expert_block=0,
        weights=kernel.collect_weights(None, spec),
        decoder_layers=targets,
        device="cpu",
        compile_fn=lambda f: f,
        op_namespace=f"dsv4_mk_test{next(_op_ids)}",
    )
    runner = runner_for(groups)
    mk.bind_caches(runner)
    assert mk.cfg.swa_page == 128
    assert list(mk.cfg.group_widths) == [
        runner.input_batch.block_table[g].max_num_blocks_per_req
        for g in mk.caches.group_ids
    ]
    before = [t.clone() for t in mk.caches.arrays]
    mk.warmup(runner)
    assert all(torch.equal(t, b) for t, b in zip(mk.caches.arrays, before))
    # per bucket: the step on prep's outputs and on the idle operands, each
    # synced (torch_tpu compiles at the first sync), then every output fit
    steps = [b for b in kernel.built]
    assert len(synced) >= 2 * len(steps)
    widths = [
        runner.input_batch.block_table[g].max_num_blocks_per_req
        for g in mk.caches.group_ids
    ]
    md = fake_metadata(mk, num_reqs=2, padded_reqs=4, bt_cols=widths)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    mk.decide(VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 6))
    ids = torch.arange(8, dtype=torch.int32) + 100
    hidden, aux = mk.forward(_never, ids, ids, None, None)
    assert hidden.shape == (8, HIDDEN) and len(aux) == 2
    np.testing.assert_array_equal(hidden[:6, 0].numpy(), ids[:6].numpy())
    # the echo kernel adds 1 to every cache when a request is active
    assert all(torch.equal(t, b + 1) for t, b in zip(mk.caches.arrays, before))


def test_bind_refuses_caches_and_weights_off_the_default_layout(adapter, monkeypatch):
    mk, _, _ = adapter()
    caches = mk.caches
    bad_cache, bad_w = caches.arrays[2], mk.weights[5]
    for bad, what in ((bad_cache, "cache array 2"), (bad_w, "weight operand 5")):
        monkeypatch.setattr(VA, "default_layout", lambda t, b=bad: t is not b)
        with pytest.raises(VA.Unsupported, match=f"{what} .* default device layout"):
            mk.bind(caches, mk.cfg.group_widths)
        # still unreported after a sync: a view, copied on every call
        monkeypatch.setattr(
            VA, "default_layout", lambda t, b=bad: None if t is b else True
        )
        with pytest.raises(VA.Unsupported, match=f"{what} .* a view"):
            mk.bind(caches, mk.cfg.group_widths)
    # unreported until materialized: fine once it reports the default
    seen = set()

    def first_none(t):
        if id(t) in seen:
            return True
        seen.add(id(t))
        return None

    monkeypatch.setattr(VA, "default_layout", first_none)
    mk.bind(caches, mk.cfg.group_widths)


def test_warmup_runs_idle_and_compiles_every_reachable_shape(adapter):
    compiled = []
    mk, _, _ = adapter(compile_fn=lambda f: compiled.append(f.__code__.co_name) or f)
    before = [t.clone() for t in mk.caches.arrays]
    mk.decide(VA.LocalStep(VA.STOCK, "stock_prefill"))
    mk.warmup(NS(num_tokens_paddings=[8, 16, 32], num_reqs_max_model_len=4))
    for t, b in zip(mk.caches.arrays, before):
        assert torch.equal(t, b)
    # own rows (4 slots x 3 = 12) to the runner's 8 / 16 padded rows, per
    # bucket: fewer rows are views, only padding up to 16 compiles a graph
    assert sorted(n for n in compiled if "fit" in n) == [
        "dsv4_mk_fit_q3_t16_p16",
        "dsv4_mk_fit_q3_t32_p16",
    ]
    assert sorted(n for n in compiled if "prep" in n) == [
        "dsv4_mk_prep_q3_t16",
        "dsv4_mk_prep_q3_t32",
    ]
    assert mk._step_mix == {} and mk.ab.counts == {}


def test_stock_paths_and_one_shot_decision(adapter):
    mk, _, _ = adapter()
    calls = []

    def stock(*a):
        calls.append(a)
        return "hidden", None

    ids = torch.zeros(16, dtype=torch.int32)
    # no decision (warmup, precompile, dummy runs outside a step): stock
    assert mk.forward(stock, ids, ids, None, None) == ("hidden", None)
    mk.decide(VA.LocalStep(VA.STOCK, "stock_prefill"))
    assert not mk.owns_step()
    assert mk.forward(stock, ids, ids, None, None) == ("hidden", None)
    assert len(calls) == 2
    assert mk._step_mix == {"stock_prefill": 1}


def test_last_run_names_only_the_forward_that_ran_the_megakernel(adapter):
    """The runner (logits pairing) and the drafter read `last_run` off the
    forward just made: a megakernel step sets it, the next forward without a
    decision (idle-pairing chunks, warmups) or with a stock one clears it."""
    mk = adapter(rank=1, mode=VA.MODE_DP)[0]
    ids = torch.zeros(16, dtype=torch.int32)

    def stock(*a):
        return "hidden", None

    d = mk.decide_dp(None, max_tokens=6, max_chunks=1, any_prefill=False,
                     entries=[0, 0, 6, 0])  # fmt: skip
    assert d.run and d.idle
    mk.forward(_never, ids, ids, None, None)
    assert mk.last_run == d.bucket
    mk.forward(stock, ids, ids, None, None)  # a second dummy chunk: stock
    assert mk.last_run is None
    mk.decide_dp(None, max_tokens=6, max_chunks=1, any_prefill=False,
                 entries=[0, 0, 6, 0])  # fmt: skip
    mk.forward(_never, ids, ids, None, None)
    mk.decide(VA.LocalStep(VA.STOCK, "stock_prefill"))
    mk.forward(stock, ids, ids, None, None)
    assert mk.last_run is None


def _classify_inputs(batch_reqs, scheduled_reqs, q=3, computed=50):
    runner = NS(
        input_batch=NS(
            num_reqs=batch_reqs,
            num_computed_tokens_cpu=np.full(max(batch_reqs, 1), computed),
            req_ids=[f"r{i}" for i in range(batch_reqs)],
        ),
        _num_static_verify_reqs=batch_reqs,
    )
    sched = NS(
        num_scheduled_tokens={f"r{i}": q for i in range(scheduled_reqs)},
        total_num_scheduled_tokens=q * scheduled_reqs,
        scheduled_encoder_inputs={},
    )
    return runner, sched


def test_classify_counts_scheduled_requests_and_traces_to_stderr(adapter, capsys):
    mk, _, _ = adapter()
    mk._trace_left = 3
    runner, sched = _classify_inputs(2, 2)
    local = mk.classify(runner, sched, 0, 2, 1)
    assert (local.kind, local.rows) == (VA.ELIGIBLE, 16)
    mk.decide(local)
    # the persistent batch holds a request this step does not schedule
    runner, sched = _classify_inputs(3, 2)
    local = mk.classify(runner, sched, 0, 3, 1)
    assert (local.kind, local.reason) == (VA.STOCK, "stock_unscheduled_reqs")
    mk.decide_dp(None, max_tokens=6, max_chunks=1, any_prefill=False,
                 entries=[0, 60, 16])  # fmt: skip
    mk.decide(VA.LocalStep(VA.STOCK, "stock_prefill"))  # past the trace budget
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 3 and all(
        ln.startswith("[dsv4-megakernel rank 0]") for ln in lines
    )
    assert (
        "batch_reqs=2 scheduled_reqs=2" in lines[0] and "-> mk_q3_t16 run=1" in lines[0]
    )
    assert "step: idle | dp: veto=0 prefill=0 total_rows=16 max_ctx=60" in lines[1]
    assert "-> stock_prefill" in lines[2]


def test_unbound_adapter_never_runs(adapter):
    mk, _, _ = adapter()
    mk.cfg = None
    d = mk.decide(VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 9))
    assert not d.run and d.key == "stock_unbound"


def test_dp_all_or_nothing_across_ranks(adapter, monkeypatch):
    """Four ranks: the MAX-reduced votes give every rank the same path and
    bucket; idle ranks run the kernel with every request inactive."""
    ranks = [adapter(rank=r, mode=VA.MODE_DP)[0] for r in range(4)]
    md = fake_metadata(
        ranks[0], num_reqs=2, padded_reqs=4, bt_cols=ranks[0].cfg.group_widths
    )
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)

    def step(locals_, tokens):
        mine = [
            mk.dp_entries(loc, i, 4) for i, (mk, loc) in enumerate(zip(ranks, locals_))
        ]
        reduced = dict(
            max_tokens=max(tokens),
            max_chunks=max(int(t > 0) for t in tokens),
            any_prefill=any(
                loc is not None and loc.reason == "stock_prefill" for loc in locals_
            ),
            entries=[max(col) for col in zip(*mine)],  # the MAX all-reduce
        )
        return [mk.decide_dp(loc, **reduced) for mk, loc in zip(ranks, locals_)]

    ok6 = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 6)
    ok12 = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 12)
    pre = VA.LocalStep(VA.STOCK, "stock_prefill")

    ds = step([ok6, None, ok12, None], [6, 0, 12, 0])
    # T holds the group's 6 + 12 rows
    assert all(d.run for d in ds) and {d.bucket.rows for d in ds} == {32}
    assert [d.idle for d in ds] == [False, True, False, True]
    before = [t.clone() for t in ranks[1].caches.arrays]
    hidden, aux = ranks[1].forward(
        _never,
        torch.zeros(16, dtype=torch.int32),
        torch.zeros(16, dtype=torch.int32),
        None,
        None,
    )
    # idle: nothing read (zero operands), nothing written; rank from meta
    np.testing.assert_array_equal(hidden.numpy()[:, :4], 0)
    np.testing.assert_array_equal(hidden.numpy()[:12, 4], 1)  # 12 own rows
    for t, b in zip(ranks[1].caches.arrays, before):
        assert torch.equal(t, b)
    busy = ranks[0].forward(
        _never,
        torch.ones(16, dtype=torch.int32),
        torch.ones(16, dtype=torch.int32),
        None,
        None,
    )[0]
    np.testing.assert_array_equal(busy.numpy()[:6, 2], [100] * 3 + [110] * 3)

    # one prefill anywhere sends every rank to stock
    ds = step([ok6, pre, None, ok6], [6, 900, 0, 6])
    assert not any(d.run for d in ds)
    assert [d.key for d in ds] == [
        "stock_dp_prefill",
        "stock_prefill",
        "stock_dp_prefill",
        "stock_dp_prefill",
    ]


def test_dp_ab_alternation_stays_in_lockstep(adapter):
    ranks = [adapter(rank=r, mode=VA.MODE_DP, ab=True)[0] for r in range(3)]
    seq = []
    for tokens in ([6, 0, 3], [9, 9, 0], [6, 6, 6], [12, 3, 0], [3, 0, 0]):
        locals_ = [
            VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, t) if t else None for t in tokens
        ]
        entries = [0, 0, *tokens]
        ds = [
            mk.decide_dp(
                loc, max_tokens=max(tokens), max_chunks=1, any_prefill=False,
                entries=entries,
            )
            for mk, loc in zip(ranks, locals_)
        ]  # fmt: skip
        assert len({(d.run, d.key, d.bucket) for d in ds}) == 1
        seq.append(ds[0].key)
    # buckets by the group's total rows: 9, 18, 18, 15, 3 -> T 16, 32, 32, 16, 16
    assert seq == [
        "mk_q3_t16",
        "mk_q3_t32",
        "ab_stock_q3_t32",
        "ab_stock_q3_t16",
        "mk_q3_t16",
    ]


def test_ab_forward_times_both_paths(adapter, monkeypatch):
    mk, _, _ = adapter(ab=True, aux=())
    md = fake_metadata(mk, num_reqs=1, padded_reqs=2, bt_cols=mk.cfg.group_widths)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    one = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 3)
    ids = torch.zeros(8, dtype=torch.int32)
    stock_calls = []

    def stock(*a):
        stock_calls.append(1)
        return torch.zeros(8, HIDDEN), None

    for _ in range(4):
        mk.decide(one)
        hidden, aux = mk.forward(stock, ids, ids, None, None)
        assert aux is None and hidden.shape == (8, HIDDEN)
    assert len(stock_calls) == 2
    assert len(adapter.syncs) == 8  # before and after each timed step
    summary = mk.ab.summary_dict()["mk_q3_t16"]
    assert summary["mk"]["n"] == 2 and summary["stock"]["n"] == 2


def test_maybe_create_is_off_without_the_flag(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_DSV4_MEGAKERNEL", raising=False)
    # never touches the runner
    assert VA.DSv4Megakernel.maybe_create(object()) is None
    monkeypatch.setenv("VLLM_TPU_DSV4_MEGAKERNEL", "1")
    runner = NS(model_config=NS(hf_config=NS(model_type="qwen3")))
    with pytest.raises(VA.Unsupported, match="model_type"):
        VA.DSv4Megakernel.maybe_create(runner)


def test_placeholder_kernel_refuses_to_build():
    k = KI.default_kernel()
    with pytest.raises(NotImplementedError):
        k.collect_weights(None, None)
    with pytest.raises(NotImplementedError):
        k.build_step(None, KI.Bucket(q=7, rows=16))


def _two_steps(mk, md, device):
    """Two consecutive megakernel steps (3 then 2 requests of 3 rows)."""
    outs = []
    for num_reqs in (3, 2):
        for name in md:
            md[name].request_distribution = torch.tensor(
                [0, num_reqs, num_reqs], dtype=torch.int32
            ).to(device)
        rows = 3 * num_reqs
        mk.decide(VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows))
        ids = (torch.arange(16, dtype=torch.int32) + 1000 * num_reqs).to(device)
        pos = (torch.arange(16, dtype=torch.int32) + 7 * num_reqs).to(device)
        hidden, aux = mk.forward(_never, ids, pos, None, None)
        outs.append([hidden.cpu()] + [a.cpu() for a in aux])
    return outs, [t.cpu() for t in mk.caches.arrays]


@pytest.mark.tpu_test
def test_tpu_export_compile_and_donation_match_cpu(monkeypatch):
    """The real path -- torch_tpu export of the shard_map'd kernel, the
    list[Tensor] custom op, torch.compile(backend="tpu") of prep / step /
    fit, cache donation and copy-back -- gives bit-identical outputs and
    caches to the CPU reference path, over two steps and an idle warmup."""
    mk_tpu, _, _ = build_adapter(
        device="tpu", compile_fn=VA.DSv4Megakernel._default_compile
    )
    widths = mk_tpu.cfg.group_widths
    bt_cols = [w - 1 if g == 0 else w for g, w in enumerate(widths)]
    md_tpu = fake_metadata(mk_tpu, 3, 4, bt_cols, device="tpu")
    caches_before = [t.cpu() for t in mk_tpu.caches.arrays]
    mk_tpu.warmup(NS(num_tokens_paddings=[8, 16, 32], num_reqs_max_model_len=4))
    for t, b in zip(mk_tpu.caches.arrays, caches_before):
        assert torch.equal(t.cpu(), b)  # idle warmup writes nothing
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md_tpu)
    got, got_caches = _two_steps(mk_tpu, md_tpu, "tpu")

    monkeypatch.setattr(VA, "sharded_jax_callable", cpu_sharded_jax_callable)
    monkeypatch.setattr(VA.DSv4Megakernel, "direct_dispatch", False)
    mk_cpu, _, _ = build_adapter()
    md_cpu = fake_metadata(mk_cpu, 3, 4, bt_cols)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md_cpu)
    want, want_caches = _two_steps(mk_cpu, md_cpu, "cpu")

    for g_step, w_step in zip(got, want):
        for g, w in zip(g_step, w_step):
            torch.testing.assert_close(g, w, rtol=0, atol=0)
    for g, w, b in zip(got_caches, want_caches, caches_before):
        assert torch.equal(g, w) and torch.equal(g, b + 2)


def test_expert_block_is_the_ep_rank_not_the_mesh_index(monkeypatch):
    """tpu7x hosts put local ranks 0..7 on devices [0, 1, 4, 5, 6, 7, 2, 3],
    so the device-id-ordered mesh index 6 serves EP rank 4's experts."""
    import sys
    import types

    order = (0, 1, 6, 7, 2, 3, 4, 5)  # EP rank at each mesh index
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("ep",))
    ep_mesh = types.ModuleType("vllm_torchtpu.distributed.ep_mesh")
    ep_mesh.build_ep_mesh = lambda: mesh
    ep_mesh.ep_mesh_index = lambda: 6
    ep_mesh.ep_rank_order = lambda: order
    monkeypatch.setitem(sys.modules, ep_mesh.__name__, ep_mesh)
    assert VA._mesh_and_rank(VA.MODE_DP) == (mesh, 6, 4)

    monkeypatch.setattr(VA, "sharded_jax_callable", cpu_sharded_jax_callable)
    monkeypatch.setattr(VA.DSv4Megakernel, "direct_dispatch", False)
    mk = build_adapter(rank=6, expert_block=4)[0]
    assert int(mk.meta[KI.META_RANK]) == 6
    assert int(mk.meta[KI.META_EBLK]) == 4


# ------------------------------------------------ vLLM TP x DP (TP8 x DP2)


def test_engine_mesh_order_is_engine_major():
    """TP rank j of engine g sits at mesh index g*8 + j whatever the device ids
    (tpu7x: local ranks on devices [0, 1, 4, 5, 6, 7, 2, 3] per host)."""
    host = [0, 1, 4, 5, 6, 7, 2, 3]
    members = [
        VA.GroupMember(key=g * 8 + j, device_id=8 * g + host[j], ep_rank=g * 8 + j)
        for g in (1, 0)
        for j in reversed(range(8))
    ]
    assert VA.engine_mesh_order(members) == host + [8 + d for d in host]
    dup = members[:-1] + [dataclasses.replace(members[-1], key=members[0].key)]
    with pytest.raises(VA.Unsupported, match="keys"):
        VA.engine_mesh_order(dup)
    same_dev = members[:-1] + [
        dataclasses.replace(members[-1], device_id=members[0].device_id)
    ]
    with pytest.raises(VA.Unsupported, match="device id"):
        VA.engine_mesh_order(same_dev)


def test_engine_mesh_and_rank_from_the_vllm_groups(monkeypatch):
    import sys

    host = [0, 1, 4, 5, 6, 7, 2, 3]
    me_g, me_j = 1, 2
    ep = NS(world_size=16, rank_in_group=me_g * 8 + me_j, cpu_group="cpu")

    def fake(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    ps = fake(
        "vllm.distributed.parallel_state",
        get_dp_group=lambda: NS(world_size=2, rank_in_group=me_g),
        get_tp_group=lambda: NS(world_size=8, rank_in_group=me_j),
    )
    fake("vllm.distributed", parallel_state=ps)
    fake("vllm_torchtpu.distributed.ep_mesh", EP_AXIS_NAME="d", get_ep_group=lambda: ep)
    fake(
        "vllm_torchtpu.distributed.mesh_utils",
        _get_tpu_global_device_id=lambda: 8 + host[me_j],
    )
    dev = [NS(id=i) for i in range(16)]
    fake(
        "torch_tpu._internal.pallas",
        get_pallas_mesh=lambda axis_names: NS(devices=np.array(dev)),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)

    def gather(out, mine, group):
        assert group == "cpu" and mine == VA.GroupMember(10, 8 + host[2], 10)
        for g in range(2):
            for j in range(8):
                out[g * 8 + j] = VA.GroupMember(g * 8 + j, 8 * g + host[j], g * 8 + j)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(jax.sharding, "Mesh", lambda d, axis_names: (d, axis_names))
    (devices, axes), rank, eblk = VA._engine_mesh_and_rank(8)
    assert (rank, eblk, axes) == (10, 10, ("d",))
    assert [d.id for d in devices] == host + [8 + d for d in host]
    ep.world_size = 8
    with pytest.raises(VA.Unsupported, match="EP 8 is not TP 8 x DP 2"):
        VA._engine_mesh_and_rank(8)


def test_two_engine_vote_sums_engine_rows_and_agrees_on_every_rank():
    """16 ranks, 2 engines: an engine's 8 ranks classify the same batch, each
    DP group {j, 8 + j} MAX-reduces its pair, and every rank picks the bucket
    of the two engines' summed rows (QP = 8 rows per request)."""
    buckets = VA.parse_row_buckets("16,32,64", 2, reqs=4)

    def vote(per_engine):
        decisions = []
        for j in range(8):
            mine = [VA.dp_entries(per_engine[g], g, 2) for g in range(2)]
            reduced = [max(c) for c in zip(*mine)]
            for g in range(2):
                red = VA.DPReduced(
                    max_tokens=6, max_chunks=1, any_prefill=False,
                    any_veto=bool(reduced[0]), total_rows=sum(reduced[2:]),
                    max_ctx=reduced[1],
                )  # fmt: skip
                loc = per_engine[g] or VA.LocalStep(VA.IDLE, VA.IDLE)
                decisions.append(VA.decide_step(loc, red, buckets))
        # all 16 ranks take the same path and bucket (keys are log labels)
        assert len({b for _, b in decisions}) == 1
        return decisions

    one = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows=8, ctx=100)
    four = VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows=32, ctx=900)
    assert vote([one, None])[0][1].rows == 16  # C=1: the idle engine runs too
    assert vote([four, four])[0][1].rows == 64  # c4: 2 x 4 requests
    pre = VA.LocalStep(VA.STOCK, "stock_prefill")
    assert set(vote([four, pre])) == {("stock_dp_peer", None), ("stock_prefill", None)}


class _SpecKernel:
    cache_inplace = False

    def collect_weights(self, model, spec):
        self.spec = spec
        return [torch.zeros(8)]

    def build_step(self, cfg, bucket):  # pragma: no cover - never bound here
        raise AssertionError


def _tp_runner(tp=8, dp=2, drafter=None):
    layers = [NS(attn=NS(compress_ratio=r)) for r in (128, 4)]
    backbone = NS(
        layers=layers,
        aux_hidden_state_layers=(1, 2),
        embed_tokens=NS(shard_indices=NS(org_vocab_start_index=2 * 16160)),
    )
    hf = NS(
        model_type="deepseek_v4", hidden_size=64, sliding_window=128,
        rms_norm_eps=1e-6, hc_eps=1e-6, vocab_size=129280, num_hash_layers=1,
        n_routed_experts=384,
    )  # fmt: skip
    return NS(
        model_config=NS(hf_config=hf, dtype=torch.bfloat16),
        parallel_config=NS(
            pipeline_parallel_size=1, tensor_parallel_size=tp, data_parallel_size=dp
        ),
        _dp_lockstep_enabled=lambda: True,
        speculative_config=None,
        drafter=drafter,
        model=NS(model=backbone, parameters=lambda: [], buffers=lambda: []),
        device="cpu",
    )


def test_from_runner_tp8_dp2(monkeypatch):
    mesh = NS(devices=np.zeros(16), axis_names=("d",))  # never bound here
    monkeypatch.setattr(VA, "_engine_mesh_and_rank", lambda tp: (mesh, 10, 10))
    monkeypatch.setattr(
        VA, "_mesh_and_rank", lambda mode: pytest.fail("DP16 mesh under TP")
    )
    for k in (
        "VLLM_TPU_DSV4_MK_LOGITS",
        "VLLM_TPU_DSV4_MK_DRAFT",
        "TPU_EXPERT_PLACEMENT",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_DSV4_INDEXER_KV_LAYOUT", "sal")
    kernel = _SpecKernel()
    mk = VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)
    assert kernel.spec.mode == VA.MODE_DP and kernel.spec.attn_tp == 8
    assert kernel.spec.num_ranks == 16 and kernel.spec.dp_groups == 2
    assert int(mk.meta[KI.META_RANK]) == 10 and int(mk.meta[KI.META_EBLK]) == 10
    assert int(mk.meta[KI.META_VSTART]) == 2 * 16160

    monkeypatch.setenv("VLLM_TPU_DSV4_MK_LOGITS", "1")
    with pytest.raises(VA.Unsupported, match="logits mode under vLLM TP"):
        VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)
    monkeypatch.delenv("VLLM_TPU_DSV4_MK_LOGITS")
    # the shipped EP16 placement: a permutation, the kernel reads its tables
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", "dsv4_pro_agentx_ep16")
    VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)
    assert (kernel.spec.moe_placement, kernel.spec.expert_slots) == ("perm", 24)


def test_from_runner_refuses_any_indexer_layout_but_sal(monkeypatch):
    """Both indexer K-cache layouts have the same allocated shape, so only
    this check stops the kernel from reading head-along-sublane records as
    seq-along-lane ones (and writing records the stock path then misreads)."""
    mesh = NS(devices=np.zeros(16), axis_names=("d",))
    monkeypatch.setattr(VA, "_engine_mesh_and_rank", lambda tp: (mesh, 10, 10))
    for k in ("VLLM_TPU_DSV4_MK_LOGITS", "TPU_EXPERT_PLACEMENT"):
        monkeypatch.delenv(k, raising=False)
    kernel = _SpecKernel()
    monkeypatch.delenv("VLLM_TPU_DSV4_INDEXER_KV_LAYOUT", raising=False)
    with pytest.raises(VA.Unsupported, match="INDEXER_KV_LAYOUT='has'.*=sal"):
        VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)  # the env default
    monkeypatch.setenv("VLLM_TPU_DSV4_INDEXER_KV_LAYOUT", "has")
    with pytest.raises(VA.Unsupported, match="seq-along-lane layout only"):
        VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)
    assert not hasattr(kernel, "spec")  # refused before any weight is touched
    monkeypatch.setenv("VLLM_TPU_DSV4_INDEXER_KV_LAYOUT", "sal")
    VA.DSv4Megakernel.from_runner(_tp_runner(), kernel)
    assert kernel.spec.attn_tp == 8


def test_placement_operands_are_this_ep_ranks_table_rows():
    from vllm_torchtpu import expert_placement as XP

    spec = KI.ModelSpec(
        hidden_size=7168, num_layers=3, compress_ratios=(128, 4, 128), dtype="bfloat16",
        q=7, mode="dp", axis_name="d", num_ranks=4, aux_layers=(),
        n_routed_experts=8,
    )  # fmt: skip
    assert VA.placement_operands(spec, 0, "cpu") == (spec, {})  # none set
    rows = ((5, 0), (1, 7), (2, 3), (6, 4))  # layer 1: a permutation
    perm = XP.Placement(experts=8, shards=4, layers={1: rows})
    got, ops = VA.placement_operands(spec, 2, "cpu", perm)
    assert (got.moe_placement, got.expert_slots) == ("perm", 2)
    assert list(ops) == ["moe_held"]
    # layers 0 and 2 unlisted: contiguous blocks; layer 1: rank 2's slots
    assert ops["moe_held"].tolist() == [[4, 5], [2, 3], [4, 5]]
    real = dataclasses.replace(spec, num_ranks=16, n_routed_experts=384)
    man = KI.weight_manifest(
        dataclasses.replace(got, num_ranks=16, n_routed_experts=384), logits=True
    )
    assert [e.name for e in man][-2:] == ["moe_held", "lm_head"]
    assert man[-2].shape == (3, 2) and "moe_held" not in [
        e.name for e in KI.weight_manifest(real)
    ]
    rep = XP.Placement(
        experts=8, shards=4, layers={1: ((5, 0, 1), (1, 7, 6), (2, 3, 0), (6, 4, 5))}
    )
    got, ops = VA.placement_operands(spec, 2, "cpu", rep)
    assert got.moe_placement == "replicas" and list(ops) == ["moe_held", "moe_copy"]
    assert ops["moe_held"][1].tolist() == [2, 3, 0]
    with pytest.raises(VA.Unsupported, match="expert placement for 8 experts over 2"):
        VA.placement_operands(spec, 0, "cpu", XP.Placement(8, 2, {}))


def test_multi_q_buckets_and_step_q():
    b = VA.parse_row_buckets("16,32,64@4:131072", q=7, reqs=4, max_ctx=1 << 20)
    assert [(x.q, x.rows, x.max_ctx) for x in b] == [
        (7, 16, 1 << 20),
        (7, 32, 1 << 20),
        (4, 64, 131072),
    ]
    extra = VA.parse_row_buckets("16,32", q=7, reqs=4, extra_qs=(4,))
    assert [(x.q, x.rows) for x in extra] == [(7, 16), (4, 16), (7, 32), (4, 32)]
    assert VA.select_bucket(24, b, q=7) == b[1]
    assert VA.select_bucket(40, b, q=7) is None
    assert VA.select_bucket(40, b, q=4) == b[2]
    # step_q=4 classifies against q=4 buckets
    local_q4 = _classify(
        num_reqs=4,
        num_windowed=4,
        num_static_verify=0,
        num_tokens=16,
        buckets=b,
        step_q=4,
    )
    assert local_q4 == VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, 32, 0, q=4)
    assert VA.decide_step(local_q4, None, b) == ("mk_q4_t64", b[2])
