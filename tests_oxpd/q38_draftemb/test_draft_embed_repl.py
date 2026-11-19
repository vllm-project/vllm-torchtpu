# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-2 (c): the MTP draft embeds from a replicated table
(VLLM_TPU_DRAFT_EMBED_REPL, default off, compile-hashed).

A. ``replicated_lookup`` on the table ``gather_sharded_weight`` assembles is the TP-summed stock
   lookup, bit for bit, for every id class (owned, other ranks', negative, vocab padding, beyond).
B. Replay (the deciding numerics): the real ``Qwen4ExpMultiTokenPredictor`` (stub decoder layer)
   over 3-pass draft steps, the next pass fed the previous pass's greedy token and multi-stream
   hidden, on all 8 simulated TP ranks. Stock (XLA all-reduce emulated as the rank-order sum) and
   DRAFT-2 (c) give bit-identical embeddings, fused inputs, outputs and draft tokens; so does (c)
   with DRAFT-2 (a) armed, which never calls H8.
C. 0-chip census of the draft step (3 passes): embedding all-reduces 3 -> 0 (XLA, and H8 with
   DRAFT-2 (a) armed), lookups 3 -> 3; by dispatch census of the real forward and by make_fx of
   the traced embed graph.
D. All-rank agreement (fail-stop): every rank of a TP > 1 draft agrees on the flag, on or off;
   a divergent rank stops every rank before anything is assembled; a failed gather stops.
E. Refusals and wiring: unbound table, quantized embedding, TP 1, the bind_shared_selection hook,
   the build-time read and the flag default.
"""

import ast
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from torch import nn

V, H, HC, TP = 61, 8, 4, 8  # 61 ids over 8 ranks of 8 rows: rank 7 owns 5 ids + 3 padding rows
PER = -(-V // TP)
CPU_GROUP = "tp-cpu-group"


# ---------------------------------------------------------------- helpers


def _bits(t):
    """The exact bit pattern (tells -0.0 from +0.0)."""
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
    return t.contiguous().view(view[t.dtype])


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _weight(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(V, H, generator=g) * 0.7).to(torch.bfloat16)
    w[3, 1] = -0.0  # signed zeros: the stock sum and the assembly sum both yield +0.0
    w[V - 1, 0] = -0.0
    return w


def _shard(rank, weight):
    """Rank `rank`'s TPU-patched VocabParallelEmbedding over `weight` (vLLM's layout: PER rows per
    rank, the last rank's padding rows zero-filled as vLLM's weight_loader leaves them)."""
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    import vllm_torchtpu

    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()
    emb = object.__new__(vpe.VocabParallelEmbedding)
    nn.Module.__init__(emb)
    start = rank * PER
    end = min(start + PER, V)
    rows = torch.zeros(PER, weight.shape[1], dtype=weight.dtype)
    rows[: end - start] = weight[start:end]
    emb.tp_size = TP
    emb.org_vocab_size = V
    emb.embedding_dim = weight.shape[1]
    emb.weight = nn.Parameter(rows, requires_grad=False)
    emb.quant_method = NS(embedding=lambda layer, ids: F.embedding(ids, layer.weight))
    emb.shard_indices = NS(
        org_vocab_start_index=start,
        org_vocab_end_index=end,
        num_org_vocab_padding=PER - (end - start),
        added_vocab_start_index=V,
        added_vocab_end_index=V,
    )
    return emb


class StockTP:
    """The stock embedding all-reduce over TP simulated ranks: rank r's partial must be its own
    masked lookup, and the result is the rank-order bf16 sum of every rank's partial for the same
    ids (one owner row plus exact zeros). The ids reach it through a forward pre-hook."""

    def __init__(self, shards):
        self.shards = shards
        self.ids = None
        self.calls = 0
        for s in shards:
            s.register_forward_pre_hook(lambda m, args: setattr(self, "ids", args[0]))

    def all_reduce_for(self, rank):
        def all_reduce(partial):
            parts = [s._tpu_partial(self.ids) for s in self.shards]
            assert bits_equal(partial, parts[rank])
            out = parts[0].clone()
            for p in parts[1:]:
                out = out + p
            self.calls += 1
            return out

        return all_reduce


def _emulate_gather(monkeypatch, shards, rank):
    """`gather_sharded_weight` on rank `rank`: its fp32 CPU-group all-reduce sums every rank's
    rows-into-zeros contribution (gloo's sum; exact, one nonzero per element)."""
    from vllm_torchtpu.spec_decode import utils as su

    contrib = []
    for s in shards:
        si = s.shard_indices
        c = torch.zeros(V, s.embedding_dim, dtype=torch.float32)
        n = si.org_vocab_end_index - si.org_vocab_start_index
        c[si.org_vocab_start_index : si.org_vocab_end_index] = s.weight.data[:n].float()
        contrib.append(c)
    seen = []

    def all_reduce(t, group=None):
        assert group == CPU_GROUP
        assert torch.equal(t, contrib[rank])
        out = contrib[0].clone()
        for c in contrib[1:]:
            out += c
        t.copy_(out)
        seen.append(rank)

    monkeypatch.setattr(su, "get_tp_group", lambda: NS(cpu_group=CPU_GROUP))
    monkeypatch.setattr(su.dist, "all_reduce", all_reduce)
    return seen


def _emulate_agreement(monkeypatch, votes, rank):
    """DRAFT-ASM's real `_gather` over a TP group of len(votes) ranks: the CPU-group all_gather
    returns every rank's (VLLM_TPU_DRAFT_EMBED_REPL, VLLM_TPU_TARGET_EMBED_REPL) vote (an int vote
    means the target flag is 0); this rank must contribute `votes[rank]`."""
    import vllm.distributed.parallel_state as ps

    vecs = [list(v) if isinstance(v, (tuple, list)) else [int(v), 0] for v in votes]
    calls = []

    def all_gather(out, t, group=None):
        assert group == CPU_GROUP and len(out) == len(vecs)
        assert t.tolist() == vecs[rank]
        for o, v in zip(out, vecs):
            o.copy_(torch.tensor(v, dtype=torch.int64))
        calls.append(rank)

    monkeypatch.setattr(ps, "get_tp_group", lambda: NS(world_size=len(votes), cpu_group=CPU_GROUP))
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    return calls


def _holder(shard, repl):
    m = nn.Module()
    m.embed_tokens = shard
    m.embed_repl = repl
    return m


def _bind(holder):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    return mtp.Qwen4ExpMTP.bind_replicated_embedding(NS(model=holder))


# ---------------------------------------------------------------- A. lookup identity

IDS = torch.tensor([0, 3, 7, 8, 15, 33, 55, 56, 60, V - 1, -1, -7, V, V + 2, PER * TP - 1, 999, 3, 0])


@pytest.mark.parametrize("ids_dtype", [torch.int64, torch.int32])
def test_replicated_lookup_is_the_tp_summed_stock_lookup(monkeypatch, ids_dtype):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp
    from vllm_torchtpu.spec_decode.utils import gather_sharded_weight

    weight = _weight()
    shards = [_shard(r, weight) for r in range(TP)]
    ids = IDS.to(ids_dtype)
    stock = shards[0]._tpu_partial(ids)
    for s in shards[1:]:
        stock = stock + s._tpu_partial(ids)
    for rank in (0, 5, TP - 1):
        seen = _emulate_gather(monkeypatch, shards, rank)
        table, org_vocab, dim, _ = gather_sharded_weight(shards[rank])
        assert seen == [rank] and (org_vocab, dim) == (V, H) and table.dtype == torch.bfloat16
        got = mtp.replicated_lookup(table, ids)
        assert bits_equal(got, stock), rank
    # Owned rows are the checkpoint rows (signed zeros summed to +0.0); every other id is zeros.
    valid = (IDS >= 0) & (IDS < V)
    assert torch.equal(stock[valid], weight[IDS[valid]])
    assert bits_equal(stock[~valid], torch.zeros_like(stock[~valid]))
    assert _bits(stock)[1, 1].item() == 0 and _bits(table)[3, 1].item() == 0  # -0.0 -> +0.0 on both


# ---------------------------------------------------------------- B. replay through the real predictor


@pytest.fixture
def cpu_vllm_config():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _predictor(monkeypatch, *, h8=False, repl=False):
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    text = q4exp_text_config(
        hidden_size=H, hc_count=HC, hc_lowrank=4, vocab_size=V, num_hidden_layers=2
    )

    class Layer(nn.Module):
        def __init__(self, cfg, index, prefix, *, layer_type):
            super().__init__()
            self.self_attn = NS(prefix=prefix + ".self_attn")

        def forward(self, hidden, positions, input_ids, starts, ngram, slots, context, valid,
                    main, compressed):
            return (torch.tanh(hidden.float() * 1.5) * 2).to(hidden.dtype)

    monkeypatch.setattr(mtp, "Qwen4ExpDecoderLayer", Layer)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    for module in ("parameter", "layers.linear"):
        for name, value in (
            ("get_tensor_model_parallel_rank", 0),
            ("get_tensor_model_parallel_world_size", 1),
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.{module}.{name}", lambda value=value: value, raising=False
            )
    monkeypatch.setattr(mtp.vllm_torchtpu_envs, "VLLM_TPU_DRAFT_EMBED_H8", h8)
    monkeypatch.setattr(mtp.vllm_torchtpu_envs, "VLLM_TPU_DRAFT_EMBED_REPL", repl)
    cfg = NS(model_config=NS(hf_text_config=text), additional_config={})
    with torch.device("cpu"):
        predictor = mtp.Qwen4ExpMultiTokenPredictor(cfg, ["mtp.layers.2"])
    predictor = predictor.to(torch.bfloat16)
    g = torch.Generator().manual_seed(11)
    for p in predictor.parameters():
        p.data.copy_((torch.randn(p.shape, generator=g) * 0.3).to(p.dtype))
    assert predictor.embed_h8 is h8 and predictor.embed_repl is repl
    return predictor


def _forward_context(monkeypatch, rows):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    metadata = NS(
        query_start_loc=torch.arange(rows + 1),
        seq_lens=torch.full((rows,), 9),
        block_tables=torch.zeros(rows, dtype=torch.int32),
    )
    monkeypatch.setattr(
        mtp, "get_forward_context", lambda: NS(attn_metadata={"mtp.layers.2.self_attn": metadata})
    )


STEPS, ROWS, PASSES = 24, 16, 3


def _replay(predictor, head):
    """Draft steps of PASSES passes: pass 0 takes chunk tokens (incl. padding / out-of-range ids)
    and a target hidden; pass k > 0 takes pass k-1's greedy token and multi-stream hidden."""
    g = torch.Generator().manual_seed(5)
    records = []
    with torch.no_grad():
        return _replay_steps(predictor, head, g, records)


def _replay_steps(predictor, head, g, records):
    for _ in range(STEPS):
        ids = torch.randint(0, V, (ROWS,), generator=g)
        ids[0], ids[1], ids[2], ids[3] = -1, V, V + 2, 0
        hidden = (torch.randn(ROWS, HC * H, generator=g) * 0.6).to(torch.bfloat16)
        positions = torch.arange(ROWS) + 100
        for k in range(PASSES):
            emb = predictor.embed(ids)
            fused = predictor.fuse_inputs(ids, hidden)
            sample, multi = predictor(ids, positions, hidden, None, k)
            token = (sample.float() @ head.float().T).argmax(-1)
            records.append((emb, fused, sample, multi, token))
            ids, hidden, positions = token, multi, positions + 1
    return records


def _run_stock(monkeypatch, predictor, shards, head):
    import vllm.distributed as vd

    tp = StockTP(shards)
    runs = []
    for rank in range(TP):
        predictor.embed_tokens = shards[rank]
        monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", tp.all_reduce_for(rank))
        runs.append(_replay(predictor, head))
    # embed + fuse_inputs + forward embed once each per pass
    assert tp.calls == TP * STEPS * PASSES * 3
    return runs


def _run_repl(monkeypatch, predictor, shards, head):
    import vllm.distributed as vd

    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    def no_all_reduce(x):
        raise AssertionError("DRAFT-2 (c) must issue no embedding all-reduce")

    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", no_all_reduce)
    runs = []
    for rank in range(TP):
        predictor.embed_tokens = shards[rank]
        if hasattr(predictor, "embed_full"):
            del predictor.embed_full
        _emulate_agreement(monkeypatch, [1] * TP, rank)
        seen = _emulate_gather(monkeypatch, shards, rank)
        assert mtp.Qwen4ExpMTP.bind_replicated_embedding(NS(model=predictor)) is True
        assert seen == [rank] and predictor.embed_full.shape == (V, H)
        runs.append(_replay(predictor, head))
    return runs


def _same_weights(a, b):
    """Both predictors were seeded alike; every parameter but the (replaced) embedding is equal."""
    pa = {k: v for k, v in a.named_parameters() if not k.startswith("embed_tokens.")}
    pb = {k: v for k, v in b.named_parameters() if not k.startswith("embed_tokens.")}
    assert pa.keys() == pb.keys() and all(bits_equal(pa[k].data, pb[k].data) for k in pa)


def _compare(ref, got):
    names = ("embed", "fused", "sample", "multi")
    bad = {n: 0 for n in names}
    agree = total = 0
    for r, x in zip(ref, got):
        for i, n in enumerate(names):
            bad[n] += not bits_equal(r[i], x[i])
        agree += int((r[4] == x[4]).sum())
        total += r[4].numel()
    return bad, agree, total


@pytest.mark.usefixtures("cpu_vllm_config")
def test_replay_stock_vs_replicated_is_bitwise(monkeypatch):
    weight = _weight(1)
    shards = [_shard(r, weight) for r in range(TP)]
    head = (torch.randn(V, H, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16)
    _forward_context(monkeypatch, ROWS)

    stock_pred = _predictor(monkeypatch, h8=False, repl=False)
    stock = _run_stock(monkeypatch, stock_pred, shards, head)
    for rank in range(1, TP):  # every simulated rank computes the same draft
        bad, agree, total = _compare(stock[0], stock[rank])
        assert not any(bad.values()) and agree == total, (rank, bad)

    repl_pred = _predictor(monkeypatch, h8=False, repl=True)
    _same_weights(stock_pred, repl_pred)
    repl = _run_repl(monkeypatch, repl_pred, shards, head)

    from vllm_torchtpu.layers.adapter import tp_ar_decode

    h8_calls = []
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (lambda x: h8_calls.append(1) or x, H))
    both_pred = _predictor(monkeypatch, h8=True, repl=True)
    _same_weights(stock_pred, both_pred)
    both = _run_repl(monkeypatch, both_pred, shards, head)
    assert not h8_calls  # (c) takes precedence over DRAFT-2 (a)

    pass0_odd = sum(int(((r[4] < 0) | (r[4] >= V)).sum()) for r in stock[0])
    assert pass0_odd == 0  # draft tokens are always in-vocab ids
    for label, runs in (("repl", repl), ("repl+h8", both)):
        for rank in range(TP):
            bad, agree, total = _compare(stock[0], runs[rank])
            pct = 100.0 * agree / total
            if rank in (0, TP - 1):
                print(f"\nDRAFTEMB_REPLAY arm={label} rank={rank} tp={TP} steps={STEPS} passes={PASSES} "
                      f"rows={ROWS} draft_tokens={total} agree={agree}/{total} ({pct:.3f}%) "
                      f"bits_mismatch embed={bad['embed']} fused={bad['fused']} "
                      f"sample={bad['sample']} multi={bad['multi']}")
            assert pct >= 99.9  # the brief's bar ...
            assert agree == total and not any(bad.values()), (label, rank, bad)  # ... and bitwise


# ---------------------------------------------------------------- C. 0-chip census


@torch.library.custom_op("q38draftemb::xla_all_reduce", mutates_args=())
def _xla_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_xla_all_reduce.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("q38draftemb::h8_all_reduce", mutates_args=())
def _h8_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_h8_all_reduce.register_fake
def _(x):
    return torch.empty_like(x)


class _OpCount(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name()
        self.ops[name] = self.ops.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


def _graph_counts(fn, ids):
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(lambda x: fn(x))(ids)  # a plain callable: make_fx would count a bound method's self
    counts = {"xla": 0, "h8": 0, "lookup": 0}
    for node in gm.graph.nodes:
        t = str(node.target)
        counts["xla"] += "q38draftemb.xla_all_reduce" in t
        counts["h8"] += "q38draftemb.h8_all_reduce" in t
        counts["lookup"] += t.startswith("aten.embedding")
    return counts


@pytest.mark.usefixtures("cpu_vllm_config")
@pytest.mark.parametrize(
    "variant,h8,repl,want",
    [
        ("off", False, False, (3, 0, 3)),
        ("a_h8", True, False, (0, 3, 3)),
        ("c_repl", False, True, (0, 0, 3)),
        ("a_and_c", True, True, (0, 0, 3)),
    ],
)
def test_census_draft_step_embedding_all_reduces(monkeypatch, variant, h8, repl, want):
    import vllm.distributed as vd

    from vllm_torchtpu.layers.adapter import tp_ar_decode
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    weight = _weight(2)
    shards = [_shard(r, weight) for r in range(TP)]
    rank = 2
    rows = 12
    _forward_context(monkeypatch, rows)
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce",
                        lambda x: torch.ops.q38draftemb.xla_all_reduce(x))
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (lambda x: torch.ops.q38draftemb.h8_all_reduce(x), H))
    pred = _predictor(monkeypatch, h8=h8, repl=repl)
    pred.embed_tokens = shards[rank]
    _emulate_agreement(monkeypatch, [int(repl)] * TP, rank)
    _emulate_gather(monkeypatch, shards, rank)
    assert mtp.Qwen4ExpMTP.bind_replicated_embedding(NS(model=pred)) is repl
    ids = torch.tensor([0, 5, 17, 23, 31, 40, 47, 52, 60, -1, V, 9])
    hidden = (torch.randn(rows, HC * H, generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16)
    positions = torch.arange(rows)
    census = _OpCount()
    with census, torch.no_grad():
        for k in range(PASSES):  # one draft step
            sample, hidden = pred(ids, positions, hidden, None, k)
            ids = torch.randint(0, V, (rows,), generator=torch.Generator().manual_seed(k))
    xla = census.ops.get("q38draftemb::xla_all_reduce", 0)
    h8n = census.ops.get("q38draftemb::h8_all_reduce", 0)
    lookups = census.ops.get("aten::embedding", 0)
    graph = {"xla": 0, "h8": 0, "lookup": 0}
    for _ in range(PASSES):
        with torch.no_grad():
            counts = _graph_counts(pred.embed, ids)
        for key, n in counts.items():
            graph[key] += n
    print(f"\nDRAFTEMB_CENSUS variant={variant} per_draft_step dispatch: embed_all_reduce_xla={xla} "
          f"embed_all_reduce_h8={h8n} lookups={lookups} | traced embed graph x{PASSES}: "
          f"xla={graph['xla']} h8={graph['h8']} lookups={graph['lookup']}")
    assert (xla, h8n, lookups) == want
    assert (graph["xla"], graph["h8"], graph["lookup"]) == want


# ---------------------------------------------------------------- D. all-rank agreement (fail-stop)


def test_unanimous_on_binds_the_gathered_table(monkeypatch):
    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]
    for rank in (0, TP - 1):
        holder = _holder(shards[rank], True)
        calls = _emulate_agreement(monkeypatch, [1] * TP, rank)
        seen = _emulate_gather(monkeypatch, shards, rank)
        assert _bind(holder) is True
        assert calls == [rank] and seen == [rank]
        assert holder.embed_full.dtype == torch.bfloat16 and holder.embed_full.shape == (V, H)
        assert torch.equal(holder.embed_full, weight)  # == holds -0.0 == +0.0
        assert "embed_full" not in holder.state_dict()  # non-persistent: never in a checkpoint


def test_unanimous_off_agrees_and_assembles_nothing(monkeypatch):
    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]
    holder = _holder(shards[3], False)
    calls = _emulate_agreement(monkeypatch, [0] * TP, 3)
    seen = _emulate_gather(monkeypatch, shards, 3)
    assert _bind(holder) is False
    assert calls == [3]  # an off rank takes part in the agreement
    assert seen == [] and not hasattr(holder, "embed_full")


@pytest.mark.parametrize("votes", [[1, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1]])
def test_divergent_flag_stops_every_rank(monkeypatch, votes):
    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]
    for rank in range(TP):
        holder = _holder(shards[rank], bool(votes[rank]))
        calls = _emulate_agreement(monkeypatch, votes, rank)
        seen = _emulate_gather(monkeypatch, shards, rank)
        with pytest.raises(RuntimeError, match="across the TP ranks"):
            _bind(holder)
        assert calls == [rank] and seen == [] and not hasattr(holder, "embed_full")


def test_failed_agreement_gather_stops(monkeypatch):
    import vllm.distributed.parallel_state as ps

    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]

    def broken(out, t, group=None):
        raise RuntimeError("gloo: connection reset")

    monkeypatch.setattr(ps, "get_tp_group", lambda: NS(world_size=TP, cpu_group=CPU_GROUP))
    monkeypatch.setattr(torch.distributed, "all_gather", broken)
    seen = _emulate_gather(monkeypatch, shards, 0)
    with pytest.raises(RuntimeError, match="could not agree"):
        _bind(_holder(shards[0], True))
    assert seen == []


# ---------------------------------------------------------------- E. refusals and wiring


@pytest.mark.usefixtures("cpu_vllm_config")
def test_flag_on_without_a_bound_table_refuses_to_trace(monkeypatch):
    weight = _weight(4)
    pred = _predictor(monkeypatch, repl=True)
    pred.embed_tokens = _shard(0, weight)
    with pytest.raises(RuntimeError, match="never bound"):
        pred.embed(torch.tensor([1, 2, 3]))


@pytest.mark.parametrize("kind", ["weight_scale", "fp8"])
def test_quantized_embedding_is_refused(monkeypatch, kind):
    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]
    shard = shards[1]
    if kind == "weight_scale":
        shard.weight_scale = torch.ones(PER)
    else:
        shard.weight = nn.Parameter(shard.weight.data.to(torch.float8_e4m3fn), requires_grad=False)
    _emulate_agreement(monkeypatch, [1] * TP, 1)
    seen = _emulate_gather(monkeypatch, shards, 1)
    holder = _holder(shard, True)
    with pytest.raises(ValueError, match="unquantized shared embedding"):
        _bind(holder)
    assert seen == [] and not hasattr(holder, "embed_full")


@pytest.mark.usefixtures("cpu_vllm_config")
def test_tp1_embedding_is_left_alone(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    agreed = []
    monkeypatch.setattr(mtp, "agree_draft_embed_repl", lambda *a: agreed.append(a) or [a])
    plain = nn.Embedding(V, H)
    holder = _holder(plain, True)
    assert _bind(holder) is False and not hasattr(holder, "embed_full")
    pred = _predictor(monkeypatch, repl=True)
    pred.embed_tokens = plain
    ids = torch.tensor([0, 4, 60])
    assert torch.equal(pred.embed(ids), plain(ids))
    assert _bind(NS(embed_tokens=None)) is False  # a draft without an embedding module
    assert agreed == []  # no collective at TP 1, so nothing to agree on


def test_bind_shared_selection_runs_the_replicated_bind_first(monkeypatch):
    from tests.models.vllm.test_qwen4_exp_shared_selection_bind import TOKENS, _models

    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    weight = _weight(4)
    shards = [_shard(r, weight) for r in range(TP)]
    for shared in (False, True):
        monkeypatch.setattr(mtp.vllm_torchtpu_envs, "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", shared)
        target, draft = _models()
        draft.model.embed_tokens = shards[6]
        draft.model.embed_repl = True
        calls = _emulate_agreement(monkeypatch, [1] * TP, 6)
        _emulate_gather(monkeypatch, shards, 6)
        assert draft.bind_shared_selection(target, TOKENS) is shared
        assert calls == [6] and torch.equal(draft.model.embed_full, weight)


def test_flag_default_and_build_time_read():
    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_DRAFT_EMBED_REPL"}
    code = "from vllm_torchtpu import envs; print('FLAG', envs.VLLM_TPU_DRAFT_EMBED_REPL)"
    for extra, want in (({}, "FLAG False"), ({"VLLM_TPU_DRAFT_EMBED_REPL": "1"}, "FLAG True")):
        out = subprocess.run([sys.executable, "-c", code], env={**env, **extra},
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        assert [ln for ln in out.stdout.splitlines() if ln.startswith("FLAG ")] == [want], out.stdout
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "src", "vllm_torchtpu")
    src = open(os.path.join(root, "models", "vllm", "qwen4_exp", "mtp.py")).read()
    tree = ast.parse(src)
    cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    pred = {n.name: n for n in cls["Qwen4ExpMultiTokenPredictor"].body if isinstance(n, ast.FunctionDef)}
    init_src = ast.get_source_segment(src, pred["__init__"])
    assert "self.embed_repl = bool(vllm_torchtpu_envs.VLLM_TPU_DRAFT_EMBED_REPL)" in init_src
    # The proposer reaches the bind through bind_shared_selection, after the sharing and before
    # the fused loop is built (eagle3 load_model; eagle3.py is unchanged by this lever).
    e3 = open(os.path.join(root, "spec_decode", "eagle3.py")).read()
    fn = next(n for n in ast.walk(ast.parse(e3))
              if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    body = ast.get_source_segment(e3, fn)
    i_share = body.index("maybe_share_embeddings(")
    i_bind = body.index('getattr(self.draft_model, "bind_shared_selection", None)')
    i_call = body.index("bind(\n", i_bind)
    i_loop = body.index("self._build_mtp_fused_loop()")
    assert i_share < i_bind < i_call < i_loop
