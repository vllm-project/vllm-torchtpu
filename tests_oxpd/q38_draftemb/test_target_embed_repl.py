# SPDX-License-Identifier: Apache-2.0
"""q38 L4 EMB-REP-TGT (VLLM_TPU_TARGET_EMBED_REPL, default off, compile-hashed, needs DRAFT-2 (c)):
the target's embedding module (the one the MTP draft borrows) looks up DRAFT-2 (c)'s replicated
table through the TPU-patched VocabParallelEmbedding.forward.

A. Bitwise against the vocab-parallel path on all 8 simulated ranks, for pad / out-of-range ids and
   for -0.0 entries: the stock all-reduce turns -0.0 into +0.0, and so does the table's assembly
   (the same sum), so the outputs match bit for bit. Positive control: a raw-row table that keeps
   -0.0 differs from the stock output in exactly those bits, so the check can tell.
B. 0-chip traced-graph census of the module call: embedding all-reduce 1 -> 0, lookup 1 -> 1.
C. Bind: the target module gets the draft's tensor (same storage, no extra HBM) only with both
   flags; the target flag alone is refused; a divergent target flag stops every rank.
D. Wiring (C): every Qwen3.8 target site (stock, segment, whole-step / D14 driver) calls the
   embedding module, so the module-level lookup covers them; the build-time read and the default.
"""

import ast
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from test_draft_embed_repl import (  # one module object: its custom ops register once
    IDS,
    TP,
    V,
    StockTP,
    _bits,
    _emulate_agreement,
    _emulate_gather,
    _graph_counts,
    _shard,
    _weight,
    bits_equal,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "vllm_torchtpu")


def _holder(shard, repl, tgt):
    m = nn.Module()
    m.embed_tokens = shard
    m.embed_repl = repl
    m.embed_repl_tgt = tgt
    return m


def _bind(holder):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    return mtp.Qwen4ExpMTP.bind_replicated_embedding(NS(model=holder))


def _bound_shards(monkeypatch, weight):
    shards = [_shard(r, weight) for r in range(TP)]
    tables = []
    for rank in range(TP):
        _emulate_agreement(monkeypatch, [(1, 1)] * TP, rank)
        _emulate_gather(monkeypatch, shards, rank)
        holder = _holder(shards[rank], True, True)
        assert _bind(holder) is True
        tables.append(holder.embed_full)
    return shards, tables


# ---------------------------------------------------------------- A. bitwise, incl. -0.0


@pytest.mark.parametrize("ids_dtype", [torch.int64, torch.int32])
def test_target_lookup_is_bitwise_including_signed_zero(monkeypatch, ids_dtype):
    import vllm.distributed as vd

    weight = _weight(6)
    neg0 = [(3, 1), (V - 1, 0)]
    assert all(_bits(weight)[r, c].item() == -32768 for r, c in neg0)  # bf16 -0.0
    ref = [_shard(r, weight) for r in range(TP)]
    tp = StockTP(ref)
    ids = IDS.to(ids_dtype)
    stock = []
    for rank in range(TP):
        monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", tp.all_reduce_for(rank))
        stock.append(ref[rank](ids))
    assert tp.calls == TP and all(bits_equal(stock[0], s) for s in stock)
    # The stock sum turns -0.0 into +0.0 (every output row of ids 3 and V-1).
    at = sorted((k, c) for r, c in neg0 for k, i in enumerate(IDS.tolist()) if i == r)
    n_at = sum(IDS.tolist().count(r) for r, _ in neg0)  # ids 3 and V-1 each occur twice in IDS
    assert len(at) == n_at == 4 and all(_bits(stock[0])[k, c].item() == 0 for k, c in at)

    shards, tables = _bound_shards(monkeypatch, weight)

    def no_all_reduce(x):
        raise AssertionError("EMB-REP-TGT must issue no embedding all-reduce")

    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", no_all_reduce)
    for rank in range(TP):
        assert shards[rank]._q38_repl_table is tables[rank]
        got = shards[rank](ids)  # the TPU-patched forward
        assert bits_equal(got, stock[0]), rank
    # Positive control: a table of raw checkpoint rows (keeping -0.0) differs in exactly those bits.
    raw = weight.clone()
    shards[0]._q38_repl_table = raw
    got = shards[0](ids)
    differ = (_bits(got) != _bits(stock[0])).nonzero().tolist()
    print(f"\nTGTEMB_BITS ranks={TP} ids={IDS.numel()} dtype={str(ids_dtype).removeprefix('torch.')} "
          f"bitwise_vs_stock=8/8 signed_zero_bits_stock_and_table=+0.0 raw_table_control_diffs={len(differ)}")
    assert sorted(map(tuple, differ)) == at


# ---------------------------------------------------------------- B. 0-chip census


def test_census_target_embedding_all_reduce(monkeypatch):
    import vllm.distributed as vd

    weight = _weight(7)
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce",
                        lambda x: torch.ops.q38draftemb.xla_all_reduce(x))
    plain = _shard(4, weight)
    ids = torch.tensor([0, 9, 33, 60, -1, V, 47, 2])
    with torch.no_grad():
        off = _graph_counts(plain, ids)
    shards, _ = _bound_shards(monkeypatch, weight)
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce",
                        lambda x: torch.ops.q38draftemb.xla_all_reduce(x))
    with torch.no_grad():
        on = _graph_counts(shards[4], ids)
    for label, c in (("off", off), ("tgt", on)):
        print(f"\nTGTEMB_CENSUS variant={label} per_target_forward traced graph: embed_all_reduce={c['xla']} "
              f"lookups={c['lookup']}")
    assert (off["xla"], off["lookup"]) == (1, 1)
    assert (on["xla"], on["h8"], on["lookup"]) == (0, 0, 1)


# ---------------------------------------------------------------- C. bind, refusal, agreement


@pytest.mark.parametrize("tgt", [True, False])
def test_bind_shares_the_draft_table_only_with_the_target_flag(monkeypatch, tgt):
    weight = _weight(8)
    shards = [_shard(r, weight) for r in range(TP)]
    calls = _emulate_agreement(monkeypatch, [(1, int(tgt))] * TP, 2)
    _emulate_gather(monkeypatch, shards, 2)
    holder = _holder(shards[2], True, tgt)
    assert _bind(holder) is True and calls == [2]
    if tgt:
        table = shards[2]._q38_repl_table
        assert table is holder.embed_full and table.data_ptr() == holder.embed_full.data_ptr()
        assert "_q38_repl_table" not in shards[2].state_dict()
    else:
        assert getattr(shards[2], "_q38_repl_table", None) is None


def test_target_flag_without_the_draft_flag_is_refused(monkeypatch):
    weight = _weight(8)
    shards = [_shard(r, weight) for r in range(TP)]
    calls = _emulate_agreement(monkeypatch, [(0, 1)] * TP, 5)
    seen = _emulate_gather(monkeypatch, shards, 5)
    holder = _holder(shards[5], False, True)
    with pytest.raises(ValueError, match="needs VLLM_TPU_DRAFT_EMBED_REPL=1"):
        _bind(holder)
    assert calls == [5] and seen == []
    assert getattr(shards[5], "_q38_repl_table", None) is None and not hasattr(holder, "embed_full")


def test_divergent_target_flag_stops_every_rank(monkeypatch):
    weight = _weight(8)
    shards = [_shard(r, weight) for r in range(TP)]
    votes = [(1, 1)] * TP
    votes[6] = (1, 0)
    for rank in range(TP):
        holder = _holder(shards[rank], True, bool(votes[rank][1]))
        calls = _emulate_agreement(monkeypatch, votes, rank)
        seen = _emulate_gather(monkeypatch, shards, rank)
        with pytest.raises(RuntimeError, match="across the TP ranks"):
            _bind(holder)
        assert calls == [rank] and seen == []
        assert getattr(shards[rank], "_q38_repl_table", None) is None


# ---------------------------------------------------------------- D. wiring and flag


def _calls_to_embed_tokens(path):
    src = open(path).read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "embed_tokens"]
    weight_reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("weight", "weight_scale")
                    and isinstance(n.value, ast.Attribute) and n.value.attr == "embed_tokens"]
    return len(calls), len(weight_reads)


def test_every_target_site_calls_the_embedding_module():
    model = os.path.join(ROOT, "models", "vllm", "qwen4_exp", "model.py")
    driver = os.path.join(ROOT, "megakernel_runtime", "qwen38_driver.py")
    n_model, w_model = _calls_to_embed_tokens(model)
    n_driver, w_driver = _calls_to_embed_tokens(driver)
    print(f"\nTGTEMB_SITES model.py module_calls={n_model} weight_reads={w_model} "
          f"qwen38_driver.py module_calls={n_driver} weight_reads={w_driver}")
    # stock forward, segment path, whole-step path, embed_input_ids; the D14 driver's two forwards
    assert n_model == 4 and n_driver == 2
    assert w_model == 0 and w_driver == 0  # nothing reads the table around the module


def test_target_flag_default_and_build_time_read():
    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_TARGET_EMBED_REPL"}
    code = "from vllm_torchtpu import envs; print('FLAG', envs.VLLM_TPU_TARGET_EMBED_REPL)"
    for extra, want in (({}, "FLAG False"), ({"VLLM_TPU_TARGET_EMBED_REPL": "1"}, "FLAG True")):
        out = subprocess.run([sys.executable, "-c", code], env={**env, **extra},
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        assert [ln for ln in out.stdout.splitlines() if ln.startswith("FLAG ")] == [want], out.stdout
    src = open(os.path.join(ROOT, "models", "vllm", "qwen4_exp", "mtp.py")).read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen4ExpMultiTokenPredictor")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "self.embed_repl_tgt = bool(vllm_torchtpu_envs.VLLM_TPU_TARGET_EMBED_REPL)" in ast.get_source_segment(src, init)
    pkg = open(os.path.join(ROOT, "__init__.py")).read()
    fwd = next(n for n in ast.walk(ast.parse(pkg)) if isinstance(n, ast.FunctionDef) and n.name == "patched_forward")
    first = ast.get_source_segment(pkg, fwd.body[0])
    assert first == 'table = getattr(self, "_q38_repl_table", None)'
