# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill matches the replicated TP path.

TP ranks are simulated by threads that exchange tensors through a barrier, so
the decoder layer runs its real control flow on each rank's shard.
"""

import threading
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

HIDDEN, HC, LOWRANK, TOKENS = 128, 2, 16, 32


@pytest.fixture(autouse=True)
def cpu_default():
    with torch.device("cpu"):
        yield


class SimulatedTP:
    """Collectives over threads; each thread is one TP rank."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def exchange(self, x):
        self.barrier.wait()
        self.slots[self.rank] = x
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def total(self, x):
        values = self.exchange(x)
        out = values[0].float()
        for v in values[1:]:
            out = out + v.float()
        return out.to(x.dtype)

    def shard(self, x):
        return x.chunk(self.size, dim=0)[self.rank].clone()

    def gather(self, x, num_tokens):
        return torch.cat(self.exchange(x), dim=0).narrow(0, 0, num_tokens)

    def reduce(self, x, sequence_parallel):
        summed = self.total(x)
        return self.shard(summed) if sequence_parallel else summed

    def run(self, fn):
        results, errors = [None] * self.size, []

        def body(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


@pytest.fixture
def tp2(monkeypatch):
    tp = SimulatedTP(2)
    monkeypatch.setattr(sp, "shard", tp.shard)
    monkeypatch.setattr(sp, "gather", tp.gather)
    monkeypatch.setattr(sp, "reduce", tp.reduce)
    monkeypatch.setattr(sp, "tp_world_size", lambda: tp.size)
    return tp


def hc_module():
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=HIDDEN,
            hc_count=HC,
            hc_lowrank=LOWRANK,
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return module


class PartialAttention(nn.Module):
    """Rank r returns a fixed share of a linear map, like a row-parallel o_proj."""

    decode_token_shard = False
    replicated_token_shard = False

    def __init__(self, tp, weight, layer_type):
        super().__init__()
        self.tp, self.weight, self.layer_type = tp, weight, layer_type

    def _partial(self, block):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        return (block.float() @ self.weight.T * share).to(block.dtype)

    def forward(self, block, gdn_step_metadata_cache=None):
        # GDN: un-reduced output projection
        return self._partial(block)

    def forward_with_indices(
        self,
        block,
        positions,
        metadata,
        reduce=True,
        replicated_indexer=False,
        token_shard=False,
        decode_token_shard=False,
    ):
        assert reduce is False
        return self._partial(block), None


class RowwiseMoE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, block):
        return torch.tanh(block.float() @ self.weight.T).to(block.dtype)


class SplitMoE(nn.Module):
    """Routed experts on any rows + a TP-sliced shared expert, like MoERunner."""

    def __init__(self, tp, weight):
        super().__init__()
        self.tp, self.weight = tp, weight
        self.replicate_shared_expert = False
        self.gate = lambda x: (x.float() @ weight[:4].T, None)
        method = NS(_adaptive_fused_moe_op_sp=object())
        self.experts = NS(
            routed_experts=NS(
                quant_method=method,
                w13_weight=None,
                w2_weight=None,
                w13_weight_scale_inv=None,
                w2_weight_scale_inv=None,
            )
        )
        self.shared_expert = self._shared_partial

    def routed(self, block, logits):
        scale = torch.sigmoid(logits[:, :1])
        return (torch.tanh(block.float() @ self.weight.T) * scale).to(block.dtype)

    def _shared_partial(self, block):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        return (torch.sin(block.float() @ self.weight) * share).to(block.dtype)

    def forward(self, block):
        logits, _ = self.gate(block)
        return self.tp.total(self._shared_partial(block)) + self.routed(block, logits)


def make_layer(tp, layer_type, attn_hc, mlp_hc, attn_weight, moe_weight):
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpDecoderLayer

    layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = layer_type
    attention = PartialAttention(tp, attn_weight, layer_type)
    if layer_type == "linear_attention":
        layer.linear_attn = attention
    else:
        layer.self_attn = attention
    layer.attn_hyper_connection = attn_hc
    layer.mlp_hyper_connection = mlp_hc
    layer.mlp = RowwiseMoE(moe_weight)
    return layer


@pytest.mark.parametrize("sp_pad", [0, 24])
@pytest.mark.parametrize(
    "split_moe,shared_local", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_decoder_layer_sequence_parallel_matches_tp(
    tp2, monkeypatch, layer_type, split_moe, shared_local, sp_pad
):
    """shared_local: QWEN4_EXP_PREFILL_SP_SHARED_LOCAL, the shared expert's replicated
    copy on
    this rank's rows (no gather / reduce); one bf16 sum instead of the TP partials'
    reduce."""
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as adaptive_moe,
    )

    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MOE", split_moe, raising=False)
    calls = []

    def fake_adaptive(method, block, w1, w2, s1, s2, logits, attr):
        calls.append((attr, block.shape[0]))
        return layers[tp2.rank].mlp.routed(block, logits)

    monkeypatch.setattr(adaptive_moe, "run_adaptive_fused_moe", fake_adaptive)
    layers = {}
    torch.random.default_generator.manual_seed(0)
    attn_hc, mlp_hc = hc_module(), hc_module()
    attn_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    moe_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    hidden = torch.randn(TOKENS, HC * HIDDEN).bfloat16()
    input_ids = torch.zeros(TOKENS, dtype=torch.long)
    valid = torch.arange(TOKENS) < TOKENS - 5  # trailing padding rows
    # NaN padding rows must not reach the MoE on either path.
    hidden[~valid] = float("nan")

    def run(rank, sequence_parallel):
        layer = make_layer(tp2, layer_type, attn_hc, mlp_hc, attn_w, moe_w)
        layer.mlp = SplitMoE(tp2, moe_w)
        if shared_local and sequence_parallel:

            def gathered_shared(block):
                raise AssertionError(
                    "shared_expert_local must replace the gathered TP call"
                )

            layer.mlp.shared_expert = gathered_shared
            layer.mlp.shared_expert_local = lambda b: torch.sin(b.float() @ moe_w).to(
                b.dtype
            )
        layers[rank] = layer
        x = tp2.shard(hidden) if sequence_parallel else hidden
        pad = sp_pad if sequence_parallel else 0
        out = layer(
            sp.pad_shard(x, pad),
            None,
            input_ids,
            None,
            None,
            None,
            None,
            valid,
            None,
            None,
            sequence_parallel=sequence_parallel,
            sp_pad=pad,
        )
        if pad:
            # 16 real rows per rank, padded to 24: the tail stays zero.
            assert out.shape[0] == pad
            torch.testing.assert_close(
                out[x.shape[0] :], torch.zeros_like(out[x.shape[0] :])
            )
        return sp.unpad(out, x.shape[0])

    replicated = tp2.run(lambda r: run(r, False))
    sharded = tp2.run(lambda r: run(r, True))
    torch.testing.assert_close(
        replicated[0], replicated[1], rtol=0, atol=0, equal_nan=True
    )
    joined = torch.cat(sharded, dim=0)
    assert joined.shape == replicated[0].shape
    rows = valid
    # The replicated shared expert sums in one bf16 rounding, the TP path in two.
    atol = 3e-2 if shared_local else 0
    torch.testing.assert_close(joined[rows], replicated[0][rows], rtol=0, atol=atol)
    expected_calls = [("_adaptive_fused_moe_op_sp", TOKENS // 2)] * 2
    assert calls == (expected_calls if split_moe else [])


def test_ple_gate_sequence_parallel_matches_tp(tp2):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import Qwen4ExpPLELayer

    torch.random.default_generator.manual_seed(1)
    embed = 24
    key_w = torch.randn(HC * HIDDEN, embed).bfloat16()
    value_w = torch.randn(HIDDEN, embed).bfloat16()

    def rms(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )

    owner = NS(
        key_proj=lambda e: (e @ key_w.T, None),
        value_proj=lambda e: (e @ value_w.T, None),
        norm_key=rms,
        norm_query=rms,
        gate_dot=None,
        hc_count=HC,
        hidden_size=HIDDEN,
    )
    hidden = torch.randn(TOKENS, HC * HIDDEN).bfloat16()
    embeddings = torch.randn(TOKENS, embed).bfloat16()
    expected = Qwen4ExpPLELayer.gated_values(owner, hidden, embeddings)
    got = tp2.run(
        lambda r: Qwen4ExpPLELayer.gated_values(
            owner, tp2.shard(hidden), embeddings, TOKENS
        )
    )
    for g in got:  # values stay complete on every rank
        torch.testing.assert_close(g, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "enabled,minimum,tokens,tp,expected",
    [
        (False, 1024, 8192, 8, False),
        (True, 1024, 8192, 8, True),
        (True, 1024, 1024, 8, True),
        (True, 1024, 512, 8, False),
        (True, 1024, 1056, 8, False),  # shard would not be 8-row aligned
        (True, 1024, 8192, 1, False),
        (True, 0, 64, 8, False),  # decode buckets stay under the 1024 floor
        (True, 0, 1088, 8, True),
        (True, 2048, 1088, 8, False),
    ],
)
def test_sequence_parallel_eligibility(
    monkeypatch, enabled, minimum, tokens, tp, expected
):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", enabled, raising=False)
    monkeypatch.setattr(
        sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", minimum, raising=False
    )
    assert sp.sequence_parallel_eligible(tokens, tp) is expected


def test_shard_rows_keep_every_hc_kernel_on_its_pallas_path():
    # Full mix tile 128 runs on 256-row multiples, up and norm on 128.
    assert sp.shard_rows([256, 128, 128]) == 256
    assert sp.shard_rows([0, 0, 0]) == sp.ROW_ALIGNMENT


@pytest.mark.parametrize(
    "tokens,expected", [(1024, False), (2048, True), (3072, False), (8192, True)]
)
def test_eligibility_with_hc_shard_rows(monkeypatch, tokens, expected):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", True, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024, raising=False)
    assert sp.sequence_parallel_eligible(tokens, 8, rows=256) is expected


def test_pad_shard_rounds_up_with_zero_rows_and_unpad_restores():
    x = torch.randn(384, 4)
    padded = sp.pad_shard(x, 256)
    assert padded.shape == (512, 4)
    assert torch.equal(padded[384:], torch.zeros(128, 4))
    assert torch.equal(sp.unpad(padded, 384), x)
    flags = sp.pad_shard(torch.ones(128, dtype=torch.bool), 256)
    assert flags.dtype == torch.bool and not flags[128:].any()
    assert sp.pad_shard(x, 0) is x
    assert sp.padded_rows(256, 256) == 256


@pytest.mark.parametrize(
    "tokens,expected", [(960, False), (1024, True), (3072, True), (1088, True)]
)
def test_padded_shards_admit_every_row_aligned_bucket(monkeypatch, tokens, expected):
    # QWEN4_EXP_PREFILL_SP_PAD_HC: the rule falls back to ROW_ALIGNMENT rows.
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", True, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024, raising=False)
    assert sp.sequence_parallel_eligible(tokens, 8) is expected


@pytest.mark.parametrize(
    "pad,moe,expected", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_padded_shards_only_with_the_sequence_parallel_moe(
    monkeypatch, pad, moe, expected
):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_PAD_HC", pad, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MOE", moe, raising=False)
    assert sp.pad_hc_enabled() is expected


# --- VLLM_TPU_SP_MICROBATCH_OVERLAP ------------------------------------------------

_SC_AG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather"
_SC_RS = "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter"


@pytest.fixture
def microbatch_env(monkeypatch):
    """Clean decision state around each test."""
    sp.reset_microbatch()
    monkeypatch.setattr(sp.envs, "VLLM_TPU_MAMBA_TAIL_FOLD", False, raising=False)
    yield monkeypatch
    sp.reset_microbatch()


@pytest.mark.parametrize(
    "flag,force_off,libtpu,tail_fold,expected",
    [
        (False, False, f"{_SC_AG}=true", False, False),
        (True, False, f"{_SC_AG}=true", False, True),
        (True, False, f"{_SC_RS}=true", False, True),
        (True, False, "", False, False),  # nothing on the SparseCore
        (True, False, f"{_SC_AG}=true {_SC_AG}=false", False, False),  # last wins
        (True, True, f"{_SC_AG}=true", False, False),  # kill switch
        (True, False, f"{_SC_AG}=true", True, False),  # tail-fold checkpoints
    ],
)
def test_microbatch_guard(microbatch_env, flag, force_off, libtpu, tail_fold, expected):
    mp = microbatch_env
    mp.setenv("LIBTPU_INIT_ARGS", libtpu)
    mp.setattr(sp.envs, "VLLM_TPU_SP_MICROBATCH_OVERLAP", flag, raising=False)
    mp.setattr(
        sp.envs, "VLLM_TPU_SP_MICROBATCH_OVERLAP_FORCE_OFF", force_off, raising=False
    )
    mp.setattr(sp.envs, "VLLM_TPU_MAMBA_TAIL_FOLD", tail_fold, raising=False)
    assert sp.microbatch_enabled() is expected
    assert sp.microbatch_groups() == (2 if expected else 1)
    # Decided once: later environment changes do not reach traced code.
    mp.setenv("LIBTPU_INIT_ARGS", "")
    assert sp.microbatch_enabled() is expected


def test_microbatch_block(microbatch_env):
    mp = microbatch_env
    mp.setenv("LIBTPU_INIT_ARGS", f"{_SC_AG}=true")
    mp.setattr(sp.envs, "VLLM_TPU_SP_MICROBATCH_OVERLAP", True, raising=False)
    sp.block_microbatch("decode rows do not fit")
    assert sp.microbatch_enabled() is False


# A step of 32 tokens in halves of 16: two decode/verify windows, a prefill that
# resumes a prefix-cache hit, one that crosses the half boundary (resuming from a
# non-zero read offset), one that starts in half B, an idle slot and 5 padding rows.
STARTS = [0, 2, 3, 12, 24, 27, 27]
CONTEXT = [5, 9, 4, 7, 0, 0]
DISTRIBUTION = [2, 4, 5]  # [decode_end, prefill_end, mixed_end]
READ_OFFSET = [1, 0, 1, 1, 0, 0]
SPEC = 2  # checkpoints per request (one per verify-window position)


def _metadata():
    starts = torch.tensor(STARTS, dtype=torch.int32)
    context = torch.tensor(CONTEXT, dtype=torch.int64)
    lengths = starts[1:] - starts[:-1]
    seq_lens = (context + lengths).to(torch.int32)
    seq_lens[-1] = 0  # idle slot
    state_ids = torch.tensor([3, 5, 7, 9, 11, 0], dtype=torch.int32)
    # Per-block read offsets (indexed by state id), as the runner fills them.
    slot_read = torch.zeros(16, dtype=torch.int32)
    slot_read[state_ids.long()] = torch.tensor(READ_OFFSET, dtype=torch.int32)
    slot_read[0] = 0
    ckpt = torch.stack([state_ids, state_ids + 1], dim=1)  # [seqs, SPEC]
    return NS(
        query_start_loc=starts,
        context=context,
        seq_lens=seq_lens,
        distribution=torch.tensor(DISTRIBUTION, dtype=torch.int32),
        state_ids=state_ids,
        slot_read=slot_read,
        ckpt=ckpt,
    )


def reference_gdn(x, pool, starts, seq_lens, distribution, state_ids, read, ckpt):
    """The fused GDN kernel's state contract on a toy recurrence
    (``kernels/gdn/v3``): windowed sequences ``[0, decode_end)`` read
    checkpoint ``read[s]`` and write one checkpoint per window position;
    PER_SEQ sequences ``[decode_end, mixed_end)`` take tiles only for query
    rows, read checkpoint ``read[s]`` iff ``seq_lens - query_lens > 0`` and
    write their final state to checkpoint 0. The state carries the previous
    input too, standing in for the conv state."""
    pool = pool.clone()
    out = torch.zeros_like(x)
    starts, seq_lens = starts.tolist(), seq_lens.tolist()
    decode_end, end = int(distribution[0]), int(distribution[-1])
    for s in range(end):
        lo, hi = starts[s], starts[s + 1]
        if hi == lo:
            continue
        state = (
            pool[ckpt[s, read[s]]].clone()
            if seq_lens[s] - (hi - lo) > 0
            else torch.zeros_like(pool[0])
        )
        for i, t in enumerate(range(lo, hi)):
            h, prev = state.chunk(2)
            h = 0.5 * h + x[t] + 0.25 * prev
            state = torch.cat([h, x[t]])
            out[t] = h
            if s < decode_end:
                pool[ckpt[s, i]] = state
        if s >= decode_end:
            pool[ckpt[s, 0]] = state
    del state_ids
    return out, pool


def test_gdn_halves_continue_the_state_of_a_whole_step():
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import _half_step

    md = _metadata()
    torch.random.default_generator.manual_seed(2)
    x = torch.randn(32, 4, dtype=torch.float64)
    pool = torch.randn(16, 8, dtype=torch.float64)

    def run(x, pool, starts, seq_lens, distribution, slot_read):
        read = slot_read[md.state_ids.long()].tolist()
        return reference_gdn(
            x, pool, starts, seq_lens, distribution, md.state_ids, read, md.ckpt
        )

    whole, whole_pool = run(
        x, pool, md.query_start_loc, md.seq_lens, md.distribution, md.slot_read
    )
    plan = sp.MicroBatchPlan(32)
    attn_md = NS(query_start_loc=md.query_start_loc, seq_lens=md.seq_lens)
    outs, state = [], pool
    for g in range(2):
        key, starts, seq_lens, distribution, slot_read = _half_step(
            (plan, g), attn_md, md.state_ids, md.distribution, md.slot_read
        )
        assert key is plan.gdn(attn_md, None, None, None, g)  # memoized per group
        out, state = run(
            x[16 * g : 16 * (g + 1)], state, starts, seq_lens, distribution, slot_read
        )
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs), whole, rtol=0, atol=0)
    torch.testing.assert_close(state, whole_pool, rtol=0, atol=0)


def test_gdn_half_metadata():
    md = _metadata()
    a = sp.gdn_half(md.query_start_loc, md.seq_lens, md.distribution, 0, 16)
    b = sp.gdn_half(md.query_start_loc, md.seq_lens, md.distribution, 1, 16)
    assert a.query_start_loc.tolist() == [0, 2, 3, 12, 16, 16, 16]
    assert b.query_start_loc.tolist() == [0, 0, 0, 0, 8, 11, 11]
    # The crossing request ends half A after 4 of its 12 rows.
    assert a.seq_lens.tolist() == [7, 10, 13, 11, 0, 0]
    assert b.seq_lens.tolist() == md.seq_lens.tolist()
    assert a.distribution.tolist() == DISTRIBUTION and a.resume_from_carry is None
    assert b.distribution.tolist() == [0, 4, 5]
    assert b.resume_from_carry.tolist() == [False, False, False, True, False, False]


def reference_qsa(x, cache, starts, context):
    """Causal attention on a toy cache: every query row writes its key at
    ``context + offset`` of its request, then reads all keys up to there."""
    cache = cache.clone()
    out = torch.zeros_like(x)
    starts, context = starts.tolist(), context.tolist()
    for r in range(len(context)):
        for i, t in enumerate(range(starts[r], starts[r + 1])):
            pos = context[r] + i
            cache[r, pos] = x[t]
            weights = torch.arange(1, pos + 2, dtype=x.dtype)[:, None]
            out[t] = (cache[r, : pos + 1] * weights).sum(0)
    return out, cache


def test_qsa_halves_attend_the_keys_of_earlier_halves():
    md = _metadata()
    torch.random.default_generator.manual_seed(3)
    x = torch.randn(32, 4, dtype=torch.float64)
    cache = torch.randn(len(CONTEXT), 40, 4, dtype=torch.float64)
    whole, whole_cache = reference_qsa(x, cache, md.query_start_loc, md.context)
    plan = sp.MicroBatchPlan(32)
    outs, state = [], cache
    for g in range(2):
        starts, context = plan.qsa(md.query_start_loc, md.context, g)
        out, state = reference_qsa(x[16 * g : 16 * (g + 1)], state, starts, context)
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs), whole, rtol=0, atol=0)
    torch.testing.assert_close(state, whole_cache, rtol=0, atol=0)
    assert plan.qsa(md.query_start_loc, md.context, 1)[1].tolist() == [
        7,
        10,
        13,
        11,
        0,
        0,
    ]


class MicrobatchTP(SimulatedTP):
    """SimulatedTP behind sequence_parallel's collective primitives, so the
    module's own layout code (shard / gather / reduce, the halves) runs."""

    def reduce_scatter(self, x):
        return self.shard(self.total(x))


def _primitives(mp, decision):
    tp = MicrobatchTP(2)
    mp.setattr(sp, "all_gather_rows", lambda x: torch.cat(tp.exchange(x)))
    mp.setattr(sp, "reduce_scatter_rows", tp.reduce_scatter)
    mp.setattr(sp, "chunk_rows", tp.shard)
    mp.setattr(sp, "tensor_model_parallel_all_reduce", tp.total)
    mp.setattr(sp, "tp_world_size", lambda: tp.size)
    mp.setattr(sp, "_microbatch_decision", decision)
    tp.microbatch = decision
    return tp


@pytest.fixture(params=[False, True], ids=["mb_off", "mb_on"])
def tp2_primitives(request, microbatch_env):
    return _primitives(microbatch_env, request.param)


class CausalPartialAttention(nn.Module):
    """A causal attention core over per-rank caches (the toy contracts above),
    with a row-parallel output share per rank. GDN reads its group metadata as
    the real layer reads the forward context; both honour token halves."""

    decode_token_shard = False
    replicated_token_shard = False

    def __init__(self, tp, weight, layer_type, md, init_state):
        super().__init__()
        self.tp, self.weight, self.layer_type, self.md = tp, weight, layer_type, md
        self.state = init_state.clone()
        self.halves = []

    def _project(self, core, num_rows):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        out = (core.float() @ self.weight.T * share).to(torch.bfloat16)
        assert out.shape[0] == num_rows
        return out

    def forward(self, block, gdn_step_metadata_cache=None, sp_half=None):
        from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
            _half_step,
        )

        md = self.md
        starts, seq_lens, distribution, slot_read = (
            md.query_start_loc,
            md.seq_lens,
            md.distribution,
            md.slot_read,
        )
        if sp_half is not None:
            self.halves.append(sp_half[1])
            _, starts, seq_lens, distribution, slot_read = _half_step(
                sp_half, md, md.state_ids, distribution, slot_read
            )
        read = slot_read[md.state_ids.long()].tolist()
        core, self.state = reference_gdn(
            block.double(),
            self.state,
            starts,
            seq_lens,
            distribution,
            md.state_ids,
            read,
            md.ckpt,
        )
        return self._project(core, block.shape[0])

    def forward_with_indices(
        self,
        block,
        positions,
        metadata,
        reduce=True,
        replicated_indexer=False,
        token_shard=False,
        decode_token_shard=False,
        write_sink=True,
    ):
        assert reduce is False
        if not write_sink:  # a token half; the layer writes the sink once
            self.halves.append(write_sink)
        core, self.state = reference_qsa(
            block.double(),
            self.state,
            metadata.query_start_loc,
            metadata.context_lens,
        )
        return self._project(core, block.shape[0]), positions


@pytest.mark.parametrize("sp_pad", [0, 24])
@pytest.mark.parametrize("split_moe", [False, True])
@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_decoder_layer_microbatch_matches_tp(
    microbatch_env, monkeypatch, layer_type, split_moe, sp_pad
):
    """The micro-batched sequence-parallel layer (two token halves through a
    causal core) is bitwise equal to the plain sequence-parallel layer, and
    both match the replicated TP layer."""
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as adaptive_moe,
    )

    tp = _primitives(microbatch_env, False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MOE", split_moe, raising=False)
    layers = {}

    def fake_adaptive(method, block, w1, w2, s1, s2, logits, attr):
        return layers[tp.rank].mlp.routed(block, logits)

    monkeypatch.setattr(adaptive_moe, "run_adaptive_fused_moe", fake_adaptive)
    md = _metadata()
    torch.random.default_generator.manual_seed(4)
    attn_hc, mlp_hc = hc_module(), hc_module()
    attn_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    moe_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    hidden = torch.randn(TOKENS, HC * HIDDEN).bfloat16()
    input_ids = torch.zeros(TOKENS, dtype=torch.long)
    positions = torch.arange(TOKENS)
    valid = torch.arange(TOKENS) < STARTS[-1]
    hidden[~valid] = float("nan")
    if layer_type == "linear_attention":
        init_state = torch.randn(16, 2 * HIDDEN, dtype=torch.float64)
    else:
        init_state = torch.randn(len(CONTEXT), 48, HIDDEN, dtype=torch.float64)

    def run(rank, sequence_parallel):
        layer = make_layer(tp, layer_type, attn_hc, mlp_hc, attn_w, moe_w)
        attention = CausalPartialAttention(tp, attn_w, layer_type, md, init_state)
        if layer_type == "linear_attention":
            layer.linear_attn = attention
        else:
            layer.self_attn = attention
        layer.mlp = SplitMoE(tp, moe_w)
        if split_moe:
            layer.mlp.shared_expert_local = lambda b: torch.sin(b.float() @ moe_w).to(
                b.dtype
            )
        layers[rank] = layer
        x = sp.shard(hidden) if sequence_parallel else hidden
        pad = sp_pad if sequence_parallel else 0
        plan = (
            sp.MicroBatchPlan(TOKENS) if sequence_parallel and tp.microbatch else None
        )
        out = layer(
            sp.pad_shard(x, pad),
            positions,
            input_ids,
            md.query_start_loc,
            None,
            None,
            md.context,
            valid,
            None,
            None,
            sequence_parallel=sequence_parallel,
            sp_pad=pad,
            sp_microbatch=plan,
        )
        expected_halves = [0, 1] if layer_type == "linear_attention" else [False, False]
        assert attention.halves == (expected_halves if plan is not None else [])
        out = sp.unpad(out, x.shape[0])
        # Back to token order: this rank's rows of each half.
        return sp.gather(out, TOKENS) if sequence_parallel else out

    replicated = tp.run(lambda r: run(r, False))
    sharded = tp.run(lambda r: run(r, True))
    monkeypatch.setattr(sp, "_microbatch_decision", True)
    tp.microbatch = True
    halves = tp.run(lambda r: run(r, True))
    for got_sp, got_mb in zip(sharded, halves):
        torch.testing.assert_close(got_mb[valid], got_sp[valid], rtol=0, atol=0)
        # The replicated shared expert sums in one bf16 rounding, the TP path in
        # two (as in test_decoder_layer_sequence_parallel_matches_tp).
        tol = 3e-2 if split_moe else 0
        torch.testing.assert_close(
            got_sp[valid], replicated[0][valid], rtol=tol, atol=tol
        )


def test_microbatch_layout_round_trips(tp2_primitives):
    """shard / gather / reduce agree on the layout, on and off."""
    tp = tp2_primitives
    x = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)

    def body(rank):
        mine = sp.shard(x)
        if tp.microbatch:  # rank r: its rows of half A, then of half B
            expected = torch.cat(
                [x[8 * rank : 8 * rank + 8], x[16 + 8 * rank : 16 + 8 * rank + 8]]
            )
        else:
            expected = x[16 * rank : 16 * rank + 16]
        assert torch.equal(mine, expected)
        assert torch.equal(sp.gather(mine, 32), x)
        assert torch.equal(sp.reduce(x * (rank + 1), True), 3 * mine)
        return True

    assert all(tp.run(body))


def test_microbatch_off_keeps_one_collective_per_call(microbatch_env):
    """Flag off: shard, gather and reduce issue exactly today's single
    collective each, on the whole tensor (the graph is unchanged)."""
    mp = microbatch_env
    calls = []
    mp.setattr(sp, "_microbatch_decision", False)
    mp.setattr(sp, "tp_world_size", lambda: 2)
    mp.setattr(sp, "chunk_rows", lambda x: calls.append(("chunk", x.shape)) or x)
    mp.setattr(sp, "all_gather_rows", lambda x: calls.append(("ag", x.shape)) or x)
    mp.setattr(sp, "reduce_scatter_rows", lambda x: calls.append(("rs", x.shape)) or x)
    x = torch.zeros(16, 4)
    sp.shard(x), sp.gather(x, 16), sp.reduce(x, True)
    assert calls == [("chunk", (16, 4)), ("ag", (16, 4)), ("rs", (16, 4))]


def test_microbatch_plan_traces_without_graph_breaks(microbatch_env):
    """The micro-batch layout and the half metadata trace in one dynamo graph
    with a symbolic token count (the SP graph's dynamic dim), using
    single-process stand-ins for the collectives."""
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import _half_step

    mp = microbatch_env
    mp.setattr(sp, "_microbatch_decision", True)
    mp.setattr(sp, "tp_world_size", lambda: 2)
    mp.setattr(sp, "chunk_rows", lambda x: x.narrow(0, 0, x.shape[0] // 2))
    mp.setattr(sp, "all_gather_rows", lambda x: torch.cat([x, x]))
    mp.setattr(sp, "reduce_scatter_rows", lambda x: 2 * x.narrow(0, 0, x.shape[0] // 2))
    md = _metadata()

    # The runner's metadata enters the graph as an input object.
    attn_md = NS(query_start_loc=md.query_start_loc, seq_lens=md.seq_lens)

    def step(hidden, positions, starts, context, distribution, slot_read):
        num_tokens = hidden.shape[0]
        plan = sp.MicroBatchPlan(num_tokens)
        shard = sp.shard(hidden)
        out = []
        for g in range(plan.groups):
            rows = sp.all_gather_rows(sp.group_rows(shard, plan.groups)[g])
            q_starts, q_context = plan.qsa(starts, context, g)
            key, g_starts, g_lens, g_dist, g_read = _half_step(
                (plan, g), attn_md, md.state_ids, distribution, slot_read
            )
            assert key is plan.gdn(attn_md, None, None, None, g)
            scale = sum(
                x.sum() for x in (q_starts, q_context, g_starts, g_lens, g_dist, g_read)
            ).to(rows.dtype)
            mixed = rows * scale + plan.positions(positions, g)[:, None].to(rows.dtype)
            out.append(sp.reduce_scatter_rows(mixed))
        return sp.gather(torch.cat(out), num_tokens)

    args = (
        torch.randn(32, 4),
        torch.arange(32),
        md.query_start_loc,
        md.context,
        md.distribution,
        md.slot_read,
    )
    torch._dynamo.reset()
    compiled = torch.compile(step, backend="eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(*args), step(*args), rtol=0, atol=0)
    bigger = (torch.randn(64, 4), torch.arange(64), *args[2:])
    torch.testing.assert_close(compiled(*bigger), step(*bigger), rtol=0, atol=0)
