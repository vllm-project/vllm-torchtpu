# SPDX-License-Identifier: Apache-2.0
"""Two-pass MTP draft step 0 (TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS).

CPU tests of the proposer side: when the two-pass path is taken, the narrowed
pass-2 metadata, and that a propose with the flag on leaves the same draft
tokens, KV / indexer-K caches and IndexShare rows as the single full pass.
The draft is a toy that keeps the property the argument rests on: rows couple
only through a causal read of caches every row writes first.
"""

import contextlib
import dataclasses
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import vllm_torchtpu.spec_decode.eagle3 as e3
from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs, Eagle3Proposer

DRAFT_LAYER = "model.layers.78.mtp_block.self_attn.attn"
HIDDEN = 4
VOCAB = 32
TOPK = 2


def _make_proposer(method: str = "mtp") -> Eagle3Proposer:
    speculative_config = SimpleNamespace(
        draft_tensor_parallel_size=1, method=method, draft_model_config=None
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return Eagle3Proposer(
        runner=mock.MagicMock(_pcp_enabled=False, _mtp_enabled=method == "mtp"),
        vllm_config=vllm_config,
    )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def _gating_proposer(monkeypatch, *, min_tokens=8, lockstep=False, target_md=True):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", min_tokens)
    proposer = _make_proposer()
    proposer._mtp_kv_only_model = object()
    proposer._mtp_kv_only_min_tokens = min_tokens
    proposer._dp_lockstep_sharded = lambda: lockstep
    proposer._target_draft_attn_metadata = lambda chunk: (
        {DRAFT_LAYER: object()} if target_md else None
    )
    return proposer


@pytest.mark.parametrize(
    "kwargs, num_tokens, num_rows, expected",
    [
        ({}, 16, 4, True),
        ({}, 8, 4, True),  # exactly the gate
        ({}, 7, 4, False),  # below the gate
        ({"min_tokens": 4}, 4, 4, False),  # nothing to drop: T == p
        ({"min_tokens": 0}, 4096, 16, False),  # flag off
        ({"lockstep": True}, 16, 4, False),  # EP-DP dummy replay
        ({"target_md": False}, 16, 4, False),  # dummy chunk / no target md
    ],
)
def test_use_kv_only_two_pass_gate(monkeypatch, kwargs, num_tokens, num_rows, expected):
    proposer = _gating_proposer(monkeypatch, **kwargs)
    assert proposer._use_kv_only_two_pass(object(), num_tokens, num_rows) is expected


def test_use_kv_only_two_pass_needs_the_program(monkeypatch):
    proposer = _gating_proposer(monkeypatch)
    proposer._mtp_kv_only_model = None
    assert proposer._use_kv_only_two_pass(object(), 4096, 16) is False


def _build_proposer(monkeypatch, *, min_tokens=1024, method="mtp"):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", min_tokens)
    proposer = _make_proposer(method)
    proposer.draft_model = object()
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(
        _pcp_enabled=False,
        mesh=object(),
        _dp_lockstep_enabled=lambda: False,
        # Loop buckets up to 32 (max_num_reqs 32).
        num_tokens_paddings=[16, 32, 64, 128, 1024, 4096],
        max_num_reqs=32,
    )
    proposer._draft_attn_layer_names = {"mtp.attn", "mtp.indexer.k_cache"}
    return proposer


def _record_logs(monkeypatch):
    logs = []
    for level in ("info", "warning"):
        monkeypatch.setattr(
            e3.logger,
            level,
            lambda msg, *args, _level=level, **kw: logs.append((_level, msg % args)),
        )
    return logs


def _patch_builder(monkeypatch, reason=None, raises=False):
    import vllm_torchtpu.spec_decode.mtp_kv_only as kv_only

    built = []

    class _FakeModel:
        def __init__(self, **kwargs):
            if raises:
                raise RuntimeError("boom")
            built.append(kwargs)

        def cache_layer_names(self):
            return ("mtp.attn", "mtp.indexer.k_cache")

    monkeypatch.setattr(kv_only, "kv_only_unsupported_reason", lambda mtp: reason)
    monkeypatch.setattr(kv_only, "MtpKvOnlyModel", _FakeModel)
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        e3, "set_current_vllm_config", lambda *a, **k: contextlib.nullcontext()
    )
    return built


def test_build_kv_only_model_off_by_default(monkeypatch):
    proposer = _build_proposer(monkeypatch, min_tokens=0)
    built = _patch_builder(monkeypatch)
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is None
    assert built == []


def test_build_kv_only_model_builds_with_draft_config_and_own_prefix(monkeypatch):
    proposer = _build_proposer(monkeypatch)
    built = _patch_builder(monkeypatch)
    logs = _record_logs(monkeypatch)
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is not None
    # Above every loop bucket: the gate is the flag, as set.
    assert proposer._mtp_kv_only_min_tokens == 1024
    assert [lvl for lvl, _ in logs] == ["info"]
    assert ">= 1024 tokens" in logs[0][1]
    (kwargs,) = built
    assert kwargs["vllm_config"] is proposer.draft_vllm_config
    assert kwargs["mtp"] is proposer.draft_model
    assert kwargs["mesh"] is proposer.runner.mesh
    assert kwargs["prefix"] == "mtp_prefill_kv_only"


@pytest.mark.parametrize("min_tokens", [1, 16, 32])
def test_build_kv_only_model_raises_the_gate_above_the_loop_buckets(
    monkeypatch, min_tokens
):
    """A step 0 no larger than the largest loop bucket keeps the single pass:
    the warmup's single-request specs at those sizes warm the draft at p = T."""
    proposer = _build_proposer(monkeypatch, min_tokens=min_tokens)
    _patch_builder(monkeypatch)
    logs = _record_logs(monkeypatch)
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is not None
    assert proposer._mtp_kv_only_min_tokens == 33
    ((lvl, msg), (lvl2, msg2)) = logs
    assert lvl == "warning" and "largest draft loop bucket (32)" in msg
    assert lvl2 == "info" and ">= 33 tokens" in msg2
    proposer._dp_lockstep_sharded = lambda: False
    proposer._target_draft_attn_metadata = lambda chunk: {DRAFT_LAYER: object()}
    assert proposer._use_kv_only_two_pass(object(), 32, 1) is False
    assert proposer._use_kv_only_two_pass(object(), 64, 1) is True


@pytest.mark.parametrize(
    "setup",
    [
        "eagle3",
        "kv_shared",
        "pcp",
        "lockstep",
        "unsupported",
        "build_fails",
        "not_draft_layers",
    ],
)
def test_build_kv_only_model_refusals_keep_the_single_pass(monkeypatch, setup):
    proposer = _build_proposer(
        monkeypatch, method="eagle3" if setup == "eagle3" else "mtp"
    )
    built = _patch_builder(
        monkeypatch,
        reason="not DSA" if setup == "unsupported" else None,
        raises=setup == "build_fails",
    )
    if setup == "kv_shared":
        proposer._draft_kv_shared = True
    if setup == "pcp":
        proposer.runner._pcp_enabled = True
    if setup == "lockstep":
        proposer._dp_lockstep_sharded = lambda: True
    logs = _record_logs(monkeypatch)
    if setup == "not_draft_layers":
        # The indexer cache is not among the draft's attention layers, so the
        # step-0 metadata pass 1 reads would not carry it.
        proposer._draft_attn_layer_names = {"mtp.attn"}
    proposer._build_mtp_kv_only_model()
    assert proposer._mtp_kv_only_model is None
    assert proposer._mtp_kv_only_min_tokens == 0
    if setup != "not_draft_layers":
        assert built == []
    # The log says why, and never claims the two-pass step 0 is active.
    assert logs[-1][0] == "warning" and "keeps the single full pass" in logs[-1][1]
    assert not any(lvl == "info" for lvl, _ in logs)
    if setup == "lockstep":
        assert "EP-DP lockstep" in logs[-1][1]


# ---------------------------------------------------------------------------
# Pass-2 metadata
# ---------------------------------------------------------------------------


def _ctx(seq_lens, qsl, num_reqs, knr):
    seq = np.zeros(knr, dtype=np.int32)
    seq[: len(seq_lens)] = seq_lens
    qsl_full = np.full(knr + 1, qsl[-1], dtype=np.int32)
    qsl_full[: len(qsl)] = qsl
    return SimpleNamespace(
        use_max_model_len=True,
        seq_lens=torch.from_numpy(seq),
        query_start_loc=torch.from_numpy(qsl_full),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
        position_ids_override=None,
    )


@pytest.mark.parametrize("async_rejected", [False, True])
def test_pass2_attn_metadata_is_one_query_per_request(async_rejected):
    proposer = _make_proposer()
    knr = 6
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=knr, num_reqs_most_model_len=None, device="cpu"
    )
    chunk = SimpleNamespace(
        num_reqs=3, attn_ctx=_ctx([10, 6, 8], [0, 4, 10, 15], 3, knr)
    )
    md = AttentionMetadata(
        input_positions=torch.zeros(16, dtype=torch.int32),
        block_tables=torch.arange(12, dtype=torch.int32),
        seq_lens=chunk.attn_ctx.seq_lens,
        query_start_loc=chunk.attn_ctx.query_start_loc,
        request_distribution=chunk.attn_ctx.request_distribution,
    )
    other = AttentionMetadata(input_positions=torch.zeros(16, dtype=torch.int32))
    target_md = {"a": md, "b": md, "c": other}
    rejected = np.array([2, 0, 0], dtype=np.int32)
    if async_rejected:
        rejected = torch.from_numpy(rejected)

    loop_inputs = proposer._chunk_loop_inputs(chunk, rejected)
    out = proposer._pass2_attn_metadata(chunk, target_md, rejected, loop_inputs)

    assert out["a"] is out["b"]  # shared metadata stays shared
    new = out["a"]
    assert new.block_tables is md.block_tables
    assert new.seq_lens.tolist() == [8, 6, 8, 0, 0, 0]
    assert new.query_start_loc.tolist() == [0, 1, 2, 3, 3, 3, 3]
    assert new.request_distribution.tolist() == [3, 3, 3]
    # The loop steps reuse the same query starts and distribution.
    assert new.query_start_loc is loop_inputs[1]
    assert new.request_distribution is loop_inputs[2]
    # The target's own metadata is untouched.
    assert md.seq_lens is chunk.attn_ctx.seq_lens
    assert md.query_start_loc is chunk.attn_ctx.query_start_loc


def test_save_topk_snapshots_the_front_rows():
    proposer = _make_proposer()
    buf = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    proposer._mtp_topk_buffers = [buf]
    (saved,) = proposer._mtp_save_topk(2)
    assert torch.equal(saved, buf[:2])
    buf.zero_()
    assert saved.sum() > 0  # a copy, not a view


# ---------------------------------------------------------------------------
# Flag on == flag off, on a toy draft
# ---------------------------------------------------------------------------


class _ToyWorld:
    """Per-request caches indexed by absolute position, plus the shared top-k
    table. `write` and `attend` are the only row couplings, as in the MTP
    layer (KV + indexer-K writes, causal sparse attention over them)."""

    def __init__(self, num_reqs: int, max_len: int, prefix_lens):
        g = torch.Generator().manual_seed(0)
        self.kv = torch.zeros(num_reqs, max_len, HIDDEN)
        self.idx = torch.zeros(num_reqs, max_len)
        for r, n in enumerate(prefix_lens):
            self.kv[r, :n] = torch.randn(n, HIDDEN, generator=g)
            self.idx[r, :n] = torch.randn(n, generator=g)
        self.topk_buf = torch.full((32, TOPK), -1, dtype=torch.int32)
        self.md = None
        self.skip_topk = False
        self.step0_rows_seen: list[int] = []
        self.kv_only_rows_seen: list[int] = []

    def rows(self, num_tokens: int, positions: torch.Tensor):
        """(row, request, position) of every row the metadata places."""
        (md,) = {id(m): m for m in self.md.values()}.values()
        qsl = md.query_start_loc.tolist()
        seq_lens = md.seq_lens.tolist()
        num_reqs = int(md.request_distribution[2])
        # The toy's "block table": entry r * PAGES_PER_REQ is request r's
        # batch-wide id, i.e. which cache rows it owns.
        owners = md.block_tables.tolist()
        out = []
        for t in range(num_tokens):
            r = next((r for r in range(num_reqs) if qsl[r] <= t < qsl[r + 1]), None)
            if r is None:
                continue
            pos = seq_lens[r] - (qsl[r + 1] - qsl[r]) + (t - qsl[r])
            # The metadata and the row's own position agree.
            assert pos == int(positions[t]), (t, r, pos, int(positions[t]))
            out.append((t, owners[r * PAGES_PER_REQ], pos))
        return out

    def write(self, input_ids, positions, hidden):
        placed = self.rows(input_ids.shape[0], positions)
        for t, r, pos in placed:
            self.kv[r, pos] = hidden[t] * 0.5 + input_ids[t] * 0.01 + pos * 0.001
            self.idx[r, pos] = hidden[t].sum() + input_ids[t] * 0.1
        return placed

    def forward(self, input_ids, positions, hidden_states, spec_step_idx=0):
        placed = self.write(input_ids, positions, hidden_states)
        out = hidden_states.clone()
        for t, r, pos in placed:
            if self.skip_topk:
                sel = self.topk_buf[t]
            else:
                scores = self.idx[r, : pos + 1]
                k = min(TOPK, pos + 1)
                sel = torch.full((TOPK,), -1, dtype=torch.int32)
                sel[:k] = torch.topk(scores, k).indices.to(torch.int32)
                self.topk_buf[t] = sel
            valid = sel[sel >= 0].long()
            out[t] = (
                hidden_states[t] * 0.9
                + self.kv[r, : pos + 1].mean(dim=0) * 0.3
                + self.kv[r, valid].sum(dim=0) * 0.2
            )
        if spec_step_idx == 0 and not self.skip_topk:
            self.step0_rows_seen.append(input_ids.shape[0])
        return out, out * 2.0


class _ToyMtpModel:
    def __init__(self, world):
        self.world = world

    def set_skip_topk(self, skip):
        self.world.skip_topk = skip

    def compact_topk_indices(self, slot_ids):
        buf = self.world.topk_buf
        buf[: slot_ids.numel()] = buf[slot_ids]


class _ToyDraft:
    def __init__(self, world):
        self.world = world
        self.model = _ToyMtpModel(world)
        g = torch.Generator().manual_seed(1)
        self.head = torch.randn(HIDDEN, VOCAB, generator=g)

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)

    def compute_logits(self, hidden):
        return hidden @ self.head


class _ToyLoopProgram:
    """IndexShare's loop-step program: the same forward, run with the
    indexer off (it reads the saved top-k rows)."""

    def __init__(self, world):
        self.world = world

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        assert self.world.skip_topk
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)


class _ToyFusedLoop:
    """The fused IndexShare program: draft steps 1..K-1 of one chunk in one
    call, indexer off (it reads the rows restored before it), bumping the
    loop metadata's seq_lens in place between steps."""

    def __init__(self, world, proposer, num_steps):
        self.world = world
        self.proposer = proposer
        self.num_steps = num_steps
        self.calls = 0

    def __call__(self, input_ids, positions, hidden_states, seq_lens):
        assert self.world.skip_topk
        self.calls += 1
        tokens = []
        for step in range(1, self.num_steps + 1):
            last_hidden, hidden_states = self.world.forward(
                input_ids, positions, hidden_states, spec_step_idx=step
            )
            input_ids = self.proposer._draft_propose_token(last_hidden)
            tokens.append(input_ids)
            positions = positions + 1
            seq_lens.add_(1)
        return tuple(tokens)


class _ToyKvOnly:
    def __init__(self, world):
        self.world = world
        self.calls = 0

    def __call__(self, input_ids, positions, hidden_states):
        self.calls += 1
        placed = self.world.write(input_ids, positions, hidden_states)
        self.world.kv_only_rows_seen.append(len(placed))
        return hidden_states


class _Req:
    def __init__(self, tokens, num_computed):
        self.tokens = tokens
        self.num_computed_tokens = num_computed

    def get_token_id(self, i):
        return self.tokens[i]


PAGES_PER_REQ = 4
KNR = 4
GATE = 9


@dataclasses.dataclass
class _ReqSpec:
    rid: str
    context: int
    q_len: int
    rejected: int = 0
    # Sync: the sampled tokens; async: the rejection-sampler row.
    sampled: tuple = ()
    async_row: tuple = (0, -1, -1, -1)
    num_drafts: int = 0
    discard: bool = False


# Chunk 0 (16 padded tokens, at the gate or above -> two-pass), K = 3:
#   r0 verify: context 6, 1 + K = 4 tokens (positions 6..9), 2 drafts rejected;
#   r1 prefill whose prompt completes: 6 tokens (positions 0..5);
#   r2 intermediate prefill chunk: 5 tokens (positions 3..7), discarded.
_CHUNK0 = [
    _ReqSpec("r0", 6, 4, 2, (11, 12), (11, 12, -1, -1), 3),
    _ReqSpec("r1", 0, 6, 0, (13,), (13, -1, -1, -1)),
    _ReqSpec("r2", 3, 5, discard=True),
]
# Chunk 1 (8 tokens, below the gate -> single pass): two verify requests.
_CHUNK1 = [
    _ReqSpec("r3", 4, 4, 1, (21, 22, 23), (21, 22, 23, -1), 3),
    _ReqSpec("r4", 2, 4, 0, (31, 32, 33, 34), (31, 32, 33, 34), 3),
]


def _chunk(specs, start, padded):
    num_reqs = len(specs)
    q_lens = [sp.q_len for sp in specs]
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    seq_lens = [sp.context + sp.q_len for sp in specs]
    positions = np.zeros(padded, dtype=np.int32)
    for r, sp in enumerate(specs):
        positions[qsl[r] : qsl[r + 1]] = np.arange(sp.context, seq_lens[r])
    real = int(qsl[-1])
    input_ids = np.zeros(padded, dtype=np.int32)
    input_ids[:real] = (np.arange(real) + 5 * start) % 7 + 3
    g = torch.Generator().manual_seed(2 + start)
    hidden = torch.randn(padded, HIDDEN, generator=g)
    owners = np.zeros(KNR * PAGES_PER_REQ, dtype=np.int32)
    owners[: num_reqs * PAGES_PER_REQ : PAGES_PER_REQ] = np.arange(
        start, start + num_reqs
    )
    md = AttentionMetadata(
        input_positions=torch.from_numpy(positions),
        block_tables=torch.from_numpy(owners),
        seq_lens=torch.tensor(seq_lens + [0] * (KNR - num_reqs), dtype=torch.int32),
        query_start_loc=torch.tensor(
            list(qsl) + [qsl[-1]] * (KNR - num_reqs), dtype=torch.int32
        ),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
    )
    attn_ctx = SimpleNamespace(
        use_max_model_len=True,
        seq_lens=md.seq_lens,
        query_start_loc=md.query_start_loc,
        request_distribution=md.request_distribution,
        position_ids_override=None,
    )
    return DraftChunkInputs(
        input_ids=torch.from_numpy(input_ids),
        position_ids=torch.from_numpy(positions),
        query_start_loc_np=qsl,
        attn_ctx=attn_ctx,
        start_index=start,
        num_reqs=num_reqs,
        aux_hidden_states=[],
        hidden_states=hidden,
        attn_metadata={DRAFT_LAYER: md, "target.attn": md},
        draft_lengths=torch.tensor([sp.num_drafts for sp in specs], dtype=torch.int32),
    )


def _scenario(mode: str, num_chunks: int):
    chunk_specs = [_CHUNK0, _CHUNK1][:num_chunks]
    specs = [sp for cs in chunk_specs for sp in cs]
    chunks, start = [], 0
    for cs, padded in zip(chunk_specs, (16, 8)):
        chunks.append(_chunk(cs, start, padded))
        start += len(cs)
    requests = {
        sp.rid: _Req(list(range(100, 120)), sp.context) for sp in specs if sp.discard
    }
    kwargs = dict(
        discard_sampled_tokens_req_indices=[
            i for i, sp in enumerate(specs) if sp.discard
        ],
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens={sp.rid: sp.q_len for sp in specs},
            scheduled_spec_decode_tokens={},
        ),
    )
    if mode == "sync":
        kwargs.update(
            sampled_token_ids=[list(sp.sampled) for sp in specs],
            num_rejected_tokens_np=np.array(
                [sp.rejected for sp in specs], dtype=np.int32
            ),
        )
    else:
        # The rejection-sampler rows that give the same seeds and rejections.
        kwargs.update(
            sampled_token_ids=None,
            num_rejected_tokens_np=None,
            next_tokens_per_chunk=[
                torch.tensor([sp.async_row for sp in cs], dtype=torch.int32)
                for cs in chunk_specs
            ],
            return_device=True,
        )
    return chunks, requests, kwargs, [sp.context for sp in specs], specs


# Eagle3Proposer's @torch.compile(backend="tpu") helpers that propose() calls.
_COMPILED_DRAFT_HELPERS = (
    "_draft_seed_input_ids",
    "_draft_combine_hidden_states",
    "_draft_gather_carries",
    "_draft_gather_rows",
    "_draft_propose_token",
)


def _eager_body(method):
    """The Python body under `_static_shape_args(torch.compile(...)(fn))`
    (idempotent: `_run` runs twice per test)."""
    compiled = getattr(method, "__wrapped__", method)
    return getattr(compiled, "_torchdynamo_orig_callable", compiled)


def _run(
    monkeypatch,
    *,
    two_pass: bool,
    mode: str,
    num_chunks: int = 1,
    topk_inplace: bool = True,
    fused: bool = False,
):
    import torch._dynamo

    from vllm_torchtpu.kernels import partial_kv_update

    torch._dynamo.reset()
    # propose() runs its small helpers in @torch.compile(backend="tpu")
    # regions, which need a TPU device; on the CPU lane run their bodies
    # eagerly (the TPU equality run covers the compiled path).
    for name in _COMPILED_DRAFT_HELPERS:
        monkeypatch.setattr(
            e3.Eagle3Proposer, name, _eager_body(getattr(e3.Eagle3Proposer, name))
        )
    # TPU_MTP_TOPK_INPLACE: snapshot by gather, restore in place. The Pallas
    # scatter is TPU-only; index_copy_ writes the same rows.
    monkeypatch.setattr(envs, "TPU_MTP_TOPK_INPLACE", topk_inplace)
    monkeypatch.setattr(
        partial_kv_update,
        "pallas_index_copy_",
        lambda buf, ids, rows: buf.index_copy_(0, ids.long(), rows),
    )
    chunks, requests, kwargs, prefix_lens, specs = _scenario(mode, num_chunks)
    world = _ToyWorld(len(specs), 24, prefix_lens)
    K = 3

    proposer = _make_proposer()
    proposer.speculative_config.num_speculative_tokens = K
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=len(specs), req_ids=[sp.rid for sp in specs]
        ),
        requests=requests,
        device=torch.device("cpu"),
        num_tokens_paddings=[4, 8, 16, 32],
        max_num_reqs=4,
        num_reqs_max_model_len=KNR,
        num_reqs_most_model_len=None,
        uses_mrope=False,
        mesh=object(),
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: n,
        _has_mamba_state=False,
        _unified_kv_layout=False,
        # TPU_H2D_REUSE_STEADY's upload (no reuse here: a plain copy).
        _upload_unless_unchanged=lambda slot, host, device=None: host.to("cpu"),
    )
    proposer._draft_attn_layer_names = {DRAFT_LAYER}
    proposer.draft_model = _ToyDraft(world)
    proposer.draft_chunks = chunks
    # IndexShare, on the toy's shared top-k table.
    proposer._share_mtp_indices = True
    proposer._mtp_topk_buffers = [world.topk_buf]
    proposer._mtp_loop_model = _ToyLoopProgram(world)
    proposer._mtp_loop_model_warm = True
    fused_loop = _ToyFusedLoop(world, proposer, K - 1)
    if fused:
        # VLLM_TPU_MTP_FUSED_INDEXSHARE: steps 1..K-1 in one program, fed each
        # chunk's saved step-0 rows.
        proposer._mtp_fused_loop = fused_loop
        proposer._mtp_fused_indexshare = True
        proposer._mtp_fused_loop_sizes = None
        proposer._mtp_fused_loop_warm = False

    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", GATE)
    kv_only = _ToyKvOnly(world)
    proposer._mtp_kv_only_model = kv_only if two_pass else None
    proposer._mtp_kv_only_min_tokens = GATE if two_pass else 0

    @contextlib.contextmanager
    def fake_forward_context(attn_metadata, *_a, **_k):
        world.md = attn_metadata
        yield

    monkeypatch.setattr(e3, "set_forward_context", fake_forward_context)
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())

    drafts = proposer.propose(**kwargs)
    if isinstance(drafts, torch.Tensor):
        drafts = drafts.tolist()
    return SimpleNamespace(
        drafts=drafts,
        world=world,
        kv_only_calls=kv_only.calls,
        fused_calls=fused_loop.calls,
    )


@pytest.mark.parametrize("fused", [False, True], ids=["separate", "fused_indexshare"])
@pytest.mark.parametrize("topk_inplace", [True, False], ids=["inplace", "copy"])
@pytest.mark.parametrize("num_chunks", [1, 2])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_two_pass_step0_matches_the_single_pass(
    monkeypatch, mode, num_chunks, topk_inplace, fused
):
    opts = dict(mode=mode, num_chunks=num_chunks, topk_inplace=topk_inplace)
    ref = _run(monkeypatch, two_pass=False, fused=fused, **opts)
    new = _run(monkeypatch, two_pass=True, fused=fused, **opts)
    if fused:
        # Steps 1..K-1 ran in the program for every chunk (the first call
        # runs it twice: trace, then compiled).
        assert ref.fused_calls == new.fused_calls == num_chunks + 1
        # And match the separate per-step loop.
        sep = _run(monkeypatch, two_pass=False, fused=False, **opts)
        assert new.drafts == sep.drafts
        assert torch.equal(new.world.kv, sep.world.kv)

    # The paths actually differ: one full step 0 on all 16 rows vs a KV-only
    # pass over the 15 real rows and a full pass on p = 4 rows. A second
    # chunk (8 tokens) is below the gate and keeps the single pass.
    assert ref.kv_only_calls == 0
    assert ref.world.step0_rows_seen == [16, 8][:num_chunks]
    assert new.kv_only_calls == 1
    assert new.world.kv_only_rows_seen == [15]
    assert new.world.step0_rows_seen == [4, 8][:num_chunks]

    assert new.drafts == ref.drafts
    # Same cache entries everywhere: prefix, this step's rows (rejected tail
    # included), and the K - 1 loop steps' draft entries.
    assert torch.equal(new.world.kv, ref.world.kv)
    assert torch.equal(new.world.idx, ref.world.idx)
    # The last chunk's per-request top-k rows (IndexShare) are the same.
    num_last = len(_CHUNK0 if num_chunks == 1 else _CHUNK1)
    assert torch.equal(new.world.topk_buf[:num_last], ref.world.topk_buf[:num_last])


def test_two_pass_step0_is_not_taken_below_the_gate(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", 32)
    proposer = _make_proposer()
    proposer._mtp_kv_only_model = object()
    proposer._mtp_kv_only_min_tokens = 32
    proposer._dp_lockstep_sharded = lambda: False
    proposer._target_draft_attn_metadata = lambda chunk: {DRAFT_LAYER: object()}
    assert proposer._use_kv_only_two_pass(object(), 16, 4) is False
