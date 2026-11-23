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

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_torchtpu.spec_decode import utils


@pytest.fixture(autouse=True)
def _gather_exchange(monkeypatch):
    """These tests cover the all-gather exchange; TPU_MTP_ONEHOT_AR (on by
    default) needs an initialized TP group and per-rank slots, and its tests
    below build their slots explicitly."""
    monkeypatch.setenv("TPU_MTP_ONEHOT_AR", "0")


def test_own_vocab_weights_are_kept_when_tp_layout_matches():
    target_embed = object()
    target_head = object()
    draft_embed = object()
    draft_head = object()
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_head,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)
    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    assert draft.model.embed_tokens is draft_embed
    assert draft.lm_head is draft_head


@pytest.mark.parametrize("has_own_embed_tokens", [None, True])
def test_draft_keeps_own_embed_when_target_embed_is_on_another_pp_stage(
    has_own_embed_tokens,
):
    # Only the first PP stage builds a DeepseekV2 target's embed_tokens.
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PPMissingLayer()))
    draft_embed = object()
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft_embed))
    if has_own_embed_tokens is not None:
        draft.has_own_embed_tokens = has_own_embed_tokens

    utils.maybe_share_embeddings(draft, target, tp_layout_matches=True)

    assert draft.model.embed_tokens is draft_embed


@pytest.mark.parametrize(
    ("draft_embed", "has_own_embed_tokens", "force_share"),
    [(None, None, False), (object(), None, True), (object(), False, False)],
    ids=["no-draft-embed", "force-share", "declares-no-own-embed"],
)
def test_draft_needing_target_embed_raises_when_it_is_on_another_pp_stage(
    draft_embed, has_own_embed_tokens, force_share
):
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PPMissingLayer()))
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=draft_embed))
    if has_own_embed_tokens is not None:
        draft.has_own_embed_tokens = has_own_embed_tokens

    with pytest.raises(RuntimeError, match="not on this pipeline stage"):
        utils.maybe_share_embeddings(
            draft, target, tp_layout_matches=True, force_share=force_share
        )


def test_handoff_seed_tokens_replaces_only_rows_with_a_published_send():
    from types import SimpleNamespace

    seed = torch.tensor([11, 22, 33], dtype=torch.int32)
    metadata = SimpleNamespace(
        reqs_to_send={
            "a": SimpleNamespace(next_prompt_token_id=7),
            "c": SimpleNamespace(next_prompt_token_id=None),
        }
    )
    out = utils.handoff_seed_tokens(seed, ["a", "b", "c"], metadata)
    assert out.tolist() == [7, 22, 33]
    assert seed.tolist() == [11, 22, 33]
    # No metadata, or no send with a token: the seed is returned as is.
    assert utils.handoff_seed_tokens(seed, ["a"], None) is seed
    assert (
        utils.handoff_seed_tokens(
            seed, ["b"], SimpleNamespace(reqs_to_send={"a": object()})
        )
        is seed
    )


class _SharedHeadLayer(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.shared_head = torch.nn.Module()
        self.shared_head.head = torch.nn.Linear(hidden, vocab, bias=False)


def _make_mtp_draft(num_layers: int = 1, hidden: int = 2, vocab: int = 3):
    """A DeepSeek/GLM-5.2-shaped MTP draft: no top-level lm_head, one head per
    MTP layer at model.layers.<N>.shared_head.head, LogitsProcessor on the
    inner model."""
    layers = torch.nn.ModuleDict(
        {str(78 + i): _SharedHeadLayer(hidden, vocab) for i in range(num_layers)}
    )
    inner = torch.nn.Module()
    inner.layers = layers
    inner.logits_processor = SimpleNamespace(
        _gather_logits=lambda logits: logits.clone()
    )
    draft = torch.nn.Module()
    draft.model = inner
    return draft


def _make_target(hidden: int = 2, vocab: int = 3):
    return SimpleNamespace(lm_head=torch.nn.Linear(hidden, vocab, bias=False))


def test_mtp_shared_head_is_bound_to_target_lm_head():
    """G1: DeepSeekMTP keeps its head at model.layers.<N>.shared_head.head, so
    the plain lm_head attribute walk never reaches it."""
    draft = _make_mtp_draft(num_layers=2)
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=False, tp_layout_matches=True
    )

    heads = [sh.head for sh in utils.iter_mtp_shared_heads(draft)]
    assert len(heads) == 2
    assert all(h is target.lm_head for h in heads)
    # No dead top-level attribute is invented when shared heads were bound.
    assert not hasattr(draft, "lm_head")


def test_mtp_replicated_draft_overrides_inner_logits_processor():
    """G2: DeepSeekMTP's LogitsProcessor lives on the inner model, so the
    top-level getattr used to return None and skip the TP-gather override."""
    draft = _make_mtp_draft()
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    logits = torch.randn(2, 3)
    assert draft.model.logits_processor._gather_logits(logits) is logits


def test_replicated_draft_without_logits_processor_raises():
    draft = _make_mtp_draft()
    del draft.model.logits_processor
    target = _make_target()

    with pytest.raises(RuntimeError, match="LogitsProcessor"):
        utils.maybe_share_lm_head(
            draft, target, draft_replicated=True, tp_layout_matches=True
        )


def test_top_level_lm_head_draft_is_unchanged():
    """Regression: eagle3 / Qwen3.5-MTP drafts have a top-level lm_head and no
    shared_head, so the shared-head walk must be a no-op for them."""
    draft = SimpleNamespace(
        model=SimpleNamespace(),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        logits_processor=SimpleNamespace(_gather_logits=lambda logits: logits.clone()),
    )
    target = _make_target()

    utils.maybe_share_lm_head(
        draft, target, draft_replicated=True, tp_layout_matches=True
    )

    assert draft.lm_head is target.lm_head
    assert list(utils.iter_mtp_shared_heads(draft)) == []
    logits = torch.randn(2, 3)
    assert draft.logits_processor._gather_logits(logits) is logits


# --- TPU_SAMPLER_VOCAB_PARALLEL: vocab-parallel draft argmax ---


def _fake_sharded_head(start: int, end: int, tp: int):
    head = torch.nn.Module()
    head.tp_size = tp
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=start, org_vocab_end_index=end
    )
    return head


def test_install_vocab_parallel_draft_argmax_on_mtp_draft():
    draft = _make_mtp_draft(num_layers=1)
    draft.model.logits_processor.scale = 1.0
    head = _fake_sharded_head(9680 * 3, 9680 * 4, 16)
    for sh in utils.iter_mtp_shared_heads(draft):
        sh.head = head
    info = utils.install_vocab_parallel_draft_argmax(draft)
    assert info == utils.VocabParallelArgmax(9680 * 3, 9680, 16)
    logits = torch.randn(2, 3)
    assert draft.model.logits_processor._gather_logits(logits) is logits


def test_install_vocab_parallel_draft_argmax_skips_unsharded_head():
    draft = _make_mtp_draft(num_layers=1)
    draft.model.logits_processor.scale = 1.0
    gather = draft.model.logits_processor._gather_logits
    assert utils.install_vocab_parallel_draft_argmax(draft) is None
    assert draft.model.logits_processor._gather_logits is gather


@pytest.mark.parametrize("tp,width,pad", [(16, 9680, 0), (4, 37, 5), (2, 8, 0)])
def test_vocab_parallel_argmax_matches_full_argmax(monkeypatch, tp, width, pad):
    """Each rank reduces its shard and all-gathers (max, id); the result must be
    the full-vocab torch.argmax, first-occurrence ties across and within
    shards included, with padding columns never chosen."""
    import vllm.distributed as vd

    rows = 16
    g = torch.Generator().manual_seed(tp * 100 + width)
    full = torch.randn(rows, tp * width, generator=g).to(torch.bfloat16)
    org = tp * width - pad
    full[0, :] = 1.0  # all tied -> id 0
    full[1, width - 1] = full[1, width + 1] = 50.0  # tie across shards
    full[2, 3] = full[2, 5] = 50.0  # tie within a shard
    if pad:
        full[3, org:] = 100.0  # padding must lose
    shards = list(full.split(width, dim=-1))  # padding: end of global vocab
    infos = [
        utils.VocabParallelArgmax(r * width, max(0, min(width, org - r * width)), tp)
        for r in range(tp)
    ]
    pairs = []
    monkeypatch.setattr(
        vd,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: pairs.append(x) or x.repeat(tp, 1),
    )
    for r in range(tp):
        utils.vocab_parallel_argmax(shards[r], infos[r])
    gathered = torch.cat(pairs, dim=0)
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_gather", lambda x, dim=0: gathered
    )
    masked = full.clone()
    masked[:, org:] = float("-inf")
    expected = torch.argmax(masked, dim=-1).to(torch.int32)
    for r in range(tp):
        out = utils.vocab_parallel_argmax(shards[r], infos[r])
        assert torch.equal(out, expected), (r, out, expected)


def _simulated_all_reduce(per_rank_inputs):
    """What the small all-reduce returns: float32 sum over ranks in rank
    order, rounded once to the input dtype."""
    acc = torch.zeros_like(per_rank_inputs[0], dtype=torch.float32)
    for x in per_rank_inputs:
        acc = acc + x.float()
    return acc.to(per_rank_inputs[0].dtype)


@pytest.mark.parametrize("tp,width,pad", [(16, 9680, 0), (4, 37, 5), (2, 8, 0)])
def test_onehot_vocab_parallel_argmax_matches_full_argmax(monkeypatch, tp, width, pad):
    """TPU_MTP_ONEHOT_AR: the (max, id) exchange as a one-hot all-reduce gives
    the full-vocab torch.argmax, ties across and within shards included."""
    import vllm.distributed as vd

    rows = 16
    g = torch.Generator().manual_seed(tp * 7 + width)
    full = torch.randn(rows, tp * width, generator=g).to(torch.bfloat16)
    org = tp * width - pad
    full[0, :] = 1.0
    full[1, width - 1] = full[1, width + 1] = 50.0
    full[2, 3] = full[2, 5] = 50.0
    full[4, :] = -0.0
    full[4, width + 2] = 0.0  # signed zeros tie: lowest id wins
    full[5, :] = -3.0
    full[5, tp * width - pad - 1] = 7.0  # last real id (largest digits)
    if pad:
        full[3, org:] = 100.0
    shards = list(full.split(width, dim=-1))
    infos = [
        utils.VocabParallelArgmax(
            r * width,
            max(0, min(width, org - r * width)),
            tp,
            slot=utils.rank_slot(r, tp, "cpu"),
        )
        for r in range(tp)
    ]
    bufs = []
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_reduce", lambda x: bufs.append(x) or x
    )
    for r in range(tp):
        utils.vocab_parallel_argmax(shards[r], infos[r])
    assert all(b.shape == (rows, tp * 128) and b.dtype == torch.bfloat16 for b in bufs)
    summed = _simulated_all_reduce(bufs)
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", lambda x: summed)
    masked = full.clone()
    masked[:, org:] = float("-inf")
    expected = torch.argmax(masked, dim=-1).to(torch.int32)
    for r in range(tp):
        out = utils.vocab_parallel_argmax(shards[r], infos[r])
        assert out.dtype == torch.int32
        assert torch.equal(out, expected), (r, out, expected)


def test_onehot_all_gather_is_the_gather(monkeypatch):
    """Each rank's block lands at its column slot, bitwise (bar -0)."""
    import vllm.distributed as vd

    tp, rows, n = 16, 4, 384
    g = torch.Generator().manual_seed(3)
    blocks = [
        (torch.randn(rows, n, generator=g) * 1e3).to(torch.bfloat16) for _ in range(tp)
    ]
    bufs = []
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_reduce", lambda x: bufs.append(x) or x
    )
    for r in range(tp):
        utils.onehot_all_gather(blocks[r], utils.rank_slot(r, tp, "cpu"))
    out = _simulated_all_reduce(bufs)
    assert torch.equal(out, torch.cat(blocks, dim=-1))


@pytest.mark.parametrize(
    "tp,rows,lead3", [(16, 16, False), (8, 5, False), (4, 6, True)]
)
def test_column_parallel_eh_proj_matches_replicated(monkeypatch, tp, rows, lead3):
    """TPU_MTP_EH_PROJ_TP: every rank's gathered output equals the replicated
    nn.Linear, for 2-D and 3-D inputs. Small-integer f32 values make every dot
    product exact under any summation order, so equality tests the column
    split and the gather layout, not BLAS blocking."""
    import vllm.distributed as vd

    hidden = 8 * tp
    g = torch.Generator().manual_seed(tp * 7 + rows)
    full = torch.nn.Linear(2 * hidden, hidden, bias=False)
    with torch.no_grad():
        full.weight.copy_(
            torch.randint(-4, 5, (hidden, 2 * hidden), generator=g).float()
        )
    shape = (2, rows, 2 * hidden) if lead3 else (rows, 2 * hidden)
    x = torch.randint(-4, 5, shape, generator=g).float()
    expected = full(x).detach()
    mods = [utils.ColumnParallelEhProj(full.weight, r, tp) for r in range(tp)]
    for r, m in enumerate(mods):
        n = hidden // tp
        assert m.weight.shape == (n, 2 * hidden)
        assert torch.equal(m.weight, full.weight[r * n : (r + 1) * n])
    # all-gather(dim=0) = concatenation of every rank's local [rows, n] block
    flat = x.reshape(-1, 2 * hidden)
    locals_ = [torch.nn.functional.linear(flat, m.weight) for m in mods]
    monkeypatch.setattr(
        vd, "tensor_model_parallel_all_gather", lambda t, dim=0: torch.cat(locals_, 0)
    )
    for r, m in enumerate(mods):
        out = m(x)
        assert out.shape == expected.shape
        assert torch.equal(out, expected), r


def test_shard_draft_eh_proj_swaps_only_plain_linear():
    draft = _make_mtp_draft(num_layers=2)
    layers = list(draft.model.layers.values())
    layers[0].eh_proj = torch.nn.Linear(32, 16, bias=False)
    layers[1].eh_proj = torch.nn.Linear(32, 16, bias=True)  # has bias: kept
    w0 = layers[0].eh_proj.weight.detach().clone()
    assert utils.shard_draft_eh_proj(draft, tp_rank=2, tp_size=4) == 1
    assert isinstance(layers[0].eh_proj, utils.ColumnParallelEhProj)
    assert torch.equal(layers[0].eh_proj.weight, w0[8:12])
    assert type(layers[1].eh_proj) is torch.nn.Linear
    # tp 1, or a width tp does not divide: untouched
    d2 = _make_mtp_draft(num_layers=1)
    next(iter(d2.model.layers.values())).eh_proj = torch.nn.Linear(10, 6, bias=False)
    assert utils.shard_draft_eh_proj(d2, 0, 1) == 0
    assert utils.shard_draft_eh_proj(d2, 0, 4) == 0
