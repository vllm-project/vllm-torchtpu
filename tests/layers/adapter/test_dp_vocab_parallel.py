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
"""TPU_DP_VOCAB_PARALLEL: the embedding and lm_head sharded over every DP x TP
rank match the unsharded embedding and projection.

Every rank of a DP x TP layout runs in its own thread on CPU, with fake groups
whose collectives exchange real tensors, through vLLM's own
VocabParallelEmbedding / ParallelLMHead / LogitsProcessor."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding
from vllm_torchtpu.layers.adapter import dp_vocab_parallel

VOCAB = 250  # pads to 256 = 8 shards of 32 at DP 4 x TP 2
HIDDEN = 16


class _Exchange:
    """One rendezvous per collective call among a group's members."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.slots: dict[tuple[int, int], torch.Tensor] = {}

    def exchange(self, call: int, rank: int, tensor: torch.Tensor):
        self.slots[(call, rank)] = tensor
        self.barrier.wait()
        parts = [self.slots[(call, r)] for r in range(self.world)]
        self.barrier.wait()
        return parts


class _FakeGroup:
    """The GroupCoordinator surface the module uses, for one rank."""

    def __init__(self, exchange: _Exchange, rank_in_group: int):
        self.exchange = exchange
        self.world_size = exchange.world
        self.rank_in_group = rank_in_group
        self.device_group = self
        self.calls = 0

    def _parts(self, tensor: torch.Tensor):
        self.calls += 1
        return self.exchange.exchange(self.calls, self.rank_in_group, tensor)

    def all_gather(self, tensor, dim=0):
        return torch.cat(self._parts(tensor), dim=dim)

    def reduce_scatter(self, tensor, dim=0):
        total = torch.stack(self._parts(tensor)).sum(0)
        return total.chunk(self.world_size, dim=dim)[self.rank_in_group]

    def all_reduce(self, tensor):
        return torch.stack(self._parts(tensor)).sum(0)

    def all_to_all(self, tensor):
        parts = self._parts(tensor)
        return torch.stack([p[self.rank_in_group] for p in parts])


def _layout(dp: int, tp: int):
    """Per global rank (dp_rank * tp + tp_rank): its EP, DP and TP groups."""
    ep_x = _Exchange(dp * tp)
    dp_x = [_Exchange(dp) for _ in range(tp)]
    tp_x = [_Exchange(tp) for _ in range(dp)]
    return [
        SimpleNamespace(
            ep=_FakeGroup(ep_x, d * tp + t),
            dp=_FakeGroup(dp_x[t], d),
            tp=_FakeGroup(tp_x[d], t),
            dp_rank=d,
        )
        for d in range(dp)
        for t in range(tp)
    ]


def _run_ranks(ranks, fn):
    """fn(rank) on every rank at once; the results in rank order."""
    local = threading.local()
    results: list = [None] * len(ranks)
    errors: list = []

    def run(i):
        local.rank = ranks[i]
        try:
            results[i] = fn(ranks[i])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            for r in ranks:  # release the peers
                for g in (r.ep, r.dp, r.tp):
                    g.exchange.barrier.abort()

    with (
        patch.object(
            dp_vocab_parallel, "_groups", lambda: (local.rank.dp, local.rank.tp)
        ),
        patch.object(dp_vocab_parallel, "_all_to_all", lambda x, g: g.all_to_all(x)),
    ):
        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(ranks))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return results


@pytest.fixture(autouse=True)
def _patched():
    _patch_vllm_vocab_parallel_embedding()
    dp_vocab_parallel.install_logits_hook()
    with set_current_vllm_config(VllmConfig()):
        yield


def _build(cls, ranks, table):
    layers = []
    for r in ranks:
        layer = cls(VOCAB, HIDDEN, params_dtype=torch.float32, parallel_group=r.ep)
        layer.weight_loader(layer.weight, table)
        setattr(layer, dp_vocab_parallel.DP_VOCAB_ATTR, True)
        layers.append(layer)
    return layers


def _lm_head_cls():
    # ParallelLMHead does not forward parallel_group; the TPU patch injects it
    # through VocabParallelEmbedding.__init__, which this subclass mimics.
    class _Head(ParallelLMHead):
        def __init__(self, num_embeddings, embedding_dim, params_dtype, parallel_group):
            VocabParallelEmbedding.__init__(
                self,
                num_embeddings,
                embedding_dim,
                params_dtype,
                parallel_group=parallel_group,
            )
            self.register_parameter("bias", None)

    return _Head


@pytest.mark.parametrize("dp, tp", [(4, 2), (2, 1), (1, 4)])
def test_embedding_matches_unsharded(dp, tp):
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN)
    ranks = _layout(dp, tp)
    layers = _build(VocabParallelEmbedding, ranks, table)
    assert {layer.weight.shape[0] for layer in layers} == {256 // (dp * tp)}
    # Each engine its own ids (shared by its TP ranks), the same count on all.
    ids = [torch.randint(0, VOCAB, (7,)) for _ in range(dp)]

    out = _run_ranks(
        ranks,
        lambda r: dp_vocab_parallel.embedding_forward(
            layers[ranks.index(r)], ids[r.dp_rank]
        ),
    )

    for r, got in zip(ranks, out, strict=True):
        torch.testing.assert_close(got, table[ids[r.dp_rank]], rtol=0, atol=0)


@pytest.mark.parametrize("dp, tp", [(4, 2), (2, 1)])
def test_lm_head_logits_match_unsharded(dp, tp):
    torch.manual_seed(1)
    table = torch.randn(VOCAB, HIDDEN)
    ranks = _layout(dp, tp)
    heads = _build(_lm_head_cls(), ranks, table)
    processor = LogitsProcessor(VOCAB)
    rows = 8
    dp_vocab_parallel.set_logits_rows(torch.nn.ModuleList(heads), rows)
    # Each engine its own rows (shared by its TP ranks), the fixed count on all.
    hidden = [torch.randn(rows, HIDDEN) for _ in range(dp)]

    out = _run_ranks(
        ranks, lambda r: processor(heads[ranks.index(r)], hidden[r.dp_rank])
    )

    for r, got in zip(ranks, out, strict=True):
        assert got.shape == (rows, VOCAB)
        torch.testing.assert_close(
            got, hidden[r.dp_rank] @ table.T, rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("num_rows", [3, 9])
def test_lm_head_rejects_any_other_row_count(num_rows):
    """A rank-dependent row count would give the ranks different programs
    around the collectives; the runner pads before the call."""
    table = torch.randn(VOCAB, HIDDEN)
    ranks = _layout(2, 1)
    heads = _build(_lm_head_cls(), ranks, table)
    dp_vocab_parallel.set_logits_rows(torch.nn.ModuleList(heads), 8)
    with pytest.raises(ValueError, match="must pass exactly 8"):
        LogitsProcessor(VOCAB)(heads[0], torch.randn(num_rows, HIDDEN))


def test_lm_head_without_rows_fails_loudly():
    ranks = _layout(2, 1)
    heads = _build(_lm_head_cls(), ranks, torch.randn(VOCAB, HIDDEN))
    with pytest.raises(RuntimeError, match="row count was never set"):
        LogitsProcessor(VOCAB)(heads[0], torch.randn(2, HIDDEN))


def test_set_logits_rows_reports_whether_any_layer_is_sharded_over_dp():
    plain = torch.nn.Linear(2, 2)
    assert not dp_vocab_parallel.set_logits_rows(plain, 8)
    flagged = torch.nn.Linear(2, 2)
    setattr(flagged, dp_vocab_parallel.DP_VOCAB_ATTR, True)
    model = torch.nn.ModuleList([plain, flagged])
    assert dp_vocab_parallel.set_logits_rows(model, 8)
    assert getattr(flagged, dp_vocab_parallel.ROWS_ATTR) == 8


def test_tp_only_lm_head_keeps_the_stock_path():
    """An lm_head without the flag goes through vLLM's own _get_logits."""
    head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True)
    head.weight.data.copy_(
        torch.nn.functional.pad(torch.randn(VOCAB, HIDDEN), (0, 0, 0, 6))
    )
    hidden = torch.randn(3, HIDDEN)
    got = LogitsProcessor(VOCAB)(head, hidden)
    torch.testing.assert_close(got, hidden @ head.weight[:VOCAB].T)


def _config(dp=4, ep=True, pp=1, pcp=1, dcp=1, spec=None, lora=None):
    pc = SimpleNamespace(
        data_parallel_size=dp,
        enable_expert_parallel=ep,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )
    return SimpleNamespace(
        parallel_config=pc, speculative_config=spec, lora_config=lora
    )


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"dp": 1}, "data parallelism with expert parallelism"),
        ({"ep": False}, "data parallelism with expert parallelism"),
        ({"pp": 2}, "pipeline"),
        ({"pcp": 2}, "context parallelism"),
        ({"spec": object()}, "speculative decoding"),
        ({"lora": object()}, "LoRA"),
    ],
)
def test_ineligible_layouts(kwargs, reason):
    got = dp_vocab_parallel._ineligible_reason(_config(**kwargs))
    assert got is not None and reason in got


def test_eligible_layout(monkeypatch):
    monkeypatch.delenv("TPU_GREEDY_TOP_TOKENS", raising=False)
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    assert dp_vocab_parallel._ineligible_reason(_config()) is None
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")
    assert "TPU_GREEDY_TOP_TOKENS" in dp_vocab_parallel._ineligible_reason(_config())


def test_flag_off_builds_no_group(monkeypatch):
    monkeypatch.delenv("TPU_DP_VOCAB_PARALLEL", raising=False)
    assert dp_vocab_parallel.group_for_new_layer() is None


def test_group_layout_must_be_dp_major(monkeypatch):
    monkeypatch.setenv("TPU_DP_VOCAB_PARALLEL", "1")
    ep = SimpleNamespace(world_size=8, rank_in_group=1)
    dp = SimpleNamespace(world_size=4, rank_in_group=1)  # dp-major would be 2
    tp = SimpleNamespace(world_size=2, rank_in_group=0)
    with (
        patch("vllm.config.get_current_vllm_config", return_value=_config()),
        patch.object(dp_vocab_parallel, "_ineligible_reason", return_value=None),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp),
        pytest.raises(RuntimeError, match="dp_rank \\* tp_size"),
    ):
        dp_vocab_parallel.group_for_new_layer()


def test_patched_init_marks_layers_built_over_the_dp_group(monkeypatch):
    """The TPU vocab patch hands layers built without a group of their own the
    EP group, and leaves layers that bring one (or disable TP) alone."""
    ranks = _layout(2, 1)
    with patch.object(
        dp_vocab_parallel, "group_for_new_layer", return_value=ranks[0].ep
    ):
        flagged = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.float32)
        untouched = VocabParallelEmbedding(
            VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True
        )
    assert getattr(flagged, dp_vocab_parallel.DP_VOCAB_ATTR, False)
    assert flagged.parallel_group is ranks[0].ep
    assert flagged.weight.shape[0] == 128
    assert not getattr(untouched, dp_vocab_parallel.DP_VOCAB_ATTR, False)
