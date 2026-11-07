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
VocabParallelEmbedding / ParallelLMHead / LogitsProcessor (as the
TpuLogitsProcessor the TPU plugin builds it as)."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
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
        patch.object(dp_vocab_parallel, "all_to_all", lambda x, g: g.all_to_all(x)),
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
    # A CPU device: the CPU unit-test agents have no TPU device to resolve.
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
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

    # Through the layer's own forward: the TPU vocab patch dispatches it.
    out = _run_ranks(ranks, lambda r: layers[ranks.index(r)](ids[r.dp_rank]))

    for r, got in zip(ranks, out, strict=True):
        torch.testing.assert_close(got, table[ids[r.dp_rank]], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [3, 8])
@pytest.mark.parametrize("dp, tp", [(4, 2), (2, 1)])
def test_lm_head_logits_match_unsharded(dp, tp, rows):
    torch.manual_seed(1)
    table = torch.randn(VOCAB, HIDDEN)
    ranks = _layout(dp, tp)
    heads = _build(_lm_head_cls(), ranks, table)
    processor = dp_vocab_parallel.TpuLogitsProcessor(VOCAB)
    # Each engine its own rows (shared by its TP ranks), the same count on all.
    hidden = [torch.randn(rows, HIDDEN) for _ in range(dp)]

    out = _run_ranks(
        ranks, lambda r: processor(heads[ranks.index(r)], hidden[r.dp_rank])
    )

    for r, got in zip(ranks, out, strict=True):
        assert got.shape == (rows, VOCAB)
        torch.testing.assert_close(
            got, hidden[r.dp_rank] @ table.T, rtol=1e-5, atol=1e-5
        )


def test_lm_head_projects_each_engines_rows_once_per_tp_index():
    """An engine's TP ranks hold the same rows, so each rank projects the rows
    of the DP x 1 ranks sharing its TP index, not of all DP x TP ranks."""
    dp, tp, rows = 4, 2, 3
    ranks = _layout(dp, tp)
    heads = _build(_lm_head_cls(), ranks, torch.randn(VOCAB, HIDDEN))
    processor = dp_vocab_parallel.TpuLogitsProcessor(VOCAB)
    projected: list[int] = []
    apply_head = processor._apply_head

    def counting_apply_head(lm_head, hidden_states, embedding_bias):
        projected.append(hidden_states.shape[0])
        return apply_head(lm_head, hidden_states, embedding_bias)

    processor._apply_head = counting_apply_head
    hidden = [torch.randn(rows, HIDDEN) for _ in range(dp)]

    _run_ranks(ranks, lambda r: processor(heads[ranks.index(r)], hidden[r.dp_rank]))

    assert projected == [dp * rows] * (dp * tp)


def test_check_layers_sharded_over_dp():
    ranks = _layout(4, 2)
    tp_only = VocabParallelEmbedding(
        VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True
    )
    assert not dp_vocab_parallel.check_layers_sharded_over_dp(tp_only)
    (head,) = _build(_lm_head_cls(), ranks[:1], torch.randn(VOCAB, HIDDEN))
    model = torch.nn.ModuleList(
        [tp_only, head, dp_vocab_parallel.TpuLogitsProcessor(VOCAB)]
    )
    assert dp_vocab_parallel.check_layers_sharded_over_dp(model)


def test_quantized_lm_head_layout_passes_the_load_check():
    """The TPU FP8 method stores an lm_head weight as [hidden, vocab shard];
    the check keys on shared parameters, not on the weight's shape."""
    (head,) = _build(_lm_head_cls(), _layout(4, 2)[:1], torch.randn(VOCAB, HIDDEN))
    head.weight = torch.nn.Parameter(torch.zeros(HIDDEN, 256 // 8))
    model = torch.nn.ModuleList([head, dp_vocab_parallel.TpuLogitsProcessor(VOCAB)])
    assert dp_vocab_parallel.check_layers_sharded_over_dp(model)


@pytest.mark.parametrize("dp_side", ["lm_head", "embedding"])
def test_tie_across_shardings_fails_at_load(dp_side):
    """A tied lm_head and input embedding that took different shardings would
    project, or look up, with the other layer's rows."""
    ranks = _layout(4, 2)
    (dp_layer,) = _build(_lm_head_cls(), ranks[:1], torch.randn(VOCAB, HIDDEN))
    tp_layer = VocabParallelEmbedding(
        VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True
    )
    if dp_side == "lm_head":
        dp_layer.weight = tp_layer.weight
    else:
        tp_layer.weight = dp_layer.weight
    with pytest.raises(NotImplementedError, match="shares a weight"):
        dp_vocab_parallel.check_layers_sharded_over_dp(
            torch.nn.ModuleList([tp_layer, dp_layer])
        )


class _OwnLogitsProcessor(LogitsProcessor):
    """A model's LogitsProcessor subclass, which vLLM's out-of-tree
    replacement of LogitsProcessor does not reach."""


@pytest.mark.parametrize(
    "processors, match",
    [
        ([], "no LogitsProcessor"),
        ([LogitsProcessor], r"\(LogitsProcessor\)"),
        ([dp_vocab_parallel.TpuLogitsProcessor, _OwnLogitsProcessor], "_OwnLogits"),
    ],
)
def test_lm_head_over_dp_needs_the_tpu_logits_processor(processors, match):
    (head,) = _build(_lm_head_cls(), _layout(4, 2)[:1], torch.randn(VOCAB, HIDDEN))
    model = torch.nn.ModuleList([head, *(cls(VOCAB) for cls in processors)])
    with pytest.raises(NotImplementedError, match=match):
        dp_vocab_parallel.check_layers_sharded_over_dp(model)


@pytest.mark.parametrize(
    "processor, raises",
    [
        (LogitsProcessor, True),
        (dp_vocab_parallel.TpuLogitsProcessor, False),
        (None, False),
    ],
)
def test_embedding_over_dp_needs_the_tpu_logits_processor(processor, raises):
    """A model may hand its input embedding to its LogitsProcessor as the
    lm_head; a stock one would then gather the DP-split logits over TP only."""
    (embed,) = _build(
        VocabParallelEmbedding, _layout(4, 2)[:1], torch.randn(VOCAB, HIDDEN)
    )
    model = torch.nn.ModuleList([embed] + ([processor(VOCAB)] if processor else []))
    if raises:
        with pytest.raises(NotImplementedError, match="a layer is sharded"):
            dp_vocab_parallel.check_layers_sharded_over_dp(model)
    else:
        assert dp_vocab_parallel.check_layers_sharded_over_dp(model)


def test_register_logits_processor_only_with_the_flag(monkeypatch):
    """Models build LogitsProcessor; under the flag they get TpuLogitsProcessor
    through vLLM's out-of-tree PluggableLayer registry."""
    from vllm.model_executor import custom_op

    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    monkeypatch.setattr(
        dp_vocab_parallel.TpuLogitsProcessor,
        "name",
        dp_vocab_parallel.TpuLogitsProcessor.name,
    )
    monkeypatch.delenv("TPU_DP_VOCAB_PARALLEL", raising=False)
    dp_vocab_parallel.register_logits_processor()
    assert type(LogitsProcessor(VOCAB)) is LogitsProcessor

    monkeypatch.setenv("TPU_DP_VOCAB_PARALLEL", "1")
    dp_vocab_parallel.register_logits_processor()
    dp_vocab_parallel.register_logits_processor()  # once per process stage
    assert type(LogitsProcessor(VOCAB)) is dp_vocab_parallel.TpuLogitsProcessor


def test_tp_only_lm_head_keeps_the_stock_path():
    """An lm_head without the flag goes through vLLM's own _get_logits."""
    head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True)
    head.weight.data.copy_(
        torch.nn.functional.pad(torch.randn(VOCAB, HIDDEN), (0, 0, 0, 6))
    )
    hidden = torch.randn(3, HIDDEN)
    got = dp_vocab_parallel.TpuLogitsProcessor(VOCAB)(head, hidden)
    torch.testing.assert_close(got, hidden @ head.weight[:VOCAB].T)


def _config(
    dp=4, ep=True, pp=1, pcp=1, dcp=1, spec=None, lora=None, runner="generate", mm=False
):
    pc = SimpleNamespace(
        data_parallel_size=dp,
        enable_expert_parallel=ep,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
    )
    return SimpleNamespace(
        parallel_config=pc,
        speculative_config=spec,
        lora_config=lora,
        model_config=SimpleNamespace(
            runner_type=runner, get_vocab_size=lambda: VOCAB, is_multimodal_model=mm
        ),
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
        ({"runner": "pooling"}, "pooling models"),
        ({"mm": True}, "multimodal models"),
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
    assert dp_vocab_parallel.group_for_new_layer(VOCAB, None, 64) is None


def _group_for(monkeypatch, dp_size, tp_size, *vocab, ep_rank=None):
    """group_for_new_layer on an eligible config at DP rank 1, TP rank 0, with
    the EP rank dp-major unless given."""
    monkeypatch.setenv("TPU_DP_VOCAB_PARALLEL", "1")
    dp_rank, tp_rank = 1, 0
    if ep_rank is None:
        ep_rank = dp_rank * tp_size + tp_rank
    ep = SimpleNamespace(world_size=dp_size * tp_size, rank_in_group=ep_rank)
    dp = SimpleNamespace(world_size=dp_size, rank_in_group=dp_rank)
    tp = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank)
    with (
        patch("vllm.config.get_current_vllm_config", return_value=_config()),
        patch.object(dp_vocab_parallel, "_ineligible_reason", return_value=None),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        return ep, dp_vocab_parallel.group_for_new_layer(*vocab)


def test_group_layout_must_be_dp_major(monkeypatch):
    with pytest.raises(RuntimeError, match="dp_rank \\* tp_size"):
        _group_for(monkeypatch, 4, 2, VOCAB, None, 64, ep_rank=1)  # dp-major: 2


def test_vocab_that_does_not_split_over_dp_x_tp_stays_tp_sharded(monkeypatch):
    """250 rows pad to 256 (padding_size 64): 8 ranks split it, 6 do not, and
    vLLM would fail at init instead of falling back."""
    ep, group = _group_for(monkeypatch, 4, 2, VOCAB, None, 64)
    assert group is ep
    _, group = _group_for(monkeypatch, 3, 2, VOCAB, None, 64)
    assert group is None


def test_only_tables_over_the_text_vocab_are_resharded(monkeypatch):
    """Position, multimodal or per-layer tables may run outside the decoder's
    lockstep forward (here a 128-row multimodal table)."""
    _, group = _group_for(monkeypatch, 4, 2, 128, None, 64)
    assert group is None


def test_patched_init_marks_layers_built_over_the_dp_group(monkeypatch):
    """The TPU vocab patch hands the stock embedding and lm_head, built without
    a group of their own, the EP group. It leaves alone layers that bring a
    group, disable TP, skip the reduction, or are subclasses that may run their
    own forward."""

    class _OwnForward(VocabParallelEmbedding):
        def forward(self, input_):
            return input_

    monkeypatch.setenv("TPU_DP_VOCAB_PARALLEL", "1")
    vpe = "vllm.model_executor.layers.vocab_parallel_embedding"
    ranks = _layout(2, 1)
    with (
        patch.object(
            dp_vocab_parallel, "group_for_new_layer", return_value=ranks[0].ep
        ) as group_for,
        # Layers left on the TP group look it up; a 1-rank one here.
        patch(f"{vpe}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{vpe}.get_tensor_model_parallel_world_size", return_value=1),
    ):
        # params_dtype, org_num_embeddings and padding_size by position.
        flagged = VocabParallelEmbedding(VOCAB, HIDDEN, torch.float32, None, 32)
        head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.float32)
        untouched = [
            VocabParallelEmbedding(
                VOCAB, HIDDEN, params_dtype=torch.float32, parallel_group=ranks[0].tp
            ),
            VocabParallelEmbedding(
                VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True
            ),
            VocabParallelEmbedding(
                VOCAB, HIDDEN, params_dtype=torch.float32, reduce_results=False
            ),
            _OwnForward(VOCAB, HIDDEN, params_dtype=torch.float32),
        ]
    for layer in (flagged, head):
        assert getattr(layer, dp_vocab_parallel.DP_VOCAB_ATTR, False)
        assert layer.parallel_group is ranks[0].ep
    assert flagged.weight.shape[0] == 128
    # The vocab arguments reach the split check, positional ones included.
    assert group_for.call_args_list[0].args == (VOCAB, None, 32)
    assert group_for.call_args_list[1].args == (VOCAB, None, 64)
    assert group_for.call_count == 2
    for layer in untouched:
        assert not getattr(layer, dp_vocab_parallel.DP_VOCAB_ATTR, False)


def test_patched_init_skips_the_dp_check_with_the_flag_off(monkeypatch):
    """With the flag off, building a stock embedding takes vLLM's path as is."""
    monkeypatch.delenv("TPU_DP_VOCAB_PARALLEL", raising=False)
    vpe = "vllm.model_executor.layers.vocab_parallel_embedding"
    with (
        patch.object(dp_vocab_parallel, "group_for_new_layer") as group_for,
        patch(f"{vpe}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{vpe}.get_tensor_model_parallel_world_size", return_value=1),
    ):
        layer = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.float32)
    group_for.assert_not_called()
    assert not getattr(layer, dp_vocab_parallel.DP_VOCAB_ATTR, False)
