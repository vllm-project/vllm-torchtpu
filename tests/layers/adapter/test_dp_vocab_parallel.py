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

import os
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
    """Stock layers built through the TPU vocab patch, one per rank, each handed
    its rank's EP group."""
    layers = []
    for r in ranks:
        with (
            patch.dict(os.environ, {"TPU_DP_VOCAB_PARALLEL": "1"}),
            patch.object(dp_vocab_parallel, "group_for_new_layer", return_value=r.ep),
        ):
            layer = cls(VOCAB, HIDDEN, params_dtype=torch.float32)
        assert getattr(layer, dp_vocab_parallel.DP_VOCAB_ATTR, False)
        layer.weight_loader(layer.weight, table)
        layers.append(layer)
    return layers


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
    heads = _build(ParallelLMHead, ranks, table)
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
    heads = _build(ParallelLMHead, ranks, torch.randn(VOCAB, HIDDEN))
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [3, 8])
@pytest.mark.parametrize("dp, tp", [(4, 2), (2, 1), (1, 4), (8, 1)])
def test_lm_head_top_tokens_match_unsharded_argmax(dp, tp, rows, dtype):
    """TpuLogitsProcessor.get_top_tokens exchanges only [ep_size, rows, 2]
    pairs and reproduces torch.argmax over the full-vocab logits across
    dtype, scale, soft_cap, embedding_bias, and padded vocabulary tails."""
    torch.manual_seed(dp * 10 + tp + rows)
    table = torch.randn(VOCAB, HIDDEN, dtype=dtype)
    bias = torch.randn(VOCAB, dtype=dtype) * 0.1
    ranks = _layout(dp, tp)
    heads = []
    for r in ranks:
        with (
            patch.dict(os.environ, {"TPU_DP_VOCAB_PARALLEL": "1"}),
            patch.object(dp_vocab_parallel, "group_for_new_layer", return_value=r.ep),
        ):
            head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=dtype)
        head.weight_loader(head.weight, table)
        # Fill padded tail rows (indices >= VOCAB) on the last shard with large
        # values that would win argmax if _mask_outside_vocab were missing.
        pad = head.shard_indices.num_org_vocab_padding
        if pad > 0:
            head.weight.data[-pad:].fill_(1e3)
        heads.append(head)

    processor = dp_vocab_parallel.TpuLogitsProcessor(VOCAB, scale=0.75, soft_cap=5.0)
    hidden = [torch.randn(rows, HIDDEN, dtype=dtype) for _ in range(dp)]
    bias_shards = []
    for head in heads:
        start = head.shard_indices.org_vocab_start_index
        width = head.weight.shape[0]
        shard_b = torch.zeros(width, dtype=dtype)
        valid = max(0, min(width, VOCAB - start))
        if valid > 0:
            shard_b[:valid] = bias[start : start + valid]
        bias_shards.append(shard_b)

    sent_shapes: list[tuple[int, ...]] = []
    orig_rank0_a2a = ranks[0].ep.all_to_all

    def recording_all_to_all(x):
        sent_shapes.append(tuple(x.shape))
        return orig_rank0_a2a(x)

    ranks[0].ep.all_to_all = recording_all_to_all
    top = _run_ranks(
        ranks,
        lambda r: processor.get_top_tokens(
            heads[ranks.index(r)],
            hidden[r.dp_rank],
            bias_shards[ranks.index(r)],
        ),
    )

    assert sent_shapes == [(dp * tp, rows, 2)]
    full = _run_ranks(
        ranks,
        lambda r: processor(
            heads[ranks.index(r)],
            hidden[r.dp_rank],
            bias_shards[ranks.index(r)],
        ),
    )
    for got, logits in zip(top, full, strict=True):
        assert got.dtype == torch.int64 and got.shape == (rows,)
        assert torch.equal(got, logits.argmax(dim=-1))


def test_lm_head_top_tokens_ties_resolve_to_lowest_token_id():
    """Exact ties both within one EP shard and across distinct EP shards break
    to the lowest global token ID, matching torch.argmax."""
    dp, tp = 4, 2
    table = torch.zeros(VOCAB, HIDDEN)
    # Row 0 ties ids 35 (shard 1) and 195 (shard 6); row 1 ties ids 66 and 67
    # inside shard 2; row 2 ties all 250 tokens at 0.0.
    table[[35, 195], 0] = 2.0
    table[[66, 67], 1] = 2.0
    ranks = _layout(dp, tp)
    heads = _build(ParallelLMHead, ranks, table)
    processor = dp_vocab_parallel.TpuLogitsProcessor(VOCAB)
    h = torch.zeros(3, HIDDEN)
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    h[2, 2] = 1.0
    hidden = [h.clone() for _ in range(dp)]

    out = _run_ranks(
        ranks,
        lambda r: processor.get_top_tokens(heads[ranks.index(r)], hidden[r.dp_rank]),
    )
    for got in out:
        assert got.tolist() == [35, 66, 0]


def test_tp_only_lm_head_top_tokens_keeps_the_stock_path():
    head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True)
    table = torch.randn(VOCAB, HIDDEN)
    head.weight.data[:VOCAB].copy_(table)
    head.weight.data[VOCAB:].fill_(1e4)
    hidden = torch.randn(4, HIDDEN)
    got = dp_vocab_parallel.TpuLogitsProcessor(VOCAB).get_top_tokens(head, hidden)
    assert torch.equal(got, (hidden @ table.T).argmax(dim=-1))
    bad_proc = dp_vocab_parallel.TpuLogitsProcessor(VOCAB, scale=-1.0)
    (dp_head,) = _build(ParallelLMHead, _layout(4, 2)[:1], table)
    with pytest.raises(ValueError, match="non-positive"):
        bad_proc.get_top_tokens(dp_head, hidden)


class _OwnLogitsProcessor(LogitsProcessor):
    """A model's LogitsProcessor subclass, which vLLM's out-of-tree
    replacement of LogitsProcessor does not reach."""


def _over_dp(cls):
    (layer,) = _build(cls, _layout(4, 2)[:1], torch.randn(VOCAB, HIDDEN))
    return layer


def _over_tp():
    return VocabParallelEmbedding(
        VOCAB, HIDDEN, params_dtype=torch.float32, disable_tp=True
    )


def _fp8_layout(head):
    # The TPU FP8 method stores an lm_head weight as [hidden, vocab shard].
    head.weight = torch.nn.Parameter(torch.zeros(HIDDEN, 256 // 8))
    return head


def _tied(dp_side):
    dp_layer, tp_layer = _over_dp(ParallelLMHead), _over_tp()
    if dp_side == "lm_head":
        dp_layer.weight = tp_layer.weight
    else:
        tp_layer.weight = dp_layer.weight
    return [tp_layer, dp_layer]


_TPU = dp_vocab_parallel.TpuLogitsProcessor


@pytest.mark.parametrize(
    "build, expected",
    [
        pytest.param(lambda: [_over_tp()], False, id="nothing_over_dp"),
        pytest.param(
            lambda: [_over_dp(ParallelLMHead), _TPU(VOCAB)], True, id="lm_head_over_dp"
        ),
        pytest.param(
            lambda: [_fp8_layout(_over_dp(ParallelLMHead)), _TPU(VOCAB)],
            True,
            id="fp8_lm_head_layout",
        ),
        pytest.param(
            lambda: [_over_dp(VocabParallelEmbedding)],
            True,
            id="embedding_over_dp_no_processor",
        ),
        pytest.param(
            lambda: [_over_dp(VocabParallelEmbedding), _TPU(VOCAB)],
            True,
            id="embedding_over_dp",
        ),
        pytest.param(
            lambda: _tied("lm_head"), "shares a weight", id="tied_lm_head_over_dp"
        ),
        pytest.param(
            lambda: _tied("embedding"), "shares a weight", id="tied_embedding_over_dp"
        ),
        pytest.param(
            lambda: [_over_dp(ParallelLMHead)],
            "no LogitsProcessor",
            id="lm_head_over_dp_no_processor",
        ),
        pytest.param(
            lambda: [_over_dp(ParallelLMHead), LogitsProcessor(VOCAB)],
            r"\(LogitsProcessor\)",
            id="stock_processor",
        ),
        pytest.param(
            lambda: [_over_dp(ParallelLMHead), _TPU(VOCAB), _OwnLogitsProcessor(VOCAB)],
            "_OwnLogits",
            id="processor_subclass",
        ),
        pytest.param(
            lambda: [_over_dp(VocabParallelEmbedding), LogitsProcessor(VOCAB)],
            "a layer is sharded",
            id="embedding_over_dp_stock_processor",
        ),
    ],
)
def test_check_layers_sharded_over_dp(build, expected):
    """Whether the model has a layer over DP, or why it cannot run one: tied
    layers that took different shardings would use the other layer's rows, and
    a stock LogitsProcessor or a subclass would gather DP-split logits over TP
    only (a model may pass its embedding to it as the lm_head). The check keys
    on shared parameters, not weight shapes, so a transposed FP8 lm_head
    passes."""
    model = torch.nn.ModuleList(build())
    if isinstance(expected, str):
        with pytest.raises(NotImplementedError, match=expected):
            dp_vocab_parallel.check_layers_sharded_over_dp(model)
    else:
        assert dp_vocab_parallel.check_layers_sharded_over_dp(model) is expected


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


def _group_for(
    monkeypatch, config=None, vocab=(VOCAB, None, 64), dp_size=4, ep_rank=None, env=()
):
    """group_for_new_layer for a table of `vocab` under `config` (default: an
    eligible DP x TP 2 layout), at DP rank 1, TP rank 0, with the flag and `env`
    set, and the EP rank dp-major unless given."""
    tp_size, dp_rank, tp_rank = 2, 1, 0
    for name in ("TPU_GREEDY_TOP_TOKENS", "TORCH_TPU_DP_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TPU_DP_VOCAB_PARALLEL", "1")
    for name, value in env:
        monkeypatch.setenv(name, value)
    if ep_rank is None:
        ep_rank = dp_rank * tp_size + tp_rank
    ep = SimpleNamespace(world_size=dp_size * tp_size, rank_in_group=ep_rank)
    dp = SimpleNamespace(world_size=dp_size, rank_in_group=dp_rank)
    tp = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank)
    with (
        patch(
            "vllm.config.get_current_vllm_config",
            return_value=config or _config(dp=dp_size),
        ),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp),
    ):
        return ep, dp_vocab_parallel.group_for_new_layer(*vocab)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({}, "ep", id="eligible"),
        pytest.param({"env": [("TPU_DP_VOCAB_PARALLEL", "0")]}, None, id="flag_off"),
        pytest.param({"config": _config(dp=1)}, None, id="no_dp"),
        pytest.param({"config": _config(ep=False)}, None, id="no_ep"),
        pytest.param({"config": _config(pp=2)}, None, id="pipeline_parallelism"),
        pytest.param({"config": _config(pcp=2)}, None, id="prefill_context_parallel"),
        pytest.param({"config": _config(dcp=2)}, None, id="decode_context_parallel"),
        pytest.param({"config": _config(spec=object())}, None, id="spec_decoding"),
        pytest.param({"config": _config(lora=object())}, None, id="lora"),
        pytest.param({"config": _config(runner="pooling")}, None, id="pooling_model"),
        pytest.param({"config": _config(mm=True)}, None, id="multimodal_model"),
        pytest.param(
            {"env": [("TPU_GREEDY_TOP_TOKENS", "1")]}, "ep", id="greedy_top_tokens"
        ),
        # 250 rows pad to 256 (padding_size 64): 8 ranks split it, 6 do not.
        pytest.param({"dp_size": 3}, None, id="vocab_does_not_split_over_dp_x_tp"),
        # Position, per-layer or multimodal tables may run outside the forward.
        pytest.param({"vocab": (128, None, 64)}, None, id="not_the_text_vocab"),
        pytest.param({"ep_rank": 1}, RuntimeError, id="ep_rank_not_dp_major"),
    ],
)
def test_group_for_new_layer(monkeypatch, kwargs, expected):
    """The EP group only for a table over the text vocabulary on an eligible DP
    + EP layout whose padded vocab splits over DP x TP; else None (vLLM's TP
    sharding), or an error for an EP group laid out other than ep_rank =
    dp_rank * tp_size + tp_rank."""
    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="dp_rank \\* tp_size"):
            _group_for(monkeypatch, **kwargs)
        return
    ep, group = _group_for(monkeypatch, **kwargs)
    assert group is (ep if expected == "ep" else None)


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
