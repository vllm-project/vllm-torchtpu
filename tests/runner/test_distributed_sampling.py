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
"""Tests for vocabulary-parallel distributed non-greedy sampling."""

import contextlib
import logging
import math
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm import logger as vllm_logger
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner import distributed_sampling, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

Mode = distributed_sampling.DistributedSamplingMode
_MIN_SHARD_ROWS = 128


class _HeadShard(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        start: int,
        tp_size: int,
        tp_rank: int,
        num_pad: int,
        *,
        num_added: int = 0,
        transposed: bool = False,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        shard_vocab = weight.shape[0]
        self.num_embeddings_per_partition = shard_vocab
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.weight = weight.T.contiguous() if transposed else weight
        self.bias = bias
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            num_org_vocab_padding=num_pad,
            num_added_elements=num_added,
        )
        # Mirrors the buffers registered by the TPU VocabParallelEmbedding patch.
        self.register_buffer(
            "org_vocab_start_index", torch.tensor(start), persistent=False
        )
        self.register_buffer(
            "num_org_vocab_padding", torch.tensor(num_pad), persistent=False
        )

        def _apply(
            layer: _HeadShard, x: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            proj = x @ layer.weight if transposed else x @ layer.weight.T
            return proj + bias if bias is not None else proj

        self.quant_method = SimpleNamespace(apply=_apply)


class _Owner(torch.nn.Module):
    def __init__(
        self,
        head: torch.nn.Module,
        vocab_size: int,
        *,
        scale: float = 1.0,
        soft_cap: float | None = None,
        logits_as_input: bool = False,
        processor_cls: type[LogitsProcessor] = LogitsProcessor,
    ) -> None:
        super().__init__()
        self.lm_head = head
        with patch.object(
            lp_module,
            "get_current_vllm_config",
            return_value=SimpleNamespace(model_config=None),
        ):
            self.logits_processor = processor_cls(
                vocab_size,
                scale=scale,
                logits_as_input=logits_as_input,
                soft_cap=soft_cap,
            )


def _shards(
    weight: torch.Tensor,
    tp_size: int,
    *,
    min_shard_rows: int | None = None,
    transposed: bool = False,
    bias: torch.Tensor | None = None,
) -> list[_HeadShard]:
    vocab, hidden = weight.shape
    floor_rows = _MIN_SHARD_ROWS if min_shard_rows is None else min_shard_rows
    per_rank = max(math.ceil(vocab / tp_size), floor_rows)
    pad_rows = torch.full((per_rank * tp_size - vocab, hidden), 1e4, dtype=weight.dtype)
    padded = torch.cat([weight, pad_rows])
    padded_bias = None
    if bias is not None:
        padded_bias = torch.cat(
            [bias, torch.zeros(per_rank * tp_size - vocab, dtype=bias.dtype)]
        )
    shards = []
    for rank in range(tp_size):
        start = rank * per_rank
        num_pad = max(0, start + per_rank - vocab) if start < vocab else per_rank
        shard_bias = (
            None if padded_bias is None else padded_bias[start : start + per_rank]
        )
        shards.append(
            _HeadShard(
                padded[start : start + per_rank],
                start,
                tp_size,
                rank,
                num_pad,
                transposed=transposed,
                bias=shard_bias,
            )
        )
    return shards


def _run_across_ranks(fn_per_rank, tp_size: int, monkeypatch):
    """Executes ``fn_per_rank(rank)`` in lockstep across ``tp_size`` TP ranks."""
    tls = threading.local()
    barrier = threading.Barrier(tp_size)
    slots: list[torch.Tensor | None] = [None] * tp_size
    outputs: list[torch.Tensor | None] = [None] * tp_size
    errors: list[BaseException] = []

    def lockstep_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        slots[tls.rank] = tensor
        barrier.wait()
        gathered = torch.cat([t for t in slots if t is not None], dim=dim)
        barrier.wait()
        return gathered

    monkeypatch.setattr(
        distributed_sampling, "tensor_model_parallel_all_gather", lockstep_gather
    )

    def worker(rank: int) -> None:
        tls.rank = rank
        try:
            outputs[rank] = fn_per_rank(rank)
        except BaseException as exc:
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return [out for out in outputs if out is not None]


def _make_batch(*params: SamplingParams, max_num_logprobs=None, vocab_size=100000):
    requests = {
        f"r{i}": SimpleNamespace(sampling_params=p) for i, p in enumerate(params)
    }
    top_k_vals = [p.top_k if p.top_k > 0 else vocab_size for p in params]
    input_batch = SimpleNamespace(
        num_reqs=len(params),
        all_greedy=all(p.temperature == 0 for p in params),
        max_num_logprobs=max_num_logprobs,
        req_ids=[*requests, None],
        req_id_to_index={r: i for i, r in enumerate(requests)},
        temperature_cpu_tensor=torch.tensor(
            [p.temperature for p in params], dtype=torch.float32
        ),
        top_k_cpu_tensor=torch.tensor(top_k_vals, dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor([p.top_p for p in params], dtype=torch.float32),
        no_top_k=all(p.top_k <= 0 for p in params),
        no_top_p=all(p.top_p >= 1.0 for p in params),
    )
    return input_batch, requests


def test_softcap_precedes_scale_matching_upstream_logits_processor(monkeypatch):
    """_apply_softcap_and_scale matches real upstream LogitsProcessor.forward."""
    gen = torch.Generator().manual_seed(11)
    weight = torch.randn(16, 4, generator=gen, dtype=torch.float32)
    hidden = torch.randn(3, 4, generator=gen, dtype=torch.float32)
    shard = _HeadShard(weight, 0, tp_size=1, tp_rank=0, num_pad=0)
    owner = _Owner(shard, 16, scale=0.5, soft_cap=10.0)
    monkeypatch.setattr(
        lp_module, "tensor_model_parallel_all_gather", lambda t, *a, **k: t
    )
    upstream_out = owner.logits_processor(owner.lm_head, hidden)
    got = distributed_sampling._apply_softcap_and_scale(
        hidden @ weight.T, owner.logits_processor
    )
    assert upstream_out is not None
    torch.testing.assert_close(got, upstream_out, rtol=1e-6, atol=1e-6)


def test_topk_topp_greedy_row_caps_before_argmax_on_bf16_ties(monkeypatch):
    """Greedy rows in TOPK_TOPP tie-break on softcapped bf16 logits, not raw."""
    vocab, hidden, tp_size = 256, 4, 2
    weight = torch.zeros((vocab, hidden), dtype=torch.bfloat16)
    # In bf16 with soft_cap=30.0, 60.0 and 60.5 both round to 29.0:
    # full-logits argmax picks the lower ID (3), whereas raw argmax would pick
    # 9 intra-shard or 135 cross-shard.
    weight[3, 0] = 60.0
    weight[9, 0] = 60.5
    weight[135, 0] = 60.5
    shards = _shards(weight, tp_size, min_shard_rows=128)
    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, soft_cap=30.0), dp_lockstep=False
        )
        for r in range(tp_size)
    ]

    hidden = torch.zeros((2, hidden), dtype=torch.bfloat16)
    hidden[:, 0] = 1.0
    full_capped = torch.tanh((hidden @ weight.T) / 30.0) * 30.0
    assert full_capped[0, 3] == full_capped[0, 9] == full_capped[0, 135]
    ref_greedy = torch.argmax(full_capped[:1], dim=-1, keepdim=True)
    assert int(ref_greedy.item()) == 3

    temps = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    top_k = torch.tensor([[0], [16]], dtype=torch.int32)
    top_p = torch.ones((2, 1), dtype=torch.float32)
    u = torch.full(
        (2, distributed_sampling.rng_width(Mode.TOPK_TOPP, tp_size, 128)),
        0.5,
        dtype=torch.float32,
    )
    outs = _run_across_ranks(
        lambda r: distributed_sampling.sample_topk_topp(
            owners[r], hidden, temps, top_k, top_p, u
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(outs[0][:1], ref_greedy)


def test_softcap_preserves_masked_vocab_padding(monkeypatch):
    """soft_cap must never turn -inf vocab-padding slots into finite -soft_cap."""
    gen = torch.Generator().manual_seed(99)
    # At vocab=128, tp_size=2, min_shard_rows=128: rank 0 owns [0, 128) and
    # rank 1 owns [128, 256) which is 100% vocab padding. Give valid tokens
    # strongly negative raw logits (-400 -> softcapped -30) so any padded slot
    # turned into -30 would be sampled ~50% of the time at high temperature.
    vocab, hidden, tp_size = 128, 8, 2
    weight = torch.full((vocab, hidden), -50.0, dtype=torch.float32)
    shards = _shards(weight, tp_size, min_shard_rows=128)
    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, soft_cap=30.0), dp_lockstep=False
        )
        for r in range(tp_size)
    ]

    hidden_states = torch.ones((64, hidden), dtype=torch.float32)
    temps = torch.full((64, 1), 50.0, dtype=torch.float32)
    top_k = torch.full((64, 1), 64, dtype=torch.int32)
    top_p = torch.ones((64, 1), dtype=torch.float32)

    u_tktp = torch.rand(
        (64, distributed_sampling.rng_width(Mode.TOPK_TOPP, tp_size, 128)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_tktp = _run_across_ranks(
        lambda r: distributed_sampling.sample_topk_topp(
            owners[r], hidden_states, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(outs_tktp[0], outs_tktp[1])
    assert int(outs_tktp[0].max().item()) < vocab

    u_temp = torch.rand(
        (64, distributed_sampling.rng_width(Mode.PURE_TEMP, tp_size, 128)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_temp = _run_across_ranks(
        lambda r: distributed_sampling.sample_pure_temp(
            owners[r], hidden_states, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(outs_temp[0], outs_temp[1])
    assert int(outs_temp[0].max().item()) < vocab


def test_quantized_transposed_head_weight_and_bias(monkeypatch):
    """Transposed [hidden, shard_vocab] quantized weights and head bias work."""
    gen = torch.Generator().manual_seed(21)
    vocab, hidden, tp_size = 300, 16, 2
    weight = torch.randn(vocab, hidden, generator=gen, dtype=torch.float32)
    bias = torch.randn(vocab, generator=gen, dtype=torch.float32)

    ref_shards = _shards(weight, tp_size, transposed=False, bias=bias)
    quant_shards = _shards(weight, tp_size, transposed=True, bias=bias)
    assert quant_shards[0].weight.shape == (hidden, 150)

    ref_owners = [
        distributed_sampling.resolve_head_owner(_Owner(ref_shards[r], vocab), False)
        for r in range(tp_size)
    ]
    quant_owners = [
        distributed_sampling.resolve_head_owner(_Owner(quant_shards[r], vocab), False)
        for r in range(tp_size)
    ]
    assert distributed_sampling.head_tp_and_shard_vocab(quant_owners[0]) == (2, 150)

    h = torch.randn(8, hidden, generator=gen, dtype=torch.float32)
    temps = torch.tensor([[0.0], [1.0], [0.8], [1.2]] * 2, dtype=torch.float32)
    top_k = torch.full((8, 1), 32, dtype=torch.int32)
    top_p = torch.full((8, 1), 0.9, dtype=torch.float32)

    u_tktp = torch.rand(
        (8, distributed_sampling.rng_width(Mode.TOPK_TOPP, tp_size, 150)),
        generator=gen,
        dtype=torch.float32,
    )
    ref_tktp = _run_across_ranks(
        lambda r: distributed_sampling.sample_topk_topp(
            ref_owners[r], h, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    quant_tktp = _run_across_ranks(
        lambda r: distributed_sampling.sample_topk_topp(
            quant_owners[r], h, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(ref_tktp[0], quant_tktp[0])

    u_temp = torch.rand(
        (8, distributed_sampling.rng_width(Mode.PURE_TEMP, tp_size, 150)),
        generator=gen,
        dtype=torch.float32,
    )
    ref_temp = _run_across_ranks(
        lambda r: distributed_sampling.sample_pure_temp(
            ref_owners[r], h, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    quant_temp = _run_across_ranks(
        lambda r: distributed_sampling.sample_pure_temp(
            quant_owners[r], h, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(ref_temp[0], quant_temp[0])


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("soft_cap", [None, 30.0])
def test_distributed_sampling_modes_lockstep_padding_and_ties(
    monkeypatch, tp_size, soft_cap
):
    gen = torch.Generator().manual_seed(42 + tp_size)
    # Use a vocab with non-multiple shard padding and strided multi-block extraction.
    vocab, hidden = 2053 if tp_size <= 2 else 9205, 16
    weight = (torch.randn(vocab, hidden, generator=gen) * 0.5).to(torch.bfloat16)
    # Row 0 is greedy with an exact tie across shards: lowest token ID 5 wins.
    weight[:, 0] = 0.0
    weight[5, 0] = 4.0
    weight[vocab - 7, 0] = 4.0
    shards = _shards(weight, tp_size, min_shard_rows=1792 if tp_size <= 2 else 1280)

    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, scale=0.8, soft_cap=soft_cap),
            dp_lockstep=False,
        )
        for r in range(tp_size)
    ]

    hidden_states = torch.randn(4, hidden, generator=gen).to(torch.bfloat16)
    hidden_states[0] = 0.0
    hidden_states[0, 0] = 1.0

    temps = torch.tensor([[0.0], [1.0], [0.7], [1.2]], dtype=torch.float32)
    top_k = torch.tensor([[0], [64], [16], [1]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [0.95], [0.85], [1.0]], dtype=torch.float32)
    _, shard_vocab = distributed_sampling.head_tp_and_shard_vocab(owners[0])

    u_tktp = torch.rand(
        (4, distributed_sampling.rng_width(Mode.TOPK_TOPP, tp_size, shard_vocab)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_tktp = _run_across_ranks(
        lambda r: distributed_sampling.sample_topk_topp(
            owners[r], hidden_states, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    expected_greedy = torch.tensor(5, dtype=torch.int64)
    for out in outs_tktp:
        assert torch.equal(out, outs_tktp[0])
        assert torch.equal(out[0, 0], expected_greedy)
        assert bool((out < vocab).all())

    u_temp = torch.rand(
        (4, distributed_sampling.rng_width(Mode.PURE_TEMP, tp_size, shard_vocab)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_temp = _run_across_ranks(
        lambda r: distributed_sampling.sample_pure_temp(
            owners[r], hidden_states, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    for out in outs_temp:
        assert torch.equal(out, outs_temp[0])
        assert torch.equal(out[0, 0], expected_greedy)
        assert bool((out < vocab).all())


@pytest.mark.parametrize(
    "tp_size, mode, top_k_val, top_p_val, max_kl, max_tvd",
    [
        (2, Mode.TOPK_TOPP, 32, 0.90, 0.005, 0.020),
        (4, Mode.TOPK_TOPP, 100, 0.90, 0.010, 0.030),
        (2, Mode.PURE_TEMP, -1, 1.0, 0.060, 0.075),
    ],
)
def test_statistical_kl_and_support_parity(
    monkeypatch, tp_size, mode, top_k_val, top_p_val, max_kl, max_tvd
):
    gen = torch.Generator().manual_seed(123 + tp_size)
    # At TP=4 use 8192 vocab (2048 per shard > 4 * 256 = 1024) so multi-block
    # 4-round extraction runs rather than the small-shard early return.
    vocab = 8192 if tp_size == 4 else 4096
    hidden = 16
    weight = torch.randn(vocab, hidden, generator=gen, dtype=torch.float32) * 1.4
    h = torch.randn(1, hidden, generator=gen, dtype=torch.float32)
    raw_logits = (30.0 * torch.tanh((h @ weight.T) / 30.0)) * 0.85
    raw_probs = raw_logits.softmax(dim=-1).squeeze(0)
    if top_k_val > 0 or top_p_val < 1.0:
        ref_masked = apply_top_k_top_p(
            raw_logits.clone(),
            torch.tensor([[max(top_k_val, 0)]], dtype=torch.int32),
            torch.tensor([[top_p_val]], dtype=torch.float32),
        )
        target_probs = ref_masked.softmax(dim=-1).squeeze(0)
    else:
        target_probs = raw_probs

    shards = _shards(weight, tp_size)
    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, scale=0.85, soft_cap=30.0),
            dp_lockstep=False,
        )
        for r in range(tp_size)
    ]
    _, shard_vocab = distributed_sampling.head_tp_and_shard_vocab(owners[0])
    n_draws = 16384
    batch = 256
    h_rep = h.expand(batch, hidden).contiguous()
    t_rep = torch.ones((batch, 1), dtype=torch.float32)
    k_rep = torch.full((batch, 1), max(top_k_val, 0), dtype=torch.int32)
    p_rep = torch.full((batch, 1), top_p_val, dtype=torch.float32)
    counts = torch.zeros(vocab, dtype=torch.float32)

    u_width = distributed_sampling.rng_width(mode, tp_size, shard_vocab)
    for _ in range(n_draws // batch):
        u = torch.rand((batch, u_width), generator=gen, dtype=torch.float32)
        if mode is Mode.TOPK_TOPP:
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling.sample_topk_topp(
                    owners[r], h_rep, t_rep, k_rep, p_rep, u_step
                ),
                tp_size,
                monkeypatch,
            )
        else:
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling.sample_pure_temp(
                    owners[r], h_rep, t_rep, u_step
                ),
                tp_size,
                monkeypatch,
            )
        for r in range(1, tp_size):
            assert torch.equal(outs[0], outs[r])
        counts.scatter_add_(0, outs[0].view(-1), torch.ones(batch, dtype=torch.float32))

    emp_probs = counts / n_draws
    support = target_probs > 0
    min_kept_prob = float(raw_probs[support].min().item())
    relaxed_support = support | (raw_probs >= min_kept_prob * (1.0 - 1e-5))
    assert counts[~relaxed_support].sum().tolist() == 0.0
    nz = (emp_probs > 0) & relaxed_support
    p_ref_nz = (
        target_probs[nz]
        if bool((target_probs[nz] > 0).all())
        else raw_probs[nz] / raw_probs[relaxed_support].sum()
    )
    kl = float((emp_probs[nz] * torch.log(emp_probs[nz] / p_ref_nz)).sum().tolist())
    tvd = float((0.5 * (emp_probs - target_probs).abs().sum()).tolist())
    max_err = float((emp_probs - target_probs).abs().max().tolist())
    assert kl < max_kl
    assert tvd < max_tvd
    assert max_err < 0.010


def test_resolve_head_owner_admission_checks():
    shard = _shards(torch.zeros(16, 2), 2)[0]

    # Accepts TpuLogitsProcessor when DP_VOCAB_ATTR is not set.
    tpu_lp_owner = _Owner(
        _shards(torch.zeros(16, 2), 2)[0],
        16,
        processor_cls=dp_vocab_parallel.TpuLogitsProcessor,
    )
    assert (
        distributed_sampling.resolve_head_owner(tpu_lp_owner, dp_lockstep=False)
        is tpu_lp_owner
    )

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 16), dp_lockstep=False, speculative=True
        )

    tp1_shard = _shards(torch.zeros(8, 2), 1)[0]
    with pytest.raises(NotImplementedError, match="tp_size > 1"):
        distributed_sampling.resolve_head_owner(_Owner(tp1_shard, 8), dp_lockstep=False)

    with pytest.raises(NotImplementedError, match="expert-parallel data parallelism"):
        distributed_sampling.resolve_head_owner(_Owner(shard, 16), dp_lockstep=True)

    # DP_VOCAB_ATTR must report TPU_DP_VOCAB_PARALLEL even when dp_lockstep=True.
    dp_shard = _HeadShard(torch.zeros(8, 2), 0, tp_size=2, tp_rank=0, num_pad=0)
    setattr(dp_shard, dp_vocab_parallel.DP_VOCAB_ATTR, True)
    with pytest.raises(NotImplementedError, match="TPU_DP_VOCAB_PARALLEL"):
        distributed_sampling.resolve_head_owner(_Owner(dp_shard, 16), dp_lockstep=True)

    with pytest.raises(NotImplementedError, match="logits_as_input=True"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 16, logits_as_input=True), dp_lockstep=False
        )
    with pytest.raises(NotImplementedError, match="non-positive soft_cap"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 16, soft_cap=0.0), dp_lockstep=False
        )

    added_shard = _HeadShard(
        torch.zeros(8, 2), 0, tp_size=2, tp_rank=0, num_pad=0, num_added=4
    )
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        distributed_sampling.resolve_head_owner(
            _Owner(added_shard, 16), dp_lockstep=False
        )

    class _CustomLogitsProcessor(LogitsProcessor):
        pass

    with pytest.raises(NotImplementedError, match="_CustomLogitsProcessor"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 16, processor_cls=_CustomLogitsProcessor), dp_lockstep=False
        )


def test_resolve_head_owner_rejects_non_passthrough_compute_logits():
    """Models whose compute_logits edits hidden_states or logits are rejected."""

    class _PassthroughLM(_Owner):
        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, hidden_states)

    class _CleanWrapper(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm

        def compute_logits(self, hidden_states):
            return self.language_model.compute_logits(hidden_states)

    clean_lm = _PassthroughLM(_shards(torch.zeros(16, 2), 2)[0], 16)
    assert (
        distributed_sampling.resolve_head_owner(
            _CleanWrapper(clean_lm), dp_lockstep=False
        )
        is clean_lm
    )

    class _SuppressingWrapper(_CleanWrapper):
        def compute_logits(self, hidden_states):
            logits = super().compute_logits(hidden_states)
            logits[..., 0] = float("-inf")
            return logits

    suppressed_lm = _PassthroughLM(_shards(torch.zeros(16, 2), 2)[0], 16)
    with pytest.raises(NotImplementedError, match="custom pre- or post-processing"):
        distributed_sampling.resolve_head_owner(
            _SuppressingWrapper(suppressed_lm), dp_lockstep=False
        )

    class _PreNormLM(_Owner):
        def __init__(self, head):
            super().__init__(head, 16)
            self.norm = torch.nn.RMSNorm(2)

        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, self.norm(hidden_states))

    with pytest.raises(NotImplementedError, match="custom pre- or post-processing"):
        distributed_sampling.resolve_head_owner(
            _PreNormLM(_shards(torch.zeros(16, 2), 2)[0]), dp_lockstep=False
        )


def test_step_eligible_and_topp_only_fallback_routing():
    ib_greedy, reqs_greedy = _make_batch(SamplingParams(temperature=0.0))
    assert not distributed_sampling.step_eligible(ib_greedy, reqs_greedy)

    grammar = SamplingParams(
        temperature=1.0, structured_outputs=StructuredOutputsParams(regex="[ab]+")
    )
    assert not distributed_sampling.step_eligible(*_make_batch(grammar))
    assert not distributed_sampling.step_eligible(
        *_make_batch(SamplingParams(temperature=1.0), max_num_logprobs=1)
    )

    ib_k, reqs_k = _make_batch(
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
    )
    assert distributed_sampling.step_eligible(ib_k, reqs_k)
    assert distributed_sampling.select_chunk_mode(ib_k, 0, 2, 100000) is Mode.TOPK_TOPP

    ib_t, _ = _make_batch(
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=-1, top_p=1.0),
    )
    assert distributed_sampling.select_chunk_mode(ib_t, 0, 2, 100000) is Mode.PURE_TEMP

    # Pure top_p-only (top_k=-1, top_p=0.95) -> None (full-logits fallback).
    ib_p, reqs_p = _make_batch(SamplingParams(temperature=0.8, top_k=-1, top_p=0.95))
    assert distributed_sampling.step_eligible(ib_p, reqs_p)
    assert distributed_sampling.select_chunk_mode(ib_p, 0, 1, 100000) is None
    assert not distributed_sampling.step_has_eligible_request(ib_p, 100000)

    # Mixed batches containing a top_p-only request fall back on the combined
    # slice, while per-chunk selection routes chunk 0 to TOPK_TOPP and chunk 1
    # to full logits.
    ib_mix_kp, _ = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    assert distributed_sampling.select_chunk_mode(ib_mix_kp, 0, 2, 100000) is None
    assert (
        distributed_sampling.select_chunk_mode(ib_mix_kp, 0, 1, 100000)
        is Mode.TOPK_TOPP
    )
    assert distributed_sampling.select_chunk_mode(ib_mix_kp, 1, 2, 100000) is None
    assert distributed_sampling.step_has_eligible_request(ib_mix_kp, 100000)

    ib_mix_tp, _ = _make_batch(
        SamplingParams(temperature=1.0, top_k=-1, top_p=1.0),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    assert distributed_sampling.select_chunk_mode(ib_mix_tp, 0, 2, 100000) is None

    ib_wide, _ = _make_batch(SamplingParams(temperature=1.0, top_k=200, top_p=0.95))
    assert distributed_sampling.select_chunk_mode(ib_wide, 0, 1, 100000) is None
    assert not distributed_sampling.step_has_eligible_request(ib_wide, 100000)


def test_runner_distributed_sampling_step_gating():
    """Step gating checks chunk eligibility plus active logits-level features."""
    runner = object.__new__(TPUModelRunner)
    runner._distributed_sampling_owner = object()
    runner.vocab_size = 100000
    runner._min_tokens = SimpleNamespace(any_in=lambda _idx: False)
    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: False)
    runner._thinking_budget = None

    ib_topp, reqs_topp = _make_batch(
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95)
    )
    runner.input_batch = ib_topp
    runner.requests = reqs_topp
    assert not runner._distributed_sampling_this_step(target_num_chunks=1)
    assert not runner._distributed_sampling_this_step(target_num_chunks=2)

    ib_mix, reqs_mix = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    runner.input_batch = ib_mix
    runner.requests = reqs_mix
    assert not runner._distributed_sampling_this_step(target_num_chunks=1)
    assert runner._distributed_sampling_this_step(target_num_chunks=2)

    ib_ok, reqs_ok = _make_batch(SamplingParams(temperature=1.0, top_k=32, top_p=0.95))
    runner.input_batch = ib_ok
    runner.requests = reqs_ok
    assert runner._distributed_sampling_this_step(target_num_chunks=1)

    # Preempted thinking-budget request not in input_batch does not block step.
    runner._thinking_budget = SimpleNamespace(requests={"preempted_req": 100})
    assert runner._distributed_sampling_this_step(target_num_chunks=1)

    # Active thinking-budget request in input_batch gates distributed sampling off.
    runner._thinking_budget = SimpleNamespace(requests={"r0": 100})
    assert not runner._distributed_sampling_this_step(target_num_chunks=1)
    runner._thinking_budget = None

    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: True)
    assert not runner._distributed_sampling_this_step(target_num_chunks=1)
    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: False)

    runner._min_tokens = SimpleNamespace(any_in=lambda _idx: True)
    assert not runner._distributed_sampling_this_step(target_num_chunks=1)


def test_sample_plain_chunk_dispatches_distributed_and_falls_back_per_chunk():
    """Eligible chunks run _sample_distributed; ineligible chunks project logits."""
    ib_mix, _ = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    dist_calls: list[tuple[Mode, int, int]] = []
    projected_shapes: list[tuple[int, ...]] = []
    full_sampled: list[tuple[int, ...]] = []

    runner = SimpleNamespace(
        input_batch=ib_mix,
        vocab_size=128,
        _stage_timing=MagicMock(),
        _sample_distributed=lambda h, mode, s, e, _g: (
            dist_calls.append((mode, s, e))
            or torch.full((h.shape[0], 1), 7, dtype=torch.int64)
        ),
        compute_logits_from_hidden_states=lambda h: (
            projected_shapes.append(tuple(h.shape)) or torch.zeros((h.shape[0], 128))
        ),
        _sample_chunk_logits=lambda logits, **_kw: (
            full_sampled.append(tuple(logits.shape))
            or (torch.full((logits.shape[0], 1), 11, dtype=torch.int64), None)
        ),
    )
    runner._sample_plain_chunk = TPUModelRunner._sample_plain_chunk.__get__(runner)
    sched = SimpleNamespace(scheduled_spec_decode_tokens={})

    tok0, _ = runner._sample_plain_chunk(
        torch.zeros((8, 16)),
        cur_start_idx=0,
        cur_end_idx=1,
        top_tokens=False,
        use_distributed_sampling=True,
        all_greedy=False,
        sampling_generator=None,
        grammar_output=None,
        scheduler_output=sched,
        is_spec_step=False,
        min_toks={},
    )
    assert dist_calls == [(Mode.TOPK_TOPP, 0, 1)]
    assert projected_shapes == []
    assert int(tok0[0, 0].item()) == 7

    tok1, _ = runner._sample_plain_chunk(
        torch.zeros((8, 16)),
        cur_start_idx=1,
        cur_end_idx=2,
        top_tokens=False,
        use_distributed_sampling=True,
        all_greedy=False,
        sampling_generator=None,
        grammar_output=None,
        scheduler_output=sched,
        is_spec_step=False,
        min_toks={},
    )
    assert projected_shapes == [(8, 16)]
    assert full_sampled == [(8, 128)]
    assert int(tok1[0, 0].item()) == 11


@pytest.mark.parametrize("mode", list(Mode))
def test_runner_sample_distributed_routes_mode_and_preserves_fp32_temperature(mode):
    shard = _shards(torch.zeros(512, 8), 2, min_shard_rows=4096)[0]
    owner = distributed_sampling.resolve_head_owner(
        _Owner(shard, 512), dp_lockstep=False
    )
    ib, _ = _make_batch(
        SamplingParams(temperature=0.7, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.3, top_k=64, top_p=0.95),
    )
    ib.req_ids = ["__warmup_0__", None]
    runner = object.__new__(TPUModelRunner)
    runner._distributed_sampling_owner = owner
    runner.input_batch = ib
    runner.vocab_size = 512

    seen: dict[Mode, tuple[torch.Tensor, tuple[int, ...]]] = {}

    def record(m: Mode):
        def fn(*args):
            seen[m] = (args[1], tuple(args[-1].shape))
            return torch.zeros((args[0].shape[0], 1), dtype=torch.int64)

        return fn

    runner.sample_distributed_topk_topp = record(Mode.TOPK_TOPP)
    runner.sample_distributed_pure_temp = record(Mode.PURE_TEMP)
    runner._sample_distributed(
        torch.zeros((4, 8), dtype=torch.bfloat16), mode, 0, 2, None
    )
    temps, u_shape = seen[mode]
    assert temps.dtype == torch.float32
    torch.testing.assert_close(
        temps[:2, 0], torch.tensor([0.7, 0.3], dtype=torch.float32), rtol=0, atol=0
    )
    assert u_shape == (4, distributed_sampling.rng_width(mode, 2, 4096))


def test_log_step_path_skips_synthetic_warmup_and_logs_once():
    if hasattr(vllm_logger, "_print_info_once"):
        vllm_logger._print_info_once.cache_clear()
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    vllm_log = logging.getLogger("vllm")
    old_level = vllm_log.level
    old_disable = logging.root.manager.disable
    vllm_log.addHandler(handler)
    vllm_log.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        distributed_sampling.log_step_path(Mode.TOPK_TOPP, ["__warmup_0__", None])
        assert records == []
        for mode in (Mode.TOPK_TOPP, Mode.TOPK_TOPP, Mode.PURE_TEMP):
            distributed_sampling.log_step_path(mode, ["__warmup_0__", "req-1"])
    finally:
        vllm_log.removeHandler(handler)
        vllm_log.setLevel(old_level)
        logging.disable(old_disable)
    assert [r.getMessage() for r in records] == [
        "TPU_DISTRIBUTED_SAMPLING: step took mode=topk_topp",
        "TPU_DISTRIBUTED_SAMPLING: step took mode=pure_temp",
    ]


def test_env_default_off_and_forwarded_to_ray_workers(monkeypatch):
    assert "TPU_DISTRIBUTED_SAMPLING" in TpuPlatform.additional_env_vars
    monkeypatch.delenv("TPU_DISTRIBUTED_SAMPLING", raising=False)
    assert envs.TPU_DISTRIBUTED_SAMPLING is False
    monkeypatch.setenv("TPU_DISTRIBUTED_SAMPLING", "1")
    assert envs.TPU_DISTRIBUTED_SAMPLING is True
    monkeypatch.setenv("TPU_DISTRIBUTED_SAMPLING", "0")
    assert envs.TPU_DISTRIBUTED_SAMPLING is False


def test_precompile_distributed_sampling_uses_no_grad_and_fresh_fp32_inputs(
    monkeypatch,
):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    shard = _shards(torch.zeros(512, 16), 2, min_shard_rows=256)[0]
    owner = distributed_sampling.resolve_head_owner(
        _Owner(shard, 512), dp_lockstep=False
    )
    select_grad_modes: list[bool] = []
    seen_u: list[torch.Tensor] = []
    seen_temp_dtypes: list[torch.dtype] = []

    def _rec_select(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        select_grad_modes.append(torch.is_grad_enabled())
        return h.index_select(0, idx)

    def _rec_sample(temps: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        seen_temp_dtypes.append(temps.dtype)
        seen_u.append(u)
        return torch.zeros((u.shape[0], 1), dtype=torch.int32)

    runner = SimpleNamespace(
        _distributed_sampling_owner=owner,
        get_pcp_planner=lambda: SimpleNamespace(
            uses_selected_logits_hidden_states=False
        ),
        model_config=SimpleNamespace(get_hidden_size=lambda: 16),
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.bfloat16,
        num_tokens_paddings=[8, 16],
        num_reqs_paddings=[4, 8],
        max_num_reqs=8,
        _precompile_timed=lambda _name: contextlib.nullcontext(),
        select_sample_hidden_states=_rec_select,
        sample_distributed_topk_topp=lambda _h, t, _k, _p, u: _rec_sample(t, u),
        sample_distributed_pure_temp=lambda _h, t, u: _rec_sample(t, u),
    )
    TPUModelRunner._precompile_distributed_sampling(runner)
    assert select_grad_modes and not any(select_grad_modes)
    assert seen_temp_dtypes == [torch.float32] * 4
    assert len(seen_u) == 4
    assert len({id(t) for t in seen_u}) == 4
    assert not torch.equal(seen_u[0][:4, :8], seen_u[1][:4, :8])


@pytest.mark.parametrize("tp_size", [2, 8])
def test_spmd_compile_graph_symmetry_across_tp_ranks(monkeypatch, tp_size):
    """Compiled graphs must be rank-symmetric across TP=2 and TP=8 (no rank ints)."""
    monkeypatch.setattr(
        distributed_sampling,
        "tensor_model_parallel_all_gather",
        lambda t, dim=-1: torch.cat([t] * tp_size, dim=dim),
    )
    vocab, hidden = (475 if tp_size == 2 else 9501), 16
    shards = _shards(
        torch.zeros((vocab, hidden), dtype=torch.bfloat16),
        tp_size,
        min_shard_rows=1792 if tp_size == 2 else 1280,
    )
    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, scale=0.9, soft_cap=30.0), dp_lockstep=False
        )
        for r in range(tp_size)
    ]

    for r in range(tp_size):
        assert isinstance(owners[r].lm_head.org_vocab_start_index, torch.Tensor)
        assert isinstance(owners[r].lm_head.num_org_vocab_padding, torch.Tensor)
        assert isinstance(owners[r].lm_head._tpu_tp_rank, torch.Tensor)
        assert owners[r].lm_head._tpu_tp_rank.tolist() == r

    h = torch.zeros((4, hidden), dtype=torch.bfloat16)
    temps = torch.ones((4, 1), dtype=torch.float32)
    top_k = torch.full((4, 1), 16, dtype=torch.int32)
    top_p = torch.full((4, 1), 0.9, dtype=torch.float32)
    _, v_loc = distributed_sampling.head_tp_and_shard_vocab(owners[0])
    u_tktp = torch.full(
        (4, distributed_sampling.rng_width(Mode.TOPK_TOPP, tp_size, v_loc)), 0.5
    )
    u_temp = torch.full(
        (4, distributed_sampling.rng_width(Mode.PURE_TEMP, tp_size, v_loc)), 0.5
    )

    for fn, args in (
        (distributed_sampling.sample_topk_topp, (h, temps, top_k, top_p, u_tktp)),
        (distributed_sampling.sample_pure_temp, (h, temps, u_temp)),
    ):
        codes: list[str] = []
        for r in range(tp_size):
            torch._dynamo.reset()

            def backend(gm, _example_inputs, bucket=codes):
                bucket.append(gm.code)
                return gm.forward

            compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
            compiled(owners[r], *args)
        assert len(codes) == tp_size
        for r in range(1, tp_size):
            assert codes[r] == codes[0]
