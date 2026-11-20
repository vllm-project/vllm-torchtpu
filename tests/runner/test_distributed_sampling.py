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

import numpy as np
import pytest
import torch
from runner_output_test_utils import _host_copy_state
from vllm import SamplingParams
from vllm import logger as vllm_logger
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams

from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample import (
    distributed_sampling as distributed_sampling_ops,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.runner import distributed_sampling, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState
from vllm_torchtpu.runner.utils import VOCAB_PARALLEL_SAMPLING_KEY

Mode = distributed_sampling_ops.DistributedSamplingMode
Config = distributed_sampling.VocabParallelSamplingConfig
_MIN_SHARD_ROWS = 128


class _HeadShard(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        start: int,
        tp_size: int,
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
        distributed_sampling_ops, "tensor_model_parallel_all_gather", lockstep_gather
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


@contextlib.contextmanager
def _capture_vllm_logs(level: int):
    for fn_name in ("_print_info_once", "_print_warning_once"):
        fn = getattr(vllm_logger, fn_name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    vllm_log = logging.getLogger("vllm")
    old_level = vllm_log.level
    old_disable = logging.root.manager.disable
    vllm_log.addHandler(handler)
    vllm_log.setLevel(level)
    logging.disable(logging.NOTSET)
    try:
        yield records
    finally:
        vllm_log.removeHandler(handler)
        vllm_log.setLevel(old_level)
        logging.disable(old_disable)


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
    assert all(o is not None for o in owners)

    hidden_t = torch.zeros((2, hidden), dtype=torch.bfloat16)
    hidden_t[:, 0] = 1.0
    full_capped = torch.tanh((hidden_t @ weight.T) / 30.0) * 30.0
    assert full_capped[0, 3] == full_capped[0, 9] == full_capped[0, 135]
    ref_greedy = torch.argmax(full_capped[:1], dim=-1, keepdim=True)
    assert int(ref_greedy.item()) == 3

    temps = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    top_k = torch.tensor([[0], [16]], dtype=torch.int32)
    top_p = torch.ones((2, 1), dtype=torch.float32)
    u = torch.full(
        (2, distributed_sampling_ops.rng_width(Mode.TOPK_TOPP, tp_size, 128)),
        0.5,
        dtype=torch.float32,
    )
    outs = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_topk_topp(
            owners[r], hidden_t, temps, top_k, top_p, u
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(outs[0][:1], ref_greedy)


def test_softcap_preserves_masked_vocab_padding(monkeypatch):
    """soft_cap must never turn -inf vocab-padding slots into finite -soft_cap."""
    gen = torch.Generator().manual_seed(99)
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
        (64, distributed_sampling_ops.rng_width(Mode.TOPK_TOPP, tp_size, 128)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_tktp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_topk_topp(
            owners[r], hidden_states, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(outs_tktp[0], outs_tktp[1])
    assert int(outs_tktp[0].max().item()) < vocab

    u_temp = torch.rand(
        (64, distributed_sampling_ops.rng_width(Mode.PURE_TEMP, tp_size, 128)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_temp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_pure_temp(
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
    assert quant_owners[0] is not None
    assert distributed_sampling.head_tp_and_shard_vocab(quant_owners[0]) == (2, 150)

    h = torch.randn(8, hidden, generator=gen, dtype=torch.float32)
    temps = torch.tensor([[0.0], [1.0], [0.8], [1.2]] * 2, dtype=torch.float32)
    top_k = torch.full((8, 1), 32, dtype=torch.int32)
    top_p = torch.full((8, 1), 0.9, dtype=torch.float32)

    u_tktp = torch.rand(
        (8, distributed_sampling_ops.rng_width(Mode.TOPK_TOPP, tp_size, 150)),
        generator=gen,
        dtype=torch.float32,
    )
    ref_tktp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_topk_topp(
            ref_owners[r], h, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    quant_tktp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_topk_topp(
            quant_owners[r], h, temps, top_k, top_p, u_tktp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(ref_tktp[0], quant_tktp[0])

    u_temp = torch.rand(
        (8, distributed_sampling_ops.rng_width(Mode.PURE_TEMP, tp_size, 150)),
        generator=gen,
        dtype=torch.float32,
    )
    ref_temp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_pure_temp(
            ref_owners[r], h, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    quant_temp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_pure_temp(
            quant_owners[r], h, temps, u_temp
        ),
        tp_size,
        monkeypatch,
    )
    assert torch.equal(ref_temp[0], quant_temp[0])


def test_two_stage_extraction_exact_when_all_top_k_cluster_in_one_lane():
    """Two-stage block-max + topk preserves all top-100 tokens even when every
    top token shares a single mod-256 lane or lies contiguously on one shard."""
    shard_vocab = 4205  # > 128 * 32 = 4096 and not divisible by 32 (exercises padding)
    vocab_start = torch.tensor(1000, dtype=torch.int32)
    logits = torch.full((2, shard_vocab), -10.0, dtype=torch.float32)

    # Row 0: 16 top tokens all in the same mod-256 lane (7, 263, 519, ...).
    same_lane_ids = [7 + 256 * i for i in range(16)]
    for rank, idx in enumerate(same_lane_ids):
        logits[0, idx] = 20.0 - 0.1 * rank

    # Row 1: 100 top tokens packed contiguously at [0, 100) on one shard.
    contig_ids = list(range(100))
    for rank, idx in enumerate(contig_ids):
        logits[1, idx] = 20.0 - 0.05 * rank

    vals, ids = distributed_sampling_ops._extract_candidates(logits, vocab_start)
    assert vals.shape == (2, distributed_sampling_ops.CANDIDATES_PER_SHARD)
    assert {i + 1000 for i in same_lane_ids}.issubset(set(ids[0].tolist()))
    assert {i + 1000 for i in contig_ids}.issubset(set(ids[1].tolist()))


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("soft_cap", [None, 30.0])
def test_distributed_sampling_modes_lockstep_padding_and_ties(
    monkeypatch, tp_size, soft_cap
):
    gen = torch.Generator().manual_seed(42 + tp_size)
    # Non-multiple shard width (> 4096 for TP=2 to exercise two-stage + padding).
    vocab, hidden = (8501 if tp_size == 2 else 9205), 16
    weight = (torch.randn(vocab, hidden, generator=gen) * 0.5).to(torch.bfloat16)
    # Row 0 is greedy with an exact tie across shards: lowest token ID 5 wins.
    weight[:, 0] = 0.0
    weight[5, 0] = 4.0
    weight[vocab - 7, 0] = 4.0
    shards = _shards(weight, tp_size, min_shard_rows=4205 if tp_size == 2 else 1280)

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
    assert owners[0] is not None
    _, shard_vocab = distributed_sampling.head_tp_and_shard_vocab(owners[0])

    u_tktp = torch.rand(
        (4, distributed_sampling_ops.rng_width(Mode.TOPK_TOPP, tp_size, shard_vocab)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_tktp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_topk_topp(
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
        (4, distributed_sampling_ops.rng_width(Mode.PURE_TEMP, tp_size, shard_vocab)),
        generator=gen,
        dtype=torch.float32,
    )
    outs_temp = _run_across_ranks(
        lambda r: distributed_sampling_ops.sample_pure_temp(
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
    # Use shard_vocab > 4096 for TOPK_TOPP to exercise the two-stage block-max
    # + topk path; keep PURE_TEMP at 4096 so finite-sample KL bias (V / 2N)
    # stays small with 16k draws.
    vocab = 4096 if mode is Mode.PURE_TEMP else (9000 if tp_size == 2 else 17000)
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
    assert owners[0] is not None
    _, shard_vocab = distributed_sampling.head_tp_and_shard_vocab(owners[0])
    n_draws = 16384
    batch = 256
    h_rep = h.expand(batch, hidden).contiguous()
    t_rep = torch.ones((batch, 1), dtype=torch.float32)
    k_rep = torch.full((batch, 1), max(top_k_val, 0), dtype=torch.int32)
    p_rep = torch.full((batch, 1), top_p_val, dtype=torch.float32)
    counts = torch.zeros(vocab, dtype=torch.float32)

    u_width = distributed_sampling_ops.rng_width(mode, tp_size, shard_vocab)
    for _ in range(n_draws // batch):
        u = torch.rand((batch, u_width), generator=gen, dtype=torch.float32)
        if mode is Mode.TOPK_TOPP:
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling_ops.sample_topk_topp(
                    owners[r], h_rep, t_rep, k_rep, p_rep, u_step
                ),
                tp_size,
                monkeypatch,
            )
        else:
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling_ops.sample_pure_temp(
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


def test_config_enum_validation():
    assert distributed_sampling.get_vocab_parallel_sampling_config(None) is Config.OFF
    assert (
        distributed_sampling.get_vocab_parallel_sampling_config(
            SimpleNamespace(additional_config={})
        )
        is Config.OFF
    )
    for raw, expected in (
        ("off", Config.OFF),
        ("greedy", Config.GREEDY),
        ("all", Config.ALL),
    ):
        cfg = SimpleNamespace(additional_config={VOCAB_PARALLEL_SAMPLING_KEY: raw})
        assert distributed_sampling.get_vocab_parallel_sampling_config(cfg) is expected

    for bad in (True, 1, "distributed", None):
        cfg = SimpleNamespace(additional_config={VOCAB_PARALLEL_SAMPLING_KEY: bad})
        with pytest.raises(ValueError, match=VOCAB_PARALLEL_SAMPLING_KEY):
            distributed_sampling.get_vocab_parallel_sampling_config(cfg)


def test_resolve_head_owner_ignores_ineligible_configs_with_warning():
    shard = _shards(torch.zeros(16, 2), 2)[0]

    tpu_lp_owner = _Owner(
        _shards(torch.zeros(16, 2), 2)[0],
        16,
        processor_cls=dp_vocab_parallel.TpuLogitsProcessor,
    )
    assert (
        distributed_sampling.resolve_head_owner(tpu_lp_owner, dp_lockstep=False)
        is tpu_lp_owner
    )

    tp1_shard = _shards(torch.zeros(8, 2), 1)[0]
    dp_shard = _HeadShard(torch.zeros(8, 2), 0, tp_size=2, num_pad=0)
    setattr(dp_shard, dp_vocab_parallel.DP_VOCAB_ATTR, True)
    added_shard = _HeadShard(torch.zeros(8, 2), 0, tp_size=2, num_pad=0, num_added=4)
    no_buf_shard = _HeadShard(torch.zeros(8, 2), 0, tp_size=2, num_pad=0)
    del no_buf_shard._buffers["org_vocab_start_index"]

    class _CustomLogitsProcessor(LogitsProcessor):
        pass

    lora_owner = _Owner(shard, 16)
    lora_owner.logits_processor = MagicMock(spec=LogitsProcessorWithLoRA)
    lora_owner.logits_processor.scale = 1.0

    ineligible_cases = [
        (_Owner(shard, 16), False, True, "speculative decoding"),
        (_Owner(tp1_shard, 8), False, False, "tp_size <= 1"),
        (_Owner(shard, 16), True, False, "expert-parallel data parallelism"),
        (_Owner(dp_shard, 16), True, False, "TPU_DP_VOCAB_PARALLEL"),
        (_Owner(shard, 16, logits_as_input=True), False, False, "logits_as_input=True"),
        (_Owner(shard, 16, soft_cap=0.0), False, False, "non-positive soft_cap"),
        (_Owner(shard, 16, scale=0.0), False, False, "non-positive logit scale"),
        (_Owner(added_shard, 16), False, False, "added vocabulary"),
        (
            _Owner(shard, 16, processor_cls=_CustomLogitsProcessor),
            False,
            False,
            "_CustomLogitsProcessor",
        ),
        (lora_owner, False, False, "LoRA-wrapped"),
        (_Owner(no_buf_shard, 16), False, False, "not registered as buffers"),
    ]
    for model, dp_lockstep, speculative, expected_substr in ineligible_cases:
        with _capture_vllm_logs(logging.WARNING) as records:
            assert (
                distributed_sampling.resolve_head_owner(
                    model, dp_lockstep=dp_lockstep, speculative=speculative
                )
                is None
            )
        assert any(expected_substr in r.getMessage() for r in records), (
            expected_substr,
            [r.getMessage() for r in records],
        )


def test_probe_enforces_exact_head_bias_and_allows_vllm_torchtpu_passthrough():
    """Probe accepts matching lm_head.bias and vllm_torchtpu passthroughs, and
    rejects omitted/mismatched bias or custom compute_logits."""

    class _PassthroughLM(_Owner):
        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, hidden_states)

    class _BiasForwardingLM(_Owner):
        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)

    class _WrongBiasLM(_Owner):
        def __init__(self, head, vocab):
            super().__init__(head, vocab)
            self.other_bias = torch.zeros_like(head.bias)

        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, hidden_states, self.other_bias)

    class _CleanWrapper(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm

        def compute_logits(self, hidden_states):
            return self.language_model.compute_logits(hidden_states)

    class _SuppressingWrapper(_CleanWrapper):
        def compute_logits(self, hidden_states):
            logits = super().compute_logits(hidden_states)
            logits[..., 0] = float("-inf")
            return logits

    class _PreNormLM(_Owner):
        def __init__(self, head):
            super().__init__(head, 16)
            self.norm = torch.nn.RMSNorm(2)

        def compute_logits(self, hidden_states):
            return self.logits_processor(self.lm_head, self.norm(hidden_states))

    # Clean passthrough wrapper -> accepted.
    clean_lm = _PassthroughLM(_shards(torch.zeros(16, 2), 2)[0], 16)
    assert (
        distributed_sampling.resolve_head_owner(
            _CleanWrapper(clean_lm), dp_lockstep=False
        )
        is clean_lm
    )

    # Passthrough method defined under vllm_torchtpu.* -> accepted (not rejected
    # by module name, e.g. DeepseekV4ForCausalLM / KimiLinearForCausalLM).
    tpu_passthrough = _PassthroughLM(_shards(torch.zeros(16, 2), 2)[0], 16)
    tpu_cls = type(
        "_TpuPassthroughLM",
        (_PassthroughLM,),
        {"__module__": "vllm_torchtpu.models.vllm.deepseek_v4.model"},
    )
    tpu_cls.compute_logits = lambda self, h: self.logits_processor(self.lm_head, h)
    tpu_cls.compute_logits.__module__ = "vllm_torchtpu.models.vllm.deepseek_v4.model"
    tpu_model = tpu_cls(_shards(torch.zeros(16, 2), 2)[0], 16)
    assert distributed_sampling.resolve_head_owner(tpu_model, dp_lockstep=False) is (
        tpu_model
    )
    assert (
        distributed_sampling.resolve_greedy_head_owner(
            tpu_model, dp_lockstep=False, config=Config.ALL
        )
        is tpu_model
    )
    _ = tpu_passthrough

    # Head with bias: accepted iff compute_logits passes the exact head.bias.
    bias_shard = _shards(torch.zeros(16, 2), 2, bias=torch.ones(16))[0]
    bias_ok = _BiasForwardingLM(bias_shard, 16)
    assert distributed_sampling.resolve_head_owner(bias_ok, dp_lockstep=False) is (
        bias_ok
    )
    # Greedy top-tokens ignores head bias until upstream forwarding lands.
    assert (
        distributed_sampling.resolve_greedy_head_owner(
            bias_ok, dp_lockstep=False, config=Config.GREEDY
        )
        is None
    )
    # Omitting head.bias or passing a different tensor is rejected.
    assert (
        distributed_sampling.resolve_head_owner(
            _PassthroughLM(bias_shard, 16), dp_lockstep=False
        )
        is None
    )
    assert (
        distributed_sampling.resolve_head_owner(
            _WrongBiasLM(bias_shard, 16), dp_lockstep=False
        )
        is None
    )

    # Custom pre/post processing -> rejected (returns None).
    suppressed = _SuppressingWrapper(
        _PassthroughLM(_shards(torch.zeros(16, 2), 2)[0], 16)
    )
    assert (
        distributed_sampling.resolve_head_owner(suppressed, dp_lockstep=False) is None
    )
    assert (
        distributed_sampling.resolve_head_owner(
            _PreNormLM(_shards(torch.zeros(16, 2), 2)[0]), dp_lockstep=False
        )
        is None
    )


def test_step_eligible_and_mixed_chunk_routing():
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

    # Mixed TOPK_TOPP + PURE_TEMP (+ greedy) chunk -> Mode.MIXED (eligible!).
    ib_mixed, _ = _make_batch(
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.9, top_k=-1, top_p=1.0),
    )
    assert distributed_sampling.select_chunk_mode(ib_mixed, 0, 3, 100000) is Mode.MIXED

    # Pure top_p-only (top_k=-1, top_p=0.95) -> None (full-logits fallback).
    ib_p, reqs_p = _make_batch(SamplingParams(temperature=0.8, top_k=-1, top_p=0.95))
    assert distributed_sampling.step_eligible(ib_p, reqs_p)
    assert distributed_sampling.select_chunk_mode(ib_p, 0, 1, 100000) is None

    # A chunk mixing an eligible row with a top_p-only row falls back to None,
    # while separate chunks route individually.
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

    ib_wide, _ = _make_batch(SamplingParams(temperature=1.0, top_k=200, top_p=0.95))
    assert distributed_sampling.select_chunk_mode(ib_wide, 0, 1, 100000) is None


def test_runner_distributed_sampling_step_gating():
    """Step gating checks owner plus active logits-level features."""
    runner = object.__new__(TPUModelRunner)
    runner._distributed_sampling_owner = object()
    runner.vocab_size = 100000
    runner._min_tokens = SimpleNamespace(any_in=lambda _idx: False)
    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: False)
    runner._thinking_budget = None

    ib_ok, reqs_ok = _make_batch(SamplingParams(temperature=1.0, top_k=32, top_p=0.95))
    runner.input_batch = ib_ok
    runner.requests = reqs_ok
    assert runner._distributed_sampling_step_eligible()

    # Preempted thinking-budget request not in input_batch does not block step.
    runner._thinking_budget = SimpleNamespace(requests={"preempted_req": 100})
    assert runner._distributed_sampling_step_eligible()

    # Active thinking-budget request in input_batch gates distributed sampling off.
    runner._thinking_budget = SimpleNamespace(requests={"r0": 100})
    assert not runner._distributed_sampling_step_eligible()
    runner._thinking_budget = None

    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: True)
    assert not runner._distributed_sampling_step_eligible()
    runner._repetition_penalty = SimpleNamespace(active=lambda _ib: False)

    runner._min_tokens = SimpleNamespace(any_in=lambda _idx: True)
    assert not runner._distributed_sampling_step_eligible()


@pytest.mark.parametrize("mode", list(Mode))
def test_runner_sample_distributed_routes_mode_and_preserves_fp32_temperature(mode):
    shard = _shards(torch.zeros(512, 8), 2, min_shard_rows=4096)[0]
    owner = distributed_sampling.resolve_head_owner(
        _Owner(shard, 512), dp_lockstep=False
    )
    ib, _ = _make_batch(
        SamplingParams(temperature=0.7, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.3, top_k=-1, top_p=1.0),
    )
    ib.req_ids = ["__warmup_0__", "__warmup_1__", None]
    runner = object.__new__(TPUModelRunner)
    runner._distributed_sampling_owner = owner
    runner.input_batch = ib
    runner.vocab_size = 512

    seen: dict[Mode, tuple[torch.Tensor, tuple[int, ...]]] = {}

    def record(m: Mode, tok_val: int):
        def fn(*args):
            seen[m] = (args[1], tuple(args[-1].shape))
            return torch.full((args[0].shape[0], 1), tok_val, dtype=torch.int64)

        return fn

    runner.sample_distributed_topk_topp = record(Mode.TOPK_TOPP, 11)
    runner.sample_distributed_pure_temp = record(Mode.PURE_TEMP, 22)
    runner.select_mixed_distributed_tokens = lambda tk, topk_t, pure_t: torch.where(
        tk > 0, topk_t, pure_t
    )
    out = runner._sample_distributed(
        torch.zeros((4, 8), dtype=torch.bfloat16), mode, 0, 2, None
    )
    if mode is Mode.MIXED:
        assert set(seen) == {Mode.TOPK_TOPP, Mode.PURE_TEMP}
        assert out[:2, 0].tolist() == [11, 22]
        for sub_mode in (Mode.TOPK_TOPP, Mode.PURE_TEMP):
            temps, u_shape = seen[sub_mode]
            assert temps.dtype == torch.float32
            assert u_shape == (
                4,
                distributed_sampling_ops.rng_width(sub_mode, 2, 4096),
            )
    else:
        temps, u_shape = seen[mode]
        assert temps.dtype == torch.float32
        torch.testing.assert_close(
            temps[:2, 0], torch.tensor([0.7, 0.3], dtype=torch.float32), rtol=0, atol=0
        )
        assert u_shape == (4, distributed_sampling_ops.rng_width(mode, 2, 4096))


@pytest.mark.parametrize("preselected_hidden", [False, True])
def test_runner_execute_model_and_sample_tokens_with_distributed_owner(
    preselected_hidden,
):
    """Real TPUModelRunner.execute_model + sample_tokens with an active owner:
    per-chunk modes route eligible chunks through hidden states and fallback
    chunks through full logits."""
    # Chunk 0: req 0 (TOPK_TOPP); Chunk 1: req 1,2 (MIXED); Chunk 2: req 3 (top_p-only).
    ib, reqs = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=1.0, top_k=32, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=1.0),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
        vocab_size=128,
    )
    ib.num_tokens_no_spec = [1, 1, 1, 1]
    ib.token_ids_cpu = np.zeros((4, 8), dtype=np.int32)
    ib.req_ids = ["r0", "r1", "r2", "r3"]
    for rid, st in reqs.items():
        st.num_computed_tokens = 0
        st.num_tokens = 1
        st.output_token_ids = []
        _ = rid

    shard = _shards(torch.zeros(128, 8), 2, min_shard_rows=128)[0]
    owner = distributed_sampling.resolve_head_owner(
        _Owner(shard, 128), dp_lockstep=False
    )
    assert owner is not None

    select_calls: list[int] = []
    full_selected_calls: list[int] = []
    full_from_hidden_calls: list[int] = []
    dist_sample_calls: list[tuple[Mode, int, int]] = []
    full_sample_calls: list[tuple[int, int]] = []

    chunk_spans = [(0, 1), (1, 3), (3, 4)]
    chunk_iter = iter(chunk_spans)

    def _prepare_inputs(_so, start_index, _num_decode, _num_windowed):
        s, e = next(chunk_iter)
        assert s == start_index
        n = e - s
        empty = torch.empty(0, dtype=torch.int32)
        return (
            None,
            torch.arange(n, dtype=torch.int64),
            n,
            n,
            e,
            empty,
            empty,
            None,
            None,
            None,
        )

    runner = SimpleNamespace(
        _pp_handoff=None,
        _dp_step_other_decode=False,
        _thinking_budget=None,
        _routed_experts=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pp_pending_scheduler_output=None,
        _pp_pending_kv_save_nan_check=None,
        execute_model_state=None,
        _update_states=lambda _so: None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _mtp_prefill_only=False,
        _reorder_batch_for_rpa=lambda _so: (0, 0),
        _count_input_chunks=lambda _so: (len(chunk_spans), 2),
        _dp_coordinated_step=lambda *_a, **_kw: (None, len(chunk_spans)),
        _dp_local_prompt_logprob_rows=lambda _so: 0,
        _dp_step_prompt_logprob_rows=0,
        _dp_num_tokens_across_dp=lambda _n: None,
        supports_mm_inputs=False,
        phase_based_profiler=None,
        _is_async_drafter=False,
        _greedy_head_owner=None,
        _distributed_sampling_owner=owner,
        vocab_size=128,
        _min_tokens=SimpleNamespace(any_in=lambda _idx: False),
        _min_tokens_batch_state=lambda: {},
        _repetition_penalty=SimpleNamespace(
            active=lambda _ib: False, begin_step=lambda _ib: False
        ),
        vllm_config=SimpleNamespace(),
        maybe_setup_kv_connector=lambda *_a, **_kw: None,
        input_batch=ib,
        requests=reqs,
        num_prompt_logprobs={},
        _run_dp_dummy_chunk=MagicMock(),
        query_start_loc_np=np.zeros(4, dtype=np.int32),
        _last_sequence_layout_plan=None,
        position_ids=torch.zeros(4, dtype=torch.int32),
        _flush_mamba_state_seed_copies=lambda: None,
        _reset_read_offsets_for_new_requests=lambda *_a: None,
        input_ids=torch.zeros(4, dtype=torch.int32),
        _apply_async_token_substitution=lambda ids, *_a: ids,
        _record_input_staging_fence=lambda: None,
        _get_model_inputs=lambda ids, _mm: (ids, None),
        _token_padding_update=lambda _n: None,
        _pp_take_topk_indices=lambda *_a: None,
        mesh=None,
        _kv_cache_bundle=None,
        _stage_timing=MagicMock(),
        is_pooling_model=False,
        pcp_manager=None,
        sequence_layout_planner=SimpleNamespace(
            maybe_select_logits_hidden_states=(
                (lambda hs, _plan, idx: hs.index_select(0, idx))
                if preselected_hidden
                else (lambda _hs, _plan, _idx: None)
            ),
            finalize_hidden_states=lambda hs, _plan: hs,
        ),
        select_sample_hidden_states=lambda h, idx: (
            select_calls.append(idx.shape[0]) or h.index_select(0, idx)
        ),
        _logits_at_dp_vocab_rows=lambda h, idx: (
            full_selected_calls.append(idx.shape[0]) or torch.zeros((idx.shape[0], 128))
        ),
        compute_logits_from_hidden_states=lambda h: (
            full_from_hidden_calls.append(h.shape[0]) or torch.zeros((h.shape[0], 128))
        ),
        _sample_distributed=lambda h, mode, s, e, _g: (
            dist_sample_calls.append((mode, s, e))
            or torch.full((h.shape[0], 1), 7, dtype=torch.int64)
        ),
        _sample_chunk_logits=lambda logits, cur_start_idx, cur_end_idx, **_kw: (
            full_sample_calls.append((cur_start_idx, cur_end_idx))
            or (torch.full((logits.shape[0], 1), 11, dtype=torch.int64), None)
        ),
        _attn_metadata_builder_ctx=SimpleNamespace(
            mamba_state_indices=None, unified_mamba_state_indices=None, seq_lens=None
        ),
        mamba_slot_read_offsets=None,
        speculative_config=None,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        _adaptive_verification=None,
        _get_sampling_generator=lambda: None,
        _update_mamba_slot_read_offsets=lambda *_a: None,
        _update_num_xla_graphs=lambda *_a: None,
        _prepare_inputs=_prepare_inputs,
        forward_model=MagicMock(
            side_effect=[(torch.zeros((4, 8)), None) for _ in chunk_spans]
        ),
    )
    runner._stage_timing.stage.return_value = contextlib.nullcontext()
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in (
        "_any_decode",
        "_readmit_trimmed_drafts",
        "_greedy_top_tokens_this_step",
        "_distributed_sampling_step_eligible",
        "_finalize_hidden_states",
        "_maybe_select_logits_hidden_states",
        "_sample_plain_chunk",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_build_kv_connector_output",
        "_sample_tokens_pp_intermediate_stage",
        "sample_tokens",
        "execute_model",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4,
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r0": 1, "r1": 1, "r2": 1, "r3": 1},
        finished_req_ids=set(),
        has_structured_output_requests=False,
        scheduled_spec_decode_tokens={},
    )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_forward_context",
            side_effect=lambda *_a, **_kw: contextlib.nullcontext(),
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
            side_effect=lambda *_a, **_kw: contextlib.nullcontext(),
        ),
        patch.object(
            AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy_state
        ),
    ):
        runner.execute_model(scheduler_output)
        assert runner.execute_model_state is not None
        assert runner.execute_model_state.distributed_sampling_modes == [
            Mode.TOPK_TOPP,
            Mode.MIXED,
            None,
        ]
        out = runner.sample_tokens(None)

    if preselected_hidden:
        assert select_calls == []
        assert full_selected_calls == []
        assert full_from_hidden_calls == [1]
    else:
        assert select_calls == [1, 2]
        assert full_selected_calls == [1]
        assert full_from_hidden_calls == []
    assert dist_sample_calls == [(Mode.TOPK_TOPP, 0, 1), (Mode.MIXED, 1, 3)]
    assert full_sample_calls == [(3, 4)]
    assert out.sampled_token_ids == [[7], [7], [7], [11]]


def test_log_step_path_skips_synthetic_warmup_and_logs_once():
    with _capture_vllm_logs(logging.INFO) as records:
        distributed_sampling.log_step_path(Mode.TOPK_TOPP, ["__warmup_0__", None])
        assert records == []
        for mode in (Mode.TOPK_TOPP, Mode.TOPK_TOPP, Mode.PURE_TEMP, Mode.MIXED):
            distributed_sampling.log_step_path(mode, ["__warmup_0__", "req-1"])
    assert [r.getMessage() for r in records] == [
        f"{VOCAB_PARALLEL_SAMPLING_KEY}: step took mode=topk_topp",
        f"{VOCAB_PARALLEL_SAMPLING_KEY}: step took mode=pure_temp",
        f"{VOCAB_PARALLEL_SAMPLING_KEY}: step took mode=mixed",
    ]


def test_precompile_distributed_sampling_warms_all_subgraphs(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    shard = _shards(torch.zeros(512, 16), 2, min_shard_rows=256)[0]
    owner = distributed_sampling.resolve_head_owner(
        _Owner(shard, 512), dp_lockstep=False
    )
    select_grad_modes: list[bool] = []
    seen_u: list[torch.Tensor] = []
    seen_temp_dtypes: list[torch.dtype] = []
    mixed_calls: list[int] = []

    def _rec_select(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        select_grad_modes.append(torch.is_grad_enabled())
        return h.index_select(0, idx)

    def _rec_sample(temps: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        seen_temp_dtypes.append(temps.dtype)
        seen_u.append(u)
        return torch.zeros((u.shape[0], 1), dtype=torch.int64)

    @contextlib.contextmanager
    def _timed(_name: str):
        with torch.no_grad():
            yield

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
        _precompile_timed=_timed,
        select_sample_hidden_states=_rec_select,
        sample_distributed_topk_topp=lambda _h, t, _k, _p, u: _rec_sample(t, u),
        sample_distributed_pure_temp=lambda _h, t, u: _rec_sample(t, u),
        select_mixed_distributed_tokens=lambda tk, a, b: (
            mixed_calls.append(tk.shape[0]) or torch.where(tk > 0, a, b)
        ),
    )
    TPUModelRunner._precompile_distributed_sampling(runner)
    assert select_grad_modes and not any(select_grad_modes)
    assert seen_temp_dtypes == [torch.float32] * 4
    assert len(seen_u) == 4
    assert len({id(t) for t in seen_u}) == 4
    assert mixed_calls == [4, 8]


@pytest.mark.parametrize("tp_size", [2, 8])
def test_spmd_compile_graph_symmetry_across_tp_ranks(monkeypatch, tp_size):
    """Compiled graphs must be rank-symmetric across TP=2 and TP=8 (no rank ints)."""
    monkeypatch.setattr(
        distributed_sampling_ops,
        "tensor_model_parallel_all_gather",
        lambda t, dim=-1: torch.cat([t] * tp_size, dim=dim),
    )
    vocab, hidden = (8501 if tp_size == 2 else 33501), 16
    shards = _shards(
        torch.zeros((vocab, hidden), dtype=torch.bfloat16),
        tp_size,
        min_shard_rows=4205,
    )
    owners = [
        distributed_sampling.resolve_head_owner(
            _Owner(shards[r], vocab, scale=0.9, soft_cap=30.0), dp_lockstep=False
        )
        for r in range(tp_size)
    ]

    for r in range(tp_size):
        assert owners[r] is not None
        assert isinstance(owners[r].lm_head.org_vocab_start_index, torch.Tensor)
        assert isinstance(owners[r].lm_head.num_org_vocab_padding, torch.Tensor)
        assert not hasattr(owners[r].lm_head, "_tpu_tp_rank")

    h = torch.zeros((4, hidden), dtype=torch.bfloat16)
    temps = torch.ones((4, 1), dtype=torch.float32)
    top_k = torch.full((4, 1), 16, dtype=torch.int32)
    top_p = torch.full((4, 1), 0.9, dtype=torch.float32)
    assert owners[0] is not None
    _, v_loc = distributed_sampling.head_tp_and_shard_vocab(owners[0])
    u_tktp = torch.full(
        (4, distributed_sampling_ops.rng_width(Mode.TOPK_TOPP, tp_size, v_loc)), 0.5
    )
    u_temp = torch.full(
        (4, distributed_sampling_ops.rng_width(Mode.PURE_TEMP, tp_size, v_loc)), 0.5
    )

    for fn, args in (
        (distributed_sampling_ops.sample_topk_topp, (h, temps, top_k, top_p, u_tktp)),
        (distributed_sampling_ops.sample_pure_temp, (h, temps, u_temp)),
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
