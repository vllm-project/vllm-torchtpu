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
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm import logger as vllm_logger
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams

from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    _float32_bsearch,
    _prob_bsearch,
    apply_top_k_top_p,
    safe_logsumexp,
)
from vllm_torchtpu.runner import distributed_sampling, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _HeadShard(torch.nn.Module):
    def __init__(self, weight, start, tp_size, num_pad, num_added=0):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            num_org_vocab_padding=num_pad,
            num_added_elements=num_added,
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


class _Owner(torch.nn.Module):
    def __init__(
        self, head, vocab_size, scale=1.0, soft_cap=None, logits_as_input=False
    ):
        super().__init__()
        self.lm_head = head
        with patch.object(
            lp_module,
            "get_current_vllm_config",
            return_value=SimpleNamespace(model_config=None),
        ):
            self.logits_processor = LogitsProcessor(
                vocab_size,
                scale=scale,
                logits_as_input=logits_as_input,
                soft_cap=soft_cap,
            )


def _shards(weight, tp_size, min_shard_rows=None):
    vocab, hidden = weight.shape
    per_rank = math.ceil(vocab / tp_size)
    floor_rows = (
        distributed_sampling.LOCAL_TOPK_CANDIDATES
        if min_shard_rows is None
        else min_shard_rows
    )
    per_rank = max(per_rank, floor_rows)
    pad_rows = torch.full((per_rank * tp_size - vocab, hidden), 1e4, dtype=weight.dtype)
    padded = torch.cat([weight, pad_rows])
    shards = []
    for rank in range(tp_size):
        start = rank * per_rank
        num_pad = max(0, start + per_rank - vocab) if start < vocab else per_rank
        shards.append(
            _HeadShard(padded[start : start + per_rank], start, tp_size, num_pad)
        )
    return shards


def _run_across_ranks(fn_per_rank, tp_size, monkeypatch):
    """Execute a distributed sampling function in lockstep across TP ranks."""
    if tp_size == 1:
        monkeypatch.setattr(
            distributed_sampling,
            "tensor_model_parallel_all_gather",
            lambda t, dim=-1: t,
        )
        return [fn_per_rank(0)]

    tls = threading.local()
    barrier = threading.Barrier(tp_size)
    slots: list[torch.Tensor | None] = [None] * tp_size
    outputs: list[torch.Tensor | None] = [None] * tp_size
    errors: list[BaseException] = []

    monkeypatch.setattr(
        distributed_sampling,
        "get_tensor_model_parallel_rank",
        lambda: tls.rank,
    )

    def lockstep_gather(tensor, dim=-1):
        slots[tls.rank] = tensor
        barrier.wait()
        gathered = torch.cat([t for t in slots if t is not None], dim=dim)
        barrier.wait()
        return gathered

    monkeypatch.setattr(
        distributed_sampling, "tensor_model_parallel_all_gather", lockstep_gather
    )

    def worker(r: int) -> None:
        tls.rank = r
        try:
            outputs[r] = fn_per_rank(r)
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
    return outputs


def _make_batch(*params, max_num_logprobs=None, vocab_size=100000):
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


def test_prob_bsearch_matches_float32_bsearch_bitwise():
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(24, 512, generator=gen, dtype=torch.float32) * 3.0
    logits[0, :64] = 5.0
    probs = logits.softmax(dim=-1)
    ps = torch.linspace(0.05, 0.999, 24, dtype=torch.float32).unsqueeze(1)

    def pred(th):
        return torch.where(probs > th, probs, 0.0).sum(dim=-1, keepdim=True) < ps

    th_orig = _float32_bsearch((24, 1), pred, probs.device)
    th_prob = _prob_bsearch((24, 1), pred, probs.device)
    assert torch.equal(th_orig, th_prob)


def test_safe_logsumexp_handles_all_neg_inf_without_nan():
    x = torch.tensor([[-float("inf"), -float("inf")], [1.0, 2.0]], dtype=torch.float32)
    out = safe_logsumexp(x, dim=-1)
    assert not torch.isnan(out).any()
    assert out.tolist()[0] == -float("inf")
    assert torch.allclose(out[1], torch.logsumexp(x[1], dim=-1))


def test_softcap_precedes_scale_matching_upstream_logits_processor():
    """LogitsProcessor.forward applies soft_cap first and scale second."""
    owner = _Owner(_shards(torch.zeros(8, 2), 1)[0], 8, scale=0.5, soft_cap=10.0)
    x = torch.tensor([[20.0, -5.0]], dtype=torch.float32)
    got = distributed_sampling._apply_softcap_and_scale(x, owner.logits_processor)
    expected = torch.tanh(x / 10.0) * 10.0 * 0.5
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("soft_cap", [None, 30.0])
def test_distributed_sampling_modes_lockstep_padding_and_ties(
    monkeypatch, tp_size, soft_cap
):
    gen = torch.Generator().manual_seed(42 + tp_size)
    # Use a vocab with non-multiple shard padding and strided multi-block extraction.
    vocab, hidden = 2053 if tp_size <= 2 else 9205, 16
    weight = (torch.randn(vocab, hidden, generator=gen) * 0.5).to(torch.bfloat16)
    # Row 0 is greedy with an exact tie across shards: lowest token ID 5 wins.
    weight[:] = weight.clone()
    weight[:, 0] = 0.0
    weight[5, 0] = 4.0
    weight[vocab - 7, 0] = 4.0
    shards = _shards(weight, tp_size, min_shard_rows=1792 if tp_size <= 2 else 1280)

    owners = []
    for r in range(tp_size):
        monkeypatch.setattr(
            distributed_sampling, "get_tensor_model_parallel_rank", lambda r=r: r
        )
        owners.append(
            distributed_sampling.resolve_head_owner(
                _Owner(shards[r], vocab, scale=0.8, soft_cap=soft_cap),
                dp_lockstep=False,
            )
        )

    hidden_states = torch.randn(4, hidden, generator=gen).to(torch.bfloat16)
    hidden_states[0] = 0.0
    hidden_states[0, 0] = 1.0

    temps = torch.tensor([[0.0], [1.0], [0.7], [1.2]], dtype=torch.bfloat16)
    top_k = torch.tensor([[0], [64], [16], [1]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [0.95], [0.85], [1.0]], dtype=torch.float32)

    u_tktp = torch.rand(
        (
            4,
            distributed_sampling.topk_topp_rng_width(
                tp_size, shards[0].weight.shape[0]
            ),
        ),
        generator=gen,
        dtype=torch.float32,
    )
    outs_tktp = _run_across_ranks(
        lambda r: distributed_sampling.distributed_topk_topp_sample(
            owners[r],
            hidden_states,
            temps,
            top_k,
            top_p,
            u_tktp,
        ),
        tp_size,
        monkeypatch,
    )
    expected_greedy = torch.tensor(5, dtype=torch.int64)
    for out in outs_tktp:
        assert torch.equal(out, outs_tktp[0])
        assert torch.equal(out[0, 0], expected_greedy)
        assert torch.equal(out < vocab, torch.ones_like(out, dtype=torch.bool))

    u_temp = torch.rand(
        (
            4,
            distributed_sampling.pure_temp_rng_width(
                shards[0].weight.shape[0], tp_size
            ),
        ),
        generator=gen,
        dtype=torch.float32,
    )
    outs_temp = _run_across_ranks(
        lambda r: distributed_sampling.distributed_pure_temp_sample(
            owners[r],
            hidden_states,
            temps,
            u_temp,
        ),
        tp_size,
        monkeypatch,
    )
    for out in outs_temp:
        assert torch.equal(out, outs_temp[0])
        assert torch.equal(out[0, 0], expected_greedy)
        assert torch.equal(out < vocab, torch.ones_like(out, dtype=torch.bool))


@pytest.mark.parametrize(
    "mode, top_k_val, top_p_val, max_kl, max_tvd",
    [
        ("topk_topp", 32, 0.90, 0.005, 0.020),
        ("pure_temp", -1, 1.0, 0.060, 0.075),
    ],
)
def test_statistical_kl_and_support_parity(
    monkeypatch, mode, top_k_val, top_p_val, max_kl, max_tvd
):
    gen = torch.Generator().manual_seed(123)
    vocab, hidden, tp_size = 4096, 16, 2
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
    owners = []
    for r in range(tp_size):
        monkeypatch.setattr(
            distributed_sampling, "get_tensor_model_parallel_rank", lambda r=r: r
        )
        owners.append(
            distributed_sampling.resolve_head_owner(
                _Owner(shards[r], vocab, scale=0.85, soft_cap=30.0),
                dp_lockstep=False,
            )
        )
    n_draws = 16384
    batch = 256
    h_rep = h.expand(batch, hidden).contiguous()
    t_rep = torch.ones((batch, 1), dtype=torch.float32)
    k_rep = torch.full((batch, 1), max(top_k_val, 0), dtype=torch.int32)
    p_rep = torch.full((batch, 1), top_p_val, dtype=torch.float32)
    counts = torch.zeros(vocab, dtype=torch.float32)

    u_width = (
        distributed_sampling.topk_topp_rng_width(tp_size, shards[0].weight.shape[0])
        if mode == "topk_topp"
        else distributed_sampling.pure_temp_rng_width(
            shards[0].weight.shape[0], tp_size
        )
    )
    for _ in range(n_draws // batch):
        u = torch.rand((batch, u_width), generator=gen, dtype=torch.float32)
        if mode == "topk_topp":
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling.distributed_topk_topp_sample(
                    owners[r],
                    h_rep,
                    t_rep,
                    k_rep,
                    p_rep,
                    u_step,
                ),
                tp_size,
                monkeypatch,
            )
        else:
            outs = _run_across_ranks(
                lambda r, u_step=u: distributed_sampling.distributed_pure_temp_sample(
                    owners[r],
                    h_rep,
                    t_rep,
                    u_step,
                ),
                tp_size,
                monkeypatch,
            )
        assert torch.equal(outs[0], outs[1])
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


def test_resolve_head_owner_rejects_unsupported_processors():
    shard = _shards(torch.zeros(8, 2), 1)[0]
    with pytest.raises(NotImplementedError, match="logits_as_input=True"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 8, logits_as_input=True), dp_lockstep=False
        )
    with pytest.raises(NotImplementedError, match="non-positive soft_cap"):
        distributed_sampling.resolve_head_owner(
            _Owner(shard, 8, soft_cap=0.0), dp_lockstep=False
        )

    dp_shard = _HeadShard(torch.zeros(8, 2), 0, 1, 0)
    setattr(dp_shard, dp_vocab_parallel.DP_VOCAB_ATTR, True)
    with pytest.raises(NotImplementedError, match="TPU_DP_VOCAB_PARALLEL"):
        distributed_sampling.resolve_head_owner(_Owner(dp_shard, 8), dp_lockstep=False)

    added_shard = _HeadShard(torch.zeros(8, 2), 0, 1, 0, num_added=4)
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        distributed_sampling.resolve_head_owner(
            _Owner(added_shard, 8), dp_lockstep=False
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

    # Greedy + top_k=64, top_p=0.95 -> DistributedSamplingMode.TOPK_TOPP
    ib_k, reqs_k = _make_batch(
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
    )
    assert distributed_sampling.step_eligible(ib_k, reqs_k)
    assert (
        distributed_sampling.select_chunk_mode(ib_k, 0, 2, 100000)
        == distributed_sampling.DistributedSamplingMode.TOPK_TOPP
    )

    # Pure temperature (top_k=-1, top_p=1.0) + greedy -> PURE_TEMP
    ib_t, _ = _make_batch(
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=-1, top_p=1.0),
    )
    assert (
        distributed_sampling.select_chunk_mode(ib_t, 0, 2, 100000)
        == distributed_sampling.DistributedSamplingMode.PURE_TEMP
    )

    # Pure top_p-only (top_k=-1, top_p=0.95) -> None (full-logits fallback)
    ib_p, reqs_p = _make_batch(SamplingParams(temperature=0.8, top_k=-1, top_p=0.95))
    assert distributed_sampling.step_eligible(ib_p, reqs_p)
    assert distributed_sampling.select_chunk_mode(ib_p, 0, 1, 100000) is None
    assert not distributed_sampling.step_has_eligible_request(ib_p, 100000)

    # Mixed batches containing a top_p-only or wide-top_k request -> None on mixed slice
    ib_mix_kp, _ = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    assert distributed_sampling.select_chunk_mode(ib_mix_kp, 0, 2, 100000) is None
    # Chunk 0 alone is eligible, chunk 1 alone falls back (multi-chunk step support)
    assert (
        distributed_sampling.select_chunk_mode(ib_mix_kp, 0, 1, 100000)
        == distributed_sampling.DistributedSamplingMode.TOPK_TOPP
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


def test_runner_distributed_sampling_step_and_chunk_fallback():
    """Verify runner skips deferred LM head on single-chunk top_p-only steps
    and cleanly falls back per-chunk on multi-chunk mixed steps."""
    runner = object.__new__(TPUModelRunner)
    runner._dist_sampling_head_owner = object()
    runner.vocab_size = 100000
    runner._min_tokens = SimpleNamespace(any_in=lambda _idx: False)
    sched = SimpleNamespace(scheduled_spec_decode_tokens={})

    # 1. Single-chunk top_p-only step: _distributed_sampling_this_step is False
    # so execute_model uses main's compute_selected_logits directly.
    ib_topp, reqs_topp = _make_batch(
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95)
    )
    runner.input_batch = ib_topp
    runner.requests = reqs_topp
    assert not runner._distributed_sampling_this_step(sched, target_num_chunks=1)
    assert not runner._distributed_sampling_this_step(sched, target_num_chunks=2)

    # 2. Multi-chunk mixed step (chunk 0: top_k=64, chunk 1: top_p=0.95):
    # 1-chunk step returns False; 2-chunk step returns True and falls back on chunk 1.
    ib_mix, reqs_mix = _make_batch(
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.8, top_k=-1, top_p=0.95),
    )
    runner.input_batch = ib_mix
    runner.requests = reqs_mix
    assert not runner._distributed_sampling_this_step(sched, target_num_chunks=1)
    assert runner._distributed_sampling_this_step(sched, target_num_chunks=2)

    called_fallback = []
    runner.compute_logits_from_hidden_states = lambda h: torch.ones(
        (h.shape[0], 16), dtype=h.dtype
    )
    runner._build_padded_sampling_params = lambda s, e, t: (
        torch.ones((t.shape[0], 1), dtype=t.dtype),
        torch.zeros((t.shape[0], 1), dtype=torch.int32),
        torch.full((t.shape[0], 1), 0.95, dtype=torch.float32),
    )
    runner._sampler_topk_candidates = lambda _s, _e, _g: 0
    runner._sampler_applies_topk_topp = lambda: True

    def fake_sample_from_logits(logits, *args, **kwargs):
        called_fallback.append(tuple(logits.shape))
        return torch.full((logits.shape[0], 1), 7, dtype=torch.int64)

    runner.sample_from_logits_func = fake_sample_from_logits
    out = runner._sample_distributed_or_fallback(
        torch.zeros((4, 8), dtype=torch.bfloat16),
        cur_start_idx=1,
        cur_end_idx=2,
        all_greedy=False,
        grammar_output=None,
        sampling_generator=None,
    )
    assert called_fallback == [(4, 16)]
    assert torch.equal(out, torch.full((4, 1), 7, dtype=torch.int64))


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
    mode_cls = distributed_sampling.DistributedSamplingMode
    try:
        distributed_sampling.log_step_path(mode_cls.TOPK_TOPP, ["__warmup_0__", None])
        assert records == []
        for mode in (mode_cls.TOPK_TOPP, mode_cls.TOPK_TOPP, mode_cls.PURE_TEMP):
            distributed_sampling.log_step_path(mode, ["__warmup_0__", "req-1"])
    finally:
        vllm_log.removeHandler(handler)
        vllm_log.setLevel(old_level)
        logging.disable(old_disable)
    assert [r.getMessage() for r in records] == [
        "TPU_DISTRIBUTED_SAMPLING: non-speculative step took mode=topk_topp",
        "TPU_DISTRIBUTED_SAMPLING: non-speculative step took mode=pure_temp",
    ]


@pytest.mark.parametrize(
    "accel, env_val, expected",
    [
        ("v7x-8", None, True),
        ("v6e-8", None, False),
        ("v6e-8", "1", True),
        ("v7x-8", "0", False),
    ],
)
def test_distributed_sampling_enabled_by_chip_and_env(
    monkeypatch, accel, env_val, expected
):
    from vllm_torchtpu import tpu_info

    tpu_info.get_chip_version.cache_clear()
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", accel)
    if env_val is None:
        monkeypatch.delenv("TPU_DISTRIBUTED_SAMPLING", raising=False)
    else:
        monkeypatch.setenv("TPU_DISTRIBUTED_SAMPLING", env_val)
    try:
        runner = object.__new__(TPUModelRunner)
        assert runner._distributed_sampling_enabled() is expected
    finally:
        tpu_info.get_chip_version.cache_clear()


def test_distributed_sampling_auto_fallback_vs_explicit_raise(monkeypatch):
    from vllm_torchtpu import tpu_info

    tpu_info.get_chip_version.cache_clear()
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v7x-8")
    runner = object.__new__(TPUModelRunner)
    runner._dist_sampling_head_owner = None
    runner._pp_is_last = True
    runner.is_pooling_model = False
    runner.model = torch.nn.Linear(2, 2)
    runner._dp_lockstep_enabled = lambda: False

    try:
        monkeypatch.delenv("TPU_DISTRIBUTED_SAMPLING", raising=False)
        runner._resolve_distributed_sampling_owner()
        assert runner._dist_sampling_head_owner is None

        monkeypatch.setenv("TPU_DISTRIBUTED_SAMPLING", "1")
        with pytest.raises(NotImplementedError, match="no module with an lm_head"):
            runner._resolve_distributed_sampling_owner()
    finally:
        tpu_info.get_chip_version.cache_clear()


def test_precompile_distributed_sampling_uses_fresh_u_per_mode(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(
        tpu_runner,
        "_get_sequence_layout_planner_for_runner",
        lambda _r: SimpleNamespace(uses_selected_logits_hidden_states=True),
    )
    seen_u: list[torch.Tensor] = []

    def _rec(u: torch.Tensor) -> torch.Tensor:
        seen_u.append(u)
        return torch.zeros((u.shape[0], 1), dtype=torch.int32)

    runner = SimpleNamespace(
        _dist_sampling_head_owner=object(),
        model_config=SimpleNamespace(get_hidden_size=lambda: 16),
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.bfloat16,
        num_reqs_paddings=[4, 8],
        _dist_sampling_tp_and_shard_vocab=lambda: (2, 256),
        _precompile_timed=lambda _name: contextlib.nullcontext(),
        sample_distributed_topk_topp=lambda _h, _t, _k, _p, u: _rec(u),
        sample_distributed_pure_temp=lambda _h, _t, u: _rec(u),
    )
    TPUModelRunner._precompile_distributed_sampling(runner)
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
    owners = []
    for r in range(tp_size):
        monkeypatch.setattr(
            distributed_sampling, "get_tensor_model_parallel_rank", lambda r=r: r
        )
        owners.append(
            distributed_sampling.resolve_head_owner(
                _Owner(shards[r], vocab, scale=0.9, soft_cap=30.0), dp_lockstep=False
            )
        )

    for r in range(tp_size):
        assert isinstance(owners[r].lm_head.org_vocab_start_index, torch.Tensor)
        assert isinstance(owners[r].lm_head.num_org_vocab_padding, torch.Tensor)
        assert isinstance(owners[r].lm_head._tpu_tp_rank, torch.Tensor)
        assert owners[r].lm_head._tpu_tp_rank.tolist() == r

    h = torch.zeros((4, hidden), dtype=torch.bfloat16)
    temps = torch.ones((4, 1), dtype=torch.bfloat16)
    top_k = torch.full((4, 1), 16, dtype=torch.int32)
    top_p = torch.full((4, 1), 0.9, dtype=torch.float32)
    v_loc = shards[0].weight.shape[0]
    u_tktp = torch.full(
        (4, distributed_sampling.topk_topp_rng_width(tp_size, v_loc)), 0.5
    )
    u_temp = torch.full(
        (4, distributed_sampling.pure_temp_rng_width(v_loc, tp_size)), 0.5
    )

    for fn, args in (
        (
            distributed_sampling.distributed_topk_topp_sample,
            (h, temps, top_k, top_p, u_tktp),
        ),
        (distributed_sampling.distributed_pure_temp_sample, (h, temps, u_temp)),
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
