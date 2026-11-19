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
"""TPU_SPEC_VERIFY_COARSE_PAD: verify rows padded to a coarse ladder give
the tokens, accepted counts and random draws of the token-bucket padding,
and the precompile sweeps cover every shape a step can reach with fewer
programs."""

import contextlib
import functools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
    coarse_pad_enabled,
    coarse_verify_paddings,
    verify_logits_paddings,
)
from vllm_torchtpu.runner.tpu_runner import (
    MIN_NUM_SEQS,
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len,
    _get_req_paddings,
)

K = 7
SAMPLING_EPS = 1e-5
ENV = "TPU_SPEC_VERIFY_COARSE_PAD"
# VPAD_TEST_DEVICE=tpu runs the exactness cases through the compiled TPU
# programs instead of eager CPU.
DEVICE = os.environ.get("VPAD_TEST_DEVICE", "cpu")
if DEVICE != "cpu":
    # One compiled program per verify shape, as the runner allows (tpu_runner init).
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 1024
# The served deployments' (max_num_batched_tokens, max_num_seqs,
# TPU_TOKEN_BUCKET_EXTRA) of the MiniMax-M3.1 unified launcher
# (launch_unified.sh; MNBT 8192 at every conc).
CONFIGS = {
    "c4": (8192, 4, [4, 8, 24, 48, 96, 384, 1536, 2560, 3072, 5120, 6144, 7168]),
    "c16": (
        8192,
        16,
        [4, 8, 24, 40, 48, 80, 96, 384, 1536, 2560, 3072, 5120, 6144, 7168],
    ),
    "c48": (
        8192,
        48,
        [48, 96, 160, 192, 224, 288, 320, 352, 384, 3072, 5120, 6144, 7168],
    ),
    "c64": (8192, 64, [48, 96, 384, 448, 480, 3072, 5120, 6144, 7168, 7680]),
}


def _reference_sample_from_logits(
    logits,
    temperatures,
    u,
    top_k,
    top_p,
    all_greedy=False,
    topk_candidates=0,
    apply_topk_topp=True,
    apply_top_k=True,
):
    """An eager Gumbel-max sampler with TPUModelRunner.sample_from_logits'
    signature (the exactness checks compare the switch off vs on, so any
    deterministic function of the uniforms does)."""
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled_logits = logits / safe_temperatures
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(
            scaled_logits, top_k, top_p, topk_candidates, apply_top_k
        )
    else:
        masked_logits = scaled_logits
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy_logits = masked_logits + (-torch.log(-torch.log(u_clamped)))
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)


def _spec_config(synthetic: bool):
    if not synthetic:
        return None
    return SimpleNamespace(
        rejection_sample_method="synthetic",
        # Mean acceptance length ~4 of 7, so both bonus and recovered tokens
        # occur.
        synthetic_acceptance_rates=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    )


def _shape_runner(conc: str, coarse: bool) -> SimpleNamespace:
    mnbt, seqs, extra = CONFIGS[conc]
    max_num_reqs = max(seqs, MIN_NUM_SEQS)
    return SimpleNamespace(
        num_tokens_paddings=_get_token_paddings(16, mnbt, extra),
        num_reqs_paddings=_get_req_paddings(MIN_NUM_SEQS, max_num_reqs),
        max_num_reqs=max_num_reqs,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _spec_verify_coarse_pad=coarse,
    )


def test_env_parses_and_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert coarse_pad_enabled() is False
    monkeypatch.setenv(ENV, "1")
    assert coarse_pad_enabled() is True
    monkeypatch.setenv(ENV, "yes please")
    with pytest.raises(ValueError):
        coarse_pad_enabled()


def test_ladder():
    c48 = _shape_runner("c48", True)
    ladder = verify_logits_paddings(c48)
    assert ladder[:6] == [8, 16, 32, 64, 128, 256]
    # Top of the powers of two: the token bucket of a full batch of windows
    # (48 * 8 = 384), then the token buckets above it.
    assert ladder[6] == 384
    assert ladder[7:] == [p for p in c48.num_tokens_paddings if p > 384]
    c64 = _shape_runner("c64", True)
    assert verify_logits_paddings(c64)[:7] == [8, 16, 32, 64, 128, 256, 512]
    assert verify_logits_paddings(_shape_runner("c4", True))[:3] == [8, 16, 32]
    # Off: the token buckets.
    off = _shape_runner("c48", False)
    assert verify_logits_paddings(off) == off.num_tokens_paddings
    assert coarse_verify_paddings([16, 32, 96, 128], 80) == [8, 16, 32, 64, 96, 128]


# --- exactness --------------------------------------------------------------


class _Step:
    """One verify chunk laid out by the real get_spec_decode_metadata, with
    the switch off and on, plus the logits table of its scheduled tokens."""

    def __init__(self, conc, num_rows, num_reqs, seed, vocab):
        rng = np.random.default_rng(seed)
        self.conc = conc
        # num_rows verify rows over num_reqs requests: drafts <= K, at
        # least one request with drafts; draft-less requests may be prefill
        # chunks (extra scheduled tokens, one verify row).
        drafts = np.zeros(num_reqs, np.int32)
        extra = num_rows - num_reqs
        assert 0 < extra <= num_reqs * K
        while extra:
            i = int(rng.integers(num_reqs))
            if drafts[i] < K:
                drafts[i] += 1
                extra -= 1
        prefill = np.where(drafts == 0, rng.integers(0, 40, size=num_reqs), 0).astype(
            np.int32
        )
        scheduled = drafts + 1 + prefill
        cu = np.cumsum(scheduled).astype(np.int32)
        total = int(cu[-1])
        self.num_reqs = num_reqs
        self.padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, max(CONFIGS[conc][1], MIN_NUM_SEQS)
        )
        g = torch.Generator().manual_seed(seed)
        self.table = (torch.randn((total, vocab), generator=g) * 3).to(torch.bfloat16)
        self.table_dev = self.table.to(DEVICE)
        argmax = self.table.float().argmax(-1)
        # Drafts that often match the target's argmax, so real acceptance
        # accepts some.
        ids = torch.randint(0, vocab, (total + 1,), generator=g, dtype=torch.int32)
        row_starts = cu - scheduled + prefill
        for r in range(num_reqs):
            for j in range(int(drafts[r])):
                pos = int(row_starts[r]) + j
                if rng.random() < 0.7:
                    ids[pos + 1] = int(argmax[pos])
        self.mds = {}
        for coarse in (False, True):
            runner = _shape_runner(conc, coarse)
            runner.input_ids_cpu = ids
            runner.arange_np = np.arange(total + 8, dtype=np.int64)
            runner.device = DEVICE
            self.mds[coarse] = SpeculativeDecodingManager(
                runner
            ).get_spec_decode_metadata(drafts, cu, self.padded_num_reqs)
        temperature = rng.choice([0.0, 0.6, 1.0, 1.3], size=num_reqs)
        temperature[0] = 1.0
        top_p = rng.choice([0.9, 1.0], size=num_reqs)
        top_k = np.full(num_reqs, vocab)
        top_k[: min(2, num_reqs)] = 20
        self.input_batch = SimpleNamespace(
            temperature_cpu_tensor=torch.tensor(temperature, dtype=torch.float32),
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
        )


def _verify_runner(step, synthetic, vocab):
    runner = SimpleNamespace(
        _synthetic_rng_counter=None,
        _spec_glue_logged=set(),
        model_config=SimpleNamespace(seed=0),
        device=DEVICE,
        vocab_size=vocab,
        input_batch=step.input_batch,
        rejection_sampler=RejectionSampler(_spec_config(synthetic), DEVICE),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        sample_from_logits_func=_reference_sample_from_logits,
        _sampler_topk_candidates=lambda *a: 0,
        _sampler_applies_topk_topp=lambda: True,
        _sampler_applies_top_k=lambda: True,
        _note_sampler_variant=lambda *a: None,
    )
    for name in (
        "_build_padded_sampling_params",
        "_padded_sampling_params",
        "_greedy_sample",
        "_plan_synthetic_verify",
        "_sample_synthetic_verify",
        "_get_synthetic_rng_counter",
        "_log_spec_glue_once",
        "spec_gather_bonus_and_target_logits",
        "spec_bonus_and_target_logits",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


# (synthetic, in-graph rng, all_greedy, one-row)
PATHS = {
    "greedy": (False, False, True, False),
    "real_sampled": (False, False, False, False),
    "synthetic_greedy": (True, False, True, False),
    "synthetic_sampled": (True, False, False, False),
    "synthetic_one_row": (True, False, False, True),
    "synthetic_one_row_ingraph_rng": (True, True, False, True),
}


def _run(step, coarse, path, vocab, seed, monkeypatch):
    synthetic, ingraph, all_greedy, one_row = PATHS[path]
    monkeypatch.setenv("TPU_SPEC_SYNTHETIC_INGRAPH_RNG", "1" if ingraph else "0")
    md = step.mds[coarse]
    md.synthetic_verify = None
    md.synthetic_output = None
    md.synthetic_row_rank = None
    runner = _verify_runner(step, synthetic, vocab)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1000)
    runner._get_sampling_generator = lambda: gen
    stance = (
        torch.compiler.set_stance("force_eager")
        if DEVICE == "cpu"
        else contextlib.nullcontext()
    )
    with stance:
        if one_row:
            rows = runner._plan_synthetic_verify(md)
            logits = step.table_dev[rows]
        elif md.fine_logits_indices is not None:
            # As execute_model: the LM head at the token-bucket rows, padded.
            logits = tpu_runner._fit_verify_rows(
                step.table_dev[md.fine_logits_indices],
                md.final_logits_indices.shape[0],
            )
        else:
            # compute_selected_logits: the verify rows of the logits.
            logits = step.table_dev[md.final_logits_indices]
        out = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits,
            md,
            grammar_output=None,
            scheduler_output=SimpleNamespace(),
            cur_start_idx=0,
            cur_end_idx=step.num_reqs,
            all_greedy=all_greedy,
            sampling_generator=gen,
        )
    counter = runner._synthetic_rng_counter
    # The generator's position: its next draw.
    after = torch.rand(16, generator=gen, device=DEVICE)
    return out.cpu(), after.cpu(), None if counter is None else int(counter)


def _cases(conc):
    """(rows, requests) per token bucket up to a full batch: each bucket's
    smallest and largest row count, with the fewest and the most requests
    that can hold them."""
    off = _shape_runner(conc, False)
    pads = off.num_tokens_paddings
    most = off.max_num_reqs * (1 + K)
    cases = set()
    prev = 0
    for b in pads:
        if prev >= most:
            break
        for n in {prev + 1, min(b, most)}:
            lo = -(-n // (1 + K))
            hi = min(n - 1, off.max_num_reqs)
            if lo > hi:
                continue
            cases.update({(n, lo), (n, hi), (n, (lo + hi) // 2)})
        prev = b
    return sorted(cases)


@pytest.mark.parametrize("conc", sorted(CONFIGS))
def test_coarse_pad_is_exact(conc, monkeypatch):
    vocab = 512
    padded_further = 0
    cases = _cases(conc)[:: int(os.environ.get("VPAD_CASE_STRIDE", "1"))]
    for i, (num_rows, num_reqs) in enumerate(cases):
        step = _Step(conc, num_rows, num_reqs, seed=i, vocab=vocab)
        off, on = step.mds[False], step.mds[True]
        n_off = off.final_logits_indices.shape[0]
        n_on = on.final_logits_indices.shape[0]
        assert on.rng_rows == (n_off if n_on != n_off else None)
        padded_further += n_on > n_off
        for name in ("final_logits_indices", "target_logits_indices", "draft_token_ids",
                     "segment_ids", "group_indices"):  # fmt: skip
            a, b = getattr(off, name), getattr(on, name)
            assert torch.equal(a[:num_rows], b[:num_rows]), name
        # The LM-head indices of a full-logits chunk are the off path's.
        if on.rng_rows is None:
            assert on.fine_logits_indices is None
        else:
            assert torch.equal(on.fine_logits_indices, off.final_logits_indices)
        # Padded verify rows belong to no real request.
        assert bool((on.segment_ids[num_rows:] == num_reqs).all())
        for path in PATHS:
            if (
                DEVICE == "cpu"
                and path.endswith("_sampled")
                and num_reqs == step.padded_num_reqs
            ):
                # The unfused sampled verify gathers per-row temperatures with
                # the padding sentinel num_reqs, one past the table when no
                # request is padded: XLA clamps it on TPU, eager CPU raises
                # (with the switch off too, whenever a row is padded).
                continue
            want = _run(step, False, path, vocab, i, monkeypatch)
            got = _run(step, True, path, vocab, i, monkeypatch)
            assert torch.equal(got[0], want[0]), (conc, num_rows, num_reqs, path)
            assert torch.equal(got[1], want[1]), ("generator", path)
            assert got[2] == want[2], ("counter", path)
    # Rows past the token bucket of the off path, so the test cannot pass by
    # never padding further.
    assert padded_further >= len(cases) // 4, (padded_further, len(cases))


def test_metadata_cache_hit_keeps_rng_rows():
    step = _Step("c48", 130, 20, seed=3, vocab=64)
    runner = _shape_runner("c48", True)
    runner.input_ids_cpu = torch.zeros(4096, dtype=torch.int32)
    runner.arange_np = np.arange(4096, dtype=np.int64)
    runner.device = "cpu"
    manager = SpeculativeDecodingManager(runner)
    drafts = step.mds[True].draft_lengths_cpu
    cu = np.cumsum(drafts + 1).astype(np.int32)
    first = manager.get_spec_decode_metadata(drafts, cu, 32)
    again = manager.get_spec_decode_metadata(drafts, cu, 32)
    assert first.rng_rows == again.rng_rows == 160
    assert again.draft_token_ids.shape[0] == 256
    assert again.fine_logits_indices is first.fine_logits_indices
    assert torch.equal(first.fine_logits_indices, first.final_logits_indices[:160])


def test_fit_verify_rows_pads_and_cuts():
    t = torch.arange(12.0).view(3, 4)
    with torch.compiler.set_stance("force_eager"):
        padded = tpu_runner._fit_verify_rows(t, 5)
        cut = tpu_runner._fit_verify_rows(t, 2)
    assert torch.equal(padded[:3], t) and bool((padded[3:] == 0).all())
    assert torch.equal(cut, t[:2])


def test_uniform_draw_matches_off_path_stream():
    md = SimpleNamespace(rng_rows=160)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    u = tpu_runner._verify_rows_uniform(md, (256, 3), "cpu", g1)
    ref = torch.rand((160, 3), generator=g2)
    assert torch.equal(u[:160], ref) and bool((u[160:] == 0).all())
    assert torch.equal(g1.get_state(), g2.get_state())
    # A ladder entry below the token bucket cuts the draw.
    md = SimpleNamespace(rng_rows=16)
    u = tpu_runner._verify_rows_uniform(md, (8,), "cpu", g1)
    assert torch.equal(u, torch.rand((16,), generator=g2)[:8])
    # Off (rng_rows None): a plain draw.
    md = SimpleNamespace(rng_rows=None)
    u = tpu_runner._verify_rows_uniform(md, (8,), "cpu", g1)
    assert torch.equal(u, torch.rand((8,), generator=g2))


# --- precompile coverage and program counts ---------------------------------


class _Sampler:
    synthetic_mode = False

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def __call__(self, *, draft_token_ids, num_draft_tokens, **kwargs):
        self.calls.append((draft_token_ids.shape[0], num_draft_tokens.shape[0]))
        return draft_token_ids


def _sweep_runner(conc, coarse):
    runner = _shape_runner(conc, coarse)
    runner.__dict__.update(
        _is_async_drafter=True,
        rejection_sampler=_Sampler(),
        vocab_size=4,
        _hidden_states_dtype=torch.float32,
        device="cpu",
        spec_bonus_and_target_logits=lambda d, b, t: (b, d),
        spec_gather_bonus_and_target_logits=lambda d, b, t: (b, d),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _greedy_head_owner=None,
        _topk_candidate_blocks=lambda: 0,
        _precompile_synthetic_verify_heads=lambda pairs: None,
        sample_from_logits_func=lambda *a, **k: torch.zeros((a[0].shape[0], 1)),
    )
    for name in ("_spec_verify_bucket_reachable", "_build_padded_sampling_params",
                 "_padded_sampling_params", "_greedy_sample"):  # fmt: skip
        if hasattr(TPUModelRunner, name):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _rs_pairs(monkeypatch, conc, coarse):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
    monkeypatch.delenv("TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY", raising=False)
    monkeypatch.delenv("TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS", raising=False)
    runner = _sweep_runner(conc, coarse)
    with torch.compiler.set_stance("force_eager"):
        TPUModelRunner._precompile_rejection_sampler(runner)  # type: ignore[arg-type]
    return sorted(set(runner.rejection_sampler.calls)), runner


def _fits(runner):
    runner._verify_rows_fits = functools.partial(
        TPUModelRunner._verify_rows_fits, runner
    )
    return set(runner._verify_rows_fits())


@pytest.mark.parametrize("conc", sorted(CONFIGS))
def test_sweeps_cover_every_reachable_shape(monkeypatch, conc):
    """Every (padded_logits_length, padded_num_reqs) a step can produce is in
    the rejection-sampler sweep, and every (token bucket, coarse) row pair a
    full-logits chunk pads in the `_fit_verify_rows` warmup, switch on and
    off."""
    for coarse in (False, True):
        pairs, runner = _rs_pairs(monkeypatch, conc, coarse)
        fits = _fits(runner)
        vpads = verify_logits_paddings(runner)
        tpads = runner.num_tokens_paddings
        pair_set = set(pairs)
        for r in range(1, runner.max_num_reqs + 1):
            pnr = _get_padded_num_reqs_with_upper_limit(r, runner.max_num_reqs)
            for n in range(r + 1, r * (1 + K) + 1):
                length = _get_padded_token_len(vpads, n)
                assert (length, pnr) in pair_set, (coarse, n, r)
                fine = _get_padded_token_len(tpads, n)
                if length != fine:
                    assert (fine, length) in fits, (coarse, n)
        if not coarse:
            assert not fits


def test_program_counts(monkeypatch, capsys):
    """Prints the verify-program shape counts off vs on per deployment and
    checks that the switch never adds rejection-sampler shapes."""
    rows = []
    for conc in sorted(CONFIGS):
        counts = {}
        for coarse in (False, True):
            pairs, _ = _rs_pairs(monkeypatch, conc, coarse)
            counts[coarse] = (len(pairs), len({p[0] for p in pairs}))
        rows.append((conc, counts[False], counts[True]))
        for a, b in zip(counts[False], counts[True], strict=True):
            assert b <= a, (conc, counts)
    with capsys.disabled():
        print("\nconc  (rs_pairs, rs_lengths) off -> on")
        for conc, off, on in rows:
            print(f"{conc:4s}  {off} -> {on}")


def test_runner_reads_the_switch(monkeypatch):
    """`_init_speculative_decoding` sets `_spec_verify_coarse_pad` from the
    env (off without a speculative config)."""
    cases = (("1", True, True), ("0", True, False), ("1", False, False))
    for value, spec, want in cases:
        monkeypatch.setenv(ENV, value)
        assert envs.TPU_SPEC_VERIFY_COARSE_PAD is (value == "1")
        got = bool(
            (SimpleNamespace(num_speculative_tokens=K) if spec else None)
            and coarse_pad_enabled()
        )
        assert got is want
