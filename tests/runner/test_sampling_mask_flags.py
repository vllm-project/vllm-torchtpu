# SPDX-License-Identifier: Apache-2.0
"""The samplers leave out the top-k and top-p searches no row of a step uses.

The runner reads from the host copies of a chunk's sampling parameters whether
any row applies top-k and whether any applies top-p, and passes both to the
samplers as static arguments. A search no row uses is left out with the same
tokens; when any row applies a mask, the search runs.
"""

import contextlib
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample import top_k_top_p
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import MASK_FLAG_COMBINATIONS
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import PaddedSamplingParams, TPUModelRunner

VOCAB = 64
ROWS = 8


def _params_runner(top_k, top_p):
    n = len(top_k)
    runner = SimpleNamespace(
        vocab_size=VOCAB,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.ones(n),
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
        ),
    )
    runner.build = MethodType(TPUModelRunner._build_padded_sampling_params, runner)
    return runner


@pytest.mark.parametrize(
    "top_k, top_p, expected",
    [
        ([0, VOCAB, VOCAB + 7], [1.0, 1.0, 1.0], (False, False)),
        ([0, 3, VOCAB], [1.0, 1.0, 1.0], (True, False)),
        ([0, VOCAB, 0], [1.0, 0.9, 1.0], (False, True)),
        ([5, VOCAB, 0], [1.0, 1.0, 0.5], (True, True)),
    ],
)
def test_padded_params_say_which_masks_a_row_applies(top_k, top_p, expected):
    """top_k 0 or at least the vocab and top_p 1 apply no mask."""
    params = _params_runner(top_k, top_p).build(0, 3, torch.zeros((4, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == expected
    assert params.top_k[3].item() == 0 and params.top_p[3].item() == 1.0


def test_padded_params_read_only_the_chunk_rows():
    """A mask in another chunk's rows does not turn a search on."""
    runner = _params_runner([3, 0, VOCAB, 0], [0.5, 1.0, 1.0, 1.0])
    params = runner.build(1, 3, torch.zeros((2, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == (False, False)
    params = runner.build(0, 2, torch.zeros((2, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == (True, True)


def _sampler_inputs(seed):
    """Logits with grammar-masked entries, a greedy row, and drafts."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((ROWS, VOCAB), generator=g) * 3
    logits[1, ::5] = float("-inf")
    temperatures = torch.tensor([[1.0], [0.7], [0.0], [1.3]] * (ROWS // 4))
    u = torch.rand((ROWS, VOCAB), generator=g)
    noise = torch.randn((ROWS, VOCAB), generator=g)
    draft = torch.argmax(logits + noise, dim=-1).to(torch.int32)
    return logits, temperatures, u, draft


def _sample(logits, temperatures, u, top_k, top_p, use_top_k, use_top_p):
    runner = SimpleNamespace()
    runner._apply_temperature = MethodType(TPUModelRunner._apply_temperature, runner)
    return TPUModelRunner.sample_from_logits.__wrapped__(
        runner,
        logits,
        temperatures,
        u,
        top_k,
        top_p,
        all_greedy=False,
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )


@pytest.fixture
def eager_rejection(monkeypatch):
    """Runs the rejection sampler's program body eagerly on CPU."""
    monkeypatch.setattr(
        rs,
        "_random_rejection_sample_with_segment",
        rs._random_rejection_sample_with_segment.__wrapped__,
    )
    return rs.RejectionSampler()


def _reject(sampler, inputs, top_k, top_p, use_top_k, use_top_p):
    logits, temperatures, u, draft = inputs
    return sampler(
        draft_token_ids=draft,
        num_draft_tokens=torch.tensor([3, 3, 2], dtype=torch.int32),
        target_logits=logits,
        bonus_token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64),
        group_indices=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.int64),
        max_draft_tokens=3,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=u[:, 0].contiguous(),
        recover_u=u,
        do_sampling=True,
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )


@pytest.mark.parametrize("seed", range(4))
def test_sampling_without_masks_skips_searches_with_same_tokens(seed, eager_rejection):
    inputs = _sampler_inputs(seed)
    logits, temperatures, u, _ = inputs
    top_k = torch.zeros((ROWS, 1), dtype=torch.int32)
    top_p = torch.ones((ROWS, 1))
    full = _sample(logits, temperatures, u, top_k, top_p, True, True)
    skipped = _sample(logits, temperatures, u, top_k, top_p, False, False)
    assert torch.equal(full, skipped)

    full = _reject(eager_rejection, inputs, top_k, top_p, True, True)
    skipped = _reject(eager_rejection, inputs, top_k, top_p, False, False)
    assert torch.equal(full, skipped)


@pytest.fixture
def searches(monkeypatch):
    """Records which cutoff searches run: topk_mask and topp_mask."""
    ran = []
    for name in ("topk_mask", "topp_mask"):
        search = getattr(top_k_top_p, name)

        def spy(*args, _name=name, _search=search, **kwargs):
            ran.append(_name)
            return _search(*args, **kwargs)

        monkeypatch.setattr(top_k_top_p, name, spy)
    return ran


@pytest.mark.parametrize("use_top_k, use_top_p", MASK_FLAG_COMBINATIONS)
def test_every_sampler_runs_only_the_searches_its_flags_ask_for(
    use_top_k, use_top_p, searches, eager_rejection
):
    """Rows here set top-k and top-p, so a search that ran despite a False
    flag would show up; each entry point runs exactly the flagged ones."""
    inputs = _sampler_inputs(0)
    logits, temperatures, u, _ = inputs
    top_k = torch.full((ROWS, 1), 3, dtype=torch.int32)
    top_p = torch.full((ROWS, 1), 0.5)
    expected = ["topk_mask"] * use_top_k + ["topp_mask"] * use_top_p

    top_k_top_p.apply_top_k_top_p(
        logits, top_k, top_p, use_top_k=use_top_k, use_top_p=use_top_p
    )
    assert searches == expected
    searches.clear()
    _sample(logits, temperatures, u, top_k, top_p, use_top_k, use_top_p)
    assert searches == expected
    searches.clear()
    _reject(eager_rejection, inputs, top_k, top_p, use_top_k, use_top_p)
    assert searches == expected


@pytest.mark.parametrize(
    "mask_row_top_k, mask_row_top_p", [(1, 1.0), (0, 0.05), (2, 0.05)]
)
def test_a_row_with_a_mask_is_still_searched(mask_row_top_k, mask_row_top_p):
    """Row 0 applies a mask that removes the token its noise would pick."""
    logits = torch.linspace(4.0, 0.0, VOCAB).repeat(ROWS, 1)
    temperatures = torch.ones((ROWS, 1))
    u = torch.full((ROWS, VOCAB), 0.5)
    u[:, -1] = 1.0 - 1e-6
    top_k = torch.zeros((ROWS, 1), dtype=torch.int32)
    top_p = torch.ones((ROWS, 1))
    top_k[0], top_p[0] = mask_row_top_k, mask_row_top_p
    runner = _params_runner(top_k.view(-1).tolist(), top_p.view(-1).tolist())
    params = runner.build(0, ROWS, logits)
    assert (params.use_top_k, params.use_top_p) == (
        mask_row_top_k > 0,
        mask_row_top_p < 1.0,
    )

    flagged = _sample(
        logits, temperatures, u, top_k, top_p, params.use_top_k, params.use_top_p
    )
    full = _sample(logits, temperatures, u, top_k, top_p, True, True)
    unmasked = _sample(logits, temperatures, u, top_k, top_p, False, False)
    assert torch.equal(flagged, full)
    assert flagged[0].item() == 0
    assert unmasked[0].item() == VOCAB - 1
    assert torch.equal(flagged[1:], unmasked[1:])


class _RejectionSampler:
    """Stands in for RejectionSampler; non-greedy calls go to `fn`."""

    synthetic_mode = False

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, **kwargs):
        return self._fn(**kwargs)

    def greedy_from_token_ids(self, **kwargs):
        return torch.zeros((2, 2), dtype=torch.int32)


def _record(calls):
    def sample_from_logits(logits, temperatures, u, top_k, top_p, **kwargs):
        calls.append(("sample", kwargs))
        return torch.zeros((logits.shape[0], 1), dtype=torch.int32)

    return sample_from_logits


@pytest.mark.parametrize("use_top_k, use_top_p", MASK_FLAG_COMBINATIONS)
def test_verify_chunk_passes_the_flags_to_both_samplers(use_top_k, use_top_p):
    calls = []
    logits = torch.zeros((2, VOCAB))
    params = PaddedSamplingParams(
        temperatures=torch.ones((2, 1)),
        top_k=torch.zeros((2, 1), dtype=torch.int32),
        top_p=torch.ones((2, 1)),
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )

    def rejection_sampler(**kwargs):
        calls.append(("reject", kwargs))
        return torch.zeros((2, 2), dtype=torch.int32)

    runner = SimpleNamespace(
        spec_gather_bonus_and_target_logits=lambda logits, b, t: (logits, logits),
        _build_padded_sampling_params=lambda start, end, logits: params,
        sample_from_logits_func=_record(calls),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        rejection_sampler=_RejectionSampler(rejection_sampler),
    )
    md = SimpleNamespace(
        bonus_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        draft_token_ids=torch.zeros((2,), dtype=torch.int32),
        draft_lengths=torch.ones((2,), dtype=torch.int32),
        segment_ids=torch.tensor([0, 1], dtype=torch.int64),
        group_indices=torch.tensor([0, 0], dtype=torch.int64),
    )
    TPUModelRunner._sample_spec_verify_chunk(
        runner,
        logits,
        md,
        grammar_output=None,
        scheduler_output=SimpleNamespace(),
        cur_start_idx=0,
        cur_end_idx=2,
        all_greedy=False,
        sampling_generator=None,
    )
    assert [kind for kind, _ in calls] == ["sample", "reject"]
    for _, kwargs in calls:
        assert (kwargs["use_top_k"], kwargs["use_top_p"]) == (use_top_k, use_top_p)


def test_warmup_compiles_every_flag_combination(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **kw: None)
    calls = []
    runner = SimpleNamespace(
        num_reqs_paddings=[2, 4],
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        _dummy_logits=lambda n: torch.zeros((n, VOCAB)),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        sample_from_logits_func=_record(calls),
    )
    TPUModelRunner._precompile_sample_from_logits(runner)
    warmed = [(k["all_greedy"], k["use_top_k"], k["use_top_p"]) for _, k in calls]
    expected = [(True, True, True)] + [(False, *f) for f in MASK_FLAG_COMBINATIONS]
    assert warmed == expected * 2


def _signature(args, kwargs):
    """Dtype and shape of each tensor argument, and every other value."""

    def describe(value):
        if isinstance(value, torch.Tensor):
            return value.dtype, tuple(value.shape)
        return value

    return [describe(a) for a in args], {k: describe(v) for k, v in kwargs.items()}


def _flags_of(signature):
    return signature[1]["use_top_k"], signature[1]["use_top_p"]


def _verify_runner(top_k, top_p, samples, rejections):
    """A runner whose samplers and gathers run eagerly and record their inputs."""
    runner = _params_runner(top_k, top_p)
    runner._build_padded_sampling_params = runner.build
    runner._apply_temperature = MethodType(TPUModelRunner._apply_temperature, runner)
    sample = MethodType(TPUModelRunner.sample_from_logits.__wrapped__, runner)

    def sample_from_logits(*args, **kwargs):
        samples.append(_signature(args, kwargs))
        return sample(*args, **kwargs)

    def rejection_sampler(**kwargs):
        if kwargs.get("do_sampling"):
            rejections.append(_signature((), kwargs))
        return torch.zeros((2, 2), dtype=torch.int32)

    runner.sample_from_logits_func = sample_from_logits
    runner.rejection_sampler = _RejectionSampler(rejection_sampler)
    runner._greedy_sample = MethodType(TPUModelRunner._greedy_sample, runner)
    for name in (
        "spec_bonus_and_target_token_ids",
        "spec_gather_bonus_and_target_logits",
    ):
        method = getattr(TPUModelRunner, name).__wrapped__
        setattr(runner, name, MethodType(method, runner))
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    return runner


def test_rejection_warmup_matches_the_verify_step_inputs(monkeypatch):
    """Each non-greedy rejection program the precompile warms gets the dtypes,
    shapes and flags a verify step passes, so the step finds it compiled. The
    bonus tokens are sample_from_logits' int64, not the greedy prelude's int32.
    """
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **kw: None)
    rows, reqs, dtype = 4, 2, torch.bfloat16
    warm_samples, warm_rejections = [], []
    runner = _verify_runner([0, 0], [1.0, 1.0], warm_samples, warm_rejections)
    runner.__dict__.update(
        _is_async_drafter=True,
        num_tokens_paddings=[rows],
        num_reqs_paddings=[reqs],
        max_num_reqs=reqs,
        _fused_greedy_head=None,
        _hidden_states_dtype=dtype,
        device=torch.device("cpu"),
        _spec_verify_bucket_reachable=lambda tokens, n: True,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    TPUModelRunner._precompile_rejection_sampler(runner)
    assert [_flags_of(r) for r in warm_rejections] == list(MASK_FLAG_COMBINATIONS)
    bonus_samples = [s for s in warm_samples if not s[1]["all_greedy"]]
    assert [_flags_of(s) for s in bonus_samples] == list(MASK_FLAG_COMBINATIONS)

    md = SimpleNamespace(
        bonus_logits_indices=torch.zeros(reqs, dtype=torch.int32),
        target_logits_indices=torch.zeros(rows, dtype=torch.int32),
        draft_token_ids=torch.zeros(rows, dtype=torch.int32),
        draft_lengths=torch.zeros(reqs, dtype=torch.int32),
        segment_ids=torch.zeros(rows, dtype=torch.int64),
        group_indices=torch.zeros(rows, dtype=torch.int32),
    )
    for flags, warm_sample, warm_rejection in zip(
        MASK_FLAG_COMBINATIONS, bonus_samples, warm_rejections
    ):
        samples, rejections = [], []
        runner = _verify_runner(
            [3 if flags[0] else 0, 0],
            [0.5 if flags[1] else 1.0, 1.0],
            samples,
            rejections,
        )
        TPUModelRunner._sample_spec_verify_chunk(
            runner,
            torch.zeros((rows, VOCAB), dtype=dtype),
            md,
            grammar_output=None,
            scheduler_output=SimpleNamespace(),
            cur_start_idx=0,
            cur_end_idx=2,
            all_greedy=False,
            sampling_generator=None,
        )
        assert samples == [warm_sample]
        assert rejections == [warm_rejection]
        assert rejections[0][1]["bonus_token_ids"] == (torch.int64, (reqs,))


def _capture_warmup_requests(monkeypatch):
    """A runner whose warmup steps record their new requests' SamplingParams."""
    monkeypatch.setattr(tpu_runner, "_spec_warmup_group_num_blocks", lambda *a: 1)
    params = []

    def execute_model(scheduler_output):
        params.extend(r.sampling_params for r in scheduler_output.scheduled_new_reqs)

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        kv_cache_config=SimpleNamespace(num_blocks=64, kv_cache_groups=[None]),
        max_model_len=64,
        max_num_tokens=64,
        max_num_reqs=4,
        execute_model=execute_model,
        sample_tokens=lambda grammar_output: None,
        take_draft_token_ids=lambda: None,
        _warmup_spec_decode_cleanup=lambda rids: True,
    )
    return runner, params


def _flags_of_request(sampling_params):
    """The step flags a request with these SamplingParams gives its chunk."""
    runner = _params_runner([sampling_params.top_k], [sampling_params.top_p])
    params = runner.build(0, 1, torch.zeros((1, VOCAB)))
    return params.use_top_k, params.use_top_p


@pytest.mark.parametrize("mask_flags", MASK_FLAG_COMBINATIONS)
def test_spec_warmup_requests_set_the_masks_they_warm(monkeypatch, mask_flags):
    """A non-greedy warmup request runs the real verify step with the top-k and
    top-p flags it is asked to warm."""
    runner, params = _capture_warmup_requests(monkeypatch)
    assert TPUModelRunner._warmup_one_spec_request(
        runner, 8, idx=0, greedy=False, mask_flags=mask_flags
    )
    TPUModelRunner._warmup_spec_batch(
        runner, 2, 8, greedy=False, short=True, mask_flags=mask_flags
    )
    assert len(params) == 3
    for sampling_params in params:
        assert sampling_params.temperature == 1.0
        assert _flags_of_request(sampling_params) == mask_flags


def test_spec_warmup_runs_every_flag_combination(monkeypatch):
    """The real spec-decode warmup runs the greedy path and the non-greedy path
    once per flag combination, for single requests and for batches."""
    monkeypatch.setattr(tpu_runner.envs, "SPEC_WARMUP", True, raising=False)
    single, batched = [], []

    def warmup_one_spec_request(P, idx, quiet=False, greedy=True, mask_flags=None):
        single.append((P, greedy, mask_flags))
        return True

    def warmup_spec_batch(R, P, greedy=True, mask_flags=None):
        batched.append((R, greedy, mask_flags))

    runner = SimpleNamespace(
        enforce_eager=False,
        _is_async_drafter=True,
        input_batch=SimpleNamespace(num_reqs=0),
        num_xla_graphs=0,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        num_tokens_paddings=[8, 16],
        num_reqs_paddings=[1, 2, 4],
        max_model_len=64,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        drafter=SimpleNamespace(),
        _warmup_one_spec_request=warmup_one_spec_request,
        _warmup_spec_batch=warmup_spec_batch,
    )
    TPUModelRunner._warmup_spec_decode(runner)
    variants = [(True, (False, False))] + [(False, f) for f in MASK_FLAG_COMBINATIONS]
    assert single == [(P, *v) for P in (8, 16) for v in variants]
    assert batched == [(R, *v) for R in (2, 4) for v in variants]


class _Event:
    def record(self):
        pass

    def synchronize(self):
        pass


def _sample_tokens_runner(spec_decode_metadata_list, params_by_start, calls):
    """Two chunks of one request each, sampled at temperature 1.0."""
    req_ids = ["r0", "r1"]

    def rejection_sampler(**kwargs):
        calls.append(("reject", kwargs))
        return torch.zeros((2, 2), dtype=torch.int32)

    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 1, "r1": 1}, finished_req_ids=set()
            ),
            draft_chunks=[],
            logits_list=[torch.zeros((2, VOCAB)), torch.zeros((2, VOCAB))],
            num_reqs_list=[1, 1],
            spec_decode_metadata_list=spec_decode_metadata_list,
            mamba_state_indices_list=None,
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=2,
            req_ids=req_ids,
            req_id_to_index={"r0": 0, "r1": 1},
            all_greedy=False,
            num_tokens_no_spec=[1, 1],
            token_ids_cpu=np.zeros((2, 8), dtype=np.int32),
        ),
        requests={
            r: SimpleNamespace(num_computed_tokens=0, num_tokens=1, output_token_ids=[])
            for r in req_ids
        },
        speculative_config=SimpleNamespace(method="ngram", num_speculative_tokens=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _is_async_drafter=False,
        is_pooling_model=False,
        spec_decode_manager=SimpleNamespace(propose_draft_token_ids=lambda **kw: None),
        _get_sampling_generator=lambda: None,
        _build_padded_sampling_params=lambda start, end, logits: params_by_start[start],
        sample_from_logits_func=_record(calls),
        rejection_sampler=_RejectionSampler(rejection_sampler),
        spec_gather_bonus_and_target_logits=lambda logits, b, t: (logits, logits),
        _update_mamba_slot_read_offsets=lambda *args: None,
        _update_num_xla_graphs=lambda *args: None,
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in (
        "_sample_spec_verify_chunk",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
    ):
        setattr(runner, name, MethodType(getattr(TPUModelRunner, name), runner))
    return runner


@pytest.mark.parametrize("use_top_k, use_top_p", MASK_FLAG_COMBINATIONS)
@pytest.mark.parametrize("spec_step", [True, False], ids=["spec", "non_spec"])
def test_sample_tokens_passes_each_chunk_its_own_flags(
    monkeypatch, spec_step, use_top_k, use_top_p
):
    """A spec step with a verify chunk and a chunk without drafts, and a step
    without drafts: every sampler call gets the flags of its own chunk."""
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=_Event), raising=False)
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: False)
    first, second = (use_top_k, use_top_p), (not use_top_k, not use_top_p)
    params_by_start = {
        start: PaddedSamplingParams(
            temperatures=torch.ones((2, 1)),
            top_k=torch.zeros((2, 1), dtype=torch.int32),
            top_p=torch.ones((2, 1)),
            use_top_k=flags[0],
            use_top_p=flags[1],
        )
        for start, flags in ((0, first), (1, second))
    }
    md = SimpleNamespace(
        bonus_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        draft_token_ids=torch.zeros((2,), dtype=torch.int32),
        draft_lengths=torch.ones((2,), dtype=torch.int32),
        segment_ids=torch.tensor([0, 1], dtype=torch.int64),
        group_indices=torch.tensor([0, 0], dtype=torch.int64),
    )
    calls = []
    runner = _sample_tokens_runner(
        [md if spec_step else None, None], params_by_start, calls
    )

    TPUModelRunner.sample_tokens(runner, None)

    seen = [(kind, (k["use_top_k"], k["use_top_p"])) for kind, k in calls]
    if spec_step:
        assert seen == [("sample", first), ("reject", first), ("sample", second)]
    else:
        assert seen == [("sample", first), ("sample", second)]
