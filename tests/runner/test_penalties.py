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
"""Penalties in the TPU runner: the inputs read from vLLM's InputBatch, where
sample_tokens applies them, async settling and precompile coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from runner_output_test_utils import _plain_step_runner, _run_sample_tokens
from vllm import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.layers.adapter.sample import penalties as sampling_penalties
from vllm_torchtpu.runner import penalties as runner_penalties
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

VOCAB = 16
WIDTH = 4  # a small compiled width, so short histories take several passes


@pytest.fixture
def small_width(monkeypatch):
    monkeypatch.setattr(runner_penalties, "PENALTY_HISTORY_WIDTH", WIDTH)
    monkeypatch.setattr(sampling_penalties, "PENALTY_HISTORY_WIDTH", WIDTH)


def _request(req_id, prompt, output, **penalty_kwargs):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, **penalty_kwargs),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=len(prompt),
        output_token_ids=list(output),
    )


def _input_batch(*requests):
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[16],
        max_num_blocks_per_req=[4],
        logitsprocs=LogitsProcessors(),
        logitsprocs_need_output_token_ids=True,
    )
    for r in requests:
        batch.add_request(r)
    batch.refresh_metadata()
    return batch


@pytest.mark.usefixtures("small_width")
class TestInputBatchInputs:
    def test_reads_the_rows_after_a_reorder(self):
        """The runner reorders the batch (swap_states) after vLLM fills it;
        the inputs must follow the requests to their new rows."""
        a = _request("a", [1, 2, 3], [4, 4], repetition_penalty=1.3)
        b = _request("b", [5], [6])  # no penalty
        batch = _input_batch(a, b)
        assert not batch.no_penalties
        batch.swap_states(0, 1)  # as _reorder_batch_for_rpa does

        rows = runner_penalties.penalized_rows(batch, 0, 2)
        assert rows.tolist() == [1]
        params = np.zeros((4, runner_penalties.NUM_PENALTY_PARAMS), np.float32)
        runner_penalties.fill_params(params, batch, 0, rows)
        np.testing.assert_allclose(
            params, [[1, 0, 0], [1.3, 0, 0], [1, 0, 0], [1, 0, 0]], rtol=1e-6
        )
        passes = runner_penalties.num_passes(batch, 0, rows)
        assert passes == 2  # 5 tokens, width 4
        tables = [np.zeros((4, WIDTH), np.int32) for _ in range(passes)]
        runner_penalties.fill_history_tables(tables, batch, 0, rows)
        assert tables[0][1].tolist() == [1, 2, 3, -6]  # output 4 -> -(4 + 2)
        assert tables[1][1].tolist() == [-6, -1, -1, -1]
        assert (tables[0][0] == -1).all() and (tables[0][2:] == -1).all()

    def test_chunk_offset(self):
        batch = _input_batch(
            _request("a", [1], [2], frequency_penalty=0.5),
            _request("b", [3], [4], presence_penalty=0.5),
        )
        rows = runner_penalties.penalized_rows(batch, 1, 2)
        assert rows.tolist() == [0]  # relative to the chunk start
        tables = [np.zeros((2, WIDTH), np.int32)]
        runner_penalties.fill_history_tables(tables, batch, 1, rows)
        assert tables[0][0, :2].tolist() == [3, -6]


def _penalized(runner, rep=1.0, pres=0.0, freq=0.0, history=(3, 3), prompt=1):
    """Request r0 penalized, with `history` (prompt then output) so far."""
    batch = runner.input_batch
    batch.no_penalties = False
    batch.repetition_penalties_cpu[0] = rep
    batch.presence_penalties_cpu[0] = pres
    batch.frequency_penalties_cpu[0] = freq
    batch.token_ids_cpu[0, : len(history)] = history
    batch.num_tokens_no_spec[0] = len(history)
    batch.num_prompt_tokens[0] = prompt


def _logits():
    logits = torch.zeros(1, 8)
    logits[0, 3], logits[0, 5] = 4.0, 3.5  # 3 repeats in the history
    return logits


@pytest.mark.parametrize("penalized, expected", [(False, 3), (True, 5)])
def test_greedy_step_is_penalized(penalized, expected):
    runner = _plain_step_runner(None, _logits())
    if penalized:
        _penalized(runner, rep=2.0)  # 4.0 / 2 < 3.5
    assert _run_sample_tokens(runner).sampled_token_ids == [[expected]]


@pytest.mark.parametrize("penalized, expected", [(False, 3), (True, 5)])
def test_sampled_step_is_penalized(penalized, expected):
    """Temperature 1 through the runner's own sampler: token 3 holds all but
    e^-10 of the mass until its frequency penalty."""
    logits = torch.full((1, 8), float("-inf"))
    logits[0, 3], logits[0, 5] = 10.0, 0.0
    runner = _plain_step_runner(None, logits)
    runner.input_batch.all_greedy = False
    runner._build_padded_sampling_params = lambda *_args: (
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones((1, 1)),
    )
    runner._sampler_topk_candidates = lambda *_args: 0
    runner._sampler_applies_topk_topp = (
        TPUModelRunner._sampler_applies_topk_topp.__get__(runner)
    )
    runner._get_sampling_generator = lambda: torch.Generator().manual_seed(0)
    runner._apply_temperature = TPUModelRunner._apply_temperature.__get__(runner)
    runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(runner)
    if penalized:
        _penalized(runner, freq=10.0, history=(1, 3), prompt=1)  # 3 in the output
    assert _run_sample_tokens(runner).sampled_token_ids == [[expected]]


def test_second_chunk_is_penalized_at_its_own_rows():
    """Two chunks; only r1 (row 1, start 1) is penalized."""
    runner = _plain_step_runner(None, _logits())
    st = runner.execute_model_state
    st.scheduler_output.num_scheduled_tokens = {"r0": 1, "r1": 1}
    st.logits_list = [_logits(), _logits()]
    st.num_reqs_list = [1, 1]
    st.spec_decode_metadata_list = [None, None]
    batch = runner.input_batch
    batch.num_reqs, batch.req_ids = 2, ["r0", "r1"]
    batch.req_id_to_index = {"r0": 0, "r1": 1}
    batch.num_tokens_no_spec = np.array([2, 2])
    batch.num_prompt_tokens = np.array([1, 1])
    batch.token_ids_cpu = np.full((2, 8), 3, dtype=np.int32)
    batch.no_penalties = False
    batch.repetition_penalties_cpu = np.array([1.0, 2.0], dtype=np.float32)
    batch.presence_penalties_cpu = np.zeros(2, dtype=np.float32)
    batch.frequency_penalties_cpu = np.zeros(2, dtype=np.float32)
    runner.requests["r1"] = SimpleNamespace(
        num_computed_tokens=0, num_tokens=1, output_token_ids=[]
    )
    assert _run_sample_tokens(runner).sampled_token_ids == [[3], [5]]


def test_speculative_config_skips_penalties_and_warns():
    """Verify steps do not apply penalties yet, so with a speculative config
    every step skips them (documented); the draft-less chunk of a spec step
    is then sampled from the unpenalized logits."""
    runner = _plain_step_runner(None, _logits(), req_id="r1")
    _penalized(runner, rep=2.0)
    st = runner.execute_model_state
    st.scheduler_output.num_scheduled_tokens = {"r0": 2, "r1": 1}
    st.logits_list = [torch.zeros(2, 8), _logits()]
    st.num_reqs_list = [1, 1]
    st.spec_decode_metadata_list = [object(), None]
    batch = runner.input_batch
    batch.num_reqs, batch.req_ids = 2, ["r0", "r1"]
    batch.req_id_to_index = {"r0": 0, "r1": 1}
    batch.num_tokens_no_spec = np.array([1, 1])
    batch.token_ids_cpu = np.zeros((2, 8), dtype=np.int32)
    runner.requests["r1"] = SimpleNamespace(
        num_computed_tokens=0, num_tokens=1, output_token_ids=[]
    )
    runner.speculative_config = SimpleNamespace(
        method="ngram", num_speculative_tokens=1
    )
    runner._sample_spec_verify_chunk = lambda *_args: torch.tensor([[7, -1]])
    runner.drafter = SimpleNamespace()
    runner.spec_decode_manager = SimpleNamespace(
        propose_draft_token_ids=lambda **_kwargs: None
    )
    runner._sync_replicated_drafts_across_tp = lambda drafts: drafts
    runner._apply_penalties = MagicMock(side_effect=AssertionError("not skipped"))
    with patch("vllm_torchtpu.runner.tpu_runner.logger") as logger:
        output = _run_sample_tokens(runner)
    assert output.sampled_token_ids == [[7], [3]]
    logger.warning_once.assert_called_once()


def test_logprobs_come_from_the_unpenalized_logits():
    runner = _plain_step_runner(None, _logits())
    _penalized(runner, rep=2.0)
    runner.input_batch.max_num_logprobs = 0
    seen = []

    def gather_logprobs(x, tokens):
        seen.append(x.clone())
        return LogprobsTensors(
            tokens.view(-1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64)
        )

    runner.gather_logprobs = gather_logprobs
    assert _run_sample_tokens(runner).sampled_token_ids == [[5]]
    assert len(seen) == 1 and seen[0][0, 3] == 4.0


def test_top_tokens_step_with_penalties_is_refused():
    runner = _plain_step_runner(None, _logits())
    _penalized(runner, rep=2.0)
    runner.execute_model_state.top_tokens = True
    with pytest.raises(AssertionError, match="penalties"):
        TPUModelRunner.sample_tokens(runner, None)


def test_top_tokens_path_is_off_with_a_penalized_request():
    runner = object.__new__(TPUModelRunner)
    runner._greedy_head_owner = object()
    runner.requests = {"r0": SimpleNamespace(sampling_params=SamplingParams())}
    runner._min_tokens = SimpleNamespace(any_in=lambda _index: False)
    runner.input_batch = SimpleNamespace(
        all_greedy=True,
        max_num_logprobs=None,
        no_penalties=True,
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
    )
    assert runner._greedy_top_tokens_this_step()
    runner.input_batch.no_penalties = False
    assert not runner._greedy_top_tokens_this_step()


class TestAsyncSettle:
    def _runner(self, rep):
        runner = _plain_step_runner(None, _logits())
        # rep 1.0: the batch has a penalized request, but not in this chunk.
        _penalized(runner, rep=rep)
        runner._pre_async_results = object()
        runner._modify_prev_results = MagicMock()
        return runner

    def test_batch_state_has_no_side_effect(self):
        runner = self._runner(rep=2.0)
        assert runner._penalties_batch_state()
        runner._modify_prev_results.assert_not_called()
        assert runner._pre_async_results is not None

    def test_settled_before_a_penalized_chunk_reads_its_history(self):
        runner = self._runner(rep=2.0)
        with torch.compiler.set_stance("force_eager"):
            runner._apply_penalties(_logits(), 0, 1)
        runner._modify_prev_results.assert_called_once()
        assert runner._pre_async_results is None

    def test_not_settled_for_a_chunk_without_penalized_rows(self):
        runner = self._runner(rep=1.0)
        logits = _logits()
        assert runner._apply_penalties(logits, 0, 1) is logits
        runner._modify_prev_results.assert_not_called()


def _precompile_runner(dtype):
    runner = object.__new__(TPUModelRunner)
    runner.num_reqs_paddings = [8, 16]
    runner.speculative_config = None
    runner.device = torch.device("cpu")
    runner._hidden_states_dtype = dtype
    runner.vocab_size = VOCAB
    runner._precompile_timed = MagicMock()
    runner._topk_candidate_blocks = lambda: 0
    return runner


@pytest.mark.usefixtures("small_width")
def test_precompile_covers_the_penalty_calls_serving_makes():
    """Every penalty graph and sampler call of a penalized step (two passes,
    greedy and sampled) uses a shape, dtype and input kind (no views) that
    warmup compiled."""
    runner = _precompile_runner(torch.bfloat16)
    calls = set()
    real = {
        name: getattr(sampling_penalties, name)
        for name in ("penalty_counts", "add_penalty_counts", "apply_penalty_counts")
    }

    def spy(name):
        def wrapped(*args):
            # A view (e.g. a device-side .view of a staged upload) reaches the
            # TPU graph as a different input and compiles a new program.
            calls.add((name, *((a.shape, a.dtype, a._base is None) for a in args)))
            return real[name](*args)

        return wrapped

    def sampler(logits, temperatures, u, top_k, top_p, **kwargs):
        calls.add(("sample", logits.shape, logits.dtype, temperatures.dtype))
        return torch.zeros((logits.shape[0], 1), dtype=torch.int64)

    runner.sample_from_logits_func = sampler
    with (
        patch.multiple(sampling_penalties, **{n: spy(n) for n in real}),
        patch(
            "vllm_torchtpu.runner.tpu_runner.random_sampler_variants",
            return_value=[(0, True)],
        ),
        torch.compiler.set_stance("force_eager"),
    ):
        runner._precompile_penalties()
        runner._precompile_sample_from_logits()
        precompiled, calls = calls, set()
        for num_rows in (8, 16):
            step = _plain_step_runner(None, torch.zeros(num_rows, VOCAB))
            _penalized(step, rep=1.5, history=(1, 2, 3, 3, 3, 3), prompt=2)
            logits = step._penalties_chunk(0, 1).apply(
                torch.zeros(num_rows, VOCAB, dtype=torch.bfloat16)
            )
            temperatures = torch.ones((num_rows, 1), dtype=logits.dtype)
            sampler(logits, temperatures, None, None, None)
    assert {c[0] for c in calls} == {
        "penalty_counts",
        "add_penalty_counts",
        "apply_penalty_counts",
        "sample",
    }
    assert calls <= precompiled


class _RecordingProcessor:
    """A stand-in for a later processor (e.g. the thinking budget's)."""

    def __init__(self):
        self.seen = []
        self.sampled = None

    def apply(self, logits):
        self.seen.append(logits.clone())
        return logits

    def override_top_tokens(self, token_ids):
        return token_ids

    def on_sampled(self, token_ids):
        self.sampled = token_ids


def test_penalties_run_before_the_thinking_budget():
    """vLLM's Sampler.apply_logits_processors applies penalties, then the
    thinking-token budget; the thinking budget's processor here gets the
    penalized float32 logits, and the token sampled after both."""
    runner = _plain_step_runner(None, _logits())
    _penalized(runner, rep=2.0)
    later = _RecordingProcessor()
    tb_step = MagicMock()
    tb_step.plain_chunk.return_value = later
    runner._thinking_budget = MagicMock()
    runner._thinking_budget.step_state.return_value = tb_step
    assert _run_sample_tokens(runner).sampled_token_ids == [[5]]
    assert len(later.seen) == 1
    assert later.seen[0].dtype == torch.float32
    assert later.seen[0][0, 3] == 2.0  # 4.0 / repetition 2.0
    assert later.sampled.view(-1).tolist() == [5]


class TestPenaltiesChunk:
    def test_none_for_a_chunk_without_penalized_rows(self):
        runner = _plain_step_runner(None, _logits())
        _penalized(runner, rep=1.0)
        assert runner._penalties_chunk(0, 1) is None

    def test_processor_hooks(self):
        runner = _plain_step_runner(None, _logits())
        _penalized(runner, rep=2.0)
        chunk = runner._penalties_chunk(0, 1)
        with torch.compiler.set_stance("force_eager"):
            out = chunk.apply(_logits())
        assert out.dtype == torch.float32 and out[0, 3] == 2.0
        assert chunk.on_sampled(torch.zeros(1, 1, dtype=torch.int64)) is None
        with pytest.raises(RuntimeError, match="full logits"):
            chunk.override_top_tokens(torch.zeros(1, 1, dtype=torch.int64))


@pytest.mark.parametrize(
    "spec, dtypes",
    [
        (None, [torch.bfloat16, torch.float32]),
        (SimpleNamespace(num_speculative_tokens=1), [torch.bfloat16]),
    ],
)
def test_thinking_budget_precompile_covers_penalized_float32_logits(spec, dtypes):
    """The thinking budget runs after the penalties, so without a speculative
    config (when penalties can apply) its graphs are also warmed for float32."""
    runner = object.__new__(TPUModelRunner)
    runner._thinking_budget = MagicMock(enabled=True)
    runner.speculative_config = spec
    runner._mtp_prefill_only = False
    runner._verify_logits_paddings = lambda: []
    runner.num_reqs_paddings = [8]
    runner._precompile_timed = MagicMock()
    runner._greedy_head_owner = None
    runner._hidden_states_dtype = torch.bfloat16
    runner._dummy_logits = lambda n: torch.zeros(n, VOCAB, dtype=torch.bfloat16)
    runner._precompile_thinking_budget()
    calls = runner._thinking_budget.precompile.call_args_list
    assert [c.kwargs["dummy_logits_fn"](8).dtype for c in calls] == dtypes
