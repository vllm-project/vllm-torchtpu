# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The megakernel runner: asynchronous responses preserve worker state and KV
metadata; the program drops finished and preempted requests' state; a pipeline
stage runs each step between the wave's hand-offs."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.megakernel.runner import MegakernelModelRunner


@pytest.mark.parametrize("output_rank", [False, True])
@pytest.mark.parametrize("kv_transfer", [False, True])
def test_async_sample_tokens_preserves_local_state_and_transfer_metadata(
    monkeypatch, output_rank, kv_transfer
):
    import torch
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.megakernel import runner
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

    tokens = torch.tensor([[101], [202]], dtype=torch.int32)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[tokens],
        chunk_real_lens=[2],
        copy_ready_event=None,
        completed=True,
    )
    monkeypatch.setattr(
        AsyncTPUCopyState, "from_device_chunks", lambda *args: copy_state
    )
    monkeypatch.setattr(runner, "has_kv_transfer_group", lambda: kv_transfer)
    connector_output = KVConnectorOutput(expected_finished_count=32)
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})
    instance = SimpleNamespace(
        execute_model_state=runner.MegakernelStepState(scheduled, tokens, {0: 0, 1: 0}),
        input_batch=SimpleNamespace(
            num_reqs=2, req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}
        ),
        requests={
            request_id: SimpleNamespace(num_computed_tokens=10, num_tokens=11)
            for request_id in ("a", "b")
        },
        scheduler_config=SimpleNamespace(async_scheduling=True),
        _pp_wave=None,
        _is_output_rank=output_rank,
        _modify_prev_results=lambda: None,
        maybe_wait_for_kv_save=lambda: None,
        _build_kv_connector_output=lambda _: connector_output,
        _update_placeholder=lambda *args: {"a": 0, "b": 1},
    )
    result = MegakernelModelRunner.sample_tokens(instance).get_output()
    assert result.kv_connector_output is connector_output
    assert instance._pre_async_results.req_ids == ["a", "b"]
    assert instance._pre_async_results.copy_state is copy_state
    assert copy_state.sampled_token_ids_cpu.tolist() == [[101], [202]]
    if output_rank or not kv_transfer:
        assert result.req_ids == ["a", "b"]
        assert result.sampled_token_ids == [[101], [202]]
    else:
        assert result.req_ids == []
        assert result.sampled_token_ids == []


def test_preempted_requests_reach_the_program_with_finished_ones(monkeypatch):
    from vllm_torchtpu.megakernel import runner

    monkeypatch.setattr(runner, "has_kv_transfer_group", lambda: False)
    instance = SimpleNamespace(
        execute_model_state=None,
        _update_states=lambda scheduler_output: None,
        _finished_req_ids_pending=set(),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _pp_wave=None,
    )
    # A zero-token step: the program runs on a later one.
    scheduled = SimpleNamespace(
        finished_req_ids={"done"},
        preempted_req_ids={"preempted"},
        total_num_scheduled_tokens=0,
    )
    MegakernelModelRunner.execute_model(instance, scheduled)
    assert instance._finished_req_ids_pending == {"done", "preempted"}


class _Wave:
    """Records the runner's calls into the pipeline wave."""

    def __init__(self, received=None):
        self.calls = []
        self.received = received

    def start_forward(self, rows):
        self.calls.append(("start", rows))
        return None if self.received is None else SimpleNamespace(tensors=self.received)

    def end_forward(self, tensors, rows):
        self.calls.append(("end", None if tensors is None else sorted(tensors), rows))


class _Stage:
    """A pipeline-stage program that records what each step received."""

    def __init__(self, last):
        self.last = last
        self.inputs = []

    def rows(self, step_input):
        return 24

    def step(self, step_input):
        from vllm_torchtpu.megakernel.program import StepOutput

        self.inputs.append(step_input.hidden)
        if self.last:
            return StepOutput("tokens")
        return StepOutput(None, {"hidden_states": "activations"})


def _stage_runner(wave, last):
    program = _Stage(last)
    return SimpleNamespace(
        program=program,
        _stage=lambda: program,
        _pp_wave=wave,
        _pp_is_last=last,
    )


@pytest.mark.parametrize("last", [False, True])
def test_a_stage_runs_each_step_between_the_hand_offs(last):
    from vllm_torchtpu.megakernel.program import StepInput

    wave = _Wave(received={"hidden_states": "from the previous stage"})
    instance = _stage_runner(wave, last)
    output = MegakernelModelRunner._run_program(instance, StepInput([]))
    assert instance.program.inputs == [{"hidden_states": "from the previous stage"}]
    if last:
        assert output.tokens == "tokens"
        assert wave.calls == [("start", 24), ("end", None, 24)]
    else:
        assert output is None
        assert wave.calls == [("start", 24), ("end", ["hidden_states"], 24)]


def test_the_first_stage_receives_no_hand_off():
    from vllm_torchtpu.megakernel.program import StepInput

    instance = _stage_runner(_Wave(received=None), last=False)
    MegakernelModelRunner._run_program(instance, StepInput([]))
    assert instance.program.inputs == [None]


def test_an_earlier_stage_reports_only_its_connector_progress():
    instance = SimpleNamespace(
        execute_model_state=None,
        _pp_wave=None,
        _pp_is_last=False,
        _sample_tokens_pp_intermediate_stage=lambda: "connector progress",
    )
    assert MegakernelModelRunner.sample_tokens(instance) == "connector progress"


def test_capture_model_builds_the_wave_from_the_program(monkeypatch):
    import torch

    from vllm_torchtpu.megakernel import runner

    order = []

    class Wave:
        def __init__(self, device, max_rows, template):
            order.append(("wave", device, max_rows, template))

        def warmup(self):
            order.append("wave warmup")

    template = {"hidden_states": ((8,), torch.bfloat16)}
    program = SimpleNamespace(
        max_rows=lambda: 4096,
        handoff_template=lambda: template,
        warmup=lambda: order.append("program warmup"),
    )
    monkeypatch.setattr(runner, "PPWave", Wave)
    instance = SimpleNamespace(
        program=program,
        _stage=lambda: program,
        _pp_wave=None,
        device="tpu:0",
        parallel_config=SimpleNamespace(pipeline_parallel_size=4),
    )
    assert MegakernelModelRunner.capture_model(instance) == 0
    assert order == [
        ("wave", "tpu:0", 4096, template),
        "wave warmup",
        "program warmup",
    ]
    assert isinstance(instance._pp_wave, Wave)


def test_the_chunk_profile_times_the_program(monkeypatch):
    from vllm_torchtpu.megakernel import backend

    monkeypatch.setattr(
        backend,
        "capabilities",
        lambda config: SimpleNamespace(page_aligned_chunks=True),
    )
    timed = []

    def time_steps(points):
        timed.extend(points)
        return [float(tokens + prefix) / 1000 for tokens, prefix in points]

    instance = SimpleNamespace(
        vllm_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        cache_config=SimpleNamespace(block_size=2048),
        kv_cache_config=None,
        max_model_len=65536,
        _stage=lambda: SimpleNamespace(time_steps=time_steps),
    )
    report = MegakernelModelRunner.profile_pipeline_chunks(instance)
    # An eighth of the step, rounded up to whole KV blocks.
    assert report["granularity"] == 2048
    assert report["schedule"] is None
    assert [(t, p) for t, p, _ in report["samples"]] == timed
    assert (8192, 0) in timed
    assert all(ms == (t + p) / 1000 for t, p, ms in report["samples"])


def test_the_chunk_granule_holds_whole_scheduler_blocks_in_align_mode(monkeypatch):
    from vllm_torchtpu.megakernel import backend

    monkeypatch.setattr(
        backend,
        "capabilities",
        lambda config: SimpleNamespace(page_aligned_chunks=True),
    )

    def granularity(mode):
        instance = SimpleNamespace(
            vllm_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
            cache_config=SimpleNamespace(block_size=640, mamba_cache_mode=mode),
            kv_cache_config=SimpleNamespace(has_mamba_layers=True),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=8),
            max_model_len=262144,
            _stage=lambda: SimpleNamespace(
                time_steps=lambda points: [1.0] * len(points)
            ),
        )
        return MegakernelModelRunner.profile_pipeline_chunks(instance)["granularity"]

    # The scheduler clips align-mode chunks to its 8 x 640-token blocks.
    assert granularity("align") == 5120
    assert granularity("none") == 4480


def test_a_program_that_is_no_pipeline_stage_refuses_a_pipeline(monkeypatch):
    from vllm_torchtpu.megakernel import backend, config
    from vllm_torchtpu.megakernel.kimi_k3 import program

    monkeypatch.setattr(backend, "role", lambda vllm_config: "prefill")
    monkeypatch.setattr(
        config.MegakernelEngineConfig, "from_vllm_config", classmethod(lambda cls, c: c)
    )
    monkeypatch.setattr(
        program, "capabilities", lambda c: SimpleNamespace(pipeline=False)
    )

    def vllm_config(pp):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["KimiK3ForConditionalGeneration"]
                )
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        )

    assert backend.capabilities(vllm_config(1)).pipeline is False
    with pytest.raises(ValueError, match="pipeline stage"):
        backend.capabilities(vllm_config(4))
