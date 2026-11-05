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
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp)
        )

    assert backend.capabilities(vllm_config(1)).pipeline is False
    with pytest.raises(ValueError, match="pipeline stage"):
        backend.capabilities(vllm_config(4))


def _kimi_k31_runner(budget: int, draft_layers: int = 0):
    """A runner stub over Kimi-K3.1's KV specs: 24 MLA layers (each with an
    index-key cache) and 69 KDA layers on one unified page, plus the MLA
    DSpark draft's latent layers (the target's MLA spec)."""
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    page = 327680
    mla = [*range(3, 92, 4), 92]
    context = {
        f"language_model.model.layers.{i}.self_attn.indexer.k_cache": object()
        for i in mla
    }
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=None,
        kv_cache_memory_bytes=budget,
        cache_dtype="fp8",
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
    )
    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            attention_config=SimpleNamespace(hisparse_config=None),
            cache_config=cache_config,
            speculative_config=None,
            model_config=None,
        ),
        cache_config=cache_config,
        device="cpu",
        kv_cache_raw_tensors=[],
    )
    attention = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        page_size_padded=page,
    )
    assert attention.page_size_bytes == page
    kda = MambaSpec(
        block_size=512,
        shapes=((4, 128, 128),),
        dtypes=(torch.float32,),
        page_size_padded=page,
    )
    specs = {
        f"language_model.model.layers.{i}.self_attn.attn": attention
        if i in mla
        else kda
        for i in range(93)
    }
    for j in range(draft_layers):
        specs[f"model.layers.{93 + j}.self_attn.attn"] = attention
    return fake, specs, attention


@pytest.mark.cpu_test
def test_runner_reserves_and_allocates_the_index_cache():
    budget = 30 * 2**30
    fake, specs, spec = _kimi_k31_runner(budget)
    MegakernelModelRunner._reserve_index_cache(fake, specs)
    blocks = fake.cache_config.num_gpu_blocks_override
    assert blocks == budget // (24 * (327680 + 544 * 128))
    group = SimpleNamespace(
        kv_cache_spec=spec, layer_names=["language_model.model.layers.3.self_attn.attn"]
    )
    kv_config = SimpleNamespace(num_blocks=4, kv_cache_groups=[group])
    caches = MegakernelModelRunner.extra_kv_cache_tensors(fake, kv_config)
    assert len(caches) == 24
    assert next(iter(caches.values())).shape == (4, 24, 544, 128)
    no_mla = SimpleNamespace(num_blocks=4, kv_cache_groups=[])
    with pytest.raises(RuntimeError, match="one MLA block size"):
        MegakernelModelRunner.extra_kv_cache_tensors(fake, no_mla)


@pytest.mark.cpu_test
def test_index_cache_budget_counts_the_draft_layers():
    """The MLA DSpark draft's five latent layers widen vLLM's MLA group to 29
    unified pages a block: the budget keeps the pools and the 24 layers'
    index pages within the KV budget."""
    import torch

    budget = 18 * 2**30
    fake, specs, spec = _kimi_k31_runner(budget, draft_layers=5)
    MegakernelModelRunner._reserve_index_cache(fake, specs)
    blocks = fake.cache_config.num_gpu_blocks_override
    per_block = 29 * 327680 + 24 * 544 * 128
    assert blocks == budget // per_block == 1729
    fake.kv_cache_raw_tensors = [
        torch.empty(blocks * 327680, dtype=torch.uint8, device="meta")
        for _ in range(29)
    ]
    group = SimpleNamespace(
        kv_cache_spec=spec, layer_names=["language_model.model.layers.3.self_attn.attn"]
    )
    kv_config = SimpleNamespace(num_blocks=blocks, kv_cache_groups=[group])
    caches = MegakernelModelRunner.extra_kv_cache_tensors(fake, kv_config)
    assert next(iter(caches.values())).shape == (blocks, 24, 544, 128)


@pytest.mark.cpu_test
def test_runner_has_no_index_cache_for_kimi_k3():
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=None, kv_cache_memory_bytes=1, cache_dtype="fp8"
    )
    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={"x.attn": object()}
            )
        ),
        cache_config=cache_config,
    )
    MegakernelModelRunner._reserve_index_cache(fake, {})
    assert cache_config.num_gpu_blocks_override is None
    kv_config = SimpleNamespace(num_blocks=4, kv_cache_groups=[])
    assert MegakernelModelRunner.extra_kv_cache_tensors(fake, kv_config) == {}
