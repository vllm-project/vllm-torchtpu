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

"""Opt-in worker RPC for real MTP acceptance and idle-DP integration tests.

Load Qwen4ExpMTPValidationWorker as --worker-extension-cls on a fresh, disposable
validation engine with prefix caching disabled. Invoke validate_qwen4_exp_mtp via
collective_rpc on all DP ranks, then shut down the engine. Never run this against
a paused serving engine: synthetic physical blocks bypass scheduler ownership.
Never loaded by the production warmup path.
"""

import json
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path

import torch
from vllm.distributed.parallel_state import get_dp_group
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.outputs import ModelRunnerOutput

from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput


@dataclass
class _ValidationState:
    async_scheduling: bool
    contexts: dict[str, int] = field(default_factory=dict)
    output_counts: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        output: ModelRunnerOutput,
        scheduled: dict[str, int],
        drafts: dict[str, list[int]],
    ) -> None:
        for rid, tokens in zip(output.req_ids, output.sampled_token_ids):
            if rid not in scheduled:
                continue
            rejected = len(drafts[rid]) + 1 - len(tokens) if rid in drafts else 0
            assert 0 <= rejected <= len(drafts.get(rid, []))
            self.contexts[rid] = self.contexts.get(rid, 0) + scheduled[rid] - rejected
            self.output_counts[rid] = self.output_counts.get(rid, 0) + len(tokens)
            self.rejected[rid] = rejected

    def lengths(self, req_ids: list[str]) -> tuple[list[int], list[int]]:
        correction = [
            self.rejected[rid] if self.async_scheduling else 0 for rid in req_ids
        ]
        return (
            [self.contexts[rid] + n for rid, n in zip(req_ids, correction)],
            [self.output_counts[rid] + n for rid, n in zip(req_ids, correction)],
        )


def validate_live_hidden(hidden: torch.Tensor, valid_tokens: int) -> None:
    if not torch.isfinite(hidden[:valid_tokens]).all().item():
        raise RuntimeError("MTP validation produced non-finite target hidden states")


def _run_request(runner, *, accepted: int | None, greedy: bool, idle_peers: bool):
    assert runner.input_batch.num_reqs == 0
    assert not runner.requests, "Validation requires an empty service"
    k = runner.speculative_config.num_speculative_tokens
    prompt = int(runner.num_tokens_paddings[0])
    steps = max(5, k + 2)
    max_pos = prompt + steps * (k + 1)
    assert max_pos <= runner.max_model_len
    counts = [
        group.kv_cache_spec.max_num_blocks_per_req(runner.vllm_config, max_pos)
        for group in runner.kv_cache_config.kv_cache_groups
    ]
    offsets = list(accumulate([1, *counts]))
    assert offsets[-1] <= runner.kv_cache_config.num_blocks
    blocks = tuple(list(range(a, b)) for a, b in zip(offsets, offsets[1:]))
    active = not idle_peers or get_dp_group().rank_in_group == 0
    progress = _ValidationState(bool(runner.scheduler_config.async_scheduling))
    rid = "__mtp_state_validation__"
    cur = [0] * k
    observed = []
    try:
        for step in range(steps + 1):
            so = SchedulerOutput.make_empty()
            if step == 0:
                so.scheduled_new_reqs = [
                    NewRequestData(
                        req_id=rid,
                        prompt_token_ids=[0] * prompt,
                        mm_features=[],
                        sampling_params=SamplingParams(
                            temperature=0.0 if greedy else 1.0
                        ),
                        pooling_params=None,
                        block_ids=blocks,
                        num_computed_tokens=0,
                        lora_request=None,
                    )
                ]
            elif active:
                contexts, outputs = progress.lengths([rid])
                so.scheduled_cached_reqs = CachedRequestData(
                    req_ids=[rid],
                    resumed_req_ids=set(),
                    new_token_ids=[],
                    all_token_ids={rid: [0] * (contexts[0] + 1)},
                    new_block_ids=[None],
                    num_computed_tokens=contexts,
                    num_output_tokens=outputs,
                )
                so.scheduled_spec_decode_tokens = {rid: cur}
            count = prompt if step == 0 else k + 1
            so.num_scheduled_tokens = {rid: count}
            so.total_num_scheduled_tokens = count
            if not active:
                bucket, chunks = runner._dp_coordinated_step(0, 0)
                assert bucket is not None
                runner._run_dp_idle_pairing(bucket, chunks)
                continue
            assert runner.execute_model(so) is None
            state = runner.execute_model_state
            assert state is not None
            for chunk in state.draft_chunks:
                assert chunk.hidden_states is not None
                validate_live_hidden(
                    chunk.hidden_states, int(chunk.query_start_loc_np[chunk.num_reqs])
                )
            result = runner.sample_tokens(None)
            if isinstance(result, AsyncTPUModelRunnerOutput):
                result = result.get_output()
            if step:
                actual = len(result.sampled_token_ids[0]) - 1
                assert 0 <= actual <= k
                if accepted is not None:
                    assert actual == accepted, (step, actual, accepted)
                observed.append(actual)
            progress.record(
                result, so.num_scheduled_tokens, so.scheduled_spec_decode_tokens
            )
            draft = runner.take_draft_token_ids()
            if draft and draft.draft_token_ids:
                cur = [int(token) for token in draft.draft_token_ids[0][:k]]
    finally:
        assert runner._warmup_spec_decode_cleanup(rid)
    assert runner.input_batch.num_reqs == 0 and not runner.requests
    return observed


class Qwen4ExpMTPValidationWorker:
    def validate_qwen4_exp_mtp(self, output_dir: str):
        runner = self.model_runner
        if runner.cache_config.enable_prefix_caching:
            raise ValueError(
                "MTP validation requires a disposable engine with prefix caching "
                "disabled: synthetic blocks can overwrite scheduler-owned cache"
            )
        sampler = runner.rejection_sampler
        assert sampler is not None
        k = runner.speculative_config.num_speculative_tokens
        saved_mode = sampler.synthetic_mode
        saved_rates = sampler.synthetic_conditional_rates
        cases = []
        try:
            sampler.synthetic_mode = True
            for accepted in range(k + 1):
                sampler.synthetic_conditional_rates = torch.tensor(
                    [1.0] * accepted + [0.0] * (k - accepted),
                    dtype=torch.float32,
                    device=runner.device,
                )
                for greedy in (True, False):
                    observed = _run_request(
                        runner, accepted=accepted, greedy=greedy, idle_peers=False
                    )
                    cases.append(
                        dict(accepted=accepted, greedy=greedy, observed=observed)
                    )
        finally:
            sampler.synthetic_mode = saved_mode
            sampler.synthetic_conditional_rates = saved_rates
        if runner._dp_lockstep_enabled():
            for greedy in (True, False):
                observed = _run_request(
                    runner, accepted=None, greedy=greedy, idle_peers=True
                )
                cases.append(dict(idle_peers=True, greedy=greedy, observed=observed))
        result = dict(
            dp_rank=self.parallel_config.data_parallel_rank,
            k=k,
            async_scheduling=bool(runner.scheduler_config.async_scheduling),
            cases=cases,
        )
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"rank{result['dp_rank']}.json").write_text(json.dumps(result))
        return result
