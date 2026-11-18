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
"""A simulated engine for the device-decode tests.

`SimEngine` drives the real `TPUModelRunner.execute_model` and `sample_tokens`
on a runner object holding a real vLLM InputBatch and the real attention
metadata builders, fed by a small V1-style scheduler. The model's logits
depend only on each row's input token and position, so two engines given the
same schedule must sample the same tokens whichever path each step takes.
"""

import contextlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.device_decode import controller as controller_mod
from vllm_torchtpu.runner.min_tokens import MinTokensTracker
from vllm_torchtpu.runner.tpu_runner import (
    MIN_NUM_SEQS,
    TPUModelRunner,
    _get_req_paddings,
)
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

FLAG = "VLLM_TPU_DEVICE_DECODE"
MAX_NUM_REQS = 16
TOKEN_PADDINGS = (16, 32, 64)
BLOCK = 16
MAX_MODEL_LEN = 160
VOCAB = 96
HIDDEN = 8
MAX_NUM_BATCHED_TOKENS = 48
MAX_LOGPROBS = 5


def _no_context(*_args, **_kwargs):
    return contextlib.nullcontext()


def fake_tpu_events(monkeypatch) -> None:
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: event), raising=False
    )


class FakeModel:
    """Hidden states and logits that depend only on a row's input token and
    position."""

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.token_embedding = torch.randn(VOCAB, HIDDEN, generator=generator)
        self.position_embedding = torch.randn(
            MAX_MODEL_LEN, HIDDEN, generator=generator
        )
        self.lm_head = 2 * torch.randn(HIDDEN, VOCAB, generator=generator)

    def __call__(self, input_ids, positions, inputs_embeds=None):
        return (
            self.token_embedding[input_ids.long()]
            + self.position_embedding[positions.long()]
        )

    def compute_logits(self, hidden_states):
        return hidden_states @ self.lm_head


def make_runner(
    monkeypatch, *, async_scheduling: bool = False, target_rows: int = MAX_NUM_REQS
) -> TPUModelRunner:
    """A TPUModelRunner object (no __init__) with one attention KV cache group
    and what execute_model, sample_tokens and the device path read.
    `target_rows` is num_reqs_max_model_len: a step with more requests runs in
    several stock chunks."""
    monkeypatch.setattr(
        gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    fake_tpu_events(monkeypatch)
    # The fake model reads neither context.
    for module in (tpu_runner, controller_mod):
        monkeypatch.setattr(module, "set_forward_context", _no_context)
        monkeypatch.setattr(module, "set_vllm_model_wrapper_context", _no_context)

    r = object.__new__(TPUModelRunner)
    r.device = torch.device("cpu")
    r.model = FakeModel()
    r._stage_timing = MagicMock()
    r._stage_timing.take_samples.return_value = ([], [])
    r.model_config = SimpleNamespace(
        seed=0, max_logprobs=MAX_LOGPROBS, uses_mrope=False, enforce_eager=False
    )
    r.enforce_eager = False
    r.vocab_size = VOCAB
    r.max_num_reqs = MAX_NUM_REQS
    r.max_num_local_reqs = MAX_NUM_REQS
    r.num_reqs_max_model_len = target_rows
    r.num_reqs_most_model_len = None
    r.most_model_len = None
    r.num_tokens_paddings = list(TOKEN_PADDINGS)
    r.max_num_tokens = max(TOKEN_PADDINGS)
    r.num_reqs_paddings = _get_req_paddings(MIN_NUM_SEQS, MAX_NUM_REQS)
    r.max_model_len = MAX_MODEL_LEN
    r.block_size = BLOCK
    r.uses_mrope = False
    r.supports_mm_inputs = False
    r.lora_config = None
    r.speculative_config = None
    r.reorder_batch_threshold = 1
    r._pcp_enabled = False
    r._mtp_enabled = False
    r._mtp_prefill_only = False
    r._pp_is_first = True
    r._pp_is_last = True
    r._pp_handoff = None
    r._adaptive_verification = None
    r._is_dspark_pd_producer = False
    r.phase_based_profiler = None
    r.scheduler_config = SimpleNamespace(
        async_scheduling=async_scheduling,
        max_num_seqs=MAX_NUM_REQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )
    r.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=1,
        cp_kv_cache_interleave_size=BLOCK,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
    r.vllm_config = SimpleNamespace(
        parallel_config=r.parallel_config,
        scheduler_config=r.scheduler_config,
        kv_transfer_config=None,
    )
    r.sequence_layout_planner = AllSequenceLayoutPlanner()
    r.pcp_manager = None
    r.cache_config = SimpleNamespace(block_size=BLOCK, mamba_cache_mode="none")
    r.mesh = None
    r._kv_cache_bundle = None

    max_tokens = max(TOKEN_PADDINGS)
    r.input_ids_cpu = torch.zeros(max_tokens, dtype=torch.int32)
    r.positions_cpu = torch.zeros(max_tokens, dtype=torch.int32)
    r.positions_np = r.positions_cpu.numpy()
    r.query_start_loc_cpu = torch.zeros(max_tokens + 1, dtype=torch.int32)
    r.query_start_loc_np = r.query_start_loc_cpu.numpy()
    r.seq_lens_cpu = torch.zeros(max_tokens, dtype=torch.int32)
    r.seq_lens_np = r.seq_lens_cpu.numpy()
    r.arange_np = np.arange(max_tokens, dtype=np.int64)
    r._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    r._decode_device_cache_key = None
    r._cached_query_start_loc = None
    r._cached_logits_indices = None
    r._cached_request_distribution = None
    r._input_staging_fence = None
    r._unfenced_h2d_sources = []
    r._fenced_h2d_sources = []
    r._free_h2d_sources = {}
    r._registered_h2d_sources = set()
    r._h2d_reuse = {}
    r._dp_target_bucket = None
    r._attention_capacity = {}
    r._attention_runs_batched_kernel = False
    r.mamba_slot_read_offsets = None
    r._mamba_ckpt_window = 1
    r._unified_kv_layout = False
    r._has_mamba_state = False
    r._mamba_slot_by_req_id = {}
    r._free_mamba_slots = []
    r._mamba_state_pos = {}
    r._mamba_offset_seeded = set()
    r._mamba_copy_plan = []
    r._pending_mamba_state_copies = []
    r._prefill_ckpt_units = None
    r._prefill_ckpt_offsets_cpu = None
    r._token_padding_state = None
    r.empty_slot_mappings = {}
    r._pre_async_results = None
    r._last_sequence_layout_plan = None
    r._fast_token_substitution = False
    r._sampling_generator = None
    r.execute_model_state = None
    r._hidden_states_dtype = torch.float32
    r._update_num_xla_graphs = lambda *_args, **_kwargs: None
    r._greedy_head_owner = None
    r._dp_vocab_logits = False
    r.sample_from_logits_func = r.sample_from_logits
    r._device_decode = None

    r.requests = {}
    r.num_prompt_logprobs = {}
    r._prompt_logprobs_unsupported = "the CPU test runner"
    r._min_tokens = MinTokensTracker()
    r.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _ids: None,
        register_request=lambda _req_id, _params: None,
    )
    r.encoder_cache = {}
    r.is_pooling_model = False
    r.uses_xdrope_dim = 0
    r.use_async_scheduling = async_scheduling
    r.use_async_spec_decode = False
    r._may_reorder_batch = lambda _output: None
    r._install_spec_token_room_guard = lambda: None

    r.input_batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=max_tokens,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        max_num_blocks_per_req=[-(-MAX_MODEL_LEN // BLOCK)],
    )
    spec = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=16384,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["attn.0"],
        vllm_config=MagicMock(),
        device=r.device,
        runner=r,
        kv_cache_group_id=0,
    )
    r.kv_cache_config = SimpleNamespace(
        num_blocks=4096,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=spec)],
        has_mamba_layers=False,
    )
    r.attn_groups = [
        [SimpleNamespace(metadata_builders=[builder], layer_names=["attn.0"])]
    ]
    return r


def enable_device_decode(monkeypatch, runner):
    """Builds and initializes the runner's controller, as __init__,
    profile_run and capture_model do."""
    monkeypatch.setenv(FLAG, "1")
    runner._device_decode = controller_mod.DeviceDecodeController.maybe_create(runner)
    runner._device_decode.precompile_heads()
    runner._device_decode.initialize_and_precompile()
    assert runner._device_decode.enabled, runner._device_decode.disabled_reason
    return runner._device_decode


@dataclass
class SimRequest:
    req_id: str
    prompt: list[int]
    params: SamplingParams
    output: list[int] = field(default_factory=list)
    num_computed: int = 0
    blocks: list[int] = field(default_factory=list)
    status: str = "waiting"  # waiting | running | preempted
    known_to_runner: bool = False

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)


class SimScheduler:
    """Running requests first (decode or a prefill chunk) under a token
    budget, then waiting and preempted requests while the batch has room."""

    def __init__(self):
        self.reqs: dict[str, SimRequest] = {}
        self.queue: list[str] = []
        self.running: list[str] = []
        self.finished: set[str] = set()
        self.preempted: set[str] = set()
        self.next_block = 1

    def add(self, req_id: str, prompt: list[int], params: SamplingParams) -> None:
        assert req_id not in self.reqs, req_id
        self.reqs[req_id] = SimRequest(req_id, list(prompt), params)
        self.queue.append(req_id)

    def abort(self, req_id: str) -> None:
        req = self.reqs.pop(req_id)
        for ids in (self.running, self.queue):
            if req_id in ids:
                ids.remove(req_id)
        if req.known_to_runner:
            self.finished.add(req_id)

    def preempt(self, req_id: str) -> None:
        req = self.reqs[req_id]
        assert req.status == "running"
        self.running.remove(req_id)
        req.num_computed = 0
        req.blocks = []
        req.status = "preempted"
        self.queue.insert(0, req_id)
        self.preempted.add(req_id)

    def _grow(self, req: SimRequest, seq_len: int) -> list[int]:
        """Allocates the blocks `req` needs to hold `seq_len` tokens."""
        count = max(0, -(-seq_len // BLOCK) - len(req.blocks))
        new = list(range(self.next_block, self.next_block + count))
        self.next_block += count
        req.blocks.extend(new)
        return new

    def schedule(self) -> SchedulerOutput:
        budget = MAX_NUM_BATCHED_TOKENS
        num_scheduled: dict[str, int] = {}
        new_reqs: list[NewRequestData] = []
        cached = SimpleNamespace(
            req_ids=[], resumed=set(), new_block_ids=[], computed=[], outputs=[]
        )
        all_token_ids: dict[str, list[int]] = {}
        for req_id in list(self.running):
            req = self.reqs[req_id]
            n = min(req.num_tokens - req.num_computed, budget)
            if n <= 0:
                continue
            new_ids = self._grow(req, req.num_computed + n)
            cached.req_ids.append(req_id)
            cached.new_block_ids.append((new_ids,) if new_ids else None)
            cached.computed.append(req.num_computed)
            cached.outputs.append(len(req.output))
            num_scheduled[req_id] = n
            budget -= n
        while (
            self.queue
            and not self.preempted
            and budget > 0
            and len(self.running) < MAX_NUM_REQS
        ):
            req_id = self.queue.pop(0)
            req = self.reqs[req_id]
            n = min(req.num_tokens - req.num_computed, budget)
            self._grow(req, req.num_computed + n)
            if req.known_to_runner:
                cached.req_ids.append(req_id)
                cached.resumed.add(req_id)
                cached.new_block_ids.append((list(req.blocks),))
                cached.computed.append(req.num_computed)
                cached.outputs.append(len(req.output))
                all_token_ids[req_id] = req.prompt + req.output
            else:
                new_reqs.append(
                    NewRequestData(
                        req_id=req_id,
                        prompt_token_ids=list(req.prompt),
                        mm_features=[],
                        sampling_params=req.params,
                        pooling_params=None,
                        block_ids=(list(req.blocks),),
                        num_computed_tokens=req.num_computed,
                        lora_request=None,
                    )
                )
                req.known_to_runner = True
            req.status = "running"
            self.running.append(req_id)
            num_scheduled[req_id] = n
            budget -= n
        out = SchedulerOutput.make_empty()
        out.scheduled_new_reqs = new_reqs
        out.scheduled_cached_reqs = CachedRequestData(
            req_ids=cached.req_ids,
            resumed_req_ids=cached.resumed,
            new_token_ids=[],
            all_token_ids=all_token_ids,
            new_block_ids=cached.new_block_ids,
            num_computed_tokens=cached.computed,
            num_output_tokens=cached.outputs,
        )
        out.num_scheduled_tokens = num_scheduled
        out.total_num_scheduled_tokens = sum(num_scheduled.values())
        out.finished_req_ids = set(self.finished)
        out.preempted_req_ids = set(self.preempted) or None
        self.finished.clear()
        self.preempted.clear()
        return out

    def complete(self, out: SchedulerOutput, tokens: dict[str, list[int]]) -> None:
        for req_id, n in out.num_scheduled_tokens.items():
            req = self.reqs[req_id]
            req.num_computed += n
            req.output.extend(tokens.get(req_id, []))
            if len(req.output) >= req.params.max_tokens:
                self.abort(req_id)


class SimEngine:
    """One runner and one scheduler, stepped through the runner's real
    execute_model and sample_tokens."""

    def __init__(
        self,
        monkeypatch,
        *,
        async_scheduling: bool,
        device: bool,
        target_rows: int = MAX_NUM_REQS,
    ):
        self.runner = make_runner(
            monkeypatch, async_scheduling=async_scheduling, target_rows=target_rows
        )
        self.sched = SimScheduler()
        self.controller = (
            enable_device_decode(monkeypatch, self.runner) if device else None
        )
        # Per request id, per submission: the sampled tokens, and the
        # logprobs rows of a request that asks for them.
        self.tokens: dict[str, list[list[int]]] = {}
        self.logprobs: dict[str, list[list[tuple]]] = {}

    def add(self, req_id: str, prompt_len: int, **params) -> None:
        params.setdefault("max_tokens", 12)
        prompt = [
            (7 * ord(req_id[0]) + 3 * i) % (VOCAB - 1) + 1 for i in range(prompt_len)
        ]
        self.sched.add(req_id, prompt, SamplingParams(**params))
        self.tokens.setdefault(req_id, []).append([])
        self.logprobs.setdefault(req_id, []).append([])

    def run_until_done(self, max_steps: int = 60) -> None:
        for _ in range(max_steps):
            if not self.sched.reqs:
                return
            self.step()
        raise AssertionError("requests left after max_steps")

    def step(self) -> SchedulerOutput:
        out = self.sched.schedule()
        with torch.compiler.set_stance("force_eager"):
            output = TPUModelRunner.execute_model(self.runner, out)
            if output is None:
                output = TPUModelRunner.sample_tokens(self.runner, None)
        if isinstance(output, AsyncTPUModelRunnerOutput):
            output = output.get_output()
        sampled: dict[str, list[int]] = {}
        for req_id, row in output.req_id_to_index.items():
            if row >= len(output.sampled_token_ids):
                continue
            tokens = output.sampled_token_ids[row]
            if not tokens:
                continue
            sampled[req_id] = tokens
            self.tokens[req_id][-1].extend(tokens)
            want = self.sched.reqs[req_id].params.logprobs
            if want is not None:
                lists = output.logprobs
                cols = want + 1
                self.logprobs[req_id][-1].append(
                    (
                        lists.logprob_token_ids[row, :cols].tolist(),
                        lists.logprobs[row, :cols].view(np.int32).tolist(),
                        int(lists.sampled_token_ranks[row]),
                    )
                )
        self.sched.complete(out, sampled)
        return out

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()
