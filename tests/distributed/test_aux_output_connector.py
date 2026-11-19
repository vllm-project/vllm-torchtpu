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
"""vLLM's auxiliary-output connectors on the TPU routed-experts capturer."""

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from itertools import takewhile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
)
from vllm.platforms import current_platform

from tests.layers.adapter.routed_experts_test_utils import (
    BLOCK_SIZE,
    NUM_LAYERS,
    PADDING,
    TOKEN_PADDINGS,
    TOPK,
    expert_ids,
    make_capturer,
    make_moe_model,
    make_vllm_connector,
    model_forward,
    use_host_events,
)
from vllm_torchtpu.distributed import aux_output_connector

ROW_SHAPE = (NUM_LAYERS, TOPK)
# A rejected draft's row; every known row is below it.
REJECTED = PADDING - 1


@pytest.fixture(autouse=True)
def on_host(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    use_host_events(monkeypatch)


def test_every_layer_writes_the_given_capturer(monkeypatch):
    """vLLM binds its own capturer first; `use_capturer` rebinds each layer."""
    capturer = make_capturer()
    model = make_moe_model()
    make_vllm_connector(monkeypatch, capturer, model).close()
    ids = expert_ids(16)

    model_forward(model, ids)

    assert torch.equal(capturer.check_forward(16).to(torch.int32), ids)


@pytest.mark.parametrize("other", ["dtype", "shape"])
def test_a_capturer_of_other_rows_is_refused(monkeypatch, other):
    """vLLM sized the connector's rows from the capturer it built."""
    capturer = make_capturer(num_experts=256 if other == "dtype" else 512)
    if other == "shape":
        capturer.shape_per_token = (ROW_SHAPE[0], ROW_SHAPE[1] + 1)
    model = make_moe_model()
    connector = make_vllm_connector(monkeypatch, None, model)
    try:
        with pytest.raises(ValueError, match="differ"):
            aux_output_connector.use_capturer(connector, model, capturer)
    finally:
        connector.close()


def test_rejected_rows_follow_the_schedulers_rollback():
    """No drafts, all rejected, some, none, and nothing sampled."""
    num_drafts = np.array([0, 3, 3, 3, 3])
    num_sampled = np.array([1, 1, 2, 4, 0])

    rejected = aux_output_connector.num_rejected_rows(num_drafts, num_sampled)

    assert rejected.tolist() == [0, 3, 2, 0, 0]


def _hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).digest()


def _known_row(tokens):
    """The row of a position, a function of the tokens up to it."""
    base = int.from_bytes(_hash(tokens)[:4], "little")
    return ((base + np.arange(np.prod(ROW_SHAPE))) % REJECTED).reshape(ROW_SHAPE)


@dataclass
class _Request:
    """What vLLM's scheduler connector reads, plus the simulated scheduler's
    counters."""

    request_id: str
    prompt: list[int]
    max_tokens: int
    seed: int
    out: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_output_placeholders: int = 0
    num_spec_tokens: int = 0
    finished: bool = False
    rows: list[np.ndarray] = field(default_factory=list)
    sampling_params: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(routed_experts_prompt_start=0)
    )

    @property
    def num_tokens(self):
        return len(self.prompt) + len(self.out)

    @property
    def num_prompt_tokens(self):
        return len(self.prompt)

    @property
    def num_output_tokens(self):
        return len(self.out)

    @property
    def block_hashes(self):
        tokens = self.prompt + self.out
        ends = range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
        return [_hash(tokens[:end]) for end in ends]

    def is_finished(self):
        return self.finished

    def token(self, position):
        """The token the request holds, or will hold, at `position`."""
        if position < len(self.prompt):
            return self.prompt[position]
        return (self.seed * 7919 + position * 104729) % 50000


def _serve(monkeypatch, seed, num_spec, flaw=None):
    """Serves 8 requests through vLLM's scheduler and worker connectors in
    TPU steps, and returns them finished: async scheduling with two steps in
    flight, two requests per forward, every forward's table taken into the
    step's snapshot, `num_spec` drafts per decode, and repeated prompts that
    hit the prefix cache.

    `flaw` breaks one step on purpose: "zero_rejected" keeps rejected draft
    rows, "padded_rows" takes each forward's padding rows as real,
    "no_wait" hands a step's rows over before its snapshot has waited,
    "vllm_capturer" keeps vLLM's own capturer.
    """
    rng = random.Random(seed)
    capturer = make_capturer()
    model = make_moe_model()
    connector = make_vllm_connector(
        monkeypatch, None if flaw == "vllm_capturer" else capturer, model
    )
    cached = set()
    pool = SimpleNamespace(
        get_cached_block=lambda h, _: [object()] if h in cached else None,
        hash_block_size=BLOCK_SIZE,
    )
    manager = SimpleNamespace(block_size=BLOCK_SIZE, block_pool=pool)
    scheduler = AuxOutputSchedulerConnector(
        SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=[manager]),
            block_pool=pool,
            kv_cache_config=SimpleNamespace(num_blocks=1024),
        )
    )
    prompts = [[rng.randrange(50000) for _ in range(n)] for n in (5, 9, 18)]
    arrivals = deque(
        _Request(f"r{i}", list(prompts[i % 3]), rng.choice([1, 2, 5, 14]), i)
        for i in range(8)
    )
    running, device_ends, in_flight, finished = {}, {}, deque(), []

    def finish_output(step):
        """The output thread: hand the step's counts to vLLM's connector."""
        if "outputs" not in step:
            num_sampled = np.array([len(step["sampled"][r]) for r in step["ids"]])
            num_drafts = np.array([step["drafts"][r] for r in step["ids"]])
            rejected = aux_output_connector.num_rejected_rows(num_drafts, num_sampled)
            if flaw == "zero_rejected":
                rejected[:] = 0
            if flaw != "no_wait":
                step["snapshot"].wait()
            step["pending"].enqueue_cpu_copy(num_sampled, rejected)
            step["outputs"] = step["pending"].process_output()

    def update_from_output(step):
        """vLLM's scheduler: roll back rejected drafts, append sampled tokens."""
        for rid in step["ids"]:
            req, sampled = running.get(rid), step["sampled"][rid]
            if req is None or not sampled:
                continue
            if step["drafts"][rid]:
                num_rejected = step["drafts"][rid] - (len(sampled) - 1)
                req.num_computed_tokens -= num_rejected
                req.num_output_placeholders -= num_rejected
            for token in sampled[: req.max_tokens - len(req.out)]:
                req.out.append(token)
                req.num_output_placeholders -= 1
            req.finished = len(req.out) == req.max_tokens
            rows = scheduler.take_output(req, step["outputs"])
            if rows is not None:
                req.rows.append(rows.copy())
            # A block is a prefix hit once its step's output is in, never in
            # the step that writes it (as on TPU).
            cached.update(req.block_hashes)
            if req.finished:
                scheduler.request_finished(req)
                finished.append(running.pop(rid))

    def schedule():
        """vLLM's async scheduler: placeholders stand for unsampled tokens."""
        for _ in range(rng.choice([0, 1, 1, 2])):
            if arrivals and len(running) < 6:
                req = arrivals.popleft()
                running[req.request_id] = req
                kv_blocks = len(list(takewhile(cached.__contains__, req.block_hashes)))
                hit = min(kv_blocks * BLOCK_SIZE, scheduler.max_cache_hit(req))
                req.num_computed_tokens = hit - hit % BLOCK_SIZE
                device_ends[req.request_id] = req.num_computed_tokens
        num_new, drafts, starts = {}, {}, {}
        for rid, req in running.items():
            end = req.num_tokens + req.num_output_placeholders
            n = end + req.num_spec_tokens - req.num_computed_tokens
            if n > 0:
                if req.num_computed_tokens < req.num_prompt_tokens:
                    n = min(n, 6)
                num_new[rid], starts[rid] = n, req.num_computed_tokens
                drafts[rid] = max(req.num_computed_tokens + n - end, 0)
        metadata = scheduler.build_connector_meta(
            SimpleNamespace(num_scheduled_tokens=num_new), running
        )
        for rid, n in num_new.items():
            req = running[rid]
            req.num_computed_tokens += n
            if req.num_computed_tokens >= req.num_tokens + req.num_output_placeholders:
                req.num_output_placeholders += 1 + drafts[rid]
                req.num_spec_tokens = num_spec
        return metadata, num_new, drafts, starts

    def forward(rids, num_new, drafts, sampled):
        """One forward padded to a bucket; drafts are accepted at random."""
        num_rows = sum(num_new[r] for r in rids)
        bucket = next(b for b in TOKEN_PADDINGS if b >= num_rows)
        ids = np.full((bucket, *ROW_SHAPE), PADDING, dtype=np.int32)
        row = 0
        for rid in rids:
            req, start, n, k = running[rid], device_ends[rid], num_new[rid], drafts[rid]
            kept = 1 + rng.randint(0, k) if k else n
            for i in range(kept):
                ids[row + i] = _known_row([req.token(p) for p in range(start + i + 1)])
            ids[row + kept : row + n] = REJECTED
            row += n
            device_ends[rid] = end = start + kept
            first = start + 1 if k else end
            produces = end >= req.num_prompt_tokens
            sampled[rid] = [req.token(p) for p in range(first, end + 1) if produces]
        model_forward(model, torch.from_numpy(ids))
        return bucket, num_rows

    try:
        while arrivals or running or in_flight:
            metadata, num_new, drafts, starts = schedule()
            if in_flight and rng.random() < 0.5:
                finish_output(in_flight[-1])
            connector.begin_step(metadata)
            if num_new:
                # The runner reorders the batch.
                rids = rng.sample(list(num_new), len(num_new))
                sampled = {}
                for lo in range(0, len(rids), 2):
                    bucket, num_rows = forward(
                        rids[lo : lo + 2], num_new, drafts, sampled
                    )
                    capturer.take_forward(
                        bucket, bucket if flaw == "padded_rows" else num_rows
                    )
                batch = SimpleNamespace(
                    req_ids=rids,
                    query_start_loc_np=np.cumsum([0, *(num_new[r] for r in rids)]),
                    num_computed_tokens_np=np.array([starts[r] for r in rids]),
                )
                pending = connector.prepare_output(batch)
                in_flight.append(
                    dict(
                        ids=rids,
                        pending=pending,
                        snapshot=capturer.take_snapshot(),
                        sampled=sampled,
                        drafts=drafts,
                    )
                )
            # The engine waits for the oldest step when two are in flight or
            # nothing new was scheduled.
            if len(in_flight) == 2 or (not num_new and in_flight):
                oldest = in_flight.popleft()
                finish_output(oldest)
                update_from_output(oldest)
    finally:
        connector.close()
    return finished


def _assert_known_rows(requests):
    assert len(requests) == 8
    for req in requests:
        tokens = req.prompt + req.out
        expected = np.stack([_known_row(tokens[:p]) for p in range(1, len(tokens))])
        assert np.array_equal(np.concatenate(req.rows), expected), (
            f"{req.request_id}: rows differ from the known answer"
        )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_every_request_row_matches_the_known_answer(monkeypatch, num_spec):
    """Every row a request gets back, computed or read from the prefix cache,
    is the row its position had; padding and rejected drafts never show."""
    for seed in range(20):
        _assert_known_rows(_serve(monkeypatch, seed, num_spec))


@pytest.mark.parametrize(
    ("flaw", "num_spec", "message"),
    [
        ("zero_rejected", 3, "capture moved backwards"),
        ("padded_rows", 0, "took .* rows for a step of"),
        ("no_wait", 0, "known answer"),
        ("vllm_capturer", 0, r"routed-experts table of \(0, "),
    ],
)
def test_a_flawed_step_fails(monkeypatch, flaw, num_spec, message):
    """Each way a step can go wrong fails a check: the known-answer check,
    or one of the capture's own."""
    with pytest.raises(AssertionError, match=message):
        _assert_known_rows(_serve(monkeypatch, 0, num_spec, flaw))
