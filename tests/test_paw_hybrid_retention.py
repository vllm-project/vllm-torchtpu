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
"""``tpu_scheduler.hybrid_state_retention``: PrefixAwareScheduler keeps the
GDN state snapshots a later turn hits on, through the retention hooks of its
base (core/retention_scheduler.py)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import (
    prefix_aware_scheduler,
    retention_scheduler,
    scheduler_options,
)
from vllm_torchtpu.core.hybrid_retention import HybridStateRetention
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.core.retention_scheduler import HybridRetentionScheduler

ROLES = [None, "kv_producer", "kv_both", "kv_consumer"]
QWEN3_5_MOE = "Qwen3_5MoeForConditionalGeneration"


def _built(
    monkeypatch,
    role="kv_both",
    section=None,
    architecture=QWEN3_5_MOE,
    kv_cache_manager=None,
):
    """Runs PrefixAwareScheduler.__init__ over a stub base __init__ for an
    engine with KV role ``role`` serving ``architecture``, with ``section``
    as its tpu_scheduler options. The connector is None: retention does not
    read it."""
    kv_transfer_config = (
        None
        if role is None
        else KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=role)
    )

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            additional_config=None if section is None else {"tpu_scheduler": section},
            model_config=SimpleNamespace(architectures=[architecture]),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
        )
        self.connector = None
        self.kv_cache_manager = (
            MagicMock() if kv_cache_manager is None else kv_cache_manager
        )

    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    monkeypatch.setattr(PrefixAwareScheduler, "_enable_memo_xstep", lambda self: None)
    return PrefixAwareScheduler()


def _fake_retention(enabled=True):
    built = []

    class FakeRetention:
        def __init__(self, kvm):
            built.append(kvm)
            self.enabled = enabled

    return FakeRetention, built


# ---------------- parsing ----------------


def test_the_option_defaults_off():
    assert scheduler_options.parse({}).hybrid_state_retention is False
    assert scheduler_options.parse(None).hybrid_state_retention is False


def test_parse_reads_the_option():
    opts = scheduler_options.get(
        SimpleNamespace(
            additional_config={"tpu_scheduler": {"hybrid_state_retention": True}}
        )
    )
    assert opts.hybrid_state_retention is True
    assert opts.prefix_aware_levers is False


@pytest.mark.parametrize("value", [0, "true", None])
def test_the_option_takes_a_bool_only(value):
    with pytest.raises(ValueError, match="hybrid_state_retention must be a bool"):
        scheduler_options.parse({"hybrid_state_retention": value})


# ---------------- building ----------------


@pytest.mark.parametrize("role", ROLES)
def test_without_the_option_no_engine_builds_retention(monkeypatch, role):
    """A KV producer, for which ``retention_wanted`` holds, included: the
    pipeline chunk scheduler's rule does not apply to this scheduler."""
    fake, built = _fake_retention()
    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", fake)
    for section in (None, {"hybrid_state_retention": False}):
        s = _built(monkeypatch, role, section)
        assert s._retention is None
    assert built == []


@pytest.mark.parametrize("role", ROLES)
def test_the_option_builds_retention_on_every_engine_role(monkeypatch, role):
    fake, built = _fake_retention()
    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", fake)
    s = _built(monkeypatch, role, {"hybrid_state_retention": True})
    assert isinstance(s._retention, fake)
    assert built == [s.kv_cache_manager]


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3NextForCausalLM", "Qwen3_5ForConditionalGeneration", "JambaForCausalLM"],
)
def test_other_architectures_log_that_retention_is_off(
    monkeypatch, caplog, architecture
):
    fake, built = _fake_retention()
    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", fake)
    with caplog.at_level("INFO"):
        s = _built(
            monkeypatch, "kv_both", {"hybrid_state_retention": True}, architecture
        )
    assert s._retention is None and built == []
    assert "hybrid state retention is off" in caplog.text
    assert architecture in caplog.text


def test_a_retention_that_fails_to_build_is_logged_and_left_off(monkeypatch, caplog):
    def broken(kvm):
        raise AttributeError("single_type_managers")

    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", broken)
    with caplog.at_level("INFO"):
        s = _built(monkeypatch, "kv_both", {"hybrid_state_retention": True})
    assert s._retention is None
    assert "hybrid state retention is off: building it failed" in caplog.text


def test_a_retention_that_cannot_run_is_logged_and_left_off(monkeypatch, caplog):
    fake, built = _fake_retention(enabled=False)
    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", fake)
    with caplog.at_level("INFO"):
        s = _built(monkeypatch, "kv_both", {"hybrid_state_retention": True})
    assert s._retention is None and len(built) == 1
    assert "it needs prefix caching and a Mamba" in caplog.text


# ---------------- the hooks ----------------


class _Retention:
    """Records the retention calls and clips every chunk at ``stop``."""

    enabled = True

    def __init__(self, stop=None, fail=False, pinned=0):
        self.stop = stop
        self.fail = fail
        self.pinned = pinned
        self.calls: list[tuple] = []

    def pin_running(self, running):
        self.calls.append(("pin", tuple(r.request_id for r in running)))
        if self.fail:
            raise RuntimeError("pin")

    def begin_step(self):
        self.calls.append(("begin",))

    def prepare_prefill_split(self, request, start, num_new_tokens):
        self.calls.append(("prepare", request.request_id, start, num_new_tokens))
        if self.fail:
            raise RuntimeError("prepare")

    def adjust_prefill_split(self, request, start, aligned):
        self.calls.append(("adjust", request.request_id, start, aligned))
        if self.stop is not None and start < self.stop < start + aligned:
            return self.stop - start
        return aligned

    def snapshot_attention_blocks(self, request):
        self.calls.append(("snapshot", request.request_id))
        return ["attn"]

    def on_request_freed(self, request, attn_blocks):
        self.calls.append(("freed", request.request_id, attn_blocks))

    def num_pinned_blocks(self):
        return self.pinned

    def shutdown(self):
        self.calls.append(("shutdown",))


def _with_retention(monkeypatch, retention, role="kv_both", levers=False):
    fake, _ = _fake_retention()
    monkeypatch.setattr(prefix_aware_scheduler, "HybridStateRetention", fake)
    s = _built(
        monkeypatch,
        role,
        {"hybrid_state_retention": True, "prefix_aware_levers": levers},
    )
    s._retention = retention
    s.waiting = FCFSRequestQueue()
    s.kv_holding_waiting = FCFSRequestQueue()
    # No prefill checkpoint blocks: the backend writes no state inside a
    # chunk.
    s.mamba_has_prefill_checkpoint_blocks = False
    return s


def _decoding(request_id):
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=100,
        num_computed_tokens=101,
        status=RequestStatus.RUNNING,
    )


@pytest.mark.parametrize("levers", [False, True])
def test_schedule_pins_the_running_requests_first(monkeypatch, levers):
    retention = _Retention()
    s = _with_retention(monkeypatch, retention, levers=levers)
    s.running = [_decoding("d")]
    s._inflight_prefills = set()
    seen = []

    def fake_schedule(self, *args, **kwargs):
        seen.append(list(retention.calls))
        return SimpleNamespace(scheduled_new_reqs=[])

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule, raising=False)
    s._waiting_loop_stop_reason = lambda out, waiting: "idle"
    s._record_step_telemetry = lambda out, waiting, stop: None
    s._last_sched_log_t = float("inf")
    s.schedule()
    assert seen == [[("pin", ("d",)), ("begin",)]]


def test_the_split_applies_the_retention_stops_after_the_mamba_alignment(
    monkeypatch,
):
    retention = _Retention(stop=1024)
    s = _with_retention(monkeypatch, retention)
    request = SimpleNamespace(request_id="r", num_computed_tokens=0)
    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", lambda *a: 3840)
    assert s._mamba_block_aligned_split(request, 8192, 256) == 1024 - 256
    assert retention.calls == [
        ("prepare", "r", 256, 8192),
        ("adjust", "r", 256, 3840),
    ]


def test_the_split_starts_after_the_local_and_external_hits(monkeypatch):
    retention = _Retention(stop=4096)
    s = _with_retention(monkeypatch, retention)
    request = SimpleNamespace(request_id="r", num_computed_tokens=512)
    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", lambda *a: 3840)
    # Starts at 512 computed + 1024 local + 2048 external tokens.
    assert s._mamba_block_aligned_split(request, 8192, 1024, 2048) == 4096 - 3584
    assert retention.calls == [
        ("prepare", "r", 3584, 8192),
        ("adjust", "r", 3584, 3840),
    ]


@pytest.mark.parametrize(
    ("checkpoint_blocks", "num_tokens", "written"),
    [
        # The chunk ends the prefill: the backend writes the state at the
        # last boundary less the dropped EAGLE block.
        (True, 635, 2304),
        # A chunk short of the prefill end writes nothing inside, though it
        # spans that boundary.
        (True, 512, 0),
        (False, 635, 0),
    ],
)
def test_where_the_backend_writes_the_state_inside_a_chunk(
    monkeypatch, checkpoint_blocks, num_tokens, written
):
    s = _with_retention(monkeypatch, _Retention())
    s.mamba_has_prefill_checkpoint_blocks = checkpoint_blocks
    s.hash_block_size = s.block_size = 256
    s.cache_config = SimpleNamespace(block_size=256)
    s.mamba_partial_cache_hit = False
    s.use_eagle_block_drop = True
    s.num_hash_lookahead_tokens = 0
    s.mamba_prefill_checkpoint_alignment = 256
    request = SimpleNamespace(num_prompt_tokens=2683, num_tokens=2683)
    assert s._in_chunk_checkpoint(request, 2048, num_tokens) == written


def test_without_retention_the_split_is_vllms(monkeypatch):
    s = _built(monkeypatch, "kv_both")
    seen = []

    def base(self, request, num_new_tokens, local=0, external=0):
        seen.append((num_new_tokens, local, external))
        return 640

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", base)
    s._retention_split = lambda *a: pytest.fail("retention split")
    request = SimpleNamespace(request_id="r", num_computed_tokens=0)
    assert s._mamba_block_aligned_split(request, 8192, 128, 64) == 640
    assert seen == [(8192, 128, 64)]


def test_freeing_a_request_snapshots_its_attention_blocks_first(monkeypatch):
    """The snapshot reads the request's blocks, which vLLM's free drops."""
    retention = _Retention()
    s = _with_retention(monkeypatch, retention)
    request = SimpleNamespace(request_id="f")

    def base(self, req):
        retention.calls.append(("base", req.request_id))

    monkeypatch.setattr(Scheduler, "_free_request_blocks", base)
    s._free_request_blocks(request)
    assert retention.calls == [
        ("snapshot", "f"),
        ("base", "f"),
        ("freed", "f", ["attn"]),
    ]


def test_a_request_the_connector_still_holds_is_pinned(monkeypatch):
    retention = _Retention()
    s = _with_retention(monkeypatch, retention)
    request = SimpleNamespace(request_id="c")
    with patch.object(Scheduler, "_connector_finished", return_value=(True, None)):
        assert s._connector_finished(request) == (True, None)
    with patch.object(Scheduler, "_connector_finished", return_value=(False, None)):
        s._connector_finished(request)
    assert retention.calls == [("pin", ("c",))]


def test_the_free_block_count_counts_the_pinned_blocks(monkeypatch):
    s = _built(monkeypatch, "kv_both")
    s.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 10)
    )
    assert s._get_num_free_blocks() == 10
    s._retention = _Retention(pinned=5)
    assert s._get_num_free_blocks() == 15


def test_a_failing_retention_is_turned_off_and_the_split_goes_on(monkeypatch):
    retention = _Retention(fail=True)
    s = _with_retention(monkeypatch, retention)
    request = SimpleNamespace(request_id="r", num_computed_tokens=0)
    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", lambda *a: 2048)
    with patch.object(retention_scheduler.logger, "exception") as log:
        assert s._mamba_block_aligned_split(request, 4096) == 2048
    assert s._retention is None
    # It releases its blocks; its log line names this scheduler.
    assert retention.calls == [("prepare", "r", 0, 4096), ("shutdown",)]
    assert log.call_args_list[0].args[1] == "PrefixAwareScheduler"


def test_the_scheduler_runs_the_base_hooks():
    for name in (
        "_retention_failed",
        "_retention_begin_step",
        "_connector_finished",
        "_free_request_blocks",
        "_retention_split",
    ):
        hook = getattr(HybridRetentionScheduler, name)
        assert getattr(PrefixAwareScheduler, name) is hook, name
    assert issubclass(PrefixAwareScheduler, AsyncScheduler)


# ---------------- on vLLM's own KV cache manager ----------------

_BS = 16


def _manager(num_blocks: int, mamba: bool = True) -> KVCacheManager:
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=_BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
    ]
    if mamba:
        spec = MambaSpec(
            block_size=_BS,
            shapes=(1, 1),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        groups.append(KVCacheGroupSpec(["mamba"], spec))
    config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=_BS,
        hash_block_size=_BS,
        enable_caching=True,
    )


def _request(request_id: str, token_ids) -> Request:
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64)
    params.update_from_generation_config({}, eos_token_id=100_000)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(_BS, sha256),
    )


def _engine(monkeypatch, kvm, section):
    s = _built(monkeypatch, "kv_both", section, kv_cache_manager=kvm)
    s.waiting = FCFSRequestQueue()
    s.kv_holding_waiting = FCFSRequestQueue()
    s.running = []
    # vLLM's _free_request_blocks frees at once without deferred frees.
    s.defer_block_free = False
    return s


def _run_turn(s, kvm, request_id, prompt, decode_tokens):
    """Prefills ``prompt`` in block-aligned chunks, decodes with a
    ``schedule()`` call per step (which pins the running requests'
    snapshots), then frees the request through the scheduler. Returns the
    boundary state block."""
    req = _request(request_id, prompt)
    boundary = (len(prompt) // _BS) * _BS
    for chunk in (boundary, len(prompt) - boundary):
        kvm.new_step_starts()
        assert kvm.allocate_slots(req, chunk) is not None
        req.num_computed_tokens += chunk
    mamba_blocks = kvm.coordinator.single_type_managers[1].req_to_blocks[request_id]
    state = mamba_blocks[boundary // _BS - 1]
    s.running = [req]
    for i in range(decode_tokens):
        req.append_output_token_ids(5000 + i)
        kvm.new_step_starts()
        assert kvm.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        s.schedule()
    s.running = []
    s._free_request_blocks(req)
    return state


def _evict_one_and_look_up(kvm, prompt):
    """Allocates every free block up to the first one holding a cached
    prefix, frees them, and looks up a request that shares ``prompt``.
    Returns the evicted block and the hit length."""
    pool = kvm.block_pool
    queue = pool.free_block_queue
    block, position = queue.fake_free_list_head.next_free_block, 0
    while block.block_hash is None:
        block, position = block.next_free_block, position + 1
    pool.free_blocks(pool.get_new_blocks(position + 1))
    kvm.new_step_starts()
    nxt = _request("next", prompt + list(range(7000, 7020)))
    return block, kvm.get_computed_blocks(nxt)[1]


@pytest.mark.parametrize("on", [False, True])
def test_on_a_kv_both_engine_the_next_turn_hits_the_kept_state(monkeypatch, on):
    """Allocation pressure evicts the released boundary state first without
    the option; with it the state stays pinned, the next turn hits it, and
    the pinned block counts as free."""
    monkeypatch.setattr(AsyncScheduler, "schedule", lambda self, *a, **k: None)
    kvm = _manager(48)
    s = _engine(monkeypatch, kvm, {"hybrid_state_retention": on})
    assert isinstance(s._retention, HybridStateRetention) is on
    prompt = list(range(1000, 1088))
    state = _run_turn(s, kvm, "turn0", prompt, decode_tokens=40)
    victim, hit = _evict_one_and_look_up(kvm, prompt)
    pool_free = kvm.block_pool.get_num_free_blocks()
    if on:
        assert victim is not state and hit == 80
        assert s._retention is not None and s._retention.num_pinned_blocks() == 1
        assert s._get_num_free_blocks() == pool_free + 1
    else:
        assert victim is state and hit == 0
        assert s._get_num_free_blocks() == pool_free


def test_a_model_without_a_mamba_group_runs_without_retention(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        s = _engine(
            monkeypatch, _manager(48, mamba=False), {"hybrid_state_retention": True}
        )
    assert s._retention is None
    assert "hybrid state retention is off" in caplog.text


# ---------------- on vLLM's own scheduler ----------------

_CKPT_BS = 256


def _real_scheduler(tmp_path, monkeypatch, retention: bool, checkpoint_blocks: int):
    """A PrefixAwareScheduler over vLLM's own scheduler and KV cache manager
    with a Mamba "align" group of ``checkpoint_blocks`` prefill checkpoint
    blocks (the backend then writes the prompt's last boundary state inside
    the final chunk). The tiny model stands in for a retention architecture."""
    model = tmp_path / "tiny-opt"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 32768,
                "vocab_size": 50272,
                "torch_dtype": "float16",
                "word_embed_proj_dim": 64,
                "do_layer_norm_before": True,
            }
        )
    )
    monkeypatch.setattr(
        prefix_aware_scheduler, "RETENTION_ARCHITECTURES", frozenset({"OPTForCausalLM"})
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            max_model_len=32768,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model=str(model), dtype="float16", seed=42, skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(
            block_size=_CKPT_BS,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        device_config=DeviceConfig("cpu"),
        additional_config={"tpu_scheduler": {"hybrid_state_retention": retention}},
    )
    num_blocks = 400
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=_CKPT_BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=_CKPT_BS,
                shapes=(1, 1),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_prefill_checkpoint_blocks=checkpoint_blocks,
                prefill_checkpoint_alignment=_CKPT_BS if checkpoint_blocks else None,
            ),
        ),
    ]
    s = PrefixAwareScheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=_CKPT_BS,
        log_stats=True,
    )
    assert s.mamba_has_prefill_checkpoint_blocks is bool(checkpoint_blocks)
    assert (s._retention is not None) is retention
    return s


def _turn_chunks(s, request_id: str, prompt: list[int]) -> list[int]:
    """Runs a request from admission to finish (4 output tokens) and returns
    its prefill chunks."""
    init_none_hash(sha256)
    s.add_request(
        Request(
            request_id=request_id,
            prompt_token_ids=prompt,
            sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
            pooling_params=None,
            block_hasher=get_request_block_hasher(_CKPT_BS, sha256),
        )
    )
    chunks = []
    while request_id in s.requests:
        request = s.requests[request_id]
        prefilling = request.num_computed_tokens < request.num_prompt_tokens
        out = s.schedule()
        ids = list(out.num_scheduled_tokens)
        if prefilling:
            chunks.append(out.num_scheduled_tokens.get(request_id, 0))
        s.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=[
                    [] if s.requests[rid].is_prefill_chunk else [0] for rid in ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
            ),
        )
    return chunks


@pytest.mark.parametrize(
    ("retention", "checkpoint_blocks", "turn0", "turn1"),
    [
        # The backend writes the boundary state inside the final chunk: a warm
        # turn prefills in one chunk with retention as without it. The cold
        # turn still stops at its fork junction (8 blocks - 1).
        (False, 1, [2148], [635]),
        (True, 1, [1792, 356], [635]),
        # Without checkpoint blocks the chunk ends at the boundary either way.
        (False, 0, [2048, 100], [512, 123]),
        (True, 0, [1792, 256, 100], [512, 123]),
    ],
)
def test_a_warm_turn_is_not_split_where_the_backend_writes_the_state(
    tmp_path, monkeypatch, retention, checkpoint_blocks, turn0, turn1
):
    s = _real_scheduler(tmp_path, monkeypatch, retention, checkpoint_blocks)
    prompt = list(range(1, 2149))
    assert _turn_chunks(s, "turn0", prompt) == turn0
    # The next turn hits the 2048-token boundary and has 635 tokens to go;
    # its last boundary is 2560.
    nxt = prompt[:2048] + list(range(9000, 9635))
    assert _turn_chunks(s, "turn1", nxt) == turn1
    if retention:
        # The retention keeps the boundary state the backend wrote.
        kept = {e.req_id: e.hit for e in s._retention._conv_lru.values()}
        assert kept == {"turn0": 2048, "turn1": 2560}
