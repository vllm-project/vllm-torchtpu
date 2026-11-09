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
"""Real CPU scheduler probes plus deterministic offline model invariants."""

import ast
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cpu_test


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def simulator():
    return load_file(
        "dsv41_simulator", ROOT / "scripts/dsv41_bench/simulate_prefill.py"
    )


@pytest.fixture(scope="module")
def planner(simulator):
    return simulator.load_replay_plan(ROOT)


def test_ced_mixed_batch_uses_middle_bucket(planner):
    counts = [6] * 47 + [1300]
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    metadata = SimpleNamespace(
        query_start_loc_cpu=tuple(bounds),
        seq_lens_cpu=tuple(counts),
        prompt_lens_cpu=(0,) * 47 + (1300,),
        replay_starts_cpu=(0,) * 48,
    )
    plan = planner(metadata, 128, 2048, (288, 512, 2048))
    assert len(plan.indices) == 47 * 6 + 128
    assert plan.padded_num_tokens == 512
    assert planner(metadata, 128, 2048, (288, 2048)).padded_num_tokens == 2048


def test_ced_middle_chunk_needs_no_decoder_rows(planner):
    metadata = SimpleNamespace(
        query_start_loc_cpu=(0, 224),
        seq_lens_cpu=(224,),
        prompt_lens_cpu=(1300,),
        replay_starts_cpu=(0,),
    )
    assert planner(metadata, 128, 512, (288, 512, 2048)).padded_num_tokens == 0


@pytest.mark.parametrize(
    "interval,window,budget", [(1, 0, 2048), (4, 0, 2048), (4, 1, 2048), (8, 2, 512)]
)
def test_model_conserves_work_with_long_chunks_and_short_outputs(
    simulator, planner, interval, window, budget
):
    chains = {
        0: [simulator.Turn(0, 0, 5000, 1), simulator.Turn(0, 0, 128, 19)],
        1: [simulator.Turn(1, 1, 1300, 77)],
    }
    result = simulator.simulate(
        chains,
        simulator.Policy(
            "probe", interval=interval, strict_window=window, token_budget=budget
        ),
        2,
        3.15,
        planner,
    )
    assert result["completed_turns"] == 3
    assert round(result["output_tokens"]) == 97
    assert result["prefill_tokens"] == 6428
    assert result["steps"] == sum(result["stage_bucket_steps"].values())
    if budget == 512:
        assert all(not key.startswith("2048/") for key in result["stage_bucket_steps"])


def test_adding_bucket_preserves_admission_and_output(simulator, planner):
    chains = {i: [simulator.Turn(i, i % 2, 1300, 300)] for i in range(96)}
    old = simulator.simulate(
        chains, simulator.Policy("old", buckets=(288, 2048)), 2, 3.15, planner
    )
    new = simulator.simulate(chains, simulator.Policy("new"), 2, 3.15, planner)
    for key in (
        "steps",
        "output_tokens",
        "prefill_tokens",
        "mean_admission_wait_steps",
    ):
        assert old[key] == new[key]
    assert "2048/512" in new["stage_bucket_steps"]


def test_surrogate_output_scaling_respects_recorded_request_limit(simulator):
    data = {
        "summary": {
            "metrics_steady": {
                "prompt_tokens_s": 128,
                "prefix_cache_hit_rate": 0.0,
                "seconds": 1.0,
                "generation_tokens_s": 10,
            },
            "responses_steady": {"requests": 1},
            "source": {"max_tokens": {"32": 1}},
        },
        "requests": [
            {
                "slot": 0,
                "rank": 0,
                "outcome": "completed",
                "trajectory": 0,
                "turn": 0,
                "started": 0.0,
                "prompt_tokens": 128,
                "completion_tokens": 10,
                "cached_tokens": None,
            }
        ],
    }
    chains, inputs = simulator.workload(data, 1, "fixed", output_scale=100.0)
    assert inputs["maximum_output_tokens"] == 32
    assert all(turn.output == 32 for chain in chains.values() for turn in chain)


def test_bucket_routes_trace_real_kernels(tmp_path):
    out = tmp_path / "routes.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/dsv41_bench/audit_bucket_routes.py"),
            "--trace",
            "--output",
            str(out),
        ],
        check=True,
    )
    rows = {row["tokens"]: row for row in json.loads(out.read_text())["rows"]}
    for tokens in (288, 512, 2048):
        assert rows[tokens]["fused_output_projection"]
        assert rows[tokens]["shifted_mhc"]
        assert all(graph["pallas_call"] for graph in rows[tokens]["jaxpr"].values())
    assert rows[512]["sorted_fp8_index_write"]
    assert not rows[512]["qa"]["tuned"]
    assert not any(rows[512]["metadata_fusions"].values())


@pytest.fixture
def cpu_scheduler(tmp_path, monkeypatch):
    import torch
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "vocab_size": 128,
                "hidden_size": 64,
                "ffn_dim": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "max_position_embeddings": 8192,
                "word_embed_proj_dim": 64,
                "torch_dtype": "float32",
            }
        )
    )
    model = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        seed=42,
        max_model_len=8192,
        skip_tokenizer_init=True,
    )
    upstream = SpeculativeConfig.max_num_new_slots_for_drafting.fget
    if getattr(upstream, "_tpu_drafting_slots_patch", False):
        upstream = upstream.__wrapped__
    monkeypatch.setattr(
        SpeculativeConfig, "max_num_new_slots_for_drafting", property(upstream)
    )

    def build(*, budget=2048, seqs=48, speculative=False, strict=False):
        config = VllmConfig(
            model_config=model,
            scheduler_config=SchedulerConfig(
                max_num_seqs=seqs,
                max_num_batched_tokens=budget,
                max_model_len=8192,
                enable_chunked_prefill=True,
                is_encoder_decoder=False,
                watermark=0.0,
                async_scheduling=True,
            ),
            cache_config=CacheConfig(
                block_size=16, gpu_memory_utilization=0.9, enable_prefix_caching=True
            ),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
        if speculative:
            # Exercise the real property/methods without loading a draft model.
            spec = object.__new__(SpeculativeConfig)
            spec.method = "dspark"
            spec.num_speculative_tokens = 5
            spec.parallel_drafting = True
            spec.disable_eagle_block_drop = False
            spec.num_speculative_tokens_per_batch_size = None
            spec.draft_model_config = SimpleNamespace(
                hf_config=SimpleNamespace(sample_from_anchor=True)
            )
            config.speculative_config = spec
        config.cache_config.num_gpu_blocks = 4096
        kv = KVCacheConfig(
            num_blocks=4096,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer0"],
                    FullAttentionSpec(
                        block_size=16,
                        num_kv_heads=1,
                        head_size=64,
                        dtype=torch.float32,
                    ),
                ),
            ],
        )
        register_all_kvcache_specs(config)
        module = load_file(
            "dsv41_tpu_scheduler", ROOT / "src/vllm_torchtpu/core/tpu_scheduler.py"
        )
        scheduler_cls = (
            module.TpuPrefillWindowScheduler if strict else module.TpuDpScheduler
        )
        scheduler = scheduler_cls(
            vllm_config=config,
            kv_cache_config=kv,
            structured_output_manager=StructuredOutputManager(config),
            block_size=16,
        )
        init_none_hash(sha256)
        return scheduler, get_request_block_hasher(16, sha256)

    return build


def add_request(scheduler, hasher, name, tokens):
    from vllm import SamplingParams
    from vllm.v1.request import Request

    params = SamplingParams(max_tokens=512, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    scheduler.add_request(
        Request(name, [7] * tokens, params, None, block_hasher=hasher)
    )


def install_budget_patch():
    path = ROOT / "src/vllm_torchtpu/__init__.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_patch_dflash_drafting_slot_budget"
    )
    namespace = {"logger": logging.getLogger(__name__)}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    namespace[node.name]()


def test_real_scheduler_continues_prefill_on_throttled_step(cpu_scheduler):
    scheduler, hasher = cpu_scheduler()
    add_request(scheduler, hasher, "long", 3000)
    assert scheduler.schedule(False).num_scheduled_tokens["long"] == 2048
    assert scheduler.schedule(True).num_scheduled_tokens["long"] > 0


def test_real_scheduler_drain_admits_arrival_on_throttled_step(cpu_scheduler):
    scheduler, hasher = cpu_scheduler()
    add_request(scheduler, hasher, "long", 3000)
    scheduler.schedule(False)
    add_request(scheduler, hasher, "new", 128)
    assert scheduler.schedule(True).num_scheduled_tokens["new"] > 0


def test_strict_window_holds_chunks_and_new_requests_until_aligned(cpu_scheduler):
    scheduler, hasher = cpu_scheduler(strict=True)
    add_request(scheduler, hasher, "long", 3000)
    assert scheduler.schedule(False).num_scheduled_tokens["long"] == 2048
    request = scheduler.requests["long"]
    fence = request.next_decode_eligible_step
    computed = request.num_computed_tokens
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    add_request(scheduler, hasher, "new", 128)
    for _ in range(7):
        assert not scheduler.schedule(True).num_scheduled_tokens
        assert request.num_computed_tokens == computed
        assert request.next_decode_eligible_step == fence
        assert (
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
        )
    aligned = scheduler.schedule(False)
    assert aligned.num_scheduled_tokens["long"] > 0
    assert aligned.num_scheduled_tokens["new"] > 0


@pytest.mark.parametrize("patched", [False, True])
def test_strict_window_preserves_six_row_k5_decodes(cpu_scheduler, patched):
    if patched:
        install_budget_patch()
    scheduler, hasher = cpu_scheduler(strict=True, speculative=True)
    for i in range(47):
        add_request(scheduler, hasher, f"decode-{i}", 1)
    scheduler.schedule(False)
    add_request(scheduler, hasher, "prefill", 5000)
    scheduler.schedule(False)
    for _ in range(7):
        output = scheduler.schedule(True)
        assert "prefill" not in output.num_scheduled_tokens
        assert all(output.num_scheduled_tokens[f"decode-{i}"] == 6 for i in range(47))
    assert scheduler.schedule(False).num_scheduled_tokens["prefill"] > 0


def test_strict_window_restores_existing_fence_and_pause_on_error(
    cpu_scheduler, monkeypatch
):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    scheduler, hasher = cpu_scheduler(strict=True)
    add_request(scheduler, hasher, "long", 3000)
    scheduler.schedule(False)
    request = scheduler.requests["long"]
    request.next_decode_eligible_step = 59
    pause = scheduler.pause_state

    def fail(*args):
        raise RuntimeError("probe")

    monkeypatch.setattr(AsyncScheduler, "schedule", fail)
    with pytest.raises(RuntimeError, match="probe"):
        scheduler.schedule(True)
    assert request.next_decode_eligible_step == 59
    assert scheduler.pause_state == pause


def test_drafting_budget_backport_frees_prefill_capacity(cpu_scheduler):
    scheduled = []
    for patched in (False, True):
        if patched:
            install_budget_patch()
            install_budget_patch()  # Registration is idempotent.
        scheduler, hasher = cpu_scheduler(budget=512, speculative=True)
        for i in range(47):
            add_request(scheduler, hasher, f"decode-{i}", 1)
        scheduler.schedule(False)
        add_request(scheduler, hasher, "prefill", 1300)
        output = scheduler.schedule(False)
        assert all(output.num_scheduled_tokens[f"decode-{i}"] == 6 for i in range(47))
        scheduled.append(output.num_scheduled_tokens["prefill"])
    assert scheduled == [38, 230]


@pytest.mark.parametrize(
    "method,parallel",
    [
        ("dflash", True),
        ("dspark", True),
        ("eagle3", True),
        ("mtp", False),
        ("draft_model", False),
        ("ngram", False),
    ],
)
def test_budget_patch_only_changes_separate_forward_drafters(
    cpu_scheduler, method, parallel
):
    from vllm.config import SpeculativeConfig

    spec = object.__new__(SpeculativeConfig)
    spec.method = method
    spec.num_speculative_tokens = 5
    spec.parallel_drafting = parallel
    before = spec.max_num_new_slots_for_drafting
    install_budget_patch()
    expected = 0 if method in ("dflash", "dspark") else before
    assert spec.max_num_new_slots_for_drafting == expected


@pytest.mark.parametrize("target", [[288, 2048], [288, 512, 2048], [128]])
def test_epic_draft_query_buckets_stay_independent_of_target(target):
    path = ROOT / "src/vllm_torchtpu/spec_decode/dflash.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DFlashProposer"
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_draft_token_paddings"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    proposer = SimpleNamespace(
        block_size=5,
        runner=SimpleNamespace(
            num_tokens_paddings=target,
            num_reqs_paddings=[8, 16, 32, 64],
            num_reqs_max_model_len=48,
            num_reqs_most_model_len=None,
        ),
    )
    assert namespace[node.name](proposer) == [40, 80, 160, 240]
