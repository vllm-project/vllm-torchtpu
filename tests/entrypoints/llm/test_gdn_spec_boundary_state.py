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
"""The GDN state cached for a decode-filled block under MTP is the state at
the block's boundary.

With prefix caching in align mode and dense retention, the scheduler hashes
the state block that boundary T completes once T tokens are confirmed, and a
prefix hit of T tokens resumes from it. Requests decode across T with MTP;
the state in each hashed block is then compared with fresh prefills of T - d
tokens and must be closest to d = 0, and the boundary copies must compile
nothing after warmup. Dummy weights keep the model small, and the synthetic
rejection sampler fixes the acceptance pattern.

Run with: pytest tests/entrypoints/llm/test_gdn_spec_boundary_state.py -v
"""

import inspect
import math
import multiprocessing
import os
import traceback

import numpy as np
import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "Qwen/Qwen3.5-4B"
MAX_MODEL_LEN = 1024
# Also the mamba block size: it holds the GDN state, and its multiples tile
# the attention kernel's 512-token KV chunks.
BLOCK_SIZE = 512
NUM_SPEC = 3
# Requests whose prompts end at successive positions before the boundary.
NUM_PHASES = 4
# References at T - d for every d up to MAX_OFFSET.
MAX_OFFSET = 8
# One case: engine build and compilation, decode, and the references.
TIMEOUT_S = 480


def _four_layers(config):
    text = config.get_text_config()
    text.num_hidden_layers = 4
    text.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return config


def _xla_cache_entries():
    return len(torch.tpu._get_cache_stats().per_entry_stats)


def _count_copy_graphs(worker):
    """From now on, count the XLA graphs the boundary copies compile."""
    from vllm_torchtpu.runner import spec_boundary_state

    runner = worker.model_runner
    copy = spec_boundary_state.copy_states
    runner.boundary_copy_graphs = 0

    def counted(*args):
        before = _xla_cache_entries()
        copy(*args)
        runner.boundary_copy_graphs += _xla_cache_entries() - before

    spec_boundary_state.copy_states = counted


def _boundary_copy_graphs(worker):
    return worker.model_runner.boundary_copy_graphs


def _region_geometry(worker):
    """Per mamba group: byte ranges of the SSM and conv states in a block.

    A block of the unified pool is token-major: the SSM state fills its first
    tokens, the conv state starts at the next token.
    """
    runner = worker.model_runner
    out = {}
    for gid, raws in runner._mamba_copy_plan:
        spec = runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec
        assert raws[0].element_size() == 2, raws[0].dtype
        ssm_item = torch.empty((), dtype=spec.dtypes[1]).element_size()
        assert ssm_item == 2, spec.dtypes
        conv_bytes = math.prod(spec.shapes[0]) * 2
        layout = derive_pooled_gdn_state_layout(
            ssm_bytes=math.prod(spec.shapes[1]) * ssm_item,
            conv_bytes=conv_bytes,
            token_bytes=math.prod(raws[0].shape[2:]) * raws[0].element_size(),
        )
        conv_start = layout.ssm_tokens * layout.token_bytes
        out[gid] = (layout.ssm_bytes, conv_start, conv_bytes)
    return out


def _read_blocks(worker, blocks_by_gid):
    """Bytes of manager blocks, keyed `(gid, slot)`; one block per group."""
    runner = worker.model_runner
    split = runner._pool_block_split
    out = {}
    for gid, raws in runner._mamba_copy_plan:
        if gid not in blocks_by_gid:
            continue
        block = blocks_by_gid[gid]
        index = torch.arange(block * split, (block + 1) * split, device=raws[0].device)
        for slot, raw in enumerate(raws):
            rows = raw.index_select(0, index).to("cpu").contiguous()
            out[(gid, slot)] = rows.view(torch.uint8).numpy().tobytes()
    return out


def _states(data: bytes, geometry) -> tuple[np.ndarray, np.ndarray]:
    """The (ssm, conv) states of one block as float32."""
    ssm_bytes, conv_start, conv_bytes = geometry

    def bf16(chunk: bytes) -> np.ndarray:
        return (np.frombuffer(chunk, np.uint16).astype(np.uint32) << 16).view(
            np.float32
        )

    return bf16(data[:ssm_bytes]), bf16(data[conv_start : conv_start + conv_bytes])


def _rel(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64), y.astype(np.float64)
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))


def _check_boundary_states(llm: LLM) -> None:
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    groups = scheduler.kv_cache_config.kv_cache_groups
    mamba_gids = [
        g
        for g, group in enumerate(groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    block = groups[mamba_gids[0]].kv_cache_spec.block_size
    assert block == BLOCK_SIZE, block

    # Mamba blocks as the scheduler hashes them, keyed (salt, gid, column),
    # and each finished request's block ids per group.
    hashed: dict[tuple[str, int, int], int] = {}
    freed: dict[str, tuple[list[int], ...]] = {}
    pool = scheduler.kv_cache_manager.block_pool
    cache_full_blocks = pool.cache_full_blocks
    signature = inspect.signature(cache_full_blocks)

    def record_hashes(*args, **kwargs):
        cache_full_blocks(*args, **kwargs)
        call = signature.bind(*args, **kwargs).arguments
        if call["kv_cache_group_id"] not in mamba_gids:
            return
        blocks = call["blocks"]
        for col in range(call["num_cached_blocks"], call["num_full_blocks"]):
            if not blocks[col].is_null and blocks[col].block_hash is not None:
                key = (call["request"].cache_salt, call["kv_cache_group_id"], col)
                hashed[key] = blocks[col].block_id

    pool.cache_full_blocks = record_hashes
    free_request_blocks = scheduler._free_request_blocks

    def record_frees(request):
        freed[request.cache_salt] = scheduler.kv_cache_manager.get_block_ids(
            request.request_id
        )
        free_request_blocks(request)

    scheduler._free_request_blocks = record_frees
    geometry = llm.collective_rpc(_region_geometry)[0]

    def run(requests):
        return llm.generate(
            [
                {"prompt_token_ids": ids, "cache_salt": salt}
                for ids, salt, _ in requests
            ],
            [params for _, _, params in requests],
            use_tqdm=False,
        )

    def read(salt_blocks):
        blocks_by_gid = {gid: salt_blocks[gid] for gid in mamba_gids}
        return llm.collective_rpc(_read_blocks, args=(blocks_by_gid,))[0]

    # Prompts end at successive positions before the boundary, so the
    # MTP windows meet it at every phase; all decode past it.
    boundary = block
    rng = np.random.default_rng(0)
    decode = []
    for j in range(NUM_PHASES):
        prompt = rng.integers(1000, 100000, size=boundary - 40 + j).tolist()
        params = SamplingParams(
            temperature=0.0, max_tokens=boundary + 32 - len(prompt), ignore_eos=True
        )
        decode.append((prompt, f"A{j}", params))
    llm.collective_rpc(_count_copy_graphs)
    outputs = run(decode)
    graphs = llm.collective_rpc(_boundary_copy_graphs)
    assert graphs == [0], f"boundary copies compiled {graphs} graphs after warmup"
    tokens = [
        list(prompt) + list(out.outputs[0].token_ids)
        for (prompt, _, _), out in zip(decode, outputs)
    ]
    col = boundary // block - 1
    stored = []
    for j in range(NUM_PHASES):
        ids = {gid: hashed.get((f"A{j}", gid, col)) for gid in mamba_gids}
        assert all(ids.values()), f"boundary block of A{j} not hashed: {ids}"
        stored.append(read(ids))

    # Each reference runs alone and is read before anything reuses its
    # freed state block.
    greedy1 = SamplingParams(temperature=0.0, max_tokens=1)
    failures = []
    for j in range(NUM_PHASES):
        refs = []
        for d in range(MAX_OFFSET + 1):
            salt = f"R{j}-{d}"
            (out,) = run([(tokens[j][: boundary - d], salt, greedy1)])
            assert out.num_cached_tokens == 0
            state_col = (boundary - d - 1) // block
            refs.append(read({g: freed[salt][g][state_col] for g in mamba_gids}))
        for key, data in stored[j].items():
            ssm, conv = _states(data, geometry[key[0]])
            rel_ssm, rel_conv = [], []
            for ref in refs:
                ref_ssm, ref_conv = _states(ref[key], geometry[key[0]])
                rel_ssm.append(_rel(ssm, ref_ssm))
                rel_conv.append(_rel(conv, ref_conv))
            best_ssm, best_conv = int(np.argmin(rel_ssm)), int(np.argmin(rel_conv))
            if best_ssm or best_conv:
                failures.append(
                    f"A{j} group {key[0]} slot {key[1]}: closest to T-{best_ssm}"
                    f" (ssm), T-{best_conv} (conv); relative error by d"
                    f" {np.round(rel_ssm, 3).tolist()}"
                )
    assert not failures, "\n".join(failures)


def _run_case(acceptance, async_scheduling, conn) -> None:
    """Builds the engine in this process and sends back None or a failure."""
    try:
        # In-process engine core: the scheduler hooks run in this process.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        llm = LLM(
            model=MODEL_NAME,
            hf_overrides=_four_layers,
            load_format="dummy",
            language_model_only=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            block_size=BLOCK_SIZE,
            max_num_seqs=NUM_PHASES,
            max_num_batched_tokens=MAX_MODEL_LEN,
            kv_cache_memory_bytes=1 << 30,
            enable_prefix_caching=True,
            # Retain every block's state, decode-filled ones included.
            prefix_cache_retention_interval=None,
            mamba_cache_mode="align",
            mamba_ssm_cache_dtype="bfloat16",
            async_scheduling=async_scheduling,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": NUM_SPEC,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_rates": acceptance,
            },
            seed=0,
        )
        _check_boundary_states(llm)
        conn.send(None)
    except BaseException:
        conn.send(traceback.format_exc())


@pytest.mark.parametrize(
    "acceptance,async_scheduling",
    [
        pytest.param([1.0, 1.0, 1.0], True, id="accept_all-async"),
        pytest.param([1.0, 0.0, 0.0], False, id="accept_one-sync"),
    ],
)
def test_decode_filled_block_holds_boundary_state(acceptance, async_scheduling):
    if os.environ.get("VLLM_KV_CACHE_LAYOUT") == "HND":
        pytest.skip("reads the token-major pool layout")
    # The engine runs in a child process, so the TPU is free again for the
    # tests after this one that start engines of their own.
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_case, args=(acceptance, async_scheduling, child)
    )
    process.start()
    child.close()
    try:
        failure = parent.recv() if parent.poll(TIMEOUT_S) else "timed out"
    except EOFError:
        failure = "exited without a result"
    finally:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join()
    assert failure is None, failure
