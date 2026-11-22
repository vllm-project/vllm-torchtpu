# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for synthetic warmup ownership of shared physical blocks."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_test

_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/runner/tpu_runner.py"
)
_METHODS = {
    "single": "_warmup_one_spec_request",
    "batch": "_warmup_spec_batch",
    "pcp": "_warmup_one_pcp_mtp_prefill",
}


class _CircularBufferSpec:
    block_size = 8

    def max_num_blocks_per_req(self, config, max_len):
        return 1


class _UniformTypeKVCacheSpecs:
    def __init__(self, first_spec):
        self.first_spec = first_spec
        self.block_size = first_spec.block_size


@pytest.fixture
def warmup_code():
    """Execute actual runner helpers/methods without importing the TPU stack."""
    tree = ast.parse(_RUNNER_PATH.read_text(), filename=str(_RUNNER_PATH))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_spec_warmup_")
    ]
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner"
    )
    methods = [
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _METHODS.values()
    ]
    for method in methods:
        method.body = [
            node for node in method.body if not isinstance(node, ast.ImportFrom)
        ]
    namespace = {
        "Any": Any,
        "cast": lambda _type, value: value,
        "cdiv": lambda x, y: (x + y - 1) // y,
        "CircularBufferSpec": _CircularBufferSpec,
        "UniformTypeKVCacheSpecs": _UniformTypeKVCacheSpecs,
        "logger": logging.getLogger(__name__),
        "SamplingParams": SimpleNamespace,
        "NewRequestData": SimpleNamespace,
        "CachedRequestData": SimpleNamespace,
        "SchedulerOutput": SimpleNamespace(make_empty=SimpleNamespace),
    }
    module = ast.Module(body=helpers + methods, type_ignores=[])
    exec(compile(module, str(_RUNNER_PATH), "exec"), namespace)
    return namespace


def _make_runner(num_blocks):
    calls = []
    cleanups = []
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
        SimpleNamespace(kv_cache_spec=_CircularBufferSpec()),
    ]
    runner = SimpleNamespace(
        max_model_len=128,
        max_num_tokens=12,
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        kv_cache_config=SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=groups),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        max_num_reqs=8,
        execute_model=lambda output: calls.append(output),
        sample_tokens=lambda _: None,
        take_draft_token_ids=lambda: None,
        _warmup_spec_decode_cleanup=lambda ids: cleanups.append(ids) or True,
    )
    return runner, calls, cleanups


def _run(code, mode, runner):
    method = code[_METHODS[mode]]
    if mode == "batch":
        return method(runner, R=3, P=6)
    if mode == "single":
        return method(runner, P=6, idx=0)
    return method(runner, P=6)


@pytest.mark.parametrize("mode", _METHODS)
def test_warmup_reserves_disjoint_request_group_blocks(warmup_code, mode):
    runner, calls, cleanups = _make_runner(128)
    assert _run(warmup_code, mode, runner) is not False
    requests = [
        req for call in calls for req in getattr(call, "scheduled_new_reqs", [])
    ]
    assert len(requests) == (3 if mode == "batch" else 1)
    blocks = [block for req in requests for group in req.block_ids for block in group]
    assert len(blocks) == len(set(blocks))
    assert blocks == list(range(1, len(blocks) + 1))
    assert all(len(req.block_ids[2]) == 1 for req in requests)
    assert cleanups
    if mode == "batch":
        # Chunked prefills remain resident together for the following decodes.
        assert len([call for call in calls if hasattr(call, "scheduled_new_reqs")]) == 2
    if mode == "pcp":
        assert len(calls) == 1


@pytest.mark.parametrize("mode", _METHODS)
def test_warmup_sum_capacity_and_exact_fit(warmup_code, mode):
    # Existing contexts yield [4, 1, 1] for single, [6, 2, 1] per batch
    # request, and [2, 1, 1] for PCP. Circular state stays capped at one block.
    required = {"single": 6, "batch": 27, "pcp": 4}[mode]
    runner, calls, _ = _make_runner(required + 1)
    assert _run(warmup_code, mode, runner) is not False
    assert calls

    runner, calls, cleanups = _make_runner(required)
    if mode == "pcp":
        with pytest.raises(RuntimeError, match=f"required={required + 1}"):
            _run(warmup_code, mode, runner)
    else:
        assert _run(warmup_code, mode, runner) is (False if mode == "single" else None)
    assert not calls
    assert not cleanups


@pytest.mark.parametrize(
    ("counts", "capacity", "expected"),
    [
        ([[2, 1], [1, 2]], 7, [([1, 2], [3]), ([4], [5, 6])]),
        ([[2, 1], [1, 2]], 6, None),
        ([[0, 1], [2, 0]], 4, [([], [1]), ([2, 3], [])]),
        ([], 1, []),
    ],
)
def test_block_plan_capacity_and_empty_groups(warmup_code, counts, capacity, expected):
    assert warmup_code["_spec_warmup_block_plan"](counts, capacity) == expected


@pytest.mark.parametrize("capacity, expected_extras", [(67, 3), (66, 1), (34, 0)])
def test_mixed_warmup_keeps_variants_disjoint(warmup_code, capacity, expected_extras):
    runner, calls, cleanups = _make_runner(capacity)
    runner.max_num_tokens = 48
    warmup_code["_warmup_spec_batch"](runner, R=2, P=6, mixed=True)
    requests = [
        req for call in calls for req in getattr(call, "scheduled_new_reqs", [])
    ]
    assert len(requests) == 2 + expected_extras
    blocks = [b for req in requests for group in req.block_ids for b in group]
    assert blocks == list(range(1, len(blocks) + 1))
    assert max(blocks) < capacity
    assert all(len(req.block_ids[2]) == 1 for req in requests)
    assert set(cleanups[-1]) == {req.req_id for req in requests}
    if expected_extras == 3:
        mixed_calls = [
            call for call in calls if getattr(call, "finished_req_ids", set())
        ]
        assert len(mixed_calls) == 1
        assert mixed_calls[0].finished_req_ids == {requests[2].req_id}


@pytest.mark.parametrize("mode", _METHODS)
def test_wrapped_circular_state_retains_ownership_cap(warmup_code, mode):
    runner, calls, _ = _make_runner(128)
    runner.kv_cache_config.kv_cache_groups[2].kv_cache_spec = _UniformTypeKVCacheSpecs(
        _CircularBufferSpec()
    )
    assert _run(warmup_code, mode, runner) is not False
    for call in calls:
        for req in getattr(call, "scheduled_new_reqs", []):
            assert len(req.block_ids[2]) == 1


@pytest.mark.parametrize(("counts", "capacity"), [([[1]], 0), ([[-1]], 4)])
def test_block_plan_rejects_invalid_capacity(warmup_code, counts, capacity):
    with pytest.raises(ValueError, match="nonnegative counts and a null block"):
        warmup_code["_spec_warmup_block_plan"](counts, capacity)
