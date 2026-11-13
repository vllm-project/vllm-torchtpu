# SPDX-License-Identifier: Apache-2.0
"""Capture decode draft shapes without consuming device output."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from server_step_worker import DECODE_SCENARIO, Recorder, ServerStepWorker  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_decode_captures_scheduled_drafts_and_native_prepare_shapes():
    output = (SimpleNamespace(shape=(128, 32)), SimpleNamespace(shape=(128,)))
    calls = []

    def prepare(chunk):
        calls.append(chunk.num_reqs)
        return output

    runner = SimpleNamespace(
        execute_model=lambda scheduler: None,
        sample_tokens=lambda: None,
        drafter=SimpleNamespace(_prepare_dflash_inputs=prepare, block_size=5),
    )
    recorder = Recorder(runner, DECODE_SCENARIO, 0)
    recorder.enabled = True
    recorder.execute_model(
        SimpleNamespace(scheduled_spec_decode_tokens={"r": [1, 2, 3, 4, 5]})
    )
    result = runner.drafter._prepare_dflash_inputs(SimpleNamespace(num_reqs=16))
    assert result is output
    assert calls == [16]
    assert recorder.pending["scheduled_draft_lengths"] == {"r": 5}
    stage = recorder.pending["stages"][0]
    assert (stage["requests"], stage["rows"], stage["padded_rows"]) == (16, 80, 128)
    recorder.enabled = False
    runner.drafter._prepare_dflash_inputs(SimpleNamespace(num_reqs=16))
    assert len(recorder.stages) == 1


@pytest.mark.parametrize("dp_size", [8, 16])
def test_synchronized_profile_writes_to_requested_directory(
    tmp_path, monkeypatch, dp_size
):
    rank = 0
    calls, barriers, syncs = [], [], []
    group = SimpleNamespace(world_size=dp_size, cpu_group="cpu")

    def gather(target, value, group):
        target[:] = [{**value, "rank": i} for i in range(dp_size)]

    dist = SimpleNamespace(
        all_gather_object=gather, barrier=lambda **kw: barriers.append(kw)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_dp_group=lambda: group)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_torchtpu.utils",
        SimpleNamespace(synchronize_device=lambda: syncs.append(True)),
    )
    runner = SimpleNamespace(
        _dp_coordinated_step=lambda *a, **kw: "native",
        input_batch=SimpleNamespace(
            num_reqs=48,
            num_computed_tokens_cpu=[DECODE_SCENARIO.prefix_tokens + 44] * 48,
        ),
    )
    original = runner._dp_coordinated_step
    worker = ServerStepWorker()
    worker.model_runner = runner
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=rank)
    )
    worker.profile = lambda **kw: calls.append(kw)
    worker.synchronized_profile_arm(
        output_dir=str(tmp_path), scenario="decode" if dp_size == 8 else "decode_dp16"
    )
    for _ in range(7):
        assert runner._dp_coordinated_step(288) == "native"
    assert runner._dp_coordinated_step is original
    assert len(calls) == 2
    assert len(barriers) == 4 and len(syncs) == 2
    capture = worker.synchronized_profile_check()
    assert len(capture) == dp_size
    assert all(r["completed"] and r["captured_steps"] == 6 for r in capture)
    assert (tmp_path / "synchronized-profile" / f"rank-{rank}-capture.json").is_file()


@pytest.mark.parametrize(
    "mesh_size,impl,prepush,exchange,eplb,mode,static_layout,error",
    [
        (16, "push", 2, True, True, "decode_only", None, None),
        (16, "push", 2, True, True, "prefill_only", None, "decode_only"),
        (16, "push", 2, True, False, "off", "static", None),
        (16, "push", 2, True, False, "off", None, "expert layout"),
        (None, "push", 2, True, True, "decode_only", None, "EP16 mesh"),
        (8, "push", 2, True, True, "decode_only", None, "EP16 mesh"),
        (16, "v2", 2, True, True, "decode_only", None, "requires push"),
        (16, "push", 1, True, True, "decode_only", None, "SC_PREPUSH=2"),
        (16, "push", 2, False, True, "decode_only", None, "TC_EXCHANGE=1"),
    ],
)
def test_dp16_ci_requires_push_and_dynamic_eplb_without_split_topology(
    monkeypatch, mesh_size, impl, prepush, exchange, eplb, mode, static_layout, error
):
    from server_step_worker import require_dp16_recipe

    mesh = None if mesh_size is None else SimpleNamespace(size=mesh_size)
    monkeypatch.setitem(
        sys.modules,
        "vllm_torchtpu.distributed.ep_mesh",
        SimpleNamespace(build_ep_mesh=lambda: mesh),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_torchtpu.envs",
        SimpleNamespace(
            MOE_FUSED_EP_KERNEL_IMPL=impl,
            MOE_FUSED_EP_PUSH_SC_PREPUSH=prepush,
            MOE_FUSED_EP_PUSH_TC_EXCHANGE=exchange,
            TPU_MOE_EPLB_DYNAMIC=eplb,
            TPU_MOE_EPLB_MODE=mode,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_torchtpu.expert_placement",
        SimpleNamespace(configured_value=lambda: static_layout),
    )
    # Push works without #2646: no topology preprocessing is required.
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.compilation.moe_preprocessing", None
    )
    if error is None:
        require_dp16_recipe()
    else:
        with pytest.raises(ValueError, match=error):
            require_dp16_recipe()
