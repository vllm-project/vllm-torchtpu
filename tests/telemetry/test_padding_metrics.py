# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Unit tests for direct TPU bucket padding runner hooks and Prometheus collector."""

import ast
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry import (
    AgenticTelemetryService,
    finish_tpu_padding_warmup,
    record_tpu_round_padding,
    register_tpu_padding_collector,
)


@pytest.fixture(autouse=True)
def _ensure_vllm_logger_for_standalone_cpu(monkeypatch) -> None:
    if importlib.util.find_spec("vllm") is not None:
        return

    class _StandInLogger(logging.Logger):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._seen_once: set[str] = set()

        def warning_once(self, msg: str, *args: Any, **kwargs: Any) -> None:
            if msg not in self._seen_once:
                self._seen_once.add(msg)
                self.warning(msg, *args, **kwargs)

    vllm_pkg = ModuleType("vllm")
    vllm_logger_mod: Any = ModuleType("vllm.logger")
    vllm_logger_mod.init_logger = lambda name: _StandInLogger(name)
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger_mod)


def test_tpu_runner_direct_wiring_without_monkey_patching() -> None:
    """`TPUModelRunner` wires warmup and padding callbacks without hot-path imports."""
    runner_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "vllm_torchtpu"
        / "runner"
        / "tpu_runner.py"
    )
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    runner_cls_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner"
    )
    methods = {
        node.name: node
        for node in runner_cls_node.body
        if isinstance(node, ast.FunctionDef)
    }

    init_attrs = {
        target.attr
        for node in ast.walk(methods["__init__"])
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert "_agentic_padding_recorder" in init_attrs

    capture_method_calls = [
        node.func.attr
        for node in ast.walk(methods["capture_model"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    # Called on both the `enforce_eager` early return and post-precompile exit.
    assert capture_method_calls.count("_enable_agentic_padding_telemetry") == 2

    enable_calls = [
        node.func.id
        for node in ast.walk(methods["_enable_agentic_padding_telemetry"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "finish_tpu_padding_warmup" in enable_calls

    # `_prepare_inputs` reads cached `_agentic_padding_recorder` via `getattr`
    # without per-step imports or `envs` lookups.
    prepare_calls = [
        node.func.id
        for node in ast.walk(methods["_prepare_inputs"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "record_padding" in prepare_calls
    prepare_imports = [
        node.module
        for node in ast.walk(methods["_prepare_inputs"])
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any("telemetry" in m for m in prepare_imports)

    # API-server collector registration never imports `tpu_runner`.
    if "vllm_torchtpu.runner.tpu_runner" not in sys.modules:
        register_tpu_padding_collector()
        assert "vllm_torchtpu.runner.tpu_runner" not in sys.modules


def test_runner_padding_hook_warmup_gate_eager_spec_decode_and_pcp(
    monkeypatch, tmp_path
) -> None:
    """Skips warmup rounds, supports `enforce_eager`, spec-decode, and PCP."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", str(tmp_path / ".pad_spec"))
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    svc = AgenticTelemetryService(block_size=4, replica_id="rep-spec", dp_rank=0)

    class FakeSpecPcpRunner:
        def __init__(self, *, enforce_eager: bool = False) -> None:
            self.enforce_eager = enforce_eager
            self.parallel_config = SimpleNamespace(
                rank=0,
                world_size=1,
                data_parallel_rank=0,
                data_parallel_size=1,
            )
            self._last_sequence_layout_plan: Any = SimpleNamespace(
                global_num_tokens=64, global_padded_num_tokens=128
            )
            self.pcp_manager: Any = None
            self._agentic_padding_recorder: Any = None

        def _enable_agentic_padding_telemetry(self) -> None:
            if not envs.VLLM_TPU_AGENTIC_TELEMETRY:
                return
            finish_tpu_padding_warmup(self, service=svc)
            self._agentic_padding_recorder = record_tpu_round_padding

        def capture_model(self) -> None:
            self._agentic_padding_recorder = None
            if self.enforce_eager:
                self._enable_agentic_padding_telemetry()
                return
            # Synthetic MTP / spec-decode warmup executes `_prepare_inputs` with a
            # populated layout plan; these rounds must not be counted.
            self._prepare_inputs(SimpleNamespace(), 0, 0, 0)
            self._prepare_inputs(SimpleNamespace(), 0, 0, 2)
            self._enable_agentic_padding_telemetry()

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            start_index: int,
            num_decode_reqs: int,
            num_windowed_reqs: int | None = None,
        ) -> None:
            if num_windowed_reqs is None:
                num_windowed_reqs = num_decode_reqs
            end_index = start_index + 2
            plan = (
                self.pcp_manager.sharding_plan
                if self.pcp_manager is not None
                else self._last_sequence_layout_plan
            )
            record_padding = getattr(self, "_agentic_padding_recorder", None)
            if record_padding is not None:
                record_padding(
                    self,
                    end_index=end_index,
                    num_decode_reqs=num_decode_reqs,
                    num_windowed_reqs=num_windowed_reqs,
                    real_tokens=plan.global_num_tokens if plan is not None else None,
                    padded_tokens=(
                        plan.global_padded_num_tokens if plan is not None else None
                    ),
                )

    runner = FakeSpecPcpRunner()

    # 1a. Before `capture_model()` finishes -> nothing recorded.
    assert runner._agentic_padding_recorder is None
    runner._prepare_inputs(SimpleNamespace(), 0, 0, 0)
    assert svc.sync_padding_state()["prompt_rounds"] == 0
    assert svc.sync_padding_state()["reply_rounds"] == 0

    # 1b. Synthetic warmup rounds inside `capture_model()` -> still nothing recorded,
    # and recorder + cached service bind once `capture_model()` completes.
    runner.capture_model()
    assert runner._agentic_padding_recorder is record_tpu_round_padding
    assert runner._agentic_padding_service is svc
    assert svc.sync_padding_state()["prompt_rounds"] == 0
    assert svc.sync_padding_state()["reply_rounds"] == 0

    # 1c. `enforce_eager=True` early return also enables padding recording.
    eager_runner = FakeSpecPcpRunner(enforce_eager=True)
    eager_runner.capture_model()
    assert eager_runner._agentic_padding_recorder is record_tpu_round_padding

    # 1d. `__new__`-constructed runner without `__init__` attributes does not crash.
    bare_runner = object.__new__(FakeSpecPcpRunner)
    bare_runner._last_sequence_layout_plan = SimpleNamespace(
        global_num_tokens=16, global_padded_num_tokens=32
    )
    bare_runner.pcp_manager = None
    bare_runner._prepare_inputs(SimpleNamespace(), 0, 0, 0)
    assert svc.sync_padding_state()["prompt_rounds"] == 0

    # 2. Pure MTP/Eagle spec-decode step: `_reorder_batch_for_rpa` returns
    # `num_decode_reqs=0, num_windowed_reqs=2` for two (1 + 3 draft)-token requests.
    runner._last_sequence_layout_plan = SimpleNamespace(
        global_num_tokens=8, global_padded_num_tokens=16
    )
    runner._prepare_inputs(SimpleNamespace(), 0, 0, 2)
    counts = svc.sync_padding_state()
    assert counts["reply_rounds"] == 1
    assert counts["reply_real_tokens"] == 8
    assert counts["reply_filler_tokens"] == 8
    assert counts["prompt_rounds"] == 0

    # 3. PCP sharding_plan (when `_last_sequence_layout_plan` is None) on prefill chunk,
    # also verifying layout-plan resolution fallback when token kwargs are omitted.
    runner._last_sequence_layout_plan = None
    runner.pcp_manager = SimpleNamespace(
        sharding_plan=SimpleNamespace(
            global_num_tokens=430, global_padded_num_tokens=512
        )
    )
    record_tpu_round_padding(
        runner,
        end_index=2,
        num_decode_reqs=0,
        num_windowed_reqs=0,
    )
    counts_2 = svc.sync_padding_state()
    assert counts_2["prompt_rounds"] == 1
    assert counts_2["prompt_real_tokens"] == 430
    assert counts_2["prompt_filler_tokens"] == 82

    # 4. Strict no-op when `VLLM_TPU_AGENTIC_TELEMETRY` is disabled at warmup.
    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", False)
    disabled_runner = FakeSpecPcpRunner()
    disabled_runner.capture_model()
    assert disabled_runner._agentic_padding_recorder is None
    disabled_runner._prepare_inputs(SimpleNamespace(), 0, 0, 0)
    assert svc.sync_padding_state()["prompt_rounds"] == 1


def _make_fake_tp_dp_runner(*, tp_rank: int, dp_rank: int = 0, dp_size: int = 2) -> Any:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=tp_rank,
            world_size=4,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
        ),
        _last_sequence_layout_plan=SimpleNamespace(
            global_num_tokens=12, global_padded_num_tokens=16
        ),
        pcp_manager=None,
        _agentic_padding_ready=False,
    )


def test_padding_hook_tp_rank_zero_only_and_multi_dp_sum(monkeypatch, tmp_path) -> None:
    """Only TPxPP rank 0 writes per DP engine, and API reader sums across DP ranks."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    pad_base = str(tmp_path / ".pad_tp_dp")
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", pad_base)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    svc_non_zero = AgenticTelemetryService(block_size=4, replica_id="rep-dp2")
    for tp_rank in (1, 2, 3):
        r = _make_fake_tp_dp_runner(tp_rank=tp_rank, dp_rank=0, dp_size=2)
        finish_tpu_padding_warmup(r, service=svc_non_zero)
        record_tpu_round_padding(
            r,
            end_index=12,
            num_decode_reqs=12,
            num_windowed_reqs=12,
            real_tokens=12,
            padded_tokens=16,
            service=svc_non_zero,
        )
    assert not os.path.exists(svc_non_zero._padding_sync_path(0))

    svc_dp0 = AgenticTelemetryService(block_size=4, replica_id="rep-dp2", dp_rank=0)
    r0 = _make_fake_tp_dp_runner(tp_rank=0, dp_rank=0, dp_size=2)
    finish_tpu_padding_warmup(r0, service=svc_dp0)
    record_tpu_round_padding(
        r0,
        end_index=12,
        num_decode_reqs=12,
        num_windowed_reqs=12,
        real_tokens=12,
        padded_tokens=16,
        service=svc_dp0,
    )

    svc_dp1 = AgenticTelemetryService(block_size=4, replica_id="rep-dp2", dp_rank=1)
    r1 = _make_fake_tp_dp_runner(tp_rank=0, dp_rank=1, dp_size=2)
    finish_tpu_padding_warmup(r1, service=svc_dp1)
    record_tpu_round_padding(
        r1,
        end_index=12,
        num_decode_reqs=12,
        num_windowed_reqs=12,
        real_tokens=12,
        padded_tokens=16,
        service=svc_dp1,
    )

    api_reader = AgenticTelemetryService(block_size=4, replica_id="rep-dp2")
    synced = api_reader.sync_padding_state()
    assert synced["reply_rounds"] == 2
    assert synced["reply_real_tokens"] == 24
    assert synced["reply_filler_tokens"] == 8


def test_padding_sync_prunes_stale_runs_shrunk_dp_and_foreign_servers(
    monkeypatch, tmp_path
) -> None:
    """Dead PIDs, shrunk DP ranks, and foreign servers do not pollute `/metrics`."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    pad_base = str(tmp_path / ".pad_stale")
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", pad_base)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    api_reader = AgenticTelemetryService(block_size=4, replica_id="rep-shared")
    prefix = api_reader._padding_sync_prefix()

    # 1. Leftover file from a dead worker PID is unlinked and ignored on scrape.
    dead_path = f"{prefix}_dp0_99999999"
    with open(dead_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_rounds": 99,
                "prompt_real_tokens": 50000,
                "prompt_filler_tokens": 10000,
                "reply_rounds": 0,
                "reply_real_tokens": 0,
                "reply_filler_tokens": 0,
                "dp_rank": 0,
                "dp_size": 1,
                "pid": 99999999,
                "proc_start": 12345,
                "ancestors": [99999999],
                "started_at": 1.0,
            },
            f,
        )
    assert api_reader.sync_padding_state()["prompt_rounds"] == 0
    assert not os.path.exists(dead_path)

    # 2. Foreign live server on same host (live PID 1, disjoint lineage) is ignored
    # without deleting the other server's file.
    foreign_path = f"{prefix}_dp0_1"
    init_stat = hooks_mod._proc_stat_fields(1)
    foreign_proc_start = init_stat[3] if init_stat is not None else 0
    with open(foreign_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_rounds": 77,
                "prompt_real_tokens": 7700,
                "prompt_filler_tokens": 770,
                "reply_rounds": 0,
                "reply_real_tokens": 0,
                "reply_filler_tokens": 0,
                "dp_rank": 0,
                "dp_size": 1,
                "pid": 1,
                "proc_start": foreign_proc_start,
                "ancestors": [1],
                "started_at": 2.0,
            },
            f,
        )
    assert api_reader.sync_padding_state()["prompt_rounds"] == 0
    assert os.path.exists(foreign_path)
    os.unlink(foreign_path)

    # 3. Restart from DP=4 (large totals) -> DP=2 (small totals): `_dp2`/`_dp3` are
    # dropped and `_dp0`/`_dp1` overwrite directly instead of being masked by `max()`.
    for old_rank in range(4):
        old_svc = AgenticTelemetryService(
            block_size=4, replica_id="rep-shared", dp_rank=old_rank
        )
        old_svc.on_tpu_round_padded(
            500, 1024, is_prompt_round=True, dp_rank=old_rank, dp_size=4
        )
    assert api_reader.sync_padding_state()["prompt_rounds"] == 4

    time.sleep(0.005)
    new_dp0 = AgenticTelemetryService(block_size=4, replica_id="rep-shared", dp_rank=0)
    new_dp1 = AgenticTelemetryService(block_size=4, replica_id="rep-shared", dp_rank=1)
    new_dp0.init_padding_sync_writer(dp_rank=0, dp_size=2)
    new_dp1.init_padding_sync_writer(dp_rank=1, dp_size=2)
    assert api_reader.sync_padding_state()["prompt_rounds"] == 0
    assert api_reader.sync_padding_state()["prompt_real_tokens"] == 0

    new_dp0.on_tpu_round_padded(10, 16, is_prompt_round=True, dp_rank=0, dp_size=2)
    new_dp1.on_tpu_round_padded(12, 16, is_prompt_round=True, dp_rank=1, dp_size=2)
    after_restart = api_reader.sync_padding_state()
    assert after_restart["prompt_rounds"] == 2
    assert after_restart["prompt_real_tokens"] == 22
    assert after_restart["prompt_filler_tokens"] == 10


def test_rate_limited_trailing_rounds_flush_when_idle(monkeypatch, tmp_path) -> None:
    """Rounds deferred inside the rate-limit window flush automatically once idle."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", str(tmp_path / ".pad_idle"))
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.04)

    writer = AgenticTelemetryService(block_size=4, replica_id="rep-idle", dp_rank=0)
    reader = AgenticTelemetryService(block_size=4, replica_id="rep-idle")

    writer.on_tpu_round_padded(10, 16, is_prompt_round=True, dp_rank=0, dp_size=1)
    writer.on_tpu_round_padded(4, 16, is_prompt_round=False, dp_rank=0, dp_size=1)
    writer.on_tpu_round_padded(4, 16, is_prompt_round=False, dp_rank=0, dp_size=1)

    assert writer._padding_sync_dirty is True

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        synced = reader.sync_padding_state()
        if synced["reply_rounds"] == 2:
            break
        time.sleep(0.01)

    synced = reader.sync_padding_state()
    assert synced["prompt_rounds"] == 1
    assert synced["reply_rounds"] == 2
    assert synced["reply_real_tokens"] == 8
    assert synced["reply_filler_tokens"] == 24


def test_prometheus_collector_and_isolated_registry_wrappers(
    monkeypatch, tmp_path
) -> None:
    """Collector emits expected Prometheus series with isolated registries."""
    prometheus_client = pytest.importorskip("prometheus_client")
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", str(tmp_path / ".pad_prom"))
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    svc = AgenticTelemetryService(block_size=4, replica_id="rep-prom", dp_rank=0)
    svc.on_tpu_round_padded(100, 128, is_prompt_round=True)
    svc.on_tpu_round_padded(4, 16, is_prompt_round=False)

    fresh_reg = prometheus_client.CollectorRegistry()

    class FakeMultiProcessCollector:
        def collect(self) -> Any:
            return iter(())

    register_tpu_padding_collector(
        service=svc,
        prometheus_registry=fresh_reg,
        multiprocess_cls=FakeMultiProcessCollector,
    )

    families = {m.name: m for m in fresh_reg.collect()}
    assert "vllm:tpu_padding_rounds" in families
    assert "vllm:tpu_padding_tokens" in families

    samples = {
        (s.name, tuple(sorted(s.labels.items()))): s.value
        for fam in families.values()
        for s in fam.samples
    }
    assert (
        samples[
            (
                "vllm:tpu_padding_tokens_total",
                (("kind", "filler"), ("phase", "prompt"), ("replica_id", "rep-prom")),
            )
        ]
        == 28.0
    )
    assert (
        samples[
            (
                "vllm:tpu_padding_tokens_total",
                (("kind", "filler"), ("phase", "reply"), ("replica_id", "rep-prom")),
            )
        ]
        == 12.0
    )

    mp_families = list(FakeMultiProcessCollector().collect())
    assert len(mp_families) == 2
