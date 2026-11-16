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
"""Unit tests for TPU static bucket padding runner hook and Prometheus collector."""

import importlib.util
import json
import logging
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.telemetry import AgenticTelemetryService, patch_tpu_padding_metrics


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


def test_runner_padding_hook_warmup_gate_spec_decode_and_pcp(
    monkeypatch, tmp_path
) -> None:
    """Skips warmup rounds, classifies spec-verify via `num_windowed_reqs`, and PCP."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", str(tmp_path / ".pad_spec"))
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    # API-server activation when `tpu_runner` is not loaded must not import it.
    if "vllm_torchtpu.runner.tpu_runner" not in sys.modules:
        patch_tpu_padding_metrics(patch_prometheus=False)
        assert "vllm_torchtpu.runner.tpu_runner" not in sys.modules

    svc = AgenticTelemetryService(block_size=4, replica_id="rep-spec", dp_rank=0)

    class FakeSpecPcpRunner:
        def __init__(self) -> None:
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

        def capture_model(self) -> None:
            # Synthetic MTP / spec-decode warmup executes `_prepare_inputs` with a
            # populated layout plan; these rounds must not be counted.
            self._prepare_inputs(SimpleNamespace(), 0, 0, 0)
            self._prepare_inputs(SimpleNamespace(), 0, 0, 2)

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            start_index: int,
            num_decode_reqs: int,
            num_windowed_reqs: int | None = None,
        ) -> tuple[Any, ...]:
            return (
                None,
                None,
                2,
                2,
                start_index + 2,
                None,
                None,
                None,
                None,
                self._last_sequence_layout_plan,
            )

    patch_tpu_padding_metrics(FakeSpecPcpRunner, service=svc)
    runner = FakeSpecPcpRunner()

    # 1a. Before `capture_model()` finishes (flag off) -> nothing recorded.
    assert runner._agentic_padding_ready is False
    runner._prepare_inputs(SimpleNamespace(), 0, 0, 0)
    assert svc.sync_padding_state()["prompt_rounds"] == 0
    assert svc.sync_padding_state()["reply_rounds"] == 0

    # 1b. Synthetic warmup rounds inside `capture_model()` -> still nothing recorded,
    # and readiness flips on once `capture_model()` completes.
    runner.capture_model()
    assert runner._agentic_padding_ready is True
    assert svc.sync_padding_state()["prompt_rounds"] == 0
    assert svc.sync_padding_state()["reply_rounds"] == 0

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

    # 3. PCP sharding_plan (when `_last_sequence_layout_plan` is None) on prefill chunk.
    runner._last_sequence_layout_plan = None
    runner.pcp_manager = SimpleNamespace(
        sharding_plan=SimpleNamespace(
            global_num_tokens=430, global_padded_num_tokens=512
        )
    )
    runner._prepare_inputs(SimpleNamespace(), 0, 0, 0)
    counts_2 = svc.sync_padding_state()
    assert counts_2["prompt_rounds"] == 1
    assert counts_2["prompt_real_tokens"] == 430
    assert counts_2["prompt_filler_tokens"] == 82


def _make_fake_tp_dp_runner_cls() -> type:
    class _FakeTpDpRunner:
        def __init__(self, tp_rank: int, dp_rank: int = 0, dp_size: int = 2) -> None:
            self.parallel_config = SimpleNamespace(
                rank=tp_rank,
                world_size=4,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
            )
            self._last_sequence_layout_plan = SimpleNamespace(
                global_num_tokens=12, global_padded_num_tokens=16
            )
            self.pcp_manager = None

        def capture_model(self) -> None:
            pass

        def _prepare_inputs(
            self,
            scheduler_output: Any,
            start_index: int,
            num_decode_reqs: int,
            num_windowed_reqs: int | None = None,
        ) -> tuple[Any, ...]:
            return (
                None,
                None,
                12,
                12,
                start_index + 12,
                None,
                None,
                None,
                None,
                self._last_sequence_layout_plan,
            )

    return _FakeTpDpRunner


def test_padding_hook_tp_rank_zero_only_and_multi_dp_sum(monkeypatch, tmp_path) -> None:
    """Only TPxPP rank 0 writes per DP engine, and API reader sums across DP ranks."""
    import vllm_torchtpu.telemetry.vllm_hooks as hooks_mod

    monkeypatch.setattr(envs, "VLLM_TPU_AGENTIC_TELEMETRY", True)
    pad_base = str(tmp_path / ".pad_tp_dp")
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_PATH", pad_base)
    monkeypatch.setattr(hooks_mod, "_PADDING_SYNC_MIN_INTERVAL_S", 0.0)

    cls_non_zero = _make_fake_tp_dp_runner_cls()
    svc_non_zero = AgenticTelemetryService(block_size=4, replica_id="rep-dp2")
    patch_tpu_padding_metrics(cls_non_zero, service=svc_non_zero)
    for tp_rank in (1, 2, 3):
        r = cls_non_zero(tp_rank=tp_rank, dp_rank=0, dp_size=2)
        r.capture_model()
        r._prepare_inputs(SimpleNamespace(), 0, 12, 12)
    assert not os.path.exists(svc_non_zero._padding_sync_path(0))

    # DP=2: fresh runner class per DP engine service so wrappers never stack.
    cls_dp0 = _make_fake_tp_dp_runner_cls()
    svc_dp0 = AgenticTelemetryService(block_size=4, replica_id="rep-dp2", dp_rank=0)
    patch_tpu_padding_metrics(cls_dp0, service=svc_dp0)
    r0 = cls_dp0(tp_rank=0, dp_rank=0, dp_size=2)
    r0.capture_model()
    r0._prepare_inputs(SimpleNamespace(), 0, 12, 12)

    cls_dp1 = _make_fake_tp_dp_runner_cls()
    svc_dp1 = AgenticTelemetryService(block_size=4, replica_id="rep-dp2", dp_rank=1)
    patch_tpu_padding_metrics(cls_dp1, service=svc_dp1)
    r1 = cls_dp1(tp_rank=0, dp_rank=1, dp_size=2)
    r1.capture_model()
    r1._prepare_inputs(SimpleNamespace(), 0, 12, 12)

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
    # Scrape before first new round sees 0 (not old 4 * 500 = 2000!).
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

    # Round 1 flushes immediately; Rounds 2 & 3 fall inside the 40ms window,
    # including a prompt/reply phase flip that must also respect the rate limit.
    writer.on_tpu_round_padded(10, 16, is_prompt_round=True, dp_rank=0, dp_size=1)
    writer.on_tpu_round_padded(4, 16, is_prompt_round=False, dp_rank=0, dp_size=1)
    writer.on_tpu_round_padded(4, 16, is_prompt_round=False, dp_rank=0, dp_size=1)

    # Immediately after Round 3, Rounds 2 & 3 are still pending in worker memory.
    assert writer._padding_sync_dirty is True

    # Traffic stops; the coalesced timer flushes pending rounds to `/dev/shm`.
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

    patch_tpu_padding_metrics(
        service=svc,
        patch_prometheus=True,
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
