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

import contextlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner import utils as runner_utils_module
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID


def test_spec_warmup_all_token_ids_matches_current_sequence_lengths():
    all_token_ids = tpu_runner._spec_warmup_all_token_ids(
        ["request-0", "request-1"], [7, 12]
    )

    assert all_token_ids == {
        "request-0": [0] * 8,
        "request-1": [0] * 13,
    }


def test_suspend_kv_transfer_group_restores_agent_after_failure():
    connector = object()

    with patch.object(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector):
        with (
            pytest.raises(RuntimeError, match="synthetic warmup failed"),
            tpu_runner._suspend_kv_transfer_group(),
        ):
            assert kv_transfer_state._KV_CONNECTOR_AGENT is None
            raise RuntimeError("synthetic warmup failed")

        assert kv_transfer_state._KV_CONNECTOR_AGENT is connector


@pytest.mark.parametrize(
    ("outcomes", "raises_after_retry"),
    [
        ([False, True], False),
        ([False, False], True),
    ],
)
def test_pcp_mtp_prefill_warmup_retries_once(outcomes, raises_after_retry):
    attempts = MagicMock(side_effect=outcomes)
    runner = SimpleNamespace(
        enforce_eager=False,
        _is_async_drafter=True,
        input_batch=SimpleNamespace(num_reqs=0),
        num_tokens_paddings=[4096],
        max_model_len=262144,
        num_xla_graphs=7,
        _precompile_timed=lambda _label: contextlib.nullcontext(),
        _warmup_one_pcp_mtp_prefill=attempts,
    )

    expectation = (
        pytest.raises(RuntimeError, match="PCP MTP prefill warmup failed after retry")
        if raises_after_retry
        else contextlib.nullcontext()
    )
    with (
        patch.object(
            tpu_runner,
            "_suspend_kv_transfer_group",
            return_value=contextlib.nullcontext(),
        ),
        expectation,
    ):
        TPUModelRunner._warmup_pcp_mtp_prefill(runner)

    assert attempts.call_args_list == [
        ((4096,), {"quiet": True}),
        ((4096,), {}),
    ]


def test_one_pcp_mtp_warmup_builds_prefill_without_decode():
    executed = []
    runner = SimpleNamespace(
        max_model_len=64,
        kv_cache_config=SimpleNamespace(
            num_blocks=100,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8)),
            ],
        ),
        execute_model=MagicMock(side_effect=lambda so: executed.append(so)),
        sample_tokens=MagicMock(),
        take_draft_token_ids=MagicMock(),
        _warmup_spec_decode_cleanup=MagicMock(return_value=True),
    )

    assert TPUModelRunner._warmup_one_pcp_mtp_prefill(runner, 31) is True

    assert len(executed) == 1
    scheduler_output = executed[0]
    assert scheduler_output.total_num_scheduled_tokens == 31
    assert scheduler_output.num_scheduled_tokens == {"__pcp_mtp_warmup__": 31}
    assert len(scheduler_output.scheduled_new_reqs) == 1
    request = scheduler_output.scheduled_new_reqs[0]
    assert request.req_id == "__pcp_mtp_warmup__"
    assert len(request.prompt_token_ids) == 31
    assert request.block_ids == ([98, 99], [96, 97, 98, 99])
    assert scheduler_output.scheduled_cached_reqs.req_ids == []
    runner.sample_tokens.assert_called_once_with(None)
    runner.take_draft_token_ids.assert_called_once_with()
    runner._warmup_spec_decode_cleanup.assert_called_once_with("__pcp_mtp_warmup__")


def _sub_indices(req_id_to_index_copy, req_ids, num_scheduled, spec_k=None):
    """Run _prepare_async_token_substitution_indices with a minimal fake self.
    req_id_to_index_copy holds each request's *position*; with spec_k set the
    builder uses a `1+spec_k` source stride. Returns (cur, source) index lists."""
    fake = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy=req_id_to_index_copy,
            spec_decode_num_rejected_tokens=(None if spec_k is None else object()),
        ),
        speculative_config=(
            None if spec_k is None else SimpleNamespace(num_speculative_tokens=spec_k)
        ),
        _last_sequence_layout_plan=None,
        input_batch=SimpleNamespace(req_ids=req_ids),
    )
    cur, src = TPUModelRunner._prepare_async_token_substitution_indices(
        fake,
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.array(num_scheduled, dtype=np.int32),
    )
    return cur.tolist(), src.tolist()


def test_async_sub_indices_non_spec():
    # num_scheduled == 1 per request → one placeholder slot each (the original
    # single-token decode behaviour, unchanged).
    cur, src = _sub_indices({"r0": 0, "r1": 1}, ["r0", "r1"], [1, 1])
    assert cur == [0, 1]
    assert src == [0, 1]


def test_async_sub_indices_spec_1plusk():
    # K=3 → stride 4. Positions r0=0, r1=1 → source spans at 0*4 and 1*4; each
    # request occupies 4 input slots ([bonus, draft_1..3]).
    cur, src = _sub_indices({"r0": 0, "r1": 1}, ["r0", "r1"], [4, 4], spec_k=3)
    assert cur == [0, 1, 2, 3, 4, 5, 6, 7]
    assert src == [0, 1, 2, 3, 4, 5, 6, 7]


def test_async_sub_indices_skips_new_req():
    # r0 is new (no source span) → skipped, but its scheduled tokens still
    # advance the running input offset for r1. K=3 → stride 4, r1 position 0.
    cur, src = _sub_indices({"r1": 0}, ["r0", "r1"], [3, 4], spec_k=3)
    assert cur == [3, 4, 5, 6]
    assert src == [0, 1, 2, 3]


def test_get_finished_kv_transfers_drains_invalid_block_ids():
    connector = MagicMock()
    connector.get_finished.return_value = ({"sent"}, {"loaded"})
    connector.get_block_ids_with_load_errors.return_value = {41, 43}
    connector.get_block_ids_with_load_errors_group_index.return_value = 2
    connector.build_connector_worker_meta.return_value = {"jobs": []}
    connector.get_kv_connector_stats.return_value = None
    runner = SimpleNamespace()
    scheduler_output = SimpleNamespace(finished_req_ids={"finished"})

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
    ):
        result = TPUModelRunner.get_finished_kv_transfers(runner, scheduler_output)

    assert result == ({"sent"}, {"loaded"}, {"jobs": []}, {41, 43}, 2, None)
    connector.get_finished.assert_called_once_with({"finished"})
    connector.get_block_ids_with_load_errors.assert_called_once_with()
    connector.get_block_ids_with_load_errors_group_index.assert_called_once_with()
    connector.clear_connector_metadata.assert_called_once_with()


def test_no_forward_output_preserves_invalid_block_ids():
    runner = SimpleNamespace(
        maybe_setup_kv_connector=MagicMock(),
        get_finished_kv_transfers=MagicMock(
            return_value=(set(), {"failed-load"}, None, {41, 43}, 2, None)
        ),
    )
    scheduler_output = SimpleNamespace()
    vllm_config = SimpleNamespace()

    with patch(
        "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
        return_value=False,
    ):
        output = TPUModelRunner.kv_connector_no_forward(
            runner, scheduler_output, vllm_config
        )

    assert output.kv_connector_output.finished_recving == {"failed-load"}
    assert output.kv_connector_output.invalid_block_ids == {41, 43}
    if tpu_runner._KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP:
        assert output.kv_connector_output.invalid_block_group_index == 2


def test_build_kv_connector_output_supports_vllm_023():
    with (
        patch.object(
            tpu_runner,
            "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP",
            False,
        ),
        patch.object(tpu_runner, "KVConnectorOutput") as output_cls,
    ):
        tpu_runner._build_kv_connector_output(
            finished_sending={"sent"},
            finished_recving=None,
            kv_connector_worker_meta=None,
            invalid_block_ids=set(),
            invalid_block_group_index=None,
        )

    output_cls.assert_called_once_with(
        finished_sending={"sent"},
        finished_recving=None,
        kv_connector_worker_meta=None,
        invalid_block_ids=set(),
        kv_connector_stats=None,
    )


def test_build_kv_connector_output_forwards_invalid_block_ids_for_hybrid_recovery():
    with patch.object(
        tpu_runner,
        "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP",
        False,
    ):
        out = tpu_runner._build_kv_connector_output(
            finished_sending=None,
            finished_recving={"failed-load"},
            kv_connector_worker_meta=None,
            invalid_block_ids={41, 43},
            invalid_block_group_index=2,
        )
        assert out.finished_recving == {"failed-load"}
        assert out.invalid_block_ids == {41, 43}


def _fake_phased_runner(
    additional_config=None,
    torch_profiler_dir="/tmp/phased",
    max_iterations=0,
    delay_iterations=0,
    profiler_rank=2,
    profiler_world_size=4,
):
    """A stand-in for TPUModelRunner carrying only what phased profiling reads.

    No parallel_config: the phased profiler must not reach for it.
    """
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config=additional_config
            if additional_config is not None
            else {},
            profiler_config=SimpleNamespace(
                torch_profiler_dir=torch_profiler_dir,
                max_iterations=max_iterations,
                delay_iterations=delay_iterations,
            ),
        ),
        _profiler_rank=profiler_rank,
        _profiler_world_size=profiler_world_size,
    )


def _init_phased_runner(monkeypatch, phased_enabled=True, **runner_kwargs):
    """Build a fake runner and run _init_phased_profiling over it.

    `phased_enabled` drives only the env var, independently of the trace
    directory, so a test can exercise either guard in isolation.
    """
    if phased_enabled:
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")
    else:
        monkeypatch.delenv("USE_PHASED_PROFILER", raising=False)
    runner = _fake_phased_runner(**runner_kwargs)
    TPUModelRunner._init_phased_profiling(runner)
    return runner


class TestInitPhasedProfiling:
    """Verify _init_phased_profiling is selected by USE_PHASED_PROFILER and
    reads its directory from profiler_config. It only resolves the directory;
    start_phased_profiling/stop_phased_profiling (below) own arming the
    profiler itself, since that is now triggered from TPUWorker.profile()."""

    def test_disabled_when_env_not_set(self, monkeypatch):
        runner = _init_phased_runner(monkeypatch, phased_enabled=False)
        assert runner.phased_profiling_dir == ""
        assert runner.phase_based_profiler is None

    def test_disabled_when_dir_not_set(self, monkeypatch):
        """The env var alone is not enough; traces need somewhere to go."""
        runner = _init_phased_runner(monkeypatch, torch_profiler_dir="")
        assert runner.phased_profiling_dir == ""
        assert runner.phase_based_profiler is None

    def test_enabled_resolves_dir_without_arming(self, monkeypatch):
        runner = _init_phased_runner(monkeypatch)
        assert runner.phased_profiling_dir == "/tmp/phased"
        assert runner.phase_based_profiler is None

    def test_profiler_rank_kwargs_have_no_default(self):
        """Guards the reason they are required.

        A default here can only come from parallel_config, which is the wrong
        scope for DP and fails silently: every replica reports rank 0 and the
        merge keeps one trace out of N. Better to fail loudly at a new call
        site than to lose traces at one.
        """
        params = inspect.signature(TPUModelRunner.__init__).parameters
        for name in ("profiler_rank", "profiler_world_size"):
            assert params[name].default is inspect.Parameter.empty
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


class TestStartStopPhasedProfiling:
    """start_phased_profiling/stop_phased_profiling arm/disarm the phase
    profiler; TPUWorker.profile() is the only caller, mirroring how it
    start/stops the standard torch profiler."""

    def test_start_when_dir_not_set_warns_and_noops(self, monkeypatch):
        """Phased mode is on, but there is nowhere to write traces."""
        runner = _init_phased_runner(monkeypatch, torch_profiler_dir="")
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        mock_profiler_cls.assert_not_called()
        assert runner.phase_based_profiler is None

    def test_start_uses_config_values(self, monkeypatch):
        runner = _init_phased_runner(
            monkeypatch,
            additional_config={
                "phased_profiler_decode_only_kv_len_threshold": 128,
                "phased_profiler_prefill_only_kv_len_threshold": 4096,
            },
            max_iterations=20,
            delay_iterations=3,
        )
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        mock_profiler_cls.assert_called_once_with(
            "/tmp/phased",
            worker_rank=2,
            world_size=4,
            num_steps_to_profile_for=20,
            num_decode_steps_to_skip=3,
            decode_kv_len_threshold=128,
            prefill_kv_len_threshold=4096,
            standard_opts={
                "host_tracer_level": 2,
                "device_tracer_level": 1,
                "python_tracer_level": 1,
            },
            advanced_opts={
                "tpu_trace_mode": "TRACE_COMPUTE",
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            },
        )
        assert runner.phase_based_profiler is mock_profiler_cls.return_value

    def test_profile_prefix_scopes_the_phase_dirs(self, monkeypatch):
        """Same scoping the standard profiler gives /start_profile's prefix:
        it names the run, and the phase subdirectories sit beneath it."""
        runner = _init_phased_runner(monkeypatch)
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner, "decode")

        assert mock_profiler_cls.call_args.args[0] == "/tmp/phased/decode"

    def test_profiler_kwargs_overrides_options(self, monkeypatch):
        runner = _init_phased_runner(monkeypatch)
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(
                runner,
                "decode",
                profiler_kwargs={
                    "host_tracer_level": 3,
                    "e2e_enable_fw_throttle_event": True,
                },
            )

        assert mock_profiler_cls.call_args.args[0] == "/tmp/phased/decode"
        assert (
            mock_profiler_cls.call_args.kwargs["standard_opts"]["host_tracer_level"]
            == 3
        )
        assert (
            mock_profiler_cls.call_args.kwargs["advanced_opts"][
                "e2e_enable_fw_throttle_event"
            ]
            is True
        )

    def test_start_twice_is_a_noop(self, monkeypatch):
        """Already armed; a second /start_profile must not replace it and
        lose the phases it has already marked as seen."""
        runner = _init_phased_runner(monkeypatch)
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)
            first_profiler = runner.phase_based_profiler
            TPUModelRunner.start_phased_profiling(runner)

        mock_profiler_cls.assert_called_once()
        assert runner.phase_based_profiler is first_profiler

    def test_uses_worker_supplied_slice_global_rank(self, monkeypatch):
        """parallel_config.rank is TPxPP-scoped: every DP replica calls itself
        rank 0, so their traces would collide. The worker passes the
        slice-global rank instead, and it is the only source."""
        runner = _init_phased_runner(
            monkeypatch, profiler_rank=9, profiler_world_size=16
        )
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        assert mock_profiler_cls.call_args.kwargs["worker_rank"] == 9
        assert mock_profiler_cls.call_args.kwargs["world_size"] == 16

    def test_falls_back_to_default_num_steps_when_max_iterations_unset(
        self, monkeypatch
    ):
        runner = _init_phased_runner(monkeypatch, max_iterations=0, delay_iterations=0)
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        assert (
            mock_profiler_cls.call_args.kwargs["num_steps_to_profile_for"]
            == runner_utils_module.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
        )
        assert (
            mock_profiler_cls.call_args.kwargs["decode_kv_len_threshold"]
            == runner_utils_module.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD
        )

    def test_stop_finishes_and_clears_the_profiler(self, monkeypatch):
        runner = _init_phased_runner(monkeypatch)
        with patch(
            "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)
            armed_profiler = mock_profiler_cls.return_value

            TPUModelRunner.stop_phased_profiling(runner)

        armed_profiler.finish.assert_called_once_with()
        assert runner.phase_based_profiler is None

    def test_stop_when_not_running_warns_and_noops(self, monkeypatch):
        runner = _init_phased_runner(monkeypatch)

        TPUModelRunner.stop_phased_profiling(runner)

        assert runner.phase_based_profiler is None


class TestTPURunner:
    def setup_method(self):
        self.mock_device = torch.device("cpu")

        model_config = ModelConfig(
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype=torch.bfloat16,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_model_len=1024,
            is_encoder_decoder=False,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            observability_config={},
            additional_config={},
        )

        self.runner = MagicMock(spec=TPUModelRunner)
        self.runner.device = self.mock_device
        self.runner.block_size = 16
        self.runner.kv_cache_dtype = torch.bfloat16
        self.runner.use_spmd = False
        self.runner.vllm_config = vllm_config
        self.runner.cache_config = vllm_config.cache_config
        self.runner.scheduler_config = vllm_config.scheduler_config
        self.runner._hybrid_uniform_page_size_bytes = None
        self.runner.max_num_reqs = 16
        self.runner.max_model_len = 1024
        self.runner.max_num_tokens = 2048
        self.runner.pin_memory = False
        self.runner.model_config = vllm_config.model_config
        self.runner.enforce_eager = False
        self.runner.speculative_config = None

        self._find_non_ssm_backend_patcher = patch.object(
            TpuPlatform, "_find_non_ssm_backend", return_value=PallasAttentionBackend
        )
        self._find_non_ssm_backend_patcher.start()

    def teardown_method(self):
        self._find_non_ssm_backend_patcher.stop()

    def test_maybe_setup_kv_connector_fences_preemptions_before_bind(self):
        """Upstream parity: handle_preemptions must run before
        bind_connector_metadata / start_load_kv so connectors with async
        saves (OffloadingConnector jobs_to_flush) can fence in-flight
        stores before the forward overwrites their source blocks."""
        self.runner.maybe_setup_kv_connector = (
            TPUModelRunner.maybe_setup_kv_connector.__get__(self.runner)
        )
        connector = MagicMock()
        scheduler_output = MagicMock()
        meta = scheduler_output.kv_connector_metadata

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
                return_value=connector,
            ),
        ):
            self.runner.maybe_setup_kv_connector(scheduler_output)

        connector.handle_preemptions.assert_called_once_with(meta)
        connector.bind_connector_metadata.assert_called_once_with(meta)
        connector.start_load_kv.assert_called_once_with(None)
        names = [c[0] for c in connector.mock_calls]
        assert (
            names.index("handle_preemptions")
            < names.index("bind_connector_metadata")
            < names.index("start_load_kv")
        )

    def test_mrope_positions_buffer_is_int32(self):
        """Run the real TPUModelRunner.__init__ (with parent's __init__ and
        a few heavy TPU-only helpers patched out) on the setup_method mock
        and verify the uses_mrope override block produces an int32 buffer
        — not just that _make_buffer respects an int32 dtype on its own."""
        self.runner.dtype = torch.bfloat16
        self.runner.parallel_config = self.runner.vllm_config.parallel_config
        self.runner.uses_mrope = True
        self.runner.supports_mm_inputs = False
        self.runner.vllm_config.compilation_config.compile_sizes = [16, 2048]
        self.runner._make_buffer = GPUModelRunner._make_buffer.__get__(self.runner)

        with (
            patch.object(GPUModelRunner, "__init__", return_value=None),
            patch(
                "vllm_torchtpu.runner.tpu_runner._torch_tpu_wrapper",
                side_effect=lambda: contextlib.nullcontext(),
            ),
            patch("vllm_torchtpu.runner.tpu_runner._validate_libtpu_version"),
            patch.object(
                TPUModelRunner, "_create_mesh_for_parallelism", return_value=MagicMock()
            ),
        ):
            TPUModelRunner.__init__(
                self.runner,
                self.runner.vllm_config,
                self.mock_device,
                profiler_rank=0,
                profiler_world_size=1,
            )

        assert self.runner.mrope_positions.cpu.dtype == torch.int32
        assert self.runner.mrope_positions.cpu.shape == (
            3,
            self.runner.max_num_tokens + 1,
        )
        assert self.runner.mrope_positions.np.dtype == np.int32


class TestAttentionMetadataBuilder:
    """Direct tests for AttentionMetadataBuilder.build, exercising the
    runner-state path used in _prepare_inputs and the position_ids_override
    path used in _dummy_run."""

    def _make_runner_mock(
        self,
        most_model_len=None,
        num_groups=1,
        max_num_blocks_per_req=4,
        max_num_reqs=4,
    ):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.block_size = 16
        runner.max_num_reqs = max_num_reqs
        runner.most_model_len = most_model_len
        runner._unified_kv_layout = False
        runner.position_ids = torch.full((8,), 42, dtype=torch.int32)

        block_tables = []
        for gid in range(num_groups):
            bt = MagicMock()
            bt.max_num_blocks_per_req = max_num_blocks_per_req
            bt.get_cpu_tensor.return_value = (
                torch.arange(
                    max_num_reqs * max_num_blocks_per_req, dtype=torch.int32
                ).reshape(max_num_reqs, max_num_blocks_per_req)
                + gid * 100
            )
            block_tables.append(bt)
        runner.input_batch.block_table = block_tables
        return runner

    def _make_builder(self, runner, kv_cache_group_id=0, spec=None):
        if spec is None:
            spec = FullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.bfloat16,
                page_size_padded=16384,
            )
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["attn.0"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=kv_cache_group_id,
        )

    def _make_cm(self, num_reqs):
        """Minimal CommonAttentionMetadata stand-in — build() only reads
        num_reqs off it (everything else still comes from ctx because TPU
        bypasses parent's CpuGpuBuffers)."""
        cm = MagicMock()
        cm.num_reqs = num_reqs
        return cm

    def test_build_runner_state_path(self):
        """Normal _prepare_inputs path: copy from the right per-group block
        table at the right start_index, pad remaining rows, and reuse
        runner.position_ids. target_num_reqs is sourced from cm.num_reqs
        (the only field we currently read from common_attn_metadata)."""
        runner = self._make_runner_mock(num_groups=2)
        # Build for group 1 to cover the per-group block_table lookup too.
        builder = self._make_builder(runner, kv_cache_group_id=1)

        target_num_reqs, num_reqs, start_index = 4, 2, 1
        seq_lens = torch.tensor([10, 12, 0, 0], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
        request_distribution = torch.tensor([2, 2, 2], dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=True,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        meta = builder.build(
            common_prefix_len=0, common_attn_metadata=self._make_cm(target_num_reqs)
        )

        assert isinstance(meta, AttentionMetadata)
        assert meta.input_positions is runner.position_ids
        assert meta.seq_lens is seq_lens
        assert meta.query_start_loc is query_start_loc
        assert meta.request_distribution is request_distribution

        # Group 1 (not 0) should have been read; group 0 untouched.
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        runner.input_batch.block_table[1].get_cpu_tensor.assert_called_once()

        # Flattened (target_num_reqs * max_num_blocks_per_req,); first
        # num_reqs rows from the source slice, rest zero-padded.
        max_num_blocks = runner.input_batch.block_table[1].max_num_blocks_per_req
        block_tables_2d = meta.block_tables.reshape(target_num_reqs, max_num_blocks)
        src = runner.input_batch.block_table[1].get_cpu_tensor.return_value
        assert torch.equal(
            block_tables_2d[:num_reqs], src[start_index : start_index + num_reqs]
        )
        assert torch.equal(
            block_tables_2d[num_reqs:],
            torch.zeros(
                (target_num_reqs - num_reqs, max_num_blocks), dtype=torch.int32
            ),
        )

    def test_build_position_ids_override(self):
        """_dummy_run path: position_ids_override is forwarded as-is and the
        block-table copy is skipped (no read from input_batch)."""
        runner = self._make_runner_mock()
        builder = self._make_builder(runner)

        override = torch.zeros((3, 8), dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4,), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            position_ids_override=override,
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        assert meta.input_positions is override
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        assert torch.equal(meta.block_tables, torch.zeros((4 * 4,), dtype=torch.int32))

    def test_build_most_model_len_shrinks_block_table(self):
        """When use_max_model_len is False, target_num_blocks =
        cdiv(most_model_len, block_size) — smaller than the per-group
        max_num_blocks_per_req, so the H2D copy is shorter."""
        runner = self._make_runner_mock(most_model_len=32, max_num_blocks_per_req=8)
        builder = self._make_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.tensor([8, 16, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        # cdiv(32, 16) = 2; flattened length = target_num_reqs * 2 = 8.
        assert meta.block_tables.shape == (4 * 2,)

    def test_unified_mamba_state_indices_derive_from_block_table(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(2, 8)],
            dtypes=[torch.bfloat16],
            page_size_padded=256,
        )
        builder = self._make_builder(runner, spec=mamba_spec)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 33, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        assert torch.equal(
            meta.mamba_state_indices, torch.tensor([0, 6, 0, 0], dtype=torch.int32)
        )

    def test_unified_mamba_state_indices_use_cp_adjusted_block_size(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(2, 8)],
            dtypes=[torch.bfloat16],
            page_size_padded=256,
        )
        with patch(
            "vllm_torchtpu.layers.core.attention_metadata.get_pcp_group"
        ) as mock_pcp:
            mock_pcp.return_value.world_size = 4
            builder = self._make_builder(runner, spec=mamba_spec)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([64, 65, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0, common_attn_metadata=self._make_cm(4))

        assert builder.target_block_size == 64
        assert torch.equal(
            meta.mamba_state_indices, torch.tensor([0, 5, 0, 0], dtype=torch.int32)
        )

    def test_unified_mamba_state_indices_dcp_does_not_scale_block_size(self):
        """Regression test for #1233: DCP must not scale mamba target_block_size."""
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_kv_layout = True
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(2, 8)],
            dtypes=[torch.bfloat16],
            page_size_padded=256,
        )
        with (
            patch("vllm.distributed.get_dcp_group") as mock_dcp,
            patch(
                "vllm_torchtpu.layers.core.attention_metadata.get_pcp_group"
            ) as mock_pcp,
        ):
            mock_dcp.return_value.world_size = 4
            mock_pcp.return_value.world_size = 1
            builder = self._make_builder(runner, spec=mamba_spec)

        assert builder.target_block_size == 16


class TestCompactMambaSlotPool:
    """Unit tests for the compact-mamba recurrent-slot allocator
    (_init_mamba_slot_pool + _build_mamba_state_indices). The slot index is
    correctness-critical: a wrong slot = silent recurrent-state corruption."""

    def _make_runner(self, max_num_reqs=4, mamba_num_blocks=5, slot_stride=1):
        runner = MagicMock(spec=TPUModelRunner)
        runner.device = torch.device("cpu")
        runner.max_num_reqs = max_num_reqs
        # Slots per request: 1 without spec decode, num_spec + 1 with it.
        runner._mamba_slot_stride = slot_stride
        runner.mamba_state_indices_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
        runner.input_batch = MagicMock()
        # Bind the real methods.
        runner._init_mamba_slot_pool = TPUModelRunner._init_mamba_slot_pool.__get__(
            runner
        )
        runner._build_mamba_state_indices = (
            TPUModelRunner._build_mamba_state_indices.__get__(runner)
        )
        return runner

    def _set_batch(self, runner, req_ids):
        """Set the persistent batch to the given ordered req_ids."""
        runner.input_batch.req_ids = list(req_ids)
        runner.input_batch.req_id_to_index = {r: i for i, r in enumerate(req_ids)}

    def test_unique_slots_and_null_tail(self):
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)  # usable slots 1..4, slot 0 = null
        self._set_batch(runner, ["a", "b"])

        idx = runner._build_mamba_state_indices(
            start_index=0, num_reqs=2, target_num_reqs=4
        ).cpu()
        # Two active requests get distinct non-null slots.
        assert idx[0] != 0 and idx[1] != 0
        assert idx[0] != idx[1]
        # Padded tail points at the null slot (0).
        assert idx[2] == 0 and idx[3] == 0

    def test_idle_dp_rank_gets_all_null_slots(self):
        """An idle DP rank asks for the coordinated shape with an empty batch.

        `run_dp_dummy_draft` replays a busy peer's collective trace using
        `num_reqs_max_model_len`, but the idle rank has no scheduled
        requests. Walking the real `req_ids` for that many positions used to
        raise `IndexError: list index out of range` inside the dummy draft
        and kill the engine (compact layout only -- the unified pool builds
        its indices from the block table and never reaches this code).
        """
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, [])  # idle rank: nothing scheduled

        idx = runner._build_mamba_state_indices(
            start_index=0, num_reqs=4, target_num_reqs=4
        ).cpu()
        # Every position is the null slot: no real request to address, and
        # the dummy forward must not touch a live request's state.
        assert [int(v) for v in idx] == [0, 0, 0, 0]
        # No slots were consumed, so a later real request still gets one.
        assert len(runner._free_mamba_slots) == 4

    def test_partially_filled_batch_assigns_only_live_prefix(self):
        """Coordinated `num_reqs` above the live count keeps the tail null."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a"])

        # Peer coordinated 4; this rank only has 1 request.
        idx = runner._build_mamba_state_indices(0, 4, 4).cpu()
        assert int(idx[0]) != 0
        assert [int(v) for v in idx[1:]] == [0, 0, 0]

    def test_slot_follows_req_id_through_condense(self):
        """A request keeps its slot even when its persistent-batch position
        changes (upstream condense moves it to a lower index)."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b"])
        idx0 = runner._build_mamba_state_indices(0, 2, 4).cpu()
        slot_b = int(idx0[1])

        # "a" finishes; "b" condenses to position 0.
        self._set_batch(runner, ["b"])
        idx1 = runner._build_mamba_state_indices(0, 1, 4).cpu()
        assert int(idx1[0]) == slot_b  # same physical slot as before

    def test_freed_slot_is_reused(self):
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b", "c", "d"])
        idx0 = runner._build_mamba_state_indices(0, 4, 4).cpu()
        used = {int(x) for x in idx0}
        assert used == {1, 2, 3, 4}  # all usable slots allocated, none null

        # All finish; a fresh request must reuse a freed slot (pool not
        # exhausted, no out-of-range index).
        self._set_batch(runner, ["e"])
        idx1 = runner._build_mamba_state_indices(0, 1, 4).cpu()
        assert 1 <= int(idx1[0]) <= 4

    def test_chunked_build_allocates_all(self):
        """Two chunks in one step: each chunk allocates its own requests; all
        get distinct slots."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b", "c", "d"])

        # clone: the result aliases a reused staging buffer (no copy on CPU).
        idx_chunk0 = runner._build_mamba_state_indices(0, 2, 4).clone()
        idx_chunk1 = runner._build_mamba_state_indices(2, 2, 4).clone()
        slots = {
            int(idx_chunk0[0]),
            int(idx_chunk0[1]),
            int(idx_chunk1[0]),
            int(idx_chunk1[1]),
        }
        assert slots == {1, 2, 3, 4}

    def test_spec_decode_stride_allocates_group_bases(self):
        """With speculative decoding each request owns a group of `stride`
        consecutive slots; the pool hands out only the group *bases*."""
        # num_spec = 2 -> stride 3; max_num_reqs = 3 -> 3*3 + 1 = 10 blocks.
        runner = self._make_runner(max_num_reqs=3, mamba_num_blocks=10, slot_stride=3)
        runner._init_mamba_slot_pool(10)
        # Group bases are 1, 4, 7 (slot 0 is the null block); the interior
        # checkpoint slots (2,3,5,6,8,9) are never handed out directly.
        assert sorted(runner._free_mamba_slots) == [1, 4, 7]

        self._set_batch(runner, ["a", "b", "c"])
        idx = runner._build_mamba_state_indices(0, 3, 3).cpu()
        assert sorted(int(x) for x in idx) == [1, 4, 7]

    def test_spec_decode_stride_one_matches_plain_layout(self):
        """stride 1 (spec decode disabled) reproduces the dense layout."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5, slot_stride=1)
        runner._init_mamba_slot_pool(5)
        assert sorted(runner._free_mamba_slots) == [1, 2, 3, 4]


class TestReorderBatchForRpa:
    """Unit tests for the RPA batch reordering. With spec decode the batch is
    partitioned into three contiguous segments [decode][verify][prefill]; the
    returned (num_decode, num_windowed) split is fed to both RPA (decode-only
    front) and the GDN windowed kernel (decode + verify)."""

    def _make_runner(self):
        runner = MagicMock(spec=TPUModelRunner)
        runner.input_batch = MagicMock()
        runner._reorder_batch_for_rpa = TPUModelRunner._reorder_batch_for_rpa.__get__(
            runner
        )
        return runner

    def _batch(self, runner, req_ids):
        ib = runner.input_batch
        ib.req_ids = list(req_ids)
        ib.num_reqs = len(req_ids)

        def swap_states(i, j):
            ib.req_ids[i], ib.req_ids[j] = ib.req_ids[j], ib.req_ids[i]

        ib.swap_states.side_effect = swap_states

    def _sched(self, num_scheduled, spec_reqs=()):
        return SimpleNamespace(
            num_scheduled_tokens=num_scheduled,
            scheduled_spec_decode_tokens={r: object() for r in spec_reqs},
        )

    def test_empty_batch(self):
        runner = self._make_runner()
        self._batch(runner, [])
        assert runner._reorder_batch_for_rpa(self._sched({})) == (0, 0)

    def test_no_spec_decode_first_and_windowed_equal(self):
        """Without spec decode, num_decode == num_windowed and only the
        1-token decodes move to the front."""
        runner = self._make_runner()
        self._batch(runner, ["p", "d1", "p2", "d2"])
        sched = self._sched({"p": 8, "d1": 1, "p2": 5, "d2": 1})
        num_decode, num_windowed = runner._reorder_batch_for_rpa(sched)
        assert num_decode == 2
        assert num_windowed == 2
        # Both decode requests are in the front two positions.
        assert set(runner.input_batch.req_ids[:2]) == {"d1", "d2"}

    def test_three_segments_decode_verify_prefill(self):
        """Spec decode yields [decode][verify][prefill] with the split at
        num_decode and num_windowed."""
        runner = self._make_runner()
        self._batch(runner, ["pf", "vr", "dc", "vr2", "dc2"])
        # dc, dc2: 1-token decode; vr, vr2: multi-token spec verify; pf: prefill
        sched = self._sched(
            {"pf": 16, "vr": 3, "dc": 1, "vr2": 3, "dc2": 1}, spec_reqs=("vr", "vr2")
        )
        num_decode, num_windowed = runner._reorder_batch_for_rpa(sched)
        assert num_decode == 2
        assert num_windowed == 4
        ids = runner.input_batch.req_ids
        assert set(ids[:2]) == {"dc", "dc2"}  # decode segment
        assert set(ids[2:4]) == {"vr", "vr2"}  # verify segment
        assert ids[4] == "pf"  # prefill/mixed tail

    def test_all_verify_no_plain_decode(self):
        """Verify windows with no 1-token decodes: decode segment empty,
        windowed segment covers all verify requests."""
        runner = self._make_runner()
        self._batch(runner, ["pf", "vr", "vr2"])
        sched = self._sched({"pf": 10, "vr": 3, "vr2": 3}, spec_reqs=("vr", "vr2"))
        num_decode, num_windowed = runner._reorder_batch_for_rpa(sched)
        assert num_decode == 0
        assert num_windowed == 2
        assert set(runner.input_batch.req_ids[:2]) == {"vr", "vr2"}
        assert runner.input_batch.req_ids[2] == "pf"


class TestInitializeAttentionKernelsThreshold:
    """`reorder_batch_threshold` is the smallest DECODE query width across the
    TARGET kernels `_initialize_attention_kernels` builds.

    Draft layers are excluded, and are always pinned to a 1-token tile. A
    K+1-wide decode tile describes the target's verify step; a proposer runs
    one query token per draft step and builds its own pure-decode request
    distribution, so it never reads this threshold and must never widen it."""

    def _make_runner(self, widths, draft_names=(), draft_tp=2):
        from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

        runner = MagicMock(spec=TPUModelRunner)
        runner._attention_kernels_initialized = False
        runner.model = None
        runner.mesh = MagicMock()
        runner.vllm_config = MagicMock()
        runner.reorder_batch_threshold = 1
        runner.drafter = SimpleNamespace(_draft_attn_layer_names=set(draft_names))
        runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=draft_tp)
        runner._initialize_attention_kernels = (
            TPUModelRunner._initialize_attention_kernels.__get__(runner)
        )
        layers = {}
        for name, width in widths.items():
            impl = MagicMock(spec=PallasAttentionBackendImpl)
            impl.decode_query_size = width
            layers[name] = SimpleNamespace(impl=impl)
        return runner, layers

    def _run(self, runner, layers):
        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
                return_value=layers,
            ),
            patch("vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context"),
        ):
            runner._initialize_attention_kernels()

    def test_uniform_width(self):
        runner, layers = self._make_runner({"a": 4, "b": 4})
        self._run(runner, layers)
        assert runner.reorder_batch_threshold == 4
        for layer in layers.values():
            layer.impl.initialize_kernel.assert_called_once()

    def test_min_across_layers(self):
        runner, layers = self._make_runner({"a": 4, "b": 1})
        self._run(runner, layers)
        assert runner.reorder_batch_threshold == 1

    def test_replicated_draft_relocated_and_pinned_to_one_token(self):
        runner, layers = self._make_runner(
            {"target": 4, "draft": 4}, draft_names=("draft",), draft_tp=1
        )
        self._run(runner, layers)
        assert layers["draft"].impl.decode_query_size == 1
        assert layers["target"].impl.decode_query_size == 4
        # The draft's 1 must not leak into the threshold: it would disable the
        # RPAd verify lane for the target, which is unrelated to draft TP.
        assert runner.reorder_batch_threshold == 4

    def test_sharded_draft_also_pinned_to_one_token(self):
        """Regression: `PallasBatchedRPAAttentionBackendImpl.__init__` stamps
        K+1 on every layer it builds, draft included, because it only keys on
        "a spec config exists". A sharded draft left at K+1 mistiles its single
        query token and degenerates after one token -- while acceptance reads a
        perfect 100%, because target and draft agree on the same garbage."""
        runner, layers = self._make_runner(
            {"target": 4, "draft": 4}, draft_names=("draft",), draft_tp=2
        )
        self._run(runner, layers)
        assert layers["draft"].impl.decode_query_size == 1
        assert layers["target"].impl.decode_query_size == 4
        assert runner.reorder_batch_threshold == 4

    def test_no_pallas_layers(self):
        runner, layers = self._make_runner({})
        self._run(runner, layers)
        assert runner.reorder_batch_threshold == 1


class TestMambaSlotReadOffsets:
    """Unit tests for the per-slot mamba read-offset scatter used to roll back
    rejected speculative tokens by *selection* (base_slot + offset)."""

    def _make_runner(self, num_blocks=8):
        runner = MagicMock(spec=TPUModelRunner)
        runner.mamba_slot_read_offsets = torch.zeros(num_blocks, dtype=torch.int32)
        runner._update_mamba_slot_read_offsets = (
            TPUModelRunner._update_mamba_slot_read_offsets.__get__(runner)
        )
        return runner

    def test_noop_when_offsets_disabled(self):
        """No slot buffer (spec decode disabled) -> nothing to scatter."""
        runner = MagicMock(spec=TPUModelRunner)
        runner.mamba_slot_read_offsets = None
        runner._update_mamba_slot_read_offsets = (
            TPUModelRunner._update_mamba_slot_read_offsets.__get__(runner)
        )
        # Must not raise even with real-looking args.
        runner._update_mamba_slot_read_offsets(
            [torch.tensor([1, 4])], torch.tensor([[10, -1]]), 2
        )

    def test_verify_offsets_are_num_accepted_minus_one(self):
        """offset = (#valid tokens) - 1: full accept -> k, all-rejected -> 0."""
        runner = self._make_runner(num_blocks=8)
        state_indices = torch.tensor([1, 4], dtype=torch.int32)
        # req at slot 1 accepted 3 tokens (2 drafts + bonus); req at slot 4
        # accepted only the bonus (both drafts rejected).
        next_tokens = torch.tensor(
            [[10, 11, 12], [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]]
        )
        runner._update_mamba_slot_read_offsets([state_indices], next_tokens, 2)
        assert int(runner.mamba_slot_read_offsets[1]) == 2  # 3 valid - 1
        assert int(runner.mamba_slot_read_offsets[4]) == 0  # 1 valid - 1

    def test_non_verify_chunk_resets_offsets(self):
        """A None next_tokens (prefill / plain decode) resets to 0, so the
        next step resumes from the group base checkpoint."""
        runner = self._make_runner(num_blocks=8)
        runner.mamba_slot_read_offsets[1] = 2
        runner.mamba_slot_read_offsets[4] = 1
        runner._update_mamba_slot_read_offsets(
            [torch.tensor([1, 4], dtype=torch.int32)], None, 2
        )
        assert int(runner.mamba_slot_read_offsets[1]) == 0
        assert int(runner.mamba_slot_read_offsets[4]) == 0

    def test_padded_tail_writes_to_null_slot(self):
        """Padded positions carry slot 0 (the null block); their reset writes
        land there harmlessly and never touch an active group base."""
        runner = self._make_runner(num_blocks=8)
        runner.mamba_slot_read_offsets[1] = 2
        # num_reqs = 1 active (slot 1); position 1 is padding -> slot 0.
        state_indices = torch.tensor([1, 0], dtype=torch.int32)
        next_tokens = torch.tensor([[10, 11], [INVALID_TOKEN_ID, INVALID_TOKEN_ID]])
        runner._update_mamba_slot_read_offsets([state_indices], next_tokens, 1)
        assert int(runner.mamba_slot_read_offsets[1]) == 1  # 2 valid - 1
        # Null slot only ever gets 0; active groups untouched by the tail.
        assert int(runner.mamba_slot_read_offsets[0]) == 0

    def test_unified_per_group_indices_all_scattered(self):
        """Unified pool: each mamba group has its own state block per
        request; the same offsets land at every group's blocks."""
        runner = self._make_runner(num_blocks=16)
        group0 = torch.tensor([3, 7], dtype=torch.int32)
        group1 = torch.tensor([11, 5], dtype=torch.int32)
        next_tokens = torch.tensor(
            [[10, 11, 12], [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]]
        )
        runner._update_mamba_slot_read_offsets([group0, group1], next_tokens, 2)
        for block, expected in ((3, 2), (7, 0), (11, 2), (5, 0)):
            assert int(runner.mamba_slot_read_offsets[block]) == expected

    def test_unified_empty_group_list_is_noop(self):
        runner = self._make_runner(num_blocks=8)
        runner._update_mamba_slot_read_offsets([], torch.tensor([[1, 2]]), 1)
        assert int(runner.mamba_slot_read_offsets.sum()) == 0


class TestReadOffsetsClearedOnReallocation:
    """A recycled slot/block must not carry a read offset into its next owner.

    `mamba_slot_read_offsets` is indexed by slot/block id, not by request.
    When a request leaves mid-verify-window its `num_accepted - 1` stays in
    the buffer, and both layouts hand that storage to the next request --
    compact via `_free_mamba_slots`, unified via the block manager. The new
    owner's first forward would then resume from a checkpoint belonging to
    the previous conversation: silent GDN state corruption, no crash.
    """

    def _make_runner(self, num_blocks=8, width=2):
        runner = MagicMock(spec=TPUModelRunner)
        runner.mamba_slot_read_offsets = torch.zeros(num_blocks, dtype=torch.int32)
        runner._mamba_offset_seeded = set()
        runner._read_offset_scratch = {}
        runner.input_batch = MagicMock()
        runner._mamba_state_index_groups = (
            TPUModelRunner._mamba_state_index_groups.__get__(runner)
        )
        runner._read_offset_reset_scratch = (
            TPUModelRunner._read_offset_reset_scratch.__get__(runner)
        )
        runner._reset_read_offsets_for_new_requests = (
            TPUModelRunner._reset_read_offsets_for_new_requests.__get__(runner)
        )
        return runner

    def _set_batch(self, runner, req_ids, state_indices):
        runner.input_batch.req_ids = list(req_ids)
        runner.input_batch.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        ctx = MagicMock()
        ctx.mamba_state_indices = torch.tensor(state_indices, dtype=torch.int32)
        ctx.unified_mamba_state_indices = None
        runner._attn_metadata_builder_ctx = ctx

    def test_recycled_slot_does_not_inherit_stale_offset(self):
        runner = self._make_runner()
        # Request "a" owns slot 5 and ended a verify step having accepted 3
        # tokens, so slot 5 carries offset 2.
        self._set_batch(runner, ["a"], [5, 0])
        runner._reset_read_offsets_for_new_requests(0, 1)
        runner.mamba_slot_read_offsets[5] = 2

        # "a" leaves; a new conversation "b" is handed the same slot.
        self._set_batch(runner, ["b"], [5, 0])
        runner._reset_read_offsets_for_new_requests(0, 1)

        assert int(runner.mamba_slot_read_offsets[5]) == 0, (
            "new request inherited the previous request's checkpoint"
        )

    def test_continuing_request_keeps_its_offset(self):
        """The reset must not clobber a request still mid-window."""
        runner = self._make_runner()
        self._set_batch(runner, ["a"], [5, 0])
        runner._reset_read_offsets_for_new_requests(0, 1)
        runner.mamba_slot_read_offsets[5] = 2

        # Same request, next step: its offset is still live.
        runner._reset_read_offsets_for_new_requests(0, 1)
        assert int(runner.mamba_slot_read_offsets[5]) == 2

    def test_null_slot_absorbs_padded_and_seen_rows(self):
        runner = self._make_runner()
        self._set_batch(runner, ["a", "b"], [5, 6])
        runner._reset_read_offsets_for_new_requests(0, 2)
        runner.mamba_slot_read_offsets[5] = 2
        runner.mamba_slot_read_offsets[6] = 1

        # "a" stays, "c" replaces "b" on slot 6.
        self._set_batch(runner, ["a", "c"], [5, 6])
        runner._reset_read_offsets_for_new_requests(0, 2)
        assert int(runner.mamba_slot_read_offsets[5]) == 2  # untouched
        assert int(runner.mamba_slot_read_offsets[6]) == 0  # cleared
        assert int(runner.mamba_slot_read_offsets[0]) == 0  # null stays 0


class TestUnifiedReadOffsetMigration:
    """The align-mode seed-copy collector must migrate the per-block read
    offsets together with the state when a request's state block moves.

    These pin the no-checkpoint-group gate (`_mamba_ckpt_window == 1`),
    where the offset simply follows the state block. With a checkpoint
    group the group slides with the state column and the contract is
    different -- the resumed checkpoint moves and the offset resets to 0;
    that is covered in tests/runner/test_mamba_state_seed_copies.py.
    """

    def _make_runner(
        self, *, block_table, computed, scheduled, block_size=4, ckpt_window=1
    ):
        runner = MagicMock(spec=TPUModelRunner)
        runner.cache_config = MagicMock()
        runner.cache_config.block_size = block_size
        runner._mamba_ckpt_window = ckpt_window
        runner._bucket_len = TPUModelRunner._bucket_len
        runner._pad_to_bucket = TPUModelRunner._pad_to_bucket
        for name in ("_pad_dev_to_bucket", "_expand_pool_split", "_spec_seed_sources"):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
        # The collector strides the mamba block tables by the mamba groups'
        # spec block size (times the cp world), not by the attention block
        # size; with cp=1 the two coincide here.
        runner._mamba_state_block_size = block_size
        runner.device = torch.device("cpu")
        runner.mamba_slot_read_offsets = torch.zeros(16, dtype=torch.int32)
        runner._mamba_state_pos = {}
        runner._pool_block_split = 1
        runner._pending_mamba_state_copies = []
        raw = torch.zeros(16, 2)
        runner._mamba_copy_plan = [(0, [raw])]
        runner.input_batch = MagicMock()
        req_ids = [f"r{i}" for i in range(len(computed))]
        runner.input_batch.req_ids = req_ids
        runner.input_batch.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        runner.input_batch.num_computed_tokens_cpu = np.array(computed)
        bt_obj = MagicMock()
        bt_obj.get_cpu_tensor.return_value = torch.tensor(
            block_table, dtype=torch.int32
        )
        runner.input_batch.block_table = {0: bt_obj}
        runner._collect_mamba_state_seed_copies = (
            TPUModelRunner._collect_mamba_state_seed_copies.__get__(runner)
        )
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {
            rid: s for rid, s in zip(req_ids, scheduled)
        }
        return runner, scheduler_output

    def test_offsets_follow_state_block_on_crossing(self):
        # block_size=4: req r0 computed=7 scheduled=2 -> state block moves
        # from position 1 (block id 5) to position 2 (block id 9).
        runner, scheduler_output = self._make_runner(
            block_table=[[2, 5, 9, 0]], computed=[7], scheduled=[2]
        )
        runner.mamba_slot_read_offsets[5] = 3
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        assert int(runner.mamba_slot_read_offsets[9]) == 3
        # The state seed copy itself was staged for the same pair.
        assert len(runner._pending_mamba_state_copies) == 1
        _, src_t, dst_t = runner._pending_mamba_state_copies[0]
        assert int(src_t[0]) == 5 and int(dst_t[0]) == 9

    def test_no_crossing_leaves_offsets_alone(self):
        runner, scheduler_output = self._make_runner(
            block_table=[[2, 5, 9, 0]], computed=[5], scheduled=[2]
        )
        runner.mamba_slot_read_offsets[5] = 3
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        assert int(runner.mamba_slot_read_offsets[5]) == 3
        assert int(runner.mamba_slot_read_offsets[9]) == 0
        assert not runner._pending_mamba_state_copies

    def test_checkpoint_group_crossing_seeds_the_resumed_checkpoint(self):
        # With a checkpoint group the whole group slides with the state
        # column, so the block to carry over is the checkpoint the request
        # resumes from -- not the old state block, which would leave it
        # resuming one checkpoint late. It lands as checkpoint 0, so the
        # offset resets.
        runner, scheduler_output = self._make_runner(
            block_table=[[2, 5, 9, 11]], computed=[7], scheduled=[2], ckpt_window=3
        )
        # Pre-crossing group is columns 1..3 = blocks (5, 9, 11).
        runner.mamba_slot_read_offsets[5] = 2
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        _, src_t, dst_t = runner._pending_mamba_state_copies[0]
        assert int(src_t[0]) == 11 and int(dst_t[0]) == 9
        assert int(runner.mamba_slot_read_offsets[9]) == 0

    def test_rollback_crossing_migrates_offsets_backward(self):
        # A rejected verify window can pull the state position back into the
        # previous block: prev tracked position 2 (block 9), current step
        # lands in position 1 (block 5) -> offsets follow backward.
        runner, scheduler_output = self._make_runner(
            block_table=[[2, 5, 9, 0]], computed=[6], scheduled=[1]
        )
        runner._mamba_state_pos["r0"] = 2
        runner.mamba_slot_read_offsets[9] = 1
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        assert int(runner.mamba_slot_read_offsets[5]) == 1


def _block_size_for(architecture, backend_page_size=256, preferred=None):
    """Run update_block_size_for_backend for one model architecture.

    `preferred` mimics a backend whose get_preferred_block_size ignores its
    argument, as DeepseekSparseSWABackend's hardcoded 256 does.
    """
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            block_size_unaligned=16,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_hybrid=False,
            architecture=architecture,
            hf_config=SimpleNamespace(architectures=[architecture]),
        ),
    )
    backend_mock = MagicMock()
    backend_mock.get_min_page_size.return_value = 1
    backend_mock.get_page_size.return_value = backend_page_size
    backend_mock.get_preferred_block_size.side_effect = (
        (lambda d: preferred) if preferred is not None else (lambda d: d)
    )
    with patch.object(TpuPlatform, "_find_non_ssm_backend", return_value=backend_mock):
        TpuPlatform.update_block_size_for_backend(vllm_config)
    return vllm_config.cache_config.block_size


def test_block_size_resolution_needs_no_ambient_config(monkeypatch):
    """`update_block_size_for_backend` runs before vLLM makes the config
    current, so anything it reaches that resolves the KV layout has to
    establish the context itself.

    Deliberately takes no `vllm_config_context` and uses the real backend
    rather than a MagicMock: with either of those in place the call resolves
    the layout against an ambient config and the gap stays hidden. Missing it
    both ways is what let this reach CI as an `AssertionError: Current vLLM
    config is not set` at server startup."""
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackend

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(),
        model_config=SimpleNamespace(
            use_mla=False,
            is_hybrid=False,
            max_model_len=1024,
            architecture="Qwen3ForCausalLM",
            get_head_size=lambda: 128,
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        kv_transfer_config=None,
    )
    resolve_kv_cache_layout(vllm_config, [["LBNHC", "LBHNC"]])
    with patch.object(
        TpuPlatform,
        "_find_non_ssm_backend",
        return_value=PallasBatchedRPAAttentionBackend,
    ):
        TpuPlatform.update_block_size_for_backend(vllm_config)
    assert vllm_config.cache_config.block_size in (128, 256)


def test_tpu_platform_block_size_override_is_dsv4_only(vllm_config_context):
    """Only DSv4 takes the MLA page size; other MLA models keep vLLM's path.

    DSv4's own backend does not define get_page_size, so its size comes from
    PallasMLAttentionBackend. Applying that to every `use_mla` model would
    silently re-size DeepSeek-V2/V3 caches too.
    """
    assert _block_size_for("DeepseekV4ForCausalLM") == 1024
    assert _block_size_for("DeepseekV2ForCausalLM") == 256
    assert _block_size_for("DeepseekV32ForCausalLM") == 256

    # DSv4's own backend hardcodes get_preferred_block_size to 256, which does
    # not fit the packed latent record; DSv4 must not be clamped by it.
    assert _block_size_for("DeepseekV4ForCausalLM", preferred=256) == 1024


def test_spec_token_room_guard_caps_writes_at_context_limit():
    """The wrapped update_req_spec_token_ids clamps the staged write to the
    remaining token-buffer room (unclamped it overflows and kills the engine
    near max_model_len) while leaving the scheduled draft count untouched.

    The count has to survive: num_scheduled_tokens is already fixed at 1+K and
    get_spec_decode_metadata anchors the logits window at the end of each
    request's query, so a trimmed count slides the verify window forward and
    the sampler validates the wrong drafts.
    """
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            token_ids_cpu=np.zeros((4, 2048), dtype=np.int32),
            num_tokens_no_spec=np.array([10, 2038, 2048, 0], dtype=np.int32),
            req_id_to_index={"roomy": 0, "edge": 1, "full": 2},
            spec_token_ids=[[] for _ in range(4)],
        ),
        requests={},
    )
    calls = []
    runner.input_batch.update_req_spec_token_ids = (
        lambda request, scheduled: calls.append(
            (request.req_id, list(scheduled[request.req_id]))
        )
    )
    TPUModelRunner._install_spec_token_room_guard(runner)

    scheduled = {"roomy": [1] * 15, "edge": [2] * 15, "full": [3] * 15}
    requests = {
        rid: SimpleNamespace(req_id=rid, prev_num_draft_len=0)
        for rid in ("roomy", "edge", "full")
    }
    for rid in ("roomy", "edge", "full"):
        runner.input_batch.update_req_spec_token_ids(requests[rid], scheduled)

    # Only what fits is staged into token_ids_cpu.
    assert [(r, len(ids)) for r, ids in calls] == [
        ("roomy", 15),
        ("edge", 10),
        ("full", 0),
    ]
    # ...and the clamped write keeps the first drafts, not the last.
    assert calls[1][1] == [2] * 10
    # The scheduler's counts are restored, so the verify window stays aligned
    # with num_scheduled_tokens.
    assert [len(scheduled[r]) for r in ("roomy", "edge", "full")] == [15, 15, 15]
    assert [requests[r].prev_num_draft_len for r in ("roomy", "edge", "full")] == [
        0,
        15,
        15,
    ]
    assert [len(runner.input_batch.spec_token_ids[i]) for i in (1, 2)] == [15, 15]


class TestBuildAttentionMetadataForLayers:
    """Tests TPUModelRunner.build_attention_metadata_for_layers."""

    @staticmethod
    def _group(gid, layer_names):
        builder = MagicMock()
        builder.build.return_value = SimpleNamespace(tag=gid)
        return SimpleNamespace(
            layer_names=layer_names, kv_cache_group_id=gid, metadata_builders=[builder]
        )

    def _runner(self, groups):
        runner = MagicMock()
        runner.attn_groups = [[g] for g in groups]
        return runner

    def test_builds_only_the_groups_owning_the_requested_layers(self):
        wanted = self._group(0, ["draft.0", "draft.1"])
        other = self._group(1, ["target.0"])
        runner = self._runner([wanted, other])

        out = TPUModelRunner.build_attention_metadata_for_layers(
            runner, {"draft.0", "draft.1"}, 8
        )

        assert set(out) == {"draft.0", "draft.1"}
        # One build per group, shared by that group's layers.
        wanted.metadata_builders[0].build.assert_called_once()
        assert out["draft.0"] is out["draft.1"]
        # The untouched group costs nothing -- that is the whole point.
        other.metadata_builders[0].build.assert_not_called()

    def test_passes_the_padded_request_count_through(self):
        group = self._group(0, ["draft.0"])
        runner = self._runner([group])

        TPUModelRunner.build_attention_metadata_for_layers(runner, ["draft.0"], 32)

        kwargs = group.metadata_builders[0].build.call_args.kwargs
        assert kwargs["common_attn_metadata"].num_reqs == 32
        assert kwargs["common_prefix_len"] == 0

    def test_partial_overlap_returns_only_the_requested_layers(self):
        # A group can own layers the caller did not ask for; they must not
        # leak into the forward context.
        group = self._group(0, ["draft.0", "draft.1"])
        runner = self._runner([group])

        out = TPUModelRunner.build_attention_metadata_for_layers(runner, {"draft.0"}, 8)

        assert set(out) == {"draft.0"}

    def test_no_matching_layers_builds_nothing(self):
        group = self._group(0, ["target.0"])
        runner = self._runner([group])

        out = TPUModelRunner.build_attention_metadata_for_layers(runner, {"draft.0"}, 8)

        assert out == {}
        group.metadata_builders[0].build.assert_not_called()


class _FakeTpuEvent:
    instances: list["_FakeTpuEvent"] = []

    def __init__(self):
        self.recorded = False
        self.synchronized = 0
        _FakeTpuEvent.instances.append(self)

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized += 1


class TestInputStagingFence:
    """A chunk records a fence after uploading from the reusable host staging
    tensors; the next chunk synchronizes on it before rewriting them, once."""

    def test_record_then_wait_synchronizes_once(self, monkeypatch):
        _FakeTpuEvent.instances.clear()
        monkeypatch.setattr(
            torch, "tpu", SimpleNamespace(Event=_FakeTpuEvent), raising=False
        )
        runner = SimpleNamespace(_input_staging_fence=None)
        TPUModelRunner._record_input_staging_fence(runner)
        fence = runner._input_staging_fence
        assert fence is _FakeTpuEvent.instances[-1]
        assert fence.recorded
        TPUModelRunner._wait_input_staging_fence(runner)
        assert fence.synchronized == 1
        assert runner._input_staging_fence is None
        TPUModelRunner._wait_input_staging_fence(runner)
        assert fence.synchronized == 1

    def test_wait_without_fence_is_noop(self, monkeypatch):
        monkeypatch.setattr(
            torch, "tpu", SimpleNamespace(Event=_FakeTpuEvent), raising=False
        )
        runner = SimpleNamespace(_input_staging_fence=None)
        TPUModelRunner._wait_input_staging_fence(runner)
        assert runner._input_staging_fence is None
