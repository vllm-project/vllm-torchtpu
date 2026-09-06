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
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec, MLAAttentionSpec)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilder,
    AttentionMetadataBuilderContext)
from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner import utils as runner_utils_module
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID


def _attention_layer_mock():
    """An `Attention` mock whose `get_attn_backend()` returns a real backend.

    `get_kv_cache_spec` measures page padding through the backend the layer
    reports. A bare `MagicMock` returns a mock shape, which iterates as empty
    and silently makes that arithmetic meaningless.
    """
    layer = MagicMock(spec=Attention)
    layer.get_attn_backend.return_value = PallasAttentionBackend
    return layer


def test_spec_warmup_all_token_ids_matches_current_sequence_lengths():
    all_token_ids = tpu_runner._spec_warmup_all_token_ids(
        ["request-0", "request-1"], [7, 12])

    assert all_token_ids == {
        "request-0": [0] * 8,
        "request-1": [0] * 13,
    }


def test_suspend_kv_transfer_group_restores_agent_after_failure():
    connector = object()

    with patch.object(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector):
        with pytest.raises(RuntimeError, match="synthetic warmup failed"):
            with tpu_runner._suspend_kv_transfer_group():
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

    expectation = (pytest.raises(
        RuntimeError, match="PCP MTP prefill warmup failed after retry")
                   if raises_after_retry else contextlib.nullcontext())
    with patch.object(tpu_runner,
                      "_suspend_kv_transfer_group",
                      return_value=contextlib.nullcontext()), expectation:
        TPUModelRunner._warmup_pcp_mtp_prefill(runner)

    assert attempts.call_args_list == [
        ((4096, ), {
            "quiet": True
        }),
        ((4096, ), {}),
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
    runner._warmup_spec_decode_cleanup.assert_called_once_with(
        "__pcp_mtp_warmup__")


def _sub_indices(req_id_to_index_copy, req_ids, num_scheduled, spec_k=None):
    """Run _prepare_async_token_substitution_indices with a minimal fake self.
    req_id_to_index_copy holds each request's *position*; with spec_k set the
    builder uses a `1+spec_k` source stride. Returns (cur, source) index lists."""
    fake = SimpleNamespace(
        _pre_async_results=SimpleNamespace(
            req_id_to_index_copy=req_id_to_index_copy,
            spec_decode_num_rejected_tokens=(None
                                             if spec_k is None else object())),
        speculative_config=(None if spec_k is None else SimpleNamespace(
            num_speculative_tokens=spec_k)),
        _last_sequence_layout_plan=None,
        input_batch=SimpleNamespace(req_ids=req_ids))
    cur, src = TPUModelRunner._prepare_async_token_substitution_indices(
        fake,
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.array(num_scheduled, dtype=np.int32))
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

    with patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
               return_value=True), patch(
                   "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
                   return_value=connector):
        result = TPUModelRunner.get_finished_kv_transfers(
            runner, scheduler_output)

    assert result == ({"sent"}, {"loaded"}, {"jobs": []}, {41, 43}, 2, None)
    connector.get_finished.assert_called_once_with({"finished"})
    connector.get_block_ids_with_load_errors.assert_called_once_with()
    connector.get_block_ids_with_load_errors_group_index.assert_called_once_with(
    )
    connector.clear_connector_metadata.assert_called_once_with()


def test_no_forward_output_preserves_invalid_block_ids():
    runner = SimpleNamespace(
        maybe_setup_kv_connector=MagicMock(),
        get_finished_kv_transfers=MagicMock(return_value=(set(),
                                                          {"failed-load"},
                                                          None, {41, 43}, 2,
                                                          None)),
    )
    scheduler_output = SimpleNamespace()
    vllm_config = SimpleNamespace()

    with patch(
            "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
            return_value=False):
        if not tpu_runner._KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP:
            with pytest.raises(RuntimeError,
                               match="cache-group-scoped KV load failure"):
                TPUModelRunner.kv_connector_no_forward(runner,
                                                       scheduler_output,
                                                       vllm_config)
            return
        output = TPUModelRunner.kv_connector_no_forward(
            runner, scheduler_output, vllm_config)

    assert output.kv_connector_output.finished_recving == {"failed-load"}
    assert output.kv_connector_output.invalid_block_ids == {41, 43}
    assert output.kv_connector_output.invalid_block_group_index == 2


def test_build_kv_connector_output_supports_vllm_023():
    with patch.object(
            tpu_runner,
            "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP",
            False,
    ), patch.object(tpu_runner, "KVConnectorOutput") as output_cls:
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


def test_build_kv_connector_output_rejects_ambiguous_vllm_023_failure():
    with patch.object(
            tpu_runner,
            "_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP",
            False,
    ), pytest.raises(RuntimeError, match="cache-group-scoped KV load failure"):
        tpu_runner._build_kv_connector_output(
            finished_sending=None,
            finished_recving={"failed-load"},
            kv_connector_worker_meta=None,
            invalid_block_ids={41, 43},
            invalid_block_group_index=2,
        )


def _fake_phased_runner(additional_config=None,
                        torch_profiler_dir="/tmp/phased",
                        max_iterations=0,
                        delay_iterations=0,
                        profiler_rank=2,
                        profiler_world_size=4):
    """A stand-in for TPUModelRunner carrying only what phased profiling reads.

    No parallel_config: the phased profiler must not reach for it.
    """
    return SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=additional_config
                                    if additional_config is not None else {},
                                    profiler_config=SimpleNamespace(
                                        torch_profiler_dir=torch_profiler_dir,
                                        max_iterations=max_iterations,
                                        delay_iterations=delay_iterations)),
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
                'host_tracer_level': 2,
                'device_tracer_level': 1,
                'python_tracer_level': 1
            },
            advanced_opts={
                'tpu_trace_mode': 'TRACE_COMPUTE',
                'tpu_num_sparse_cores_to_trace': 1,
                'tpu_num_sparse_core_tiles_to_trace': 1
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
        runner = _init_phased_runner(monkeypatch,
                                     profiler_rank=9,
                                     profiler_world_size=16)
        with patch(
                "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        assert mock_profiler_cls.call_args.kwargs["worker_rank"] == 9
        assert mock_profiler_cls.call_args.kwargs["world_size"] == 16

    def test_falls_back_to_default_num_steps_when_max_iterations_unset(
            self, monkeypatch):
        runner = _init_phased_runner(monkeypatch,
                                     max_iterations=0,
                                     delay_iterations=0)
        with patch(
                "vllm_torchtpu.runner.tpu_runner.runner_utils.PhaseBasedProfiler"
        ) as mock_profiler_cls:
            TPUModelRunner.start_phased_profiling(runner)

        assert (
            mock_profiler_cls.call_args.kwargs["num_steps_to_profile_for"] ==
            runner_utils_module.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR)
        assert (
            mock_profiler_cls.call_args.kwargs["decode_kv_len_threshold"] ==
            runner_utils_module.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD)

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


class DummyMamba(MambaBase):

    def __init__(self):
        super().__init__()

    def get_kv_cache_spec(self, vllm_config):
        return MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
        )

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    def get_state_shape(self):
        return ((4, 128), (8, 64, 32))

    @property
    def mamba_type(self):
        return "dummy"


class _FakeReserveSpec:
    """Offloading-spec stand-in resolved via kv_connector_extra_config by
    utils.estimate_kv_connector_hbm_reserve (same lookup as vLLM's
    OffloadingSpecFactory)."""

    RESERVE_BYTES = 1024 * 1024 * 1024  # 1 GiB

    @classmethod
    def estimate_hbm_reserve_bytes(cls, vllm_config):
        return cls.RESERVE_BYTES


class TestTPURunner:

    def setup_method(self):
        self.mock_device = torch.device('cpu')

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
        self.runner.shared_kv_cache_layers = {}
        self.runner.runner_only_attn_layers = set()
        self.runner.enforce_eager = False
        self.runner.speculative_config = None
        # Slots per request in the mamba pool (1 without spec decode).
        self.runner._mamba_slot_stride = 1

        # Bind the actual methods to our mock
        self.runner._update_mamba_page_size_padded = TPUModelRunner._update_mamba_page_size_padded.__get__(
            self.runner)
        self.runner._maybe_set_num_blocks_override = TPUModelRunner._maybe_set_num_blocks_override.__get__(
            self.runner)
        self.runner._maybe_set_compact_mamba_num_blocks_override = (
            TPUModelRunner._maybe_set_compact_mamba_num_blocks_override.
            __get__(self.runner))
        # Both override paths read the budget through this helper; bind the
        # real one so the mock runner exercises the actual budget math
        # (compute_hbm_budget minus the KV-connector HBM reserve).
        self.runner._available_kv_cache_hbm = (
            TPUModelRunner._available_kv_cache_hbm.__get__(self.runner))
        # Compact-mamba state starts unset (matches real __init__).
        self.runner._mamba_num_blocks = None
        self.runner._uniform_mamba_layout = (vllm_config.kv_transfer_config
                                             is not None)
        self.runner._unified_kv_layout = False
        self.runner.kv_cache_raw_tensors = []
        self.runner.get_kv_cache_spec = TPUModelRunner.get_kv_cache_spec.__get__(
            self.runner)
        self.runner.initialize_kv_cache = TPUModelRunner.initialize_kv_cache.__get__(
            self.runner)
        self.runner._validate_shared_kv_cache_layout = (
            TPUModelRunner._validate_shared_kv_cache_layout)
        self.runner._maybe_add_kv_sharing_layers_to_kv_cache_groups = (
            TPUModelRunner._maybe_add_kv_sharing_layers_to_kv_cache_groups.
            __get__(self.runner))
        self.runner._add_shared_kv_cache_aliases = (
            TPUModelRunner._add_shared_kv_cache_aliases.__get__(self.runner))

        self._find_non_ssm_backend_patcher = patch.object(
            TpuPlatform,
            '_find_non_ssm_backend',
            return_value=PallasAttentionBackend)
        self._find_non_ssm_backend_patcher.start()

    def teardown_method(self):
        self._find_non_ssm_backend_patcher.stop()

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_update_mamba_page_size_padded(self, mock_mem_info,
                                           mock_get_page_size):
        layers = {}
        for i in range(1):
            mock_attn = _attention_layer_mock()
            mock_attn.num_kv_heads = 2
            mock_attn.head_size = 128
            layers[f'attn_{i}'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        self.runner._update_mamba_page_size_padded(layers)

        # Mamba unpadded size:
        # (4 * 128 * 2) + (8 * 64 * 32 * 4) = 1024 + 65536 = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 4096 + 199680 = 203776
        assert self.runner._hybrid_uniform_page_size_bytes == 203776
        assert self.runner.cache_config.mamba_page_size_padded == 203776

        # Compact-mamba sizing runs first and wins (it can read HBM here).
        # It caps mamba at max_num_reqs + 1 = 17 recurrent slots and gives the
        # rest of the budget to the attention pool.
        #   1 attn + 3 mamba layers -> group_size=min=1 (3 < 1*1.5 is False),
        #   num_attn_groups=1, num_mamba_groups=3.
        #   avail = 10GB * 0.9 = 9,663,676,416 (headroom pinned to 0)
        #   avail_per_tensor = avail // 1 = 9,663,676,416
        #   mamba_per_tensor = 3 * 17 * 66560 = 3,394,560
        #   attn_num_blocks =
        #       (9,663,676,416 - 3,394,560) // (1 * 4096) = 2,358,467
        mock_mem_info.assert_called_once()
        assert self.runner._mamba_num_blocks == 17
        assert self.runner.cache_config.num_gpu_blocks_override == 2358467

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(3 * 1024 * 1024, 3 * 1024 * 1024))
    def test_compact_mamba_sizing_raises_when_mamba_exceeds_budget(
            self, mock_mem_info, mock_get_page_size):
        """Mamba slots alone (3 * 17 * 66560 = 3,394,560 B) exceed the 3 MiB
        KV budget, so compact sizing cannot fit. Raise instead of falling back
        to the uniform layout, which pads every block to the mamba page and
        silently shrinks the block pool ~50x."""
        layers = {}
        mock_attn = _attention_layer_mock()
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        layers['attn_0'] = mock_attn
        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        with pytest.raises(ValueError, match="does not fit"):
            self.runner._update_mamba_page_size_padded(layers)

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_kv_connector_reserve_shrinks_num_blocks_override(
            self, mock_mem_info, mock_get_page_size):
        """The KV-connector HBM reserve (e.g. the offload H2D staging pool)
        must come out of the budget the block-count overrides size against,
        or the pinned blocks fill the reserve back up and defeat the
        worker-side subtraction. End-to-end through the same
        kv_connector_extra_config spec lookup the worker uses."""
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {
            "spec_name": _FakeReserveSpec.__name__,
            "spec_module_path": _FakeReserveSpec.__module__,
        }
        self.runner.vllm_config.kv_transfer_config = kv_tc

        layers = {}
        mock_attn = _attention_layer_mock()
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        layers['attn_0'] = mock_attn
        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        self.runner._update_mamba_page_size_padded(layers)

        # Same sizing as test_update_mamba_page_size_padded, but avail is
        # reduced by the 1 GiB reserve:
        #   avail = 10 GiB * 0.9 - 1 GiB = 8,589,934,592
        #   attn_num_blocks = (8,589,934,592 - 3 * 17 * 66560) // 4096
        #                   = 2,096,323  (vs 2,358,467 with no reserve)
        assert self.runner.cache_config.num_gpu_blocks_override == 2096323

    def test_maybe_setup_kv_connector_fences_preemptions_before_bind(self):
        """Upstream parity: handle_preemptions must run before
        bind_connector_metadata / start_load_kv so connectors with async
        saves (OffloadingConnector jobs_to_flush) can fence in-flight
        stores before the forward overwrites their source blocks."""
        self.runner.maybe_setup_kv_connector = (
            TPUModelRunner.maybe_setup_kv_connector.__get__(self.runner))
        connector = MagicMock()
        scheduler_output = MagicMock()
        meta = scheduler_output.kv_connector_metadata

        with patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                   return_value=True), \
                patch('vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group',
                      return_value=connector):
            self.runner.maybe_setup_kv_connector(scheduler_output)

        connector.handle_preemptions.assert_called_once_with(meta)
        connector.bind_connector_metadata.assert_called_once_with(meta)
        connector.start_load_kv.assert_called_once_with(None)
        names = [c[0] for c in connector.mock_calls]
        assert names.index('handle_preemptions') < names.index(
            'bind_connector_metadata') < names.index('start_load_kv')

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.tpu_bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_hybrid_duplication(self, mock_input_batch,
                                                    mock_has_kv_transfer,
                                                    mock_bind_kv_cache,
                                                    mock_get_page_size,
                                                    mock_get_shape):
        num_blocks = 100
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=uniform_size)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128), (8, 64, 32)],
                               dtypes=[torch.bfloat16, torch.float32],
                               page_size_padded=uniform_size)

        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"],
                             kv_cache_spec=mamba_spec),
        ]

        kv_cache_tensors = [
            KVCacheTensor(size=tensor_size, shared_by=layer_names)
        ]
        kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                        kv_cache_tensors=kv_cache_tensors,
                                        kv_cache_groups=kv_cache_groups)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)

        # One block table per kv_cache_group (the per-group dtype check
        # introduced with multi-group support iterates over every group).
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables

        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        # Compact-mamba sizing not exercised here (initialize_kv_cache is
        # called directly with a fixed config): mamba shares the uniform
        # num_blocks. _mamba_num_blocks=None selects that fallback path.
        self.runner._mamba_num_blocks = None
        self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind_kv_cache.assert_called_once()
        created_caches = mock_bind_kv_cache.call_args[0][0]

        assert len(created_caches) == 4

        # Mamba shares the uniform num_blocks (compact sizing not applied).
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert isinstance(created_caches[name], tuple)
            assert len(created_caches[name]) == 2
            assert created_caches[name][0].shape == (num_blocks, 4, 128)
            assert created_caches[name][1].shape == (num_blocks, 8, 64, 32)

        # Slot pool initialized to the (uniform) mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(num_blocks)

        assert isinstance(created_caches["attn.0"], torch.Tensor)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.tpu_bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_compact_mamba(self, mock_input_batch,
                                               mock_has_kv_transfer,
                                               mock_bind_kv_cache,
                                               mock_get_page_size,
                                               mock_get_shape):
        """Compact-mamba: when _mamba_num_blocks is set, mamba layers allocate
        exactly that many recurrent slots while attention keeps num_blocks, and
        the slot pool is initialized to _mamba_num_blocks."""
        num_blocks = 100
        mamba_num_blocks = 17  # max_num_reqs(16) + 1
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=uniform_size)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128), (8, 64, 32)],
                               dtypes=[torch.bfloat16, torch.float32],
                               page_size_padded=uniform_size)
        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"],
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                        kv_cache_tensors=[
                                            KVCacheTensor(
                                                size=tensor_size,
                                                shared_by=layer_names)
                                        ],
                                        kv_cache_groups=kv_cache_groups)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        self.runner._mamba_num_blocks = mamba_num_blocks
        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        # Mamba: only mamba_num_blocks slots; attention: full num_blocks.
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert created_caches[name][0].shape == (mamba_num_blocks, 4, 128)
            assert created_caches[name][1].shape == (mamba_num_blocks, 8, 64,
                                                     32)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)
        # Slot pool sized to the compact mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(
            mamba_num_blocks)

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_get_kv_cache_spec_hybrid_padded_size(self, mock_mem_info,
                                                  mock_get_page_size,
                                                  mock_get_layers):
        layers = {}

        mock_attn = _attention_layer_mock()
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None
        layers['attn_0'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        mock_get_layers.return_value = layers

        # Call the method
        kv_cache_specs = self.runner.get_kv_cache_spec()

        # Mamba unpadded size: (4 * 128 * 2) + (8 * 64 * 32 * 4) = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 203776
        expected_padded_size = 203776

        # Verify specs
        assert 'attn_0' in kv_cache_specs
        attn_spec = kv_cache_specs['attn_0']
        assert isinstance(attn_spec, FullAttentionSpec)
        assert attn_spec.page_size_padded == expected_padded_size

        for i in range(3):
            layer_name = f'mamba_{i}'
            assert layer_name in kv_cache_specs
            mamba_spec = kv_cache_specs[layer_name]
            assert isinstance(mamba_spec, MambaSpec)
            assert mamba_spec.page_size_padded == expected_padded_size

    @patch.object(TpuPlatform,
                  '_find_non_ssm_backend',
                  return_value=PallasMLAttentionBackend)
    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasMLAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=8192)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_get_kv_cache_spec_mla_mamba_hybrid(self, mock_mem_info,
                                                mock_get_page_size,
                                                mock_get_layers,
                                                mock_find_non_ssm_backend):
        mock_mla = MagicMock(spec=MLAAttention)
        mock_mla.num_kv_heads = 1
        mock_mla.head_size = 576
        layers = {'mla_0': mock_mla}
        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()
        mock_get_layers.return_value = layers

        kv_cache_specs = self.runner.get_kv_cache_spec()

        # uniform size = one 8192-byte MLA page + three 66560-byte states
        expected_padded_size = 207872
        assert isinstance(kv_cache_specs['mla_0'], MLAAttentionSpec)
        assert kv_cache_specs['mla_0'].page_size_padded == expected_padded_size
        for i in range(3):
            assert (kv_cache_specs[f'mamba_{i}'].page_size_padded ==
                    expected_padded_size)
        mock_get_page_size.assert_called()

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    def test_get_kv_cache_spec_pure_attention_no_cache_config_updates(
            self, mock_get_page_size, mock_get_layers):
        mock_attn = _attention_layer_mock()
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None

        layers = {'layer.0': mock_attn}
        mock_get_layers.return_value = layers

        with patch.object(self.runner,
                          '_update_mamba_page_size_padded') as mock_update:
            self.runner.get_kv_cache_spec()
            mock_update.assert_not_called()

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    def test_get_kv_cache_spec_records_shared_layers(self, mock_get_page_size,
                                                     mock_get_layers):
        owner = _attention_layer_mock()
        owner.attn_type = AttentionType.DECODER
        owner.num_kv_heads = 2
        owner.head_size = 128
        owner.kv_sharing_target_layer_name = None
        owner.sliding_window = None
        owner.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        shared = _attention_layer_mock()
        shared.attn_type = AttentionType.DECODER
        shared.num_kv_heads = 2
        shared.head_size = 128
        shared.kv_sharing_target_layer_name = "layer.0"
        shared.sliding_window = None
        shared.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        mock_get_layers.return_value = {
            "layer.0": owner,
            "layer.1": shared,
        }

        kv_cache_specs = self.runner.get_kv_cache_spec()

        assert set(kv_cache_specs) == {"layer.0"}
        assert self.runner.shared_kv_cache_layers == {"layer.1": "layer.0"}

    def _make_shared_pair(self, attn_window, target_window):
        attn = _attention_layer_mock()
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = attn_window
        attn.impl = SimpleNamespace(kv_cache_quantized_dtype=None)

        target = _attention_layer_mock()
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = target_window
        target.impl = SimpleNamespace(kv_cache_quantized_dtype=None)
        return attn, target

    def test_validate_shared_layout_skips_sliding_window_when_hma_disabled(
            self):
        # HMA disabled: specs are unified to full attention, so a window
        # mismatch is harmless and must not raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=None)
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=False)

    def test_validate_shared_layout_enforces_sliding_window_when_hma_enabled(
            self):
        # HMA enabled: sliding-window layers keep a smaller window-sized cache,
        # so a window mismatch means the layers disagree on storage -> raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=None)
        with pytest.raises(ValueError, match="sliding_window"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=True)

    def test_validate_shared_layout_matching_window_ok_when_hma_enabled(self):
        # HMA enabled but windows match: no mismatch, no raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=512)
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=True)

    def _make_fp8_shared_pair(self,
                              attn_quant,
                              target_quant,
                              attn_scale=1.0,
                              target_scale=1.0):
        attn = _attention_layer_mock()
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = None
        attn._k_scale_float = attn_scale
        attn._v_scale_float = attn_scale
        attn.impl = MagicMock(kv_cache_quantized_dtype=attn_quant)

        target = _attention_layer_mock()
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = None
        target._k_scale_float = target_scale
        target._v_scale_float = target_scale
        target.impl = MagicMock(kv_cache_quantized_dtype=target_quant)
        return attn, target

    def test_validate_shared_layout_fp8_matching_ok(self):
        # Both sides fp8 with matching scales/dtype: no raise.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", "fp8_e4m3")
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=False)

    def test_validate_shared_layout_fp8_mismatched_scales_raises(self):
        # Both fp8 but scales differ: dequantizing with the wrong scales
        # corrupts attention -> raise.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3",
                                                  "fp8_e4m3",
                                                  attn_scale=1.0,
                                                  target_scale=2.0)
        with pytest.raises(ValueError, match="k/v scales"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    def test_validate_shared_layout_fp8_target_only_raises(self):
        # Asymmetric: target is fp8, shared layer is not. The shared layer would
        # read the target's packed fp8 bytes as bf16 -> silent garbage. Must
        # raise instead of being skipped.
        attn, target = self._make_fp8_shared_pair(None, "fp8_e4m3")
        with pytest.raises(ValueError, match="kv cache"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    def test_validate_shared_layout_fp8_attn_only_raises(self):
        # Asymmetric the other way: shared layer is fp8, target is not.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", None)
        with pytest.raises(ValueError, match="kv cache"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch('vllm_torchtpu.utils.tpu_bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_binds_shared_layer_to_target_cache(
            self, mock_input_batch, mock_has_kv_transfer, mock_bind_kv_cache,
            mock_get_shape):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 100, shared_by=["layer.0"])
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec)
            ],
        )

        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 1
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.return_value.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch.return_value

        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        assert set(created_caches) == {"layer.0", "layer.1"}
        assert created_caches["layer.1"] is created_caches["layer.0"]
        group_names = self.runner.kv_cache_config.kv_cache_groups[
            0].layer_names
        assert group_names == ["layer.0", "layer.1"]
        assert self.runner.attn_groups[0][0].layer_names == [
            "layer.0", "layer.1"
        ]
        assert kv_cache_config.kv_cache_groups[0].layer_names == ["layer.0"]

    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_rejects_missing_shared_target(
            self, mock_input_batch, mock_has_kv_transfer):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["layer.0"])
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec)
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "missing.layer"}

        with pytest.raises(ValueError, match="target layer is missing"):
            self.runner.initialize_kv_cache(kv_cache_config)

    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_rejects_duplicate_shared_allocation(
            self, mock_input_batch, mock_has_kv_transfer):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["layer.0"]),
                KVCacheTensor(size=16384, shared_by=["layer.1"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec),
                KVCacheGroupSpec(layer_names=["layer.1"],
                                 kv_cache_spec=attn_spec),
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}

        with pytest.raises(ValueError, match="independent KV cache"):
            self.runner.initialize_kv_cache(kv_cache_config)

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
        self.runner._make_buffer = GPUModelRunner._make_buffer.__get__(
            self.runner)

        with patch.object(GPUModelRunner, '__init__', return_value=None), \
             patch('vllm_torchtpu.runner.tpu_runner._torch_tpu_wrapper',
                   side_effect=lambda: contextlib.nullcontext()), \
             patch('vllm_torchtpu.runner.tpu_runner._validate_libtpu_version'
                   ), \
             patch.object(TPUModelRunner,
                          '_create_mesh_for_parallelism',
                          return_value=MagicMock()):
            TPUModelRunner.__init__(self.runner,
                                    self.runner.vllm_config,
                                    self.mock_device,
                                    profiler_rank=0,
                                    profiler_world_size=1)

        assert self.runner.mrope_positions.cpu.dtype == torch.int32
        assert self.runner.mrope_positions.cpu.shape == (
            3, self.runner.max_num_tokens + 1)
        assert self.runner.mrope_positions.np.dtype == np.int32

    def test_initialize_kv_cache_block_major(self):
        """Verifies that block-major initialization allocates a single shared bundle backing all layer views."""
        num_blocks = 4
        # Byte-consistent page size: 16*4*1*128 bf16 elements = 16384 B to ensure exact uniform tiling.
        per_layer_shape = (16, 4, 1, 128)
        # FullAttentionSpec.real_page_size_bytes for (16, 2, 128, bf16) is
        # block_size * (k+v) * head_size * itemsize = 16 * 4 * 128 * 2 = 16384.
        per_layer_bytes = 16384
        per_layer_size = per_layer_bytes * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=per_layer_bytes,
                                      indexes_kv_by_block_stride=True)
        layer_names = ["attn.0", "attn.1", "attn.2"]
        # Uniform KV cache group representing all layers of a dense model.
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=attn_spec)
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(size=per_layer_size, shared_by=[n])
                for n in layer_names
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []
        # The block-major topology gates read parallel_config (PCP/PP).
        self.runner.parallel_config = self.runner.vllm_config.parallel_config

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch

        cross_layer_connector = MagicMock()
        cross_layer_connector.prefer_cross_layer_blocks = True
        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(num_blocks, ) + per_layer_shape), patch(
                    'vllm_torchtpu.utils.tpu_bind_kv_cache') as mock_bind, \
             patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                   return_value=False), \
             patch('vllm.v1.worker.kv_connector_model_runner_mixin.'
                   'has_kv_transfer_group', return_value=True), \
             patch('vllm.v1.worker.kv_connector_model_runner_mixin.'
                   'get_kv_transfer_group',
                   return_value=cross_layer_connector), \
             patch('vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV', True):
            self.runner.initialize_kv_cache(kv_cache_config)

        # Verify bundle tensor shape (num_blocks, num_layers, *per_layer_shape) and layer index mappings.
        assert isinstance(self.runner._kv_cache_bundle, torch.Tensor)
        expected_bundle_shape = ((num_blocks, len(layer_names)) +
                                 per_layer_shape)
        assert tuple(
            self.runner._kv_cache_bundle.shape) == expected_bundle_shape
        assert self.runner._kv_cache_bundle.dtype == torch.bfloat16
        assert self.runner._kv_cache_bundle_layer_index == {
            "attn.0": 0,
            "attn.1": 1,
            "attn.2": 2,
        }

        # Verify that bound per-layer views alias the underlying bundle storage via strided slices.
        created = mock_bind.call_args[0][0]
        for idx, name in enumerate(layer_names):
            view = created[name]
            assert isinstance(view, torch.Tensor)
            assert tuple(view.shape) == (num_blocks, ) + per_layer_shape
            # Storage aliasing: same underlying buffer as the bundle.
            assert (view.untyped_storage().data_ptr() ==
                    self.runner._kv_cache_bundle.untyped_storage().data_ptr())

        # Verify that mutating a per-layer strided view modifies the bundle in-place while isolating other layers.
        target = 1
        created[layer_names[target]].fill_(7)
        assert torch.all(self.runner._kv_cache_bundle[:, target] == 7), (
            "write through per-layer view did not mutate bundle storage")
        # Other layers untouched.
        for idx, name in enumerate(layer_names):
            if idx == target:
                continue
            assert torch.all(self.runner._kv_cache_bundle[:, idx] == 0), (
                f"layer {idx} disturbed by write to layer {target}")

    def test_initialize_kv_cache_block_major_rejects_mamba(self):
        """Verifies that standard dense block-major initialization fails closed on unsupported multi-group configurations."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128)],
                               dtypes=[torch.bfloat16],
                               page_size_padded=16384)
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["attn.0"]),
                KVCacheTensor(size=16384, shared_by=["mamba.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch
        self.runner._mamba_num_blocks = None

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(1, 16, 2, 1, 128)
        ), patch('vllm_torchtpu.utils.tpu_bind_kv_cache'), patch(
                'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                return_value=False), patch(
                    'vllm_torchtpu.envs.VLLM_TPU_BLOCK_MAJOR_KV',
                    True), pytest.raises(NotImplementedError,
                                         match="hybrid attention"):
            self.runner.initialize_kv_cache(kv_cache_config)

        assert self.runner._kv_cache_bundle is None
        assert self.runner._kv_cache_bundle_layer_index == {}

    def test_initialize_kv_cache_multi_group(self):
        """Multi-group initialize_kv_cache: verify the three new behaviors
        added with the builder migration: (1) attn_groups gets one shared
        TPU builder per kv_cache_group, (2) may_reinitialize_input_batch is
        called with per-group block_sizes, (3) empty_slot_mappings has one
        entry per group (consumed by upstream _build_attention_metadata)."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["attn.1"], kv_cache_spec=attn_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["attn.0"]),
                KVCacheTensor(size=16384, shared_by=["attn.1"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 4
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(1, 16, 2, 1, 128)
        ), patch('vllm_torchtpu.utils.tpu_bind_kv_cache'), patch(
                'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config)

        # (1) One shared TPU builder per group, in the parent's
        # list-of-list-of-AttentionGroup layout.
        assert len(self.runner.attn_groups) == len(kv_cache_groups)
        for gid, group_list in enumerate(self.runner.attn_groups):
            assert len(group_list) == 1
            grp = group_list[0]
            assert isinstance(grp, AttentionGroup)
            assert grp.backend is None
            assert grp.kv_cache_group_id == gid
            assert len(grp.metadata_builders) == 1
            assert isinstance(grp.metadata_builders[0],
                              AttentionMetadataBuilder)
            assert grp.metadata_builders[0].kv_cache_group_id == gid

        # (2) may_reinitialize_input_batch saw the per-group block_sizes
        # list (replacing the old explicit single-group InputBatch reinit).
        self.runner.may_reinitialize_input_batch.assert_called_once_with(
            kv_cache_config, [16, 16])

        # (3) empty_slot_mappings: one zero-element tensor per kv_cache_group.
        assert set(self.runner.empty_slot_mappings.keys()) == {0, 1}
        for tensor in self.runner.empty_slot_mappings.values():
            assert tensor.numel() == 0

    def test_initialize_kv_cache_composite_specs(self):
        """Verify composite kv_cache_specs unpacking when a group's spec wraps
        a dictionary of individual per-layer specs."""
        layer_spec = FullAttentionSpec(block_size=16,
                                       num_kv_heads=2,
                                       head_size=128,
                                       dtype=torch.bfloat16,
                                       page_size_padded=16384)
        composite_spec = MagicMock()
        composite_spec.kv_cache_specs = {"attn.0": layer_spec}

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"],
                             kv_cache_spec=composite_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["attn.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(1, 16, 2, 1, 128)
        ), patch('vllm_torchtpu.utils.tpu_bind_kv_cache') as mock_bind, patch(
                'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind.assert_called_once()
        created_caches = mock_bind.call_args[0][0]
        assert "attn.0" in created_caches
        assert created_caches["attn.0"].shape == (1, 16, 2, 1, 128)

    def test_initialize_kv_cache_num_blocks_override(self):
        """Verify explicit num_blocks override on kv_cache_config."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
        ]

        # num_blocks explicitly provided on kv_cache_config
        kv_cache_config_explicit = KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 10 + 123, shared_by=["attn.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(10, 16, 2, 1, 128)) as mock_get_shape, patch(
                    'vllm_torchtpu.utils.tpu_bind_kv_cache'
                ), patch(
                    'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                    return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config_explicit)
            mock_get_shape.assert_called_once_with(10, attn_spec.block_size,
                                                   attn_spec.num_kv_heads,
                                                   attn_spec.head_size,
                                                   attn_spec.dtype)

    def test_initialize_kv_cache_rejects_unresolved_num_blocks(self):
        # num_blocks is resolved by the scheduler before reaching the runner;
        # an unresolved config must fail loudly instead of being recomputed.
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=None,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 5, shared_by=["attn.0"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["attn.0"],
                                 kv_cache_spec=attn_spec),
            ],
        )

        with pytest.raises(AssertionError, match="num_blocks"):
            self.runner.initialize_kv_cache(kv_cache_config)


class TestAttentionMetadataBuilder:
    """Direct tests for AttentionMetadataBuilder.build, exercising the
    runner-state path used in _prepare_inputs and the position_ids_override
    path used in _dummy_run."""

    def _make_runner_mock(self,
                          most_model_len=None,
                          num_groups=1,
                          max_num_blocks_per_req=4,
                          max_num_reqs=4):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.block_size = 16
        runner.max_num_reqs = max_num_reqs
        runner.most_model_len = most_model_len
        runner._unified_kv_layout = False
        runner.position_ids = torch.full((8, ), 42, dtype=torch.int32)

        block_tables = []
        for gid in range(num_groups):
            bt = MagicMock()
            bt.max_num_blocks_per_req = max_num_blocks_per_req
            bt.get_cpu_tensor.return_value = (
                torch.arange(max_num_reqs * max_num_blocks_per_req,
                             dtype=torch.int32).reshape(
                                 max_num_reqs, max_num_blocks_per_req) +
                gid * 100)
            block_tables.append(bt)
        runner.input_batch.block_table = block_tables
        return runner

    def _make_builder(self, runner, kv_cache_group_id=0, spec=None):
        if spec is None:
            spec = FullAttentionSpec(block_size=16,
                                     num_kv_heads=2,
                                     head_size=128,
                                     dtype=torch.bfloat16,
                                     page_size_padded=16384)
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
            common_prefix_len=0,
            common_attn_metadata=self._make_cm(target_num_reqs))

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
        max_num_blocks = runner.input_batch.block_table[
            1].max_num_blocks_per_req
        block_tables_2d = meta.block_tables.reshape(target_num_reqs,
                                                    max_num_blocks)
        src = runner.input_batch.block_table[1].get_cpu_tensor.return_value
        assert torch.equal(block_tables_2d[:num_reqs],
                           src[start_index:start_index + num_reqs])
        assert torch.equal(
            block_tables_2d[num_reqs:],
            torch.zeros((target_num_reqs - num_reqs, max_num_blocks),
                        dtype=torch.int32))

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
            seq_lens=torch.ones((4, ), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            position_ids_override=override,
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert meta.input_positions is override
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        assert torch.equal(meta.block_tables,
                           torch.zeros((4 * 4, ), dtype=torch.int32))

    def test_build_most_model_len_shrinks_block_table(self):
        """When use_max_model_len is False, target_num_blocks =
        cdiv(most_model_len, block_size) — smaller than the per-group
        max_num_blocks_per_req, so the H2D copy is shorter."""
        runner = self._make_runner_mock(most_model_len=32,
                                        max_num_blocks_per_req=8)
        builder = self._make_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.tensor([8, 16, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        # cdiv(32, 16) = 2; flattened length = target_num_reqs * 2 = 8.
        assert meta.block_tables.shape == (4 * 2, )

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

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert torch.equal(meta.mamba_state_indices,
                           torch.tensor([0, 6, 0, 0], dtype=torch.int32))

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
                "vllm_torchtpu.layers.common.attention_metadata.get_dcp_group"
        ) as mock_dcp, patch(
                "vllm_torchtpu.layers.common.attention_metadata.get_pcp_group"
        ) as mock_pcp:
            mock_dcp.return_value.world_size = 4
            mock_pcp.return_value.world_size = 1
            builder = self._make_builder(runner, spec=mamba_spec)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([64, 65, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert builder.target_block_size == 64
        assert torch.equal(meta.mamba_state_indices,
                           torch.tensor([0, 5, 0, 0], dtype=torch.int32))


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
        runner.mamba_state_indices_cpu = torch.zeros(max_num_reqs,
                                                     dtype=torch.int32)
        runner.input_batch = MagicMock()
        # Bind the real methods.
        runner._init_mamba_slot_pool = (
            TPUModelRunner._init_mamba_slot_pool.__get__(runner))
        runner._build_mamba_state_indices = (
            TPUModelRunner._build_mamba_state_indices.__get__(runner))
        return runner

    def _set_batch(self, runner, req_ids):
        """Set the persistent batch to the given ordered req_ids."""
        runner.input_batch.req_ids = list(req_ids)
        runner.input_batch.req_id_to_index = {
            r: i
            for i, r in enumerate(req_ids)
        }

    def test_unique_slots_and_null_tail(self):
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)  # usable slots 1..4, slot 0 = null
        self._set_batch(runner, ["a", "b"])

        idx = runner._build_mamba_state_indices(start_index=0,
                                                num_reqs=2,
                                                target_num_reqs=4).cpu()
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

        idx = runner._build_mamba_state_indices(start_index=0,
                                                num_reqs=4,
                                                target_num_reqs=4).cpu()
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
            int(idx_chunk1[1])
        }
        assert slots == {1, 2, 3, 4}

    def test_spec_decode_stride_allocates_group_bases(self):
        """With speculative decoding each request owns a group of `stride`
        consecutive slots; the pool hands out only the group *bases*."""
        # num_spec = 2 -> stride 3; max_num_reqs = 3 -> 3*3 + 1 = 10 blocks.
        runner = self._make_runner(max_num_reqs=3,
                                   mamba_num_blocks=10,
                                   slot_stride=3)
        runner._init_mamba_slot_pool(10)
        # Group bases are 1, 4, 7 (slot 0 is the null block); the interior
        # checkpoint slots (2,3,5,6,8,9) are never handed out directly.
        assert sorted(runner._free_mamba_slots) == [1, 4, 7]

        self._set_batch(runner, ["a", "b", "c"])
        idx = runner._build_mamba_state_indices(0, 3, 3).cpu()
        assert sorted(int(x) for x in idx) == [1, 4, 7]

    def test_spec_decode_stride_one_matches_plain_layout(self):
        """stride 1 (spec decode disabled) reproduces the dense layout."""
        runner = self._make_runner(max_num_reqs=4,
                                   mamba_num_blocks=5,
                                   slot_stride=1)
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
        runner._reorder_batch_for_rpa = (
            TPUModelRunner._reorder_batch_for_rpa.__get__(runner))
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
            scheduled_spec_decode_tokens={r: object()
                                          for r in spec_reqs})

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
        sched = self._sched({
            "pf": 16,
            "vr": 3,
            "dc": 1,
            "vr2": 3,
            "dc2": 1
        },
                            spec_reqs=("vr", "vr2"))
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
        sched = self._sched({
            "pf": 10,
            "vr": 3,
            "vr2": 3
        },
                            spec_reqs=("vr", "vr2"))
        num_decode, num_windowed = runner._reorder_batch_for_rpa(sched)
        assert num_decode == 0
        assert num_windowed == 2
        assert set(runner.input_batch.req_ids[:2]) == {"vr", "vr2"}
        assert runner.input_batch.req_ids[2] == "pf"


class TestMambaSlotReadOffsets:
    """Unit tests for the per-slot mamba read-offset scatter used to roll back
    rejected speculative tokens by *selection* (base_slot + offset)."""

    def _make_runner(self, num_blocks=8):
        runner = MagicMock(spec=TPUModelRunner)
        runner.mamba_slot_read_offsets = torch.zeros(num_blocks,
                                                     dtype=torch.int32)
        runner._update_mamba_slot_read_offsets = (
            TPUModelRunner._update_mamba_slot_read_offsets.__get__(runner))
        return runner

    def test_noop_when_offsets_disabled(self):
        """No slot buffer (spec decode disabled) -> nothing to scatter."""
        runner = MagicMock(spec=TPUModelRunner)
        runner.mamba_slot_read_offsets = None
        runner._update_mamba_slot_read_offsets = (
            TPUModelRunner._update_mamba_slot_read_offsets.__get__(runner))
        # Must not raise even with real-looking args.
        runner._update_mamba_slot_read_offsets([torch.tensor([1, 4])],
                                               torch.tensor([[10, -1]]), 2)

    def test_verify_offsets_are_num_accepted_minus_one(self):
        """offset = (#valid tokens) - 1: full accept -> k, all-rejected -> 0."""
        runner = self._make_runner(num_blocks=8)
        state_indices = torch.tensor([1, 4], dtype=torch.int32)
        # req at slot 1 accepted 3 tokens (2 drafts + bonus); req at slot 4
        # accepted only the bonus (both drafts rejected).
        next_tokens = torch.tensor([[10, 11, 12],
                                    [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]])
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
            [torch.tensor([1, 4], dtype=torch.int32)], None, 2)
        assert int(runner.mamba_slot_read_offsets[1]) == 0
        assert int(runner.mamba_slot_read_offsets[4]) == 0

    def test_padded_tail_writes_to_null_slot(self):
        """Padded positions carry slot 0 (the null block); their reset writes
        land there harmlessly and never touch an active group base."""
        runner = self._make_runner(num_blocks=8)
        runner.mamba_slot_read_offsets[1] = 2
        # num_reqs = 1 active (slot 1); position 1 is padding -> slot 0.
        state_indices = torch.tensor([1, 0], dtype=torch.int32)
        next_tokens = torch.tensor([[10, 11],
                                    [INVALID_TOKEN_ID, INVALID_TOKEN_ID]])
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
        next_tokens = torch.tensor([[10, 11, 12],
                                    [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]])
        runner._update_mamba_slot_read_offsets([group0, group1], next_tokens,
                                               2)
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
        runner.mamba_slot_read_offsets = torch.zeros(num_blocks,
                                                     dtype=torch.int32)
        runner._mamba_offset_seeded = set()
        runner._read_offset_scratch = {}
        runner.input_batch = MagicMock()
        runner._mamba_state_index_groups = (
            TPUModelRunner._mamba_state_index_groups.__get__(runner))
        runner._read_offset_reset_scratch = (
            TPUModelRunner._read_offset_reset_scratch.__get__(runner))
        runner._reset_read_offsets_for_new_requests = (
            TPUModelRunner._reset_read_offsets_for_new_requests.__get__(runner)
        )
        return runner

    def _set_batch(self, runner, req_ids, state_indices):
        runner.input_batch.req_ids = list(req_ids)
        runner.input_batch.req_id_to_index = {
            r: i
            for i, r in enumerate(req_ids)
        }
        ctx = MagicMock()
        ctx.mamba_state_indices = torch.tensor(state_indices,
                                               dtype=torch.int32)
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
            "new request inherited the previous request's checkpoint")

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

    def _make_runner(self,
                     *,
                     block_table,
                     computed,
                     scheduled,
                     block_size=4,
                     ckpt_window=1):
        runner = MagicMock(spec=TPUModelRunner)
        runner.cache_config = MagicMock()
        runner.cache_config.block_size = block_size
        runner._mamba_ckpt_window = ckpt_window
        runner._bucket_len = TPUModelRunner._bucket_len
        runner._pad_to_bucket = TPUModelRunner._pad_to_bucket
        for name in ("_pad_dev_to_bucket", "_expand_pool_split",
                     "_spec_seed_sources"):
            setattr(runner, name,
                    getattr(TPUModelRunner, name).__get__(runner))
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
        runner.input_batch.req_id_to_index = {
            rid: i
            for i, rid in enumerate(req_ids)
        }
        runner.input_batch.num_computed_tokens_cpu = np.array(computed)
        bt_obj = MagicMock()
        bt_obj.get_cpu_tensor.return_value = torch.tensor(block_table,
                                                          dtype=torch.int32)
        runner.input_batch.block_table = {0: bt_obj}
        runner._collect_mamba_state_seed_copies = (
            TPUModelRunner._collect_mamba_state_seed_copies.__get__(runner))
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {
            rid: s
            for rid, s in zip(req_ids, scheduled)
        }
        return runner, scheduler_output

    def test_offsets_follow_state_block_on_crossing(self):
        # block_size=4: req r0 computed=7 scheduled=2 -> state block moves
        # from position 1 (block id 5) to position 2 (block id 9).
        runner, scheduler_output = self._make_runner(block_table=[[2, 5, 9,
                                                                   0]],
                                                     computed=[7],
                                                     scheduled=[2])
        runner.mamba_slot_read_offsets[5] = 3
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        assert int(runner.mamba_slot_read_offsets[9]) == 3
        # The state seed copy itself was staged for the same pair.
        assert len(runner._pending_mamba_state_copies) == 1
        _, src_t, dst_t = runner._pending_mamba_state_copies[0]
        assert int(src_t[0]) == 5 and int(dst_t[0]) == 9

    def test_no_crossing_leaves_offsets_alone(self):
        runner, scheduler_output = self._make_runner(block_table=[[2, 5, 9,
                                                                   0]],
                                                     computed=[5],
                                                     scheduled=[2])
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
            block_table=[[2, 5, 9, 11]],
            computed=[7],
            scheduled=[2],
            ckpt_window=3)
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
        runner, scheduler_output = self._make_runner(block_table=[[2, 5, 9,
                                                                   0]],
                                                     computed=[6],
                                                     scheduled=[1])
        runner._mamba_state_pos["r0"] = 2
        runner.mamba_slot_read_offsets[9] = 1
        runner._collect_mamba_state_seed_copies(scheduler_output, 0, 1)
        assert int(runner.mamba_slot_read_offsets[5]) == 1


def test_dsv4_layer_classification():
    """Every DSv4 layer is sorted by the kernel that reads its array.

    Mirrors the reference (`tpu_inference/runner/kv_cache_manager.py`):
    MLAAttentionSpec layers anchor the overlays, `*.compressor.state_cache`
    layers are placed afterwards, and SWA layers stay grouped because layers
    of one cache group share a block table. A layer matching none of the
    three is a layout change the overlay plan cannot absorb, so it raises
    rather than being silently left without an array.
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    def mla(compress_ratio, head_size=640):
        return MLAAttentionSpec(block_size=1024,
                                num_kv_heads=1,
                                head_size=head_size,
                                dtype=torch.uint8,
                                compress_ratio=compress_ratio,
                                alignment=None)

    def swa():
        return SlidingWindowMLASpec(block_size=128,
                                    num_kv_heads=1,
                                    head_size=640,
                                    dtype=torch.uint8,
                                    sliding_window=128,
                                    alignment=None)

    specs = {
        "m.layers.0.attn": mla(4),
        "m.layers.1.attn": mla(128),
        "m.layers.0.attn.indexer.k_cache": mla(4, head_size=256),
        "m.layers.0.attn.compressor.state_cache": swa(),
        "m.layers.0.attn.swa_cache": swa(),
        "m.layers.1.attn.swa_cache": swa(),
    }
    groups = [
        SimpleNamespace(layer_names=[
            "m.layers.0.attn", "m.layers.1.attn",
            "m.layers.0.attn.indexer.k_cache",
            "m.layers.0.attn.compressor.state_cache"
        ]),
        SimpleNamespace(layer_names=[
            "m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"
        ]),
    ]

    runner = MagicMock()
    runner.shared_kv_cache_layers = {}
    runner._is_ds_v4_swa_layer = TPUModelRunner._is_ds_v4_swa_layer
    runner._DS_V4_STATE_CACHE_SUFFIX = TPUModelRunner._DS_V4_STATE_CACHE_SUFFIX

    mla_names, swa_groups, state_names = TPUModelRunner._classify_ds_v4_layers(
        runner, SimpleNamespace(kv_cache_groups=groups), specs.__getitem__)

    assert mla_names == [
        "m.layers.0.attn", "m.layers.1.attn", "m.layers.0.attn.indexer.k_cache"
    ]
    assert state_names == ["m.layers.0.attn.compressor.state_cache"]
    # Grouped, not flattened: position within the group picks the host array.
    assert swa_groups == [[
        "m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"
    ]]

    stray = SimpleNamespace(layer_names=["m.layers.0.mystery"])
    with pytest.raises(ValueError, match="no known role"):
        TPUModelRunner._classify_ds_v4_layers(
            runner, SimpleNamespace(kv_cache_groups=groups + [stray]), {
                **specs, "m.layers.0.mystery": swa()
            }.__getitem__)


def test_dsv4_swa_overlay_shares_hosts_across_groups():
    """SWA caches take a host by position *within* their own cache group.

    Mirrors the reference: `swa_host_indices[position]` with `position`
    restarting at 0 for every group, so the one-SWA-layer-per-group layout
    vLLM actually produces puts every SWA cache on the *same* CSA NoPE array.
    That is safe -- distinct cache groups never own the same block ID at the
    same time -- and it is why `_validate_ds_v4_overlay` checks for collisions
    within a group only. An earlier design forced globally distinct hosts,
    which allocated arrays the reference does not.
    """
    csa_a = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)
    csa_b = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)

    # The MLA layers form one cache group; each SWA cache is its own group,
    # which is the shape vLLM actually produces here. A CSA layer and the SWA
    # cache overlaying its NoPE array therefore never share a block table.
    groups = [
        SimpleNamespace(layer_names=[f"m.layers.{i}.attn" for i in range(3)]),
    ] + [
        SimpleNamespace(layer_names=[f"m.layers.{i}.attn.swa_cache"])
        for i in range(3)
    ]
    # Positional-within-group assignment: every group's sole SWA layer is at
    # position 0, so all three share the first CSA NoPE array.
    kv_caches = {
        "m.layers.0.attn": (csa_a, torch.zeros(4, 2, 4, 128,
                                               dtype=torch.uint8)),
        "m.layers.1.attn": (csa_b, torch.zeros(4, 2, 4, 128,
                                               dtype=torch.uint8)),
        "m.layers.2.attn": torch.zeros(4, 16, 4, 128, dtype=torch.uint8),
        "m.layers.0.attn.swa_cache": csa_a,
        "m.layers.1.attn.swa_cache": csa_a,
        "m.layers.2.attn.swa_cache": csa_a,
    }
    TPUModelRunner._validate_ds_v4_overlay(
        SimpleNamespace(kv_cache_groups=groups), kv_caches)

    # Two layers of ONE group on one array is the case that does corrupt:
    # they share a block table, so they would write each other's pages.
    one_group = [
        SimpleNamespace(layer_names=[
            "m.layers.0.attn.swa_cache", "m.layers.1.attn.swa_cache"
        ])
    ]
    with pytest.raises(ValueError, match="same array"):
        TPUModelRunner._validate_ds_v4_overlay(
            SimpleNamespace(kv_cache_groups=one_group), kv_caches)


def test_dsv4_specs_are_exempt_from_tpu_normalization():
    """DSv4 specs must reach vLLM verbatim.

    Without the exemption these specs get a `page_size_padded` derived from the
    logical 1024-token `block_size` rather than `storage_block_size`. (The
    packed `uint8` dtype is separately preserved by `_normalize_one_spec`'s
    integer-MLA carve-out, so dtype alone does not distinguish the two paths.)
    """
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    from vllm_torchtpu.kv_cache_spec_normalizer import \
        normalize_kv_cache_specs_for_tpu

    # A DSv4-shaped compressed main-latent spec: 1024-token logical block
    # compressed 4:1, packed as uint8.
    ds_v4_spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        compress_ratio=4,
        alignment=None,
    )
    ds_v4_swa_spec = SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.uint8,
        sliding_window=128,
        alignment=None,
    )
    specs = {
        "model.layers.0.attn": ds_v4_spec,
        "model.layers.0.attn.swa_cache": ds_v4_swa_spec,
    }

    out = normalize_kv_cache_specs_for_tpu(
        specs,
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
        exempt_layers=set(specs),
    )

    for name, original in specs.items():
        assert out[name] is original, f"{name} was normalized"
        assert out[name].dtype == torch.uint8
        assert out[name].page_size_padded is None

    # Without the exemption the same specs are rewritten -- guards against the
    # exemption silently becoming a no-op.
    unexempt = normalize_kv_cache_specs_for_tpu(
        specs, torch.float8_e4m3fn, attention_backend=PallasAttentionBackend)
    assert unexempt["model.layers.0.attn"] is not ds_v4_spec
    assert unexempt["model.layers.0.attn"].page_size_padded is not None
    # A page sized from the logical block, not from the compressed rows: this
    # is exactly the over-allocation the exemption avoids.
    assert (unexempt["model.layers.0.attn"].page_size_padded
            > ds_v4_spec.block_size // ds_v4_spec.compress_ratio * 640)


def test_dsv4_groups_are_uniform_type_with_mixed_block_sizes():
    """vLLM hands DSv4 its groups wrapped in `UniformTypeKVCacheSpecs`.

    Two consequences for `initialize_kv_cache`, both exercised here: the group
    spec must be unwrapped before any `isinstance` type test, and block size is
    not uniform across groups, so it cannot be asserted equal.
    """
    from vllm.v1.core.kv_cache_utils import group_and_unify_kv_cache_specs
    from vllm.v1.kv_cache_interface import (AttentionSpec, MambaSpec,
                                            SlidingWindowMLASpec,
                                            UniformTypeKVCacheSpecs)

    # DSv4's real shape: a 1024-token main latent block compressed 4:1, and
    # SWA caches whose block size is that compressed row count (256), capped
    # by the window. Two window sizes give two distinct SWA groups.
    specs: dict = {}
    for i in range(2):
        specs[f"m.layers.{i}.attn"] = MLAAttentionSpec(
            block_size=1024,
            num_kv_heads=1,
            head_size=640,
            dtype=torch.uint8,
            compress_ratio=4,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
            alignment=None)
        for window, block in ((256, 256), (128, 128)):
            specs[f"m.layers.{i}.attn.swa_cache_w{window}"] = (
                SlidingWindowMLASpec(block_size=block,
                                     num_kv_heads=1,
                                     head_size=640,
                                     dtype=torch.uint8,
                                     sliding_window=window,
                                     cache_dtype_str="fp8_ds_mla",
                                     model_version="deepseek_v4",
                                     alignment=None))

    grouped = group_and_unify_kv_cache_specs(specs)
    assert grouped is not None, (
        "vLLM did not take its DeepseekV4 grouping path for DSv4-shaped specs")
    assert all(
        isinstance(g, UniformTypeKVCacheSpecs)
        for g in grouped), (f"expected UniformTypeKVCacheSpecs groups, got "
                            f"{[type(g).__name__ for g in grouped]}")

    # A bare isinstance test against the group spec sees neither AttentionSpec
    # nor MambaSpec, which is why the representative spec is peeked at.
    assert not any(isinstance(g, (AttentionSpec, MambaSpec)) for g in grouped)

    block_sizes = {g.block_size for g in grouped}
    assert len(block_sizes) > 1, (
        f"expected mixed block sizes across DSv4 groups, got {block_sizes}")


def test_dsv4_is_cache_for_ds_v4_predicate():
    """The exemption must cover all four DSv4 cache-owning module types."""
    from vllm.models.deepseek_v4.attention import (DeepseekV4Attention,
                                                   DeepseekV4IndexerCache)
    from vllm.models.deepseek_v4.compressor import CompressorStateCache
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    from vllm_torchtpu.runner.tpu_runner import is_cache_for_ds_v4

    for cls in (DeepseekV4Attention, DeepseekV4SWACache,
                DeepseekV4IndexerCache, CompressorStateCache):
        assert is_cache_for_ds_v4(MagicMock(spec=cls)), cls.__name__

    assert not is_cache_for_ds_v4(MagicMock(spec=Attention))
    assert not is_cache_for_ds_v4(MagicMock(spec=MLAAttention))


def test_dsv4_compressor_state_cache_overlay_target():
    """A compressor state cache is named after the layer whose records it joins.

    Mirrors how vLLM wires `DeepseekCompressor.k_cache_prefix`: the main
    compressor's state rides on its own attention layer's array, the indexer
    compressor's on the indexer's `k_cache`. The mapping is derived from the
    layer NAME before anything is allocated -- an overlaid cache that gets its
    own array still costs its HBM at peak, even if aliased away afterwards.
    """
    assert TPUModelRunner._ds_v4_compressed_kv_layer_name(
        "model.layers.0.self_attn.compressor.state_cache"
    ) == "model.layers.0.self_attn"

    assert TPUModelRunner._ds_v4_compressed_kv_layer_name(
        "model.layers.0.attn.indexer.compressor.state_cache"
    ) == "model.layers.0.attn.indexer.k_cache"

    # A renamed state cache would otherwise derive a host that does not exist
    # and overlay onto whatever happened to be there.
    with pytest.raises(ValueError, match="unexpected layer"):
        TPUModelRunner._ds_v4_compressed_kv_layer_name(
            "model.layers.0.self_attn.compressor.state")


def test_dsv4_state_cache_overlays_its_own_compressed_kv_array():
    """CSA / indexer states share their own layer's array; HCA states do not.

    An HCA page holds two token states, far too small to host the f32 state
    rows, so those overlay a CSA NoPE array instead. Both cases must survive
    the group collision check: the state cache and its compressed-KV layer are
    in *different* cache groups, which is what makes sharing one array safe.
    """
    csa_nope = torch.zeros(4, 8, 4, 128, dtype=torch.uint8)
    csa_rope = torch.zeros(4, 2, 4, 128, dtype=torch.uint8)
    indexer = torch.zeros(4, 2, 4, 256, dtype=torch.uint8)
    hca = torch.zeros(4, 16, 4, 128, dtype=torch.uint8)

    kv_caches = {
        "m.layers.0.attn": (csa_nope, csa_rope),
        "m.layers.0.attn.indexer.k_cache": indexer,
        "m.layers.1.attn": hca,
        # CSA and indexer states: their own layer's array.
        "m.layers.0.attn.compressor.state_cache": csa_nope,
        "m.layers.0.attn.indexer.compressor.state_cache": indexer,
        # HCA state: a CSA NoPE array, its own page being too small.
        "m.layers.1.attn.compressor.state_cache": csa_nope,
    }
    groups = [
        SimpleNamespace(layer_names=[
            "m.layers.0.attn", "m.layers.0.attn.indexer.k_cache",
            "m.layers.1.attn"
        ]),
        SimpleNamespace(layer_names=[
            "m.layers.0.attn.compressor.state_cache",
            "m.layers.0.attn.indexer.compressor.state_cache",
        ]),
        SimpleNamespace(
            layer_names=["m.layers.1.attn.compressor.state_cache"]),
    ]
    TPUModelRunner._validate_ds_v4_overlay(
        SimpleNamespace(kv_cache_groups=groups), kv_caches)

    # The CSA layer is matched on its NoPE array, not its RoPE companion, so a
    # same-group layer landing on that NoPE array is still caught.
    clash = [
        SimpleNamespace(
            layer_names=["m.layers.0.attn", "m.layers.0.attn.swa_cache"])
    ]
    with pytest.raises(ValueError, match="same array"):
        TPUModelRunner._validate_ds_v4_overlay(
            SimpleNamespace(kv_cache_groups=clash), {
                **kv_caches, "m.layers.0.attn.swa_cache": csa_nope
            })


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
    backend_mock.get_preferred_block_size.side_effect = ((
        lambda d: preferred) if preferred is not None else (lambda d: d))
    with patch.object(TpuPlatform,
                      "_find_non_ssm_backend",
                      return_value=backend_mock):
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
    from vllm_torchtpu.layers.vllm.attention import \
        PallasBatchedRPAAttentionBackend
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            block_size_unaligned=16,
        ),
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
    with patch.object(TpuPlatform,
                      "_find_non_ssm_backend",
                      return_value=PallasBatchedRPAAttentionBackend):
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
            req_id_to_index={
                "roomy": 0,
                "edge": 1,
                "full": 2
            },
            spec_token_ids=[[] for _ in range(4)],
        ),
        requests={},
    )
    calls = []
    runner.input_batch.update_req_spec_token_ids = (
        lambda request, scheduled: calls.append(
            (request.req_id, list(scheduled[request.req_id]))))
    TPUModelRunner._install_spec_token_room_guard(runner)

    scheduled = {"roomy": [1] * 15, "edge": [2] * 15, "full": [3] * 15}
    requests = {
        rid: SimpleNamespace(req_id=rid, prev_num_draft_len=0)
        for rid in ("roomy", "edge", "full")
    }
    for rid in ("roomy", "edge", "full"):
        runner.input_batch.update_req_spec_token_ids(requests[rid], scheduled)

    # Only what fits is staged into token_ids_cpu.
    assert [(r, len(ids)) for r, ids in calls] \
        == [("roomy", 15), ("edge", 10), ("full", 0)]
    # ...and the clamped write keeps the first drafts, not the last.
    assert calls[1][1] == [2] * 10
    # The scheduler's counts are restored, so the verify window stays aligned
    # with num_scheduled_tokens.
    assert [len(scheduled[r]) for r in ("roomy", "edge", "full")] \
        == [15, 15, 15]
    assert [requests[r].prev_num_draft_len
            for r in ("roomy", "edge", "full")] == [0, 15, 15]
    assert [len(runner.input_batch.spec_token_ids[i]) for i in (1, 2)] \
        == [15, 15]


class TestUpdateAttentionPageSizePadded:
    """_update_attention_page_size_padded must size pages with the caller's
    block_size instead of self.block_size, which is initialized at __init__
    time. This is because for hybrid models the update is skipped during that
    time. It only gets updated after the model is loaded, so using
    self.block_size will cause the unify_kv_cache_spec_page_size
    `page_size_padded >= real_page_size_bytes` assertion.
    """

    @pytest.fixture(autouse=True)
    def mock_non_ssm_backend(self):
        with patch.object(TpuPlatform,
                          '_find_non_ssm_backend',
                          return_value=PallasAttentionBackend):
            yield

    @staticmethod
    def _attn(num_kv_heads, head_size):
        layer = MagicMock(spec=Attention)
        layer.num_kv_heads = num_kv_heads
        layer.head_size = head_size
        return layer

    @staticmethod
    def _stub(stale_block_size=16):
        return SimpleNamespace(
            block_size=stale_block_size,
            kv_cache_dtype=torch.bfloat16,
            cache_config=SimpleNamespace(mamba_page_size_padded=None),
            _hybrid_uniform_page_size_bytes=None,
            vllm_config=MagicMock(),
        )

    def test_uses_caller_block_size_not_stale_snapshot(self):
        # gemma-4 E4B geometry: sliding head_dim 256 vs global head_dim 512.
        layers = {
            "sliding": self._attn(num_kv_heads=2, head_size=256),
            "full": self._attn(num_kv_heads=2, head_size=512),
        }
        stub = self._stub(stale_block_size=16)

        TPUModelRunner._update_attention_page_size_padded(stub, layers, 256)

        expected = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            256, 2, 512, torch.bfloat16)
        stale = PallasAttentionBackend.get_kv_cache_page_size_bytes(
            16, 2, 512, torch.bfloat16)
        assert stub._hybrid_uniform_page_size_bytes == expected
        assert stub._hybrid_uniform_page_size_bytes != stale
        assert stub.cache_config.mamba_page_size_padded == expected

    def test_uniform_layers_leave_pinning_unset(self):
        layers = {
            "a": self._attn(num_kv_heads=2, head_size=256),
            "b": self._attn(num_kv_heads=2, head_size=256),
        }
        stub = self._stub()

        TPUModelRunner._update_attention_page_size_padded(stub, layers, 256)

        assert stub._hybrid_uniform_page_size_bytes is None
        assert stub.cache_config.mamba_page_size_padded == \
            PallasAttentionBackend.get_kv_cache_page_size_bytes(
                256, 2, 256, torch.bfloat16)


class TestBuildAttentionMetadataForLayers:
    """Tests TPUModelRunner.build_attention_metadata_for_layers."""

    @staticmethod
    def _group(gid, layer_names):
        builder = MagicMock()
        builder.build.return_value = SimpleNamespace(tag=gid)
        return SimpleNamespace(layer_names=layer_names,
                               kv_cache_group_id=gid,
                               metadata_builders=[builder])

    def _runner(self, groups):
        runner = MagicMock()
        runner.attn_groups = [[g] for g in groups]
        return runner

    def test_builds_only_the_groups_owning_the_requested_layers(self):
        wanted = self._group(0, ["draft.0", "draft.1"])
        other = self._group(1, ["target.0"])
        runner = self._runner([wanted, other])

        out = TPUModelRunner.build_attention_metadata_for_layers(
            runner, {"draft.0", "draft.1"}, 8)

        assert set(out) == {"draft.0", "draft.1"}
        # One build per group, shared by that group's layers.
        wanted.metadata_builders[0].build.assert_called_once()
        assert out["draft.0"] is out["draft.1"]
        # The untouched group costs nothing -- that is the whole point.
        other.metadata_builders[0].build.assert_not_called()

    def test_passes_the_padded_request_count_through(self):
        group = self._group(0, ["draft.0"])
        runner = self._runner([group])

        TPUModelRunner.build_attention_metadata_for_layers(
            runner, ["draft.0"], 32)

        kwargs = group.metadata_builders[0].build.call_args.kwargs
        assert kwargs["common_attn_metadata"].num_reqs == 32
        assert kwargs["common_prefix_len"] == 0

    def test_partial_overlap_returns_only_the_requested_layers(self):
        # A group can own layers the caller did not ask for; they must not
        # leak into the forward context.
        group = self._group(0, ["draft.0", "draft.1"])
        runner = self._runner([group])

        out = TPUModelRunner.build_attention_metadata_for_layers(
            runner, {"draft.0"}, 8)

        assert set(out) == {"draft.0"}

    def test_no_matching_layers_builds_nothing(self):
        group = self._group(0, ["target.0"])
        runner = self._runner([group])

        out = TPUModelRunner.build_attention_metadata_for_layers(
            runner, {"draft.0"}, 8)

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
        monkeypatch.setattr(torch,
                            "tpu",
                            SimpleNamespace(Event=_FakeTpuEvent),
                            raising=False)
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
        monkeypatch.setattr(torch,
                            "tpu",
                            SimpleNamespace(Event=_FakeTpuEvent),
                            raising=False)
        runner = SimpleNamespace(_input_staging_fence=None)
        TPUModelRunner._wait_input_staging_fence(runner)
        assert runner._input_staging_fence is None
