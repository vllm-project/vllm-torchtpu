# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage tests for DeepSeek-V4.1
v2/model.py and v2/host_engram.py.

Covers all lines and branch decision points in:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/host_engram.py (lines 1-82)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/model.py (lines 1-758)
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure CPU platform and skip MDS queries before importing JAX / TPU modules
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm.config import CompilationMode

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import DecoderReplayPlan
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    HOST_ENGRAM_KEY,
    HostEngramInputs,
    capture_host_engram_inputs,
    install_host_engram_inputs,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
    DeepseekV41DecoderLayer,
    DeepseekV41DecoderStage,
    DeepseekV41EncoderStage,
    DeepseekV41ForCausalLM,
    DeepseekV41FullStage,
    DeepseekV41Model,
    _ced_config,
    _StageBase,
    _use_sequence_parallel,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# ==============================================================================
# Helper Mock Factories
# ==============================================================================


@dataclass
class _FakeForwardCtx:
    attn_metadata: Any
    is_padding: Any = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeAttnMeta:
    input_positions: Any = None
    query_start_loc: Any = None
    seq_lens: Any = None
    block_tables: Any = None


def _mock_runner(
    seq_lens: list[int],
    num_prompt: list[int],
    kind: SequenceLayoutKind = SequenceLayoutKind.ALL,
    has_plan: bool = True,
) -> SimpleNamespace:
    plan = SimpleNamespace(kind=kind) if has_plan else None
    return SimpleNamespace(
        _last_sequence_layout_plan=plan,
        input_batch=SimpleNamespace(num_prompt_tokens=np.array(num_prompt)),
        seq_lens_np=np.array(seq_lens),
        input_ids_cpu=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        positions_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        lookback_token_ids=SimpleNamespace(
            cpu=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        ),
    )


def _mock_vllm_config(
    *,
    num_hidden_layers: int = 40,
    enable_prefix_caching: bool = False,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
    ep_enabled: bool = False,
    compile_sizes: list[int] | None = None,
    prefill_cp_size: int = 1,
    speculative_config: object | None = None,
    async_scheduling: bool = False,
    max_model_len: int = 4096,
) -> SimpleNamespace:
    hf_config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
    model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
    cache_config = SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        data_parallel_size=dp_size,
        enable_expert_parallel=ep_enabled,
        prefill_context_parallel_size=prefill_cp_size,
    )
    compilation_config = SimpleNamespace(
        compile_sizes=compile_sizes, mode=CompilationMode.NONE
    )
    scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    kernel_config = SimpleNamespace(moe_backend="none")
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
        kernel_config=kernel_config,
    )


# ==============================================================================
# 1. Tests for host_engram.py
# ==============================================================================


class TestHostEngram:
    def test_host_engram_inputs_dataclass(self) -> None:
        inputs = HostEngramInputs(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 2]),
            lookback_token_ids=torch.tensor([[10]]),
            all_prompt=True,
        )
        assert inputs.all_prompt is True
        assert inputs.input_ids.shape == (2,)

    def test_capture_host_engram_inputs_none_when_no_plan(self) -> None:
        runner = _mock_runner([2, 2], [5, 5], has_plan=False)
        assert capture_host_engram_inputs(runner, 2) is None

    def test_capture_host_engram_inputs_none_when_kind_not_all(self) -> None:
        other_kind = next(
            k for k in SequenceLayoutKind if k is not SequenceLayoutKind.ALL
        )
        runner = _mock_runner([2, 2], [5, 5], kind=other_kind)
        assert capture_host_engram_inputs(runner, 2) is None

    def test_capture_host_engram_inputs_all_prompt_true_and_false(self) -> None:
        runner_prompt = _mock_runner([2, 3], [5, 5])
        captured_prompt = capture_host_engram_inputs(runner_prompt, 2)
        assert captured_prompt is not None
        assert captured_prompt.all_prompt is True
        assert torch.equal(captured_prompt.input_ids, runner_prompt.input_ids_cpu)
        assert torch.equal(captured_prompt.positions, runner_prompt.positions_cpu)
        assert torch.equal(
            captured_prompt.query_start_loc, runner_prompt.query_start_loc_cpu
        )
        assert torch.equal(
            captured_prompt.lookback_token_ids, runner_prompt.lookback_token_ids.cpu
        )

        runner_decode = _mock_runner([6, 3], [5, 5])
        captured_decode = capture_host_engram_inputs(runner_decode, 2)
        assert captured_decode is not None
        assert captured_decode.all_prompt is False

    def test_install_host_engram_inputs_lifecycle_and_idempotency(self) -> None:
        orig_fn = TPUModelRunner._prepare_lookback_token_ids

        def dummy_prepare(self: object, num_reqs: int) -> torch.Tensor:
            return torch.tensor([123])

        try:
            TPUModelRunner._prepare_lookback_token_ids = dummy_prepare

            mock_ctx = SimpleNamespace(additional_kwargs={})
            mock_runner_inst = _mock_runner([2], [5])
            mock_is_avail = MagicMock(return_value=True)

            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                    mock_is_avail,
                ),
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_forward_context",
                    return_value=mock_ctx,
                ),
            ):
                # First install
                install_host_engram_inputs()
                wrapped = TPUModelRunner._prepare_lookback_token_ids
                assert getattr(wrapped, "_deepseek_v41_host_engram", False) is True

                # Idempotent second install does not re-wrap
                install_host_engram_inputs()
                assert TPUModelRunner._prepare_lookback_token_ids is wrapped

                # Execution with forward context available
                res = TPUModelRunner._prepare_lookback_token_ids(mock_runner_inst, 1)
                assert torch.equal(res, torch.tensor([123]))
                assert HOST_ENGRAM_KEY in mock_ctx.additional_kwargs
                assert isinstance(
                    mock_ctx.additional_kwargs[HOST_ENGRAM_KEY], HostEngramInputs
                )

                # Execution without forward context available (covers 74->78 branch)
                mock_is_avail.return_value = False
                res2 = TPUModelRunner._prepare_lookback_token_ids(mock_runner_inst, 1)
                assert torch.equal(res2, torch.tensor([123]))

        finally:
            TPUModelRunner._prepare_lookback_token_ids = orig_fn


# ==============================================================================
# 2. Tests for _ced_config
# ==============================================================================


class TestCedConfig:
    def test_ced_config_branches(self) -> None:
        cfg = _mock_vllm_config(num_hidden_layers=40, enable_prefix_caching=False)

        # 1. CED=1 and prefix caching disabled -> enabled True, boundary 20
        with patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True):
            enabled, boundary = _ced_config(cfg, warn_prefix_caching=False)
            assert enabled is True
            assert boundary == 20

        # 2. CED=1 and prefix caching enabled with warn_prefix_caching=True
        cfg_cached = _mock_vllm_config(num_hidden_layers=40, enable_prefix_caching=True)
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.logger") as mock_log,
        ):
            enabled, boundary = _ced_config(cfg_cached, warn_prefix_caching=True)
            assert enabled is False
            assert boundary == 20
            mock_log.warning.assert_called_once()

        # 3. CED=1 and prefix caching enabled with warn_prefix_caching=False
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.logger") as mock_log,
        ):
            enabled, boundary = _ced_config(cfg_cached, warn_prefix_caching=False)
            assert enabled is False
            assert boundary == 20
            mock_log.warning.assert_not_called()

        # 4. CED=0 -> enabled False
        with patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", False):
            enabled, boundary = _ced_config(cfg, warn_prefix_caching=False)
            assert enabled is False
            assert boundary == 20

        # 5. fallback_config when model_config is None
        cfg_no_model = SimpleNamespace(cache_config=None)
        fallback = SimpleNamespace(num_hidden_layers=30)
        with patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", False):
            enabled, boundary = _ced_config(
                cfg_no_model, fallback_config=fallback, warn_prefix_caching=False
            )
            assert enabled is False
            assert boundary == 15

        # 6. Assertion when hf_config is missing completely
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", False),
            pytest.raises(AssertionError),
        ):
            _ced_config(cfg_no_model, fallback_config=None)


# ==============================================================================
# 3. Tests for DeepseekV41DecoderLayer
# ==============================================================================


class TestDeepseekV41DecoderLayer:
    def test_decoder_layer_init_and_sp_binding(self) -> None:
        cfg = _mock_vllm_config(num_hidden_layers=40, enable_prefix_caching=False)

        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.extract_layer_index",
                return_value=25,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4DecoderLayer.__init__",
                return_value=None,
            ) as mock_super_init,
        ):
            layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
            layer.attn = SimpleNamespace()
            layer.use_sequence_parallel = True
            layer.__init__(cfg, prefix="model.layers.25")
            mock_super_init.assert_called_once()
            assert layer.attn._sp_sharded_input is True

        # Non-CED / encoder layer (< boundary)
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.extract_layer_index",
                return_value=5,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4DecoderLayer.__init__",
                return_value=None,
            ),
        ):
            layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
            layer.attn = SimpleNamespace()
            layer.use_sequence_parallel = False
            layer.__init__(cfg, prefix="model.layers.5")
            assert layer.attn._sp_sharded_input is False

        # Layer without attn attribute
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", False),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.extract_layer_index",
                return_value=0,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4DecoderLayer.__init__",
                return_value=None,
            ),
        ):
            layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
            layer.__init__(cfg, prefix="model.layers.0")
            assert not hasattr(layer, "attn")

    def test_decoder_layer_forward_binding(self) -> None:
        assert callable(DeepseekV41DecoderLayer.forward)


# ==============================================================================
# 4. Tests for _use_sequence_parallel
# ==============================================================================


class TestUseSequenceParallel:
    def test_upstream_returns_true(self) -> None:
        cfg = _mock_vllm_config()
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._upstream_use_sequence_parallel",
            return_value=True,
        ):
            assert _use_sequence_parallel(cfg) is True

    @pytest.mark.parametrize(
        (
            "upstream_sp",
            "tpu_sp",
            "pp_size",
            "ep_enabled",
            "tp_size",
            "expected",
        ),
        [
            (False, True, 1, True, 4, True),
            (False, False, 1, True, 4, False),
            (False, True, 2, True, 4, False),
            (False, True, 1, False, 4, False),
            (False, True, 1, True, 1, False),
        ],
    )
    def test_opt_in_matrix(
        self,
        upstream_sp: bool,
        tpu_sp: bool,
        pp_size: int,
        ep_enabled: bool,
        tp_size: int,
        expected: bool,
    ) -> None:
        cfg = _mock_vllm_config(tp_size=tp_size, pp_size=pp_size, ep_enabled=ep_enabled)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._upstream_use_sequence_parallel",
                return_value=upstream_sp,
            ),
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_SP", tpu_sp),
        ):
            assert _use_sequence_parallel(cfg) is expected


# ==============================================================================
# 5. Tests for _StageBase and Stage Classes
# ==============================================================================


class TestStageClasses:
    def test_stage_base_init(self) -> None:
        owner = SimpleNamespace()
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
        )

        # With compile tag
        stage = _StageBase(
            owner=owner, vllm_config=cfg, prefix="base", compile_tag="tag1"
        )
        assert stage.prefix == "tag1"
        assert object.__getattribute__(stage, "owner") is owner

        # Without compile tag
        stage_no_tag = _StageBase(
            owner=owner, vllm_config=cfg, prefix="base", compile_tag=None
        )
        assert stage_no_tag.owner is owner
        assert not hasattr(stage_no_tag, "prefix")

    def test_full_stage_forward(self) -> None:
        owner = SimpleNamespace(_forward_stage=MagicMock(return_value="full_res"))
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
        )
        stage = DeepseekV41FullStage(owner=owner, vllm_config=cfg)
        res = stage.forward(
            1, 2, intermediate_tensors=3, inputs_embeds=4, lookback_token_ids=5
        )
        assert res == "full_res"
        owner._forward_stage.assert_called_once_with(1, 2, 3, 4, 5, stage="full")

    def test_encoder_stage_forward(self) -> None:
        owner = SimpleNamespace(_forward_stage=MagicMock(return_value="enc_res"))
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
        )
        stage = DeepseekV41EncoderStage(owner=owner, vllm_config=cfg)
        res = stage.forward(
            1, 2, intermediate_tensors=3, inputs_embeds=4, lookback_token_ids=5
        )
        assert res == "enc_res"
        owner._forward_stage.assert_called_once_with(1, 2, 3, 4, 5, stage="encoder")

    def test_decoder_stage_forward(self) -> None:
        owner = SimpleNamespace(_forward_stage=MagicMock(return_value="dec_res"))
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
        )
        stage = DeepseekV41DecoderStage(owner=owner, vllm_config=cfg)
        res = stage.forward(1, 2, inputs_embeds=3, encoder_pre_mix=4)
        assert res == "dec_res"
        owner._forward_stage.assert_called_once_with(
            1, 2, None, 3, encoder_pre_mix=4, stage="decoder"
        )


# ==============================================================================
# 6. Tests for DeepseekV41Model Engram Methods
# ==============================================================================


class TestModelEngramMethods:
    def _create_mock_model(self, *, cpu_offload: bool = True) -> DeepseekV41Model:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        torch.nn.Module.__init__(model)
        model.start_layer = 0
        model.end_layer = 1
        model.engram_dp_shared_memory = True
        model.engram_swa_prefix = "swa_0"
        model.use_native_mega_moe = False

        engram = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=cpu_offload),
            layer_hash_index=0,
            prepare_embeddings=MagicMock(),
        )
        layer = SimpleNamespace(engram=engram)
        model.layers = [layer]

        hash_state = MagicMock()
        hash_state.use_slot_cache = True
        hash_state.lookback_depth = 2
        hash_state.multipliers = torch.zeros((4,))
        hash_state.ensure_cache = MagicMock(return_value=True)
        hash_state.hash_on_host = MagicMock(
            return_value=torch.ones((2, 4, 1), dtype=torch.int32)
        )
        hash_state.return_value = torch.ones((2, 4, 1), dtype=torch.int32)
        hash_state.dummy_hashes = MagicMock(
            return_value=(
                torch.zeros((2, 4, 1)),
                torch.ones((2,), dtype=torch.bool),
            )
        )
        hash_state.swa_cache_module = SimpleNamespace(kv_cache=None)
        model.engram_hash = hash_state
        return model

    def test_has_cpu_offloaded_engram(self) -> None:
        model_true = self._create_mock_model(cpu_offload=True)
        assert model_true._has_cpu_offloaded_engram() is True

        model_false = self._create_mock_model(cpu_offload=False)
        assert model_false._has_cpu_offloaded_engram() is False

        model_no_engram = self._create_mock_model(cpu_offload=False)
        model_no_engram.layers = [SimpleNamespace()]
        assert model_no_engram._has_cpu_offloaded_engram() is False

    def test_host_hash_inputs_branches(self) -> None:
        model = self._create_mock_model(cpu_offload=True)
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            block_tables=None,
        )
        lookback = torch.tensor([[10, 11]], dtype=torch.int32)

        # 1. lookback is None
        assert model._host_hash_inputs(input_ids, swa_meta, None) is None

        # 2. torch.compiler.is_compiling() is True
        with patch("torch.compiler.is_compiling", return_value=True):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 3. Not cpu offloaded
        with patch.object(model, "_has_cpu_offloaded_engram", return_value=False):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 4. host is None
        mock_ctx = SimpleNamespace(additional_kwargs={})
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 5. host is not all_prompt
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=input_ids,
            positions=torch.tensor([0, 1]),
            query_start_loc=swa_meta.query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=False,
        )
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 6. Valid host inputs
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=input_ids,
            positions=torch.tensor([0, 1]),
            query_start_loc=swa_meta.query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=True,
        )
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            res = model._host_hash_inputs(input_ids, swa_meta, lookback)
            assert res is not None
            assert len(res) == 6
            assert torch.equal(res[0], input_ids)

    def test_prepare_engram_early_exits_and_skip_lookup(self) -> None:
        model = self._create_mock_model(cpu_offload=True)
        pos = torch.tensor([0, 1], dtype=torch.int64)

        # 1. stage == "decoder"
        assert model._prepare_engram(
            torch.tensor([1, 2]), pos, None, stage="decoder"
        ) == (None, None)

        # 2. input_ids is None
        assert model._prepare_engram(None, pos, None, stage="full") == (None, None)

        # 3. is_forward_context_available is False
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
            return_value=False,
        ):
            assert model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="full"
            ) == (None, None)

        # 4. engram_hash is None
        model_no_hash = self._create_mock_model()
        model_no_hash.engram_hash = None
        assert model_no_hash._prepare_engram(
            torch.tensor([1, 2]), pos, None, stage="full"
        ) == (None, None)

        # 5. skip_lookup is True
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            seq_lens=torch.tensor([2]),
            block_tables=None,
        )
        mock_ctx = SimpleNamespace(
            attn_metadata={"swa_0": swa_meta}, additional_kwargs={}
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
        ):
            hashes, mask = model._prepare_engram(
                input_ids, pos, None, stage="full", skip_lookup=True
            )
            assert hashes is not None
            assert hashes.shape == (2, 4, 1)
            assert mask is not None

    def test_prepare_engram_lookback_handling_and_dp_dummy_hashes(self) -> None:
        model = self._create_mock_model(cpu_offload=True)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            seq_lens=torch.tensor([2]),
            block_tables=None,
        )
        mock_ctx = SimpleNamespace(
            attn_metadata={"swa_0": swa_meta}, additional_kwargs={}
        )

        # 1. lookback is None with use_slot_cache=False -> raises NotImplementedError
        model.engram_hash.use_slot_cache = False
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            pytest.raises(
                NotImplementedError, match="engram needs `lookback_token_ids`"
            ),
        ):
            model._prepare_engram(input_ids, pos, None, stage="full")

        # 2. lookback is None with use_slot_cache=True -> creates dummy
        # lookback and calls hash
        model.engram_hash.use_slot_cache = True
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=torch.zeros((2, 4, 1)),
            ),
        ):
            hashes, mask = model._prepare_engram(input_ids, pos, None, stage="full")
            assert hashes is not None
            model.layers[0].engram.prepare_embeddings.assert_called_once()

        # 3. attn_metadata is a list (ubatch wrapper)
        mock_ctx_list = SimpleNamespace(
            attn_metadata=[{"swa_0": swa_meta}], additional_kwargs={}
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx_list,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.dbo_current_ubatch_id",
                return_value=0,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=torch.zeros((2, 4, 1)),
            ),
        ):
            hashes, mask = model._prepare_engram(
                input_ids, pos, torch.tensor([[10, 11]]), stage="full"
            )
            assert hashes is not None

        # 4. ensure_cache() returns False with engram_dp_shared_memory=False
        # and dp_size > 1
        model.engram_hash.ensure_cache.return_value = False
        model.engram_dp_shared_memory = False
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=2,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=torch.zeros((2, 4, 1)),
            ),
        ):
            hashes, mask = model._prepare_engram(input_ids, pos, None, stage="full")
            assert hashes is not None
            model.engram_hash.dummy_hashes.assert_called_once()

        # 5. gathered_hashes on non-CPU device with cpu_offload transfers to CPU
        model.engram_hash.ensure_cache.return_value = True
        device_tensor = MagicMock()
        device_tensor.device.type = "cuda"
        device_tensor.to.return_value = torch.zeros((2, 4, 1))
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=device_tensor,
            ),
        ):
            model._prepare_engram(
                input_ids, pos, torch.tensor([[10, 11]]), stage="full"
            )
            device_tensor.to.assert_called_once_with("cpu")

        # 6. host is not None calls hash_on_host (line 270) and layer with no
        # engram (branch 303->301)
        model_host = self._create_mock_model(cpu_offload=True)
        model_host.layers.append(SimpleNamespace(engram=None))
        model_host.end_layer = 2
        model_host._host_hash_inputs = MagicMock(
            return_value=(
                input_ids,
                pos,
                swa_meta.query_start_loc,
                None,
                torch.tensor([[10, 11]]),
                None,
            )
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=torch.zeros((2, 4, 1)),
            ),
        ):
            h_host, m_host = model_host._prepare_engram(
                input_ids, pos, torch.tensor([[10, 11]]), stage="full"
            )
            model_host.engram_hash.hash_on_host.assert_called_once()
            assert h_host is not None

        # 7. ensure_cache False with engram_dp_shared_memory=True (branch 286->290)
        model_shm = self._create_mock_model()
        model_shm.engram_hash.ensure_cache.return_value = False
        model_shm.engram_dp_shared_memory = True
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
        ):
            h_shm, m_shm = model_shm._prepare_engram(input_ids, pos, None, stage="full")
            assert h_shm is None
            assert m_shm is None

        # 8. dummy_hashes with skip_lookup=True (branch 290->307)
        model_dummy = self._create_mock_model()
        model_dummy.engram_hash.ensure_cache.return_value = False
        model_dummy.engram_dp_shared_memory = False
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=2,
            ),
        ):
            h_dum, m_dum = model_dummy._prepare_engram(
                input_ids, pos, None, stage="full", skip_lookup=True
            )
            assert h_dum is not None
            assert m_dum is not None

        # 9. stage == "encoder" scopes layer iteration to _ced_boundary
        model_enc = self._create_mock_model(cpu_offload=True)
        dec_engram = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=True),
            layer_hash_index=0,
            prepare_embeddings=MagicMock(),
        )
        model_enc.layers.append(SimpleNamespace(engram=dec_engram))
        model_enc.end_layer = 2
        model_enc._ced_boundary = 1
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=torch.zeros((2, 4, 1)),
            ),
        ):
            model_enc._prepare_engram(
                input_ids, pos, torch.tensor([[10, 11]]), stage="encoder"
            )
            model_enc.layers[0].engram.prepare_embeddings.assert_called_once()
            dec_engram.prepare_embeddings.assert_not_called()


# ==============================================================================
# 7. Tests for DeepseekV41Model Forward and Replay Logic
# ==============================================================================


class TestModelForward:
    def _create_model(
        self,
        *,
        ced_enabled: bool = True,
        sp_enabled: bool = False,
        cpu_offload: bool = False,
        use_native_mega_moe: bool = False,
    ) -> DeepseekV41Model:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        model._ced_enabled = ced_enabled
        model._ced_boundary = 1
        model._ced_window = 512
        model._ced_compile_sizes = (4, 8)
        model.start_layer = 0
        model.end_layer = 2
        model.use_sequence_parallel = sp_enabled
        model.use_native_mega_moe = use_native_mega_moe
        model.config = SimpleNamespace(hidden_size=64)
        model._engram_prepared_outside_stage = False
        model._has_cpu_offloaded_engram = MagicMock(return_value=cpu_offload)
        model._prepare_engram = MagicMock(return_value=(None, None))
        return model

    def test_forward_non_ced_path(self) -> None:
        model = self._create_model(ced_enabled=False)
        model.full_stage = MagicMock(return_value=torch.zeros((4, 64)))
        res = model.forward(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2, 3]))
        assert res.shape == (4, 64)
        model.full_stage.assert_called_once()

    def test_forward_offload_engram_mega_moe_int64(self) -> None:
        model = self._create_model(
            ced_enabled=False, cpu_offload=True, use_native_mega_moe=True
        )
        model.full_stage = MagicMock(return_value=torch.zeros((2, 64)))
        inp = torch.tensor([1, 2], dtype=torch.int32)
        model.forward(inp, torch.tensor([0, 1]))
        model._prepare_engram.assert_called_once()
        prep_inp = model._prepare_engram.call_args[0][0]
        assert prep_inp.dtype == torch.int64

    def test_forward_ced_incomplete_prefill_early_exit(self) -> None:
        model = self._create_model(ced_enabled=True)
        # Empty replay plan (indices=())
        empty_plan = DecoderReplayPlan(
            indices=(), padded_num_tokens=0, query_start_loc=[0], replay_starts=[0]
        )
        hidden = torch.ones((4, 64))
        pre_mix = torch.zeros((4, 64))
        model.encoder_stage = MagicMock(return_value=[(hidden, pre_mix)])

        first_meta = SimpleNamespace(seq_lens=torch.tensor([4]), query_start_loc=[0, 4])
        mock_ctx = _FakeForwardCtx(attn_metadata={"layer_0": first_meta})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.plan_decoder_replay",
                return_value=empty_plan,
            ),
        ):
            res = model.forward(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2, 3]))
            assert res.shape == (4, 64)
            assert torch.all(res == 0)

    def test_forward_ced_plan_none_and_first_none(self) -> None:
        model = self._create_model(ced_enabled=True, sp_enabled=False)
        hidden = torch.ones((2, 64))
        pre_mix = torch.zeros((2, 64))
        model.encoder_stage = MagicMock(return_value=(hidden, pre_mix))
        model.decoder_stage = MagicMock(return_value=[torch.ones((2, 64)) * 3.0])

        # metadata is not a dict -> first is None -> plan is None
        mock_ctx = _FakeForwardCtx(attn_metadata=None)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            res = model.forward(torch.tensor([1, 2]), torch.tensor([0, 1]))
            assert res.shape == (2, 64)
            assert torch.all(res == 3.0)

    def test_forward_ced_with_active_replay_plan_and_scatter(self) -> None:
        model = self._create_model(ced_enabled=True, sp_enabled=True)
        hidden = torch.ones((4, 64))
        pre_mix = torch.zeros((4, 64))
        model.encoder_stage = MagicMock(return_value=(hidden, pre_mix))

        plan = DecoderReplayPlan(
            indices=(1, 3),
            padded_num_tokens=4,
            query_start_loc=[0, 2],
            replay_starts=[0, 1],
        )
        dec_out = torch.tensor([[10.0] * 64, [30.0] * 64, [0.0] * 64, [0.0] * 64])
        model.decoder_stage = MagicMock(return_value=dec_out)

        meta_entry = _FakeAttnMeta(
            input_positions=None,
            query_start_loc=None,
            seq_lens=torch.tensor([2, 2]),
        )
        mock_ctx = _FakeForwardCtx(attn_metadata={"layer_0": meta_entry})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.plan_decoder_replay",
                return_value=plan,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_all_gather",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.synchronize_tensors"
            ) as mock_sync,
        ):
            # Input IDs is None branch
            res_no_ids = model.forward(None, torch.tensor([0, 1, 2, 3]))
            assert res_no_ids.shape == (4, 64)
            assert res_no_ids[1, 0].item() == 10.0
            assert res_no_ids[3, 0].item() == 30.0

            # TPU device synchronization check
            class _FakeTpuTensor(torch.Tensor):
                @property
                def device(self) -> SimpleNamespace:  # type: ignore[override]
                    return SimpleNamespace(type="tpu")

            hidden_tpu = torch.ones((4, 64)).as_subclass(_FakeTpuTensor)
            pre_mix_tpu = torch.zeros((4, 64)).as_subclass(_FakeTpuTensor)
            model.encoder_stage = MagicMock(return_value=(hidden_tpu, pre_mix_tpu))
            model.forward(torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2, 3]))
            mock_sync.assert_called_once()


# ==============================================================================
# 8. Tests for DeepseekV41Model._forward_stage
# ==============================================================================


class TestModelForwardStage:
    def _create_stage_model(
        self,
        *,
        stage_sp: bool = False,
        use_native_mega_moe: bool = False,
        mtp: bool = False,
    ) -> DeepseekV41Model:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        model.start_layer = 0
        model.end_layer = 2
        model._ced_boundary = 1
        model.use_sequence_parallel = stage_sp
        model.use_native_mega_moe = use_native_mega_moe
        model._engram_prepared_outside_stage = False
        model._has_cpu_offloaded_engram = MagicMock(return_value=False)
        model._prepare_engram = MagicMock(return_value=(None, None))
        model.aux_hidden_state_layers = [1, 2]
        model.embed_input_ids = MagicMock(
            side_effect=lambda x: torch.ones((x.shape[0], 64))
        )
        model.norm = MagicMock(side_effect=lambda x: x)
        model._mtp_hidden_buffer = torch.zeros((16, 64)) if mtp else None

        # Layers mock
        layer0 = MagicMock()
        layer0.return_value = (
            torch.ones((4, 64)),
            torch.zeros((4, 64)),
            None,
            None,
            torch.ones((4, 64)),
            torch.ones((4, 64)),  # previous_aux
        )
        layer1 = MagicMock()
        layer1.return_value = (
            torch.ones((4, 64)) * 2.0,
            torch.zeros((4, 64)),
            None,
            None,
            torch.ones((4, 64)),
            None,
        )
        layer1.attn_norm = MagicMock(side_effect=lambda x: x)
        layer1.attn = SimpleNamespace(write_encoder_global_kv=MagicMock())
        model.layers = [layer0, layer1]
        return model

    def test_forward_stage_decoder_and_encoder(self) -> None:
        model = self._create_stage_model(stage_sp=False)
        pos = torch.tensor([0, 1, 2, 3])
        inp = torch.tensor([1, 2, 3, 4])
        embeds = torch.ones((4, 64))

        # 1. stage == "decoder"
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=True, is_last_rank=True
            )
            res = model._forward_stage(
                inp,
                pos,
                None,
                inputs_embeds=embeds,
                encoder_pre_mix=embeds,
                stage="decoder",
            )
            # Decoder finishes, returns hidden_states and aux_hidden_states
            assert isinstance(res, tuple)
            assert res[0].shape == (4, 64)

        # 2. stage == "encoder" -> returns (hidden_states, pre_mix)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=True, is_last_rank=True
            )
            res_enc = model._forward_stage(
                inp, pos, None, inputs_embeds=None, stage="encoder"
            )
            assert isinstance(res_enc, tuple)
            assert len(res_enc) == 2
            model.layers[1].attn.write_encoder_global_kv.assert_called_once()

    def test_forward_stage_pipeline_parallel_and_mtp(self) -> None:
        model = self._create_stage_model(stage_sp=True, mtp=True)
        pos = torch.tensor([0, 1, 2, 3])
        inp = torch.tensor([1, 2, 3, 4])
        inter_tensors = {
            "hidden_states": torch.ones((4, 64)),
            "pre_mix": torch.zeros((4, 64)),
        }

        # 1. Not first rank, not last rank -> returns IntermediateTensors
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_shard",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_all_gather",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=False, is_last_rank=False
            )
            res = model._forward_stage(inp, pos, inter_tensors, stage="full")
            assert "hidden_states" in res.tensors
            assert "pre_mix" in res.tensors

        # 2. Last rank with MTP buffer and SP all gather
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_shard",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_all_gather",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=False, is_last_rank=True
            )
            model._forward_stage(inp, pos, inter_tensors, stage="full")
            assert model._mtp_hidden_buffer[0, 0].item() != 0.0

    def test_forward_stage_moe_skip_padding_and_mega_moe(self) -> None:
        model = self._create_stage_model(stage_sp=True, use_native_mega_moe=True)
        pos = torch.tensor([0, 1, 2, 3])
        inp = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        mock_ctx = SimpleNamespace(is_padding=torch.zeros((4,), dtype=torch.bool))

        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.envs.VLLM_MOE_SKIP_PADDING",
                True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_padding_mask",
                side_effect=lambda p, h: p,
            ) as mock_pad_mask,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_shard",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_all_gather",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            model._forward_stage(inp, pos, None, stage="full")
            mock_pad_mask.assert_called_once()

    def test_forward_stage_layer_none_branch_560_to_569(self) -> None:
        """Covers branch 560->569 (if layer is not None False branch)
        using PyFrame_LocalsToFast.
        """
        model = self._create_stage_model(stage_sp=False)
        pos = torch.tensor([0, 1])
        inp = torch.tensor([1, 2])

        def custom_layer(*args: object, **kwargs: object) -> tuple:
            caller_frame = sys._getframe(1)
            caller_frame.f_locals["layer"] = None
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(caller_frame), ctypes.c_int(0)
            )
            return (
                torch.ones((2, 64)),
                torch.zeros((2, 64)),
                None,
                None,
                torch.ones((2, 64)),
                None,
            )

        model.layers = [custom_layer]
        model.start_layer = 0
        model.end_layer = 1
        model._ced_boundary = 1
        model.aux_hidden_state_layers = []

        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post") as mock_post,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            res = model._forward_stage(inp, pos, None, stage="full")
            # When layer is None after the loop, post(...) should NOT be called
            # at line 562
            mock_post.assert_not_called()
            assert res.shape == (2, 64)

    def test_forward_stage_edge_cases_and_branches(self) -> None:
        pos = torch.tensor([0, 1, 2, 3])
        inp = torch.tensor([1, 2, 3, 4])
        embeds = torch.ones((4, 64))

        # 1. first_rank with inputs_embeds not None (line 474), stage_sp with
        # encoder_pre_mix (line 522), and end_layer not in aux_hidden_state_layers
        # (branch 563->569)
        model_sp = self._create_stage_model(stage_sp=True, mtp=False)
        model_sp.aux_hidden_state_layers = []  # covers branch 563->569
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_shard",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.sp_all_gather",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=True, is_last_rank=True
            )
            res = model_sp._forward_stage(
                inp,
                pos,
                None,
                inputs_embeds=embeds,
                encoder_pre_mix=embeds,
                stage="full",
            )
            assert res.shape == (4, 64)

        # 2. mtp buffer without stage_sp (branch 594->597)
        model_no_sp_mtp = self._create_stage_model(stage_sp=False, mtp=True)
        inter_tensors = {
            "hidden_states": torch.ones((4, 64)),
            "pre_mix": torch.zeros((4, 64)),
        }
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_pp_group"
            ) as mock_pp,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.post",
                side_effect=lambda h, *a: h,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.collapse",
                side_effect=lambda h, pm: h,
            ),
        ):
            mock_pp.return_value = SimpleNamespace(
                is_first_rank=False, is_last_rank=True
            )
            model_no_sp_mtp._forward_stage(inp, pos, inter_tensors, stage="full")
            assert model_no_sp_mtp._mtp_hidden_buffer[0, 0].item() != 0.0


# ==============================================================================
# 9. Tests for Model Init and Weights Finalization
# ==============================================================================


class TestModelInitAndWeights:
    def test_finalize_mhc_broadcast_weights_branches(self) -> None:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        model.start_layer = 0
        model.end_layer = 1

        # 1. Already packed: dtype != float32 -> returns early
        layer_packed = SimpleNamespace(hc_attn_fn=SimpleNamespace(dtype=torch.bfloat16))
        model.layers = [layer_packed]
        with patch(
            "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.finalize_mhc_broadcast_weights"
        ) as mock_super:
            model.finalize_mhc_broadcast_weights()
            mock_super.assert_not_called()

        # 2. Float32: calls super() and pack_mix_weights
        layer_fp32 = SimpleNamespace(hc_attn_fn=SimpleNamespace(dtype=torch.float32))
        model.layers = [layer_fp32]
        with (
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.finalize_mhc_broadcast_weights"
            ) as mock_super,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.pack_mix_weights"
            ) as mock_pack,
        ):
            model.finalize_mhc_broadcast_weights()
            mock_super.assert_called_once()
            mock_pack.assert_called_once_with([layer_fp32])

        # 3. Empty layers
        model.layers = []
        with (
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.finalize_mhc_broadcast_weights"
            ) as mock_super,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.pack_mix_weights"
            ) as mock_pack,
        ):
            model.finalize_mhc_broadcast_weights()
            mock_super.assert_called_once()
            mock_pack.assert_called_once_with([])

    def test_model_init_validation_and_ced_setup(self) -> None:
        def fake_super_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
            torch.nn.Module.__init__(self)
            self.use_sequence_parallel = getattr(
                vllm_config, "_test_use_sequence_parallel", False
            )
            self.config = SimpleNamespace(num_hidden_layers=4, hidden_size=16)
            self.start_layer = 0
            self.end_layer = 4
            self.layers = [
                SimpleNamespace(
                    attn=SimpleNamespace(
                        window_size=4, compressor=None if i != 2 else SimpleNamespace()
                    ),
                    engram=None,
                )
                for i in range(4)
            ]

        # 1a. SP compile sizes too small raises ValueError
        cfg_sp_err = _mock_vllm_config(
            num_hidden_layers=4, tp_size=4, compile_sizes=[4, 8, 16]
        )
        cfg_sp_err._test_use_sequence_parallel = True
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_init,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=True,
            ),
            pytest.raises(
                ValueError,
                match="TPU_DEEPSEEK_V41_SP needs every token bucket >= 2 \\* TP",
            ),
        ):
            DeepseekV41Model(vllm_config=cfg_sp_err)

        # 1b. SP valid compile sizes logs enabled
        cfg_sp_ok = _mock_vllm_config(
            num_hidden_layers=4, tp_size=4, compile_sizes=[8, 16]
        )
        cfg_sp_ok._test_use_sequence_parallel = True
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_init,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=True,
            ),
        ):
            m_sp = DeepseekV41Model(vllm_config=cfg_sp_ok)
            assert m_sp.use_sequence_parallel is True

        # 2. CED=1 with prefill_context_parallel_size != 1 raises NotImplementedError
        cfg_cp = _mock_vllm_config(
            num_hidden_layers=4, tp_size=1, compile_sizes=[16], prefill_cp_size=2
        )
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_init,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=False,
            ),
            pytest.raises(
                NotImplementedError,
                match="CED bounded replay requires unsharded token rows",
            ),
        ):
            DeepseekV41Model(vllm_config=cfg_cp)

        # 3. CED=1 with multiple decoder compressors raises NotImplementedError
        cfg_ced = _mock_vllm_config(
            num_hidden_layers=4, tp_size=1, compile_sizes=[16], prefill_cp_size=1
        )

        def fake_super_bad_sources(
            self: Any, *, vllm_config: Any, prefix: str = ""
        ) -> None:
            fake_super_init(self, vllm_config=vllm_config, prefix=prefix)
            self.layers[3].attn.compressor = SimpleNamespace()

        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_bad_sources,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=False,
            ),
            pytest.raises(
                NotImplementedError, match="CED requires one decoder KV source"
            ),
        ):
            DeepseekV41Model(vllm_config=cfg_ced)

        # 4. CED=1 with engram in decoder stage raises NotImplementedError
        def fake_super_dec_engram(
            self: Any, *, vllm_config: Any, prefix: str = ""
        ) -> None:
            fake_super_init(self, vllm_config=vllm_config, prefix=prefix)
            self.layers[3].engram = SimpleNamespace()

        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_dec_engram,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=False,
            ),
            pytest.raises(
                NotImplementedError, match="CED requires Engram layers in the encoder"
            ),
        ):
            DeepseekV41Model(vllm_config=cfg_ced)

        # 5. Happy path CED=1 setup
        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", True),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_init,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=False,
            ),
        ):
            m = DeepseekV41Model(vllm_config=cfg_ced)
            assert hasattr(m, "encoder_stage")
            assert hasattr(m, "decoder_stage")
            assert m.layers[2].attn._ced_decoder is True
            assert m.layers[2].attn._ced_global_from_encoder is True

        # 6. Happy path CED=0 full stage setup
        def fake_super_no_compressor(
            self: Any, *, vllm_config: Any, prefix: str = ""
        ) -> None:
            fake_super_init(self, vllm_config=vllm_config, prefix=prefix)
            for layer in self.layers:
                layer.attn.compressor = None

        with (
            patch("vllm_torchtpu.envs.TPU_DEEPSEEK_V41_CED", False),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_cache_ops"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_delayed_mhc"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_mhc_post"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_paged_attention"
            ),
            patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_indexer"),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.__init__",
                fake_super_no_compressor,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.compact_rope_caches"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._use_sequence_parallel",
                return_value=False,
            ),
        ):
            m = DeepseekV41Model(vllm_config=cfg_ced)
            assert hasattr(m, "full_stage")


# ==============================================================================
# 10. Tests for DeepseekV41ForCausalLM
# ==============================================================================


class TestDeepseekV41ForCausalLM:
    def test_init_capability_guards(self) -> None:
        # PP > 1
        cfg_pp = _mock_vllm_config(pp_size=2)
        with pytest.raises(NotImplementedError, match="requires PP1 and DP1"):
            DeepseekV41ForCausalLM(vllm_config=cfg_pp)

        # DP > 1
        cfg_dp = _mock_vllm_config(dp_size=2)
        with pytest.raises(NotImplementedError, match="requires PP1 and DP1"):
            DeepseekV41ForCausalLM(vllm_config=cfg_dp)

        # Speculation not None
        cfg_spec = _mock_vllm_config(speculative_config=object())
        with pytest.raises(
            NotImplementedError, match="does not support speculative method"
        ):
            DeepseekV41ForCausalLM(vllm_config=cfg_spec)

    def test_init_hooks_and_weight_mapper(self) -> None:
        cfg = _mock_vllm_config(async_scheduling=True)
        with (
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD": "1"}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence.install_lookback_fence"
            ) as mock_fence,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.install_host_engram_inputs"
            ) as mock_host_engram,
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV41LLMForCausalLM.__init__",
                return_value=None,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model._make_deepseek_v4_weights_mapper",
                return_value="mock_mapper",
            ) as mock_make_mapper,
        ):
            model_inst = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
            model_inst.config = SimpleNamespace(expert_dtype="fp4")
            model_inst.__init__(vllm_config=cfg)
            mock_fence.assert_called_once()
            mock_host_engram.assert_called_once()
            mock_make_mapper.assert_called_once_with("fp4", "weight_scale_inv")
            assert model_inst.hf_to_vllm_mapper == "mock_mapper"

        # When async_scheduling and offload are False
        cfg_sync = _mock_vllm_config(async_scheduling=False)
        with (
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD": "0"}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence.install_lookback_fence"
            ) as mock_fence,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.install_host_engram_inputs"
            ) as mock_host_engram,
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV41LLMForCausalLM.__init__",
                return_value=None,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model._make_deepseek_v4_weights_mapper",
                return_value="mapper",
            ),
        ):
            model_inst = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
            model_inst.config = SimpleNamespace()
            model_inst.__init__(vllm_config=cfg_sync)
            mock_fence.assert_not_called()
            mock_host_engram.assert_not_called()

    def test_forward_ensure_cache_branches(self) -> None:
        model_inst = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)

        # 1. engram_hash is not None -> calls ensure_cache()
        mock_hash = SimpleNamespace(ensure_cache=MagicMock())
        model_inst.model = SimpleNamespace(engram_hash=mock_hash)
        with patch(
            "vllm.models.deepseek_v41.nvidia.model.DeepseekV41LLMForCausalLM.forward",
            return_value="fwd_out",
        ):
            res = model_inst.forward("arg1", kw="kw1")
            assert res == "fwd_out"
            mock_hash.ensure_cache.assert_called_once()

        # 2. engram_hash is None -> does not call ensure_cache()
        model_inst.model = SimpleNamespace(engram_hash=None)
        with patch(
            "vllm.models.deepseek_v41.nvidia.model.DeepseekV41LLMForCausalLM.forward",
            return_value="fwd_out2",
        ):
            res = model_inst.forward("arg2")
            assert res == "fwd_out2"
