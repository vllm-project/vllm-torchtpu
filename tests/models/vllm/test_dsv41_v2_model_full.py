# SPDX-License-Identifier: Apache-2.0
"""Full-module line and branch coverage tests for DeepSeek-V4.1 v2/model.py."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# Ensure CPU platform and skip MDS queries before importing JAX / TPU modules
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm.config import CompilationMode

from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import DecoderReplayPlan
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

# ==============================================================================
# Helper Mock Factories
# ==============================================================================


@dataclass
class _FakeForwardCtx:
    attn_metadata: Any
    dp_metadata: Any = None
    is_padding: Any = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeAttnMeta:
    input_positions: Any = None
    query_start_loc: Any = None
    seq_lens: Any = None
    block_tables: Any = None


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
        additional_config={},
        engram_config=None,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=scheduler_config,
        speculative_config=speculative_config,
        kernel_config=kernel_config,
    )


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
        cfg_no_model = SimpleNamespace(
            additional_config={}, engram_config=None, cache_config=None
        )
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
            (False, True, 1, True, 4, False),
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
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_SP": str(int(tpu_sp))}),
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

    @contextlib.contextmanager
    def _context(self):
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            seq_lens=torch.tensor([2]),
            block_tables=None,
        )
        ctx = SimpleNamespace(attn_metadata={"swa_0": swa_meta}, additional_kwargs={})
        prefix = "vllm_torchtpu.models.vllm.deepseek_v41.v2.model"
        with contextlib.ExitStack() as stack:
            for name, value in (
                ("upstream.is_forward_context_available", True),
                ("upstream.get_forward_context", ctx),
                ("_cache_slots", None),
                ("_block_table", None),
            ):
                stack.enter_context(patch(f"{prefix}.{name}", return_value=value))
            # The fixtures' engrams are namespaces; accept them as Engram layers.
            stack.enter_context(patch(f"{prefix}.Engram", SimpleNamespace))
            yield

    def test_prepare_engram_early_exits(self) -> None:
        model = self._create_mock_model(cpu_offload=True)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        hidden = torch.ones(2, 4)
        ids = torch.tensor([1, 2])
        assert model._prepare_engram(ids, pos, None, hidden, stage="decoder") == (
            None,
            None,
            hidden,
        )
        assert model._prepare_engram(None, pos, None, hidden) == (None, None, hidden)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
            return_value=False,
        ):
            assert model._prepare_engram(ids, pos, None, hidden) == (
                None,
                None,
                hidden,
            )
        model.engram_hash = None
        assert model._prepare_engram(ids, pos, None, hidden) == (None, None, hidden)

    def test_prepare_engram_device_table_uses_gathered_hashes(self) -> None:
        model = self._create_mock_model(cpu_offload=False)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        gathered = torch.full((2, 4, 1), 7, dtype=torch.int32)
        with (
            self._context(),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                return_value=gathered,
            ),
        ):
            hashes, mask, out = model._prepare_engram(ids, pos, None, torch.ones(2, 4))
        assert hashes.shape == (2, 4, 1) and mask is not None
        # No lookback from the runner: one dead row per request.
        assert model.engram_hash.call_args.args[4].shape == (1, 2)
        prepared = model.layers[0].engram.prepare_embeddings
        prepared.assert_called_once()
        torch.testing.assert_close(prepared.call_args.args[0], gathered[:, 0])

    def test_prepare_engram_starts_host_channels_in_layer_order(self) -> None:
        model = self._create_mock_model(cpu_offload=True)
        order = []

        def engram(index):
            def start(local, host, hidden):
                order.append((index, local.shape, host.shape))
                return hidden + 1

            return SimpleNamespace(
                embed_tokens=SimpleNamespace(host_offload=True),
                layer_hash_index=index,
                host_channel=object(),
                start_host_embeddings=start,
            )

        model.layers = [SimpleNamespace(engram=engram(0)), SimpleNamespace(engram=None)]
        model.layers.append(SimpleNamespace(engram=engram(1)))
        model.end_layer = 3
        pos = torch.tensor([0, 1], dtype=torch.int64)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        model.engram_hash.return_value = torch.ones((2, 2, 3), dtype=torch.int32)
        with (
            self._context(),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.engram_host_channel_op.gather_hashes",
                side_effect=lambda h: torch.cat([h, h]),
            ) as gather,
        ):
            _, _, out = model._prepare_engram(
                ids, pos, torch.tensor([[3, 4]]), torch.zeros(2, 4)
            )
        gather.assert_called_once()  # one ICI all-gather for every channel
        assert order == [(0, (2, 3), (4, 3)), (1, (2, 3), (4, 3))]
        # The hidden states are ordered after both requests.
        assert torch.equal(out, torch.full((2, 4), 2.0))

    def test_prepare_engram_dummy_hashes_join_dp_lookups(self) -> None:
        model = self._create_mock_model(cpu_offload=False)
        model.engram_dp_shared_memory = False
        model.engram_hash.ensure_cache.return_value = False
        pos = torch.tensor([0, 1], dtype=torch.int64)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        with (
            self._context(),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=8,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                side_effect=lambda h, **_: h,
            ),
        ):
            hashes, mask, _ = model._prepare_engram(ids, pos, None, torch.ones(2, 4))
        model.engram_hash.dummy_hashes.assert_called_once()
        assert hashes is not None and mask is not None
        model.layers[0].engram.prepare_embeddings.assert_called_once()

        # A shared table needs no collective, so an idle replica skips it.
        model.engram_dp_shared_memory = True
        with self._context():
            assert model._prepare_engram(ids, pos, None, torch.ones(2, 4))[0] is None


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
        model._prepare_engram = MagicMock(
            side_effect=lambda *a, **kw: (None, None, a[3])
        )
        return model

    def test_forward_non_ced_path(self) -> None:
        model = self._create_model(ced_enabled=False)
        model.full_stage = MagicMock(return_value=torch.zeros((4, 64)))
        ids, positions = torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 2, 3])
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.torch.tensor"
        ) as upload:
            res = model.forward(ids, positions)
        upload.assert_not_called()
        assert res.shape == (4, 64)
        model.full_stage.assert_called_once()

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
        model._prepare_engram = MagicMock(
            side_effect=lambda *a, **kw: (None, None, a[3])
        )
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

    @patch("vllm_torchtpu.models.vllm.deepseek_v41.v2.model.initialize_shifted_mhc")
    def test_model_init_validation_and_ced_setup(self, _mock_shifted_mhc) -> None:
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
                match="Sequence parallelism needs every token bucket >= 2 \\* TP",
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
    @pytest.mark.cpu_test
    @pytest.mark.parametrize(
        ("shared_memory", "resolved_shared_memory"),
        [(False, False), (True, True), (True, False)],
    )
    @pytest.mark.parametrize(
        ("dp_size", "engram_dp_size"),
        [(8, 8), (16, 8), (24, 8), (32, 8), (64, 8), (12, 4)],
    )
    def test_dp_uses_per_rank_host_tables(
        self,
        shared_memory: bool,
        resolved_shared_memory: bool,
        dp_size: int,
        engram_dp_size: int,
    ) -> None:
        cfg = _mock_vllm_config(dp_size=dp_size, ep_enabled=True)
        cfg.engram_config = SimpleNamespace(
            cpu_offload=True, dp_shared_memory=shared_memory
        )

        def init(self, *, vllm_config, prefix=""):
            # Upstream may only disable sharing (e.g. /dev/shm is too small).
            if vllm_config.engram_config.dp_shared_memory:
                vllm_config.engram_config.dp_shared_memory = resolved_shared_memory
            self.config = SimpleNamespace(expert_dtype="fp4")

        with (
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD": "1"}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=engram_dp_size,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model.DeepseekV41LLMForCausalLM.__init__",
                init,
            ),
            patch(
                "vllm.models.deepseek_v41.nvidia.model._make_deepseek_v4_weights_mapper",
                return_value="mock_mapper",
            ),
        ):
            model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
            model.__init__(vllm_config=cfg)

        # Host channels need per-rank head shards, whatever upstream resolved.
        assert cfg.engram_config.dp_shared_memory is False
        assert model.hf_to_vllm_mapper == "mock_mapper"

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("engram_dp_size", [1, 3, 32])
    def test_dp_rejects_unsupported_host_table_groups(self, engram_dp_size):
        cfg = _mock_vllm_config(dp_size=16, ep_enabled=True)
        with (
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD": "1"}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=engram_dp_size,
            ),
            pytest.raises(NotImplementedError, match="node-local Engram groups"),
        ):
            DeepseekV41ForCausalLM(vllm_config=cfg)

    def test_init_capability_guards(self) -> None:
        # PP > 1
        cfg_pp = _mock_vllm_config(pp_size=2)
        with pytest.raises(NotImplementedError, match="requires PP1"):
            DeepseekV41ForCausalLM(vllm_config=cfg_pp)

        # DP > 1
        cfg_dp = _mock_vllm_config(dp_size=2)
        with pytest.raises(NotImplementedError, match="DP requires TP1, EP"):
            DeepseekV41ForCausalLM(vllm_config=cfg_dp)

        # Speculation not None
        cfg_spec = _mock_vllm_config(speculative_config=object())
        with pytest.raises(
            NotImplementedError, match="does not support speculative method"
        ):
            DeepseekV41ForCausalLM(vllm_config=cfg_spec)

    def test_init_hooks_and_weight_mapper(self) -> None:
        cfg = _mock_vllm_config(async_scheduling=True)
        cfg.engram_config = SimpleNamespace(cpu_offload=True, dp_shared_memory=None)
        with (
            patch.dict(os.environ, {"TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD": "1"}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence.install_lookback_fence"
            ) as mock_fence,
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
            assert cfg.engram_config.dp_shared_memory is False
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


@pytest.mark.parametrize("dp_size,supplied", [(8, False), (8, True), (1, False)])
def test_dp_embeddings_materialized_before_stage(monkeypatch, dp_size, supplied):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as module

    ids = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    positions = torch.arange(2)
    lookback = torch.tensor([[3, 4]])
    embeddings = torch.arange(8).reshape(2, 4).bfloat16()
    events = []

    def lookup(tokens):
        assert tokens is ids
        events.append("lookup")
        return embeddings

    def synchronize(values, *, wait):
        assert values[0] is embeddings and wait is False
        events.append("enqueue")

    def stage(tokens, pos, intermediate, inputs_embeds, history):
        assert tokens is ids and pos is positions and history is lookback
        assert intermediate is None
        assert inputs_embeds is (embeddings if supplied or dp_size > 1 else None)
        events.append("stage")
        return inputs_embeds

    owner = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(owner)
    owner.additional_config = {}
    owner.engram_config = None
    owner.parallel_config = SimpleNamespace(data_parallel_size=dp_size)
    owner.embed_input_ids = lookup
    owner._has_cpu_offloaded_engram = lambda: False
    owner._ced_enabled = False
    owner.full_stage = stage
    monkeypatch.setattr(module, "synchronize_tensors", synchronize)
    DeepseekV41Model.forward(
        owner,
        ids,
        positions,
        inputs_embeds=embeddings if supplied else None,
        lookback_token_ids=lookback,
    )
    assert events == (
        ["lookup", "enqueue", "stage"] if dp_size > 1 and not supplied else ["stage"]
    )
