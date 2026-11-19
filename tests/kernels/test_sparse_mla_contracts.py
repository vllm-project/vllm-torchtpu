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
"""Pure-CPU contract tests for the SparseMLA modular kernel architecture."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import (
    attention_body,
    attention_kernel,
    dcp_exchange,
    dma,
    dsa_gather,
    index_prep,
    native_sc_gather,
    ref_registry,
    testing_faults,
)
from vllm_torchtpu.kernels.mla.sparse import (
    config as sparse_config,
)
from vllm_torchtpu.kernels.mla.sparse import (
    kernel as sparse_kernel,
)
from vllm_torchtpu.kernels.mla.sparse.config import (
    DECODE_BODY_DEFAULT,
    GATHER_ATOM_ROWS,
    MIXED_BODY_DEFAULT,
    MIXED_KERNEL_V2_LEVERS,
    Q_ROPE_DIM,
    SEND_EXCHANGE_PIN_ROLES,
    SEND_EXCHANGE_VMEM_LIMIT_BYTES,
    SEND_PAD_ROWS,
    SEND_SLAB_ROW0,
    SPARSECORE_CACHE_LAYOUT,
    TENSORCORE_CACHE_LAYOUT,
    UNSET,
    AttentionKernelConfig,
    BodyExecutionStrategy,
    BodyOptimizationConfig,
    CacheLayout,
    CappedWindowConfig,
    DcpConfig,
    GatherConfig,
    GatherWaitMode,
    HbmPinRole,
    LseConfig,
    SendExchangeConfig,
    SparseMlaConfig,
    align_to,
    resolve_entry_config,
    send_layout_lanes,
    send_layout_shape,
)
from vllm_torchtpu.kernels.mla.sparse.gather_dispatch import (
    GatheredChunk,
    PreparedIndices,
    apply_dcp_skip_windows,
    gather_chunk_kv,
    prepare_attention_indices,
)
from vllm_torchtpu.kernels.mla.sparse.kernel_context import (
    DoubleBufferState,
    KernelGeometry,
    compute_kv_rungs,
)
from vllm_torchtpu.kernels.mla.sparse.ref_registry import (
    AttentionRefs,
    BoundsBuilder,
    BoundsRef,
    DmaSemaphores,
    LiveFusedPrepRefs,
    PallasRefRegistry,
    build_attention_ref_registry,
    build_live_fused_prep_ref_registry,
)
from vllm_torchtpu.kernels.mla.sparse.testing_faults import (
    LIVE_PREP_FAULTS,
    NO_FAULT_HOOKS,
    OUT_PIECES_FAULTS,
    PROLOGUE_FAULTS,
    SEND_EXCHANGE_FAULTS,
    SEND_FAULTS,
    FaultHooks,
    LivePrepFault,
    OutPiecesFault,
    PrologueFault,
    SendExchangeFault,
    SendFault,
)


class _RecordingAtProxy:
    """Fake `.at[...]` accessor recording subscript keys for semaphore tests."""

    def __getitem__(self, key: Any) -> tuple[str, Any]:
        return ("sem_slot", key)


class _FakeSemaphoreRef:
    """Lightweight stand-in for a Pallas semaphore ref supporting `.at[i, slot]`."""

    at = _RecordingAtProxy()


class _RecordingMemRef:
    """Lightweight stand-in for a writable Pallas VMEM ref."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.writes: list[tuple[Any, Any]] = []

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes.append((key, value))


# ---------------------------------------------------------------------------
# 1. Config & Enum contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("enum_cls", "member", "raw_value"),
    [
        (CacheLayout, CacheLayout.TENSORCORE, "tensorcore"),
        (CacheLayout, CacheLayout.SPARSECORE, "sparsecore"),
        (GatherWaitMode, GatherWaitMode.OVERLAP_ROPE, "overlap_rope"),
        (GatherWaitMode, GatherWaitMode.BEFORE_ROPE, "before_rope"),
        (
            BodyExecutionStrategy,
            BodyExecutionStrategy.TOKEN_PIPELINED,
            "token_pipelined",
        ),
        (BodyExecutionStrategy, BodyExecutionStrategy.PAIR, "pair"),
        (BodyExecutionStrategy, BodyExecutionStrategy.QUAD, "quad"),
        (BodyExecutionStrategy, BodyExecutionStrategy.TRANSPOSED_QK, "xqk"),
        (HbmPinRole, HbmPinRole.KV, "kv"),
        (HbmPinRole, HbmPinRole.Q, "q"),
        (HbmPinRole, HbmPinRole.Q_PE, "q_pe"),
        (HbmPinRole, HbmPinRole.TAB, "tab"),
        (HbmPinRole, HbmPinRole.RECV, "recv"),
    ],
)
def test_enum_string_compatibility(
    enum_cls: type[Any], member: Any, raw_value: str
) -> None:
    assert member == raw_value
    assert member.value == raw_value
    assert enum_cls(raw_value) is member


def test_enum_legacy_string_constants() -> None:
    assert TENSORCORE_CACHE_LAYOUT is CacheLayout.TENSORCORE
    assert SPARSECORE_CACHE_LAYOUT is CacheLayout.SPARSECORE
    assert isinstance(TENSORCORE_CACHE_LAYOUT, CacheLayout)
    assert isinstance(TENSORCORE_CACHE_LAYOUT, str)
    assert isinstance(SPARSECORE_CACHE_LAYOUT, CacheLayout)
    assert isinstance(SPARSECORE_CACHE_LAYOUT, str)
    assert TENSORCORE_CACHE_LAYOUT == "tensorcore"
    assert SPARSECORE_CACHE_LAYOUT == "sparsecore"
    assert SEND_EXCHANGE_PIN_ROLES == ("kv", "q", "q_pe", "tab", "recv")


@pytest.mark.parametrize(
    ("enum_cls", "bad_value"),
    [
        (CacheLayout, "dense"),
        (GatherWaitMode, "after_rope"),
        (BodyExecutionStrategy, "triple"),
        (HbmPinRole, "lse"),
    ],
)
def test_enum_rejects_invalid_values(enum_cls: type[Any], bad_value: str) -> None:
    with pytest.raises(ValueError):
        enum_cls(bad_value)


@pytest.mark.parametrize(
    ("levers", "mode", "expected_strategy", "expected_uses_stage"),
    [
        (
            MIXED_BODY_DEFAULT,
            "mixed",
            BodyExecutionStrategy.TOKEN_PIPELINED,
            False,
        ),
        (DECODE_BODY_DEFAULT, "decode", BodyExecutionStrategy.QUAD, True),
        (MIXED_KERNEL_V2_LEVERS, "mixed", BodyExecutionStrategy.PAIR, True),
        (MIXED_KERNEL_V2_LEVERS, "any", BodyExecutionStrategy.PAIR, True),
    ],
)
def test_body_optimization_from_levers_presets(
    levers: tuple[str, ...],
    mode: Any,
    expected_strategy: BodyExecutionStrategy,
    expected_uses_stage: bool,
) -> None:
    cfg = BodyOptimizationConfig.from_levers(levers, mode=mode)
    assert cfg.strategy is expected_strategy
    assert cfg.uses_kv_stage is expected_uses_stage
    assert cfg.has_any_lever is True
    assert cfg.use_lever_loader is True
    assert set(cfg.to_levers()) == set(levers)
    assert BodyOptimizationConfig.from_levers(cfg, mode=mode) is cfg


@pytest.mark.parametrize(
    ("levers", "mode"),
    [
        (("pair", "quad"), "decode"),
        (("pair", "quad"), "any"),
        (("pair", "xqk"), "mixed"),
        (("pair", "xqk"), "any"),
        (("quad", "xqk"), "any"),
    ],
)
def test_body_optimization_mutual_exclusivity(
    levers: tuple[str, ...], mode: Any
) -> None:
    with pytest.raises(ValueError, match="alternatives"):
        BodyOptimizationConfig.from_levers(levers, mode=mode)


def test_body_optimization_zu_without_pv512_mode_behavior() -> None:
    mixed_cfg = BodyOptimizationConfig.from_levers(("zu",), mode="mixed")
    assert mixed_cfg.zero_nope_words_only is True
    assert mixed_cfg.pv_nope_only is False

    with pytest.raises(ValueError, match="'zu' needs 'pv512'"):
        BodyOptimizationConfig.from_levers(("zu",), mode="decode")

    with pytest.raises(ValueError, match="'zu' needs 'pv512'"):
        mixed_cfg.validate_for_mode("decode")

    decode_cfg = BodyOptimizationConfig.from_levers(("zu", "pv512"), mode="decode")
    assert decode_cfg.zero_nope_words_only is True
    assert decode_cfg.pv_nope_only is True


@pytest.mark.parametrize(
    ("levers", "mode"),
    [
        (("quad",), "mixed"),
        (("nofill",), "mixed"),
        (("lrows",), "mixed"),
        (("step8",), "mixed"),
        (("xqk",), "decode"),
        (("pbf16",), "decode"),
        (("bogus",), "mixed"),
        (("bogus",), "decode"),
        (("bogus",), "any"),
    ],
)
def test_body_optimization_mode_restrictions_and_unknowns(
    levers: tuple[str, ...], mode: Any
) -> None:
    with pytest.raises(ValueError):
        BodyOptimizationConfig.from_levers(levers, mode=mode)


def test_body_optimization_without_narrow_q_rope() -> None:
    base = BodyOptimizationConfig.from_levers(("tr", "pair"), mode="mixed")
    assert base.without_narrow_q_rope() is base

    with_q128 = BodyOptimizationConfig.from_levers(("tr", "pair", "q128"), mode="mixed")
    stripped = with_q128.without_narrow_q_rope()
    assert stripped.narrow_q_rope_dma is False
    assert stripped == base


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"strategy": BodyExecutionStrategy.PAIR}, True),
        ({"strategy": BodyExecutionStrategy.QUAD}, True),
        ({"strategy": BodyExecutionStrategy.TRANSPOSED_QK}, True),
        ({"transpose_nope_unpack": True}, True),
        ({"zero_nope_words_only": True}, True),
        ({"pv_nope_only": True}, True),
        ({"drop_runahead_dep": True}, True),
        ({"bf16_pv_weights": True}, True),
        ({"skip_clean_overflow_fixup": True}, True),
        ({"narrow_q_rope_dma": True}, True),
        ({"skip_redundant_lse_fill": True}, True),
        ({"fetch_live_rows_only": True}, True),
        ({"step8_attention_grid": True}, True),
    ],
)
def test_body_optimization_use_lever_loader(
    kwargs: dict[str, Any], expected: bool
) -> None:
    cfg = BodyOptimizationConfig(**kwargs)
    assert cfg.use_lever_loader is expected
    assert cfg.has_any_lever is expected


def test_body_optimization_frozen_and_hashable() -> None:
    cfg_a = BodyOptimizationConfig.from_levers(DECODE_BODY_DEFAULT, mode="decode")
    cfg_b = BodyOptimizationConfig.from_levers(DECODE_BODY_DEFAULT, mode="decode")
    assert hash(cfg_a) == hash(cfg_b)
    assert {cfg_a: 1}[cfg_b] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg_a.transpose_nope_unpack = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (
            ("dcp", None),
            SendExchangeConfig(
                axis="dcp",
                lane_axis=None,
                vmem_limit_bytes=SEND_EXCHANGE_VMEM_LIMIT_BYTES,
                hbm_pin=True,
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", None),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=SEND_EXCHANGE_VMEM_LIMIT_BYTES,
                hbm_pin=True,
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", 8 * 1024 * 1024),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=8 * 1024 * 1024,
                hbm_pin=True,
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", 8 * 1024 * 1024, False),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=8 * 1024 * 1024,
                hbm_pin=False,
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", 8 * 1024 * 1024, None),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=8 * 1024 * 1024,
                hbm_pin=False,
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", 8 * 1024 * 1024, ("kv", "q_pe")),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=8 * 1024 * 1024,
                hbm_pin=(HbmPinRole.KV, HbmPinRole.Q_PE),
                in_kernel_merge=False,
            ),
        ),
        (
            ("dcp", "tp", 8 * 1024 * 1024, ("kv", "tab"), True),
            SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                vmem_limit_bytes=8 * 1024 * 1024,
                hbm_pin=(HbmPinRole.KV, HbmPinRole.TAB),
                in_kernel_merge=True,
            ),
        ),
    ],
)
def test_send_exchange_from_raw_valid(
    raw: Any, expected: SendExchangeConfig | None
) -> None:
    parsed = SendExchangeConfig.from_raw(raw)
    assert parsed == expected
    if parsed is not None:
        assert SendExchangeConfig.from_raw(parsed) is parsed
        assert isinstance(hash(parsed), int)


@pytest.mark.parametrize(
    "raw",
    [
        ("dcp",),
        ("dcp", "tp", 1024, True, False, "extra"),
        ["dcp", "tp"],
        ("dcp", "tp", 0),
        ("dcp", "tp", -1),
        ("dcp", "tp", 1024, ("kv", "unknown_role")),
        ("dcp", "tp", 1024, ["kv"]),
        ("dcp", "tp", 1024, 123),
    ],
)
def test_send_exchange_from_raw_invalid(raw: Any) -> None:
    with pytest.raises(ValueError):
        SendExchangeConfig.from_raw(raw)


@pytest.mark.parametrize(
    ("hbm_pin", "role", "expected"),
    [
        (True, HbmPinRole.KV, True),
        (True, "q", True),
        (True, None, True),
        (False, HbmPinRole.KV, False),
        (False, None, False),
        ((HbmPinRole.KV, HbmPinRole.Q_PE), HbmPinRole.KV, True),
        ((HbmPinRole.KV, HbmPinRole.Q_PE), "q_pe", True),
        ((HbmPinRole.KV, HbmPinRole.Q_PE), HbmPinRole.Q, False),
        ((HbmPinRole.KV, HbmPinRole.Q_PE), None, False),
        ((), HbmPinRole.KV, False),
        ((), None, False),
    ],
)
def test_send_exchange_should_pin(
    hbm_pin: bool | tuple[HbmPinRole, ...],
    role: HbmPinRole | str | None,
    expected: bool,
) -> None:
    cfg = SendExchangeConfig(axis="dcp", hbm_pin=hbm_pin)
    assert cfg.should_pin(role) is expected


@pytest.mark.parametrize(
    ("layout", "wait_mode", "packed_rope", "expected_sc", "expected_rope_tpr"),
    [
        ("tensorcore", "overlap_rope", False, False, 4),
        ("sparsecore", "overlap_rope", False, True, 4),
        ("sparsecore", "before_rope", True, True, 8),
    ],
)
def test_gather_config_normalization_and_properties(
    layout: str,
    wait_mode: str,
    packed_rope: bool,
    expected_sc: bool,
    expected_rope_tpr: int,
) -> None:
    cfg = GatherConfig(
        cache_layout=layout,  # type: ignore[arg-type]
        sparsecore_wait_mode=wait_mode,  # type: ignore[arg-type]
        packed_rope=packed_rope,
    )
    assert isinstance(cfg.cache_layout, CacheLayout)
    assert isinstance(cfg.sparsecore_wait_mode, GatherWaitMode)
    assert cfg.is_sparsecore is expected_sc
    assert cfg.rope_tokens_per_row == expected_rope_tpr
    assert isinstance(hash(cfg), int)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_layout": "invalid"},
        {"sparsecore_wait_mode": "invalid"},
        {"cache_layout": "tensorcore", "packed_rope": True},
        {"chunk_size": 0},
        {"sparsecore_atoms_per_batch": 0},
        {"small_chunk_tokens": -1},
        {"small_chunk_batch_size": 0},
        {"small_chunk_gather_atoms": 0},
    ],
)
def test_gather_config_rejects_invalid_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GatherConfig(**kwargs)


def test_capped_window_resolve_disables_topk_cap_before_validation() -> None:
    # Unaligned topk_cap is disabled on TensorCore before divisibility checks.
    tc_cfg = CappedWindowConfig(topk_cap=100)
    resolved_tc = tc_cfg.resolve_for_inputs(
        sparsecore_layout=False, topk=256, has_q_rope_table=True
    )
    assert resolved_tc.topk_cap is None

    # topk_cap >= topk is disabled even when topk_cap would not divide topk.
    ge_cfg = CappedWindowConfig(topk_cap=300)
    resolved_ge = ge_cfg.resolve_for_inputs(
        sparsecore_layout=True, topk=256, has_q_rope_table=True
    )
    assert resolved_ge.topk_cap is None

    # Width ladders still fail once topk_cap has been disabled to None.
    with_widths = CappedWindowConfig(topk_cap=512, mixed_live_widths=(128,))
    with pytest.raises(ValueError, match="mixed_live_widths"):
        with_widths.resolve_for_inputs(
            sparsecore_layout=False, topk=2048, has_q_rope_table=True
        )
    with pytest.raises(ValueError, match="mixed_live_widths"):
        with_widths.resolve_for_inputs(
            sparsecore_layout=True, topk=512, has_q_rope_table=True
        )


@pytest.mark.parametrize(
    ("topk_cap", "topk", "mixed_widths", "decode_widths"),
    [
        (100, 1024, (), ()),  # Not a multiple of GATHER_ATOM_ROWS (64).
        (192, 1024, (), ()),  # Does not divide topk.
        (0, 1024, (), ()),  # Non-positive topk_cap.
        (256, 1024, (128, 256), ()),  # Width equal to topk_cap.
        (256, 1024, (), (128, 384)),  # Width above topk_cap.
        (None, 1024, (128,), ()),  # Width set with no topk_cap.
        (None, 1024, (), (128,)),  # Decode width set with no topk_cap.
    ],
)
def test_capped_window_resolve_rejects_invalid_geometry(
    topk_cap: int | None,
    topk: int,
    mixed_widths: tuple[int, ...],
    decode_widths: tuple[int, ...],
) -> None:
    cfg = CappedWindowConfig(
        topk_cap=topk_cap,
        mixed_live_widths=mixed_widths,
        decode_row_widths=decode_widths,
    )
    with pytest.raises(ValueError):
        cfg.resolve_for_inputs(sparsecore_layout=True, topk=topk, has_q_rope_table=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk_cap": -1},
        {"mixed_live_widths": (256, 128)},  # Not ascending.
        {"mixed_live_widths": (128, 128)},  # Duplicate.
        {"mixed_live_widths": (64,)},  # Not multiple of 128.
        {"decode_row_widths": (0, 128)},  # Non-positive.
    ],
)
def test_capped_window_post_init_rejects_invalid_width_ladders(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        CappedWindowConfig(**kwargs)


def test_capped_window_resolve_strips_q128_without_q_rope_table() -> None:
    mixed_body = BodyOptimizationConfig.from_levers(("tr", "q128"), mode="mixed")
    decode_body = BodyOptimizationConfig.from_levers(
        ("tr", "zu", "pv512", "q128"), mode="decode"
    )
    cfg = CappedWindowConfig(
        topk_cap=512,
        mixed_live_widths=(128, 256),
        decode_row_widths=(128, 384),
        mixed_body=mixed_body,
        decode_body=decode_body,
    )

    kept = cfg.resolve_for_inputs(
        sparsecore_layout=True, topk=2048, has_q_rope_table=True
    )
    assert kept is cfg
    assert kept.mixed_body.narrow_q_rope_dma is True
    assert kept.decode_body.narrow_q_rope_dma is True

    stripped = cfg.resolve_for_inputs(
        sparsecore_layout=True, topk=2048, has_q_rope_table=False
    )
    assert stripped.mixed_body.narrow_q_rope_dma is False
    assert stripped.decode_body.narrow_q_rope_dma is False
    assert stripped.mixed_body.transpose_nope_unpack is True
    assert stripped.decode_body.pv_nope_only is True


@pytest.mark.parametrize(
    ("lse_cfg", "num_chunks", "expected"),
    [
        (LseConfig(return_lse=True, lane_dense=True, in_kernel=True), 1, True),
        (LseConfig(return_lse=True, lane_dense=True, in_kernel=True), 2, False),
        (LseConfig(return_lse=True, lane_dense=False, in_kernel=True), 1, False),
        (LseConfig(return_lse=False, lane_dense=True, in_kernel=True), 1, False),
        (LseConfig(return_lse=True, lane_dense=True, in_kernel=False), 1, False),
    ],
)
def test_lse_config_effective_lse_fill(
    lse_cfg: LseConfig, num_chunks: int, expected: bool
) -> None:
    assert lse_cfg.effective_lse_fill(num_chunks) is expected


def test_dcp_config_validation() -> None:
    with pytest.raises(ValueError, match="send_world"):
        DcpConfig(send_world=-1)
    with pytest.raises(ValueError, match="out_pieces"):
        DcpConfig(out_pieces=-1)
    with pytest.raises(ValueError, match="send_exchange requires send_world > 0"):
        DcpConfig(send_world=0, send_exchange=SendExchangeConfig(axis="dcp"))


def test_sparse_mla_config_from_legacy_kwargs_and_aliases() -> None:
    primary = SparseMlaConfig.from_legacy_kwargs(
        sm_scale=0.125,
        k_scale=0.5,
        attention_kernel_batch_size=8,
        vmem_limit_bytes=32 * 1024 * 1024,
        cache_layout="sparsecore",
        packed_rope=True,
        gather_and_attention_chunk_size=128,
        sparsecore_gather_wait_mode="before_rope",
        topk_cap=512,
        mixed_live_widths=(128, 256),
        decode_row_widths=(128, 384),
        mixed_body_levers=MIXED_BODY_DEFAULT,
        decode_body_levers=DECODE_BODY_DEFAULT,
        return_lse=True,
        lse_lane_dense=True,
        lse_in_kernel=True,
        send_world=4,
        send_exchange=("dcp", "tp", 8 * 1024 * 1024, ("kv", "q"), True),
        out_pieces=2,
    )
    aliased = SparseMlaConfig.from_legacy_kwargs(
        sm_scale=0.125,
        k_scale=0.5,
        attention_kernel_batch_size=8,
        vmem_limit_bytes=32 * 1024 * 1024,
        cache_layout="sparsecore",
        packed_rope=True,
        chunk_size=128,
        sparsecore_gather_wait_mode="before_rope",
        topk_cap=512,
        group_widths=(128, 256),
        row_widths=(128, 384),
        group_levers=MIXED_BODY_DEFAULT,
        row_levers=DECODE_BODY_DEFAULT,
        return_lse=True,
        lse_lane_dense=True,
        lse_fill=True,
        send_world=4,
        send_exchange=("dcp", "tp", 8 * 1024 * 1024, ("kv", "q"), True),
        out_pieces=2,
    )
    assert primary == aliased
    assert isinstance(hash(primary), int)
    assert primary.gather.chunk_size == 128
    assert primary.capped.mixed_live_widths == (128, 256)
    assert primary.capped.decode_row_widths == (128, 384)
    assert primary.lse.in_kernel is True
    assert primary.dcp.send_exchange is not None
    assert primary.dcp.send_exchange.in_kernel_merge is True


@pytest.mark.parametrize(
    ("kwargs", "exc_cls"),
    [
        ({"totally_unknown_flag": True}, TypeError),
        ({"gather_and_attention_chunk_size": 64, "chunk_size": 64}, ValueError),
        ({"mixed_live_widths": (128,), "group_widths": (128,)}, ValueError),
        ({"decode_row_widths": (128,), "row_widths": (128,)}, ValueError),
        ({"mixed_body_levers": ("tr",), "group_levers": ("tr",)}, ValueError),
        ({"decode_body_levers": ("tr",), "row_levers": ("tr",)}, ValueError),
        ({"lse_in_kernel": True, "lse_fill": True}, ValueError),
    ],
)
def test_sparse_mla_config_from_legacy_kwargs_errors(
    kwargs: dict[str, Any], exc_cls: type[Exception]
) -> None:
    with pytest.raises(exc_cls):
        SparseMlaConfig.from_legacy_kwargs(**kwargs)


@pytest.mark.parametrize(
    ("layout", "packed_rope", "nope_shape", "rope_shape", "dtype", "expected_ps"),
    [
        ("tensorcore", False, (2, 8, 4, 128), (2, 2, 4, 128), jnp.uint8, 8),
        ("sparsecore", False, (2, 8, 128), (2, 2, 128), jnp.uint32, 8),
        ("sparsecore", True, (2, 16, 128), (2, 2, 128), jnp.uint32, 16),
    ],
)
def test_sparse_mla_config_validate_caches_valid(
    layout: str,
    packed_rope: bool,
    nope_shape: tuple[int, ...],
    rope_shape: tuple[int, ...],
    dtype: Any,
    expected_ps: int,
) -> None:
    cfg = SparseMlaConfig.from_legacy_kwargs(
        cache_layout=layout, packed_rope=packed_rope
    )
    nope = jnp.zeros(nope_shape, dtype=dtype)
    rope = jnp.zeros(rope_shape, dtype=dtype)
    assert cfg.validate_caches(nope, rope) == expected_ps


@pytest.mark.parametrize(
    ("layout", "packed_rope", "nope_shape", "nope_dtype", "rope_shape", "rope_dtype"),
    [
        # TensorCore wrong dtype / ndim / trailing dims / page geometry mismatch.
        ("tensorcore", False, (2, 8, 4, 128), jnp.uint32, (2, 2, 4, 128), jnp.uint8),
        ("tensorcore", False, (2, 8, 128), jnp.uint8, (2, 2, 4, 128), jnp.uint8),
        ("tensorcore", False, (2, 8, 4, 128), jnp.uint8, (2, 2, 2, 128), jnp.uint8),
        ("tensorcore", False, (2, 8, 4, 128), jnp.uint8, (3, 2, 4, 128), jnp.uint8),
        ("tensorcore", False, (2, 8, 4, 128), jnp.uint8, (2, 4, 4, 128), jnp.uint8),
        # SparseCore wrong dtype / ndim / trailing dims / packed_rope geometry.
        ("sparsecore", False, (2, 8, 128), jnp.uint8, (2, 2, 128), jnp.uint32),
        ("sparsecore", False, (2, 8, 4, 128), jnp.uint32, (2, 2, 128), jnp.uint32),
        ("sparsecore", False, (2, 8, 128), jnp.uint32, (2, 2, 64), jnp.uint32),
        ("sparsecore", True, (2, 8, 128), jnp.uint32, (2, 2, 128), jnp.uint32),
        # Same total size (2*8 = 4*4) but mismatched page counts / page geometry.
        ("sparsecore", False, (2, 16, 128), jnp.uint32, (4, 2, 128), jnp.uint32),
    ],
)
def test_sparse_mla_config_validate_caches_invalid(
    layout: str,
    packed_rope: bool,
    nope_shape: tuple[int, ...],
    nope_dtype: Any,
    rope_shape: tuple[int, ...],
    rope_dtype: Any,
) -> None:
    cfg = SparseMlaConfig.from_legacy_kwargs(
        cache_layout=layout, packed_rope=packed_rope
    )
    nope = jnp.zeros(nope_shape, dtype=nope_dtype)
    rope = jnp.zeros(rope_shape, dtype=rope_dtype)
    with pytest.raises(ValueError):
        cfg.validate_caches(nope, rope)


def test_sparse_mla_config_validate_queries_valid() -> None:
    tc_cfg = SparseMlaConfig()
    q_tc = jnp.zeros((8, 16, 128), dtype=jnp.bfloat16)
    topk_tc = jnp.zeros((8, 10), dtype=jnp.int32)
    tc_cfg.validate_queries(q_tc, None, None, topk_tc)

    sc_cfg = SparseMlaConfig.from_legacy_kwargs(cache_layout="sparsecore")
    q_sc = jnp.zeros((8, 16, 128), dtype=jnp.bfloat16)
    q_pe_plain = jnp.zeros((8, 16, 64), dtype=jnp.bfloat16)
    q_pe_raw = jnp.zeros((8, 16, 256), dtype=jnp.bfloat16)
    q_rope_tab = jnp.zeros((8, 2, 128), dtype=jnp.float32)
    topk_sc = jnp.zeros((8, 16), dtype=jnp.int32)

    sc_cfg.validate_queries(q_sc, q_pe_plain, None, topk_sc)
    sc_cfg.validate_queries(q_sc, q_pe_raw, q_rope_tab, topk_sc)


@pytest.mark.parametrize(
    ("case_name", "layout", "q_spec", "q_pe_spec", "q_tab_spec", "topk_spec"),
    [
        (
            "rope_table_without_q_pe",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            None,
            ((8, 2, 128), jnp.float32),
            ((8, 16), jnp.int32),
        ),
        (
            "rope_table_wrong_q_pe_dim",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            ((8, 16, 64), jnp.bfloat16),
            ((8, 2, 128), jnp.float32),
            ((8, 16), jnp.int32),
        ),
        (
            "rope_table_wrong_dtype",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            ((8, 16, 256), jnp.bfloat16),
            ((8, 2, 128), jnp.bfloat16),
            ((8, 16), jnp.int32),
        ),
        (
            "rope_table_wrong_shape",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            ((8, 16, 256), jnp.bfloat16),
            ((8, 4, 128), jnp.float32),
            ((8, 16), jnp.int32),
        ),
        (
            "q_pe_dtype_mismatch",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            ((8, 16, 64), jnp.float32),
            None,
            ((8, 16), jnp.int32),
        ),
        (
            "q_pe_heads_mismatch",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            ((8, 8, 64), jnp.bfloat16),
            None,
            ((8, 16), jnp.int32),
        ),
        (
            "split_q_unaligned_nope_dim",
            "sparsecore",
            ((8, 16, 96), jnp.bfloat16),
            ((8, 16, 64), jnp.bfloat16),
            None,
            ((8, 16), jnp.int32),
        ),
        (
            "split_q_unpacked_heads",
            "sparsecore",
            ((8, 3, 128), jnp.bfloat16),
            ((8, 3, 64), jnp.bfloat16),
            None,
            ((8, 16), jnp.int32),
        ),
        (
            "topk_wrong_dtype",
            "tensorcore",
            ((8, 16, 128), jnp.bfloat16),
            None,
            None,
            ((8, 16), jnp.int16),
        ),
        (
            "topk_wrong_ndim",
            "tensorcore",
            ((8, 16, 128), jnp.bfloat16),
            None,
            None,
            ((8, 16, 1), jnp.int32),
        ),
        (
            "topk_token_mismatch",
            "tensorcore",
            ((8, 16, 128), jnp.bfloat16),
            None,
            None,
            ((4, 16), jnp.int32),
        ),
        (
            "sparsecore_topk_not_rope_packing_multiple",
            "sparsecore",
            ((8, 16, 128), jnp.bfloat16),
            None,
            None,
            ((8, 10), jnp.int32),
        ),
    ],
)
def test_sparse_mla_config_validate_queries_invalid(
    case_name: str,
    layout: str,
    q_spec: tuple[tuple[int, ...], Any],
    q_pe_spec: tuple[tuple[int, ...], Any] | None,
    q_tab_spec: tuple[tuple[int, ...], Any] | None,
    topk_spec: tuple[tuple[int, ...], Any],
) -> None:
    del case_name
    cfg = SparseMlaConfig.from_legacy_kwargs(cache_layout=layout)
    q = jnp.zeros(q_spec[0], dtype=q_spec[1])
    q_pe = None if q_pe_spec is None else jnp.zeros(q_pe_spec[0], dtype=q_pe_spec[1])
    q_tab = (
        None if q_tab_spec is None else jnp.zeros(q_tab_spec[0], dtype=q_tab_spec[1])
    )
    topk = jnp.zeros(topk_spec[0], dtype=topk_spec[1])
    with pytest.raises(ValueError):
        cfg.validate_queries(q, q_pe, q_tab, topk)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sm_scale": 0.1},
        {
            "sm_scale": 0.1,
            "sparsecore_layout": True,
            "capped_windows": 4,
            "has_group_tab": True,
            "live_widths": (128, 256),
            "batch_size": 4,
            "split_q": True,
            "q_rope": True,
            "body": BodyOptimizationConfig.from_levers(
                ("tr", "pair", "q128"), mode="mixed"
            ),
        },
        {
            "sm_scale": 0.1,
            "sparsecore_layout": True,
            "capped_windows": 4,
            "row_widths": (128, 256),
            "batch_size": 4,
            "split_q": True,
            "q_rope": True,
            "return_lse": True,
            "lse_lane_dense": True,
            "lse_fill": True,
            "send_world": 4,
            "send_exchange": SendExchangeConfig(axis="dcp", in_kernel_merge=True),
            "body": BodyOptimizationConfig.from_levers(
                DECODE_BODY_DEFAULT, mode="decode"
            ),
        },
        {
            "sm_scale": 0.1,
            "return_lse": True,
            "lse_lane_dense": True,
            "out_in": True,
            "out_row_offset": 8,
        },
    ],
)
def test_attention_kernel_config_valid_modes(kwargs: dict[str, Any]) -> None:
    cfg = AttentionKernelConfig(**kwargs)
    assert isinstance(hash(cfg), int)
    assert cfg.slab_row0 == (SEND_SLAB_ROW0 if cfg.xchg_enabled else 0)


@pytest.mark.parametrize(
    ("case_name", "kwargs"),
    [
        ("batch_size_zero", {"sm_scale": 0.1, "batch_size": 0}),
        ("negative_send_world", {"sm_scale": 0.1, "send_world": -1}),
        ("negative_out_offset", {"sm_scale": 0.1, "out_row_offset": -1}),
        ("q_rope_without_split_q", {"sm_scale": 0.1, "q_rope": True}),
        (
            "xchg_without_send",
            {"sm_scale": 0.1, "send_exchange": SendExchangeConfig(axis="dcp")},
        ),
        ("recv_in_without_xchg", {"sm_scale": 0.1, "recv_in": True}),
        (
            "xchg_merge_with_recv_in",
            {
                "sm_scale": 0.1,
                "split_q": True,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "send_world": 4,
                "send_exchange": SendExchangeConfig(axis="dcp", in_kernel_merge=True),
                "recv_in": True,
            },
        ),
        (
            "lse_fill_without_lse_lane_dense",
            {"sm_scale": 0.1, "return_lse": True, "lse_fill": True},
        ),
        (
            "send_without_lse_fill",
            {"sm_scale": 0.1, "split_q": True, "send_world": 4},
        ),
        (
            "send_without_split_q",
            {
                "sm_scale": 0.1,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "send_world": 4,
            },
        ),
        (
            "send_with_preserve_suffix",
            {
                "sm_scale": 0.1,
                "split_q": True,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "preserve_suffix": True,
                "send_world": 4,
            },
        ),
        (
            "has_group_tab_without_capped",
            {"sm_scale": 0.1, "sparsecore_layout": True, "has_group_tab": True},
        ),
        (
            "live_widths_without_has_group_tab",
            {"sm_scale": 0.1, "live_widths": (128,)},
        ),
        (
            "row_widths_with_has_group_tab",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "has_group_tab": True,
                "row_widths": (128,),
            },
        ),
        (
            "body_levers_without_group_or_row_widths",
            {
                "sm_scale": 0.1,
                "body": BodyOptimizationConfig(transpose_nope_unpack=True),
            },
        ),
        (
            "q128_without_q_rope",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "has_group_tab": True,
                "body": BodyOptimizationConfig(narrow_q_rope_dma=True),
            },
        ),
        (
            "pair_without_sparsecore",
            {
                "sm_scale": 0.1,
                "batch_size": 2,
                "row_widths": (128,),
                "body": BodyOptimizationConfig(strategy=BodyExecutionStrategy.PAIR),
            },
        ),
        (
            "pair_with_odd_batch_size",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "batch_size": 3,
                "row_widths": (128,),
                "body": BodyOptimizationConfig(strategy=BodyExecutionStrategy.PAIR),
            },
        ),
        (
            "row_widths_without_capped_windows",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "row_widths": (128,),
            },
        ),
        (
            "row_widths_without_sparsecore",
            {
                "sm_scale": 0.1,
                "row_widths": (128,),
            },
        ),
        (
            "decode_lever_in_mixed_mode",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "has_group_tab": True,
                "body": BodyOptimizationConfig(strategy=BodyExecutionStrategy.QUAD),
            },
        ),
        (
            "mixed_lever_in_decode_mode",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "row_widths": (128,),
                "body": BodyOptimizationConfig(bf16_pv_weights=True),
            },
        ),
        (
            "zu_without_pv512_in_decode_mode",
            {
                "sm_scale": 0.1,
                "sparsecore_layout": True,
                "capped_windows": 4,
                "row_widths": (128,),
                "body": BodyOptimizationConfig(zero_nope_words_only=True),
            },
        ),
        (
            "capped_windows_without_sparsecore",
            {"sm_scale": 0.1, "capped_windows": 4},
        ),
        (
            "capped_windows_non_positive",
            {"sm_scale": 0.1, "sparsecore_layout": True, "capped_windows": 0},
        ),
        (
            "out_in_without_return_lse",
            {"sm_scale": 0.1, "out_in": True},
        ),
        (
            "out_in_with_lse_fill",
            {
                "sm_scale": 0.1,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "out_in": True,
            },
        ),
        (
            "preserve_suffix_with_return_lse",
            {
                "sm_scale": 0.1,
                "preserve_suffix": True,
                "return_lse": True,
            },
        ),
        (
            "preserve_suffix_with_sparsecore_layout",
            {
                "sm_scale": 0.1,
                "preserve_suffix": True,
                "sparsecore_layout": True,
            },
        ),
    ],
)
def test_attention_kernel_config_rejects_invalid_invariants(
    case_name: str, kwargs: dict[str, Any]
) -> None:
    del case_name
    with pytest.raises(ValueError):
        AttentionKernelConfig(**kwargs)


# ---------------------------------------------------------------------------
# 2. PallasRefRegistry, BoundsBuilder/BoundsRef, DmaSemaphores tests
# ---------------------------------------------------------------------------


def test_pallas_ref_registry_rejects_duplicate_names() -> None:
    arr = jnp.zeros((4,), jnp.int32)
    sds = jax.ShapeDtypeStruct((4,), jnp.int32)

    reg = PallasRefRegistry()
    reg.add_prefetch("dup", arr)
    with pytest.raises(ValueError, match="Duplicate Pallas ref name"):
        reg.add_input("dup", arr)

    reg.add_input("in0", arr)
    with pytest.raises(ValueError, match="Duplicate Pallas ref name"):
        reg.add_output("in0", sds)

    reg.add_output("out0", sds)
    with pytest.raises(ValueError, match="Duplicate Pallas ref name"):
        reg.add_scratch("out0", pltpu.VMEM((4,), jnp.int32))


def test_pallas_ref_registry_alias_validation() -> None:
    arr = jnp.zeros((4,), jnp.int32)
    sds = jax.ShapeDtypeStruct((4,), jnp.int32)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    # Unknown alias target.
    reg_unknown = PallasRefRegistry()
    reg_unknown.add_input("x", arr, alias_output="missing")
    reg_unknown.add_output("out", sds)
    with pytest.raises(ValueError, match="unknown output"):
        _ = reg_unknown.input_output_aliases

    # Aliasing an output group is rejected.
    reg_group = PallasRefRegistry()
    reg_group.add_input("x", arr, alias_output="grp")
    reg_group.add_output_group("grp", [sds, sds], [hbm, hbm])
    with pytest.raises(ValueError, match="output group"):
        _ = reg_group.input_output_aliases

    # Duplicate alias target is rejected.
    reg_dup = PallasRefRegistry()
    reg_dup.add_input("x0", arr, alias_output="out")
    reg_dup.add_input("x1", arr, alias_output="out")
    reg_dup.add_output("out", sds)
    with pytest.raises(ValueError, match="aliased by both"):
        _ = reg_dup.input_output_aliases


def test_pallas_ref_registry_output_validation_and_bind_arity() -> None:
    arr = jnp.zeros((4,), jnp.int32)
    sds = jax.ShapeDtypeStruct((4,), jnp.int32)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    empty_reg = PallasRefRegistry()
    with pytest.raises(ValueError, match="At least one output"):
        _ = empty_reg.out_shape

    with pytest.raises(ValueError, match="must be non-empty"):
        empty_reg.add_output_group("grp", [], [])

    with pytest.raises(ValueError, match="BlockSpecs"):
        empty_reg.add_output_group("grp", [sds, sds], [hbm])

    reg = PallasRefRegistry()
    reg.add_prefetch("p", arr)
    reg.add_input("x", arr)
    reg.add_output("y", sds)
    reg.add_scratch("s", pltpu.VMEM((4,), jnp.int32))
    layout = reg.layout
    assert isinstance(hash(layout), int)
    assert layout.total_slots == 4

    with pytest.raises(ValueError, match="Expected 4 Pallas refs"):
        layout.bind("r0", "r1", "r2")
    with pytest.raises(ValueError, match="Expected 4 Pallas refs"):
        layout.bind("r0", "r1", "r2", "r3", "r4")


def test_bounds_builder_and_ref_slot_access_and_errors() -> None:
    empty_builder = BoundsBuilder()
    with pytest.raises(ValueError, match="at least one slot"):
        empty_builder.build()

    builder = BoundsBuilder()
    builder.add("start_seq_idx", jnp.int32(0))
    builder.add("end_seq_idx", jnp.int32(8))
    builder.add("prefix_end", jnp.int32(6))
    builder.add("peer_rank", jnp.int32(1))
    builder.add("peer_lane", jnp.int32(2))
    builder.add("real_tokens", jnp.int32(5))

    with pytest.raises(ValueError, match="Duplicate bounds slot"):
        builder.add("end_seq_idx", jnp.int32(9))

    arr, layout = builder.build()
    assert arr.shape == (6,)
    assert arr.dtype == jnp.int32
    assert len(layout) == 6
    assert isinstance(hash(layout), int)

    raw_sentinel = ["s0", "s1", "s2", "s3", "s4", "s5"]
    ref = BoundsRef(raw_sentinel, layout)
    assert "prefix_end" in ref
    assert "missing" not in ref
    assert ref.start_seq_idx == "s0"
    assert ref.end_seq_idx == "s1"
    assert ref.prefix_end == "s2"
    assert ref.peer_rank == "s3"
    assert ref.peer_lane == "s4"
    assert ref.real_tokens == "s5"
    assert ref["peer_lane"] == "s4"

    with pytest.raises(KeyError, match="missing"):
        _ = ref["missing"]

    small_builder = BoundsBuilder()
    small_builder.add("start_seq_idx", jnp.int32(0))
    _, small_layout = small_builder.build()
    small_ref = BoundsRef(["s0"], small_layout)
    with pytest.raises(KeyError, match="peer_rank"):
        _ = small_ref.peer_rank


def test_dma_semaphores_slot_indexing_and_lse_guard() -> None:
    sems_with_lse = DmaSemaphores(_FakeSemaphoreRef(), has_lse=True)
    assert sems_with_lse.kv_nope(0) == ("sem_slot", (0, 0))
    assert sems_with_lse.q(1) == ("sem_slot", (1, 1))
    assert sems_with_lse.o(0) == ("sem_slot", (2, 0))
    assert sems_with_lse.kv_rope(1) == ("sem_slot", (3, 1))
    assert sems_with_lse.lse(0) == ("sem_slot", (4, 0))

    sems_no_lse = DmaSemaphores(_FakeSemaphoreRef(), has_lse=False)
    assert sems_no_lse.kv_rope(0) == ("sem_slot", (3, 0))
    with pytest.raises(AssertionError, match="return_lse=False"):
        sems_no_lse.lse(0)


@dataclasses.dataclass(frozen=True)
class _AttentionScenario:
    """One mode combination for the `build_attention_ref_registry` matrix."""

    name: str
    cfg: AttentionKernelConfig
    num_tokens: int = 8
    num_q_heads: int = 16
    o_dim: int = 128
    topk_tile: int = 64
    out_piece_rows: int | None = None
    out_piece_carried: bool = False


def _build_scenario_inputs(sc: _AttentionScenario) -> dict[str, Any]:
    cfg = sc.cfg
    q = jnp.zeros((sc.num_tokens, sc.num_q_heads, sc.o_dim), jnp.bfloat16)
    if cfg.sparsecore_layout:
        kv_rows = (cfg.capped_windows or 1) * sc.num_tokens
        cache_kv_nope = jnp.zeros((kv_rows, sc.topk_tile, 128), jnp.uint32)
        cache_kv_rope = jnp.zeros((kv_rows, sc.topk_tile // 4, 128), jnp.uint32)
    else:
        cache_kv_nope = jnp.zeros((sc.num_tokens, sc.topk_tile * 4, 128), jnp.uint8)
        cache_kv_rope = jnp.zeros((sc.num_tokens, sc.topk_tile, 64), jnp.uint8)

    q_pe = None
    if cfg.split_q:
        pe_dim = 256 if cfg.q_rope else 64
        q_pe = jnp.zeros((sc.num_tokens, sc.num_q_heads, pe_dim), jnp.bfloat16)
    q_rope_table = (
        jnp.zeros((sc.num_tokens, 2, 128), jnp.float32) if cfg.q_rope else None
    )

    group_tab = None
    live_views = None
    if cfg.has_group_tab:
        group_tab = jnp.zeros((4,), jnp.int32)
        live_views = (
            jnp.zeros((2, cfg.batch_size, GATHER_ATOM_ROWS, 128), jnp.uint32),
            jnp.zeros((2, cfg.batch_size, GATHER_ATOM_ROWS // 4, 128), jnp.uint32),
        )

    out_rows = sc.out_piece_rows or sc.num_tokens
    lse = None
    if cfg.return_lse and not cfg.lse_fill:
        lse_shape = (
            (out_rows, sc.num_q_heads)
            if cfg.lse_lane_dense
            else (out_rows, sc.num_q_heads, 1)
        )
        lse = jnp.zeros(lse_shape, jnp.float32)

    out_piece = None
    if sc.out_piece_rows is not None:
        carried = (
            jnp.zeros((out_rows, sc.num_q_heads, sc.o_dim), jnp.bfloat16)
            if sc.out_piece_carried
            else None
        )
        out_piece = (out_rows, cfg.out_row_offset, carried)

    send_peer = None
    if cfg.xchg_enabled:
        send_peer = (
            (jnp.int32(0), jnp.int32(0), jnp.int32(sc.num_tokens))
            if cfg.xchg_merge
            else (jnp.int32(0), jnp.int32(0))
        )

    send_recv_in = None
    if cfg.recv_in:
        s0, s1, s2 = send_layout_shape(
            sc.num_tokens, sc.num_q_heads, cfg.send_world, sc.o_dim
        )
        send_recv_in = jnp.zeros((s0 * s1, s2), jnp.int16)

    return {
        "q": q,
        "cache_kv_nope": cache_kv_nope,
        "cache_kv_rope": cache_kv_rope,
        "kv_lens": jnp.zeros((sc.num_tokens,), jnp.int32),
        "start_seq_idx": jnp.int32(0),
        "end_seq_idx": jnp.int32(sc.num_tokens),
        "token_end": jnp.int32(sc.num_tokens) if cfg.preserve_suffix else None,
        "group_tab": group_tab,
        "q_pe": q_pe,
        "q_rope_table": q_rope_table,
        "live_views": live_views,
        "lse": lse,
        "send_recv_in": send_recv_in,
        "out_piece": out_piece,
        "send_peer": send_peer,
    }


def _expected_attention_ground_truth(
    sc: _AttentionScenario, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Independent ground-truth derivation of grid spec, aliases, and scratches."""
    cfg = sc.cfg
    q = inputs["q"]
    cache_kv_nope = inputs["cache_kv_nope"]
    cache_kv_rope = inputs["cache_kv_rope"]
    q_pe = inputs["q_pe"]
    q_rope_table = inputs["q_rope_table"]
    lse = inputs["lse"]
    out_piece = inputs["out_piece"]
    num_tokens, num_q_heads, o_dim = q.shape
    out_rows = num_tokens if out_piece is None else out_piece[0]

    bounds_names = ["start_seq_idx", "end_seq_idx"]
    if cfg.preserve_suffix:
        bounds_names.append("prefix_end")
    if cfg.xchg_enabled:
        bounds_names.extend(["peer_rank", "peer_lane"])
        if cfg.xchg_merge:
            bounds_names.append("real_tokens")

    prefetch_names = ["kv_lens", "bounds", "sem_ids"]
    if cfg.has_group_tab:
        prefetch_names.append("group_tab")

    input_names = ["q", "cache_kv_nope", "cache_kv_rope"]
    if cfg.split_q:
        input_names.append("q_pe")
    if cfg.q_rope:
        input_names.append("q_rope_table")
    if cfg.has_group_tab:
        input_names.extend(["live_nope", "live_rope"])
    if cfg.return_lse and not cfg.lse_fill:
        input_names.append("lse_in")
    elif cfg.return_lse and cfg.recv_in:
        input_names.append("recv_in")
    if cfg.out_in:
        input_names.append("out_carried")

    num_pf = len(prefetch_names)
    aliases: dict[int, int] = {}
    if out_piece is None:
        aliases[num_pf + input_names.index("q")] = 0
    if "lse_in" in input_names:
        aliases[num_pf + input_names.index("lse_in")] = 1
    if "recv_in" in input_names:
        aliases[num_pf + input_names.index("recv_in")] = 1
    if "out_carried" in input_names:
        aliases[num_pf + input_names.index("out_carried")] = 0

    out_structs = [jax.ShapeDtypeStruct((out_rows, num_q_heads, o_dim), q.dtype)]
    if cfg.return_lse:
        if cfg.xchg_merge:
            lse_sds = jax.ShapeDtypeStruct(
                (num_tokens * num_q_heads // cfg.send_world, o_dim), jnp.int16
            )
        elif cfg.xchg_enabled:
            s0, s1, _ = send_layout_shape(
                num_tokens, num_q_heads, cfg.send_world, o_dim
            )
            lse_sds = jax.ShapeDtypeStruct((s0 * s1, o_dim), jnp.int16)
        elif cfg.send_enabled:
            lse_sds = jax.ShapeDtypeStruct(
                send_layout_shape(num_tokens, num_q_heads, cfg.send_world, o_dim),
                jnp.int16,
            )
        else:
            lse_shape = (num_tokens, num_q_heads) if cfg.lse_fill else lse.shape
            lse_sds = jax.ShapeDtypeStruct(lse_shape, jnp.float32)
        out_structs.append(lse_sds)

    if cfg.split_q:
        pe_dim = Q_ROPE_DIM if cfg.q_rope else q_pe.shape[-1]
        head_dim = align_to(o_dim + pe_dim, 128)
    else:
        head_dim = o_dim
    send_lanes = (
        send_layout_lanes(num_tokens, num_q_heads, cfg.send_world)
        if cfg.send_enabled
        else 0
    )

    scratches: list[tuple[str, Any]] = [
        (
            "bkv_nope_x2",
            pltpu.VMEM(
                (2, cfg.batch_size, *cache_kv_nope.shape[1:]), cache_kv_nope.dtype
            ),
        ),
        (
            "bkv_rope_x2",
            pltpu.VMEM(
                (2, cfg.batch_size, *cache_kv_rope.shape[1:]), cache_kv_rope.dtype
            ),
        ),
        (
            "bq_x2",
            pltpu.VMEM((2, cfg.batch_size, num_q_heads, o_dim), q.dtype),
        ),
        (
            "bo_x2",
            pltpu.VMEM(
                (
                    2,
                    cfg.send_world,
                    cfg.slab_row0 + cfg.batch_size * num_q_heads // cfg.send_world,
                    o_dim,
                ),
                jnp.int16,
            )
            if cfg.send_enabled
            else pltpu.VMEM((2, cfg.batch_size, num_q_heads, o_dim), q.dtype),
        ),
    ]
    if cfg.return_lse:
        blse_shape = (
            (2, cfg.batch_size, send_lanes or num_q_heads)
            if cfg.lse_lane_dense
            else (2, cfg.batch_size, num_q_heads, 1)
        )
        scratches.append(("blse_x2", pltpu.VMEM(blse_shape, jnp.float32)))
    scratches.append(
        ("dma_sems", pltpu.SemaphoreType.DMA((5 if cfg.return_lse else 4, 2)))
    )
    if cfg.capped_windows is not None:
        scratches.extend(
            [
                (
                    "fix_nope_x2",
                    pltpu.VMEM((2, *cache_kv_nope.shape[1:]), cache_kv_nope.dtype),
                ),
                (
                    "fix_rope_x2",
                    pltpu.VMEM((2, *cache_kv_rope.shape[1:]), cache_kv_rope.dtype),
                ),
                ("fix_m", pltpu.VMEM((num_q_heads, 1), jnp.float32)),
                ("fix_l", pltpu.VMEM((num_q_heads, 1), jnp.float32)),
                ("fix_acc", pltpu.VMEM((num_q_heads, head_dim), jnp.float32)),
                ("fix_sems", pltpu.SemaphoreType.DMA((2, 2))),
            ]
        )
    if cfg.lse_fill and cfg.send_enabled:
        scratches.append(
            (
                "fill",
                pltpu.VMEM(
                    (num_tokens // cfg.batch_size, cfg.batch_size, send_lanes),
                    jnp.float32,
                ),
            )
        )
        if cfg.xchg_enabled:
            scratches.append(("xchg_sems", pltpu.SemaphoreType.DMA((3,))))
        else:
            scratches.append(
                (
                    "send_wbuf",
                    pltpu.VMEM((cfg.send_world, SEND_PAD_ROWS, o_dim), jnp.int16),
                )
            )
        scratches.append(("fill_sem", pltpu.SemaphoreType.DMA(())))
    elif cfg.lse_fill:
        scratches.append(("fill", pltpu.VMEM((num_tokens, num_q_heads), jnp.float32)))
        scratches.append(("fill_sem", pltpu.SemaphoreType.DMA(())))
    if cfg.q_rope:
        scratches.append(
            (
                "bq_tab_x2",
                pltpu.VMEM((2, cfg.batch_size, *q_rope_table.shape[1:]), jnp.float32),
            )
        )
    if cfg.split_q:
        pe_lanes = 128 if cfg.body.narrow_q_rope_dma else q_pe.shape[-1]
        scratches.append(
            (
                "bq_pe_x2",
                pltpu.VMEM((2, cfg.batch_size, num_q_heads, pe_lanes), q_pe.dtype),
            )
        )
    if cfg.body.uses_kv_stage:
        kv_stage_shape = (
            (1, cfg.batch_size, cache_kv_nope.shape[1], head_dim)
            if cfg.body.strategy == BodyExecutionStrategy.QUAD
            else (2, 2, cache_kv_nope.shape[1], head_dim)
        )
        scratches.append(("kv_stage", pltpu.VMEM(kv_stage_shape, jnp.float8_e4m3fn)))
    if cfg.xchg_merge:
        scratches.append(
            (
                "recv_v",
                pltpu.VMEM(
                    (
                        cfg.send_world,
                        SEND_SLAB_ROW0 + cfg.batch_size * num_q_heads // cfg.send_world,
                        o_dim,
                    ),
                    jnp.int16,
                ),
            )
        )

    return {
        "bounds_names": tuple(bounds_names),
        "prefetch_names": prefetch_names,
        "input_names": input_names,
        "output_names": ["out", "lse"] if cfg.return_lse else ["out"],
        "out_structs": out_structs,
        "aliases": aliases,
        "scratches": scratches,
    }


_ATTENTION_SCENARIOS = [
    _AttentionScenario("tensorcore_basic", AttentionKernelConfig(sm_scale=0.1)),
    _AttentionScenario(
        "tensorcore_preserve_suffix",
        AttentionKernelConfig(sm_scale=0.1, preserve_suffix=True),
    ),
    _AttentionScenario(
        "tensorcore_lse_3d",
        AttentionKernelConfig(sm_scale=0.1, return_lse=True, lse_lane_dense=False),
    ),
    _AttentionScenario(
        "sparsecore_basic",
        AttentionKernelConfig(sm_scale=0.1, sparsecore_layout=True),
    ),
    _AttentionScenario(
        "sparsecore_split_q_lse_dense",
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            split_q=True,
            return_lse=True,
            lse_lane_dense=True,
        ),
    ),
    _AttentionScenario(
        "sparsecore_q_rope_lse_fill",
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
        ),
    ),
    _AttentionScenario(
        "sparsecore_capped_quad_q128",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            row_widths=(128, 256),
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            body=BodyOptimizationConfig.from_levers(
                (*DECODE_BODY_DEFAULT, "q128"), mode="decode"
            ),
        ),
    ),
    _AttentionScenario(
        "sparsecore_live_views_pair_q128",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            has_group_tab=True,
            live_widths=(256, 384),
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            body=BodyOptimizationConfig.from_levers(
                ("tr", "pair", "q128"), mode="mixed"
            ),
        ),
    ),
    _AttentionScenario(
        "out_piece_fresh",
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            return_lse=True,
            lse_lane_dense=True,
            out_row_offset=0,
            out_in=False,
        ),
        out_piece_rows=16,
        out_piece_carried=False,
    ),
    _AttentionScenario(
        "out_piece_carried",
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            return_lse=True,
            lse_lane_dense=True,
            out_row_offset=8,
            out_in=True,
        ),
        out_piece_rows=16,
        out_piece_carried=True,
    ),
    _AttentionScenario(
        "send_world_only",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
        ),
    ),
    _AttentionScenario(
        "send_exchange_no_recv_in",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            send_exchange=SendExchangeConfig(
                axis="dcp", lane_axis="tp", hbm_pin=(HbmPinRole.KV, HbmPinRole.Q_PE)
            ),
        ),
    ),
    _AttentionScenario(
        "send_exchange_recv_in",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            send_exchange=SendExchangeConfig(axis="dcp", lane_axis="tp", hbm_pin=True),
            recv_in=True,
        ),
    ),
    _AttentionScenario(
        "send_exchange_xchg_merge_live_pair",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            capped_windows=4,
            has_group_tab=True,
            live_widths=(128, 256),
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            send_exchange=SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                hbm_pin=(HbmPinRole.KV, HbmPinRole.Q, HbmPinRole.TAB),
                in_kernel_merge=True,
            ),
            body=BodyOptimizationConfig.from_levers(("tr", "pair"), mode="mixed"),
        ),
    ),
]


@pytest.mark.parametrize(
    "scenario", _ATTENTION_SCENARIOS, ids=[sc.name for sc in _ATTENTION_SCENARIOS]
)
def test_build_attention_ref_registry_equivalence_matrix(
    scenario: _AttentionScenario,
) -> None:
    inputs = _build_scenario_inputs(scenario)
    expected = _expected_attention_ground_truth(scenario, inputs)

    reg, bounds_layout = build_attention_ref_registry(scenario.cfg, **inputs)
    grid_spec = reg.build_grid_spec((1,))

    assert bounds_layout.names == expected["bounds_names"]
    assert grid_spec.num_scalar_prefetch == len(expected["prefetch_names"])
    assert len(grid_spec.in_specs) == len(expected["input_names"])

    if len(expected["out_structs"]) == 1:
        assert isinstance(grid_spec.out_specs, pl.BlockSpec)
        assert isinstance(reg.out_shape, jax.ShapeDtypeStruct)
        assert reg.out_shape == expected["out_structs"][0]
    else:
        assert isinstance(grid_spec.out_specs, list)
        assert len(grid_spec.out_specs) == len(expected["out_structs"])
        assert reg.out_shape == expected["out_structs"]

    assert reg.input_output_aliases == expected["aliases"]

    expected_scratch_specs = tuple(spec for _, spec in expected["scratches"])
    assert tuple(grid_spec.scratch_shapes) == expected_scratch_specs

    # Round-trip AttentionRefs binding from positional sentinel refs.
    expected_slot_names = (
        expected["prefetch_names"]
        + expected["input_names"]
        + expected["output_names"]
        + [name for name, _ in expected["scratches"]]
    )
    sentinels = [f"ref:{idx}:{name}" for idx, name in enumerate(expected_slot_names)]
    by_name = dict(zip(expected_slot_names, sentinels, strict=True))

    bound = reg.layout.bind(*sentinels)
    refs = AttentionRefs.from_bound(
        bound, bounds_layout, return_lse=scenario.cfg.return_lse
    )

    assert refs.kv_lens == by_name["kv_lens"]
    assert refs.bounds.raw == by_name["bounds"]
    assert refs.bounds.layout == bounds_layout
    assert refs.sem_ids == by_name["sem_ids"]
    assert refs.group_tab == by_name.get("group_tab")
    assert refs.q == by_name["q"]
    assert refs.cache_kv_nope == by_name["cache_kv_nope"]
    assert refs.cache_kv_rope == by_name["cache_kv_rope"]
    assert refs.q_pe == by_name.get("q_pe")
    assert refs.q_rope_table == by_name.get("q_rope_table")
    assert refs.live_nope == by_name.get("live_nope")
    assert refs.live_rope == by_name.get("live_rope")
    assert refs.out == by_name["out"]
    assert refs.lse == by_name.get("lse")
    assert refs.bkv_nope_x2 == by_name["bkv_nope_x2"]
    assert refs.bkv_rope_x2 == by_name["bkv_rope_x2"]
    assert refs.bq_x2 == by_name["bq_x2"]
    assert refs.bo_x2 == by_name["bo_x2"]
    assert refs.blse_x2 == by_name.get("blse_x2")
    assert refs.sems.raw == by_name["dma_sems"]
    assert refs.sems.has_lse is scenario.cfg.return_lse
    assert refs.bq_pe_x2 == by_name.get("bq_pe_x2")
    assert refs.bq_tab_x2 == by_name.get("bq_tab_x2")
    assert refs.kv_stage == by_name.get("kv_stage")
    assert refs.recv_v == by_name.get("recv_v")

    if scenario.cfg.capped_windows is not None:
        assert refs.fixup is not None
        assert refs.fixup.nope_x2 == by_name["fix_nope_x2"]
        assert refs.fixup.rope_x2 == by_name["fix_rope_x2"]
        assert refs.fixup.m == by_name["fix_m"]
        assert refs.fixup.l_sum == by_name["fix_l"]
        assert refs.fixup.acc == by_name["fix_acc"]
        assert refs.fixup.sems == by_name["fix_sems"]
    else:
        assert refs.fixup is None

    if scenario.cfg.lse_fill:
        assert refs.lse_fill is not None
        assert refs.lse_fill.fill == by_name["fill"]
        assert refs.lse_fill.sem == by_name["fill_sem"]
        assert refs.lse_fill.send_wbuf == by_name.get("send_wbuf")
        assert refs.lse_fill.xchg_sems == by_name.get("xchg_sems")
    else:
        assert refs.lse_fill is None


@pytest.mark.parametrize(
    ("hbm_pin", "expected_pinned_indices"),
    [
        (None, set()),
        (False, set()),
        (True, {0, 1, 2, 3, 4, 5}),
        ((HbmPinRole.KV, HbmPinRole.RECV), {1, 2, 5}),
        ((HbmPinRole.Q, HbmPinRole.Q_PE, HbmPinRole.TAB), {0, 3, 4}),
    ],
)
def test_build_attention_ref_registry_hbm_pinning(
    hbm_pin: bool | tuple[HbmPinRole, ...] | None,
    expected_pinned_indices: set[int],
) -> None:
    sc = _AttentionScenario(
        "pin_test",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            send_exchange=SendExchangeConfig(
                axis="dcp",
                lane_axis="tp",
                hbm_pin=True if hbm_pin is None else hbm_pin,
            ),
            recv_in=True,
        ),
    )
    inputs = _build_scenario_inputs(sc)
    reg, _ = build_attention_ref_registry(sc.cfg, **inputs)

    pinned_markers: list[Any] = []

    def _fake_pin(val: Any, space: Any) -> tuple[str, Any, Any]:
        marker = ("PINNED", space, val)
        pinned_markers.append(marker)
        return marker

    assert sc.cfg.send_exchange is not None
    pin_fn = None if hbm_pin is None else sc.cfg.send_exchange.should_pin
    with patch.object(
        ref_registry.pltpu, "with_memory_space_constraint", side_effect=_fake_pin
    ):
        args = reg.call_args(pin_fn=pin_fn)

    num_pf = reg.layout.num_prefetches
    input_args = args[num_pf:]
    assert len(input_args) == 6
    actual_pinned = {
        idx
        for idx, arg in enumerate(input_args)
        if isinstance(arg, tuple) and arg[0] == "PINNED" and arg[1] == pltpu.HBM
    }
    assert actual_pinned == expected_pinned_indices


@pytest.mark.parametrize(
    ("case_name", "cfg_kwargs", "input_overrides"),
    [
        ("missing_q_pe", {"split_q": True}, {"q_pe": None}),
        (
            "missing_q_rope_table",
            {"split_q": True, "q_rope": True},
            {"q_rope_table": None},
        ),
        (
            "missing_live_views",
            {"sparsecore_layout": True, "capped_windows": 2, "has_group_tab": True},
            {"live_views": None},
        ),
        (
            "missing_group_tab",
            {"sparsecore_layout": True, "capped_windows": 2, "has_group_tab": True},
            {"group_tab": None},
        ),
        (
            "missing_token_end",
            {"preserve_suffix": True},
            {"token_end": None},
        ),
        (
            "missing_lse",
            {"return_lse": True},
            {"lse": None},
        ),
        (
            "missing_send_recv_in",
            {
                "batch_size": 8,
                "split_q": True,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "send_world": 4,
                "send_exchange": SendExchangeConfig(axis="dcp"),
                "recv_in": True,
            },
            {"send_recv_in": None},
        ),
        (
            "short_send_peer_for_merge",
            {
                "batch_size": 8,
                "split_q": True,
                "return_lse": True,
                "lse_lane_dense": True,
                "lse_fill": True,
                "send_world": 4,
                "send_exchange": SendExchangeConfig(axis="dcp", in_kernel_merge=True),
            },
            {"send_peer": (jnp.int32(0), jnp.int32(0))},
        ),
        (
            "out_piece_offset_mismatch",
            {
                "return_lse": True,
                "lse_lane_dense": True,
                "out_row_offset": 4,
            },
            {"out_piece": (16, 0, None)},
        ),
    ],
)
def test_build_attention_ref_registry_rejects_mismatched_inputs(
    case_name: str,
    cfg_kwargs: dict[str, Any],
    input_overrides: dict[str, Any],
) -> None:
    del case_name
    cfg = AttentionKernelConfig(sm_scale=0.1, **cfg_kwargs)
    sc = _AttentionScenario("err", cfg)
    inputs = _build_scenario_inputs(sc)
    inputs.update(input_overrides)
    with pytest.raises(ValueError):
        build_attention_ref_registry(cfg, **inputs)


@pytest.mark.parametrize("num_chunks", [1, 4])
def test_build_live_fused_prep_ref_registry(num_chunks: int) -> None:
    chunk_size = 128
    batch_size = 16
    topk = 256
    topk_cap = 128
    num_tokens = num_chunks * chunk_size
    padded_pps = 128
    num_windows = topk // topk_cap
    region_rows = chunk_size * topk_cap // 128
    gpc = chunk_size // batch_size

    reg = build_live_fused_prep_ref_registry(
        token_end=jnp.int32(num_tokens),
        windows=jnp.zeros((num_tokens, padded_pps), jnp.int32),
        topk_indices=jnp.zeros((num_tokens, topk), jnp.int32),
        chunk_size=chunk_size,
        batch_size=batch_size,
        topk=topk,
        topk_cap=topk_cap,
    )
    grid_spec = reg.build_grid_spec((num_chunks,))

    assert grid_spec.num_scalar_prefetch == 1
    assert len(grid_spec.in_specs) == 2
    assert isinstance(grid_spec.out_specs, list)
    assert len(grid_spec.out_specs) == num_chunks + 3
    assert isinstance(reg.out_shape, list)
    assert len(reg.out_shape) == num_chunks + 3
    assert reg.input_output_aliases == {}

    expected_scratches = (
        pltpu.VMEM((2, num_windows * region_rows, 128), jnp.int32),
        pltpu.VMEM((topk_cap // 128 * chunk_size, 128), jnp.int32),
        pltpu.SMEM((num_chunks,), jnp.int32),
        pltpu.SMEM((2 * gpc,), jnp.int32),
        pltpu.SemaphoreType.DMA((2 * num_windows,)),
    )
    assert tuple(grid_spec.scratch_shapes) == expected_scratches

    total_slots = 1 + 2 + (num_chunks + 3) + 5
    sentinels = [f"slot:{i}" for i in range(total_slots)]
    refs = LiveFusedPrepRefs.from_bound(reg.layout.bind(*sentinels))

    assert refs.token_end == "slot:0"
    assert refs.windows == "slot:1"
    assert refs.topk == "slot:2"
    assert isinstance(refs.chunk_outs, tuple)
    assert len(refs.chunk_outs) == num_chunks
    assert refs.chunk_outs == tuple(f"slot:{3 + i}" for i in range(num_chunks))

    tail_base = 3 + num_chunks
    assert refs.kv == f"slot:{tail_base}"
    assert refs.tab == f"slot:{tail_base + 1}"
    assert refs.live == f"slot:{tail_base + 2}"
    assert refs.stage == f"slot:{tail_base + 3}"
    assert refs.phys == f"slot:{tail_base + 4}"
    assert refs.cw == f"slot:{tail_base + 5}"
    assert refs.ab == f"slot:{tail_base + 6}"
    assert refs.sems == f"slot:{tail_base + 7}"


# ---------------------------------------------------------------------------
# 3. FaultHooks contract tests
# ---------------------------------------------------------------------------


def test_no_fault_hooks_hashable_and_identity() -> None:
    assert isinstance(hash(NO_FAULT_HOOKS), int)
    assert FaultHooks() == NO_FAULT_HOOKS

    lo_src = jnp.array([1, 2, 3, 4], dtype=jnp.int16)
    w_lo = jnp.array([0.25, 0.5], dtype=jnp.float32)
    me = jnp.int32(1)
    kv_row = jnp.zeros((2, 8), dtype=jnp.int32)
    off = jnp.int32(7)
    pieces = [jnp.array([1]), jnp.array([2]), jnp.array([3])]
    mem_ref = _RecordingMemRef((2, 4))

    assert NO_FAULT_HOOKS.q_rope_roll_shifts() == (96, 32)
    assert NO_FAULT_HOOKS.effective_row_widths((128, 256)) == (128, 256, None)
    assert NO_FAULT_HOOKS.lse_fill_sentinel(-1e30) == -1e30
    assert NO_FAULT_HOOKS.send_dest_rank(2, 4) == 2
    assert NO_FAULT_HOOKS.transform_send_lse_lo(lo_src) is lo_src
    NO_FAULT_HOOKS.poison_merge_recv_scratch(mem_ref)
    assert mem_ref.writes == []
    assert NO_FAULT_HOOKS.wait_exchange_barrier_at_entry() is False
    assert NO_FAULT_HOOKS.wait_exchange_barrier_in_round() is True
    assert NO_FAULT_HOOKS.exchange_local_only_round() is False
    assert NO_FAULT_HOOKS.exchange_expect_peers(4) == 3
    assert NO_FAULT_HOOKS.exchange_dst_slot(me, 4) is me
    assert bool(NO_FAULT_HOOKS.exchange_should_skip_peer(1, me, 4)) is True
    assert bool(NO_FAULT_HOOKS.exchange_should_skip_peer(0, me, 4)) is False
    assert NO_FAULT_HOOKS.transform_merge_w_lo(w_lo) is w_lo
    assert NO_FAULT_HOOKS.corrupt_live_prep_kv_row(kv_row) is kv_row
    assert NO_FAULT_HOOKS.corrupt_live_prep_offset(off, 0) is off
    assert NO_FAULT_HOOKS.live_prep_overflow_windows_end(4) == 4
    assert NO_FAULT_HOOKS.transform_piece_outs(pieces) is pieces


def test_fault_hooks_from_names_valid_and_singleton() -> None:
    assert FaultHooks.from_names() is NO_FAULT_HOOKS
    assert FaultHooks.from_names(allow_send_exchange=True) is NO_FAULT_HOOKS

    for name in PROLOGUE_FAULTS:
        hooks = FaultHooks.from_names(prologue_fault=name)
        assert hooks.prologue == (None if name is None else PrologueFault(name))
        assert isinstance(hash(hooks), int)

    for name in SEND_FAULTS:
        hooks = FaultHooks.from_names(prologue_fault=name)
        assert hooks.prologue == (None if name is None else SendFault(name))

    for name in SEND_EXCHANGE_FAULTS:
        hooks = FaultHooks.from_names(prologue_fault=name, allow_send_exchange=True)
        assert hooks.prologue is SendExchangeFault(name)

    for name in LIVE_PREP_FAULTS:
        hooks = FaultHooks.from_names(live_prep_fault=name)
        assert hooks.live_prep == (None if name is None else LivePrepFault(name))

    for name in OUT_PIECES_FAULTS:
        hooks = FaultHooks.from_names(out_pieces_fault=name)
        assert hooks.out_pieces == (None if name is None else OutPiecesFault(name))

    kv_hooks = FaultHooks.from_names(kv_write_fault="skip_write")
    assert kv_hooks.kv_write == "skip_write"
    assert kv_hooks is not NO_FAULT_HOOKS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prologue_fault": "not_a_fault"},
        {"live_prep_fault": "not_a_fault"},
        {"out_pieces_fault": "not_a_fault"},
        *[
            {"prologue_fault": name, "allow_send_exchange": False}
            for name in SEND_EXCHANGE_FAULTS
        ],
    ],
)
def test_fault_hooks_from_names_rejects_unknown_and_gates_send_exchange(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        FaultHooks.from_names(**kwargs)


def test_fault_hooks_positive_controls_scalar_and_routing() -> None:
    rw_hooks = FaultHooks.from_names(prologue_fault="row_width_undersize")
    assert rw_hooks.effective_row_widths((128, 256)) == (128, 128, 256)

    rot_hooks = FaultHooks.from_names(prologue_fault="send_dest_rotate")
    assert [rot_hooks.send_dest_rank(d, 4) for d in range(4)] == [1, 2, 3, 0]

    lse_hooks = FaultHooks.from_names(prologue_fault="skip_lse_fill")
    assert lse_hooks.lse_fill_sentinel(-1e30) == 0.0

    swap_rope = FaultHooks.from_names(prologue_fault="rope_swap")
    assert swap_rope.q_rope_roll_shifts() == (32, 96)

    drop_hooks = FaultHooks.from_names(
        prologue_fault="send_peer_drop", allow_send_exchange=True
    )
    assert drop_hooks.exchange_expect_peers(4) == 2

    entry_wait = FaultHooks.from_names(
        prologue_fault="send_xchg_entry_wait", allow_send_exchange=True
    )
    assert entry_wait.wait_exchange_barrier_at_entry() is True
    assert entry_wait.wait_exchange_barrier_in_round() is False

    rm_round = FaultHooks.from_names(
        prologue_fault="send_xchg_rm_round", allow_send_exchange=True
    )
    assert rm_round.exchange_local_only_round() is True

    win1_hooks = FaultHooks.from_names(live_prep_fault="win1")
    assert win1_hooks.live_prep_overflow_windows_end(4) == 1

    swap_pieces = FaultHooks.from_names(out_pieces_fault="swap")
    piece_list: Any = ["p0", "p1", "p2"]
    assert swap_pieces.transform_piece_outs(piece_list) == ["p1", "p0", "p2"]


def test_fault_hooks_positive_controls_array_transforms() -> None:
    # Q-RoPE math: SKIP_Q_ROPE returns xf; ROPE_F32 skips intermediate bf16 cast.
    xf = jnp.array([1.001, 2.001], dtype=jnp.float32)
    rot = jnp.array([0.123, -0.456], dtype=jnp.float32)
    c = jnp.array([0.877, 0.707], dtype=jnp.float32)
    s = jnp.array([0.479, 0.707], dtype=jnp.float32)

    skip_rope = FaultHooks.from_names(prologue_fault="skip_q_rope")
    assert skip_rope.apply_q_rope_math(xf, rot, c, s, jnp.bfloat16) is xf

    f32_rope = FaultHooks.from_names(prologue_fault="rope_f32")
    res_f32 = f32_rope.apply_q_rope_math(xf, rot, c, s, jnp.bfloat16)
    res_default = NO_FAULT_HOOKS.apply_q_rope_math(xf, rot, c, s, jnp.bfloat16)
    assert res_f32.tolist() == pytest.approx((xf * c + rot * s).tolist())
    assert res_default.tolist() != res_f32.tolist()

    # Send & merge low-word zeroing faults.
    lo_src = jnp.array([7, -3, 5], dtype=jnp.int16)
    lo_skip = FaultHooks.from_names(prologue_fault="send_lse_lo_skip")
    assert lo_skip.transform_send_lse_lo(lo_src).tolist() == [0, 0, 0]

    w_lo = jnp.array([0.5, -0.25], dtype=jnp.float32)
    merge_skip = FaultHooks.from_names(
        prologue_fault="send_merge_lo_skip", allow_send_exchange=True
    )
    assert merge_skip.transform_merge_w_lo(w_lo).tolist() == [0.0, 0.0]

    # Peer misorder / drop poisoning & routing.
    misorder = FaultHooks.from_names(
        prologue_fault="send_peer_misorder", allow_send_exchange=True
    )
    assert int(misorder.exchange_dst_slot(jnp.int32(3), 4)) == 0
    mem_ref = _RecordingMemRef((2, 3))
    misorder.poison_merge_recv_scratch(mem_ref)
    assert len(mem_ref.writes) == 1
    key, payload = mem_ref.writes[0]
    assert key is Ellipsis
    assert payload.shape == (2, 3)
    assert payload.dtype == jnp.int16
    assert payload.tolist() == [[0x7FC0, 0x7FC0, 0x7FC0], [0x7FC0, 0x7FC0, 0x7FC0]]

    drop = FaultHooks.from_names(
        prologue_fault="send_peer_drop", allow_send_exchange=True
    )
    me = jnp.int32(1)
    skipped = [bool(drop.exchange_should_skip_peer(p, me, 4)) for p in range(4)]
    assert skipped == [True, True, False, False]  # peer 0 == (1+3)%4 and peer 1 == me

    # Live prep KV lane-5 and offset group-0 perturbations.
    kv_fault = FaultHooks.from_names(live_prep_fault="kv")
    corrupted = kv_fault.corrupt_live_prep_kv_row(jnp.zeros((2, 8), dtype=jnp.int32))
    assert corrupted.tolist() == [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ]

    off_fault = FaultHooks.from_names(live_prep_fault="off")
    base_off = jnp.int32(10)
    assert int(off_fault.corrupt_live_prep_offset(base_off, 0)) == 11
    assert off_fault.corrupt_live_prep_offset(base_off, 1) is base_off


# ---------------------------------------------------------------------------
# 4. Backward-compatibility export contract tests
# ---------------------------------------------------------------------------


_EXPECTED_KERNEL_EXPORTS = (
    # Legacy constants
    "DEFAULT_VMEM_LIMIT_BYTES",
    "DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE",
    "DEFAULT_ATTENTION_KERNEL_BATCH_SIZE",
    "GATHER_ATOM_ROWS",
    "FUSED_PREP_MAX_TOKENS",
    "FUSED_PREP_META_ROWS",
    "Q_ROPE_DIM",
    "Q_ROPE_HEAD_DIM",
    "DECODE_STEP8_TOKENS",
    "MIXED_BODY_MAX_SKIP_TOKENS",
    "SEND_PAD_ROWS",
    "SEND_SLAB_ROW0",
    "SEND_EXCHANGE_COLLECTIVE_ID",
    "SEND_EXCHANGE_VMEM_LIMIT_BYTES",
    "SEND_EXCHANGE_PIN_ROLES",
    "TENSORCORE_CACHE_LAYOUT",
    "SPARSECORE_CACHE_LAYOUT",
    "MIXED_BODY_LEVERS",
    "MIXED_BODY_DEFAULT",
    "DECODE_BODY_LEVERS",
    "DECODE_BODY_DEFAULT",
    "MIXED_KERNEL_V2_LEVERS",
    "PROLOGUE_FAULTS",
    "SEND_FAULTS",
    "SEND_EXCHANGE_FAULTS",
    # Typed refactor exports
    "CacheLayout",
    "GatherWaitMode",
    "BodyExecutionStrategy",
    "HbmPinRole",
    "BodyOptimizationConfig",
    "SendExchangeConfig",
    "GatherConfig",
    "CappedWindowConfig",
    "LseConfig",
    "DcpConfig",
    "SparseMlaConfig",
    "AttentionKernelConfig",
    "PallasRefRegistry",
    "FaultHooks",
    "NO_FAULT_HOOKS",
    # Helpers and entry points
    "cdiv",
    "align_to",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "live_group_supported",
    "mixed_fused_prep_supported",
    "merge_pieces_supported",
    "send_layout_lanes",
    "send_layout_supported",
    "send_layout_shape",
    "_largest_divisor",
    "_chunk_kernel_batch_size",
    "gather_page_ids",
    "scatter_padding_slots",
    "prepare_capped_window_indices",
    "prepare_capped_window_live",
    "prepare_capped_window_live_fused",
    "prepare_capped_window_fused",
    "live_group_table",
    "capped_live_windows",
    "rope_q_block",
    "prepare_q_inputs",
    "prepare_outputs",
    "prepare_gather_indices",
    "select_attention_kernel",
    "_attention_kernel",
    "_decode_attention_kernel",
    "_mixed_prefill_attention_kernel",
    "sparse_attention_chunk",
    "sparse_ragged_paged_attention",
)


def test_kernel_backward_compatible_exports() -> None:
    for name in _EXPECTED_KERNEL_EXPORTS:
        assert hasattr(sparse_kernel, name), f"kernel.py missing export {name!r}"

    # Verify re-exported modules match canonical module objects.
    assert sparse_kernel.pl is pl
    assert sparse_kernel.dsa_gather is dsa_gather
    assert sparse_kernel.native_sc_gather is native_sc_gather

    # Verify private aliases and typed singletons point to canonical definitions.
    assert sparse_kernel._largest_divisor is sparse_config.largest_divisor
    assert (
        sparse_kernel._chunk_kernel_batch_size is sparse_config.chunk_kernel_batch_size
    )
    assert sparse_kernel.NO_FAULT_HOOKS is testing_faults.NO_FAULT_HOOKS
    assert sparse_kernel.FaultHooks is testing_faults.FaultHooks
    assert sparse_kernel.PallasRefRegistry is ref_registry.PallasRefRegistry
    assert (
        sparse_kernel.select_attention_kernel
        is attention_kernel.select_attention_kernel
    )
    assert sparse_kernel._attention_kernel is attention_kernel._attention_kernel
    assert (
        sparse_kernel._decode_attention_kernel
        is attention_kernel._decode_attention_kernel
    )
    assert (
        sparse_kernel._mixed_prefill_attention_kernel
        is attention_kernel._mixed_prefill_attention_kernel
    )

    # Verify jit-wrapped entry point preserves __wrapped__ for callers/tests.
    wrapped = getattr(sparse_kernel.sparse_ragged_paged_attention, "__wrapped__", None)
    assert callable(wrapped)


# ---------------------------------------------------------------------------
# 5. Regression tests for config/hooks/typing/validation contracts
# ---------------------------------------------------------------------------


def _minimal_ragged_args() -> tuple[Any, ...]:
    q = jnp.zeros((8, 16, 128), jnp.bfloat16)
    cache_nope = jnp.zeros((2, 8, 4, 128), jnp.uint8)
    cache_rope = jnp.zeros((2, 2, 4, 128), jnp.uint8)
    topk = jnp.zeros((8, 16), jnp.int32)
    page_indices = jnp.zeros((4,), jnp.int32)
    cu_q_lens = jnp.array([0, 8], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)
    return (q, cache_nope, cache_rope, topk, page_indices, cu_q_lens, distribution)


def _minimal_chunk_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    q = jnp.zeros((8, 16, 128), jnp.bfloat16)
    cache_nope = jnp.zeros((2, 8, 128), jnp.uint32)
    cache_rope = jnp.zeros((2, 2, 128), jnp.uint32)
    kv_lens = jnp.zeros((8,), jnp.int32)
    phys = jnp.zeros((8, 128), jnp.int32)
    lse = None
    pos = (q, cache_nope, cache_rope, kv_lens, phys, lse)
    kw = {"start_pos": 0, "end_pos": 8, "token_end": jnp.int32(8)}
    return pos, kw


def test_unset_sentinel_is_hashable_and_distinct() -> None:
    assert isinstance(hash(UNSET), int)
    assert {UNSET: "ok"}[UNSET] == "ok"
    assert UNSET is not None


def test_resolve_entry_config_preserves_chunk_vs_ragged_defaults() -> None:
    ragged_cfg = resolve_entry_config(None, {})
    assert ragged_cfg.gather.small_chunk_tokens == 16
    assert ragged_cfg.gather.small_chunk_batch_size == 1
    assert ragged_cfg.gather.small_chunk_live_gather is True
    assert ragged_cfg.gather.live_gather_all_chunks is True

    chunk_cfg = resolve_entry_config(
        None,
        {"sm_scale": 0.25, "k_scale": 0.5, "topk_cap": UNSET},
        defaults=sparse_kernel._ATTENTION_CHUNK_LEGACY_DEFAULTS,
        required=("sm_scale", "k_scale"),
    )
    assert chunk_cfg.sm_scale == 0.25
    assert chunk_cfg.k_scale == 0.5
    assert chunk_cfg.gather.small_chunk_tokens == 0
    assert chunk_cfg.gather.small_chunk_batch_size is None
    assert chunk_cfg.gather.small_chunk_live_gather is False
    assert chunk_cfg.gather.live_gather_all_chunks is False


@pytest.mark.parametrize(
    ("legacy_kwargs", "expected_match"),
    [
        ({"sm_scale": 1.0}, "sm_scale"),
        ({"topk_cap": None, "return_lse": False}, "return_lse, topk_cap"),
        ({"cache_layout": "sparsecore"}, "cache_layout"),
    ],
)
def test_sparse_ragged_paged_attention_rejects_config_plus_legacy(
    legacy_kwargs: dict[str, Any], expected_match: str
) -> None:
    args = _minimal_ragged_args()
    with pytest.raises(
        ValueError,
        match=f"cannot pass both 'config' and legacy config kwargs: {expected_match}",
    ):
        sparse_kernel.sparse_ragged_paged_attention(
            *args, config=SparseMlaConfig(), **legacy_kwargs
        )


@pytest.mark.parametrize(
    ("legacy_kwargs", "expected_match"),
    [
        ({"sm_scale": 0.1}, "sm_scale"),
        ({"small_chunk_tokens": 0, "k_scale": 1.0}, "k_scale, small_chunk_tokens"),
    ],
)
def test_sparse_attention_chunk_rejects_config_plus_legacy(
    legacy_kwargs: dict[str, Any], expected_match: str
) -> None:
    pos, kw = _minimal_chunk_args()
    with pytest.raises(
        ValueError,
        match=f"cannot pass both 'config' and legacy config kwargs: {expected_match}",
    ):
        sparse_kernel.sparse_attention_chunk(
            *pos, **kw, config=SparseMlaConfig(), **legacy_kwargs
        )


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"sm_scale": 0.1},
        {"k_scale": 1.0},
    ],
)
def test_sparse_attention_chunk_requires_scales_without_config(
    extra_kwargs: dict[str, Any],
) -> None:
    pos, kw = _minimal_chunk_args()
    with pytest.raises(TypeError, match="missing required keyword-only argument"):
        sparse_kernel.sparse_attention_chunk(*pos, **kw, **extra_kwargs)


def test_sparse_ragged_paged_attention_rejects_live_prep_without_mixed_fused() -> None:
    args = _minimal_ragged_args()
    q_pe = jnp.zeros((8, 16, 64), jnp.bfloat16)
    hooks = FaultHooks(live_prep=LivePrepFault.KV)
    with pytest.raises(ValueError, match="live_prep fault requires mixed_fused_prep"):
        sparse_kernel.sparse_ragged_paged_attention(
            *args, config=SparseMlaConfig(), hooks=hooks, q_pe=q_pe
        )


def test_sparse_ragged_paged_attention_forwards_hooks_to_live_fused_prep() -> None:
    num_tokens = 128
    topk = 256
    q = jnp.zeros((num_tokens, 16, 128), jnp.bfloat16)
    q_pe = jnp.zeros((num_tokens, 16, 64), jnp.bfloat16)
    cache_nope = jnp.zeros((4, 16, 128), jnp.uint32)
    cache_rope = jnp.zeros((4, 4, 128), jnp.uint32)
    topk_indices = jnp.zeros((num_tokens, topk), jnp.int32)
    page_indices = jnp.zeros((8,), jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)

    cfg = SparseMlaConfig(
        attention_kernel_batch_size=16,
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=128),
        capped=CappedWindowConfig(
            topk_cap=128, mixed_live_gather=True, mixed_fused_prep=True
        ),
    )
    hooks = FaultHooks(live_prep=LivePrepFault.OFF)
    captured: list[FaultHooks] = []

    class _StopAfterPrep(RuntimeError):
        pass

    def _fake_prep(*_a: Any, **kw: Any) -> Any:
        captured.append(kw["hooks"])
        raise _StopAfterPrep()

    with (
        patch.object(
            sparse_kernel, "prepare_capped_window_live_fused", side_effect=_fake_prep
        ),
        pytest.raises(_StopAfterPrep),
    ):
        sparse_kernel.sparse_ragged_paged_attention(
            q,
            cache_nope,
            cache_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            config=cfg,
            hooks=hooks,
            q_pe=q_pe,
        )

    assert captured == [hooks]


def test_bounds_builder_and_registry_accept_int_and_1d_send_peer_array() -> None:
    builder = BoundsBuilder()
    builder.add("start_seq_idx", 0)
    builder.add("end_seq_idx", jnp.int32(8))
    arr, layout = builder.build()
    assert arr.shape == (2,)
    assert arr.dtype == jnp.int32
    assert arr.tolist() == [0, 8]
    assert layout.names == ("start_seq_idx", "end_seq_idx")

    for merge, peer_vals in ((False, [1, 2]), (True, [1, 2, 8])):
        cfg = AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            send_exchange=SendExchangeConfig(axis="dcp", in_kernel_merge=merge),
        )
        sc = _AttentionScenario("array_peer", cfg)
        inputs = _build_scenario_inputs(sc)
        inputs["start_seq_idx"] = 0
        inputs["end_seq_idx"] = 8
        inputs["send_peer"] = jnp.array(peer_vals, jnp.int32)
        _, bounds_layout = build_attention_ref_registry(cfg, **inputs)
        expected = (
            ("start_seq_idx", "end_seq_idx", "peer_rank", "peer_lane", "real_tokens")
            if merge
            else ("start_seq_idx", "end_seq_idx", "peer_rank", "peer_lane")
        )
        assert bounds_layout.names == expected


@pytest.mark.parametrize(
    ("merge", "bad_peer"),
    [
        (False, jnp.int32(0)),
        (False, jnp.zeros((2, 2), jnp.int32)),
        (False, jnp.array([0], jnp.int32)),
        (True, jnp.array([0, 1], jnp.int32)),
    ],
)
def test_build_attention_ref_registry_rejects_bad_send_peer_array(
    merge: bool, bad_peer: jax.Array
) -> None:
    cfg = AttentionKernelConfig(
        sm_scale=0.1,
        batch_size=8,
        sparsecore_layout=True,
        split_q=True,
        q_rope=True,
        return_lse=True,
        lse_lane_dense=True,
        lse_fill=True,
        send_world=4,
        send_exchange=SendExchangeConfig(axis="dcp", in_kernel_merge=merge),
    )
    sc = _AttentionScenario("bad_peer", cfg)
    inputs = _build_scenario_inputs(sc)
    inputs["send_peer"] = bad_peer
    with pytest.raises(ValueError, match="send_peer array must be 1D"):
        build_attention_ref_registry(cfg, **inputs)


def test_rope_q_block_supports_fault_and_hooks_and_rejects_both() -> None:
    class _Tab2D:
        def __getitem__(self, _key: Any) -> jax.Array:
            return jnp.zeros((2, 128), jnp.float32)

    x = jnp.zeros((2, 128), jnp.bfloat16)
    tab: Any = _Tab2D()

    recorded_shifts: list[int] = []

    def _fake_roll(arr: Any, shift: int, _axis: int) -> Any:
        recorded_shifts.append(shift)
        return arr

    def _fake_where(_cond: Any, a: Any, _b: Any) -> Any:
        return a

    with (
        patch.object(sparse_kernel.pltpu, "roll", side_effect=_fake_roll, create=True),
        patch.object(sparse_kernel.jnp, "where", side_effect=_fake_where, create=True),
        patch.object(
            sparse_kernel.lax,
            "broadcasted_iota",
            return_value=jnp.zeros((2, 128), jnp.int32),
        ),
        patch.object(
            type(x),
            "__lt__",
            lambda self, _other: self,
            create=True,
        ),
    ):
        # Default (no fault) uses shifts (96, 32) then 64.
        recorded_shifts.clear()
        sparse_kernel.rope_q_block(x, tab)
        assert recorded_shifts == [96, 32, 64]

        # Positional fault string.
        recorded_shifts.clear()
        sparse_kernel.rope_q_block(x, tab, "rope_swap")
        assert recorded_shifts == [32, 96, 64]

        # Keyword fault string (backward-compatible signature).
        recorded_shifts.clear()
        sparse_kernel.rope_q_block(x, tab, fault="rope_swap")
        assert recorded_shifts == [32, 96, 64]

        # Keyword hooks object.
        swap_hooks = FaultHooks(prologue=PrologueFault.ROPE_SWAP)
        recorded_shifts.clear()
        sparse_kernel.rope_q_block(x, tab, hooks=swap_hooks)
        assert recorded_shifts == [32, 96, 64]

        # Positional FaultHooks object.
        recorded_shifts.clear()
        sparse_kernel.rope_q_block(x, tab, swap_hooks)
        assert recorded_shifts == [32, 96, 64]

    with pytest.raises(ValueError, match="cannot pass both fault and hooks"):
        sparse_kernel.rope_q_block(
            x, tab, fault="rope_swap", hooks=FaultHooks(prologue=PrologueFault.ROPE_F32)
        )

    with pytest.raises(ValueError, match="unknown prologue_fault"):
        sparse_kernel.rope_q_block(x, tab, fault="not_a_fault")


def test_fault_hooks_validate_for_context_and_entry_point_enforcement() -> None:
    xchg_hooks = FaultHooks(prologue=SendExchangeFault.SEND_PEER_DROP)
    with pytest.raises(ValueError, match="unknown prologue_fault send_peer_drop"):
        xchg_hooks.validate_for_context(allow_send_exchange=False)
    xchg_hooks.validate_for_context(allow_send_exchange=True)
    FaultHooks(prologue=PrologueFault.ROPE_SWAP).validate_for_context(
        allow_send_exchange=False
    )
    NO_FAULT_HOOKS.validate_for_context(allow_send_exchange=False)

    args = _minimal_ragged_args()
    with pytest.raises(ValueError, match="unknown prologue_fault send_peer_drop"):
        sparse_kernel.sparse_ragged_paged_attention(
            *args, config=SparseMlaConfig(), hooks=xchg_hooks
        )

    pos, kw = _minimal_chunk_args()
    with pytest.raises(ValueError, match="unknown prologue_fault send_peer_drop"):
        sparse_kernel.sparse_attention_chunk(
            *pos, **kw, sm_scale=0.1, k_scale=1.0, hooks=xchg_hooks
        )


def test_width_sequences_normalized_to_tuples_and_hashable() -> None:
    capped = CappedWindowConfig(
        topk_cap=512,
        mixed_live_widths=[128, 256],  # type: ignore[arg-type]
        decode_row_widths=[128],  # type: ignore[arg-type]
    )
    assert isinstance(capped.mixed_live_widths, tuple)
    assert isinstance(capped.decode_row_widths, tuple)
    assert capped.mixed_live_widths == (128, 256)
    assert capped.decode_row_widths == (128,)
    assert isinstance(hash(capped), int)

    mixed_kernel = AttentionKernelConfig(
        sm_scale=0.1,
        sparsecore_layout=True,
        capped_windows=4,
        has_group_tab=True,
        live_widths=[128, 256],  # type: ignore[arg-type]
    )
    assert isinstance(mixed_kernel.live_widths, tuple)
    assert mixed_kernel.live_widths == (128, 256)
    assert isinstance(hash(mixed_kernel), int)

    decode_kernel = AttentionKernelConfig(
        sm_scale=0.1,
        sparsecore_layout=True,
        capped_windows=4,
        row_widths=[128, 256],  # type: ignore[arg-type]
    )
    assert isinstance(decode_kernel.row_widths, tuple)
    assert decode_kernel.row_widths == (128, 256)
    assert isinstance(hash(decode_kernel), int)


def test_sparse_attention_chunk_topk_cap_zero_raises_value_error() -> None:
    pos, kw = _minimal_chunk_args()
    with pytest.raises(ValueError, match=r"topk_cap \(0\) must be a multiple"):
        sparse_kernel.sparse_attention_chunk(
            *pos,
            **kw,
            sm_scale=0.1,
            k_scale=1.0,
            cache_layout=SPARSECORE_CACHE_LAYOUT,
            topk_cap=0,
        )


# ---------------------------------------------------------------------------
# 6. index_prep.py contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "divisor",
    [1, 2, 8, 64, 128, 256, 3, 5, 7, 13, 17, 100, 127, 255, 1000, 8191],
)
def test_floor_divmod_const_exact_match(divisor: int) -> None:
    values = [
        -1000,
        -divisor - 1,
        -divisor,
        -1,
        0,
        1,
        divisor - 1,
        divisor,
        divisor + 1,
        2 * divisor - 1,
        2 * divisor,
        100_003,
        1_048_575,
        4_194_303,
    ]
    x = jnp.array(values, dtype=jnp.int32)
    q, r = index_prep._floor_divmod_const(x, divisor)
    assert q.dtype == jnp.int32
    assert r.dtype == jnp.int32
    assert q.tolist() == [v // divisor for v in values]
    assert r.tolist() == [v % divisor for v in values]


@pytest.mark.parametrize(
    "divisor",
    [1, 2, 64, 128, 256, 3, 5, 7, 17, 100, 127, 255, 1000],
)
def test_cdiv_const_exact_match(divisor: int) -> None:
    values = [
        0,
        1,
        max(0, divisor - 1),
        divisor,
        divisor + 1,
        2 * divisor - 1,
        2 * divisor,
        99_999,
        1_000_000,
    ]
    x = jnp.array(values, dtype=jnp.int32)
    out = index_prep._cdiv_const(x, divisor)
    assert out.dtype == jnp.int32
    assert out.tolist() == [(v + divisor - 1) // divisor for v in values]


def test_token_seq_ids_maps_ragged_and_empty_sequences() -> None:
    cu_q_lens = jnp.array([0, 3, 3, 5, 8], dtype=jnp.int32)
    seq_ids = index_prep._token_seq_ids(cu_q_lens, num_tokens=8)
    assert seq_ids.shape == (8,)
    assert seq_ids.tolist() == [0, 0, 0, 2, 2, 3, 3, 3]

    # Padding slots beyond cu_q_lens[-1] carry the final sequence id.
    padded_ids = index_prep._token_seq_ids(cu_q_lens, num_tokens=10)
    assert padded_ids.shape == (10,)
    assert padded_ids.tolist() == [0, 0, 0, 2, 2, 3, 3, 3, 3, 3]


@pytest.mark.parametrize("pages_per_seq", [5, 128, 130])
def test_build_token_page_windows_padding_and_lookup(pages_per_seq: int) -> None:
    num_seqs = 3
    cu_q_lens = jnp.array([0, 2, 3, 5], dtype=jnp.int32)
    num_tokens = 5
    page_rows = [
        [seq * 1000 + p + 1 for p in range(pages_per_seq)] for seq in range(num_seqs)
    ]
    flat_pages = jnp.array(
        [entry for row in page_rows for entry in row], dtype=jnp.int32
    )

    windows, returned_pps = index_prep._build_token_page_windows(
        flat_pages, cu_q_lens, num_tokens
    )
    expected_padded = align_to(pages_per_seq, index_prep._GATHER_PAGE_CHUNK)
    assert returned_pps == pages_per_seq
    assert windows.shape == (num_tokens, expected_padded)

    expected_seq_for_token = [0, 0, 1, 2, 2]
    windows_list = windows.tolist()
    for tok, seq in enumerate(expected_seq_for_token):
        assert windows_list[tok][:pages_per_seq] == page_rows[seq]
        assert all(v == 0 for v in windows_list[tok][pages_per_seq:])

    # Verify chunked physical page lookup recovers exact pages per token.
    last_page = pages_per_seq - 1
    mid_page = pages_per_seq // 2
    logical = jnp.array(
        [[0, mid_page, last_page] for _ in range(num_tokens)], dtype=jnp.int32
    )
    looked_up = index_prep._lookup_physical_page(logical, windows)
    expected_lookup = [
        [page_rows[seq][0], page_rows[seq][mid_page], page_rows[seq][last_page]]
        for seq in expected_seq_for_token
    ]
    assert looked_up.tolist() == expected_lookup


@pytest.mark.parametrize(
    ("total_num_pages", "page_size"),
    [
        (16, 64),
        (37, 256),
        (100, 96),
        (513, 100),
    ],
)
def test_scatter_padding_slots_determinism_and_bounds(
    total_num_pages: int, page_size: int
) -> None:
    flat = jnp.arange(64, dtype=jnp.int32).reshape(8, 8)
    page_a, off_a = index_prep.scatter_padding_slots(flat, total_num_pages, page_size)
    page_b, off_b = index_prep.scatter_padding_slots(flat, total_num_pages, page_size)

    assert page_a.shape == (8, 8)
    assert off_a.shape == (8, 8)
    assert page_a.dtype == jnp.int32
    assert off_a.dtype == jnp.int32
    assert page_a.tolist() == page_b.tolist()
    assert off_a.tolist() == off_b.tolist()

    flat_pages = [p for row in page_a.tolist() for p in row]
    flat_offs = [o for row in off_a.tolist() for o in row]
    assert all(0 <= p < total_num_pages for p in flat_pages)
    assert all(0 <= o < page_size for o in flat_offs)
    assert len(set(flat_pages)) > 1
    assert len(set(flat_offs)) > 1


def test_live_group_table_counts_offsets_and_chunk_bounds() -> None:
    # 3 chunks of 8 tokens, batch size 4 -> 6 groups (2 per chunk), atoms_w = 4.
    kv_lens = jnp.array(
        [
            # Chunk 0 (chunk_windows=1): g0 max=70 -> 2 atoms, g1 max=128 -> 2 atoms.
            10,
            70,
            0,
            64,
            128,
            100,
            1,
            65,
            # Chunk 1 (chunk_windows=2): g2 max=400 -> 4 atoms, g3 max=50 -> 1 atom.
            400,
            10,
            20,
            30,
            50,
            0,
            0,
            1,
            # Chunk 2 (past token_end=16): ungridded -> 0 live atoms despite values.
            999,
            999,
            999,
            999,
            999,
            999,
            999,
            999,
        ],
        dtype=jnp.int32,
    )
    tab, num_live = index_prep.live_group_table(
        kv_lens,
        token_end=jnp.int32(16),
        chunk_size=8,
        kernel_batch_size=4,
        topk_cap=256,
        num_windows=4,
    )
    assert tab.shape == (12,)
    assert tab.dtype == jnp.int32
    assert num_live.shape == (3,)
    assert num_live.dtype == jnp.int32
    # live_atoms = [2, 2, 4, 1, 0, 0], per-chunk exclusive offsets = [0, 2, 0, 4, 0, 0]
    assert tab.tolist() == [2, 2, 4, 1, 0, 0, 0, 2, 0, 4, 0, 0]
    # Chunk 0 (1 window): (2 + 2) * (4 * 64) = 1024.
    # Chunk 1 (2 windows, grid_rows=8): (2 - 1) * 8 * 256 + 8 * 256 = 4096.
    # Chunk 2 (ungridded, grid_rows=0, 1 window): 0.
    assert num_live.tolist() == [1024, 4096, 0]


@pytest.mark.parametrize(
    (
        "num_tokens",
        "chunk_size",
        "batch_size",
        "token_end",
        "kv_lens_vals",
        "topk_cap",
        "num_windows",
        "expected",
    ),
    [
        # Aligned: all zeros clamps to 1.
        (8, 8, 4, 8, [0] * 8, 128, 4, 1),
        # Aligned: live token in chunk 1 needs 3 windows.
        (
            16,
            8,
            4,
            12,
            [64] * 8 + [300, 10, 10, 10, 9999, 9999, 9999, 9999],
            128,
            4,
            3,
        ),
        # Aligned: huge token past grid_rows is ignored; clipped to num_windows.
        (8, 8, 4, 3, [50, 60, 70, 80, 9999, 9999, 9999, 9999], 128, 4, 1),
        (8, 8, 4, 5, [50, 60, 70, 80, 9999, 0, 0, 0], 128, 4, 4),
        # Non-multiple fallback (12 % 8 != 0): live token in partial tail chunk.
        (12, 8, 4, 10, [64] * 8 + [260, 10, 10, 10], 128, 4, 3),
        # Non-multiple fallback: huge padding token past grid_rows is ignored.
        (12, 8, 4, 8, [64] * 8 + [9999, 9999, 9999, 9999], 128, 4, 1),
        # Non-multiple fallback: huge live token clips to num_windows.
        (12, 8, 4, 9, [10] * 8 + [9999, 0, 0, 0], 128, 3, 3),
    ],
)
def test_capped_live_windows_aligned_and_non_multiple_fallback(
    num_tokens: int,
    chunk_size: int,
    batch_size: int,
    token_end: int,
    kv_lens_vals: list[int],
    topk_cap: int,
    num_windows: int,
    expected: int,
) -> None:
    kv_lens = jnp.array(kv_lens_vals, dtype=jnp.int32)
    assert kv_lens.shape == (num_tokens,)
    out = index_prep.capped_live_windows(
        kv_lens,
        jnp.int32(token_end),
        chunk_size=chunk_size,
        kernel_batch_size=batch_size,
        topk_cap=topk_cap,
        num_windows=num_windows,
    )
    assert out.dtype == jnp.int32
    assert int(out) == expected


# ---------------------------------------------------------------------------
# 7. gather_dispatch.py contract tests
# ---------------------------------------------------------------------------


def test_prepared_indices_chunk_accessors() -> None:
    kv = jnp.zeros((4,), jnp.int32)
    none_prep = PreparedIndices(
        kv_lens=kv,
        physical_indices=None,
        window_indices_all=None,
        chunk_window_rows=0,
        fused_prep=False,
        group_tab=None,
        group_num_live=None,
    )
    assert none_prep.window_indices_for_chunk(0) is None
    assert none_prep.group_num_live_for_chunk(0) is None

    whole = jnp.arange(12, dtype=jnp.int32).reshape(6, 2)
    fused_prep = PreparedIndices(
        kv_lens=kv,
        physical_indices=None,
        window_indices_all=whole,
        chunk_window_rows=3,
        fused_prep=True,
        group_tab=None,
        group_num_live=None,
    )
    assert fused_prep.window_indices_for_chunk(0) is whole
    assert fused_prep.window_indices_for_chunk(1) is whole

    c0 = jnp.zeros((2, 2), jnp.int32)
    c1 = jnp.full((2, 2), 7, jnp.int32)
    live_arr = jnp.array([1024, 2048], dtype=jnp.int32)
    tuple_prep = PreparedIndices(
        kv_lens=kv,
        physical_indices=None,
        window_indices_all=(c0, c1),
        chunk_window_rows=2,
        fused_prep=False,
        group_tab=jnp.zeros((4,), jnp.int32),
        group_num_live=live_arr,
    )
    assert tuple_prep.window_indices_for_chunk(0) is c0
    assert tuple_prep.window_indices_for_chunk(1) is c1
    live0 = tuple_prep.group_num_live_for_chunk(0)
    live1 = tuple_prep.group_num_live_for_chunk(1)
    assert live0 is not None and int(live0) == 1024
    assert live1 is not None and int(live1) == 2048

    sliced_prep = PreparedIndices(
        kv_lens=kv,
        physical_indices=None,
        window_indices_all=whole,
        chunk_window_rows=3,
        fused_prep=False,
        group_tab=None,
        group_num_live=None,
    )
    chunk0 = sliced_prep.window_indices_for_chunk(0)
    chunk1 = sliced_prep.window_indices_for_chunk(1)
    assert chunk0 is not None and chunk0.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert chunk1 is not None and chunk1.tolist() == [[6, 7], [8, 9], [10, 11]]


def test_prepare_attention_indices_fused_decode_branch() -> None:
    num_tokens = 16
    topk = 256
    # First 8 tokens have 2 live entries, last 8 are DCP padding with 1 live entry.
    rows = [[0, 1] + [-1] * (topk - 2)] * 8 + [[0] + [-1] * (topk - 1)] * 8
    topk_indices = jnp.array(rows, dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 8], dtype=jnp.int32)
    token_end = jnp.int32(8)

    capped = CappedWindowConfig(topk_cap=128, fused_window_prep=True)
    cfg = SparseMlaConfig(
        attention_kernel_batch_size=8,
        gather=GatherConfig(
            cache_layout=CacheLayout.SPARSECORE,
            chunk_size=256,
            small_chunk_tokens=16,
            small_chunk_batch_size=4,
            small_chunk_live_gather=True,
            small_chunk_gather_atoms=4,
            sparsecore_atoms_per_batch=16,
        ),
        capped=capped,
        lse=LseConfig(mask_empty=True),
        dcp=DcpConfig(is_dcp_mode=True),
    )
    sentinel_win = jnp.zeros((33, 128), dtype=jnp.int32)
    recorded_kw: dict[str, Any] = {}

    def _fake_fused(*_args: Any, **kw: Any) -> jax.Array:
        recorded_kw.update(kw)
        return sentinel_win

    with patch.object(
        index_prep, "prepare_capped_window_fused", side_effect=_fake_fused
    ):
        prepared = prepare_attention_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            token_end,
            num_tokens=num_tokens,
            page_size=64,
            total_num_pages=4,
            config=cfg,
            capped=capped,
            has_kv_write=True,
        )

    assert prepared.fused_prep is True
    assert prepared.window_indices_all is sentinel_win
    assert prepared.physical_indices is None
    assert prepared.group_tab is None
    assert prepared.group_num_live is None
    assert prepared.chunk_window_rows == 0
    assert prepared.kv_lens.tolist() == [2] * 8 + [0] * 8
    assert recorded_kw["kernel_batch_size"] == 4
    assert recorded_kw["gather_atoms"] == 4


def test_prepare_attention_indices_mixed_live_fused_and_unfused() -> None:
    num_tokens = 128
    topk = 256
    topk_indices = jnp.zeros((num_tokens, topk), dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    token_end = jnp.int32(num_tokens)

    # 1) Mixed live-group fused + spy interception via prepare_live_fused_fn.
    fused_capped = CappedWindowConfig(
        topk_cap=128, mixed_live_gather=True, mixed_fused_prep=True
    )
    fused_cfg = SparseMlaConfig(
        attention_kernel_batch_size=16,
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=128),
        capped=fused_capped,
    )
    hooks = FaultHooks(live_prep=LivePrepFault.KV)
    spy_rows = (jnp.zeros((256, 128), dtype=jnp.int32),)
    spy_kv = jnp.full((num_tokens,), 128, dtype=jnp.int32)
    spy_tab = jnp.zeros((16,), dtype=jnp.int32)
    spy_live = jnp.array([4096], dtype=jnp.int32)
    spy_calls: list[dict[str, Any]] = []

    def _spy(*_a: Any, **kw: Any) -> Any:
        spy_calls.append(kw)
        return spy_rows, spy_kv, spy_tab, spy_live

    prep_fused = prepare_attention_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        token_end,
        num_tokens=num_tokens,
        page_size=64,
        total_num_pages=4,
        config=fused_cfg,
        capped=fused_capped,
        hooks=hooks,
        prepare_live_fused_fn=_spy,
    )
    assert len(spy_calls) == 1
    assert spy_calls[0]["hooks"] == hooks
    assert prep_fused.fused_prep is False
    assert prep_fused.window_indices_all is spy_rows
    assert prep_fused.kv_lens is spy_kv
    assert prep_fused.group_tab is spy_tab
    assert prep_fused.group_num_live is spy_live
    assert prep_fused.chunk_window_rows == topk * num_tokens // 128

    # 2) Mixed live-group unfused (is_dcp_mode=True must NOT zero kv_lens padding).
    unfused_capped = CappedWindowConfig(
        topk_cap=128, mixed_live_gather=True, mixed_fused_prep=False
    )
    unfused_cfg = SparseMlaConfig(
        attention_kernel_batch_size=16,
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=128),
        capped=unfused_capped,
        dcp=DcpConfig(is_dcp_mode=True),
    )
    sentinel_live = jnp.zeros((256, 128), dtype=jnp.int32)
    with patch.object(
        index_prep, "prepare_capped_window_live", return_value=sentinel_live
    ) as mock_live:
        prep_unfused = prepare_attention_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            jnp.int32(64),
            num_tokens=num_tokens,
            page_size=64,
            total_num_pages=4,
            config=unfused_cfg,
            capped=unfused_capped,
        )
    assert mock_live.call_count == 1
    assert prep_unfused.window_indices_all is sentinel_live
    assert prep_unfused.group_tab is not None
    assert prep_unfused.group_num_live is not None
    assert prep_unfused.kv_lens.tolist() == [topk] * num_tokens


def test_prepare_attention_indices_live_fused_legacy_signature_compat() -> None:
    num_tokens = 128
    topk = 256
    topk_indices = jnp.zeros((num_tokens, topk), dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    token_end = jnp.int32(num_tokens)

    fused_capped = CappedWindowConfig(
        topk_cap=128, mixed_live_gather=True, mixed_fused_prep=True
    )
    fused_cfg = SparseMlaConfig(
        attention_kernel_batch_size=16,
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=128),
        capped=fused_capped,
    )
    spy_rows = (jnp.zeros((256, 128), dtype=jnp.int32),)
    spy_kv = jnp.full((num_tokens,), 128, dtype=jnp.int32)
    spy_tab = jnp.zeros((16,), dtype=jnp.int32)
    spy_live = jnp.array([4096], dtype=jnp.int32)
    legacy_calls: list[int] = []

    # Legacy stub without a ``hooks`` parameter must succeed under NO_FAULT_HOOKS.
    def _legacy_spy(
        _topk: Any,
        _pages: Any,
        _cu: Any,
        _end: Any,
        *,
        chunk_size: int,
        kernel_batch_size: int,
        topk_cap: int,
        page_size: int,
        total_num_pages: int,
    ) -> Any:
        del chunk_size, kernel_batch_size, topk_cap, page_size, total_num_pages
        legacy_calls.append(1)
        return spy_rows, spy_kv, spy_tab, spy_live

    prep_default = prepare_attention_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        token_end,
        num_tokens=num_tokens,
        page_size=64,
        total_num_pages=4,
        config=fused_cfg,
        capped=fused_capped,
        hooks=NO_FAULT_HOOKS,
        prepare_live_fused_fn=_legacy_spy,
    )
    assert legacy_calls == [1]
    assert prep_default.window_indices_all is spy_rows

    # With an active live_prep fault, ``hooks`` is forwarded explicitly.
    received_hooks: list[FaultHooks] = []
    fault_hooks = FaultHooks(live_prep=LivePrepFault.KV)

    def _hooks_spy(*_a: Any, hooks: FaultHooks, **_kw: Any) -> Any:
        received_hooks.append(hooks)
        return spy_rows, spy_kv, spy_tab, spy_live

    prepare_attention_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        token_end,
        num_tokens=num_tokens,
        page_size=64,
        total_num_pages=4,
        config=fused_cfg,
        capped=fused_capped,
        hooks=fault_hooks,
        prepare_live_fused_fn=_hooks_spy,
    )
    assert received_hooks == [fault_hooks]


def test_prepare_attention_indices_plain_capped_and_uncapped_fallback() -> None:
    num_tokens = 128
    topk = 256
    topk_indices = jnp.zeros((num_tokens, topk), dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    token_end = jnp.int32(num_tokens)

    # Plain capped window (mixed_live_gather=False, fused_window_prep=False).
    plain_capped = CappedWindowConfig(
        topk_cap=128, fused_window_prep=False, mixed_live_gather=False
    )
    plain_cfg = SparseMlaConfig(
        attention_kernel_batch_size=16,
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=128),
        capped=plain_capped,
    )
    sentinel_win = jnp.zeros((256, 128), dtype=jnp.int32)
    with patch.object(
        index_prep, "prepare_capped_window_indices", return_value=sentinel_win
    ) as mock_plain:
        prep_plain = prepare_attention_indices(
            topk_indices,
            page_indices,
            cu_q_lens,
            token_end,
            num_tokens=num_tokens,
            page_size=64,
            total_num_pages=4,
            config=plain_cfg,
            capped=plain_capped,
        )
    assert mock_plain.call_count == 1
    assert prep_plain.window_indices_all is sentinel_win
    assert prep_plain.physical_indices is None
    assert prep_plain.group_tab is None
    assert prep_plain.chunk_window_rows == (topk // 128) * num_tokens

    # Uncapped fallback (topk_cap=None and page_indices=None).
    uncapped = CappedWindowConfig(topk_cap=None)
    uncapped_cfg = SparseMlaConfig(
        gather=GatherConfig(cache_layout=CacheLayout.TENSORCORE, chunk_size=128),
        capped=uncapped,
    )
    sentinel_kv = jnp.zeros((num_tokens,), dtype=jnp.int32)
    sentinel_phys = jnp.zeros((num_tokens, topk), dtype=jnp.int32)
    with patch.object(
        index_prep,
        "prepare_gather_indices",
        return_value=(sentinel_kv, sentinel_phys),
    ) as mock_fallback:
        prep_fb = prepare_attention_indices(
            topk_indices,
            None,
            cu_q_lens,
            token_end,
            num_tokens=num_tokens,
            page_size=64,
            total_num_pages=4,
            config=uncapped_cfg,
            capped=uncapped,
        )
    assert mock_fallback.call_count == 1
    assert prep_fb.window_indices_all is None
    assert prep_fb.physical_indices is sentinel_phys
    assert prep_fb.kv_lens is sentinel_kv
    assert prep_fb.chunk_window_rows == 0


def test_prepare_attention_indices_fail_fast_checks() -> None:
    cu_q_lens = jnp.array([0, 8], dtype=jnp.int32)
    page_indices = jnp.zeros((4,), dtype=jnp.int32)

    # 1) fused_window_prep without lse_mask_empty.
    topk_small = jnp.zeros((16, 256), dtype=jnp.int32)
    capped_fused = CappedWindowConfig(topk_cap=128, fused_window_prep=True)
    cfg_no_mask = SparseMlaConfig(
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=256),
        capped=capped_fused,
        lse=LseConfig(mask_empty=False),
    )
    with pytest.raises(ValueError, match="fused_window_prep requires lse_mask_empty"):
        prepare_attention_indices(
            topk_small,
            page_indices,
            cu_q_lens,
            jnp.int32(8),
            num_tokens=16,
            page_size=64,
            total_num_pages=4,
            config=cfg_no_mask,
            capped=capped_fused,
        )

    # 2) kv_write without fused_prep.
    capped_unfused = CappedWindowConfig(topk_cap=128, fused_window_prep=False)
    cfg_unfused = SparseMlaConfig(
        gather=GatherConfig(cache_layout=CacheLayout.SPARSECORE, chunk_size=256),
        capped=capped_unfused,
        lse=LseConfig(mask_empty=True),
    )
    with pytest.raises(ValueError, match="kv_write needs the fused-prep decode path"):
        prepare_attention_indices(
            topk_small,
            page_indices,
            cu_q_lens,
            jnp.int32(8),
            num_tokens=16,
            page_size=64,
            total_num_pages=4,
            config=cfg_unfused,
            capped=capped_unfused,
            has_kv_write=True,
        )

    # 3) live_prep fault without mixed_fused_prep.
    with pytest.raises(ValueError, match="live_prep fault requires mixed_fused_prep"):
        prepare_attention_indices(
            topk_small,
            page_indices,
            cu_q_lens,
            jnp.int32(8),
            num_tokens=16,
            page_size=64,
            total_num_pages=4,
            config=cfg_unfused,
            capped=capped_unfused,
            hooks=FaultHooks(live_prep=LivePrepFault.OFF),
        )


@pytest.mark.parametrize(
    ("sparsecore_layout", "topk", "atoms", "expected_window", "expected_subchunks"),
    [
        (True, 256, 2, 128, 16),
        (False, 512, 4, 256, 16),
    ],
)
def test_apply_dcp_skip_windows_marks_dead_windows_only(
    sparsecore_layout: bool,
    topk: int,
    atoms: int,
    expected_window: int,
    expected_subchunks: int,
) -> None:
    # 4 tokens: empty (0), 1 live, exact 1-window boundary, full topk.
    kv_lens = jnp.array([0, 1, expected_window, topk], dtype=jnp.int32)
    base = jnp.arange(4 * topk, dtype=jnp.int32).reshape(4, topk)

    out, subchunks, dcp_atoms = apply_dcp_skip_windows(
        base,
        kv_lens,
        topk=topk,
        sparsecore_layout=sparsecore_layout,
        sparsecore_atoms_per_batch=atoms,
    )
    assert out is not None
    assert subchunks == expected_subchunks
    assert dcp_atoms == atoms
    skip = dsa_gather.SKIP_INDEX
    rows = out.tolist()
    base_rows = base.tolist()
    # Token 0 (kv_len=0): every window dead -> all SKIP_INDEX.
    assert all(v == skip for v in rows[0])
    # Token 1 (kv_len=1): window 0 live, remaining windows SKIP_INDEX.
    assert rows[1][:expected_window] == base_rows[1][:expected_window]
    assert all(v == skip for v in rows[1][expected_window:])
    # Token 2 (kv_len=expected_window): window 0 live, window 1+ SKIP_INDEX.
    assert rows[2][:expected_window] == base_rows[2][:expected_window]
    assert all(v == skip for v in rows[2][expected_window:])
    # Token 3 (kv_len=topk): all windows live.
    assert rows[3] == base_rows[3]

    # None indices and sub-atom topk pass through unchanged.
    none_out, _, _ = apply_dcp_skip_windows(
        None,
        kv_lens,
        topk=topk,
        sparsecore_layout=sparsecore_layout,
        sparsecore_atoms_per_batch=atoms,
    )
    assert none_out is None

    tiny = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    tiny_out, tiny_sc, tiny_atoms = apply_dcp_skip_windows(
        tiny,
        kv_lens,
        topk=4,
        sparsecore_layout=sparsecore_layout,
        sparsecore_atoms_per_batch=atoms,
    )
    assert tiny_out is tiny
    assert tiny_sc == 32
    assert tiny_atoms == atoms


def test_gather_chunk_kv_dispatch_and_validation() -> None:
    cache_nope = jnp.zeros((2, 8, 4, 128), jnp.uint8)
    cache_rope = jnp.zeros((2, 2, 4, 128), jnp.uint8)
    kv_lens = jnp.full((8,), 64, jnp.int32)
    phys = jnp.arange(8 * 64, dtype=jnp.int32).reshape(8, 64)
    tc_cfg = SparseMlaConfig(
        gather=GatherConfig(cache_layout=CacheLayout.TENSORCORE),
        capped=CappedWindowConfig(topk_cap=None),
    )

    # 1) group_tab without topk_cap raises ValueError.
    with pytest.raises(ValueError, match="group_tab requires topk_cap"):
        gather_chunk_kv(
            cache_nope,
            cache_rope,
            kv_lens,
            phys,
            start_pos=0,
            end_pos=8,
            topk=64,
            kernel_batch_size=4,
            small_chunk=False,
            batch_end=jnp.int32(8),
            config=tc_cfg,
            group_tab=jnp.zeros((4,), jnp.int32),
        )

    # 2) pre_gathered passthrough and kv_write guard.
    pre_nope = jnp.zeros((8, 64, 128), jnp.uint32)
    pre_rope = jnp.zeros((8, 16, 128), jnp.uint32)
    pre_views = (jnp.zeros((2, 4, 64, 128), jnp.uint32), pre_rope)
    passthrough = gather_chunk_kv(
        cache_nope,
        cache_rope,
        kv_lens,
        phys,
        start_pos=0,
        end_pos=8,
        topk=64,
        kernel_batch_size=4,
        small_chunk=False,
        batch_end=jnp.int32(8),
        config=tc_cfg,
        pre_gathered=(pre_nope, pre_rope, pre_views),
    )
    assert isinstance(passthrough, GatheredChunk)
    assert passthrough.nope is pre_nope
    assert passthrough.rope is pre_rope
    assert passthrough.live_views is pre_views
    assert passthrough.cache_kv_nope is cache_nope
    assert passthrough.cache_kv_rope is cache_rope

    with pytest.raises(ValueError, match="pre_gathered does not carry kv_write"):
        gather_chunk_kv(
            cache_nope,
            cache_rope,
            kv_lens,
            phys,
            start_pos=0,
            end_pos=8,
            topk=64,
            kernel_batch_size=4,
            small_chunk=False,
            batch_end=jnp.int32(8),
            config=tc_cfg,
            kv_write=jnp.zeros((3, 128), jnp.uint32),
            pre_gathered=(pre_nope, pre_rope, None),
        )

    # 3) TensorCore dispatch via dsa_gather mock (non-DCP and DCP modes).
    def _make_fake_dsa_gather(target: dict[str, Any]) -> Any:
        def _fake_dsa_gather(
            nope_c: jax.Array,
            rope_c: jax.Array,
            flat_idx: jax.Array,
            **kw: Any,
        ) -> tuple[jax.Array, jax.Array]:
            target["args"] = (nope_c, rope_c, flat_idx)
            target["kwargs"] = kw
            return (
                jnp.zeros((8 * 64 * 4, 128), jnp.uint8),
                jnp.zeros((8 * 64, 64), jnp.uint8),
            )

        return _fake_dsa_gather

    for is_dcp in (False, True):
        cfg = SparseMlaConfig(
            gather=GatherConfig(cache_layout=CacheLayout.TENSORCORE),
            capped=CappedWindowConfig(topk_cap=None),
            dcp=DcpConfig(is_dcp_mode=is_dcp),
        )
        captured: dict[str, Any] = {}

        with patch.object(
            dsa_gather, "dsa_gather", side_effect=_make_fake_dsa_gather(captured)
        ):
            gathered = gather_chunk_kv(
                cache_nope,
                cache_rope,
                kv_lens,
                phys,
                start_pos=0,
                end_pos=8,
                topk=64,
                kernel_batch_size=4,
                small_chunk=False,
                batch_end=jnp.int32(8),
                config=cfg,
            )
        assert captured["args"][2].shape == (8 * 64,)
        assert int(captured["kwargs"]["num_valid_indices"]) == 8 * 64
        if is_dcp:
            assert captured["kwargs"]["is_dcp_mode"] is True
            assert "num_row_subchunks" in captured["kwargs"]
        else:
            assert "is_dcp_mode" not in captured["kwargs"]
            assert "num_row_subchunks" not in captured["kwargs"]
        assert gathered.nope.shape == (8, 64 * 4, 128)
        assert gathered.rope.shape == (8, 64, 64)
        assert gathered.live_views is None


# ---------------------------------------------------------------------------
# 8. kernel_context.py contract tests
# ---------------------------------------------------------------------------


class _FakeRefAt:
    """Fake `.at[...]` accessor recording subscript keys on a `_FakeRef`."""

    def __init__(self, ref: _FakeRef) -> None:
        self._ref = ref

    def __getitem__(self, key: Any) -> tuple[str, str, Any]:
        return ("ref_at", self._ref.name, key)


class _FakeRef:
    """Shape/dtype-bearing Pallas ref stand-in for CPU context and DMA tests."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        *,
        values: Any = None,
        name: str = "",
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.values = values
        self.name = name
        self.at = _FakeRefAt(self)

    def __getitem__(self, key: Any) -> Any:
        if self.values is not None:
            return self.values[key]
        return ("ref_get", self.name, key)


def _bind_fake_attention_refs(
    sc: _AttentionScenario,
    overrides: dict[str, Any] | None = None,
) -> AttentionRefs:
    """Builds a real `AttentionRefs` populated with shape-bearing fake refs."""
    inputs = _build_scenario_inputs(sc)
    if overrides:
        inputs.update(overrides)
    registry, bounds_layout = build_attention_ref_registry(sc.cfg, **inputs)
    layout = registry.layout
    call_args = registry.call_args()
    out_shapes = (
        registry.out_shape
        if isinstance(registry.out_shape, list)
        else [registry.out_shape]
    )
    scratch_shapes = registry.build_grid_spec((1,)).scratch_shapes
    raw_specs: list[Any] = [*call_args, *out_shapes, *scratch_shapes]

    raw_refs: list[Any] = []
    for slot in layout.bindings:
        spec = raw_specs[slot.start]
        if slot.name == "bounds":
            raw_refs.append(spec)
        elif slot.name == "kv_lens":
            raw_refs.append(
                _FakeRef(spec.shape, spec.dtype, values=spec, name="kv_lens")
            )
        elif slot.name == "sem_ids":
            raw_refs.append([0, 0])
        elif slot.name == "dma_sems":
            raw_refs.append(_FakeSemaphoreRef())
        else:
            raw_refs.append(_FakeRef(spec.shape, spec.dtype, name=slot.name))

    return AttentionRefs.from_bound(
        layout.bind(*raw_refs), bounds_layout, return_lse=sc.cfg.return_lse
    )


def _eager_when(cond: Any) -> Any:
    """CPU-compatible stand-in for `pl.when(cond)`."""

    def _dec(fn: Any) -> Any:
        if bool(cond):
            return fn()
        return None

    return _dec


@pytest.mark.parametrize(
    ("bkv_sz", "expected"),
    [
        (512, (64, 128, 256, 512)),
        (256, (64, 128, 256)),
        (128, (64, 128)),
        (64, (64,)),
        (192, (192,)),
        (96, (96,)),
        (1024, (128, 256, 512, 1024)),
    ],
)
def test_compute_kv_rungs(bkv_sz: int, expected: tuple[int, ...]) -> None:
    assert compute_kv_rungs(bkv_sz) == expected


def test_double_buffer_state_ping_pong_transitions() -> None:
    assert int(DoubleBufferState.next_slot(jnp.int32(0))) == 1
    assert int(DoubleBufferState.next_slot(jnp.int32(1))) == 0

    sem_ids: list[Any] = [jnp.int32(0), jnp.int32(0)]
    db = DoubleBufferState(sem_ids)
    assert int(db.current_in_slot()) == 0

    nxt = db.next_slot(db.current_in_slot())
    db.flip_in_slot(nxt)
    assert int(db.current_in_slot()) == 1
    assert int(sem_ids[1]) == 0  # Input flip leaves output slot untouched.

    out0 = db.acquire_out_slot()
    out1 = db.acquire_out_slot()
    out2 = db.acquire_out_slot()
    assert [int(out0), int(out1), int(out2)] == [0, 1, 0]
    assert int(sem_ids[1]) == 1
    assert int(db.current_in_slot()) == 1  # Output flip leaves input slot untouched.


def test_kernel_geometry_from_refs_modes() -> None:
    # 1) TensorCore with preserve_suffix=True at program_id=1.
    tc_sc = _AttentionScenario(
        "tc_suffix",
        AttentionKernelConfig(sm_scale=0.1, batch_size=4, preserve_suffix=True),
        num_tokens=8,
        topk_tile=256,
    )
    tc_refs = _bind_fake_attention_refs(tc_sc, {"token_end": jnp.int32(6)})
    with patch.object(pl, "program_id", return_value=1, create=True):
        tc_geom = KernelGeometry.from_refs(tc_refs, tc_sc.cfg)
    assert tc_geom.num_tokens == 8
    assert tc_geom.o_dim == 128
    assert tc_geom.head_dim == 128
    assert tc_geom.bkv_sz == 256
    assert tc_geom.kv_rungs == (64, 128, 256)
    assert tc_geom.nofill is False
    assert tc_geom.prefix_end is not None and int(tc_geom.prefix_end) == 6
    assert int(tc_geom.batch_start_seq_idx) == 4
    assert int(tc_geom.batch_end_seq_idx) == 7

    # 2) SparseCore split-q with q_rope and nofill (single-step vs multi-step).
    sc_body = BodyOptimizationConfig.from_levers(("nofill",), mode="decode")
    sc_single = _AttentionScenario(
        "sc_split_nofill",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=8,
            sparsecore_layout=True,
            capped_windows=4,
            row_widths=(128, 256),
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            body=sc_body,
        ),
        num_tokens=8,
        topk_tile=512,
    )
    with patch.object(pl, "program_id", return_value=0, create=True):
        sc_geom = KernelGeometry.from_refs(
            _bind_fake_attention_refs(sc_single), sc_single.cfg
        )
    assert sc_geom.o_dim == 128
    assert sc_geom.head_dim == align_to(128 + Q_ROPE_DIM, 128)
    assert sc_geom.bkv_sz == 512
    assert sc_geom.kv_rungs == (64, 128, 256, 512)
    assert sc_geom.nofill is True
    assert sc_geom.prefix_end is None

    sc_multi = dataclasses.replace(
        sc_single,
        cfg=dataclasses.replace(sc_single.cfg, batch_size=4),
    )
    with patch.object(pl, "program_id", return_value=0, create=True):
        sc_multi_geom = KernelGeometry.from_refs(
            _bind_fake_attention_refs(sc_multi), sc_multi.cfg
        )
    assert sc_multi_geom.nofill is False

    # 3) Send-layout mode (no fused exchange; nofill forced False by send).
    send_sc = _AttentionScenario(
        "send_only",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            row_widths=(128, 256),
            split_q=True,
            q_rope=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
            send_world=4,
            body=sc_body,
        ),
        num_tokens=8,
        num_q_heads=16,
    )
    with patch.object(pl, "program_id", return_value=0, create=True):
        send_geom = KernelGeometry.from_refs(
            _bind_fake_attention_refs(send_sc), send_sc.cfg
        )
    assert send_geom.send_per == 4
    assert send_geom.send_rows == 32
    assert send_geom.send_lanes == send_layout_lanes(8, 16, 4)
    assert send_geom.nofill is False
    assert send_geom.xchg_rows == 0
    assert send_geom.xchg_block == 0

    # 4) Fused-exchange modes (both in_kernel_merge=False and True).
    for merge in (False, True):
        xchg_sc = _AttentionScenario(
            f"xchg_merge_{merge}",
            AttentionKernelConfig(
                sm_scale=0.1,
                batch_size=8,
                sparsecore_layout=True,
                split_q=True,
                q_rope=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
                send_world=4,
                send_exchange=SendExchangeConfig(
                    axis="dcp", lane_axis="tp", in_kernel_merge=merge
                ),
            ),
            num_tokens=8,
            num_q_heads=16,
        )
        with patch.object(pl, "program_id", return_value=0, create=True):
            xchg_geom = KernelGeometry.from_refs(
                _bind_fake_attention_refs(xchg_sc), xchg_sc.cfg
            )
        assert xchg_geom.send_per == 4
        assert xchg_geom.send_rows == 32
        assert xchg_geom.xchg_rows == SEND_SLAB_ROW0 + 32
        assert xchg_geom.xchg_block == SEND_PAD_ROWS + 32


# ---------------------------------------------------------------------------
# 9. dma.py & dcp_exchange.py contract tests
# ---------------------------------------------------------------------------


def test_fetch_bkv_batch_branch_dispatch() -> None:
    # 1) Live-group -> dma.fetch_live_blocks.
    live_sc = _AttentionScenario(
        "dma_live",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            has_group_tab=True,
        ),
        num_tokens=8,
    )
    live_refs = _bind_fake_attention_refs(live_sc)
    with patch.object(pl, "program_id", return_value=0, create=True):
        live_geom = KernelGeometry.from_refs(live_refs, live_sc.cfg)
    with patch.object(dma, "fetch_live_blocks") as mock_blocks:
        dma.fetch_bkv_batch(live_refs, live_sc.cfg, live_geom, 0, 1, wait=True)
    mock_blocks.assert_called_once_with(
        live_refs, live_sc.cfg, live_geom, 0, 1, wait=True
    )

    # 2) fetch_live_rows_only with rows -> dma.fetch_live_rows.
    lrows_body = BodyOptimizationConfig.from_levers(("lrows",), mode="decode")
    lrows_sc = _AttentionScenario(
        "dma_lrows",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            row_widths=(128, 256),
            body=lrows_body,
        ),
        num_tokens=8,
        topk_tile=512,
    )
    lrows_refs = _bind_fake_attention_refs(lrows_sc)
    with patch.object(pl, "program_id", return_value=0, create=True):
        lrows_geom = KernelGeometry.from_refs(lrows_refs, lrows_sc.cfg)
    with patch.object(dma, "fetch_live_rows") as mock_lrows:
        dma.fetch_bkv_batch(
            lrows_refs, lrows_sc.cfg, lrows_geom, 0, 0, wait=False, rows=128
        )
    mock_lrows.assert_called_once_with(
        lrows_refs, lrows_sc.cfg, lrows_geom, 0, 0, 128, wait=False
    )

    # 3) row_widths / explicit rows -> dma.fetch_bkv_slice.
    slice_sc = _AttentionScenario(
        "dma_slice",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            capped_windows=4,
            is_dcp_mode=True,
            row_widths=(128, 256),
        ),
        num_tokens=8,
        topk_tile=512,
    )
    slice_refs = _bind_fake_attention_refs(slice_sc)
    with patch.object(pl, "program_id", return_value=0, create=True):
        slice_geom = KernelGeometry.from_refs(slice_refs, slice_sc.cfg)
    with patch.object(dma, "fetch_bkv_slice") as mock_slice:
        dma.fetch_bkv_batch(
            slice_refs, slice_sc.cfg, slice_geom, 0, 0, wait=False, rows=256
        )
        dma.fetch_bkv_batch(
            slice_refs, slice_sc.cfg, slice_geom, 0, 1, wait=True, rows=None
        )
    assert mock_slice.call_count == 2
    assert mock_slice.call_args_list[0].args == (
        slice_refs,
        slice_sc.cfg,
        slice_geom,
        0,
        0,
        256,
    )
    assert mock_slice.call_args_list[1].args == (
        slice_refs,
        slice_sc.cfg,
        slice_geom,
        0,
        1,
        512,
    )

    # 4) DCP kv_rungs ladder (bkv_sz=512 -> rungs (64, 128, 256, 512)).
    dcp_sc = _AttentionScenario(
        "dma_dcp_rungs",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=4,
            sparsecore_layout=True,
            is_dcp_mode=True,
        ),
        num_tokens=8,
        topk_tile=512,
    )
    for max_kv, expected_rung in (
        (0, None),
        (1, 64),
        (64, 64),
        (65, 128),
        (128, 128),
        (129, 256),
        (256, 256),
        (300, 512),
        (512, 512),
    ):
        kv_vals = jnp.array([0, 0, max_kv, 0, 0, 0, 0, 0], dtype=jnp.int32)
        dcp_refs = _bind_fake_attention_refs(dcp_sc, {"kv_lens": kv_vals})
        with patch.object(pl, "program_id", return_value=0, create=True):
            dcp_geom = KernelGeometry.from_refs(dcp_refs, dcp_sc.cfg)
        seen_rungs: list[int] = []

        def _rec_slice(*args: Any, out: list[int] = seen_rungs, **_kw: Any) -> None:
            out.append(args[5])

        with (
            patch.object(pl, "when", side_effect=_eager_when, create=True),
            patch.object(dma, "fetch_bkv_slice", side_effect=_rec_slice),
        ):
            dma.fetch_bkv_batch(dcp_refs, dcp_sc.cfg, dcp_geom, 0, 0, wait=False)
        assert seen_rungs == ([] if expected_rung is None else [expected_rung])

    # 5) Non-DCP full tile -> direct dma.async_copy for nope and rope.
    plain_sc = _AttentionScenario(
        "dma_plain",
        AttentionKernelConfig(sm_scale=0.1, batch_size=4, sparsecore_layout=True),
        num_tokens=8,
        topk_tile=128,
    )
    plain_refs = _bind_fake_attention_refs(plain_sc)
    with patch.object(pl, "program_id", return_value=0, create=True):
        plain_geom = KernelGeometry.from_refs(plain_refs, plain_sc.cfg)
    for wait_flag in (False, True):
        with patch.object(dma, "async_copy") as mock_copy:
            dma.fetch_bkv_batch(
                plain_refs, plain_sc.cfg, plain_geom, 0, 1, wait=wait_flag
            )
        assert mock_copy.call_count == 2
        assert mock_copy.call_args_list[0].kwargs == {"wait": wait_flag}
        assert mock_copy.call_args_list[1].kwargs == {"wait": wait_flag}
        assert mock_copy.call_args_list[0].args[2] == ("sem_slot", (0, 1))
        assert mock_copy.call_args_list[1].args[2] == ("sem_slot", (3, 1))


@pytest.mark.parametrize(
    ("kv_vals", "hooks", "expected_width"),
    [
        ([32, 128, 0, 0], NO_FAULT_HOOKS, 128),
        ([64, 129, 0, 0], NO_FAULT_HOOKS, 256),
        ([100, 256, 0, 0], NO_FAULT_HOOKS, 256),
        ([10, 257, 0, 0], NO_FAULT_HOOKS, None),
        (
            [32, 128, 0, 0],
            FaultHooks(prologue=PrologueFault.ROW_WIDTH_UNDERSIZE),
            128,
        ),
        (
            [64, 129, 0, 0],
            FaultHooks(prologue=PrologueFault.ROW_WIDTH_UNDERSIZE),
            128,
        ),
        (
            [100, 256, 0, 0],
            FaultHooks(prologue=PrologueFault.ROW_WIDTH_UNDERSIZE),
            128,
        ),
        (
            [10, 257, 0, 0],
            FaultHooks(prologue=PrologueFault.ROW_WIDTH_UNDERSIZE),
            256,
        ),
    ],
)
def test_with_row_width_ladder_normal_and_undersize_fault(
    kv_vals: list[int],
    hooks: FaultHooks,
    expected_width: int | None,
) -> None:
    sc = _AttentionScenario(
        "row_widths",
        AttentionKernelConfig(
            sm_scale=0.1,
            batch_size=2,
            sparsecore_layout=True,
            capped_windows=4,
            row_widths=(128, 256),
        ),
        num_tokens=4,
    )
    refs = _bind_fake_attention_refs(
        sc, {"kv_lens": jnp.array(kv_vals, dtype=jnp.int32)}
    )
    chosen: list[int | None] = []
    with patch.object(pl, "when", side_effect=_eager_when, create=True):
        dma.with_row_width(refs, sc.cfg, hooks, 0, chosen.append)
    assert chosen == [expected_width]

    # Empty row_widths invokes callback directly with None.
    empty_cfg = dataclasses.replace(sc.cfg, row_widths=())
    direct: list[int | None] = []
    dma.with_row_width(refs, empty_cfg, hooks, 0, direct.append)
    assert direct == [None]


def test_xchg_peer_with_and_without_lane_axis() -> None:
    base_cfg = AttentionKernelConfig(
        sm_scale=0.1,
        batch_size=8,
        sparsecore_layout=True,
        split_q=True,
        q_rope=True,
        return_lse=True,
        lse_lane_dense=True,
        lse_fill=True,
        send_world=4,
        send_exchange=SendExchangeConfig(axis="dcp", lane_axis=None),
    )
    sc_no_lane = _AttentionScenario("xchg_no_lane", base_cfg, num_tokens=8)
    refs_no_lane = _bind_fake_attention_refs(
        sc_no_lane, {"send_peer": (jnp.int32(1), jnp.int32(3))}
    )
    peer_no_lane = dcp_exchange.xchg_peer(refs_no_lane, base_cfg, 2)
    assert list(peer_no_lane.keys()) == ["dcp"]
    assert int(peer_no_lane["dcp"]) == 2

    lane_cfg = dataclasses.replace(
        base_cfg,
        send_exchange=SendExchangeConfig(axis="dcp", lane_axis="tp"),
    )
    sc_with_lane = _AttentionScenario("xchg_with_lane", lane_cfg, num_tokens=8)
    refs_with_lane = _bind_fake_attention_refs(
        sc_with_lane, {"send_peer": (jnp.int32(1), jnp.int32(3))}
    )
    peer_with_lane = dcp_exchange.xchg_peer(refs_with_lane, lane_cfg, 2)
    assert set(peer_with_lane.keys()) == {"dcp", "tp"}
    assert int(peer_with_lane["dcp"]) == 2
    assert int(peer_with_lane["tp"]) == 3


# ---------------------------------------------------------------------------
# 10. attention_body.py & attention_kernel.py contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q_dtype", "k_scale"),
    [
        (jnp.float32, 1.0),
        (jnp.float32, 0.5),
        (jnp.bfloat16, 1.0),
        (jnp.bfloat16, 0.5),
    ],
)
def test_normalize_pv_output_dtypes_and_k_scale(q_dtype: Any, k_scale: float) -> None:
    pv = jnp.array([[2.0, 4.0], [6.0, 8.0]], dtype=jnp.float32)
    l_sum = jnp.array([[2.0], [4.0]], dtype=jnp.float32)

    recip_calls: list[bool] = []
    div_calls: list[int] = []
    orig_div = lax.div

    def _spy_recip(x: Any, approx: bool = False) -> Any:
        recip_calls.append(approx)
        return jnp.full(x.shape, 1.0, jnp.float32) / x

    def _spy_div(a: Any, b: Any) -> Any:
        div_calls.append(1)
        return orig_div(a, b)

    with (
        patch.object(pl, "reciprocal", side_effect=_spy_recip, create=True),
        patch.object(lax, "div", side_effect=_spy_div),
    ):
        out = attention_body.normalize_pv_output(
            pv, l_sum, k_scale=k_scale, q_dtype=q_dtype
        )

    assert out.dtype == q_dtype
    expected = [[1.0 * k_scale, 2.0 * k_scale], [1.5 * k_scale, 2.0 * k_scale]]
    assert out.tolist() == pytest.approx(expected, rel=1e-2)
    if q_dtype == jnp.float32:
        assert len(div_calls) == 1
        assert recip_calls == []
    else:
        assert recip_calls == [True]
        assert div_calls == []


@pytest.mark.parametrize(
    ("strategy", "expected_target"),
    [
        (BodyExecutionStrategy.TOKEN_PIPELINED, "run_token_pipelined"),
        (BodyExecutionStrategy.PAIR, "run_paired_softmax"),
        (BodyExecutionStrategy.QUAD, "run_quad_softmax"),
        (BodyExecutionStrategy.TRANSPOSED_QK, "run_transposed_qk"),
    ],
)
def test_dispatch_attention_body_routes_all_strategies(
    strategy: BodyExecutionStrategy, expected_target: str
) -> None:
    body_cfg = BodyOptimizationConfig(strategy=strategy)
    sentinel_refs: Any = object()
    sentinel_cfg: Any = object()
    sentinel_geom: Any = object()

    with (
        patch.object(attention_body, "run_token_pipelined") as mock_pipe,
        patch.object(attention_body, "run_paired_softmax") as mock_pair,
        patch.object(attention_body, "run_quad_softmax") as mock_quad,
        patch.object(attention_body, "run_transposed_qk") as mock_xqk,
    ):
        attention_body.dispatch_attention_body(
            sentinel_refs,
            sentinel_cfg,
            sentinel_geom,
            NO_FAULT_HOOKS,
            0,
            1,
            128,
            body_cfg,
        )
        mocks = {
            "run_token_pipelined": mock_pipe,
            "run_paired_softmax": mock_pair,
            "run_quad_softmax": mock_quad,
            "run_transposed_qk": mock_xqk,
        }
        for name, mock_fn in mocks.items():
            if name == expected_target:
                mock_fn.assert_called_once_with(
                    sentinel_refs,
                    sentinel_cfg,
                    sentinel_geom,
                    NO_FAULT_HOOKS,
                    0,
                    1,
                    128,
                    body_cfg,
                )
            else:
                mock_fn.assert_not_called()


@pytest.mark.parametrize(
    ("case_name", "cfg", "expected_decode"),
    [
        (
            "decode_lse_fill",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
            ),
            True,
        ),
        (
            "decode_row_widths",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                capped_windows=4,
                row_widths=(128, 256),
            ),
            True,
        ),
        (
            "decode_send_world",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                split_q=True,
                q_rope=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
                send_world=4,
            ),
            True,
        ),
        (
            "decode_send_exchange",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                split_q=True,
                q_rope=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
                send_world=4,
                send_exchange=SendExchangeConfig(axis="dcp", in_kernel_merge=False),
            ),
            True,
        ),
        (
            "decode_xchg_merge",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                split_q=True,
                q_rope=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
                send_world=4,
                send_exchange=SendExchangeConfig(axis="dcp", in_kernel_merge=True),
            ),
            True,
        ),
        (
            "mixed_group_tab_no_lse_fill",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                capped_windows=4,
                has_group_tab=True,
                live_widths=(128, 256),
            ),
            False,
        ),
        (
            "mixed_group_tab_with_lse_fill",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                capped_windows=4,
                has_group_tab=True,
                return_lse=True,
                lse_lane_dense=True,
                lse_fill=True,
            ),
            False,
        ),
        (
            "prefill_preserve_suffix",
            AttentionKernelConfig(sm_scale=0.1, preserve_suffix=True),
            False,
        ),
        (
            "prefill_out_pieces",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                return_lse=True,
                lse_lane_dense=True,
                out_row_offset=8,
                out_in=True,
            ),
            False,
        ),
        (
            "prefill_standard_tc",
            AttentionKernelConfig(sm_scale=0.1),
            False,
        ),
        (
            "prefill_standard_sc",
            AttentionKernelConfig(sm_scale=0.1, sparsecore_layout=True),
            False,
        ),
        (
            "prefill_dcp",
            AttentionKernelConfig(
                sm_scale=0.1,
                sparsecore_layout=True,
                is_dcp_mode=True,
                return_lse=True,
                lse_lane_dense=True,
            ),
            False,
        ),
    ],
)
def test_is_decode_kernel_and_select_attention_kernel_matrix(
    case_name: str, cfg: AttentionKernelConfig, expected_decode: bool
) -> None:
    del case_name
    assert attention_kernel.is_decode_kernel(cfg) is expected_decode
    expected_fn = (
        attention_kernel._decode_attention_kernel
        if expected_decode
        else attention_kernel._mixed_prefill_attention_kernel
    )
    assert attention_kernel.select_attention_kernel(cfg) is expected_fn


@pytest.mark.parametrize(
    "cfg",
    [
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            return_lse=True,
            lse_lane_dense=True,
            lse_fill=True,
        ),
        AttentionKernelConfig(
            sm_scale=0.1,
            sparsecore_layout=True,
            capped_windows=4,
            has_group_tab=True,
        ),
    ],
)
def test_attention_kernel_delegates_to_select_attention_kernel(
    cfg: AttentionKernelConfig,
) -> None:
    recorded: dict[str, Any] = {}

    def _fake_selected(*raw_refs: Any, **kw: Any) -> None:
        recorded["raw_refs"] = raw_refs
        recorded["kwargs"] = kw

    sentinel_layout: Any = object()
    sentinel_bounds: Any = object()
    hooks = FaultHooks(prologue=PrologueFault.ROPE_SWAP)
    with patch.object(
        attention_kernel, "select_attention_kernel", return_value=_fake_selected
    ) as mock_select:
        attention_kernel._attention_kernel(
            "r0",
            "r1",
            cfg=cfg,
            ref_layout=sentinel_layout,
            bounds_layout=sentinel_bounds,
            hooks=hooks,
        )

    mock_select.assert_called_once_with(cfg)
    assert recorded["raw_refs"] == ("r0", "r1")
    assert recorded["kwargs"] == {
        "cfg": cfg,
        "ref_layout": sentinel_layout,
        "bounds_layout": sentinel_bounds,
        "hooks": hooks,
    }
