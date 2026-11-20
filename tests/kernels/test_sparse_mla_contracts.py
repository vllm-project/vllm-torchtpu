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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import ref_registry
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
    send_layout_lanes,
    send_layout_shape,
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
# 4. Regression tests for config, registry, and fault contracts
# ---------------------------------------------------------------------------


def test_unset_sentinel_is_hashable_and_distinct() -> None:
    assert isinstance(hash(UNSET), int)
    assert {UNSET: "ok"}[UNSET] == "ok"
    assert UNSET is not None


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
