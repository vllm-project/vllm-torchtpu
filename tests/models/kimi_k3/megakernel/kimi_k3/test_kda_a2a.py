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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import prefill_megakernel
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.kda_a2a import (
    build_kda_a2a_tables,
    gather_output_projection_shares,
    gather_output_projection_shares_chip_pair,
    gather_output_projection_shares_fp8,
    kda_split_prefill_body,
    kda_split_prefill_scratch_shapes,
    owner_all_to_all_join,
    owner_all_to_all_send,
    owner_all_to_all_send_range,
    owner_output_projection,
    owner_output_projection_chip_pair_pallas,
    owner_output_projection_chip_pair_ref,
    owner_output_projection_pallas,
)

RANKS, STRIPE, COLS, HIDDEN, BLOCKS = 32, 4, 8, 16, 2


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    owner_rows = rng.standard_normal((BLOCKS, RANKS, STRIPE, COLS)).astype(np.float32)
    weight = rng.standard_normal((RANKS * COLS, HIDDEN)).astype(np.float32)
    prefix = (rng.standard_normal((BLOCKS * STRIPE, HIDDEN)) + 5.0).astype(np.float32)
    return owner_rows, weight, prefix


def _reference(owner_rows, weight, prefix, *, token_owner, length, replace):
    out = np.zeros_like(prefix)
    for block in range(BLOCKS):
        for row in range(STRIPE):
            local = block * STRIPE + row
            token = block * RANKS * STRIPE + token_owner * STRIPE + row
            projection = sum(
                owner_rows[block, source, row]
                @ weight[source * COLS : (source + 1) * COLS]
                for source in range(RANKS)
            )
            residual = 0.0 if replace else prefix[local]
            out[local] = residual + (projection if token < length else 0.0)
    return out


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


@pytest.mark.parametrize(
    ("token_owner", "length", "replace"),
    [
        (5, 10_000, False),
        (
            5,
            RANKS * STRIPE + 5 * STRIPE + 2,
            False,
        ),
        (31, 10_000, True),
    ],
)
def test_should_match_per_source_reference_when_assembling_owner_heads(
    token_owner, length, replace
):
    owner_rows, weight, prefix = map(_bf16, _inputs())

    result = owner_output_projection(
        jnp.asarray(owner_rows, jnp.bfloat16),
        jnp.asarray(weight, jnp.bfloat16),
        jnp.asarray(prefix, jnp.bfloat16),
        token_owner=token_owner,
        length=length,
        replace=replace,
    )

    expected = _reference(
        owner_rows,
        weight,
        prefix,
        token_owner=token_owner,
        length=length,
        replace=replace,
    )
    np.testing.assert_allclose(
        np.asarray(result, np.float32), _bf16(expected), rtol=2e-2, atol=2e-1
    )


def test_should_keep_only_residual_when_rows_past_length():
    owner_rows, weight, prefix = map(_bf16, _inputs(1))
    length = RANKS * STRIPE + 3 * STRIPE + 2

    result = np.asarray(
        owner_output_projection(
            jnp.asarray(owner_rows, jnp.bfloat16),
            jnp.asarray(weight, jnp.bfloat16),
            jnp.asarray(prefix, jnp.bfloat16),
            token_owner=3,
            length=length,
            replace=False,
        ),
        np.float32,
    )

    np.testing.assert_array_equal(result[STRIPE + 2 :], _bf16(prefix[STRIPE + 2 :]))
    assert np.abs(result[STRIPE + 1] - prefix[STRIPE + 1]).max() > 1.0


def test_should_break_when_source_row_blocks_are_permuted():
    owner_rows, weight, prefix = map(_bf16, _inputs(2))
    wrong = weight.reshape(RANKS, COLS, HIDDEN)[np.roll(np.arange(RANKS), 1)].reshape(
        RANKS * COLS, HIDDEN
    )

    result = np.asarray(
        owner_output_projection(
            jnp.asarray(owner_rows, jnp.bfloat16),
            jnp.asarray(wrong, jnp.bfloat16),
            jnp.asarray(prefix, jnp.bfloat16),
            token_owner=0,
            length=10_000,
            replace=True,
        ),
        np.float32,
    )

    expected = _reference(
        owner_rows, weight, prefix, token_owner=0, length=10_000, replace=True
    )
    assert np.linalg.norm(result - expected) / np.linalg.norm(expected) > 0.5


def test_should_keep_ring_reduce_scatter_when_env_disabled(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A", False)

    assert KernelConfig().prefill_kda_a2a is False


def test_should_select_all_to_all_when_env_set(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A", True)

    assert KernelConfig().prefill_kda_a2a is True


@pytest.mark.parametrize("layer", [1, 9])
def test_should_match_xla_reference_when_projecting_in_pallas(layer):
    rng = np.random.default_rng(3)
    owner_rows = jnp.asarray(rng.standard_normal((2, 32, 8, 384)), jnp.bfloat16)
    family = jnp.asarray(rng.standard_normal((10, 32 * 384, 3584)) * 0.05, jnp.bfloat16)
    prefix = jnp.asarray(rng.standard_normal((16, 3584)) + 3.0, jnp.bfloat16)
    length = 32 * 8 + 6 * 8 + 5

    result = owner_output_projection_pallas(
        owner_rows,
        family,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        interpret=True,
    )

    expected = owner_output_projection(
        owner_rows,
        family[layer],
        prefix,
        token_owner=6,
        length=length,
        replace=layer % 9 == 0,
    )
    np.testing.assert_allclose(
        np.asarray(result, np.float32),
        np.asarray(expected, np.float32),
        rtol=1e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize(
    ("a2a", "builder_tag"), [(True, "(all-to-all)"), (False, "KDA requires")]
)
def test_should_dispatch_kda_layer_builder_on_flag(monkeypatch, a2a, builder_tag):
    monkeypatch.setattr(k3_config, "_ACTIVE", KernelConfig(prefill_kda_a2a=a2a))
    one_device_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    with pytest.raises(ValueError, match="TP32 mesh") as error:
        prefill_megakernel.make_norm_gather_kda_sp(
            one_device_mesh, seq_len=16384, segments=2
        )

    assert builder_tag in str(error.value)
    assert ("(all-to-all)" in str(error.value)) is a2a


def test_should_replicate_rank_ordered_row_blocks_when_gathering_shares():
    devices = jax.devices()
    ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    shares = np.arange(ranks * 2 * 3 * 4, dtype=np.float32).reshape(ranks, 2, 3, 4)
    gather = jax.jit(
        jax.shard_map(
            functools.partial(gather_output_projection_shares, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"),),
            check_vma=False,
        )
    )

    (replica,) = gather(jnp.asarray(shares))

    expected_full = shares.transpose(1, 0, 2, 3).reshape(2, ranks * 3, 4)
    np.testing.assert_array_equal(
        np.asarray(replica), np.broadcast_to(expected_full, (ranks, 2, ranks * 3, 4))
    )


def _kda_split_inputs(*, seq_len=1024, count=3, fp8_window=False, seed=7):
    rng = np.random.default_rng(seed)
    width = 7168
    hidden_dtype = jnp.float8_e4m3fn if fp8_window else jnp.bfloat16
    hidden = jnp.asarray(rng.standard_normal((seq_len, width)) * 0.25, hidden_dtype)
    norm_in = jnp.asarray(rng.standard_normal((1, width)) * 0.2 + 1.0, jnp.bfloat16)
    gate_w = jnp.asarray(rng.standard_normal((width, 128)) * 0.02, jnp.bfloat16)
    fb_w = jnp.asarray(rng.standard_normal((128, 384)) * 0.05, jnp.bfloat16)
    proj_w = jnp.asarray(rng.standard_normal((width, 1664)) * 0.02, jnp.bfloat16)
    conv_w = jnp.asarray(rng.standard_normal((4, 3, 3, 128)) * 0.1, jnp.float32)
    alog = jnp.asarray(rng.standard_normal((3, 1)) * 0.2 - 1.0, jnp.float32)
    dt = jnp.asarray(rng.standard_normal((3, 128)) * 0.1, jnp.float32)
    kda_norm = jnp.asarray(rng.standard_normal((8, 128)) * 0.2 + 1.0, jnp.float32)
    rec_in = jnp.asarray(rng.standard_normal((count, 3, 128, 128)) * 0.05, jnp.float32)
    hist_in = jnp.asarray(
        rng.standard_normal((count, 3, 3, 3, 128)) * 0.1, jnp.bfloat16
    )
    offsets = [0, 512, seq_len]
    lengths = [420, 320, 0]
    checkpoints = [256, 0, 0]
    segment_desc = jnp.asarray(offsets + lengths + checkpoints, jnp.int32)
    length = jnp.asarray([832], jnp.int32)
    return (
        length,
        segment_desc,
        hidden,
        norm_in,
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        kda_norm,
        rec_in,
        hist_in,
    )


def _run_split_prefill_standalone(
    inputs, *, blocks, tile, fp8_window, fp8_stage_p=False
):
    length, seg, hidden, norm_in, *rest = inputs
    del length
    count = seg.shape[0] // 3
    block_tiles = hidden.shape[0] // (blocks * tile)
    shapes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in rest[:7])
    scratch = kda_split_prefill_scratch_shapes(
        hidden.shape[1],
        tile,
        *shapes,
        hidden_dtype=hidden.dtype,
        fp8_stage_p=fp8_stage_p,
    )
    if fp8_window:
        scratch = (*scratch, pltpu.VMEM((1, hidden.shape[1]), jnp.bfloat16))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(seg_ref, h_ref, norm_in_ref, *args):
        in_refs = args[:9]
        out_refs = args[9:14]
        sc = args[14:]
        input_norm = (norm_in_ref, sc[-1]) if fp8_window else None
        kda_sc = sc[:-1] if fp8_window else sc
        if fp8_stage_p:
            *kda_main, proj_scale_vmem = kda_sc
        else:
            kda_main = kda_sc
            proj_scale_vmem = None
        step = pl.program_id(1)
        phase = step // prefill_megakernel.HALF_STEPS
        ring_step = step % prefill_megakernel.HALF_STEPS
        kda_split_prefill_body(
            h_ref,
            *in_refs,
            *out_refs,
            seg_ref,
            *kda_main,
            tile=tile,
            block_tiles=block_tiles,
            blocks=blocks,
            count=count,
            phase=phase,
            ring_step=ring_step,
            step=step,
            decay_lower_bound=-5.0,
            epsilon=1e-5,
            input_norm=input_norm,
            proj_scale_vmem=proj_scale_vmem,
        )

    out_shapes = (
        jax.ShapeDtypeStruct((hidden.shape[0], 384), jnp.bfloat16),
        jax.ShapeDtypeStruct(rest[7].shape, rest[7].dtype),
        jax.ShapeDtypeStruct(rest[8].shape, rest[8].dtype),
        jax.ShapeDtypeStruct(rest[7].shape, rest[7].dtype),
        jax.ShapeDtypeStruct(rest[8].shape, rest[8].dtype),
    )
    return pl.pallas_call(
        body,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1, prefill_megakernel.HALF_STEPS * (blocks + 1)),
            in_specs=(hbm,) * 11,
            out_specs=(hbm,) * 5,
            scratch_shapes=scratch,
        ),
        interpret=True,
    )(seg, hidden, norm_in, *rest)


def _run_tiled_reference(inputs, *, tile, fp8_window):
    length, seg, hidden, norm_in, *rest = inputs
    count = seg.shape[0] // 3
    shapes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in rest[:7])
    scratch = prefill_megakernel.kda_prefill_tiled_scratch_shapes(
        hidden.shape[1], tile, *shapes, store_out=True, hidden_dtype=hidden.dtype
    )
    if fp8_window:
        scratch = (*scratch, pltpu.VMEM((1, hidden.shape[1]), jnp.bfloat16))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(len_ref, seg_ref, h_ref, norm_in_ref, *args):
        in_refs = args[:9]
        out_refs = args[9:14]
        sc = args[14:]
        input_norm = (norm_in_ref, sc[-1]) if fp8_window else None
        kda_sc = sc[:-1] if fp8_window else sc
        prefill_megakernel.kda_prefill_tiled_body(
            len_ref,
            h_ref,
            *in_refs,
            out_refs[0],
            out_refs[1],
            out_refs[2],
            *kda_sc,
            tile=tile,
            decay_lower_bound=-5.0,
            epsilon=1e-5,
            checkpoint=(None, out_refs[3], out_refs[4]),
            segments=(seg_ref, count),
            input_norm=input_norm,
        )

    out_shapes = (
        jax.ShapeDtypeStruct((hidden.shape[0], 384), jnp.bfloat16),
        jax.ShapeDtypeStruct(rest[7].shape, rest[7].dtype),
        jax.ShapeDtypeStruct(rest[8].shape, rest[8].dtype),
        jax.ShapeDtypeStruct(rest[7].shape, rest[7].dtype),
        jax.ShapeDtypeStruct(rest[8].shape, rest[8].dtype),
    )
    return pl.pallas_call(
        body,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=(hbm,) * 11,
            out_specs=(hbm,) * 5,
            scratch_shapes=scratch,
        ),
        interpret=True,
    )(length, seg, hidden, norm_in, *rest)


@pytest.mark.parametrize("fp8_window", [False, True])
def test_should_match_tiled_kda_across_segments_checkpoints_and_padding_when_split(
    fp8_window,
):
    inputs = _kda_split_inputs(seq_len=1024, count=3, fp8_window=fp8_window)

    got = _run_split_prefill_standalone(
        inputs, blocks=2, tile=256, fp8_window=fp8_window
    )
    expected = _run_tiled_reference(inputs, tile=128, fp8_window=fp8_window)

    for actual, ref in zip(got, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, np.float32),
            np.asarray(ref, np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    att = np.asarray(got[0], np.float32)
    np.testing.assert_array_equal(att[420:512], 0.0)
    np.testing.assert_array_equal(att[832:], 0.0)
    assert np.linalg.norm(att[:420]) > 1.0
    np.testing.assert_array_equal(np.asarray(got[1][2]), np.asarray(inputs[11][2]))
    np.testing.assert_array_equal(np.asarray(got[2][2]), np.asarray(inputs[12][2]))
    np.testing.assert_array_equal(np.asarray(got[3][2]), np.asarray(inputs[11][2]))
    np.testing.assert_array_equal(np.asarray(got[4][2]), np.asarray(inputs[12][2]))
    np.testing.assert_array_equal(np.asarray(got[3][1]), np.asarray(inputs[11][1]))
    np.testing.assert_array_equal(np.asarray(got[4][1]), np.asarray(inputs[12][1]))


def test_should_break_split_kda_parity_when_segment_offset_is_corrupted():
    inputs = _kda_split_inputs(seq_len=1024, count=3, fp8_window=False)
    bad_inputs = (inputs[0], inputs[1].at[0].set(256), *inputs[2:])

    got_bad = _run_split_prefill_standalone(
        bad_inputs, blocks=2, tile=256, fp8_window=False
    )
    expected = _run_tiled_reference(inputs, tile=128, fp8_window=False)

    rel_l2 = np.linalg.norm(
        np.asarray(got_bad[0], np.float32) - np.asarray(expected[0], np.float32)
    ) / np.linalg.norm(np.asarray(expected[0], np.float32))
    assert rel_l2 > 0.5


def test_should_reject_invalid_split_prefill_chunk_count():
    inputs = _kda_split_inputs(seq_len=1024, count=3, fp8_window=False)

    with pytest.raises(ValueError, match="chunks/tile"):
        _run_split_prefill_standalone(inputs, blocks=2, tile=512, fp8_window=False)


def test_should_keep_kda_a2a_fp8_disabled_when_env_unset(monkeypatch):
    monkeypatch.setattr(
        k3_config,
        "DEFAULT_PREFILL_KDA_A2A_FP8",
        k3_config.PRODUCTION_DEFAULTS["DEFAULT_PREFILL_KDA_A2A_FP8"],
    )

    assert KernelConfig().prefill_kda_a2a_fp8 is False


def test_should_enable_kda_a2a_fp8_when_a2a_and_fp8_envs_set(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A", True)
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A_FP8", True)

    cfg = KernelConfig()
    assert cfg.prefill_kda_a2a is True
    assert cfg.prefill_kda_a2a_fp8 is True


def test_should_reject_kda_a2a_fp8_when_kda_a2a_is_disabled(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A", False)
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A_FP8", True)

    with pytest.raises(ValueError, match="requires prefill_kda_a2a"):
        KernelConfig()


def test_should_quantize_and_replicate_rank_ordered_fp8_shares():
    devices = jax.devices()
    ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    rng = np.random.default_rng(11)
    shares = rng.standard_normal((ranks, 2, 6, 16)).astype(np.float32) * 0.25
    shares[-1, :, 0, 3] *= 10.0
    gather_fp8 = jax.jit(
        jax.shard_map(
            functools.partial(gather_output_projection_shares_fp8, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )
    )

    replica_fp8, scale_8 = gather_fp8(jnp.asarray(shares, jnp.bfloat16))

    assert replica_fp8.dtype == jnp.float8_e4m3fn
    assert replica_fp8.shape == (ranks, 2, ranks * 6, 16)
    assert scale_8.dtype == jnp.float32
    assert scale_8.shape == (ranks, 2, 8, 16)
    scale_np = np.asarray(scale_8, np.float32)
    np.testing.assert_array_equal(
        scale_np, np.broadcast_to(scale_np[:1, :, :1, :], scale_np.shape)
    )
    expected_full = _bf16(shares).transpose(1, 0, 2, 3).reshape(2, ranks * 6, 16)
    dequant = np.asarray(replica_fp8[0], np.float32) * scale_np[0, :, :1, :]
    rel_l2 = np.linalg.norm(dequant - expected_full) / np.linalg.norm(expected_full)
    assert rel_l2 < 0.04
    np.testing.assert_array_equal(
        np.asarray(replica_fp8),
        np.broadcast_to(np.asarray(replica_fp8[:1]), replica_fp8.shape),
    )


@pytest.mark.parametrize("layer", [1, 9])
def test_should_match_fp32_reference_when_projecting_fp8_owner_rows_in_pallas(layer):
    rng = np.random.default_rng(13)
    owner_rows = jnp.asarray(rng.standard_normal((2, 32, 8, 384)) * 0.5, jnp.bfloat16)
    family_bf16 = jnp.asarray(
        rng.standard_normal((10, 32 * 384, 3584)) * 0.05, jnp.bfloat16
    )
    prefix = jnp.asarray(rng.standard_normal((16, 3584)) + 3.0, jnp.bfloat16)
    length = 32 * 8 + 6 * 8 + 5

    w_f32 = family_bf16.astype(jnp.float32)
    max_abs = jnp.maximum(jnp.max(jnp.abs(w_f32), axis=1, keepdims=True), 1e-12)
    family_fp8 = (w_f32 * (448.0 / max_abs)).astype(jnp.float8_e4m3fn)
    scales_8 = jnp.broadcast_to(max_abs * (1.0 / 448.0), (10, 8, 3584))

    result = owner_output_projection_pallas(
        owner_rows,
        family_fp8,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        output_projection_scales=scales_8,
        interpret=True,
    )

    expected_bf16 = owner_output_projection(
        owner_rows,
        family_bf16[layer],
        prefix,
        token_owner=6,
        length=length,
        replace=layer % 9 == 0,
    )
    expected_fp8 = owner_output_projection(
        owner_rows,
        family_fp8[layer],
        prefix,
        token_owner=6,
        length=length,
        replace=layer % 9 == 0,
        output_projection_scale=scales_8[layer],
    )
    res_np = np.asarray(result, np.float32)
    ref_bf16_np = np.asarray(expected_bf16, np.float32)
    ref_fp8_np = np.asarray(expected_fp8, np.float32)
    assert np.linalg.norm(res_np - ref_bf16_np) / np.linalg.norm(ref_bf16_np) < 0.06
    assert np.linalg.norm(res_np - ref_fp8_np) / np.linalg.norm(ref_fp8_np) < 0.05
    expected_pad = (
        np.zeros_like(res_np[13:])
        if layer % 9 == 0
        else np.asarray(prefix[13:], np.float32)
    )
    np.testing.assert_array_equal(res_np[13:], expected_pad)

    bad = owner_output_projection_pallas(
        owner_rows,
        family_fp8,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        output_projection_scales=scales_8 * 3.0,
        interpret=True,
    )
    bad_rel = np.linalg.norm(np.asarray(bad, np.float32) - ref_bf16_np)
    assert bad_rel / np.linalg.norm(ref_bf16_np) > 0.5


def test_should_reject_mismatched_owner_projection_fp8_scales():
    owner_rows = jnp.zeros((2, 32, 8, 384), jnp.bfloat16)
    family_bf16 = jnp.zeros((2, 32 * 384, 3584), jnp.bfloat16)
    family_fp8 = jnp.zeros((2, 32 * 384, 3584), jnp.float8_e4m3fn)
    scales_8 = jnp.ones((2, 8, 3584), jnp.float32)
    prefix = jnp.zeros((16, 3584), jnp.bfloat16)

    with pytest.raises(ValueError, match="requires output_projection_scales"):
        owner_output_projection_pallas(
            owner_rows,
            family_fp8,
            prefix,
            layer=0,
            length=100,
            token_owner=0,
            interpret=True,
        )
    with pytest.raises(ValueError, match="only valid for float8_e4m3fn"):
        owner_output_projection_pallas(
            owner_rows,
            family_bf16,
            prefix,
            layer=0,
            length=100,
            token_owner=0,
            output_projection_scales=scales_8,
            interpret=True,
        )
    with pytest.raises(ValueError, match="scales must be"):
        owner_output_projection_pallas(
            owner_rows,
            family_fp8,
            prefix,
            layer=0,
            length=100,
            token_owner=0,
            output_projection_scales=jnp.ones((2, 4, 3584), jnp.float32),
            interpret=True,
        )
    with pytest.raises(ValueError, match="requires output_projection_scale"):
        owner_output_projection(
            owner_rows, family_fp8[0], prefix, token_owner=0, length=100, replace=False
        )
    with pytest.raises(ValueError, match="only valid for float8_e4m3fn"):
        owner_output_projection(
            owner_rows,
            family_bf16[0],
            prefix,
            token_owner=0,
            length=100,
            replace=False,
            output_projection_scale=scales_8[0],
        )


@pytest.mark.parametrize("fp8_window", [False, True])
def test_should_match_tiled_kda_when_split_prefill_uses_fp8_stage_p(fp8_window):
    inputs = _kda_split_inputs(seq_len=1024, count=3, fp8_window=fp8_window, seed=17)

    got = _run_split_prefill_standalone(
        inputs, blocks=2, tile=256, fp8_window=fp8_window, fp8_stage_p=True
    )
    expected = _run_tiled_reference(inputs, tile=128, fp8_window=fp8_window)

    tolerances = (0.08, 0.05, 0.08, 0.05, 0.08)
    for actual, ref, tol in zip(got, expected, tolerances, strict=True):
        a = np.asarray(actual, np.float32)
        r = np.asarray(ref, np.float32)
        assert np.linalg.norm(a - r) / np.linalg.norm(r) < tol

    att = np.asarray(got[0], np.float32)
    np.testing.assert_array_equal(att[420:512], 0.0)
    np.testing.assert_array_equal(att[832:], 0.0)
    assert np.linalg.norm(att[:420]) > 1.0
    np.testing.assert_array_equal(np.asarray(got[1][2]), np.asarray(inputs[11][2]))
    np.testing.assert_array_equal(np.asarray(got[2][2]), np.asarray(inputs[12][2]))
    np.testing.assert_array_equal(np.asarray(got[3][2]), np.asarray(inputs[11][2]))
    np.testing.assert_array_equal(np.asarray(got[4][2]), np.asarray(inputs[12][2]))
    np.testing.assert_array_equal(np.asarray(got[3][1]), np.asarray(inputs[11][1]))
    np.testing.assert_array_equal(np.asarray(got[4][1]), np.asarray(inputs[12][1]))


def test_should_break_fp8_split_kda_when_input_norm_is_perturbed():
    inputs = _kda_split_inputs(seq_len=1024, count=3, fp8_window=True, seed=19)
    perturbed = (*inputs[:3], inputs[3] * 2.5, *inputs[4:])

    got = _run_split_prefill_standalone(
        perturbed, blocks=2, tile=256, fp8_window=True, fp8_stage_p=True
    )
    expected = _run_tiled_reference(inputs, tile=128, fp8_window=True)

    rel_l2 = np.linalg.norm(
        np.asarray(got[0], np.float32) - np.asarray(expected[0], np.float32)
    ) / np.linalg.norm(np.asarray(expected[0], np.float32))
    assert rel_l2 > 0.3


def test_should_reject_unaligned_fp8_stage_p_projection_cols():
    shapes = (
        jax.ShapeDtypeStruct((7168, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((7168, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, 128), jnp.float32),
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )

    with pytest.raises(ValueError, match="256-aligned"):
        kda_split_prefill_scratch_shapes(7168, 256, *shapes, fp8_stage_p=True)


@pytest.mark.parametrize("fp8_a2a", [False, True])
def test_should_replicate_kda_output_projection_in_prefill_load(
    monkeypatch, tmp_path, fp8_a2a
):
    monkeypatch.setattr(
        k3_config,
        "_ACTIVE",
        KernelConfig(prefill_kda_a2a=True, prefill_kda_a2a_fp8=fp8_a2a),
    )
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    assert ("k_o_scale" in prog.names["kda"]) is fp8_a2a
    prog.names = {
        "kda": ("k_o", "k_o_scale") if fp8_a2a else ("k_o",),
        "moe": ("expert_ds",),
    }

    loaded_ko = object()
    shared = {"expert_ds": object()}

    def _fake_stream(iterator, store):
        del iterator
        store("kda", "k_o", loaded_ko)
        return {"gib": 0.1, "wall": 0.1}

    registered_ops = []

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, fn, in_specs
        registered_ops.append((name, len(out_specs)))
        if len(out_specs) == 2:
            return lambda x: ((f"fp8:{name}", x), (f"scale:{name}", x))
        return lambda x: ((f"bf16:{name}", x),)

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(k3_prefill, "materialize", lambda x: None)
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared=shared)

    if fp8_a2a:
        assert registered_ops == [
            ("pallas::k3_prefill_kda_output_projection_replicate_fp8", 2)
        ]
        assert prog.weights["kda"]["k_o"] == (
            "fp8:pallas::k3_prefill_kda_output_projection_replicate_fp8",
            loaded_ko,
        )
        assert prog.weights["kda"]["k_o_scale"] == (
            "scale:pallas::k3_prefill_kda_output_projection_replicate_fp8",
            loaded_ko,
        )
    else:
        assert registered_ops == [
            ("pallas::k3_prefill_kda_output_projection_replicate", 1)
        ]
        assert prog.weights["kda"]["k_o"] == (
            "bf16:pallas::k3_prefill_kda_output_projection_replicate",
            loaded_ko,
        )
        assert "k_o_scale" not in prog.weights["kda"]


def test_should_keep_a2a_chip_shard_off_when_disabled_and_validate_dependencies(
    monkeypatch,
):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_A2A_CHIP_SHARD", False)
    assert KernelConfig().prefill_a2a_chip_shard is False

    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_A2A", True)
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_A2A_CHIP_SHARD", True)
    assert KernelConfig().prefill_a2a_chip_shard is True

    with pytest.raises(
        ValueError, match="prefill_kda_a2a=True or prefill_mla_a2a=True"
    ):
        KernelConfig(
            prefill_kda_a2a=False,
            prefill_mla_a2a=False,
            prefill_a2a_chip_shard=True,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        KernelConfig(
            prefill_kda_a2a=True,
            prefill_kda_a2a_fp8=True,
            prefill_a2a_chip_shard=True,
        )


def test_should_reject_invalid_inputs_for_kda_chip_pair_projection():
    with pytest.raises(ValueError, match="chip-pair KDA projection"):
        owner_output_projection_chip_pair_pallas(
            jnp.zeros((1, 4, 16, 384), jnp.bfloat16),
            jnp.zeros((2, 12288, 3584), jnp.bfloat16),
            jnp.zeros((16, 7168), jnp.bfloat16),
            layer=0,
            length=512,
            own_even=0,
            own_odd=1,
        )

    with pytest.raises(ValueError, match="chip-pair KDA projection"):
        owner_output_projection_chip_pair_pallas(
            jnp.zeros((1, 32, 16, 384), jnp.bfloat16),
            jnp.zeros((2, 12288, 3584), jnp.float8_e4m3fn),
            jnp.zeros((16, 7168), jnp.bfloat16),
            layer=0,
            length=512,
            own_even=0,
            own_odd=1,
        )

    with pytest.raises(ValueError, match="even R"):
        owner_output_projection_chip_pair_ref(
            jnp.zeros((1, 3, 4, 8), jnp.bfloat16),
            jnp.zeros((24, 8), jnp.bfloat16),
            jnp.zeros((4, 16), jnp.bfloat16),
            layer=0,
            length=12,
            own_even=0,
            own_odd=1,
        )


def test_should_shard_kda_output_projection_in_prefill_load_when_chip_shard_enabled(
    monkeypatch, tmp_path
):
    chip_cfg = KernelConfig(prefill_kda_a2a=True, prefill_a2a_chip_shard=True)
    monkeypatch.setattr(k3_config, "active", lambda: chip_cfg)
    monkeypatch.setattr(k3_prefill, "active", lambda: chip_cfg)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 0))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    assert "k_o_scale" not in prog.names["kda"]
    prog.names = {"kda": ("k_o",), "moe": ("expert_ds",)}

    loaded_ko = object()
    shared = {"expert_ds": object()}
    registered_ops = []

    def _fake_stream(iterator, store):
        del iterator
        store("kda", "k_o", loaded_ko)
        return {"gib": 0.1, "wall": 0.1}

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, fn, in_specs
        registered_ops.append((name, len(out_specs)))
        return lambda x: ((f"chip_pair:{name}", x),)

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(k3_prefill, "materialize", lambda x: None)
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared=shared)

    assert registered_ops == [("pallas::k3_prefill_kda_output_projection_chip_pair", 1)]
    assert prog.weights["kda"]["k_o"] == (
        "chip_pair:pallas::k3_prefill_kda_output_projection_chip_pair",
        loaded_ko,
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_should_match_replicated_oracle_bitwise_for_kda_chip_pair_across_ranks(layer):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("chip-pair KDA test requires >= 2 JAX devices")
    even_devs = devices[: len(devices) - (len(devices) % 2)]
    num_ranks = len(even_devs)
    mesh = Mesh(np.asarray(even_devs), ("k3_sp",))
    rng = np.random.default_rng(410 + layer)
    blocks, stripe, cols, hidden, slots = 2, 4, 8, 16, 2
    rows = blocks * stripe
    length = blocks * num_ranks * stripe - 5
    owners = np.roll(np.arange(num_ranks, dtype=np.int32), 1)

    shares_np = (rng.standard_normal((num_ranks, slots, cols, hidden)) * 0.2).astype(
        np.float32
    )
    owner_rows_np = (
        rng.standard_normal((num_ranks, blocks, num_ranks, stripe, cols)) * 0.3
    ).astype(np.float32)
    prefix_np = (rng.standard_normal((num_ranks, rows, hidden)) + 3.0).astype(
        np.float32
    )

    shares = jax.device_put(
        jnp.asarray(shares_np, jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P("k3_sp")),
    )
    owner_rows = jax.device_put(
        jnp.asarray(owner_rows_np, jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P("k3_sp")),
    )
    prefix = jax.device_put(
        jnp.asarray(prefix_np, jnp.bfloat16),
        jax.sharding.NamedSharding(mesh, P("k3_sp")),
    )
    owners_dev = jax.device_put(
        jnp.asarray(owners), jax.sharding.NamedSharding(mesh, P())
    )

    (ko_half,) = jax.shard_map(
        functools.partial(gather_output_projection_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(shares)
    assert ko_half.shape == (num_ranks, slots, num_ranks * cols, hidden // 2)

    def _run_pair(or_s, ko_s, px_s, ow_tbl):
        r = jax.lax.axis_index("k3_sp")
        out = owner_output_projection_chip_pair_ref(
            or_s[0],
            ko_s[0, layer],
            px_s[0],
            layer=layer,
            length=length,
            own_even=ow_tbl[r & ~1],
            own_odd=ow_tbl[r | 1],
        )
        return out[None]

    actual = np.asarray(
        jax.shard_map(
            _run_pair,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
            out_specs=P("k3_sp"),
            check_vma=False,
        )(owner_rows, ko_half, prefix, owners_dev),
        np.float32,
    )

    full_ko_bf16 = np.asarray(
        jnp.asarray(shares_np, jnp.bfloat16)
        .transpose(1, 0, 2, 3)
        .reshape(slots, num_ranks * cols, hidden),
        np.float32,
    )
    or_bf16 = np.asarray(jnp.asarray(owner_rows_np, jnp.bfloat16), np.float32)
    px_bf16 = np.asarray(jnp.asarray(prefix_np, jnp.bfloat16), np.float32)
    expected = np.zeros_like(actual)
    for r in range(num_ranks):
        for b in range(blocks):
            for s in range(stripe):
                loc = b * stripe + s
                tok = b * num_ranks * stripe + int(owners[r]) * stripe + s
                proj = (
                    or_bf16[r, b, :, s, :].reshape(num_ranks * cols)
                    @ full_ko_bf16[layer]
                )
                res = 0.0 if layer % 9 == 0 else px_bf16[r, loc]
                expected[r, loc] = res + (proj if tok < length else 0.0)
    expected_bf16 = np.asarray(jnp.asarray(expected, jnp.bfloat16), np.float32)

    np.testing.assert_array_equal(actual, expected_bf16)

    swapped_owners = jax.device_put(
        jnp.asarray(owners[[i ^ 1 for i in range(num_ranks)]]),
        jax.sharding.NamedSharding(mesh, P()),
    )
    corrupted = np.asarray(
        jax.shard_map(
            _run_pair,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
            out_specs=P("k3_sp"),
            check_vma=False,
        )(owner_rows, ko_half, prefix, swapped_owners),
        np.float32,
    )
    assert np.max(np.abs(corrupted - expected_bf16)) > 0.1


def test_should_keep_one_gather_ring_in_flight_when_dual_gather_env_unset(monkeypatch):
    monkeypatch.setattr(
        k3_config,
        "DEFAULT_PREFILL_KDA_DUAL_GATHER",
        k3_config.PRODUCTION_DEFAULTS["DEFAULT_PREFILL_KDA_DUAL_GATHER"],
    )

    assert KernelConfig().prefill_kda_dual_gather is False


def test_should_select_dual_gather_when_env_set(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_KDA_DUAL_GATHER", True)

    assert KernelConfig().prefill_kda_dual_gather is True


@pytest.mark.parametrize("layer", [0, 2])
def test_should_match_block_major_bitwise_when_projecting_rank_major_owner_rows(layer):
    rng = np.random.default_rng(501 + layer)
    owner_bm_np = (rng.standard_normal((2, 32, 8, 384)) * 0.4).astype(np.float32)
    owner_bm_np[1, :, 5:, :] = 99.0
    owner_bm = jnp.asarray(owner_bm_np, jnp.bfloat16)
    owner_rm = jnp.transpose(owner_bm, (1, 0, 2, 3))
    family_bf16 = jnp.asarray(
        rng.standard_normal((4, 32 * 384, 3584)) * 0.05, jnp.bfloat16
    )
    prefix = jnp.asarray(rng.standard_normal((16, 3584)) + 2.5, jnp.bfloat16)
    length = 32 * 8 + 6 * 8 + 5

    got_xla = owner_output_projection(
        owner_rm,
        family_bf16[layer],
        prefix,
        token_owner=6,
        length=length,
        replace=layer % 9 == 0,
        rank_major=True,
    )
    ref_xla = owner_output_projection(
        owner_bm,
        family_bf16[layer],
        prefix,
        token_owner=6,
        length=length,
        replace=layer % 9 == 0,
        rank_major=False,
    )
    np.testing.assert_array_equal(np.asarray(got_xla), np.asarray(ref_xla))

    got_pallas = owner_output_projection_pallas(
        owner_rm,
        family_bf16,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        interpret=True,
        rank_major=True,
    )
    ref_pallas = owner_output_projection_pallas(
        owner_bm,
        family_bf16,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        interpret=True,
        rank_major=False,
    )
    np.testing.assert_array_equal(np.asarray(got_pallas), np.asarray(ref_pallas))

    w_f32 = family_bf16.astype(jnp.float32)
    max_abs = jnp.maximum(jnp.max(jnp.abs(w_f32), axis=1, keepdims=True), 1e-12)
    family_fp8 = (w_f32 * (448.0 / max_abs)).astype(jnp.float8_e4m3fn)
    scales_8 = jnp.broadcast_to(max_abs * (1.0 / 448.0), (4, 8, 3584))
    got_fp8 = owner_output_projection_pallas(
        owner_rm,
        family_fp8,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        output_projection_scales=scales_8,
        interpret=True,
        rank_major=True,
    )
    ref_fp8 = owner_output_projection_pallas(
        owner_bm,
        family_fp8,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        output_projection_scales=scales_8,
        interpret=True,
        rank_major=False,
    )
    np.testing.assert_array_equal(np.asarray(got_fp8), np.asarray(ref_fp8))

    untransposed_rm = owner_bm.reshape(32, 2, 8, 384)
    corrupted = owner_output_projection_pallas(
        untransposed_rm,
        family_bf16,
        prefix,
        layer=layer,
        length=length,
        token_owner=6,
        interpret=True,
        rank_major=True,
    )
    corrupted_diff = np.abs(
        np.asarray(corrupted, np.float32) - np.asarray(ref_pallas, np.float32)
    )
    assert np.max(corrupted_diff) > 0.2


def test_should_match_chip_pair_bitwise_when_owner_rows_are_rank_major():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("chip-pair rank-major test requires >= 2 JAX devices")
    even_devs = devices[: len(devices) - (len(devices) % 2)]
    num_ranks = len(even_devs)
    mesh = Mesh(np.asarray(even_devs), ("k3_sp",))
    rng = np.random.default_rng(512)
    blocks, stripe, cols, hidden = 2, 4, 8, 16
    rows = blocks * stripe
    length = blocks * num_ranks * stripe - 3
    owners = np.roll(np.arange(num_ranks, dtype=np.int32), 1)

    bm_shape = (num_ranks, blocks, num_ranks, stripe, cols)
    or_bm_np = (rng.standard_normal(bm_shape) * 0.3).astype(np.float32)
    or_rm_np = np.transpose(or_bm_np, (0, 2, 1, 3, 4)).copy()
    ko_shape = (num_ranks // 2, num_ranks * cols, hidden // 2)
    ko_pair = (rng.standard_normal(ko_shape) * 0.2).astype(np.float32)
    ko_np = np.repeat(ko_pair, 2, axis=0)
    px_np = (rng.standard_normal((num_ranks, rows, hidden)) + 2.0).astype(np.float32)

    sh_sp = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    sh_rep = jax.sharding.NamedSharding(mesh, P())
    or_bm = jax.device_put(jnp.asarray(or_bm_np, jnp.bfloat16), sh_sp)
    or_rm = jax.device_put(jnp.asarray(or_rm_np, jnp.bfloat16), sh_sp)
    ko_half = jax.device_put(jnp.asarray(ko_np, jnp.bfloat16), sh_sp)
    px = jax.device_put(jnp.asarray(px_np, jnp.bfloat16), sh_sp)
    owners_dev = jax.device_put(jnp.asarray(owners), sh_rep)

    def _make_runner(rm: bool):
        def _fn(o_s, w_s, p_s, ow_tbl):
            r = jax.lax.axis_index("k3_sp")
            out = owner_output_projection_chip_pair_ref(
                o_s[0],
                w_s[0],
                p_s[0],
                layer=1,
                length=length,
                own_even=ow_tbl[r & ~1],
                own_odd=ow_tbl[r | 1],
                rank_major=rm,
            )
            return out[None]

        return jax.shard_map(
            _fn,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
            out_specs=P("k3_sp"),
            check_vma=False,
        )

    got_rm = np.asarray(_make_runner(True)(or_rm, ko_half, px, owners_dev), np.float32)
    ref_bm = np.asarray(_make_runner(False)(or_bm, ko_half, px, owners_dev), np.float32)
    np.testing.assert_array_equal(got_rm, ref_bm)

    mesh32 = jax.sharding.AbstractMesh((32,), ("k3_sp",))
    sh32 = jax.sharding.NamedSharding(mesh32, P("k3_sp"))
    for interp_flag in (True, False):

        def _trace_32(o_s, w_s, p_s, interp=interp_flag):
            out = owner_output_projection_chip_pair_pallas(
                o_s[0],
                w_s[0],
                p_s[0],
                layer=1,
                length=16384,
                own_even=0,
                own_odd=1,
                interpret=interp,
                rank_major=True,
            )
            return out[None]

        traced = jax.eval_shape(
            jax.shard_map(
                _trace_32,
                mesh=mesh32,
                in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp")),
                out_specs=P("k3_sp"),
                check_vma=False,
            ),
            jax.ShapeDtypeStruct((32, 32, 8, 64, 384), jnp.bfloat16, sharding=sh32),
            jax.ShapeDtypeStruct((32, 2, 12288, 3584), jnp.bfloat16, sharding=sh32),
            jax.ShapeDtypeStruct((32, 512, 7168), jnp.bfloat16, sharding=sh32),
        )
        assert traced.shape == (32, 512, 7168)
        assert traced.dtype == jnp.bfloat16

    with pytest.raises(ValueError, match="rank_major=True"):
        owner_output_projection_chip_pair_pallas(
            jnp.zeros((2, 32, 16, 384), jnp.bfloat16),
            jnp.zeros((2, 12288, 3584), jnp.bfloat16),
            jnp.zeros((32, 7168), jnp.bfloat16),
            layer=0,
            length=512,
            own_even=0,
            own_odd=1,
            rank_major=True,
        )


def test_should_invert_owner_table_and_match_full_a2a_when_streaming_ranges():
    class _FakeDev:
        def __init__(self, dev_id, coords, core):
            self.id = dev_id
            self.coords = coords
            self.core_on_chip = core

    fake_devs = [
        _FakeDev(
            ((x * 2 + y) * 4 + z) * 2 + c,
            (x, y, z),
            c,
        )
        for x in range(2)
        for y in range(2)
        for z in range(4)
        for c in range(2)
    ]
    token_ranks = tuple((r * 7 + 3) % 32 for r in range(32))
    tables_32 = build_kda_a2a_tables(fake_devs, token_ranks)
    assert tables_32.shape == (4, 3, 32)
    for peer in range(32):
        owner = int(tables_32[0, 2, peer])
        assert owner == token_ranks[peer]
        assert int(tables_32[3, 0, owner]) == peer

    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("tile-streamed A2A range test requires >= 2 JAX devices")
    num_ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    blocks, stripe, cols = 2, 8, 384
    owners = tuple((r * 3 + 1) % num_ranks for r in range(num_ranks))
    tables_small = np.zeros((4, 3, num_ranks), dtype=np.int32)
    for f in range(3):
        tables_small[f, 0, :] = np.arange(num_ranks)
        tables_small[f, 1, :] = np.arange(num_ranks)
        tables_small[f, 2, :] = owners
    for peer, own in enumerate(owners):
        tables_small[3, 0, own] = peer

    rng = np.random.default_rng(523)
    att_shape = (num_ranks, blocks * num_ranks * stripe, cols)
    att_np = (rng.standard_normal(att_shape) * 0.5).astype(np.float32)
    sh_sp = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    sh_rep = jax.sharding.NamedSharding(mesh, P())
    att_dev = jax.device_put(jnp.asarray(att_np, jnp.bfloat16), sh_sp)
    tbl_dev = jax.device_put(jnp.asarray(tables_small), sh_rep)

    def _run_a2a(tbl_in, streamed: bool):
        def _local(t_ref_in, a_in):
            hbm = pl.BlockSpec(memory_space=pltpu.HBM)

            def _kern(t_s, a_hbm, out_hbm, send_s, recv_s):
                step = pl.program_id(0)
                if streamed:
                    half = num_ranks // 2

                    @pl.when(step == 0)
                    def _b0_h0():
                        owner_all_to_all_send_range(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=0,
                            owner_start=0,
                            owner_end=half,
                            rank_major=True,
                        )

                    @pl.when(step == 1)
                    def _b0_h1():
                        owner_all_to_all_send_range(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=0,
                            owner_start=half,
                            owner_end=num_ranks,
                            rank_major=True,
                        )
                        owner_all_to_all_join(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=0,
                            rank_major=True,
                        )
                        owner_all_to_all_send_range(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=1,
                            owner_start=0,
                            owner_end=half,
                            rank_major=True,
                        )

                    @pl.when(step == 2)
                    def _b1_h1():
                        owner_all_to_all_send_range(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=1,
                            owner_start=half,
                            owner_end=num_ranks,
                            rank_major=True,
                        )
                        owner_all_to_all_join(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=1,
                            rank_major=True,
                        )
                else:

                    @pl.when(step == 0)
                    def _b0_full():
                        owner_all_to_all_send(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=0,
                            rank_major=True,
                        )

                    @pl.when(step == 1)
                    def _b0_join_b1_send():
                        owner_all_to_all_join(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=0,
                            rank_major=True,
                        )
                        owner_all_to_all_send(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=1,
                            rank_major=True,
                        )

                    @pl.when(step == 2)
                    def _b1_join():
                        owner_all_to_all_join(
                            t_s,
                            a_hbm,
                            out_hbm,
                            send_s,
                            recv_s,
                            block=1,
                            rank_major=True,
                        )

            out_type = jax.ShapeDtypeStruct(
                (num_ranks, blocks, stripe, cols), jnp.bfloat16
            )
            out = pl.pallas_call(
                _kern,
                out_shape=out_type,
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    grid=(3,),
                    in_specs=[hbm],
                    out_specs=hbm,
                    scratch_shapes=[
                        pltpu.SemaphoreType.DMA((4,)),
                        pltpu.SemaphoreType.DMA((4,)),
                    ],
                ),
                interpret=pltpu.InterpretParams(),
            )(t_ref_in, a_in[0])
            return out[None]

        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(), P("k3_sp")),
            out_specs=P("k3_sp"),
            check_vma=False,
        )(tbl_in, att_dev)

    got_streamed = np.asarray(_run_a2a(tbl_dev, streamed=True), np.float32)
    ref_full = np.asarray(_run_a2a(tbl_dev, streamed=False), np.float32)
    np.testing.assert_array_equal(got_streamed, ref_full)

    permuted_inv_tables = tables_small.copy()
    permuted_inv_tables[3, 0, :] = np.roll(permuted_inv_tables[3, 0, :], 1)
    bad_tbl_dev = jax.device_put(jnp.asarray(permuted_inv_tables), sh_rep)
    corrupted = np.asarray(_run_a2a(bad_tbl_dev, streamed=True), np.float32)
    assert np.max(np.abs(corrupted - ref_full)) > 0.2


def test_should_reuse_cached_weights_across_blocks_when_mixture_norm_skips_reload():
    rng = np.random.default_rng(537)
    blocks_cnt, stripe, hidden = 3, 16, 256
    rows = blocks_cnt * stripe
    prefix = jnp.asarray(rng.standard_normal((rows, hidden)) * 0.4, jnp.bfloat16)
    blocks = jnp.asarray(rng.standard_normal((8, rows, hidden)) * 0.3, jnp.bfloat16)
    folded_good = jnp.asarray(rng.standard_normal((1, hidden)) * 0.1, jnp.float32)
    norm_good = jnp.asarray(1.0 + 0.1 * rng.standard_normal((1, hidden)), jnp.bfloat16)
    poison_f32 = jnp.full((hidden,), 999.0, jnp.float32)
    poison_bf16 = jnp.full((hidden,), 999.0, jnp.bfloat16)
    poisoned_folded_bank = jnp.stack([folded_good[0], poison_f32, poison_f32])[:, None]
    poisoned_norm_bank = jnp.stack([norm_good[0], poison_bf16, poison_bf16])[:, None]
    active = jnp.asarray([5], jnp.int32)

    expected = prefill_megakernel.make_mixture_norm_tiled(
        seq_len=rows, tile=stripe, eps=1e-5, interpret=True
    )(prefix, blocks, folded_good, norm_good, active)

    def _run_multistep(reload_every_step: bool):
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        scratch = prefill_megakernel.mixture_norm_tiled_scratch_shapes(hidden, stripe)

        def _body(act_r, px_r, bl_r, fo_r, nr_r, out_r, *sc):
            s = pl.program_id(0)
            rs = pl.ds(s * stripe, stripe)
            if reload_every_step:
                prefill_megakernel.mixture_norm_tiled_body(
                    act_r,
                    px_r.at[rs],
                    bl_r.at[:, rs],
                    fo_r.at[s],
                    nr_r.at[s],
                    out_r.at[rs],
                    *sc,
                    tile=stripe,
                    eps=1e-5,
                    load_weights=True,
                )
            else:

                @pl.when(s == 0)
                def _first():
                    prefill_megakernel.mixture_norm_tiled_body(
                        act_r,
                        px_r.at[rs],
                        bl_r.at[:, rs],
                        fo_r.at[s],
                        nr_r.at[s],
                        out_r.at[rs],
                        *sc,
                        tile=stripe,
                        eps=1e-5,
                        load_weights=True,
                    )

                @pl.when(s > 0)
                def _rest():
                    prefill_megakernel.mixture_norm_tiled_body(
                        act_r,
                        px_r.at[rs],
                        bl_r.at[:, rs],
                        fo_r.at[s],
                        nr_r.at[s],
                        out_r.at[rs],
                        *sc,
                        tile=stripe,
                        eps=1e-5,
                        load_weights=False,
                    )

        return pl.pallas_call(
            _body,
            out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(blocks_cnt,),
                in_specs=(hbm,) * 4,
                out_specs=hbm,
                scratch_shapes=scratch,
            ),
            interpret=True,
        )(active, prefix, blocks, poisoned_folded_bank, poisoned_norm_bank)

    got_cached = _run_multistep(reload_every_step=False)
    np.testing.assert_array_equal(np.asarray(got_cached), np.asarray(expected))

    got_poisoned = _run_multistep(reload_every_step=True)
    tail_diff = np.abs(
        np.asarray(got_poisoned, np.float32)[stripe:]
        - np.asarray(expected, np.float32)[stripe:]
    )
    assert np.max(tail_diff) > 10.0
