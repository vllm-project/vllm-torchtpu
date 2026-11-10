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
"""2D bf16 qkv streaming in the Qwen3.5 GDN v3 copy (bf16_qkv_mode=2).

A bf16 qkv reaches the kernel as 2D [tokens, dim] bf16 rows in 16-row aligned
DMA windows and is unpacked to f32 inside the kernel, instead of being cast
to f32 rows in HBM first (mode 0). Both modes must give bit-identical results.
CPU interpret mode runs the decode (BATCHED) kernel only, so the prefill
(PER_SEQ) comparisons are TPU tests.
"""

import contextlib
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.models.qwen3_5.kernels.gdn import tiling, wrapper

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 1, 2, 128, 128, 4
DIM = 2 * N_KQ * D_K + N_V * D_V


@contextlib.contextmanager
def _cpu_interpret():
    """CPU interpret mode that resolves granule_view's bitcast and reshape
    refs (the same shim as tests/kernels/test_gdn_attention_v3.py)."""
    mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    interpret_utils = mosaic_interpret.interpret_utils
    shared_memory = sys.modules[f"{mosaic_interpret.__package__}.shared_memory"]
    orig_store = shared_memory.SharedMemory.store_buffer_content

    def _resolve_to_range(transforms):
        ret, viewed = (), False
        for t in transforms:
            if isinstance(t, indexing.NDIndexer):
                idx = tuple(
                    interpret_utils._transform_slice_or_index(i) for i in t.indices
                )
                if viewed:
                    idx = idx[:-3]
                ret = interpret_utils._compose_slice_or_index(ret, idx)
            elif isinstance(
                t, (state_types.BitcastTransform, state_types.ReshapeTransform)
            ):
                if not viewed:
                    ret, viewed = ret[:-3], True
            else:
                raise NotImplementedError(f"unsupported ref transform: {t}")
        return ret

    def _store_buffer_content(buf_self, key, rnge, value, *args, **kwargs):
        buff_dtype = buf_self.mem[key].dtype
        return orig_store(buf_self, key, rnge, value.view(buff_dtype), *args, **kwargs)

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            mock.patch.object(interpret_utils, "to_range", _resolve_to_range),
            mock.patch.object(
                shared_memory.SharedMemory,
                "store_buffer_content",
                _store_buffer_content,
            ),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            yield
    finally:
        pltpu.reset_tpu_interpret_mode_state()


def _verify_window_batch(num_seqs=4, window=4, seed=0):
    rngs = iter(jax.random.split(jax.random.key(seed), 9))
    num_tokens = num_seqs * window
    num_blocks = (num_seqs + 1) * window
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM), jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        states=(
            jax.random.normal(next(rngs), (num_blocks, KERNEL_SIZE - 1, DIM)),
            jax.random.normal(next(rngs), (num_blocks, N_V, D_K, D_V)),
        ),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (DIM,), jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * window,
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32) * window,
        distribution=jnp.array([num_seqs] * 3, jnp.int32),
        seq_lens=jnp.full((num_seqs,), 20, jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
    )


def _kwargs(mode, window):
    return dict(
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        num_spec_tokens=window - 1,
        decode_tile_size=2,
        mixed_tile_size=8,
        batched_only=True,
        bf16_qkv_mode=mode,
    )


def _pallas_operands(inputs, mode, window=4):
    """(shape, dtype) of every operand of the decode pallas_call."""
    with use_abstract_mesh(
        AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    ):
        jaxpr = jax.make_jaxpr(
            lambda x: wrapper.fused_conv1d_gdn.__wrapped__(**x, **_kwargs(mode, window))
        )(inputs)
    (call,) = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "pallas_call"]
    return [(tuple(v.aval.shape), jnp.dtype(v.aval.dtype)) for v in call.invars]


def test_mode_2_streams_2d_bf16_rows_and_mode_0_casts():
    inputs = _verify_window_batch()
    rows_2d = ((16, DIM), jnp.dtype(jnp.bfloat16))
    rows_f32 = ((16, 1, DIM), jnp.dtype(jnp.float32))
    mode_2 = _pallas_operands(inputs, 2)
    mode_0 = _pallas_operands(inputs, 0)
    assert rows_2d in mode_2 and rows_f32 not in mode_2
    assert rows_f32 in mode_0 and rows_2d not in mode_0


def test_mode_2_falls_back_below_16_tokens():
    """One 4-token verify window cannot fill a 16-row DMA window."""
    inputs = _verify_window_batch(num_seqs=1)
    operands = _pallas_operands(inputs, 2)
    assert ((4, 1, DIM), jnp.dtype(jnp.float32)) in operands
    assert all(dtype != jnp.bfloat16 or len(shape) != 2 for shape, dtype in operands)


def test_mode_2_falls_back_when_the_tiler_picks_a_short_chunk():
    """The dynamic tiler can cap the prefill chunk below 8 rows; the 2D rows
    need a multiple of 8, so the call falls back to f32 rows."""
    inputs = _verify_window_batch()
    with (
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True),
        mock.patch.object(wrapper.tiling, "get_tile_sizes", return_value=(2, 4)),
    ):
        operands = _pallas_operands(inputs, 2)
    assert ((16, 1, DIM), jnp.dtype(jnp.float32)) in operands
    assert ((16, DIM), jnp.dtype(jnp.bfloat16)) not in operands


def test_mode_2_matches_mode_0_on_verify_windows():
    """W=4 verify windows at decode tile 2: bit-identical outputs and states."""
    results = {}
    for mode in (0, 2):
        inputs = _verify_window_batch(seed=1)
        with _cpu_interpret():
            results[mode] = jax.block_until_ready(
                wrapper.fused_conv1d_gdn(**inputs, **_kwargs(mode, 4))
            )
    (conv_0, rec_0), out_0 = results[0]
    (conv_2, rec_2), out_2 = results[2]
    np.testing.assert_array_equal(np.asarray(out_2), np.asarray(out_0))
    np.testing.assert_array_equal(np.asarray(conv_2), np.asarray(conv_0))
    np.testing.assert_array_equal(np.asarray(rec_2), np.asarray(rec_0))


def test_unknown_modes_are_refused():
    with pytest.raises(ValueError, match="bf16_qkv_mode"):
        wrapper.fused_conv1d_gdn.__wrapped__(**_verify_window_batch(), **_kwargs(1, 4))


# Qwen3.5-397B GDN heads per rank: (kq heads, v heads) at TP1, TP2 and TP8.
_LAYOUTS = {"tp1": (16, 64), "tp2": (8, 32), "tp8": (2, 8)}


def _shape(n_kq, n_v):
    dim = 2 * n_kq * D_K + n_v * D_V
    return dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        num_lanes=128,
        conv_state_dim_size=dim,
    )


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_2d_rows_widen_only_the_prefill_vmem_estimate(layout):
    """136 bytes per qkv column for a PER_SEQ tile, nothing for BATCHED."""
    shape = _shape(*_LAYOUTS[layout])
    dim = shape["conv_state_dim_size"]
    for is_decode, extra in ((False, dim * 136), (True, 0)):
        est = [
            tiling.get_vmem_estimate_bytes(
                tile_b=1,
                chunk_sz=1 if is_decode else 64,
                act_out_bytes=2,
                rec_state_bytes=2,
                is_decode=is_decode,
                qkv_2d=qkv_2d,
                **shape,
            )
            for qkv_2d in (False, True)
        ]
        assert est[1] - est[0] == extra


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_the_tiler_sizes_the_prefill_chunk_with_the_2d_term(layout):
    """Under a limit between the f32 and 2D estimates of a 128-token chunk,
    the f32 rows keep 128 and the 2D rows get a smaller chunk that fits."""
    shape = _shape(*_LAYOUTS[layout])

    def est(chunk, qkv_2d):
        return tiling.get_vmem_estimate_bytes(
            tile_b=1,
            chunk_sz=chunk,
            act_out_bytes=2,
            rec_state_bytes=2,
            is_decode=False,
            qkv_2d=qkv_2d,
            **shape,
        )

    limit = (est(128, False) + est(128, True)) // 2
    picks = {}
    for qkv_2d in (False, True):
        _, picks[qkv_2d] = tiling.get_tile_sizes(
            batch_size=8192,
            num_seqs=1,
            padded_batch_size=8192,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.bfloat16),
            decode_vmem_limit_bytes=limit,
            mixed_vmem_limit_bytes=limit,
            qkv_2d=qkv_2d,
            **shape,
        )
    assert picks[False] == 128
    assert picks[True] < 128
    assert est(picks[True], True) <= limit


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("lengths", "num_decode", "num_spec_tokens", "decode_tile_size"),
    [
        # Three 1-token decodes, then prefills of 37 and 128 tokens.
        ([1, 1, 1, 37, 128], 3, 0, 4),
        # 22 W=4 verify windows at decode tile 8, then two prefills.
        ([4] * 22 + [37, 128], 22, 3, 8),
    ],
)
def test_mode_2_matches_mode_0_with_prefill_on_tpu(
    lengths, num_decode, num_spec_tokens, decode_tile_size
):
    n_kq, n_v, d = 2, 8, 128
    dim = 2 * n_kq * d + n_v * d
    window = num_spec_tokens + 1
    num_seqs = len(lengths)
    num_tokens = sum(lengths)
    num_blocks = (num_seqs + 2) * window
    rngs = iter(jax.random.split(jax.random.key(42), 9))
    inputs = dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, dim), jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_tokens, n_v), jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_tokens, n_v), jnp.bfloat16),
        states=(
            jax.random.normal(next(rngs), (num_blocks, KERNEL_SIZE - 1, dim)),
            jax.random.normal(next(rngs), (num_blocks, n_v, d, d)),
        ),
        conv_weight=jax.random.normal(next(rngs), (dim, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (dim,), jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (n_v,), jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (n_v,), jnp.float32),
        query_start_loc=jnp.asarray(np.cumsum([0, *lengths]), jnp.int32),
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32) * window,
        distribution=jnp.array([num_decode, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.asarray([20] * num_decode + lengths[num_decode:], jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
    )
    results = {}
    for mode in (0, 2):
        results[mode] = jax.block_until_ready(
            wrapper.fused_conv1d_gdn(
                **inputs,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d,
                d_v=d,
                kernel_size=KERNEL_SIZE,
                num_spec_tokens=num_spec_tokens,
                decode_tile_size=decode_tile_size,
                bf16_qkv_mode=mode,
            )
        )
    for got, want in zip(
        jax.tree.leaves(results[2]), jax.tree.leaves(results[0]), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
