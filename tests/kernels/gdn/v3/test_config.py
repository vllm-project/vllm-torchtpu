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
"""CPU unit tests for GDN kernel configuration and VMEM pad helpers."""

from typing import cast
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.gdn.v3 import config, tiling, vmem_ldst, wrapper

_64MIB_VMEM_LIMIT = int(
    config.GDNConfig.DEFAULT_VMEM_FRACTION
    * pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
)
# Synthetic ample budget where VMEM never binds, testing headroom gating
# without encoding an unreleased hardware capacity.
_AMPLE_VMEM_LIMIT = 1 << 30


def _make_cfg(
    *, recurrent_state_dtype: jax.typing.DTypeLike = jnp.float32, **overrides
) -> config.GDNConfig:
    f32 = jnp.dtype(jnp.float32)
    kwargs = dict(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=f32,
            act_out=jnp.dtype(jnp.bfloat16),
            compute=f32,
            recurrent_state=jnp.dtype(recurrent_state_dtype),
            conv_state=f32,
        ),
        batch_size=128,
        dim_size=5120,
        kernel_size=4,
        tile_size=128,
        num_kq_heads=4,
        num_v_heads=32,
        kq_head_dim=128,
        v_head_dim=128,
    )
    kwargs.update(overrides)
    return config.GDNConfig(**kwargs)


class QkvPadScratchTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="gdn_even_cols_40",
            overrides=dict(num_kq_heads=4, num_v_heads=32, dim_size=40 * 128),
            want=(1, 128, 1, 41 * 128),
        ),
        dict(
            testcase_name="gdn_even_cols_80",
            overrides=dict(num_kq_heads=8, num_v_heads=64, dim_size=80 * 128),
            want=(1, 128, 1, 81 * 128),
        ),
        dict(
            testcase_name="kda_excluded",
            overrides=dict(
                attention_mode=config.AttentionMode.KDA,
                num_kq_heads=32,
                num_v_heads=32,
                dim_size=96 * 128,
                tile_size=64,
            ),
            want=None,
        ),
        dict(
            testcase_name="odd_cols_already_conflict_free",
            overrides=dict(num_kq_heads=3, num_v_heads=3, dim_size=9 * 128),
            want=None,
        ),
        dict(
            testcase_name="batched_decode",
            overrides=dict(mode=config.GDNMode.BATCHED, tile_size=8),
            want=None,
        ),
        dict(
            testcase_name="single_token_chunk",
            overrides=dict(tile_size=1),
            want=None,
        ),
        dict(
            testcase_name="speculative_window",
            overrides=dict(window_size=2),
            want=None,
        ),
        dict(
            testcase_name="unaligned_head_dim",
            overrides=dict(
                num_kq_heads=4,
                num_v_heads=32,
                kq_head_dim=64,
                v_head_dim=64,
                dim_size=20 * 128,
            ),
            want=None,
        ),
    )
    def test_qkv_pad_scratch_shape(self, overrides, want):
        self.assertEqual(_make_cfg(**overrides).qkv_pad_scratch_shape(128), want)

    @parameterized.named_parameters(
        dict(
            testcase_name="kq16_v64_f32_c64_64mib_skips",
            n_kq=16,
            n_v=64,
            chunk_sz=64,
            rec_dtype=jnp.float32,
            vmem_limit=_64MIB_VMEM_LIMIT,
            want=False,
        ),
        dict(
            testcase_name="kq16_v64_bf16_c64_64mib_fits",
            n_kq=16,
            n_v=64,
            chunk_sz=64,
            rec_dtype=jnp.bfloat16,
            vmem_limit=_64MIB_VMEM_LIMIT,
            want=True,
        ),
        dict(
            testcase_name="kq16_v64_f32_c32_64mib_fits",
            n_kq=16,
            n_v=64,
            chunk_sz=32,
            rec_dtype=jnp.float32,
            vmem_limit=_64MIB_VMEM_LIMIT,
            want=True,
        ),
        dict(
            testcase_name="kq8_v64_f32_c128_ample_vmem_fits",
            n_kq=8,
            n_v=64,
            chunk_sz=128,
            rec_dtype=jnp.float32,
            vmem_limit=_AMPLE_VMEM_LIMIT,
            want=True,
        ),
        dict(
            testcase_name="kq4_v32_f32_c128_64mib_fits",
            n_kq=4,
            n_v=32,
            chunk_sz=128,
            rec_dtype=jnp.float32,
            vmem_limit=_64MIB_VMEM_LIMIT,
            want=True,
        ),
    )
    def test_fits_qkv_pad_scratch(
        self, n_kq, n_v, chunk_sz, rec_dtype, vmem_limit, want
    ):
        cfg = _make_cfg(
            recurrent_state_dtype=jnp.dtype(rec_dtype),
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            dim_size=(2 * n_kq + n_v) * 128,
            tile_size=chunk_sz,
        )
        qpad_shape = cfg.qkv_pad_scratch_shape(128)
        self.assertIsNotNone(qpad_shape)
        self.assertEqual(
            tiling.fits_qkv_pad_scratch(cfg, qpad_shape, vmem_limit, 128),
            want,
        )

    def test_odd_column_strided_load_is_bit_identical_with_nan_spacer(self):
        """Strided Q/K/V loads from an odd-padded ref ignore the spacer column."""
        rng = np.random.default_rng(5)
        chunk, num_lanes, n_kq, n_v, d = 16, 128, 2, 8, 128
        dim = (2 * n_kq + n_v) * d
        qkv = jnp.asarray(rng.standard_normal((chunk, 1, dim)), dtype=jnp.float32)
        nan_spacer = jnp.full((chunk, 1, num_lanes), jnp.nan, dtype=jnp.float32)
        qkv_padded = jnp.concat([qkv, nan_spacer], axis=-1)

        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            kq_lanes = n_kq * (d // num_lanes)
            for num_heads, offset in (
                (n_kq, 0),
                (n_kq, kq_lanes),
                (n_v, 2 * kq_lanes),
            ):
                unpadded = vmem_ldst.load_strided_heads(
                    cast(jax.Ref, qkv), num_heads, d, offset
                )
                padded = vmem_ldst.load_strided_heads(
                    cast(jax.Ref, qkv_padded), num_heads, d, offset
                )
                np.testing.assert_array_equal(np.asarray(padded), np.asarray(unpadded))

    @parameterized.named_parameters(
        dict(
            testcase_name="gdn_even_cols_allocates_spacer",
            n_kq=2,
            n_v=8,
            attention_mode=config.AttentionMode.GDN,
            mixed_tile_size=16,
            gdn_prefill_mode=None,
            want_shape="f32[1,16,1,1664]",
            expect_skip_log=False,
        ),
        dict(
            testcase_name="gdn_odd_cols_omits_spacer",
            n_kq=3,
            n_v=3,
            attention_mode=config.AttentionMode.GDN,
            mixed_tile_size=16,
            gdn_prefill_mode=None,
            want_shape=None,
            expect_skip_log=False,
        ),
        dict(
            testcase_name="kda_even_cols_omits_spacer",
            n_kq=2,
            n_v=8,
            attention_mode=config.AttentionMode.KDA,
            mixed_tile_size=16,
            gdn_prefill_mode=None,
            want_shape=None,
            expect_skip_log=False,
        ),
        dict(
            testcase_name="tight_vmem_sequential_skips_spacer",
            n_kq=16,
            n_v=64,
            attention_mode=config.AttentionMode.GDN,
            mixed_tile_size=64,
            gdn_prefill_mode="sequential",
            want_shape=None,
            expect_skip_log=True,
        ),
        dict(
            testcase_name="tight_vmem_default_block_doubling_allocates_spacer",
            n_kq=16,
            n_v=64,
            attention_mode=config.AttentionMode.GDN,
            mixed_tile_size=64,
            gdn_prefill_mode=None,
            want_shape="f32[1,64,1,12416]",
            expect_skip_log=False,
        ),
    )
    def test_wrapper_pallas_call_qkv_pad_scratch_allocation(
        self,
        n_kq,
        n_v,
        attention_mode,
        mixed_tile_size,
        gdn_prefill_mode,
        want_shape,
        expect_skip_log,
    ):
        """Verifies fused_conv1d_gdn allocates or omits qkv_pad_scratch_ref."""
        d, kernel_size = 128, 4
        tok = mixed_tile_size
        dim = (2 * n_kq + n_v) * d
        is_kda = attention_mode == config.AttentionMode.KDA
        a_dim = n_v * d if is_kda else n_v
        dt_shape = (n_v, d) if is_kda else (n_v,)
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        mock_logger = mock.MagicMock()
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(wrapper, "logger", new=mock_logger),
        ):
            jaxpr_str = str(
                jax.make_jaxpr(
                    lambda: wrapper.fused_conv1d_gdn(
                        qkv=jnp.zeros((tok, dim), jnp.float32),
                        b=jnp.zeros((tok, n_v), jnp.float32),
                        a=jnp.zeros((tok, a_dim), jnp.float32),
                        states=(
                            jnp.zeros((2, kernel_size - 1, dim), jnp.float32),
                            jnp.zeros((2, n_v, d, d), jnp.float32),
                        ),
                        conv_weight=jnp.zeros((dim, 1, kernel_size), jnp.float32),
                        conv_bias=jnp.zeros((dim,), jnp.float32),
                        a_log=jnp.zeros((n_v,), jnp.float32),
                        dt_bias=jnp.zeros(dt_shape, jnp.float32),
                        query_start_loc=jnp.array([0, tok], jnp.int32),
                        state_indices=jnp.array([1], jnp.int32),
                        distribution=jnp.array([0, 1, 1], jnp.int32),
                        seq_lens=jnp.array([tok], jnp.int32),
                        n_kq=n_kq,
                        n_v=n_v,
                        d_k=d,
                        d_v=d,
                        kernel_size=kernel_size,
                        attention_mode=attention_mode,
                        gdn_prefill_mode=gdn_prefill_mode,
                        mixed_tile_size=mixed_tile_size,
                    )
                )()
            )
        padded_dim = (2 * n_kq + n_v + 1) * d
        spacer_sig = f"f32[1,{mixed_tile_size},1,{padded_dim}]"
        if want_shape is not None:
            self.assertIn(want_shape, jaxpr_str)
        else:
            self.assertNotIn(spacer_sig, jaxpr_str)
        if expect_skip_log:
            mock_logger.info_once.assert_called_once()
        else:
            mock_logger.info_once.assert_not_called()


class GDNPrefillModeTest(parameterized.TestCase):
    @parameterized.parameters(
        ("sequential", None),
        ("block_doubling_bf16", None),
        ("block_doubling_fp8", jnp.float8_e4m3fn),
    )
    def test_mxu_operand_dtype(self, mode, want_dtype):
        got = config.GDNPrefillMode(mode).mxu_operand_dtype
        if want_dtype is None:
            self.assertIsNone(got)
        else:
            self.assertEqual(got, jnp.dtype(want_dtype))


class GDNConfigUseBlockDoublingTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="bf16_chunk_32", overrides=dict(tile_size=32), want=True),
        dict(testcase_name="bf16_chunk_128", overrides=dict(), want=True),
        dict(
            testcase_name="fp8_chunk_128",
            overrides=dict(gdn_prefill_mode=config.GDNPrefillMode.BLOCK_DOUBLING_FP8),
            want=True,
        ),
        dict(
            testcase_name="sequential",
            overrides=dict(gdn_prefill_mode=config.GDNPrefillMode.SEQUENTIAL),
            want=False,
        ),
        dict(
            testcase_name="batched_decode",
            overrides=dict(mode=config.GDNMode.BATCHED, tile_size=8),
            want=False,
        ),
        dict(
            testcase_name="speculative_window",
            overrides=dict(window_size=2),
            want=False,
        ),
        dict(
            testcase_name="kda",
            overrides=dict(attention_mode=config.AttentionMode.KDA),
            want=False,
        ),
        dict(testcase_name="small_chunk", overrides=dict(tile_size=16), want=False),
        dict(
            testcase_name="non_multiple_of_32",
            overrides=dict(tile_size=48),
            want=False,
        ),
        dict(
            testcase_name="chunk_96_rejected",
            overrides=dict(tile_size=96),
            want=False,
        ),
        dict(
            testcase_name="chunk_192_rejected",
            overrides=dict(tile_size=192),
            want=False,
        ),
        dict(
            testcase_name="chunk_256_supported",
            overrides=dict(tile_size=256),
            want=True,
        ),
    )
    def test_use_block_doubling(self, overrides, want):
        params = dict(gdn_prefill_mode=config.GDNPrefillMode.BLOCK_DOUBLING_BF16)
        params.update(overrides)
        cfg = _make_cfg(**params)
        self.assertEqual(cfg.use_block_doubling, want)
        if want:
            self.assertIn(cfg.gdn_prefill_mode.value, cfg.get_kernel_name())

    def test_fits_qkv_pad_scratch_enforces_vmem_limit_for_block_doubling(self):
        cfg = _make_cfg(
            num_kq_heads=16,
            num_v_heads=64,
            dim_size=(2 * 16 + 64) * 128,
            tile_size=64,
            gdn_prefill_mode=config.GDNPrefillMode.BLOCK_DOUBLING_BF16,
        )
        qpad_shape = cfg.qkv_pad_scratch_shape(128)
        self.assertIsNotNone(qpad_shape)
        self.assertTrue(
            tiling.fits_qkv_pad_scratch(cfg, qpad_shape, _64MIB_VMEM_LIMIT, 128)
        )
        self.assertFalse(
            tiling.fits_qkv_pad_scratch(cfg, qpad_shape, 16 * 1024 * 1024, 128)
        )


class BlockDoublingWrapperValidationTest(parameterized.TestCase):
    def _dummy_inputs(
        self, *, head_dim: int = 128, is_kda: bool = False, tok: int = 32
    ):
        n_kq, n_v, kernel_size = 2, 8, 4
        dim = (2 * n_kq + n_v) * head_dim
        a_dim = n_v * head_dim if is_kda else n_v
        dt_shape = (n_v, head_dim) if is_kda else (n_v,)
        return dict(
            qkv=jnp.zeros((tok, dim), jnp.bfloat16),
            b=jnp.zeros((tok, n_v), jnp.bfloat16),
            a=jnp.zeros((tok, a_dim), jnp.bfloat16),
            states=(
                jnp.zeros((2, kernel_size - 1, dim), jnp.float32),
                jnp.zeros((2, n_v, head_dim, head_dim), jnp.float32),
            ),
            conv_weight=jnp.zeros((dim, 1, kernel_size), jnp.float32),
            conv_bias=jnp.zeros((dim,), jnp.float32),
            a_log=jnp.zeros((n_v,), jnp.float32),
            dt_bias=jnp.zeros(dt_shape, jnp.float32),
            query_start_loc=jnp.array([0, tok], jnp.int32),
            state_indices=jnp.array([1], jnp.int32),
            distribution=jnp.array([0, 1, 1], jnp.int32),
            seq_lens=jnp.array([tok], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=head_dim,
            d_v=head_dim,
            kernel_size=kernel_size,
        )

    def _trace(
        self,
        *,
        device: str = "TPU7x",
        head_dim: int = 128,
        is_kda: bool = False,
        tok: int = 32,
        input_overrides: dict | None = None,
        **kwargs,
    ) -> str:
        inputs = self._dummy_inputs(head_dim=head_dim, is_kda=is_kda, tok=tok)
        if input_overrides:
            inputs.update(input_overrides)
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice(device, 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            return str(
                jax.make_jaxpr(lambda: wrapper.fused_conv1d_gdn(**inputs, **kwargs))()
            )

    @parameterized.named_parameters(
        dict(
            testcase_name="unknown_mode",
            err=ValueError,
            msg="not a valid GDNPrefillMode",
            kwargs=dict(gdn_prefill_mode="unknown_mode"),
        ),
        dict(
            testcase_name="kda_with_block_doubling",
            err=ValueError,
            msg="only applies to GDN",
            is_kda=True,
            kwargs=dict(
                attention_mode=config.AttentionMode.KDA,
                gdn_prefill_mode="block_doubling_bf16",
            ),
        ),
        dict(
            testcase_name="non_lane_head_dims",
            err=ValueError,
            msg="d_k and d_v",
            head_dim=64,
            kwargs=dict(gdn_prefill_mode="block_doubling_bf16"),
        ),
        dict(
            testcase_name="fp8_without_fp8_mxu",
            err=NotImplementedError,
            msg="TPU v5e.*float8_e4m3fn",
            device="TPU v5e",
            kwargs=dict(gdn_prefill_mode="block_doubling_fp8"),
        ),
    )
    def test_rejects_invalid_prefill_mode_configs(
        self, err, msg, kwargs, device="TPU7x", head_dim=128, is_kda=False
    ):
        with self.assertRaisesRegex(err, msg):
            self._trace(device=device, head_dim=head_dim, is_kda=is_kda, **kwargs)

    def test_prefill_mode_selection_and_fallbacks(self):
        # Default enables BF16 block doubling for eligible GDN; "sequential" opts out.
        self.assertIn("_block_doubling_bf16", self._trace(mixed_tile_size=64))
        self.assertNotIn(
            "_block_doubling",
            self._trace(gdn_prefill_mode="sequential", mixed_tile_size=64),
        )
        # Unsupported chunk sizes, KDA, and non-128 head dims fall back to sequential.
        self.assertNotIn(
            "_block_doubling",
            self._trace(gdn_prefill_mode="block_doubling_bf16", mixed_tile_size=16),
        )
        self.assertNotIn(
            "_block_doubling",
            self._trace(
                is_kda=True,
                attention_mode=config.AttentionMode.KDA,
                mixed_tile_size=64,
            ),
        )
        self.assertNotIn(
            "_block_doubling",
            self._trace(
                head_dim=64,
                tok=1,
                input_overrides={"distribution": jnp.array([1, 1, 1], jnp.int32)},
                batched_only=True,
            ),
        )


if __name__ == "__main__":
    absltest.main()
