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
"""100% diff coverage unit test suite for .

Targets:
1. scripts/dsv41_bench/bench_decode_indexer_split.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import runpy
import sys
from unittest.mock import MagicMock, patch

# Ensure CPU execution defaults before importing jax or TPU runtime libraries.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import numpy as np
import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "scripts" / "dsv41_bench"

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse existing common module if already loaded to avoid module shadowing.
existing_common = sys.modules.get("common")
if existing_common is not None:
    common = existing_common
else:
    from scripts.dsv41_bench import common  # noqa: E402

    sys.modules.setdefault("common", common)
    sys.modules["common"] = common

import bench_decode_indexer_split as bdis  # noqa: E402

bdis.common = common


# Helper context manager to redirect .to("tpu") and
# torch.tensor(..., device="tpu") to CPU
class _TpuToCpuContext:
    def __enter__(self):
        self._orig_to = torch.Tensor.to
        self._orig_tensor = torch.tensor

        def safe_to(tensor_self, *args, **kwargs):
            if args and args[0] == "tpu":
                args = ("cpu",) + args[1:]
            if kwargs.get("device") == "tpu":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return self._orig_to(tensor_self, *args, **kwargs)

        def safe_tensor(*args, **kwargs):
            if kwargs.get("device") == "tpu":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return self._orig_tensor(*args, **kwargs)

        torch.Tensor.to = safe_to
        torch.tensor = safe_tensor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.to = self._orig_to
        torch.tensor = self._orig_tensor


# ==============================================================================
# 2. Tests for scripts/dsv41_bench/bench_decode_indexer_split.py
# ==============================================================================


class TestBenchDecodeIndexerSplit:
    def test_constants_and_split_part(self):
        assert bdis.SUBOPS == ("candidate_write", "sparse_logits", "full_context")
        assert bdis.SPLITS == ("pad16", "balanced", "ceil")
        assert bdis.LEGACY_TILE == (16384, 2048)

        # 1. policy == "pad16"
        # count=16, tp=8 -> rows = 128 -> ((16 // 128 + 1) * 128) // 8 = 16
        assert bdis.split_part(16, 8, "pad16") == 16
        # count=128, tp=8 -> rows = 128 -> ((128 // 128 + 1) * 128) // 8 = 32
        assert bdis.split_part(128, 8, "pad16") == 32

        # 2. policy == "ceil"
        assert bdis.split_part(16, 8, "ceil") == 2
        assert bdis.split_part(17, 8, "ceil") == 3

        # 3. policy == "balanced" / other
        assert bdis.split_part(16, 8, "balanced") == 3
        assert bdis.split_part(16, 8, "unknown") == 3

    def test_build_inputs_and_rank_slice(self):
        # 1. page_records > 0
        inp1 = bdis.build_inputs(
            seed=42,
            count=4,
            kv_len=64,
            pool_rows=256,
            heads=8,
            head_dim=128,
            candidate_blocks=4,
            block_size=16,
            page_records=16,
        )
        assert inp1["pool"].shape == (128, 128)
        assert inp1["records"].shape == (4, 64)
        assert inp1["q"].shape == (4, 8, 128)
        assert inp1["weights"].shape == (4, 8)
        assert inp1["lengths"].shape == (4,)
        assert inp1["candidates"].shape == (4, 4)
        assert inp1["requests"].tolist() == [0, 1, 2, 3]

        # 2. page_records == 0 (random rows path line 88)
        inp2 = bdis.build_inputs(
            seed=42,
            count=4,
            kv_len=64,
            pool_rows=256,
            heads=8,
            head_dim=128,
            candidate_blocks=4,
            block_size=16,
            page_records=0,
        )
        assert inp2["records"].shape == (4, 64)

        # 3. rank_slice: part=2, rank=1, tp=4 -> count=4, padding = 2*4 - 4 = 4
        sliced = bdis.rank_slice(inp1, part=2, rank=1, tp=4)
        assert sliced["q"].shape == (2, 8, 128)
        assert sliced["lengths"].shape == (2,)
        assert sliced["candidates"].shape == (2, 4)
        assert sliced["requests"].shape == (2,)

    def test_worker_options_and_stages(self, tmp_path):
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)

        def dummy_op(name, fn, *args, **kwargs):
            def call_fn(*call_args):
                jax_args = [
                    (
                        jnp.asarray(a.float().cpu().numpy())
                        if (isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16)
                        else (
                            jnp.asarray(a.cpu().numpy())
                            if isinstance(a, torch.Tensor)
                            else a
                        )
                    )
                    for a in call_args
                ]
                with contextlib.suppress(Exception):
                    fn(*jax_args)
                return torch.zeros((4, 32), dtype=torch.int32)

            return call_fn

        with (
            _TpuToCpuContext(),
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context"
            ) as mock_wrap,
            patch("torch_tpu._internal.pallas.jax_op", side_effect=dummy_op),
            patch(
                "jax.eval_shape",
                side_effect=lambda f: (jnp.zeros((4, 32), dtype=jnp.int32), None),
            ),
            patch(
                "jax.lax.fori_loop",
                side_effect=lambda low, high, body, init: body(0, init),
            ),
            patch("jax.lax.optimization_barrier", side_effect=lambda x: x),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_bf16.paged_index_scores_jax",
                return_value=jnp.ones((4, 64), dtype=jnp.float32),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select.select_indices_jax",
                return_value=(
                    jnp.zeros((4, 32), dtype=jnp.int32),
                    jnp.zeros((4, 32), dtype=jnp.int32),
                ),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_bf16.paged_indexer_jax",
                return_value=(
                    jnp.zeros((4, 32), dtype=jnp.int32),
                    jnp.zeros((4, 32), dtype=jnp.int32),
                ),
            ),
            patch.object(common, "model_wrapper_context"),
            patch.object(common, "synchronize"),
            patch.object(
                common,
                "time_steps",
                side_effect=lambda c, step_fn, **kw: ([5.0], step_fn()),
            ),
        ):
            mock_tp.return_value.device_group = None
            mock_tp.return_value.rank_in_group = 0
            mock_wrap.return_value.mesh = None

            # Test keys_per_iter and max_tile overrides, plus subops & stages
            for stage in ["scores", "select", "all"]:
                for subop in (
                    "candidate_write",
                    "sparse_logits",
                    "full_context",
                ):
                    args = argparse.Namespace(
                        subop=subop,
                        stage=stage,
                        split="balanced",
                        reps=1,
                        num_seqs=4,
                        tokens_per_seq=1,
                        kv_len=64,
                        pool_rows=256,
                        layout_page=16,
                        page_records=16,
                        keys_per_iter=32768,
                        max_tile=4096,
                        legacy=False,
                        seed=42,
                        warmup=0,
                        iters=1,
                        verify=False,
                    )
                    res = bdis.worker(ctx, args)
                    assert res["bench"] == "decode_indexer_split"
                    assert res["subop"] == subop
                    assert res["stage"] == stage
                    assert res["keys_per_iter"] == 32768
                    assert res["max_tile"] == 4096
                    assert res["per_rep_median_ms"] == 5.0

                # Test legacy=True configuration
                args_legacy = argparse.Namespace(
                    subop="candidate_write",
                    stage="all",
                    split="balanced",
                    reps=1,
                    num_seqs=4,
                    tokens_per_seq=1,
                    kv_len=64,
                    pool_rows=256,
                    layout_page=16,
                    page_records=None,
                    keys_per_iter=None,
                    max_tile=None,
                    legacy=True,
                    seed=42,
                    warmup=0,
                    iters=1,
                    verify=False,
                )
                res_legacy = bdis.worker(ctx, args_legacy)
                assert res_legacy["config"] == "legacy"
                assert res_legacy["split"] == "pad16"
                assert res_legacy["keys_per_iter"] == bdis.LEGACY_TILE[0]
                assert res_legacy["max_tile"] == bdis.LEGACY_TILE[1]

    def test_worker_legacy_visible_branches(self, tmp_path):
        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
            indexer_select,
        )

        orig_vis = indexer_select._visible_top_k
        mock_vis = MagicMock(return_value="mock_result")
        indexer_select._visible_top_k = mock_vis

        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)
        args_legacy = argparse.Namespace(
            subop="candidate_write",
            stage="all",
            split="balanced",
            reps=1,
            num_seqs=4,
            tokens_per_seq=1,
            kv_len=64,
            pool_rows=256,
            layout_page=16,
            page_records=None,
            keys_per_iter=None,
            max_tile=None,
            legacy=True,
            seed=42,
            warmup=0,
            iters=1,
            verify=False,
        )
        try:
            with (
                _TpuToCpuContext(),
                patch("vllm.distributed.get_tp_group") as mock_tp,
                patch(
                    "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context"
                ) as mock_wrap,
                patch(
                    "torch_tpu._internal.pallas.jax_op",
                    side_effect=lambda *a, **k: (
                        lambda *x, **y: torch.zeros((4, 32), dtype=torch.int32)
                    ),
                ),
                patch.object(common, "model_wrapper_context"),
                patch.object(common, "synchronize"),
                patch.object(
                    common,
                    "time_steps",
                    side_effect=lambda c, step_fn, **kw: ([5.0], step_fn()),
                ),
            ):
                mock_tp.return_value.device_group = None
                mock_tp.return_value.rank_in_group = 0
                mock_wrap.return_value.mesh = None

                bdis.worker(ctx, args_legacy)
                legacy_vis = indexer_select._visible_top_k

                # 1. scores.shape[0] < 16 -> lengths set to None
                scores_small = np.zeros((8, 2048), dtype=np.float32)
                legacy_vis(scores_small, 64, 128, lengths="not_none", unordered=True)
                args_small, _ = mock_vis.call_args
                assert args_small[3] is None  # lengths is None

                # 2. scores.shape[0] >= 16 -> lengths preserved
                scores_large = np.zeros((16, 2048), dtype=np.float32)
                legacy_vis(scores_large, 64, 128, lengths="orig_len", unordered=True)
                args_large, _ = mock_vis.call_args
                assert args_large[3] == "orig_len"
        finally:
            indexer_select._visible_top_k = orig_vis

    def test_worker_verify_scores_stage(self, tmp_path):
        ctx = common.WorkerContext(0, 2, 2, None, tmp_path)

        def dummy_op(name, fn, *args, **kwargs):
            if "scores_pair" in name:

                def call_pair(*call_args):
                    jax_args = [
                        (
                            jnp.asarray(a.float().cpu().numpy())
                            if (
                                isinstance(a, torch.Tensor)
                                and a.dtype == torch.bfloat16
                            )
                            else (
                                jnp.asarray(a.cpu().numpy())
                                if isinstance(a, torch.Tensor)
                                else a
                            )
                        )
                        for a in call_args
                    ]
                    with contextlib.suppress(Exception):
                        fn(*jax_args)
                    return torch.zeros((2, 4, 64), dtype=torch.float32)

                return call_pair
            return lambda *a, **k: torch.zeros((2, 128), dtype=torch.int32)

        args = argparse.Namespace(
            subop="full_context",
            stage="scores",
            split="balanced",
            reps=1,
            num_seqs=4,
            tokens_per_seq=1,
            kv_len=64,
            pool_rows=256,
            layout_page=16,
            page_records=16,
            keys_per_iter=None,
            max_tile=None,
            legacy=False,
            seed=42,
            warmup=0,
            iters=1,
            verify=True,
        )

        with (
            _TpuToCpuContext(),
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context"
            ) as mock_wrap,
            patch("torch_tpu._internal.pallas.jax_op", side_effect=dummy_op),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_bf16.paged_index_scores_jax",
                return_value=jnp.ones((4, 64), dtype=jnp.float32),
            ),
            patch.object(
                common,
                "gather_object",
                side_effect=lambda c, obj: [obj] * c.world_size,
            ),
            patch.object(common, "model_wrapper_context"),
            patch.object(common, "synchronize"),
            patch.object(
                common,
                "time_steps",
                side_effect=lambda c, step_fn, **kw: ([5.0], step_fn()),
            ),
        ):
            mock_tp.return_value.device_group = None
            mock_tp.return_value.rank_in_group = 0
            mock_wrap.return_value.mesh = None

            # 1. Success verify
            res = bdis.worker(ctx, args)
            assert res["verify"]["page_vs_row_scores_bitwise"] is True

            # 2. ctx.rank != 0 branch (lines 325->355)
            ctx_rank1 = common.WorkerContext(1, 2, 2, None, tmp_path)
            res_r1 = bdis.worker(ctx_rank1, args)
            assert "page_vs_row_scores_bitwise" not in res_r1.get("verify", {})

            # 3. Failure verify: scores differ
            def diff_pair_op(name, fn, *args, **kwargs):
                if "scores_pair" in name:
                    t = torch.zeros((2, 4, 64), dtype=torch.float32)
                    t[0, 0, 0] = 1.0  # make pair[0] != pair[1]
                    return lambda *a, **k: t
                return lambda *a, **k: torch.zeros((2, 128), dtype=torch.int32)

            with (
                patch("torch_tpu._internal.pallas.jax_op", side_effect=diff_pair_op),
                pytest.raises(
                    common.VerificationError, match="page gather scores differ"
                ),
            ):
                bdis.worker(ctx, args)

    def test_worker_verify_legacy_stage(self, tmp_path):
        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)

        def dummy_op(name, fn, *args, **kwargs):
            return lambda *a, **k: torch.zeros((4, 32), dtype=torch.int32)

        args = argparse.Namespace(
            subop="candidate_write",
            stage="all",
            split="balanced",
            reps=1,
            num_seqs=4,
            tokens_per_seq=1,
            kv_len=64,
            pool_rows=256,
            layout_page=16,
            page_records=None,
            keys_per_iter=None,
            max_tile=None,
            legacy=False,
            seed=42,
            warmup=0,
            iters=1,
            verify=True,
        )

        with (
            _TpuToCpuContext(),
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context"
            ) as mock_wrap,
            patch("torch_tpu._internal.pallas.jax_op", side_effect=dummy_op),
            patch.object(
                common,
                "gather_object",
                side_effect=lambda c, obj: [obj] * c.world_size,
            ),
            patch.object(common, "model_wrapper_context"),
            patch.object(common, "synchronize"),
            patch.object(
                common,
                "time_steps",
                side_effect=lambda c, step_fn, **kw: ([5.0], step_fn()),
            ),
        ):
            mock_tp.return_value.device_group = None
            mock_tp.return_value.rank_in_group = 0
            mock_wrap.return_value.mesh = None

            # 1. Success verify
            res = bdis.worker(ctx, args)
            assert res["verify"]["legacy_vs_balanced_src_exact"] is True

            # 2. ctx.rank != 0 branch (lines 342->355)
            ctx_rank1 = common.WorkerContext(1, 2, 2, None, tmp_path)
            res_r1 = bdis.worker(ctx_rank1, args)
            assert "legacy_vs_balanced_src_exact" not in res_r1.get("verify", {})

            # 3. Failure verify: selection differs from legacy
            diff_call_count = [0]

            def diff_op(name, fn, *args, **kwargs):
                def inner(*a, **k):
                    diff_call_count[0] += 1
                    val = diff_call_count[0]
                    return torch.full((4, 32), val, dtype=torch.int32)

                return inner

            with (
                patch("torch_tpu._internal.pallas.jax_op", side_effect=diff_op),
                pytest.raises(
                    common.VerificationError, match="selection differs from legacy"
                ),
            ):
                bdis.worker(ctx, args)

    def test_bdis_main_and_entrypoint(self, monkeypatch):
        mock_run = MagicMock()
        with patch.object(common, "run_main", mock_run):
            bdis.main()
            assert mock_run.call_count == 1
            args, _ = mock_run.call_args
            assert args[0] == "decode_indexer_split"

        # Test __main__ block
        monkeypatch.setattr(sys, "argv", ["bench_decode_indexer_split.py", "--help"])
        with patch.object(common, "run_main") as mock_run:
            runpy.run_path(
                str(_BENCH_DIR / "bench_decode_indexer_split.py"), run_name="__main__"
            )
            assert mock_run.call_count == 1
