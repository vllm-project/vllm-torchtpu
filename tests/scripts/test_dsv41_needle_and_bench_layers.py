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
"""Unit tests for needle_32k and layer benchmark scripts.

Targets:
- scripts/dsv41_bench/needle_32k.py
- scripts/dsv41_bench/bench_kv_compressor.py
- scripts/dsv41_bench/bench_decoder_layer.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import runpy
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

BENCH_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import bench_decoder_layer  # noqa: E402
import bench_kv_compressor  # noqa: E402
import common  # noqa: E402
import needle_32k  # noqa: E402

pytestmark = pytest.mark.cpu_test


# ==============================================================================
# 1. Tests for scripts/dsv41_bench/needle_32k.py
# ==============================================================================


class TestNeedle32k:
    """Comprehensive line and branch tests for needle_32k.py."""

    def test_post_success(self) -> None:
        mock_resp_data = {"count": 42}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            res = needle_32k._post(
                "http://127.0.0.1:8012/test", {"key": "value"}, timeout=5.0
            )
            assert res == mock_resp_data
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8012/test"
            assert req.data == b'{"key": "value"}'
            assert req.headers["Content-type"] == "application/json"

    def test_count_tokens(self) -> None:
        with patch.object(
            needle_32k, "_post", return_value={"count": 128}
        ) as mock_post:
            count = needle_32k._count_tokens(
                "http://127.0.0.1:8012", "model-x", "text", timeout=10.0
            )
            assert count == 128
            mock_post.assert_called_once_with(
                "http://127.0.0.1:8012/tokenize",
                {"model": "model-x", "prompt": "text"},
                10.0,
            )

    def test_filler(self) -> None:
        rng = random.Random(42)
        sentences = needle_32k._filler(rng, 5)
        assert len(sentences) == 5
        assert all(s.endswith(".") for s in sentences)

    def test_build_prompt_boundaries_and_validation(self) -> None:
        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        prompt = needle_32k.build_prompt(sentences, 0.0, "NEEDLE", "QUESTION")
        assert prompt.startswith("Below is a long document.")
        assert "NEEDLE" in prompt
        assert prompt.endswith("QUESTION")

        prompt_end = needle_32k.build_prompt(sentences, 1.0, "NEEDLE", "QUESTION")
        assert "NEEDLE" in prompt_end

        with pytest.raises(ValueError, match="depth must be in \\[0, 1\\]"):
            needle_32k.build_prompt(sentences, -0.1, "NEEDLE", "QUESTION")
        with pytest.raises(ValueError, match="depth must be in \\[0, 1\\]"):
            needle_32k.build_prompt(sentences, 1.1, "NEEDLE", "QUESTION")

    def test_main_all_pass_with_out(self, tmp_path: pathlib.Path) -> None:
        out_file = tmp_path / "needle_out.json"

        def fake_post(url: str, payload: dict, timeout: float) -> dict:
            if "/tokenize" in url:
                return {"count": 400}
            if "/v1/chat/completions" in url:
                prompt = payload["messages"][0]["content"]
                match = re.search(
                    r"The special magic number for \w+ is (\d+)\.", prompt
                )
                key = match.group(1) if match else "1234567"
                return {
                    "choices": [{"message": {"content": f"The number is {key}."}}],
                    "usage": {"prompt_tokens": 100},
                }
            return {}

        with (
            patch.object(needle_32k, "_post", side_effect=fake_post),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.5",
                    f"--out={out_file}",
                    "--target-tokens=100",
                ],
            ),
        ):
            rc = needle_32k.main()
            assert rc == 0
            assert out_file.is_file()
            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert data["all_pass"] is True
            assert data["score"] == "1/1"
            assert len(data["trials"]) == 1
            assert data["trials"][0]["pass"] is True

    def test_main_failure_and_content_none(self) -> None:
        def fake_post(url: str, payload: dict, timeout: float) -> dict:
            if "/tokenize" in url:
                return {"count": 400}
            return {
                "choices": [{"message": {"content": None}}],
                "usage": {"prompt_tokens": 50},
            }

        with (
            patch.object(needle_32k, "_post", side_effect=fake_post),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.2",
                    "--target-tokens=50",
                ],
            ),
        ):
            rc = needle_32k.main()
            assert rc == 1

    def test_needle_entrypoint_via_runpy(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            b'{"count": 400, "choices": [{"message": {"content": "1234567"}}], '
            b'"usage": {"prompt_tokens": 100}}'
        )
        mock_resp.__enter__.return_value = mock_resp
        with (
            patch("urllib.request.urlopen", return_value=mock_resp),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.5",
                    "--target-tokens=100",
                ],
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(str(BENCH_DIR / "needle_32k.py"), run_name="__main__")
            assert exc_info.value.code in (0, 1)


# ==============================================================================
# 2. Tests for scripts/dsv41_bench/bench_kv_compressor.py
# ==============================================================================


class TestBenchKVCompressor:
    """Comprehensive line and branch tests for bench_kv_compressor.py."""

    def test_sys_path_insertion(self) -> None:
        bench_dir_str = str(BENCH_DIR)
        assert bench_dir_str in sys.path

    def test_resolve_device(self) -> None:
        assert bench_kv_compressor._resolve_device("cpu") == torch.device("cpu")
        assert bench_kv_compressor._resolve_device("meta") == torch.device("meta")

        # auto without TPU -> returns cpu
        with patch.dict(sys.modules, {"torch_tpu": None}):
            assert bench_kv_compressor._resolve_device("auto") == torch.device("cpu")

        # tpu without TPU -> raises
        with (
            patch.dict(sys.modules, {"torch_tpu": None}),
            pytest.raises(ModuleNotFoundError),
        ):
            bench_kv_compressor._resolve_device("tpu")

        # auto/tpu with mock TPU working
        mock_tpu_dev = SimpleNamespace(type="tpu")
        orig_device = torch.device
        with (
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
        ):
            assert bench_kv_compressor._resolve_device("auto") == mock_tpu_dev
            assert bench_kv_compressor._resolve_device("tpu") == mock_tpu_dev

    def test_sync_if_tpu(self) -> None:
        cpu_dev = torch.device("cpu")
        bench_kv_compressor._sync_if_tpu(cpu_dev, torch.zeros(1))

        tpu_dev = SimpleNamespace(type="tpu")
        # None tensors -> no-op
        bench_kv_compressor._sync_if_tpu(tpu_dev, None)
        # Non-none tensors -> call common.synchronize
        with patch.object(common, "synchronize") as mock_sync:
            t = torch.zeros(1)
            bench_kv_compressor._sync_if_tpu(tpu_dev, t)
            mock_sync.assert_called_once_with(t)

    def test_bench_ms(self) -> None:
        calls = []

        def dummy() -> None:
            calls.append(1)

        dur = bench_kv_compressor._bench_ms(dummy, warmup=2, iters=3)
        assert len(calls) == 5
        assert dur >= 0.0

    def test_rms_norm_ref(self) -> None:
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        w = torch.ones(64, dtype=torch.bfloat16)
        out = bench_kv_compressor._rms_norm_ref(x, w, eps=1e-6)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16

    def test_run_project_only_verify_and_no_verify(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")
        # verify=True
        r1, v1 = bench_kv_compressor.run_project_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True
        assert r1["name"] == "project_only"

        # verify=False (covers branch 176->189)
        r2, v2 = bench_kv_compressor.run_project_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=False
        )
        assert v2 is None
        assert r2["name"] == "project_only"

    def test_run_save_compress_norm_only_verify_and_no_verify(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")
        # verify=True
        r1, v1 = bench_kv_compressor.run_save_compress_norm_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True
        assert r1["name"] == "save_compress_norm_only"

        # verify=False (covers branch 243->259)
        r2, v2 = bench_kv_compressor.run_save_compress_norm_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=False
        )
        assert v2 is None
        assert r2["name"] == "save_compress_norm_only"

    def test_run_write_fp8_only_branches(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")

        # total_slots < num_tokens assertion
        with pytest.raises(AssertionError, match="must be >= num_tokens"):
            bench_kv_compressor.run_write_fp8_only(
                dev,
                hw,
                num_tokens=64,
                num_pages=2,
                page_size=16,
                warmup=1,
                iters=1,
                verify=False,
            )

        # verify=True on CPU
        r1, v1 = bench_kv_compressor.run_write_fp8_only(
            dev,
            hw,
            num_tokens=32,
            num_pages=4,
            page_size=16,
            warmup=1,
            iters=1,
            verify=True,
        )
        assert v1 is not None
        assert v1["ok"] is True

        # verify=False on CPU (covers branch 305->321)
        r2, v2 = bench_kv_compressor.run_write_fp8_only(
            dev,
            hw,
            num_tokens=32,
            num_pages=4,
            page_size=16,
            warmup=1,
            iters=1,
            verify=False,
        )
        assert v2 is None

        # non-CPU device covers lines 278-280 (device.type != "cpu")
        meta_dev = torch.device("meta")
        with (
            patch.object(bench_kv_compressor, "initialize_cache_ops") as mock_init_ops,
            patch.object(bench_kv_compressor.PagedCacheRows, "write"),
        ):
            r3, v3 = bench_kv_compressor.run_write_fp8_only(
                meta_dev,
                hw,
                num_tokens=16,
                num_pages=2,
                page_size=16,
                warmup=1,
                iters=1,
                verify=False,
            )
            assert mock_init_ops.called
            assert v3 is None

    def test_run_engram_host_vs_hbm_branches(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")

        # verify=True on CPU
        rh1, rm1, v1 = bench_kv_compressor.run_engram_host_vs_hbm(
            dev, hw, num_tokens=16, engram_rows=128, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True

        # verify=False on CPU (covers branch 428->439)
        rh2, rm2, v2 = bench_kv_compressor.run_engram_host_vs_hbm(
            dev, hw, num_tokens=16, engram_rows=128, warmup=1, iters=1, verify=False
        )
        assert v2 is None

        # non-CPU device covers lines 373-374 (emb_hbm.to(device))
        meta_dev = torch.device("meta")
        with (
            patch.object(
                bench_kv_compressor.ParallelEngramEmbedding,
                "to",
                return_value=MagicMock(),
            ) as mock_to,
            patch.object(bench_kv_compressor.ParallelEngramEmbedding, "lookup"),
            patch.object(torch.Tensor, "copy_"),
        ):
            rh3, rm3, v3 = bench_kv_compressor.run_engram_host_vs_hbm(
                meta_dev,
                hw,
                num_tokens=16,
                engram_rows=128,
                warmup=1,
                iters=1,
                verify=False,
            )
            assert mock_to.called
            assert v3 is None

    def test_main_individual_subops_and_rank_nonzero(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Run individual subops to cover 542->550, 550->558, 558->573, 573->587
        subops = [
            "project_only",
            "save_compress_norm_only",
            "write_fp8_only",
            "engram_host_vs_hbm",
        ]
        for subop in subops:
            summary = bench_kv_compressor.main(
                [
                    "--device=cpu",
                    f"--subop={subop}",
                    "--num-tokens=16",
                    "--num-pages=2",
                    "--page-size=16",
                    "--engram-rows=128",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary["subop"] == subop

        # Run with rank != 0 and --dump-llo (covers lines 536 and 596->619)
        llo_dir = tmp_path / "llo"
        with (
            patch.dict(os.environ, {"RANK": "1"}),
            patch.object(common, "_llo_env") as mock_llo,
        ):
            summary_rank1 = bench_kv_compressor.main(
                [
                    "--device=cpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    f"--dump-llo={llo_dir}",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_rank1["subop"] == "project_only"
            mock_llo.assert_called_once()

    def test_main_tpu_orchestration_and_subprocess_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Case 1: device=auto, RANK not in os.environ, tpu_count > 0, json_out exists
        out_json = tmp_path / "summary_tpu.json"
        out_json.write_text(json.dumps({"from": "subprocess"}), encoding="utf-8")

        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=8,
            ),
            patch.object(common, "prepare_worker_env", return_value={"FAKE": "1"}),
            patch("subprocess.run") as mock_subproc,
        ):
            res1 = bench_kv_compressor.main(["--device=auto", f"--json-out={out_json}"])
            assert res1 == {"from": "subprocess"}
            mock_subproc.assert_called_once()

        # Case 2: device=auto, RANK not in os.environ, tpu_count > 0, no json_out
        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=8,
            ),
            patch.object(common, "prepare_worker_env", return_value={"FAKE": "1"}),
            patch("subprocess.run"),
        ):
            res2 = bench_kv_compressor.main(["--device=auto", "--subop=write_fp8_only"])
            assert res2 == {"device": "tpu", "subop": "write_fp8_only"}

        # Case 3: device=tpu, RANK not in os.environ, tpu_count == 0 -> raises
        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=0,
            ),
            pytest.raises(RuntimeError, match="No TPU devices found."),
        ):
            bench_kv_compressor.main(["--device=tpu"])

        # Case 4: device=auto, tpu_count == 0 -> sets device = cpu (line 522)
        with patch(
            "torch_tpu._internal.utils.hardware.get_tpu_device_count", return_value=0
        ):
            summary_auto = bench_kv_compressor.main(
                [
                    "--device=auto",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_auto["device"] == "cpu"
        # Case 4b: hardware.get_tpu_device_count raises exception (lines 499-500)
        with patch(
            "torch_tpu._internal.utils.hardware.get_tpu_device_count",
            side_effect=RuntimeError("simulated error"),
        ):
            summary_err = bench_kv_compressor.main(
                [
                    "--device=auto",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_err["device"] == "cpu"

        # Case 5a: worker branch with dist uninitialized initially
        mock_tpu_dev = SimpleNamespace(type="tpu")
        orig_device = torch.device
        init_state = {"init": False}

        def mock_is_initialized() -> bool:
            return init_state["init"]

        def mock_init_pg(*args: Any, **kwargs: Any) -> None:
            init_state["init"] = True

        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", side_effect=mock_is_initialized),
            patch("torch.distributed.init_process_group", side_effect=mock_init_pg),
            patch("torch.distributed.barrier") as mock_barrier,
            patch("torch.distributed.destroy_process_group") as mock_destroy,
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=({"name": "project_only", "lower_bound_ms": 0.01}, None),
            ),
        ):
            summary_worker = bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_worker["subop"] == "project_only"
            assert mock_barrier.called
            assert mock_destroy.called

        # Case 5b: worker branch with dist already initialized (line 527 False branch)
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.init_process_group") as mock_init_pg_skip,
            patch("torch.distributed.barrier"),
            patch("torch.distributed.destroy_process_group"),
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=({"name": "project_only", "lower_bound_ms": 0.01}, None),
            ),
        ):
            bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            mock_init_pg_skip.assert_not_called()

        # Case 5c: worker branch where dist is uninitialized at exit (lines 622->625)
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.barrier") as mock_barrier_skip,
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=({"name": "project_only", "lower_bound_ms": 0.01}, None),
            ),
        ):
            bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            mock_barrier_skip.assert_not_called()

    def test_main_json_out_write(self, tmp_path: pathlib.Path) -> None:
        out_file = tmp_path / "nested" / "summary.json"
        summary = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=project_only",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                f"--json-out={out_file}",
            ]
        )
        assert out_file.is_file()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded["subop"] == summary["subop"]
        assert loaded["device"] == "cpu"

    def test_main_all_subops_with_and_without_verify(self) -> None:
        summary_verify = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=all",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                "--verify",
            ]
        )
        assert len(summary_verify["verifications"]) == 4
        assert len(summary_verify["rooflines"]) == 5

        summary_no_verify = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=all",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                "--no-verify",
            ]
        )
        assert len(summary_no_verify["verifications"]) == 0
        assert len(summary_no_verify["rooflines"]) == 5

    def test_bench_kv_compressor_sys_path_insert(self) -> None:
        bench_dir_str = str(BENCH_DIR.resolve())
        orig_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p != bench_dir_str]
            code = compile(
                (BENCH_DIR / "bench_kv_compressor.py").read_text(encoding="utf-8"),
                str(BENCH_DIR / "bench_kv_compressor.py"),
                "exec",
            )
            globs = {
                "__name__": "bench_kv_compressor_test_mod",
                "__file__": str(BENCH_DIR / "bench_kv_compressor.py"),
            }
            exec(code, globs)
            assert bench_dir_str in sys.path
        finally:
            sys.path = orig_path

    def test_bench_kv_compressor_entrypoint_via_runpy(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "bench_kv_compressor.py",
                "--device=cpu",
                "--subop=project_only",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
            ],
        ):
            runpy.run_path(
                str(BENCH_DIR / "bench_kv_compressor.py"), run_name="__main__"
            )


# ==============================================================================
# 3. Tests for scripts/dsv41_bench/bench_decoder_layer.py
# ==============================================================================


class TestBenchDecoderLayer:
    """Comprehensive line and branch tests for bench_decoder_layer.py."""

    def test_bench_decoder_layer_package_branch(self) -> None:
        # Test execution when __package__ is a non-empty string
        code = compile(
            (BENCH_DIR / "bench_decoder_layer.py").read_text(encoding="utf-8"),
            str(BENCH_DIR / "bench_decoder_layer.py"),
            "exec",
        )
        globs = {
            "__name__": "bench_decoder_layer_test_pkg",
            "__file__": str(BENCH_DIR / "bench_decoder_layer.py"),
            "__package__": "dsv41_bench",
        }
        exec(code, globs)
        assert "compute_layer_roofline" in globs

    def test_compute_layer_roofline_errors(self) -> None:
        with pytest.raises(ValueError, match="Unsupported subop"):
            bench_decoder_layer.compute_layer_roofline("invalid_subop", num_tokens=256)

        with pytest.raises(ValueError, match="Unsupported mode"):
            bench_decoder_layer.compute_layer_roofline(
                "encoder_source", num_tokens=256, mode="invalid_mode"
            )

    def test_compute_layer_roofline_sp_and_ced(self) -> None:
        for subop in [
            "encoder_source",
            "encoder_reuse",
            "encoder_engram",
            "decoder_ced",
            "four_layer_e2e",
            "all",
        ]:
            for mode in ["tp", "sp"]:
                r = bench_decoder_layer.compute_layer_roofline(
                    subop, num_tokens=256, ced_decoder_tokens=64, tp_size=8, mode=mode
                )
                assert r["mode"] == mode
                assert r["lower_bound_ms"] > 0.0

    def test_run_decoder_layer_verification_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="Unsupported subop"):
            bench_decoder_layer.run_decoder_layer_verification(
                "invalid_subop", num_tokens=64
            )

        with pytest.raises(ValueError, match="divisible by tp_size"):
            bench_decoder_layer.run_decoder_layer_verification(
                "encoder_source", num_tokens=65, tp_size=8
            )

    def test_run_decoder_layer_verification_subops(self) -> None:
        # Test individual subops in run_decoder_layer_verification
        for subop in [
            "encoder_source",
            "encoder_reuse",
            "encoder_engram",
            "decoder_ced",
            "four_layer_e2e",
        ]:
            res = bench_decoder_layer.run_decoder_layer_verification(
                subop, num_tokens=64, tp_size=8, seed=42, device="cpu"
            )
            assert res["subop"] == subop
            assert len(res["comparisons"]) > 0
            for _name, comp in res["comparisons"].items():
                assert comp["ok"] is True

        # Also test subop="all" includes reconciliation_40_layer_decode
        res_all = bench_decoder_layer.run_decoder_layer_verification(
            "all", num_tokens=64, tp_size=8, seed=42, device="cpu"
        )
        assert "reconciliation_40_layer_decode" in res_all
        rec_dec = bench_decoder_layer.reconcile_40_layer_decode_step()
        assert rec_dec["decode_q1_T16"]["isolated_delta_ms"] < 0.0
        assert rec_dec["verify_q6_T128"]["sparse_mla_38L_delta_ms"] < 0.0

    def test_run_decoder_layer_verification_failure_raises(self) -> None:
        # Test that failed comparison triggers line 832: raise common.VerificationError
        bad_comp = {"ok": False, "max_abs_diff": 99.0}
        with (
            patch.object(common, "compare_tensors", return_value=bad_comp),
            pytest.raises(common.VerificationError),
        ):
            bench_decoder_layer.run_decoder_layer_verification(
                "encoder_source", num_tokens=64, tp_size=8, seed=42, device="cpu"
            )

    def test_worker(self) -> None:
        args = argparse.Namespace(
            subop="encoder_source",
            chunk_size=64,
            num_seqs=1,
            tp_size=8,
            seed=42,
            kv_len=1024,
            mode="tp",
        )
        ctx = SimpleNamespace(rank=0)
        with (
            patch.object(common, "init_worker", return_value=ctx),
            patch.object(common, "write_rank_result", create=True) as mock_write,
            patch.object(
                common, "summarize_samples", create=True, return_value={"median": 1.0}
            ),
            patch.object(common, "hbm_stats", return_value={}),
        ):
            rc = bench_decoder_layer._worker(args)
            assert rc == 0
            mock_write.assert_called_once()

    def test_main_cli_paths(self) -> None:
        # 1. cpu-only mode
        rc1 = bench_decoder_layer.main(
            ["--cpu-only", "--subop=encoder_source", "--chunk-size=64", "--tp-size=8"]
        )
        assert rc1 == 0

        # 2. worker mode
        with patch.object(
            bench_decoder_layer, "_worker", return_value=0
        ) as mock_worker:
            rc2 = bench_decoder_layer.main(
                ["--worker", "--subop=encoder_source", "--chunk-size=64", "--tp-size=8"]
            )
            assert rc2 == 0
            mock_worker.assert_called_once()

        # 3. parent mode
        with patch.object(
            common, "run_parent", create=True, return_value=0
        ) as mock_parent:
            rc3 = bench_decoder_layer.main(
                ["--subop=encoder_source", "--chunk-size=64", "--tp-size=8"]
            )
            assert rc3 == 0
            mock_parent.assert_called_once()

    def test_entrypoint_via_runpy(self) -> None:
        with (
            patch.object(bench_decoder_layer, "main", return_value=0),
            patch.object(
                sys,
                "argv",
                ["bench_decoder_layer.py", "--cpu-only", "--subop=encoder_source"],
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(
                    str(BENCH_DIR / "bench_decoder_layer.py"), run_name="__main__"
                )
            assert exc_info.value.code == 0
