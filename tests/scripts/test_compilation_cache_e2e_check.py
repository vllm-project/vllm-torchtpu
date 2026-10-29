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
"""Unit tests for scripts/vllm/integration/check_tier3_cache_consistency.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "integration"
    / "check_tier3_cache_consistency.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "check_tier3_cache_consistency", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


_COLD_MANIFEST = (
    "aac511fe0c7c7f94db48b9c6b80b6614  /tmp/cache/8ae7/749ade7f.bin\n"
    "ab05c68501cb1d47eaa09b1f70e1f1e0  /tmp/cache/8ae7/b5890755.bin\n"
)

_ADDED_EAGER_LINE = (
    "addbf302a8f3ba493d1033fe4fca0e30  "
    "/tmp/cache/8ae7/414a3bcdb281ad91_761b6dd1fdb0fa32_4519c327401d407a.bin\n"
)

_WARM_LOG_PRECOMPILE_ONLY = (
    "\x1b[0;36m(EngineCore pid=2005)\x1b[0;0m \x1b[32mINFO\x1b[0m "
    "\x1b[90m10-29 01:59:11\x1b[0m \x1b[90m[tpu_runner.py:5318]\x1b[0m "
    "Loading model from scratch...\n"
    "\x1b[0;36m(EngineCore pid=2005)\x1b[0;0m \x1b[32mINFO\x1b[0m "
    "\x1b[90m10-29 01:59:12\x1b[0m \x1b[90m[default_loader.py:501]\x1b[0m "
    "Loading weights took 0.74 seconds\n"
    "\x1b[0;36m(EngineCore pid=2005)\x1b[0;0m \x1b[32mINFO\x1b[0m "
    "\x1b[90m10-29 01:59:13\x1b[0m \x1b[90m[monitor.py:40]\x1b[0m "
    "[backbone] Starting torch.compile\n"
    "(EngineCore pid=2005) INFO 10-29 01:59:25 [tpu_runner.py:1559]   "
    "27 first read since the last check, 0.355s compiling\n"
    "(EngineCore pid=2005) INFO 10-29 01:59:25 [tpu_runner.py:1565]   "
    "graph 1/27: compilation_duration=0:00:00.355158 "
    "last_read=2026-10-29 01:59:11.499802 read_count=1\n"
    "(EngineCore pid=2005) INFO 10-29 01:59:25 [tpu_runner.py:1565]   "
    "graph 2/27: compilation_duration=0:00:00 "
    "last_read=2026-10-29 01:59:24.286428 read_count=1\n"
    "(EngineCore pid=2005) INFO 10-29 02:00:58 [tpu_runner.py:1559]   "
    "5 first read since the last check, 0.000s compiling\n"
    "(EngineCore pid=2005) INFO 10-29 02:00:58 [tpu_runner.py:1565]   "
    "graph 1/5: compilation_duration=0:00:00 "
    "last_read=2026-10-29 02:00:57.100000 read_count=1\n"
)


class Tier3CacheConsistencyTest(unittest.TestCase):
    def test_exact_match_passes_without_log_inspection(self) -> None:
        ok, msg = _MOD.verify_tier3_consistency(_COLD_MANIFEST, _COLD_MANIFEST, "")
        self.assertTrue(ok, msg)
        self.assertIn("100% identical", msg)

    def test_empty_cold_manifest_fails(self) -> None:
        ok, msg = _MOD.verify_tier3_consistency("", _COLD_MANIFEST, "")
        self.assertFalse(ok)
        self.assertIn("No Tier-3 binary cache files", msg)

    def test_missing_cold_file_fails(self) -> None:
        first_only = _COLD_MANIFEST.splitlines(keepends=True)[0]
        ok, msg = _MOD.verify_tier3_consistency(
            _COLD_MANIFEST, first_only, _WARM_LOG_PRECOMPILE_ONLY
        )
        self.assertFalse(ok)
        self.assertIn("missing after Warm run", msg)

    def test_mutated_cold_md5_fails(self) -> None:
        mutated = _COLD_MANIFEST.replace(
            "aac511fe0c7c7f94db48b9c6b80b6614",
            "ffffffffffffffffffffffffffffffff",
        )
        ok, msg = _MOD.verify_tier3_consistency(
            _COLD_MANIFEST, mutated, _WARM_LOG_PRECOMPILE_ONLY
        )
        self.assertFalse(ok)
        self.assertIn("changed MD5", msg)

    def test_additive_precompile_eager_bin_with_ansi_log_passes(self) -> None:
        warm = _COLD_MANIFEST + _ADDED_EAGER_LINE
        ok, msg = _MOD.verify_tier3_consistency(
            _COLD_MANIFEST, warm, _WARM_LOG_PRECOMPILE_ONLY
        )
        self.assertTrue(ok, msg)
        self.assertIn("pre-torch.compile eager weight-load ops", msg)

    def test_post_compile_recompilation_fails(self) -> None:
        warm = _COLD_MANIFEST + _ADDED_EAGER_LINE
        bad_log = _WARM_LOG_PRECOMPILE_ONLY + (
            "(EngineCore pid=2005) INFO 10-29 02:00:59 [tpu_runner.py:1559]   "
            "1 first read since the last check, 1.923s compiling\n"
            "(EngineCore pid=2005) INFO 10-29 02:00:59 [tpu_runner.py:1565]   "
            "graph 1/1: compilation_duration=0:00:01.922874 "
            "last_read=2026-10-29 02:00:58.310903 read_count=1\n"
        )
        ok, msg = _MOD.verify_tier3_consistency(_COLD_MANIFEST, warm, bad_log)
        self.assertFalse(ok)
        self.assertIn("at or after torch.compile start", msg)

    def test_more_added_bins_than_logged_precompile_graphs_fails(self) -> None:
        second_extra = "11111111111111111111111111111111  /tmp/cache/8ae7/extra2.bin\n"
        warm = _COLD_MANIFEST + _ADDED_EAGER_LINE + second_extra
        ok, msg = _MOD.verify_tier3_consistency(
            _COLD_MANIFEST, warm, _WARM_LOG_PRECOMPILE_ONLY
        )
        self.assertFalse(ok)
        self.assertIn("exceeding 1 logged pre-compile eager graph", msg)

    def test_unaccounted_summary_compiling_seconds_fails(self) -> None:
        warm = _COLD_MANIFEST + _ADDED_EAGER_LINE
        # Hidden 21st+ graph in summary with non-zero compiling time not in top-20 lines
        hidden_recomp_log = _WARM_LOG_PRECOMPILE_ONLY.replace(
            "5 first read since the last check, 0.000s compiling",
            "25 first read since the last check, 1.250s compiling",
        )
        ok, msg = _MOD.verify_tier3_consistency(_COLD_MANIFEST, warm, hidden_recomp_log)
        self.assertFalse(ok)
        self.assertIn("unaccounted", msg)

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cold_f = tmp_path / "tier3_cold.txt"
            warm_f = tmp_path / "tier3_warm.txt"
            log_f = tmp_path / "warm_run.log"
            cold_f.write_text(_COLD_MANIFEST, encoding="utf-8")
            warm_f.write_text(_COLD_MANIFEST + _ADDED_EAGER_LINE, encoding="utf-8")
            log_f.write_text(_WARM_LOG_PRECOMPILE_ONLY, encoding="utf-8")

            rc = _MOD.main(
                [
                    "--cold-md5",
                    str(cold_f),
                    "--warm-md5",
                    str(warm_f),
                    "--warm-log",
                    str(log_f),
                ]
            )
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
