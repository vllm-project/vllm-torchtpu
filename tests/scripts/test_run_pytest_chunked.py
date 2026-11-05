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
"""Unit tests for .buildkite/scripts/run_pytest_chunked.py."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "scripts"
    / "run_pytest_chunked.py"
)
_SPEC = importlib.util.spec_from_file_location("run_pytest_chunked", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_pytest_chunked = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_pytest_chunked)


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_classify_and_plan_chunks_tpu_spawn_first_and_negative_controls(
    tmp_path: Path,
) -> None:
    run_pytest_chunked._analyze_test_file.cache_clear()

    # Positive controls for TPU spawn isolation.
    entrypoint_test = _write_file(
        tmp_path / "tests" / "entrypoints" / "llm" / "test_generate_tp.py",
        "def test_ok():\n    assert True\n",
    )
    marker_spawn_test = _write_file(
        tmp_path / "tests" / "models" / "vllm" / "test_ple_multi.py",
        "import pytest\npytestmark = [pytest.mark.spawns_tpu_workers]\n",
    )
    torchrun_test = _write_file(
        tmp_path / "tests" / "layers" / "test_torchrun.py",
        "import sys\nCMD = [sys.executable, '-m', 'torch.distributed.run']\n",
    )

    # Negative control: comments, docstring, and plain string mentioning spawn/pallas.
    lookalike_batch = _write_file(
        tmp_path / "tests" / "layers" / "test_lookalike_plain.py",
        "# spawns_tpu_workers prepare_tpu_environment torch.distributed.run pallas\n"
        '"""Mentions spawns_tpu_workers, pooled_qsa, and torch.distributed.run."""\n'
        'MSG = "spawns_tpu_workers torch.distributed.run pallas"\n'
        "def test_plain():\n    assert MSG\n",
    )
    batch_b = _write_file(
        tmp_path / "tests" / "models" / "test_batch_b.py",
        "def test_b():\n    assert True\n",
    )
    batch_a = _write_file(
        tmp_path / "tests" / "kernels" / "test_batch_a.py",
        "def test_a():\n    assert True\n",
    )

    selectors = [
        str(batch_b),
        str(lookalike_batch),
        str(marker_spawn_test),
        str(batch_a),
        str(entrypoint_test),
        str(torchrun_test),
    ]
    chunks = run_pytest_chunked.plan_chunks(
        selectors, chunk_size=2, has_vfio=True, repo_root=tmp_path
    )

    # All 3 spawn_isolated files come first in 1-file chunks (sorted by rel path),
    # while lookalike_batch stays in batch chunks after all spawn files!
    assert chunks == [
        [str(entrypoint_test)],
        [str(torchrun_test)],
        [str(marker_spawn_test)],
        [str(batch_a), str(lookalike_batch)],
        [str(batch_b)],
    ]


def test_classify_and_plan_chunks_cpu_heavy_isolated_and_negative_control(
    tmp_path: Path,
) -> None:
    run_pytest_chunked._analyze_test_file.cache_clear()

    heavy_pallas = _write_file(
        tmp_path / "tests" / "kernels" / "test_heavy_pallas.py",
        "from jax.experimental import pallas as pl\n",
    )
    heavy_qsa = _write_file(
        tmp_path / "tests" / "kernels" / "test_heavy_qsa.py",
        "from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa\n",
    )
    heavy_layer_mla = _write_file(
        tmp_path / "tests" / "layers" / "core" / "test_sparse_mla_attention.py",
        "from vllm_torchtpu.layers.core import sparse_mla_attention\n",
    )
    heavy_megakernel = _write_file(
        tmp_path / "tests" / "megakernel" / "test_b1_tiles_profiles.py",
        "from vllm_torchtpu import megakernel as mk\n"
        "def run(w):\n    return mk.PallasWriter().make_step()\n",
    )
    lookalike_megakernel = _write_file(
        tmp_path / "tests" / "megakernel" / "test_decls.py",
        '# make_step PallasWriter pallas\n"""Docstring mentioning make_step."""\n'
        'TXT = "make_step PallasWriter sparse_mla"\n'
        "def test_decls():\n    assert TXT\n",
    )
    lookalike_kernel = _write_file(
        tmp_path / "tests" / "kernels" / "test_light_kernel.py",
        '# pallas pooled_qsa streamindex\n"""Docstring mentioning pallas."""\n'
        "def test_fast():\n    assert 1 == 1\n",
    )
    other_batch = _write_file(
        tmp_path / "tests" / "runner" / "test_runner.py",
        "def test_runner():\n    assert True\n",
    )

    chunks = run_pytest_chunked.plan_chunks(
        [
            str(other_batch),
            str(heavy_megakernel),
            str(lookalike_megakernel),
            str(heavy_qsa),
            str(lookalike_kernel),
            str(heavy_layer_mla),
            str(heavy_pallas),
        ],
        chunk_size=6,
        has_vfio=False,
        repo_root=tmp_path,
    )
    assert chunks == [
        [str(heavy_pallas)],
        [str(heavy_qsa)],
        [str(heavy_layer_mla)],
        [str(heavy_megakernel)],
        [str(lookalike_kernel), str(lookalike_megakernel), str(other_batch)],
    ]

    # When split-by-example nodeids are passed with isolated_chunk_size=2 and
    # chunk_size=4, CPU-heavy nodeids stay grouped per file in <=2-example
    # chunks while lightweight batch nodeids pack up to 4 per chunk.
    split_chunks = run_pytest_chunked.plan_chunks(
        [
            f"{other_batch}::test_r1",
            f"{heavy_pallas}::test_p1",
            f"{heavy_pallas}::test_p2",
            f"{heavy_pallas}::test_p3",
            f"{heavy_qsa}::test_q1",
            f"{lookalike_kernel}::test_l1",
            f"{lookalike_kernel}::test_l2",
            f"{lookalike_megakernel}::test_m1",
            f"{other_batch}::test_r2",
        ],
        chunk_size=4,
        isolated_chunk_size=2,
        has_vfio=False,
        repo_root=tmp_path,
    )
    assert split_chunks == [
        [f"{heavy_pallas}::test_p1", f"{heavy_pallas}::test_p2"],
        [f"{heavy_pallas}::test_p3"],
        [f"{heavy_qsa}::test_q1"],
        [
            f"{lookalike_kernel}::test_l1",
            f"{lookalike_kernel}::test_l2",
            f"{lookalike_megakernel}::test_m1",
            f"{other_batch}::test_r1",
        ],
        [f"{other_batch}::test_r2"],
    ]

    with pytest.raises(
        ValueError, match="--isolated-chunk-size must be a positive integer"
    ):
        run_pytest_chunked.plan_chunks(
            [str(other_batch)],
            chunk_size=4,
            isolated_chunk_size=0,
            has_vfio=False,
            repo_root=tmp_path,
        )


def test_write_merged_junit_xml_crash_repeat_and_concurrent_paths(
    tmp_path: Path,
) -> None:
    out_xml = tmp_path / "out" / "merged.xml"
    c1 = _write_file(
        tmp_path / "c1.xml",
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="s1" tests="2"/></testsuites>',
    )
    c2 = _write_file(
        tmp_path / "c2.xml",
        '<?xml version="1.0" encoding="utf-8"?><testsuite name="s2" tests="1"/>',
    )
    corrupt = _write_file(tmp_path / "corrupt.xml", "<testsuites><truncated")

    # Initial write with 2 valid + 1 truncated chunk + 1 missing chunk.
    run_pytest_chunked._write_merged_junit_xml(
        [str(c1), str(corrupt), str(tmp_path / "missing.xml"), str(c2)],
        str(out_xml),
    )
    assert os.listdir(out_xml.parent) == ["merged.xml"]
    suites = ET.parse(out_xml).getroot().findall("testsuite")
    assert [s.attrib["name"] for s in suites] == ["s1", "s2"]

    # Crash path: failure during os.fsync preserves prior valid merged.xml and leaves
    # zero .tmp files behind.
    with (
        mock.patch.object(run_pytest_chunked.os, "fsync", side_effect=OSError("disk")),
        pytest.raises(OSError, match="disk"),
    ):
        run_pytest_chunked._write_merged_junit_xml([str(c2)], str(out_xml))
    assert os.listdir(out_xml.parent) == ["merged.xml"]
    suites_after_crash = ET.parse(out_xml).getroot().findall("testsuite")
    assert [s.attrib["name"] for s in suites_after_crash] == ["s1", "s2"]

    # Repeat path: second run with fewer chunks replaces output cleanly (no stale s1).
    run_pytest_chunked._write_merged_junit_xml([str(c2)], str(out_xml))
    suites_after_repeat = ET.parse(out_xml).getroot().findall("testsuite")
    assert [s.attrib["name"] for s in suites_after_repeat] == ["s2"]

    # Concurrent path: 8 threads + 4 subprocesses writing concurrently to out_xml.
    def _thread_writer(idx: int) -> None:
        src = str(c1) if idx % 2 == 0 else str(c2)
        run_pytest_chunked._write_merged_junit_xml([src], str(out_xml))

    proc_code = (
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('m', {str(_SCRIPT_PATH)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"m._write_merged_junit_xml([{str(c1)!r}, {str(c2)!r}], {str(out_xml)!r})"
    )
    procs = [subprocess.Popen([sys.executable, "-c", proc_code]) for _ in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_thread_writer, i) for i in range(16)]
        for fut in futs:
            fut.result()
    for p in procs:
        assert p.wait(timeout=30) == 0

    assert os.listdir(out_xml.parent) == ["merged.xml"]
    parsed_final = ET.parse(out_xml).getroot().findall("testsuite")
    assert len(parsed_final) in (1, 2)


def test_cli_flag_orders_exit_codes_and_validation(tmp_path: Path) -> None:
    pass_test = _write_file(
        tmp_path / "test_ok.py",
        "import pytest\n\n"
        "@pytest.mark.slow\ndef test_skipped_by_mark():\n    assert False\n\n"
        "def test_passes():\n    assert 2 + 2 == 4\n",
    )
    xml_order_a = tmp_path / "order_a.xml"
    xml_order_b = tmp_path / "order_b.xml"

    # Flag order A (standard pipeline_tests.yml order with -- separator).
    proc_a = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--isolated-chunk-size",
            "4",
            "--chunk-size",
            "64",
            "--timeout",
            "60",
            "--mark",
            "not slow",
            f"--junit-xml={xml_order_a}",
            "--",
            str(pass_test),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_a.returncode == 0, proc_a.stdout + proc_a.stderr
    assert len(ET.parse(xml_order_a).getroot().findall("testsuite")) == 1

    # Flag order B (reversed flag order, positional selector interleaved).
    proc_b = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            str(pass_test),
            f"--junit-xml={xml_order_b}",
            "--mark",
            "not slow",
            "--timeout",
            "60",
            "--chunk-size",
            "1",
            "--isolated-chunk-size",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_b.returncode == 0, proc_b.stdout + proc_b.stderr
    assert len(ET.parse(xml_order_b).getroot().findall("testsuite")) == 1

    # Empty selector list -> exit code 5 and valid empty <testsuites/>.
    empty_xml = tmp_path / "empty.xml"
    assert run_pytest_chunked.main([f"--junit-xml={empty_xml}"]) == 5
    assert ET.parse(empty_xml).getroot().findall("testsuite") == []

    # Invalid --chunk-size / --isolated-chunk-size -> actionable ValueError.
    with pytest.raises(ValueError, match="--chunk-size must be a positive integer"):
        run_pytest_chunked.main(["--chunk-size", "0", str(pass_test)])
    with pytest.raises(
        ValueError, match="--isolated-chunk-size must be a positive integer"
    ):
        run_pytest_chunked.main(["--isolated-chunk-size", "-1", str(pass_test)])


def test_whole_file_cpu_heavy_ast_subfile_expansion_and_negative_controls(
    tmp_path: Path,
) -> None:
    run_pytest_chunked._analyze_test_file.cache_clear()
    run_pytest_chunked._extract_pytest_targets.cache_clear()

    # Positive control: whole-file heavy module with 3 top-level test_* defs,
    # 1 parameterized-style TestCase class, 1 private helper class, and 1
    # class with __init__ (ignored by pytest).
    heavy_multi = _write_file(
        tmp_path / "tests" / "megakernel" / "test_m3_like.py",
        "from jax.experimental import pallas as pl\n"
        "class _PrivateHelper:\n"
        "    def test_ignored(self): pass\n"
        "class HelperWithInit:\n"
        "    def __init__(self): pass\n"
        "    def test_ignored(self): pass\n"
        "def test_alpha(): assert pl is not None\n"
        "async def test_beta(): assert True\n"
        "class SparseMlaCaseTest:\n"
        "    def test_case_one(self): assert True\n"
        "def test_gamma(): assert True\n",
    )
    # Negative control 1: same-file class inheritance keeps whole-file selector
    # so abstract base classes are never selected directly.
    heavy_inherited = _write_file(
        tmp_path / "tests" / "kernels" / "test_inherited_base.py",
        "from jax.experimental import pallas as pl\n"
        "class BaseCase:\n"
        "    def test_shared(self): assert pl is not None\n"
        "class ChildCaseTest(BaseCase):\n"
        "    def test_child(self): assert True\n",
    )
    # Negative control 2: single-test heavy file stays whole-file selector.
    heavy_single = _write_file(
        tmp_path / "tests" / "kernels" / "test_single_heavy.py",
        "from jax.experimental import pallas as pl\n"
        "def test_only(): assert pl is not None\n",
    )
    # Negative control 3: non-heavy batch file with multiple test_* functions
    # is NOT sub-file split (avoids extra subprocess overhead).
    light_multi = _write_file(
        tmp_path / "tests" / "runner" / "test_light_multi.py",
        "def test_one(): assert True\ndef test_two(): assert True\n",
    )

    chunks = run_pytest_chunked.plan_chunks(
        [
            str(light_multi),
            str(heavy_multi),
            str(heavy_inherited),
            str(heavy_single),
        ],
        chunk_size=8,
        isolated_chunk_size=2,
        has_vfio=False,
        repo_root=tmp_path,
    )
    assert chunks == [
        [str(heavy_inherited)],
        [str(heavy_single)],
        [f"{heavy_multi}::test_alpha", f"{heavy_multi}::test_beta"],
        [f"{heavy_multi}::SparseMlaCaseTest", f"{heavy_multi}::test_gamma"],
        [str(light_multi)],
    ]


def test_signal_exit_normalization_and_vfio_settle_transitions(
    tmp_path: Path,
) -> None:
    f1 = _write_file(tmp_path / "tests" / "a.py", "def test_1(): assert True\n")
    f2 = _write_file(tmp_path / "tests" / "b.py", "def test_2(): assert True\n")
    f3 = _write_file(tmp_path / "tests" / "c.py", "def test_3(): assert True\n")

    recorded_cmds: list[list[str]] = []
    returns = iter(
        [5, -9, 0]
    )  # chunk 1 empty (rc=5), chunk 2 SIGKILL (-9 -> 137), chunk 3 ok

    class _FakeProc:
        pid = 999999

        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            recorded_cmds.append(cmd)
            self._rc = next(returns)

        def wait(self) -> int:
            return self._rc

    with (
        mock.patch.object(
            run_pytest_chunked, "has_tpu_vfio_devices", return_value=True
        ),
        mock.patch.object(
            run_pytest_chunked.importlib.util, "find_spec", return_value=object()
        ),
        mock.patch.object(
            run_pytest_chunked.subprocess, "Popen", side_effect=_FakeProc
        ),
        mock.patch.object(run_pytest_chunked, "wait_for_vfio_idle") as mock_wait,
        mock.patch.object(run_pytest_chunked.os, "killpg"),
    ):
        rc = run_pytest_chunked.main(
            ["--chunk-size", "1", "--timeout", "300", str(f1), str(f2), str(f3)]
        )

    assert rc == 137
    assert all("--timeout=300" in cmd for cmd in recorded_cmds)
    # Chunk 1 returned rc=5 (no tests collected) so VFIO settle skipped;
    # Chunk 2 returned -9 (active failure before final chunk) so VFIO settle ran once;
    # Chunk 3 was final chunk so no trailing settle.
    assert mock_wait.call_count == 1
