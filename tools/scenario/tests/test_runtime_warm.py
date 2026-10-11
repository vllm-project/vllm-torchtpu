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
"""The remote-code warm, run as P-19 runs it, against stand-in modules."""

import pathlib
import subprocess
import sys

from plan_fixtures import Data, build
from runtime_helpers import records, stub_env
from scenario.plan import plumbing
from scenario.runtime import warm

URI = "gs://bucket/kimi/k3"


def run_warm(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", warm.__name__, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_warm_pulls_then_imports_from_the_pulled_dir(
    tmp_path: pathlib.Path,
) -> None:
    env = stub_env(tmp_path)
    done = run_warm(env, URI)
    assert done.returncode == 0, done.stderr
    pulled = env["STUB_PULLED_DIR"]
    assert done.stdout == f"remote code for {URI} imported from {pulled}\n"
    calls = [
        (r["program"], r["argv"], r.get("trust_remote_code")) for r in records(env)
    ]
    assert calls == [
        ("vllm_torchtpu runai_repull", [URI], None),
        ("vllm runai pull", [URI, URI], None),
        ("transformers AutoConfig", [pulled], True),
        ("transformers AutoTokenizer", [pulled], True),
    ]


def test_a_failed_import_fails_the_warm(tmp_path: pathlib.Path) -> None:
    done = run_warm(stub_env(tmp_path, STUB_WARM_FAIL="1"), URI)
    assert done.returncode == 1
    assert "has no attribute 'TikTokenTokenizer'" in done.stderr


def test_the_warm_takes_one_uri(tmp_path: pathlib.Path) -> None:
    done = run_warm(stub_env(tmp_path))
    assert done.returncode == 2
    assert done.stderr == warm.USAGE + "\n"
    assert records(stub_env(tmp_path)) == []


def test_importing_the_warm_imports_no_vllm(tmp_path: pathlib.Path) -> None:
    probe = (
        "import sys, scenario.runtime.warm; "
        "print(sorted({'vllm', 'transformers', 'vllm_torchtpu'} & set(sys.modules)))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        env=stub_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "[]\n"


def test_p19_runs_this_module(variants: dict[str, Data]) -> None:
    (host_0, host_1) = build(variants["mp"]).roles[0].host_plans
    assert host_0.pre == ()
    (command,) = host_1.pre
    assert command.argv[:3] == ("python3", "-m", warm.__name__)
    assert warm.__name__ == plumbing.WARM_MODULE
