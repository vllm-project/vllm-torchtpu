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
"""validate, show and run --dry-run: output snapshots and exit codes.

Set SCENARIO_UPDATE_SNAPSHOTS=1 to rewrite the snapshots after an intended
change, and review the diff.
"""

import os
import pathlib
import shutil

import pytest
from click.testing import CliRunner, Result
from plan_fixtures import Data, build
from scenario import __version__
from scenario.cli import main
from scenario.plan import codec
from scenario.plan.build import plan, select_point
from scenario.resolve import resolve
from scenario.show import describe, render_plan

TESTS = pathlib.Path(__file__).resolve().parent
REPO = TESTS / "data" / "repo"
SNAPSHOTS = TESTS / "snapshots"

GLM = "scenarios/glm-5.2/fp8-dp16-ep.yaml"
GLM_MP = "scenarios/glm-5.2/fp8-dp16-ep-mp.yaml"
QWEN_PD = "scenarios/qwen3.5-397b/fp8-pcp8-dp8-1p1d.yaml"
K3_PD = "scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml"
AGENTX = "scenarios/kimi-k3/agentx.yaml"
INFERENCEX = "scenarios/qwen3.5-397b/inferencex-sweep.yaml"


def invoke(*args: str, stdin: str | None = None) -> Result:
    return CliRunner().invoke(main, list(args), input=stdin)


def assert_snapshot(name: str, text: str) -> None:
    text = text.replace(f"tool {__version__}", "tool <version>")
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("SCENARIO_UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    assert path.is_file(), (
        f"no snapshot {path.name}; run with SCENARIO_UPDATE_SNAPSHOTS=1"
    )
    assert text == path.read_text()


@pytest.fixture
def in_repo(monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(REPO)
    return REPO


@pytest.fixture
def scratch(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A writable copy of the fixture repo, as the current directory."""
    root = tmp_path / "repo"
    shutil.copytree(REPO, root)
    monkeypatch.chdir(root)
    return root


def test_validate_everything(in_repo: pathlib.Path) -> None:
    result = invoke("validate")
    assert result.exit_code == 0, result.output
    assert_snapshot("validate", result.stdout)


def test_validate_from_a_subdirectory(
    in_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO / "scenarios" / "kimi-k3")
    result = invoke("validate")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == "7 scenarios, 0 errors"


def test_validate_files_and_set(in_repo: pathlib.Path) -> None:
    result = invoke("validate", GLM, K3_PD, "--set", "server.args.max-num-seqs=4")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == "2 scenarios, 0 errors"


def test_validate_reports_every_problem(scratch: pathlib.Path) -> None:
    glm = scratch / GLM
    glm.write_text(
        glm.read_text()
        .replace("chips_per_host: 4", "chips_per_host: 2")
        .replace("multihost: ray", "multihostt: ray")
    )
    copy = scratch / "scenarios" / "glm-5.2" / "copy.yaml"
    copy.write_text((scratch / GLM_MP).read_text())
    result = invoke("validate")
    assert result.exit_code == 1
    assert_snapshot("validate_errors", result.stdout)


def test_validate_usage(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = invoke("validate")
    assert result.exit_code == 2
    assert "no scenarios/defaults.yaml" in result.stderr
    assert invoke("validate", "nope.yaml").exit_code == 2


def test_show_resolved(in_repo: pathlib.Path) -> None:
    result = invoke("show", K3_PD)
    assert result.exit_code == 0, result.output
    assert_snapshot("show_k3_pd", result.stdout)


def test_show_why(in_repo: pathlib.Path) -> None:
    result = invoke(
        "show",
        K3_PD,
        "--why",
        "server.args.tensor-parallel-size",
        "--set",
        "server.args.tensor-parallel-size=8",
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "server.args.tensor-parallel-size = 8\n"
        "  set by   --set[0] server.args.tensor-parallel-size=8  (set:0)\n"
        "  over     scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml:14  (scenario)\n"
        "  over     scenarios/kimi-k3/tp32-ep.yaml:16  (extends:kimi-k3/tp32-ep)\n"
        "  over     scenarios/defaults.yaml:6  (defaults)\n"
    )
    deleted = invoke("show", K3_PD, "--why", "server.hosts")
    assert deleted.stdout == (
        "server.hosts: deleted by scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml:9  "
        "(scenario)\n"
    )
    default = invoke("show", GLM, "--why", "tasks.perf.client")
    assert default.stdout == "tasks.perf.client = vllm-bench  (tool default)\n"
    typo = invoke("show", GLM, "--why", "server.argz")
    assert typo.exit_code == 1
    assert "did you mean server.args?" in typo.stderr


def test_show_why_per_point(in_repo: pathlib.Path) -> None:
    result = invoke("show", AGENTX, "--why", "tasks.agentx.concurrency")
    assert result.exit_code == 0, result.output
    assert result.stdout.count("point c") == 4
    assert result.stdout.endswith(
        "tasks.agentx.concurrency = 8\n"
        "  set by   scenarios/kimi-k3/agentx.yaml:24  (sweep:c8)\n"
        "  over     scenarios/kimi-k3/agentx.yaml:16  (scenario)\n"
    )


def test_show_plan_multi_host(in_repo: pathlib.Path) -> None:
    result = invoke("show", GLM_MP, "--plan")
    assert result.exit_code == 0, result.output
    assert "ready: GET :8000/health within 2h10m" in result.stdout
    assert_snapshot("plan_glm_mp", result.stdout)


def test_show_plan_pd(in_repo: pathlib.Path) -> None:
    result = invoke("show", QWEN_PD, "--plan")
    assert result.exit_code == 0, result.output
    assert_snapshot("plan_qwen_pd", result.stdout)


def test_show_plan_custom_and_eval(in_repo: pathlib.Path) -> None:
    result = invoke("show", K3_PD, "--plan")
    assert result.exit_code == 0, result.output
    # Readiness is startup_timeout (4h) + 600 s, the one value the runtime uses.
    assert "within 4h10m" in result.stdout
    assert_snapshot("plan_k3_pd", result.stdout)


def test_show_plan_clients(in_repo: pathlib.Path) -> None:
    agentx = invoke("show", AGENTX, "--plan", "--point", "c4")
    assert agentx.exit_code == 0, agentx.output
    assert_snapshot("plan_agentx_c4", agentx.stdout)
    inferencex = invoke("show", INFERENCEX, "--plan", "--point", "dp8-c256")
    assert inferencex.exit_code == 0, inferencex.output
    assert_snapshot("plan_inferencex_dp8", inferencex.stdout)


def test_show_plan_inferencex_eval(variants: dict[str, Data]) -> None:
    """InferenceX's eval client, with its env and the score floor that gates it."""
    planned = build(variants["ix_eval"])
    assert_snapshot("plan_ix_eval", render_plan(planned))
    assert describe(planned) == (
        "prefill tpu7x 2x2x1, decode tpu7x 2x2x1, router, driver client; "
        "eval inferencex lm-eval c4"
    )


def test_show_plan_needs_a_point(in_repo: pathlib.Path) -> None:
    result = invoke("show", AGENTX, "--plan")
    assert result.exit_code == 2
    assert result.stderr == (
        "error: kimi-k3/agentx is a sweep of 4 points, each its own workload; "
        "pick one with --point: c1, c2, c4, c8\n"
    )
    assert invoke("show", AGENTX, "--plan", "--point", "c3").exit_code == 1


def test_show_resolved_sweep(in_repo: pathlib.Path) -> None:
    result = invoke("show", INFERENCEX)
    assert result.exit_code == 0, result.output
    assert result.stdout.count("point: ") == 2
    assert "sweep:dp8-c256" in result.stdout


def test_show_encoded(in_repo: pathlib.Path) -> None:
    planned = plan(select_point(resolve(REPO / K3_PD)))
    encoded = codec.encode(planned)
    direct = invoke("show", K3_PD, "--plan")
    assert invoke("show", "--encoded", encoded).stdout == direct.stdout
    assert (
        invoke("show", "--encoded", "-", stdin=encoded + "\n").stdout == direct.stdout
    )
    bad = invoke("show", "--encoded", "abc")
    assert bad.exit_code == 2 and "not an encoded Plan" in bad.stderr
    assert invoke("show", K3_PD, "--encoded", encoded).exit_code == 2


def test_show_usage(in_repo: pathlib.Path) -> None:
    assert invoke("show").exit_code == 2
    assert invoke("show", GLM, "--plan", "--why", "server").exit_code == 2
    broken = invoke("show", GLM, "--set", "server.argz.x=1")
    assert broken.exit_code == 1
    assert broken.stderr.startswith(
        "--set[0] server.argz.x=1: server.argz: unknown key"
    )


def test_run_dry_run_one_host(in_repo: pathlib.Path) -> None:
    result = invoke("run", K3_PD, "--dry-run", "--role", "decode", "--host-index", "1")
    assert result.exit_code == 0, result.output
    assert "role prefill" not in result.stdout and "driver:" not in result.stdout
    assert_snapshot("run_k3_pd_decode_1", result.stdout)


def test_run_dry_run_every_host(in_repo: pathlib.Path) -> None:
    result = invoke("run", K3_PD, "--dry-run")
    assert result.exit_code == 0, result.output
    assert result.stdout == invoke("show", K3_PD, "--plan").stdout
    driver = invoke("run", K3_PD, "--dry-run", "--role", "prefill", "--host-index", "0")
    assert "driver: prefill host 0" in driver.stdout
    assert "role decode" not in driver.stdout
    client = invoke("run", QWEN_PD, "--dry-run", "--role", "client")
    assert client.stdout.count("role ") == 0 and "driver: a chipless" in client.stdout
    one_role = invoke("run", GLM, "--dry-run", "--host-index", "1")
    assert "host 1" in one_role.stdout and "host 0" not in one_role.stdout


def test_run_dry_run_only(in_repo: pathlib.Path) -> None:
    result = invoke("run", K3_PD, "--dry-run", "--only", "eval")
    assert result.exit_code == 0, result.output
    assert "custom correctness" not in result.stdout and "only: eval" in result.stdout
    unknown = invoke("run", K3_PD, "--dry-run", "--only", "perf")
    assert unknown.exit_code == 2
    assert "tasks: correctness, eval" in unknown.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((K3_PD,), "pass --dry-run"),
        (
            (K3_PD, "--dry-run", "--role", "nope"),
            "no role nope; roles: prefill, decode",
        ),
        ((K3_PD, "--dry-run", "--host-index", "1"), "--host-index needs --role"),
        ((K3_PD, "--dry-run", "--role", "decode", "--host-index", "4"), "hosts 0..3"),
        ((AGENTX, "--dry-run"), "pick one with --point"),
    ],
)
def test_run_usage(in_repo: pathlib.Path, args: tuple[str, ...], message: str) -> None:
    result = invoke("run", *args)
    assert result.exit_code == 2
    assert message in result.stderr
