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
"""Finding today's pipeline steps by the command they run."""

import pathlib

import pytest
from golden_capture import find_step
from golden_cases import Case, all_cases

PIPELINE = """\
env:
  ARTIFACTS_DIR: results
steps:
  - label: build
    key: build
    commands:
      - bash build.sh
  - group: outer
    steps:
      - label: eval qwen
        env:
          RUNS: 1
        commands:
          - |
            .buildkite/kubernetes/run.sh bash ./scripts/run_eval_flow.sh \\
              --config qwen-tp1 --run-lm-eval
      - group: inner
        steps:
          - label: mp qwen
            command: .buildkite/kubernetes/run_multihost.sh --backend mp qwen-tp1-mp
  - label: two commands
    commands:
      - .buildkite/kubernetes/run.sh first
      - second
  - label: replay
    command: .buildkite/kubernetes/run_multihost.sh --manifest pair.yaml
  - label: eval only
    env:
      EVAL_ONLY: "1"
    command: .buildkite/kubernetes/run_multihost.sh --manifest pair.yaml
"""


@pytest.fixture
def pipeline(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "pipeline.yml"
    path.write_text(PIPELINE)
    return path


def test_a_step_in_a_group_is_found_by_its_command(pipeline: pathlib.Path) -> None:
    step = find_step(pipeline, "run_eval_flow.sh --config qwen-tp1")
    assert step.command.startswith(".buildkite/kubernetes/run.sh bash")
    assert step.command.endswith("--config qwen-tp1 --run-lm-eval")
    assert step.env == {"RUNS": "1"}
    assert step.pipeline_env == {"ARTIFACTS_DIR": "results"}


def test_a_step_without_a_key_in_a_nested_group_is_found(
    pipeline: pathlib.Path,
) -> None:
    step = find_step(pipeline, "run_multihost.sh --backend mp qwen-tp1-mp")
    assert step.command == (
        ".buildkite/kubernetes/run_multihost.sh --backend mp qwen-tp1-mp"
    )
    assert step.env == {}


def test_words_match_whole(pipeline: pathlib.Path) -> None:
    with pytest.raises(LookupError, match="0 steps run `--config qwen`, not 1"):
        find_step(pipeline, "--config qwen")


def test_an_ambiguous_command_names_the_command_and_the_steps(
    pipeline: pathlib.Path,
) -> None:
    with pytest.raises(LookupError) as raised:
        find_step(pipeline, "run.sh")
    assert str(raised.value) == (
        "pipeline.yml: 2 steps run `run.sh`, not 1: eval qwen; two commands"
    )


def test_env_tells_apart_steps_that_run_the_same_command(
    pipeline: pathlib.Path,
) -> None:
    command = "run_multihost.sh --manifest pair.yaml"
    with pytest.raises(LookupError, match="2 steps run"):
        find_step(pipeline, command)
    assert find_step(pipeline, command, {"EVAL_ONLY": None}).env == {}
    assert find_step(pipeline, command, {"EVAL_ONLY": "1"}).env == {"EVAL_ONLY": "1"}
    with pytest.raises(LookupError, match="with EVAL_ONLY=2, not 1"):
        find_step(pipeline, command, {"EVAL_ONLY": "2"})


def test_a_step_needs_exactly_one_command(pipeline: pathlib.Path) -> None:
    with pytest.raises(LookupError, match="`second` has 2 commands, not 1"):
        find_step(pipeline, "second")


CASES = all_cases()


@pytest.mark.parametrize("which", CASES, ids=[c.id for c in CASES])
def test_every_case_finds_its_step(which: Case) -> None:
    which.launch()
