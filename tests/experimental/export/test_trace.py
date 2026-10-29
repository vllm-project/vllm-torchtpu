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

from __future__ import annotations

import re

import pytest

from vllm_torchtpu.experimental.export.trace import trace

# The line CDK's trace generator parses.
_EVENT = re.compile(
    r"^trace-(start|end)-event: name=(\S+) ts=(\d+\.\d+) pid=(\S+) tid=(\S+)$"
)


def _events(stderr: str) -> list[tuple[str, ...]]:
    lines = stderr.strip().splitlines()
    matches = [_EVENT.match(line) for line in lines]
    assert all(matches), lines
    return [m.groups() for m in matches if m]


def test_context_manager_logs_a_start_and_end_event(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "3")
    with trace("export.lower.forward_bucket16"):
        pass
    (start, end) = _events(capsys.readouterr().err)
    assert start[:2] == ("start", "export.lower.forward_bucket16")
    assert end[:2] == ("end", "export.lower.forward_bucket16")
    assert float(start[2]) <= float(end[2])
    assert start[3] == end[3] and "/export-rank3/" in start[3]
    assert start[4] == end[4]


def test_decorator_names_the_span_after_the_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @trace()
    def load_model() -> int:
        return 7

    assert load_model() == 7
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert names == [f"export.{load_model.__qualname__}"] * 2


def test_one_unnamed_decorator_names_each_function_it_wraps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deco = trace()

    @deco
    def first() -> None:
        pass

    @deco
    def second() -> None:
        pass

    first()
    second()
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert (
        names
        == [f"export.{first.__qualname__}"] * 2 + [f"export.{second.__qualname__}"] * 2
    )


def test_end_event_is_logged_when_the_block_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError), trace("export.export"):
        raise RuntimeError("boom")
    assert [e[0] for e in _events(capsys.readouterr().err)] == ["start", "end"]


def test_names_with_whitespace_are_rejected() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        trace("export load")


def test_saving_the_checkpoint_is_a_span(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_torchtpu.experimental.export import checkpoint

    saved = []
    monkeypatch.setattr(checkpoint, "abstract_params_inputs", lambda _: [])
    monkeypatch.setattr(checkpoint, "mesh_axis_sizes", lambda _: {})
    monkeypatch.setattr(checkpoint, "iter_checkpoint_shards", lambda *_: iter(()))
    monkeypatch.setattr(checkpoint, "save_checkpoint", lambda *a: saved.append(a))
    checkpoint.save_model_checkpoint({}, "artifact", "artifact/data")
    assert len(saved) == 1
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert names == ["export.save_checkpoint"] * 2


@pytest.mark.parametrize(
    ("load_format", "span"),
    [("auto", "export.load_safetensors"), ("dummy", "export.load_dummy_weights")],
)
def test_loading_the_weights_is_a_span(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    load_format: str,
    span: str,
) -> None:
    import types

    from vllm.model_executor import model_loader

    from vllm_torchtpu.experimental.export import universal_adapter

    class Loader:
        def load_model(self, *, vllm_config: object, model_config: object) -> str:
            return "model"

    monkeypatch.setattr(model_loader, "get_model_loader", lambda _: Loader())
    config = types.SimpleNamespace(
        load_config=types.SimpleNamespace(load_format=load_format), model_config=None
    )
    assert universal_adapter.run_model_loader(config) == "model"
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert names == [span] * 2
