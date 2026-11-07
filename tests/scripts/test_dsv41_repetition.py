# SPDX-License-Identifier: Apache-2.0
"""CPU tests for bounded replay, loop detection and failure evidence."""

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps")
)
import check_repetition as check  # noqa: E402

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Let's run We'll need to compile Let's check " * 16, "words"),
        ("Then you have got more " * 16, "words"),
        ("AgainAnd" * 100 + " end", "characters"),
        ((" ".join(f"word{i}" for i in range(176)) + " ") * 8, "words"),
        (",..,," * 18 + " text", "punctuation"),
    ],
)
def test_observed_loop_families_are_detected(text, kind):
    assert check.repetition(text)["kind"] == kind


@pytest.mark.parametrize(
    "text",
    [
        "Let's inspect the implementation, run the focused test, and read its output.",
        "Then you have got more " * 3,
        ".... ... ,.,.,. --- (()) [[], []]",
    ],
)
def test_short_repetitions_are_not_sustained_loops(text):
    assert check.repetition(text) is None


def event(delta, finish=None):
    return json.dumps(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    )


def test_fragmented_tools_and_reasoning_are_validated_without_execution():
    state = check.Stream()
    state.consume(event({"reasoning": "Inspect the implementation."}))
    state.consume(
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "',
                        },
                    }
                ]
            }
        )
    )
    state.consume(
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": "AgainAnd" * 100 + '"}',
                        },
                    }
                ]
            },
            "tool_calls",
        )
    )
    state.consume("[DONE]")
    state.validate()
    assert state.candidate is None
    assert state.tools[0]["name"] == "terminal"


@pytest.mark.parametrize(
    "mode", ["missing_done", "malformed_tool", "done_only", "after_done"]
)
def test_protocol_failures_cannot_pass(mode):
    state = check.Stream()
    with pytest.raises(ValueError):
        if mode == "done_only":
            state.consume("[DONE]")
        elif mode == "malformed_tool":
            state.consume(
                event(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "terminal",
                                    "arguments": "{incomplete",
                                },
                            }
                        ]
                    },
                    "tool_calls",
                )
            )
            state.consume("[DONE]")
            state.validate()
        else:
            state.consume(event({"content": "some output"}, "stop"))
            if mode == "after_done":
                state.consume("[DONE]")
                state.consume(event({"content": "extra"}))
            state.validate()


def test_length_finish_is_only_a_bounded_observation():
    state = check.Stream()
    state.consume(event({"reasoning": "Partial reasoning"}, "length"))
    state.consume("[DONE]")
    state.validate()
    assert state.finish_reason == "length" and state.candidate is None


def metrics(engines=16):
    return "\n".join(
        f'vllm:{name}{{engine="{i}"}} 0'
        for name in (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
        )
        for i in range(engines)
    )


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "nan", "negative"])
def test_reset_requires_every_engine_and_valid_metrics(invalid):
    text = metrics(15 if invalid == "missing" else 16)
    if invalid == "duplicate":
        text += '\nvllm:num_requests_running{engine="0"} 0'
    if invalid in ("nan", "negative"):
        text = text.replace("} 0", "} NaN" if invalid == "nan" else "} -1", 1)
    with pytest.raises(ValueError):
        check.engine_metrics(text, 16)


def response(data=None, text=""):
    return SimpleNamespace(json=lambda: data, text=text, raise_for_status=lambda: None)


def test_reset_never_forces_running_requests(tmp_path, monkeypatch):
    posts = []
    monkeypatch.setattr(
        check.requests, "get", lambda *a, **kw: response(text=metrics())
    )

    def post(url, **kwargs):
        posts.append(kwargs)
        return response({"success": True})

    monkeypatch.setattr(check.requests, "post", post)
    check.reset(tmp_path / "reset.json")
    assert posts[0]["params"] == {
        "reset_running_requests": "false",
        "reset_external": "false",
    }
    assert json.loads((tmp_path / "reset.json").read_text())["success"] is True


def fixture(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    path = private / "request.json"
    path.write_text(
        json.dumps(
            {
                "model": "old-alias",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 1.0,
                "top_p": 0.95,
                "tools": [],
            }
        )
    )
    monkeypatch.setattr(check, "REQUEST_SHA256", check.digest(path.read_bytes()))
    check.write_json(
        private / "provenance.json", {"evidence_uri": "gs://private/unit-test"}
    )
    return path


def test_fixture_only_changes_alias_stream_and_generation_bound(tmp_path, monkeypatch):
    path = fixture(tmp_path, monkeypatch)
    original = json.loads(path.read_text())
    effective = check.load_request(path)
    for key, value in original.items():
        if key != "model":
            assert effective[key] == value
    assert effective["max_tokens"] == 8192 and effective["stream"] is True
    assert "seed" not in effective
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        check.load_request(path)


@pytest.mark.parametrize(
    "failure,expected", [(None, 5), ("repetition", 3), ("protocol", 1), ("timeout", 1)]
)
def test_suite_runs_at_most_five_and_stops_on_first_failure(
    tmp_path, monkeypatch, failure, expected
):
    path = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        check.requests,
        "get",
        lambda *a, **kw: response({"data": [{"max_model_len": 524288}]}),
    )
    monkeypatch.setattr(
        check.requests, "post", lambda *a, **kw: response({"count": 332523})
    )
    resets, uploads, launches, signals = [], [], [], []
    monkeypatch.setattr(check, "reset", lambda path: resets.append(path.name))
    monkeypatch.setattr(check, "gcs_copy", lambda *args: uploads.append(args))

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            launches.append(command)
            directory = Path(command[command.index("--out") + 1])
            status = (
                "repetition"
                if failure == "repetition" and len(launches) == 3
                else "passed"
            )
            if failure in ("protocol", "timeout"):
                status = "inconclusive"
                self.returncode = 1
            check.write_json(directory / "result.json", {"status": status})

        def wait(self, timeout=None):
            if failure == "timeout" and timeout == 300:
                raise subprocess.TimeoutExpired("capture", timeout)
            return self.returncode

        def send_signal(self, sig):
            signals.append(sig)

    monkeypatch.setattr(check.subprocess, "Popen", Process)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            check.run_suite(path, tmp_path / "out")
    else:
        check.run_suite(path, tmp_path / "out")
    assert len(launches) == expected
    assert resets == [
        *(f"reset-{i}.json" for i in range(1, expected + 1)),
        "final-reset.json",
    ]
    assert len(uploads) == 1 and uploads[0][1].endswith("evidence.tar.gz")
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert len(report["attempts"]) == expected
    assert report["status"] == (
        "passed"
        if failure is None
        else "repetition"
        if failure == "repetition"
        else "inconclusive"
    )
    if failure == "timeout":
        assert signals == [signal.SIGINT]


@pytest.mark.parametrize("failure", [None, "malformed", "missing_done", "loop"])
def test_http_capture_preserves_transcript_and_fails_closed(
    tmp_path, monkeypatch, failure
):
    path = fixture(tmp_path, monkeypatch)
    frames = [event({"reasoning": "Read the function."})]
    if failure == "malformed":
        frames.append("{invalid")
    elif failure == "loop":
        frames.append(event({"reasoning": "AgainAnd" * 32}))
    else:
        frames.append(event({"content": "Completed."}, "stop"))
        if failure != "missing_done":
            frames.append("[DONE]")
    raw = "".join(f"data: {frame}\r\n\r\n" for frame in frames).encode()

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, **kwargs):
            return (raw[i : i + 7] for i in range(0, len(raw), 7))

    def post(url, **kwargs):
        assert url.endswith("/v1/chat/completions")
        assert kwargs["stream"] is True
        assert json.loads(kwargs["data"])["max_tokens"] == check.MAX_TOKENS
        return Response()

    monkeypatch.setattr(check.requests, "post", post)
    out = tmp_path / "capture"
    if failure in ("malformed", "missing_done"):
        with pytest.raises(ValueError):
            check.capture(path, out)
    else:
        check.capture(path, out)
    report = json.loads((out / "result.json").read_text())
    expected = (
        "passed"
        if failure is None
        else "repetition"
        if failure == "loop"
        else "inconclusive"
    )
    assert report["status"] == expected
    assert report["response_sha256"] == check.digest(
        (out / "response.sse").read_bytes()
    )
    if failure is None:
        assert (out / "response.sse").read_bytes() == raw
    assert report["effective_request_sha256"] == check.digest(
        (out / "effective-request.json").read_bytes()
    )
    assert (out / "partial-text.json").exists()


@pytest.mark.parametrize("invalid", ["window", "token_count", "fixture"])
def test_identity_failure_stops_before_first_generation(tmp_path, monkeypatch, invalid):
    path = fixture(tmp_path, monkeypatch)
    if invalid == "fixture":
        path.write_text(path.read_text() + " ")
    monkeypatch.setattr(
        check.requests,
        "get",
        lambda *a, **kw: response(
            {"data": [{"max_model_len": 98816 if invalid == "window" else 524288}]}
        ),
    )
    monkeypatch.setattr(
        check.requests,
        "post",
        lambda *a, **kw: response(
            {"count": 100 if invalid == "token_count" else 332523}
        ),
    )
    monkeypatch.setattr(check, "reset", lambda path: None)
    monkeypatch.setattr(check, "gcs_copy", lambda *args: None)
    monkeypatch.setattr(
        check.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not generate")
    )
    with pytest.raises(ValueError):
        check.run_suite(path, tmp_path / "out")
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["status"] == "inconclusive" and not report["attempts"]
