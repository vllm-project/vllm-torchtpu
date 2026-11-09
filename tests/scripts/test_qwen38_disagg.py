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
"""Check PD completion accounting and optional exact-token diagnostics."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = load("check_pd_completions", ROOT / "examples/disagg/check_pd_completions.py")


def test_partial_request_failure_keeps_all_completed_and_failed_receipts(
    tmp_path, monkeypatch
):
    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, rank + 42])
        if rank == 1:
            raise ValueError("rank request failed")
        return completion(len(body["prompt"]))

    monkeypatch.setattr(SMOKE, "post", post)
    with pytest.raises(ValueError, match="rank request failed"):
        SMOKE.run_checks("proxy", "prefill", "decode", "model", 4, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 3 and summary["failed"] == 1
    assert summary["unattempted"] == summary["expected"] - 4
    assert len(list(tmp_path.glob("*-reference.json"))) == 4


def completion(length=129, token=11):
    return dict(
        choices=[dict(token_ids=[token] * 16, finish_reason="length")],
        usage=dict(
            prompt_tokens=length,
            completion_tokens=16,
            total_tokens=length + 16,
            prompt_tokens_details=dict(cached_tokens=length - 1),
        ),
    )


def test_exact_repeat_and_reference_gates_reject_mutation_and_bad_accounting():
    import copy

    reference = completion()
    SMOKE.check_triplet(reference, completion(), completion(), 129)
    with pytest.raises(ValueError, match="Repeated"):
        SMOKE.check_triplet(reference, completion(), completion(token=12), 129)
    with pytest.raises(ValueError, match="reference"):
        SMOKE.check_triplet(reference, completion(token=12), completion(token=12), 129)
    broken = copy.deepcopy(reference)
    broken["usage"]["prompt_tokens"] = 128
    with pytest.raises(ValueError, match="accounting"):
        SMOKE.completion_ids(broken, 129)
    broken = copy.deepcopy(reference)
    broken["choices"][0].pop("token_ids")
    with pytest.raises(ValueError, match="every generated"):
        SMOKE.completion_ids(broken, 129)


@pytest.mark.parametrize("dp", [4, 8])
def test_actual_smoke_runner_preserves_prompts_routing_and_failure_receipts(
    dp, tmp_path, monkeypatch
):
    calls = []

    def post(url, body, rank):
        calls.append((url, body.copy(), rank))
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank, 43])
        return completion(len(body["prompt"]))

    monkeypatch.setattr(SMOKE, "post", post)
    summary = SMOKE.run_checks("proxy", "prefill", "decode", "model", dp, tmp_path)
    assert summary["passed"] and summary["completed"] == dp * len(SMOKE.LENGTHS) * 3
    rows = [row for row in calls if row[0].endswith("/v1/completions")]
    assert {row[2] for row in rows} == set(range(dp))
    for rank in range(dp):
        for length in SMOKE.LENGTHS:
            group = [
                body
                for _, body, worker in rows
                if worker == rank and len(body["prompt"]) == length
            ]
            assert len(group) == 3
            assert group[0]["prompt"] == group[1]["prompt"] == group[2]["prompt"]
            assert length + group[0]["max_tokens"] <= 33792
    monkeypatch.setattr(
        SMOKE, "post", lambda *_: (_ for _ in ()).throw(ValueError("transport failed"))
    )
    with pytest.raises(ValueError, match="transport failed"):
        SMOKE.run_checks(
            "proxy", "prefill", "decode", "model", dp, tmp_path / "failure"
        )
    failed = json.loads((tmp_path / "failure/summary.json").read_text())
    assert not failed["passed"] and failed["errors"] and failed["completed"] == 0


def test_proxy_rejects_missing_handle_before_decode_dispatch(monkeypatch):
    import asyncio
    from types import SimpleNamespace as NS

    proxy = load("pd_proxy_gate", ROOT / "examples/disagg/toy_proxy_server.py")
    monkeypatch.setattr(
        proxy, "global_args", NS(require_kv_transfer=True), raising=False
    )
    monkeypatch.setattr(proxy, "get_next_client", lambda *_: {})

    async def prefill(*_):
        return NS(json=lambda: dict(choices=[{}]))

    async def body():
        return dict(prompt=[41], model="model")

    monkeypatch.setattr(proxy, "send_request_to_prefill", prefill)
    request = NS(json=body, headers={}, app=NS(state=NS()))
    with pytest.raises(ValueError, match="transfer handle"):
        asyncio.run(proxy._handle_completions("/v1/completions", request))
    proxy.require_transfer_handle({}, False)
    proxy.require_transfer_handle(dict(uuid=123), True)


def test_nonempty_transfer_handle_with_local_recompute_cannot_pass():
    proxy = load("pd_proxy_gate", ROOT / "examples/disagg/toy_proxy_server.py")
    proxy.require_transfer_handle(dict(uuid=123), True)
    local = completion()
    local["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    with pytest.raises(ValueError, match="complete remote prefix"):
        SMOKE.check_triplet(local, local, local, 129)
    # The reference is deliberately decode-local; only the PD outputs must consume KV.
    SMOKE.check_triplet(local, completion(), completion(), 129)


@pytest.mark.parametrize("numeric_mode", ["report", "strict"])
@pytest.mark.parametrize("difference", ["reference", "repeat"])
def test_numeric_policy_preserves_failure_separately(
    numeric_mode, difference, tmp_path, monkeypatch
):
    monkeypatch.setattr(SMOKE, "LENGTHS", (129,))

    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank])
        token = 11 if url.startswith("decode") else 12
        if difference == "repeat" and body["request_id"].endswith("repeat"):
            token = 13
        return completion(len(body["prompt"]), token=token)

    monkeypatch.setattr(SMOKE, "post", post)
    if numeric_mode == "strict":
        with pytest.raises(ValueError, match="reference|Repeated"):
            SMOKE.run_checks(
                "proxy", "prefill", "decode", "model", 4, tmp_path, numeric_mode
            )
    else:
        summary = SMOKE.run_checks("proxy", "prefill", "decode", "model", 4, tmp_path)
        assert summary["performance_checks_passed"]
        assert summary["completed"] == summary["expected"] == 12
        assert len(summary["numeric_diagnostics"]) == 4
    recorded = json.loads((tmp_path / "summary.json").read_text())
    assert not recorded["passed"]
    assert recorded["numeric_reference_passed"] is False
    assert recorded["numeric_repeat_passed"] == (difference != "repeat")
    assert recorded["diagnostic_failed"]
    assert recorded["performance_checks_passed"] == (numeric_mode == "report")
    assert len(json.loads((tmp_path / "results.json").read_text())["requests"]) == 12


@pytest.mark.parametrize("numeric_mode", ["report", "strict"])
@pytest.mark.parametrize("failure", ["HTTP", "tokens", "usage", "remote"])
def test_transport_gates_fail_in_both_numeric_modes(
    numeric_mode, failure, tmp_path, monkeypatch
):
    monkeypatch.setattr(SMOKE, "LENGTHS", (129,))

    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank])
        if failure == "HTTP":
            raise ValueError("HTTP failed")
        result = completion(len(body["prompt"]))
        if failure == "tokens":
            result["choices"][0].pop("token_ids")
        elif failure == "usage":
            result["usage"]["total_tokens"] += 1
        elif failure == "remote" and url.startswith("proxy"):
            result["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        return result

    monkeypatch.setattr(SMOKE, "post", post)
    with pytest.raises(ValueError):
        SMOKE.run_checks(
            "proxy", "prefill", "decode", "model", 4, tmp_path, numeric_mode
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["performance_checks_passed"] and summary["errors"]
