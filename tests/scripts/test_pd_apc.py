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
"""Client acceptance controls; HTTP simulations are not native qualification."""

import json
import subprocess

import pytest

from tests.scripts.test_qwen38_disagg import KUBE, PROFILE, ROOT, load, stubs

APC = load("check_pd_apc", ROOT / "examples/disagg/check_pd_apc.py")


class Engines:
    def __init__(self, fault=None):
        self.fault = fault
        self.cache = {role: [set(), set()] for role in ("P", "D")}
        self.counts = {role: [[0, 0, 0], [0, 0, 0]] for role in ("P", "D")}

    def __call__(self, url, body=None, rank=None):
        role = "P" if url.startswith("P/") else "D"
        if url.endswith("/collective_rpc"):
            assert body == {"method": "get_kv_cache_plan"}
            return {
                "results": [
                    dict(
                        dp_rank=i,
                        dp_size=2,
                        tp_rank=0,
                        tp_size=1,
                        prefix_caching=self.fault != "disabled",
                        mamba_cache_mode="align",
                        cache_hit_alignment_tokens=4,
                        max_model_len=64,
                        groups=[
                            dict(block_tokens=4, prefix_cacheable=True),
                            dict(block_tokens=2, prefix_cacheable=False),
                        ],
                    )
                    for i in range(1)
                ]
            }
        if url.endswith("/reset_prefix_cache"):
            self.cache[role] = [set(), set()]
            return dict(success=True)
        if url.endswith("/metrics"):
            return "\n".join(
                f'{name}{{engine="{engine}"}} {value}'
                for engine, values in enumerate(self.counts[role])
                for name, value in zip(
                    (
                        "vllm:prefix_cache_queries_total",
                        "vllm:prefix_cache_hits_total",
                        "vllm:external_prefix_cache_hits_total",
                    ),
                    values,
                )
            )
        if url.endswith("/tokenize"):
            return {"tokens": [ord(char) for char in body["prompt"][-24:]]}
        if url.endswith("/v1/chat/completions"):
            for name in ("P", "D"):
                self.counts[name][rank][0] += 1
            question = body["messages"][0]["content"]
            expected = next(
                answer
                for _, prompt, answer in APC.CORRECTNESS.SHORT_QA
                if prompt == question
            )
            if self.fault == "known-answer":
                expected = "incorrect"
            return {
                "choices": [dict(finish_reason="stop", message=dict(content=expected))]
            }
        assert url == "proxy/v1/completions"
        prompt = tuple(body["prompt"])
        routed = (rank + 1) % 2 if self.fault == "routing" else rank
        local_hits = []
        for name, prefix in [("P", prompt[:-1]), ("D", prompt)]:
            stored = self.cache[name][routed]
            hit = max(
                (
                    len(key)
                    for key in stored
                    if len(key) < len(prefix) and prefix[: len(key)] == key
                ),
                default=0,
            )
            if self.fault == "ignored":
                hit = 0
            queries = self.counts[name][routed]
            queries[0] += len(prefix)
            queries[1] += hit
            if name == "D" and self.fault != "recompute":
                queries[2] += max(len(prompt) - 1 - hit, 0)
            for count in range(4, len(prefix) + 1, 4):
                stored.add(prefix[:count])
            local_hits.append(hit)
        tokens = [(sum(prompt) + rank + i) % 97 for i in range(APC.OUTPUT_TOKENS)]
        if self.fault == "drift" and local_hits[1]:
            tokens[0] += 1
        logprob = -0.5 - (0.3 if self.fault == "logprob" and local_hits[1] else 0)
        return dict(
            choices=[
                dict(
                    token_ids=tokens,
                    finish_reason="length",
                    logprobs=dict(token_logprobs=[logprob] * APC.OUTPUT_TOKENS),
                )
            ],
            usage=dict(
                prompt_tokens=len(prompt),
                completion_tokens=APC.OUTPUT_TOKENS,
                total_tokens=len(prompt) + APC.OUTPUT_TOKENS,
            ),
        )


def test_complete_cold_role_warm_branch_and_interleaved_matrix(tmp_path):
    summary = APC.run("P", "D", "proxy", "model", 2, tmp_path, call=Engines())
    assert summary["passed"] and summary["completed"] == summary["expected"] == 44
    assert summary["failed"] == 0
    for rank in range(2):
        partial = json.loads((tmp_path / f"rank{rank}-partial-branch.json").read_text())
        assert (
            partial["counter_deltas"][1][
                f"vllm:external_prefix_cache_hits_total/engine{rank}"
            ]
            == 4
        )
        full = json.loads((tmp_path / f"rank{rank}-D-warm-A.json").read_text())
        assert (
            full["counter_deltas"][1][
                f"vllm:external_prefix_cache_hits_total/engine{rank}"
            ]
            == 0
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("disabled", "APC"),
        ("ignored", "cache-hit"),
        ("recompute", "Remote consumption"),
        ("routing", "routing"),
        ("drift", "token IDs"),
        ("logprob", "logprob differs"),
        ("known-answer", "known-answer"),
    ],
)
def test_flags_recompute_misrouting_and_output_drift_cannot_pass(
    tmp_path, fault, reason
):
    with pytest.raises(ValueError, match=reason):
        APC.run("P", "D", "proxy", "model", 2, tmp_path, call=Engines(fault))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["passed"] and summary["failed"] > 0 and summary["errors"]
    if fault != "disabled":
        assert any(
            "error" in json.loads(path.read_text())
            for path in tmp_path.glob("rank*.json")
        )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_apc_role_uses_existing_profile_and_explicit_aligned_state(role):
    profile = PROFILE.role_profile(role, 8, "model", "127.0.0.1", apc=True)
    assert "--enable-prefix-caching" in profile["argv"]
    assert "--mamba-cache-mode=align" in profile["argv"]
    assert profile["env"]["TPU_RAIDEN_PREFIX_AWARE_LOAD"] == "1"
    assert profile["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert profile["env"]["VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND"] == (
        "torch" if role == "prefill" else "native_dma"
    )
    performance = PROFILE.role_profile(role, 8, "model", "127.0.0.1")
    assert "--no-enable-prefix-caching" in performance["argv"]
    assert "VLLM_SERVER_DEV_MODE" not in performance["env"]


def test_apc_ci_entry_reuses_role_manifest_without_performance_upload():
    import yaml

    pipeline = yaml.safe_load((ROOT / ".buildkite/pipeline_perf.yml").read_text())
    group = next(step for step in pipeline["steps"] if "steps" in step)
    lane = next(
        step
        for step in group["steps"]
        if step.get("env", {}).get("DISAGG_TEST_MODE") == "apc"
    )
    assert lane["env"]["BQ_UPLOAD"] == "0" and lane["env"]["DISAGG_DP_SIZE"] == "8"
    assert "qwen3.8-flash-1p1d.yaml" in lane["commands"][0]
    assert (KUBE / "manifests/workloads/qwen3.8-flash-1p1d.yaml").is_file()


@pytest.mark.parametrize("rc", [0, 7])
def test_actual_apc_driver_dispatch_keeps_failure_receipts_and_cleans_proxy(
    tmp_path, rc
):
    env = stubs(tmp_path)
    env.update(DISAGG_TEST_MODE="apc", STUB_APC_RC=str(rc))
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == rc, (result.stdout, result.stderr)
    args = json.loads((tmp_path / "apc-client.json").read_text())
    assert args[args.index("--dp") + 1] == "8"
    assert args[args.index("--prefill") + 1].endswith("prefill-0-0.cpu-test:8000")
    assert not (tmp_path / "vllm.jsonl").exists()
    assert not (tmp_path / "bq-called").exists()
    assert len((tmp_path / "artifacts.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "results/apc/summary.json").read_text())[
        "passed"
    ] is (rc == 0)
    from pathlib import Path

    pid = int((tmp_path / "proxy.pid").read_text())
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_actual_apc_engine_dispatch_uses_aligned_prefix_profile(tmp_path, role):
    env = stubs(tmp_path)
    env["DISAGG_TEST_MODE"] = "apc"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_engine.sh"), role],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / "vllm.jsonl").read_text())
    assert "--enable-prefix-caching" in row["argv"]
    assert "--mamba-cache-mode=align" in row["argv"]
    assert row["env"]["TPU_RAIDEN_PREFIX_AWARE_LOAD"] == "1"
