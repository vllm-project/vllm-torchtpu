# SPDX-License-Identifier: Apache-2.0
"""Bounded serial DeepSWE checks on the DP16 benchmark's existing server."""

import argparse
import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests

REQUEST_SHA256 = "45871ebc0e5c9adb23e877b82bb605749a0124c3bd4b5f998412ee6e2adb4f1f"
FIXTURE_URI = (
    "gs://tpu-for-training-falcon-logs/experiments/exp-8243e2um0p/"
    "artifacts/art-ainhcr1xnw/seq201-ab/run-01/capture/original-request.json"
)
EVIDENCE_ROOT = "gs://tpu-for-training-falcon-logs/ci-replays/deepswe"
PROMPT_TOKENS = 332523
MODEL_WINDOW = 524288
ATTEMPTS = 5
MAX_TOKENS = 8192
REQUEST_SECONDS = 300
RESET_SECONDS = 60
ROOT = "http://127.0.0.1:8000"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_request(path: Path) -> dict:
    raw = path.read_bytes()
    if digest(raw) != REQUEST_SHA256:
        raise ValueError("DeepSWE fixture SHA256 mismatch")
    request = json.loads(raw)
    if request.get("seed") is not None or (
        request["temperature"],
        request["top_p"],
    ) != (1.0, 0.95):
        raise ValueError("DeepSWE fixture sampling contract changed")
    # The served alias and the two bounded-capture fields are the only changes.
    return {
        **request,
        "model": "deepseek-v41",
        "stream": True,
        "max_tokens": MAX_TOKENS,
    }


def gcs_copy(source: str, destination: str) -> None:
    subprocess.run(
        ["gcloud", "storage", "cp", source, destination],
        check=True,
        capture_output=True,
        timeout=120,
    )


def prepare(directory: Path) -> Path:
    """Check fixture/read and private evidence/write access before launching TPU."""
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    gcs_copy(FIXTURE_URI, str(request))
    effective = load_request(request)
    evidence_uri = f"{EVIDENCE_ROOT}/{uuid.uuid4()}"
    write_json(
        directory / "provenance.json",
        {
            "request_sha256": REQUEST_SHA256,
            "fixture_uri": FIXTURE_URI,
            "effective_request_sha256": digest(
                json.dumps(effective, ensure_ascii=False).encode()
            ),
            "evidence_uri": evidence_uri,
        },
    )
    gcs_copy(str(directory / "provenance.json"), f"{evidence_uri}/provenance.json")
    return request


def repetition(text: str) -> dict | None:
    """Detect sustained word, unseparated-character and punctuation loops."""
    tail = text[-65536:].strip()
    words = re.findall(r"\b\w+\b", tail.lower())[-4096:]
    for period in range(1, min(257, len(words) // 8 + 1)):
        repeats = max(8, math.ceil(32 / period))
        if len(words) >= period * repeats:
            unit = words[-period:]
            if words[-period * repeats :] == unit * repeats:
                return {"kind": "words", "period": period, "repeats": repeats}
    # Inspect the recent window, not only its suffix: an SSE delta can contain
    # the complete repeated region followed by a new word.
    for match in re.finditer(r"(.{1,256}?)\1{7,}", tail[-4096:], re.S):
        if len(match[0]) >= 128 and any(c.isalnum() for c in match[1]):
            return {
                "kind": "characters",
                "period": len(match[1]),
                "repeats": len(match[0]) // len(match[1]),
            }
    for span in re.findall(r"[^\w\s]{64,}", tail):
        for period in range(1, 17):
            repeats = max(16, math.ceil(64 / period))
            if re.search(
                r"(.{" + str(period) + r"})\1{" + str(repeats - 1) + r"}", span
            ):
                return {"kind": "punctuation", "period": period, "repeats": repeats}
    return None


@dataclass
class Stream:
    reasoning: str = ""
    content: str = ""
    tools: dict = field(default_factory=dict)
    finish_reason: str | None = None
    done: bool = False
    candidate: dict | None = None
    events: int = 0
    checked_characters: int = 0

    def consume(self, data: str) -> None:
        if self.done:
            raise ValueError("SSE data after DONE")
        if data == "[DONE]":
            if self.finish_reason not in ("stop", "tool_calls", "length"):
                raise ValueError("DONE without a supported finish reason")
            self.done = True
            return
        event = json.loads(data)
        if "error" in event:
            raise ValueError("SSE server error")
        self.events += 1
        choices = event.get("choices", [])
        if len(choices) > 1 or (choices and self.finish_reason is not None):
            raise ValueError("Unexpected choices after completion or multiple choices")
        for choice in choices:
            if choice["index"] != 0:
                raise ValueError("Expected exactly one completion")
            delta = choice.get("delta", {})
            self.reasoning += (
                delta.get("reasoning") or delta.get("reasoning_content") or ""
            )
            self.content += delta.get("content") or ""
            for tool in delta.get("tool_calls", []):
                value = self.tools.setdefault(
                    tool["index"], {"name": "", "arguments": ""}
                )
                function = tool.get("function", {})
                value["name"] += function.get("name") or ""
                value["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason")
            if finish is not None:
                if self.finish_reason is not None:
                    raise ValueError("Duplicate finish reason")
                self.finish_reason = finish
        characters = len(self.reasoning) + len(self.content)
        if characters - self.checked_characters >= 32 or self.finish_reason:
            self.candidate = (
                self.candidate or repetition(self.reasoning) or repetition(self.content)
            )
            self.checked_characters = characters

    def validate(self) -> None:
        self.candidate = (
            self.candidate or repetition(self.reasoning) or repetition(self.content)
        )
        if not self.done or not (self.reasoning or self.content or self.tools):
            raise ValueError("Empty or incomplete SSE response")
        if self.finish_reason == "tool_calls" or (
            self.tools and self.finish_reason != "length"
        ):
            if not self.tools:
                raise ValueError("Missing tool calls")
            for tool in self.tools.values():
                if not tool["name"]:
                    raise ValueError("Missing tool name")
                json.loads(tool["arguments"])
        # 'length' is an explicit bounded observation, not a complete tool call.


def capture(request: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    stream = Stream()
    payload = load_request(request)
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    (directory / "effective-request.json").write_bytes(encoded)
    report = {"status": "inconclusive", "started_at": time.time()}
    raw = directory / "response.sse"
    try:
        with (
            requests.post(
                ROOT + "/v1/chat/completions",
                data=encoded,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=(10, REQUEST_SECONDS),
            ) as response,
            raw.open("wb") as output,
        ):
            response.raise_for_status()
            report["http_status"] = response.status_code
            lines = []
            pending = b""
            total = 0
            # Keep the received SSE body bytes, including original CRLF framing.
            # A candidate can leave a partial final frame in the raw capture;
            # partial-text contains exactly the complete frames parsed so far.
            for chunk in response.iter_content(chunk_size=1024):
                output.write(chunk)
                total += len(chunk)
                if total > 16 * 1024 * 1024:
                    raise ValueError("Response exceeded capture byte budget")
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.removesuffix(b"\r")
                    if line:
                        if line.startswith(b"data:"):
                            lines.append(line[5:].lstrip().decode("utf-8"))
                        elif not line.startswith(b":"):
                            raise ValueError("Unexpected SSE field")
                    elif lines:
                        stream.consume("\n".join(lines))
                        lines.clear()
                if stream.candidate:
                    report["status"] = "repetition"
                    break
            if not stream.candidate:
                if pending or lines:
                    raise ValueError("Unterminated SSE frame")
                stream.validate()
                report["status"] = "repetition" if stream.candidate else "passed"
    finally:
        report.update(
            finished_at=time.time(),
            done=stream.done,
            finish_reason=stream.finish_reason,
            candidate=stream.candidate,
            events=stream.events,
            response_sha256=digest(raw.read_bytes()) if raw.exists() else None,
            effective_request_sha256=digest(
                (directory / "effective-request.json").read_bytes()
            ),
        )
        write_json(
            directory / "partial-text.json",
            {
                "reasoning": stream.reasoning,
                "content": stream.content,
                "tools": stream.tools,
            },
        )
        write_json(directory / "result.json", report)


def engine_metrics(text: str, expected: int) -> dict:
    metrics = {
        k: {}
        for k in (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
        )
    }
    for line in text.splitlines():
        match = re.match(r"^vllm:(\w+)\{([^}]+)\}\s+(\S+)", line)
        if match and match[1] in metrics:
            engine = re.search(r'\bengine="([^"]+)"', match[2])
            if engine:
                if engine[1] in metrics[match[1]]:
                    raise ValueError("Duplicate engine metric")
                metrics[match[1]][engine[1]] = float(match[3])
    for values in metrics.values():
        if set(values) != {str(i) for i in range(expected)}:
            raise ValueError("Missing DP engine metrics")
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Invalid DP engine metric")
    return metrics


def reset(directory: Path, engines: int = 16) -> None:
    report = {"success": False, "started_at": time.time(), "observations": []}
    deadline = time.monotonic() + RESET_SECONDS
    try:
        while time.monotonic() < deadline:
            result = requests.get(ROOT + "/metrics", timeout=5)
            result.raise_for_status()
            metrics = engine_metrics(result.text, engines)
            report["observations"].append(metrics)
            if all(v == 0 for values in metrics.values() for v in values.values()):
                response = requests.post(
                    ROOT + "/reset_prefix_cache",
                    params={
                        "reset_running_requests": "false",
                        "reset_external": "false",
                    },
                    json={},
                    timeout=5,
                )
                response.raise_for_status()
                if response.json().get("success") is True:
                    after = requests.get(ROOT + "/metrics", timeout=5)
                    after.raise_for_status()
                    metrics = engine_metrics(after.text, engines)
                    report["observations"].append(metrics)
                    if all(
                        v == 0 for values in metrics.values() for v in values.values()
                    ):
                        report["success"] = True
                        return
            time.sleep(1)
        raise TimeoutError("DP engines did not drain/reset within budget")
    finally:
        write_json(directory, report)


def run_suite(request: Path, out: Path) -> None:
    private = request.parent
    provenance = json.loads((private / "provenance.json").read_text())
    result = {
        **provenance,
        "status": "inconclusive",
        "attempts": [],
        "max_attempts": ATTEMPTS,
        "max_tokens": MAX_TOKENS,
        "request_seconds": REQUEST_SECONDS,
        "started_at": time.time(),
    }
    try:
        payload = load_request(request)
        models = requests.get(ROOT + "/v1/models", timeout=10)
        models.raise_for_status()
        if models.json()["data"][0]["max_model_len"] != MODEL_WINDOW:
            raise ValueError("Repetition check requires the 512K service window")
        tokenized = requests.post(ROOT + "/tokenize", json=payload, timeout=60)
        tokenized.raise_for_status()
        if tokenized.json()["count"] != PROMPT_TOKENS:
            raise ValueError("Frozen prompt token count changed")
        for index in range(1, ATTEMPTS + 1):
            reset(out / f"reset-{index}.json")
            directory = private / f"run-{index}"
            with (private / f"capture-{index}.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "capture",
                        "--request",
                        str(request),
                        "--out",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                timed_out = False
                try:
                    code = process.wait(timeout=REQUEST_SECONDS)
                except subprocess.TimeoutExpired:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    timed_out = True
                    code = process.returncode
            item_path = directory / "result.json"
            item = (
                json.loads(item_path.read_text())
                if item_path.exists()
                else {
                    "status": "inconclusive",
                    "error": "capture produced no result",
                }
            )
            item["timed_out"] = timed_out
            result["attempts"].append(item)
            if item["status"] == "repetition":
                result["status"] = "repetition"
                raise ValueError(
                    "DeepSWE repetition detected; further requests stopped"
                )
            if timed_out or code != 0 or item["status"] != "passed":
                raise RuntimeError(
                    "DeepSWE capture/protocol failed; further requests stopped"
                )
        result["status"] = "passed"
    finally:
        try:
            reset(out / "final-reset.json")
        except (requests.RequestException, ValueError, TimeoutError):
            result["status"] = "inconclusive"
            raise
        finally:
            result["finished_at"] = time.time()
            # Public artifacts contain only metadata; raw prompts and model/tool
            # text go to the same private bucket as the original trajectory.
            write_json(out / "summary.json", result)
            archive = private.parent / "deepswe-evidence.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(private, arcname="replay")
                bundle.add(out, arcname="summary")
            result["archive_sha256"] = digest(archive.read_bytes())
            result["archive_uri"] = f"{provenance['evidence_uri']}/evidence.tar.gz"
            try:
                gcs_copy(str(archive), result["archive_uri"])
                result["evidence_uploaded"] = True
            finally:
                write_json(out / "summary.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "capture", "check"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.request is None:
        parser.error("--request is required for capture/check")
    elif args.action == "capture":
        capture(args.request, args.out)
    else:
        run_suite(args.request, args.out)


if __name__ == "__main__":
    main()
