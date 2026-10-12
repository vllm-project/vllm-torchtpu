# SPDX-License-Identifier: Apache-2.0
"""Check active DSpark on uneven C8 and idle C1 with saved long-context inputs."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families


def counters(text: str) -> dict[tuple[str, str, str], float]:
    names = {
        "vllm:request_success_total",
        "vllm:num_preemptions_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    }
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in names:
                key = (
                    sample.name,
                    sample.labels.get("engine", ""),
                    sample.labels.get("finished_reason", ""),
                )
                result[key] = result.get(key, 0) + sample.value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v41")
    args = parser.parse_args()
    prompts = json.loads(args.input.read_text())
    assert len(prompts) == 8 and all(len(p) == 81920 for p in prompts)
    args.out.mkdir(parents=True, exist_ok=True)

    def metrics():
        response = requests.get(args.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        return response.text, counters(response.text)

    def request(rank):
        # Distinct long documents, uneven final chunks, and sixteen output
        # tokens force both prefill and speculative decode to execute.
        prompt = prompts[rank][: 81920 - rank * 257]
        started = time.perf_counter()
        response = requests.post(
            args.base_url + "/v1/completions",
            headers={"X-data-parallel-rank": str(rank)},
            json={
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 16,
                "temperature": 0,
                "ignore_eos": True,
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        assert body["usage"]["completion_tokens"] == 16, body
        assert body["usage"]["prompt_tokens"] == len(prompt), body["usage"]
        return {
            "rank": rank,
            "prompt_tokens": len(prompt),
            "wall_s": time.perf_counter() - started,
            "response": body,
        }

    report = {}
    for name, ranks in [("uneven-c8", list(range(8))), ("idle-c1", [3])]:
        reset = requests.post(args.base_url + "/reset_prefix_cache", timeout=180)
        reset.raise_for_status()
        before_text, before = metrics()
        (args.out / f"{name}-before.prom").write_text(before_text)
        with concurrent.futures.ThreadPoolExecutor(len(ranks)) as pool:
            responses = list(pool.map(request, ranks))
        (args.out / f"{name}-responses.json").write_text(
            json.dumps(responses, indent=2)
        )
        for _ in range(60):
            after_text, after = metrics()
            delta = {key: value - before.get(key, 0) for key, value in after.items()}
            successes = sum(
                v
                for (metric, _, reason), v in delta.items()
                if metric == "vllm:request_success_total"
                and reason in {"stop", "length"}
            )
            if successes >= len(ranks):
                break
            time.sleep(1)
        (args.out / f"{name}-after.prom").write_text(after_text)
        assert successes == len(ranks), (name, successes)
        for (metric, engine, reason), value in delta.items():
            if metric == "vllm:num_preemptions_total" or reason in {"abort", "error"}:
                assert value == 0, (name, metric, engine, reason, value)
        for rank in range(8):
            completed = sum(
                value
                for (metric, engine, reason), value in delta.items()
                if metric == "vllm:request_success_total"
                and engine == str(rank)
                and reason in {"stop", "length"}
            )
            assert completed == int(rank in ranks), (name, rank, completed)
        for rank in ranks:
            assert (
                delta.get(("vllm:spec_decode_num_drafts_total", str(rank), ""), 0) > 0
            )
        report[name] = {
            "completed_ranks": ranks,
            "output_tokens_per_request": 16,
            "counters": [
                {"metric": k[0], "engine": k[1], "reason": k[2], "delta": v}
                for k, v in delta.items()
                if v
            ],
        }
    report["scope"] = "Completion, idle collectives and active DSpark only."
    (args.out / "validated.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
