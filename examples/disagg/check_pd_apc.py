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
"""Cold/warm PD equivalence with real per-engine APC and remote-hit counters."""

import argparse
import importlib.util
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

# Match run_qwen35_p8d8_disagg_correctness.sh's PD divergence criterion.
OUTPUT_TOKENS = 1
MTP_OUTPUT_TOKENS = 32
LOGPROB_ATOL = 0.2
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prefix_correctness",
    ROOT / "scripts/vllm/integration/smoke_prefix_cache_correctness.py",
)
assert SPEC is not None and SPEC.loader is not None
CORRECTNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORRECTNESS)


def http(url: str, body: dict | None = None, rank: int | None = None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            dict(status=error.code, body=error.read().decode())
        ) from error
    return text if url.endswith("/metrics") else json.loads(text)


def counters(text: str) -> dict:
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in (
                "vllm:prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
                "vllm:external_prefix_cache_hits_total",
                "vllm:generation_tokens_total",
                "vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total",
            ):
                key = (sample.name, int(sample.labels["engine"]))
                if key in result:
                    raise ValueError(
                        "Duplicate engine counter in this single-model test"
                    )
                result[key] = sample.value
    return result


def cache_contract(rows: list, dp: int, *, mtp: bool = False) -> tuple[int, int]:
    # The installed dev RPC calls the default DP engine's TP workers.
    flat = rows
    if not flat or {row["dp_size"] for row in flat} != {dp}:
        raise ValueError("Cache-plan RPC topology differs from the requested DP size")
    if len({row["dp_rank"] for row in flat}) != 1 or any(
        not 0 <= row["dp_rank"] < dp for row in flat
    ):
        raise ValueError("Cache-plan RPC must identify its actual DP engine")
    tp = flat[0]["tp_size"]
    if {row["tp_size"] for row in flat} != {tp} or {
        row["tp_rank"] for row in flat
    } != set(range(tp)):
        raise ValueError("Cache-plan RPC did not cover the engine's TP workers")
    if len({(row["dp_rank"], row["tp_rank"]) for row in flat}) != len(flat):
        raise ValueError("Duplicate worker cache-plan receipt")
    if any(
        not row["prefix_caching"] or row["mamba_cache_mode"] != "align" for row in flat
    ):
        raise ValueError("APC and aligned state must be active in actual workers")
    if mtp and any(
        not any(
            group["prefix_cacheable"]
            and any(name.startswith("mtp.") for name in group.get("layer_names", ()))
            for group in row["groups"]
        )
        for row in flat
    ):
        raise ValueError("MTP draft cache group is missing from worker cache plan")
    sizes = [
        group["block_tokens"]
        for row in flat
        for group in row["groups"]
        if group["prefix_cacheable"]
    ]
    sizes.extend(row["cache_hit_alignment_tokens"] for row in flat)
    if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Initialized cache-plan geometry is missing")
    return math.lcm(*sizes), min(row["max_model_len"] for row in flat)


def run(
    prefill: str,
    decode: str,
    proxy: str,
    model: str,
    dp: int,
    output: Path,
    *,
    call=http,
    mtp: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    criteria = dict(
        greedy_tokens=OUTPUT_TOKENS,
        logprob_atol=LOGPROB_ATOL,
        known_answers="exact normalized answer, interleaved repeats",
    )
    if mtp:
        criteria["mtp_output_tokens"] = MTP_OUTPUT_TOKENS
    summary = dict(
        passed=False,
        completed=0,
        failed=0,
        expected=dp * (27 if mtp else 22),
        cases=[],
        errors=[],
        criteria=criteria,
        geometry_scope=(
            "default DP engine TP workers; all-rank plans require "
            "native supervisor receipts"
        ),
    )
    run_id = uuid.uuid4().hex

    def save(name, value):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")

    def reset(url):
        deadline = time.monotonic() + 30
        while True:
            result = call(url + "/reset_prefix_cache", {})
            if result["success"]:
                return
            if time.monotonic() >= deadline:
                raise ValueError(
                    "Prefix reset failed: requests or producer leases remain held"
                )
            time.sleep(0.1)

    def snapshot():
        raw = [call(url + "/metrics") for url in (prefill, decode)]
        return raw, [counters(text) for text in raw]

    def settled_deltas(rank, before, receipt, *, min_decode_tokens=0):
        # Every request, including known answers, settles before the next
        # rank's baseline. HTTP completion precedes periodic metrics logging.
        deadline = time.monotonic() + 30
        while True:
            raw, after = snapshot()
            receipt["metrics_after"] = raw
            deltas = [
                {
                    key: values.get(key, 0) - old.get(key, 0)
                    for key in values.keys() | old.keys()
                }
                for old, values in zip(before, after)
            ]
            if any(
                value != 0
                for delta in deltas
                for (metric, engine), value in delta.items()
                if engine != rank
                and metric
                in (
                    "vllm:prefix_cache_queries_total",
                    "vllm:spec_decode_num_drafts_total",
                )
            ):
                raise ValueError("Actual engine routing differs from requested rank")
            queries_ready = all(
                delta.get(("vllm:prefix_cache_queries_total", rank), 0) > 0
                for delta in deltas
            )
            decode_ready = min_decode_tokens <= 0 or (
                deltas[1].get(("vllm:generation_tokens_total", rank), 0)
                >= min_decode_tokens
            )
            if queries_ready and decode_ready:
                return deltas
            if time.monotonic() >= deadline:
                raise ValueError(
                    "No actual per-engine prefix lookup counters for requested rank"
                )
            time.sleep(0.1)

    def measured(
        rank,
        name,
        prompt,
        warmth,
        external,
        expected=None,
        *,
        max_tokens=OUTPUT_TOKENS,
        check_spec=False,
    ):
        receipt = dict(rank=rank, case=name, prompt=prompt, expected=expected)
        try:
            raw_before, before = snapshot()
            body = dict(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0,
                seed=42,
                ignore_eos=True,
                logprobs=1,
                return_token_ids=True,
                request_id=f"apc-{run_id}-r{rank}-{name}",
            )
            receipt.update(body=body, metrics_before=raw_before)
            response = call(proxy + "/v1/completions", body, rank)
            receipt["response"] = response
            if len(response["choices"]) != 1:
                raise ValueError("Expected one complete choice")
            choice, usage = response["choices"][0], response["usage"]
            tokens = choice["token_ids"]
            if (
                len(tokens) != max_tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or choice["finish_reason"] != "length"
                or usage["prompt_tokens"] != len(prompt)
                or usage["completion_tokens"] != max_tokens
                or usage["total_tokens"] != len(prompt) + max_tokens
            ):
                raise ValueError("Completion token or request accounting failed")
            logprobs = choice["logprobs"]["token_logprobs"]
            if len(logprobs) != max_tokens or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in logprobs
            ):
                raise ValueError("Missing finite generated-token logprobs")
            result = dict(token_ids=tokens, logprobs=logprobs)
            if expected is not None:
                if tokens[0] != expected["token_ids"][0]:
                    raise ValueError("Cold/warm first token IDs differ")
                difference = abs(logprobs[0] - expected["logprobs"][0])
                receipt["max_logprob_absdiff"] = difference
                if max_tokens > 1:
                    receipt["exact_sequence_match"] = tokens == expected["token_ids"]
                if difference > LOGPROB_ATOL:
                    raise ValueError("Cold/warm first-token logprob differs")
            deltas = settled_deltas(
                rank,
                before,
                receipt,
                min_decode_tokens=max_tokens if check_spec else 0,
            )
            receipt["counter_deltas"] = [
                {
                    f"{metric}/engine{engine}": value
                    for (metric, engine), value in delta.items()
                }
                for delta in deltas
            ]
            for delta, warm in zip(deltas, warmth):
                if any(
                    value != 0
                    for (metric, engine), value in delta.items()
                    if engine != rank and metric == "vllm:prefix_cache_queries_total"
                ):
                    raise ValueError(
                        "Actual engine routing differs from requested rank"
                    )
                hits = delta.get(("vllm:prefix_cache_hits_total", rank), 0)
                if (warm and hits <= 0) or (not warm and hits != 0):
                    raise ValueError(
                        "Local cache-hit counters contradict cold/warm control"
                    )
            if (
                deltas[1].get(("vllm:external_prefix_cache_hits_total", rank), 0)
                != external
            ):
                raise ValueError(
                    "Remote consumption differs from full-hit/suffix contract"
                )
            if check_spec:
                drafts = deltas[1].get(("vllm:spec_decode_num_drafts_total", rank), 0)
                draft_tokens = deltas[1].get(
                    ("vllm:spec_decode_num_draft_tokens_total", rank), 0
                )
                accepted = deltas[1].get(
                    ("vllm:spec_decode_num_accepted_tokens_total", rank), 0
                )
                receipt["spec_decode"] = dict(
                    drafts=drafts,
                    draft_tokens=draft_tokens,
                    accepted_tokens=accepted,
                )
                if (
                    drafts <= 0
                    or draft_tokens <= drafts
                    or not 0 < accepted <= draft_tokens
                ):
                    raise ValueError(
                        "Speculative decoding draft/accept counters failed"
                    )
            summary["completed"] += 1
            summary["cases"].append(dict(rank=rank, case=name, passed=True))
            return result
        except Exception as error:
            receipt["error"] = repr(error)
            summary["failed"] += 1
            summary["cases"].append(dict(rank=rank, case=name, passed=False))
            raise
        finally:
            save(f"rank{rank}-{name}", receipt)

    try:
        plans = [
            call(url + "/collective_rpc", {"method": "get_kv_cache_plan"})
            for url in (prefill, decode)
        ]
        save("cache-plans", dict(prefill=plans[0], decode=plans[1]))
        contracts = [cache_contract(plan["results"], dp, mtp=mtp) for plan in plans]
        alignment = math.lcm(*(contract[0] for contract in contracts))
        max_output = MTP_OUTPUT_TOKENS if mtp else OUTPUT_TOKENS
        if 3 * alignment + 1 + max_output > min(contract[1] for contract in contracts):
            raise ValueError(
                "Configured context cannot cover shared-prefix and suffix test pages"
            )
        summary["alignment_tokens"] = alignment
        for rank in range(dp):
            seeds = [
                call(
                    decode + "/tokenize",
                    dict(
                        model=model,
                        prompt=f"Inventory ledger {tag} rank {rank}. Quantity 27.\n",
                    ),
                )["tokens"]
                for tag in ("ALPHA", "BRAVO")
            ]
            if any(not seed for seed in seeds):
                raise ValueError("Tokenizer returned an empty fixture")
            prompts = {
                tag: (seed * math.ceil((3 * alignment + 1) / len(seed)))[
                    : 2 * alignment + 1
                ]
                for tag, seed in zip(("A", "B"), seeds)
            }
            # Keep prompt A's boundary token so 1-token MTP lookahead block
            # hashes at 2 * alignment also match on multi-turn continuations.
            prompts["branch"] = (
                prompts["A"][: 2 * alignment + 1]
                + (seeds[1] * math.ceil(alignment / len(seeds[1])))[:alignment]
            )
            reference = {}
            for tag, prompt in prompts.items():
                reset(prefill)
                reset(decode)
                reference[tag] = measured(
                    rank, "cold-" + tag, prompt, (False, False), len(prompt) - 1
                )
            reset(prefill)
            reset(decode)
            for tag in ("A", "B"):
                measured(
                    rank,
                    "prime-" + tag,
                    prompts[tag],
                    (False, False),
                    len(prompts[tag]) - 1,
                    reference[tag],
                )
            for repeat in range(2):
                for tag in ("A", "B"):
                    measured(
                        rank,
                        f"both-warm-{repeat}-{tag}",
                        prompts[tag],
                        (True, True),
                        0,
                        reference[tag],
                    )
            reset(decode)
            for tag in ("A", "B"):
                measured(
                    rank,
                    "P-warm-" + tag,
                    prompts[tag],
                    (True, False),
                    len(prompts[tag]) - 1,
                    reference[tag],
                )
            reset(prefill)
            for tag in ("A", "B"):
                measured(
                    rank,
                    "D-warm-" + tag,
                    prompts[tag],
                    (False, True),
                    0,
                    reference[tag],
                )
            measured(
                rank,
                "partial-branch",
                prompts["branch"],
                (True, True),
                alignment,
                reference["branch"],
            )
            # No-load producer registrations must be released before a reset succeeds.
            reset(prefill)
            reset(decode)
            answers = CORRECTNESS.SHORT_QA[:4]
            for index in (0, 1, 0, 1, 2, 3, 2, 3):
                name, prompt, expected = answers[index]
                occurrence = sum(
                    case["case"].startswith("known-" + name)
                    for case in summary["cases"]
                    if case["rank"] == rank
                )
                body = dict(
                    model=model,
                    messages=[dict(role="user", content=prompt)],
                    max_tokens=64,
                    temperature=0,
                    seed=42,
                    request_id=f"apc-{run_id}-r{rank}-known-{name}-{occurrence}",
                    chat_template_kwargs=dict(enable_thinking=False),
                )
                receipt = dict(rank=rank, body=body, expected=expected)
                try:
                    raw_before, before = snapshot()
                    receipt["metrics_before"] = raw_before
                    response = call(proxy + "/v1/chat/completions", body, rank)
                    receipt["response"] = response
                    choice = response["choices"][0]
                    if (
                        len(response["choices"]) != 1
                        or choice["finish_reason"] != "stop"
                        or CORRECTNESS.normalize(choice["message"]["content"])
                        != CORRECTNESS.normalize(expected)
                    ):
                        raise ValueError("Independent known-answer criterion failed")
                    receipt["counter_deltas"] = [
                        {
                            f"{metric}/engine{engine}": value
                            for (metric, engine), value in delta.items()
                        }
                        for delta in settled_deltas(rank, before, receipt)
                    ]
                except Exception as error:
                    receipt["error"] = repr(error)
                    raise
                finally:
                    save(f"rank{rank}-known-{name}-{occurrence}", receipt)
                summary["completed"] += 1
                summary["cases"].append(
                    dict(rank=rank, case="known-" + name, passed=True)
                )
            if mtp:
                alt_token = next(
                    (tok for tok in seeds[1] if tok != prompts["A"][2 * alignment]),
                    prompts["A"][2 * alignment] + 1,
                )
                divergent = (
                    prompts["A"][: 2 * alignment]
                    + [alt_token]
                    + (seeds[1] * math.ceil(alignment / len(seeds[1])))[:alignment]
                )
                reset(prefill)
                reset(decode)
                cold_divergent = measured(
                    rank, "mtp-cold-divergent", divergent, (False, False), 3 * alignment
                )
                reset(prefill)
                reset(decode)
                mtp_cold_a = measured(
                    rank,
                    "mtp-spec-cold-A",
                    prompts["A"],
                    (False, False),
                    2 * alignment,
                    max_tokens=MTP_OUTPUT_TOKENS,
                    check_spec=True,
                )
                measured(
                    rank,
                    "mtp-spec-both-warm-A",
                    prompts["A"],
                    (True, True),
                    0,
                    mtp_cold_a,
                    max_tokens=MTP_OUTPUT_TOKENS,
                    check_spec=True,
                )
                measured(
                    rank,
                    "mtp-spec-partial-branch",
                    prompts["branch"],
                    (True, True),
                    alignment,
                    max_tokens=MTP_OUTPUT_TOKENS,
                    check_spec=True,
                )
                measured(
                    rank,
                    "mtp-boundary-divergent",
                    divergent,
                    (True, False),
                    3 * alignment,
                    cold_divergent,
                )
                reset(prefill)
                reset(decode)
        if summary["completed"] != summary["expected"]:
            raise ValueError("Incomplete correctness matrix")
        summary["passed"] = True
    except Exception as error:
        if not summary["failed"]:
            summary["failed"] = 1
        summary["errors"].append(repr(error))
        raise
    finally:
        save("summary", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--proxy", default="http://localhost:10000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    if args.dp <= 0:
        parser.error("--dp must be positive")
    run(
        args.prefill,
        args.decode,
        args.proxy,
        args.model,
        args.dp,
        args.output_dir,
        mtp=args.mtp,
    )


if __name__ == "__main__":
    main()
