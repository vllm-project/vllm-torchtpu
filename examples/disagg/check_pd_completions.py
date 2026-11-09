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
"""PD transport checks with separate exact token diagnostics, per DP rank."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

LENGTHS = (127, 128, 129, 130, 8191, 8192, 8193, 8194, 32768)
OUTPUT_TOKENS = 16


def post(url: str, body: dict, rank: int) -> dict:
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def completion_ids(response: dict, prompt_length: int) -> list[int]:
    if response.get("error") or len(response.get("choices", [])) != 1:
        raise ValueError("Completion error or missing/extra choices")
    choice = response["choices"][0]
    tokens = choice.get("token_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != OUTPUT_TOKENS
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("Completion must contain every generated token ID")
    usage = response["usage"]
    if (
        usage["prompt_tokens"] != prompt_length
        or usage["completion_tokens"] != OUTPUT_TOKENS
        or usage["total_tokens"] != prompt_length + OUTPUT_TOKENS
        or choice["finish_reason"] != "length"
    ):
        raise ValueError("Completion prompt/output/finish accounting mismatch")
    return tokens


def triplet_diagnostic(
    reference: dict, first: dict, repeated: dict, length: int
) -> dict:
    expected = completion_ids(reference, length)
    actual = completion_ids(first, length)
    again = completion_ids(repeated, length)
    for response in (first, repeated):
        details = response["usage"].get("prompt_tokens_details", {})
        if details.get("cached_tokens") != length - 1:
            raise ValueError("Decode did not consume the complete remote prefix")
    return dict(reference_passed=actual == expected, repeat_passed=actual == again)


def check_triplet(reference: dict, first: dict, repeated: dict, length: int) -> None:
    diagnostic = triplet_diagnostic(reference, first, repeated, length)
    if not diagnostic["repeat_passed"]:
        raise ValueError("Repeated PD token IDs differ")
    if not diagnostic["reference_passed"]:
        raise ValueError("PD token IDs differ from decode-local reference")


def run_checks(
    proxy: str,
    prefill: str,
    decode: str,
    model: str,
    dp: int,
    output: Path,
    numeric_mode: str = "report",
) -> dict:
    if numeric_mode not in ("report", "strict"):
        raise ValueError("Expected report or strict numeric mode")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    summary: dict[str, Any] = dict(
        passed=False,
        performance_checks_passed=False,
        numeric_mode=numeric_mode,
        numeric_reference_passed=None,
        numeric_repeat_passed=None,
        diagnostic_failed=False,
        numeric_diagnostics=[],
        completed=0,
        failed=0,
        expected=dp * len(LENGTHS) * 3,
        requested_ranks=list(range(dp)),
        errors=[],
    )

    def save(name: str, value: dict) -> None:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    def request(
        rank: int, length: int, label: str, prompt: list[int], url: str
    ) -> dict:
        key = f"rank{rank}-n{length}-{label}"
        body = dict(
            model=model,
            prompt=prompt,
            max_tokens=OUTPUT_TOKENS,
            temperature=0,
            seed=42,
            ignore_eos=True,
            stream=False,
            return_token_ids=True,
            request_id=key,
        )
        receipt = dict(
            requested_rank=rank,
            length=length,
            phase=label,
            body=body,
            prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        )
        try:
            receipt["response"] = post(url + "/v1/completions", body, rank)
        except Exception as error:
            receipt["error"] = repr(error)
            raise
        finally:
            save(key + ".json", receipt)
        return receipt

    try:
        for length in LENGTHS:
            prompts = []
            for rank in range(dp):
                seed = dict(
                    model=model,
                    prompt=f"Explain the sequence for worker {rank}. ",
                    add_special_tokens=False,
                )
                # Verify both tokenizers agree before constructing exact-length IDs.
                p_tokens = post(prefill + "/tokenize", seed, rank)["tokens"]
                d_tokens = post(decode + "/tokenize", seed, rank)["tokens"]
                save(
                    f"rank{rank}-n{length}-tokenize.json",
                    dict(body=seed, prefill_tokens=p_tokens, decode_tokens=d_tokens),
                )
                if not p_tokens or p_tokens != d_tokens:
                    raise ValueError("PD tokenizer identities differ")
                prompts.append(
                    (p_tokens * ((length + len(p_tokens) - 1) // len(p_tokens)))[
                        :length
                    ]
                )
            rounds = []
            for label, endpoint in [
                ("reference", decode),
                ("first", proxy),
                ("repeat", proxy),
            ]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=dp) as executor:
                    futures = [
                        executor.submit(
                            request, rank, length, label, prompts[rank], endpoint
                        )
                        for rank in range(dp)
                    ]
                    rows = []
                    failure = None
                    for future in futures:
                        try:
                            row = future.result()
                            rows.append(row)
                            results.append(row)
                            summary["completed"] += 1
                        except Exception as error:
                            summary["failed"] += 1
                            if failure is None:
                                failure = error
                    if failure is not None:
                        raise failure
                rounds.append(rows)
            for rank in range(dp):
                diagnostic = triplet_diagnostic(
                    rounds[0][rank]["response"],
                    rounds[1][rank]["response"],
                    rounds[2][rank]["response"],
                    length,
                )
                summary["numeric_diagnostics"].append(
                    dict(rank=rank, prompt_tokens=length, **diagnostic)
                )
                for field, value in (
                    ("numeric_reference_passed", diagnostic["reference_passed"]),
                    ("numeric_repeat_passed", diagnostic["repeat_passed"]),
                ):
                    summary[field] = summary[field] is not False and value
                summary["diagnostic_failed"] = not (
                    summary["numeric_reference_passed"]
                    and summary["numeric_repeat_passed"]
                )
                if numeric_mode == "strict":
                    check_triplet(
                        rounds[0][rank]["response"],
                        rounds[1][rank]["response"],
                        rounds[2][rank]["response"],
                        length,
                    )
        if summary["completed"] != summary["expected"]:
            raise ValueError("Missing completion receipts")
        summary["performance_checks_passed"] = True
        summary["passed"] = not summary["diagnostic_failed"]
    except Exception as error:
        summary["errors"].append(repr(error))
        raise
    finally:
        summary["unattempted"] = (
            summary["expected"] - summary["completed"] - summary["failed"]
        )
        save("results.json", dict(requests=results))
        save("summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp-size", choices=[4, 8], type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--numeric-mode",
        choices=("report", "strict"),
        default="report",
        help=(
            "Report exact token diagnostics separately; strict also fails on mismatch."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_checks(
                args.proxy_url,
                args.prefill_url,
                args.decode_url,
                args.model,
                args.dp_size,
                args.output_dir,
                args.numeric_mode,
            )
        )
    )


if __name__ == "__main__":
    main()
