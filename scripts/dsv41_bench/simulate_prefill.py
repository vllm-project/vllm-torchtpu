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
"""Closed-loop host scheduling model over metadata from completed replay turns.

This is a scheduling model, not the vLLM scheduler or a device simulator.
Work is fixed between policies; successor turns become ready on completion.
Unknown cache-miss lengths and shape timings remain explicit assumptions.
"""

import argparse
import ast
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class Turn:
    slot: int
    rank: int
    prefill: int
    output: int


@dataclass
class Active:
    turn: Turn
    remaining_prefill: int
    remaining_output: float


@dataclass(frozen=True)
class Policy:
    name: str
    buckets: tuple[int, ...] = (288, 512, 2048)
    interval: int = 1
    strict_window: int = 0
    token_budget: int = 2048
    input_budget: int = 2048


def load_replay_plan(root: Path):
    """Load the unchanged CED planner, bypassing torch/vLLM registration."""
    path = root / "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/ced.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("DecoderReplayPlan", "plan_decoder_replay")
    ]
    namespace = {"dataclass": dataclass, "__name__": __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["plan_decoder_replay"]


def bucket(rows: int, sizes: tuple[int, ...]) -> int:
    if rows == 0:
        return 0
    return next(size for size in sizes if size >= rows)


def workload(
    data: dict,
    dp: int,
    prefill_model: str,
    match_output_mean: bool = False,
    output_scale: float | None = None,
) -> tuple[dict, dict]:
    """Freeze observed output lengths, infer cache misses with stated models."""
    summary = data["summary"]
    metrics = summary["metrics_steady"]
    target = (
        metrics["prompt_tokens_s"]
        * (1 - metrics["prefix_cache_hit_rate"])
        * metrics["seconds"]
        / summary["responses_steady"]["requests"]
    )
    rows = [
        row
        for row in data["requests"]
        if row["outcome"] == "completed" and row["rank"] < dp
    ]
    prompts = {(row["trajectory"], row["turn"]): row["prompt_tokens"] for row in rows}
    chains = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["started"]):
        estimate = target
        if prefill_model == "delta":
            previous = prompts.get((row["trajectory"], row["turn"] - 1))
            if row["turn"] == 0:
                estimate = row["prompt_tokens"]
            elif previous is not None:
                estimate = min(
                    row["prompt_tokens"],
                    max(128, row["prompt_tokens"] - previous + 128),
                )
        chains[row["slot"]].append(
            Turn(
                row["slot"],
                row["rank"],
                max(128, round(estimate)),
                row["completion_tokens"],
            )
        )
    observed_mean = sum(t.output for c in chains.values() for t in c) / len(rows)
    stationary_mean = (
        metrics["generation_tokens_s"]
        * metrics["seconds"]
        / summary["responses_steady"]["requests"]
    )
    scale = stationary_mean / observed_mean if match_output_mean else 1.0
    if output_scale is not None:
        scale = output_scale
    maximum_output = max(int(value) for value in summary["source"]["max_tokens"])
    if scale != 1.0:
        chains = {
            slot: [
                replace(t, output=min(maximum_output, max(1, round(t.output * scale))))
                for t in chain
            ]
            for slot, chain in chains.items()
        }
    # Requests with no completion before the deadline have no known length.
    # For a steady-state surrogate, fill those slots using same-rank samples.
    imputed = 0
    for rank in range(dp):
        available = [c for c in chains.values() if c[0].rank == rank]
        for index in range(48 - len(available)):
            slot = 100_000 + rank * 48 + index
            chains[slot] = [
                replace(t, slot=slot) for t in available[index % len(available)]
            ]
            imputed += 1
    return dict(chains), {
        "prefill_model": prefill_model,
        "aggregate_miss_tokens_per_new_request": target,
        "modeled_mean_prefill": sum(t.prefill for c in chains.values() for t in c)
        / sum(map(len, chains.values())),
        "observed_completed_turns": len(rows),
        "modeled_turns": sum(map(len, chains.values())),
        "slots": len(chains),
        "imputed_slots": imputed,
        "observed_completed_output_mean": observed_mean,
        "stationary_output_per_arrival_estimate": stationary_mean,
        "output_length_scale": scale,
        "maximum_output_tokens": maximum_output,
        "excluded_censored_or_other_rank_rows": len(data["requests"]) - len(rows),
        "known_cache_miss_rows": sum(row["cached_tokens"] is not None for row in rows),
    }


def simulate(
    chains: dict[int, list[Turn]],
    policy: Policy,
    dp: int,
    accepted_per_step: float,
    replay_plan,
    capacity: int = 48,
    measure_steps: int | None = None,
    warmup_steps: int = 0,
    draft_slots: int = 4,
) -> dict:
    """Model bounded active slots, prefill chunks and synchronized stage buckets."""
    pending = [deque() for _ in range(dp)]
    active: list[list[Active]] = [[] for _ in range(dp)]
    indices = dict.fromkeys(chains, 0)
    for chain in chains.values():
        pending[chain[0].rank].append((chain[0], 0))
    drain = [False] * dp
    histogram = Counter()
    waits = []
    output = completed = steps = decode_slots = prefill_tokens = 0
    effective_encoder = effective_decoder = padded_encoder = padded_decoder = 0

    @lru_cache(maxsize=65536)
    def kept_rows(remaining: int, count: int) -> int:
        metadata = SimpleNamespace(
            query_start_loc_cpu=(0, count),
            seq_lens_cpu=(count,),
            prompt_lens_cpu=(remaining,),
            replay_starts_cpu=(0,),
        )
        return len(replay_plan(metadata, 128, count, ()).indices)

    def complete(turn: Turn) -> None:
        nonlocal completed
        completed += 1
        indices[turn.slot] += 1
        chain = chains[turn.slot]
        if measure_steps is not None or indices[turn.slot] < len(chain):
            successor = chain[indices[turn.slot] % len(chain)]
            pending[successor.rank].append((successor, steps + 1))

    while any(pending) or any(active):
        if measure_steps is not None and steps == warmup_steps:
            histogram.clear()
            waits.clear()
            output = completed = decode_slots = prefill_tokens = 0
            effective_encoder = effective_decoder = padded_encoder = padded_decoder = 0
        if measure_steps is not None and steps == warmup_steps + measure_steps:
            break
        if steps >= 1_000_000:
            raise RuntimeError("Scheduling model made insufficient progress")
        enc, dec = [], []
        finished = []
        for rank in range(dp):
            # Future successors are appended only after all ranks schedule.
            aligned = steps % policy.interval == 0
            if not policy.strict_window:
                if aligned:
                    drain[rank] = True
                if drain[rank] and not pending[rank]:
                    drain[rank] = False
                admit = drain[rank]
                continue_prefill = True
            else:
                admit = continue_prefill = (
                    steps % policy.interval < policy.strict_window
                )
            decoding = [item for item in active[rank] if not item.remaining_prefill]
            decode_slots += len(decoding)
            rows = decoder_rows = 6 * len(decoding)
            budget = policy.token_budget - rows
            input_budget = policy.input_budget - (6 + draft_slots) * len(decoding)
            assert min(budget, input_budget) >= 0
            for item in decoding:
                advance = min(item.remaining_output, accepted_per_step)
                item.remaining_output -= advance
                output += advance
                if item.remaining_output <= 1e-9:
                    finished.append(item)
            prefilling = [item for item in active[rank] if item.remaining_prefill]
            if continue_prefill:
                for item in prefilling:
                    count = min(
                        item.remaining_prefill,
                        budget,
                        max(0, input_budget - draft_slots),
                    )
                    decoder_rows += kept_rows(item.remaining_prefill, count)
                    item.remaining_prefill -= count
                    rows += count
                    prefill_tokens += count
                    budget -= count
                    if count:
                        input_budget -= count + draft_slots
                    if item.remaining_prefill == 0:
                        item.remaining_output -= 1
                        output += 1
                        if item.remaining_output <= 0:
                            finished.append(item)
            while (
                admit
                and budget
                and input_budget > draft_slots
                and pending[rank]
                and len(active[rank]) < capacity
            ):
                turn, ready = pending[rank].popleft()
                waits.append(steps - ready)
                count = min(turn.prefill, budget, input_budget - draft_slots)
                decoder_rows += kept_rows(turn.prefill, count)
                item = Active(turn, turn.prefill - count, float(turn.output))
                active[rank].append(item)
                rows += count
                prefill_tokens += count
                budget -= count
                input_budget -= count + draft_slots
                if item.remaining_prefill == 0:
                    item.remaining_output -= 1
                    output += 1
                    if item.remaining_output <= 0:
                        finished.append(item)
            enc.append(rows)
            dec.append(decoder_rows)
        encoder_bucket = bucket(max(enc), policy.buckets)
        decoder_bucket = bucket(max(dec), policy.buckets)
        histogram[f"{encoder_bucket}/{decoder_bucket}"] += 1
        effective_encoder += sum(enc)
        effective_decoder += sum(dec)
        padded_encoder += encoder_bucket * dp
        padded_decoder += decoder_bucket * dp
        for item in finished:
            active[item.turn.rank].remove(item)
            complete(item.turn)
        steps += 1
    if measure_steps is None:
        expected = sum(t.output for chain in chains.values() for t in chain)
        assert abs(output - expected) < 1e-4, (output, expected)
        assert completed == sum(map(len, chains.values()))
        assert prefill_tokens == sum(
            t.prefill for chain in chains.values() for t in chain
        )
    steps = measure_steps if measure_steps is not None else steps
    return {
        "policy": asdict(policy),
        "steps": steps,
        "stage_bucket_steps": dict(sorted(histogram.items())),
        "mean_decode_requests_per_rank": decode_slots / steps / dp,
        "mean_admission_wait_steps": sum(waits) / len(waits),
        "max_admission_wait_steps": max(waits),
        "encoder_valid_row_fraction": effective_encoder / padded_encoder,
        "decoder_valid_row_fraction": effective_decoder / padded_decoder,
        "completed_turns": completed,
        "output_tokens": output,
        "prefill_tokens": prefill_tokens,
    }


def sensitivity(
    results: list[dict], data: dict, cost_reference: dict | None = None
) -> list[dict]:
    """Fit only historical 2048 cost; vary unmeasured stage and 512 costs."""
    metrics = data["summary"]["metrics_steady"]
    decode_ms = data["formal"]["categories"]["decode"]["median_step_ms"]
    mixed_ms = metrics["engine_step_interval_ms"]["median"]
    baseline = results[0]
    timing_baseline = cost_reference["results"][0] if cost_reference else baseline
    scenarios = []
    for encoder_share in (0.35, 0.5, 0.65):
        big_fraction = (
            sum(
                count
                * (
                    encoder_share * (key.split("/")[0] == "2048")
                    + (1 - encoder_share) * (key.split("/")[1] == "2048")
                )
                for key, count in timing_baseline["stage_bucket_steps"].items()
            )
            / timing_baseline["steps"]
        )
        large_ratio = 1 + (mixed_ms / decode_ms - 1) / big_fraction
        for middle_ratio in (1.5, 2.0, 4.0, 8.0, 16.0):
            costs = {0: 0.0, 288: 1.0, 512: middle_ratio, 2048: large_ratio}
            times = {}
            for result in results:
                ms = sum(
                    count
                    * decode_ms
                    * (
                        encoder_share * costs[int(key.split("/")[0])]
                        + (1 - encoder_share) * costs[int(key.split("/")[1])]
                    )
                    for key, count in result["stage_bucket_steps"].items()
                )
                times[result["policy"]["name"]] = ms / 1000
            scenarios.append(
                {
                    "encoder_share": encoder_share,
                    "assumed_512_to_288_ratio": middle_ratio,
                    "fitted_2048_to_288_ratio": large_ratio,
                    "modeled_seconds": times,
                    "relative_to_baseline": {
                        result["policy"]["name"]: (
                            times[result["policy"]["name"]] / result["output_tokens"]
                        )
                        / (times["immediate_2bucket"] / baseline["output_tokens"])
                        for result in results
                    },
                }
            )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp-size", type=int, choices=(8, 16), default=8)
    parser.add_argument("--prefill-model", choices=("fixed", "delta"), default="fixed")
    parser.add_argument("--match-output-mean", action="store_true")
    parser.add_argument("--calibrate-arrivals", action="store_true")
    parser.add_argument(
        "--cost-reference",
        type=Path,
        help="DP16 simulation used to fit historical step costs",
    )
    parser.add_argument("--measure-steps", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=512)
    parser.add_argument(
        "--draft-slots",
        type=int,
        default=4,
        help="Pinned DSpark K5 charges K-1; main #2208 uses 0",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    source_dp = data["summary"]["slots"] / data["summary"]["per_rank"]
    cost_reference = None
    if args.cost_reference is not None:
        cost_reference = json.loads(args.cost_reference.read_text())
        if cost_reference["source_sha256"] != data["source_sha256"]:
            parser.error("--cost-reference must use the same replay artifact")
        if cost_reference["dp_size"] != source_dp:
            parser.error("--cost-reference must match the measured replay DP size")
    chains, inputs = workload(
        data, args.dp_size, args.prefill_model, args.match_output_mean
    )
    plan = load_replay_plan(Path(__file__).resolve().parents[2])
    policies = [
        Policy("immediate_2bucket", buckets=(288, 2048)),
        Policy("buckets_only"),
    ]
    for interval in (4, 8, 16):
        policies.append(Policy(f"cadence_{interval}", interval=interval))
        policies.append(
            Policy(f"strict_{interval}", interval=interval, strict_window=1)
        )
        policies.append(
            Policy(f"window2_{interval}", interval=interval, strict_window=2)
        )
    policies.append(Policy("rowcap512", token_budget=512))
    policies.append(Policy("budget512", token_budget=512, input_budget=512))
    accepted = (
        1 + data["summary"]["metrics_steady"]["mean_accepted_draft_tokens_per_step"]
    )
    calibration = []
    if args.calibrate_arrivals:
        metrics = data["summary"]["metrics_steady"]
        observed_rate = (
            data["summary"]["responses_steady"]["requests"]
            / source_dp
            / (metrics["seconds"] * 1000 / metrics["engine_step_interval_ms"]["median"])
        )
        for iteration in range(8):
            probe = simulate(
                chains,
                policies[0],
                args.dp_size,
                accepted,
                plan,
                measure_steps=args.measure_steps,
                warmup_steps=args.warmup_steps,
                draft_slots=args.draft_slots,
            )
            modeled_rate = probe["completed_turns"] / probe["steps"] / args.dp_size
            calibration.append(
                {
                    "output_length_scale": inputs["output_length_scale"],
                    "observed_arrivals_per_rank_step": observed_rate,
                    "modeled_completions_per_rank_step": modeled_rate,
                }
            )
            if abs(modeled_rate / observed_rate - 1) < 0.03 or iteration == 7:
                break
            chains, inputs = workload(
                data,
                args.dp_size,
                args.prefill_model,
                output_scale=inputs["output_length_scale"]
                * modeled_rate
                / observed_rate,
            )
    results = [
        simulate(
            chains,
            p,
            args.dp_size,
            accepted,
            plan,
            measure_steps=args.measure_steps,
            warmup_steps=args.warmup_steps,
            draft_slots=args.draft_slots,
        )
        for p in policies
    ]
    report = {
        "scope": "simulated; cyclic completed-turn cohort, closed-loop successors",
        "limitations": [
            "Not vLLM AsyncScheduler: no KV allocator, async queue "
            "or sampling execution",
            "Right-censored requests excluded; recorded completed outputs "
            "are length-biased",
            "Missing slots use same-rank samples; optional aggregate length "
            "scaling is inferred",
            "Initial turns ready together; no client think time; APC misses inferred",
            "CED planner is reused; kernel timing, padding effects and warmup "
            "are not executed",
            "DP16 formal decode and aggregate replay timing anchor "
            "hypothetical DP8 costs",
            "512 cost and encoder/decoder shares are unmeasured sensitivity parameters",
        ],
        "source_sha256": data["source_sha256"],
        "dp_size": args.dp_size,
        "warmup_steps": args.warmup_steps,
        "charged_draft_slots_per_request": args.draft_slots,
        "cost_calibration_dp_size": (
            cost_reference["dp_size"] if cost_reference else args.dp_size
        ),
        "inputs": inputs,
        "arrival_calibration": calibration,
        "accepted_output_tokens_per_decode_step": accepted,
        "results": results,
        "sensitivity": (
            sensitivity(results, data, cost_reference)
            if args.dp_size == source_dp or cost_reference is not None
            else []
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
