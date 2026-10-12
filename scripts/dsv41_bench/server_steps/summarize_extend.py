# SPDX-License-Identifier: Apache-2.0
"""Strict extend-only host cadence accounting; no device-time inference."""

import statistics

from scenarios import SCENARIOS, token_counts


def summarize_extend(ranks: list[dict]) -> dict:
    scenario = SCENARIOS["encoder_extend"]
    if sorted(r["rank"] for r in ranks) != list(range(8)):
        raise ValueError("Expected exactly eight distinct DP ranks")
    if len({r["hostname"] for r in ranks}) != 1:
        raise ValueError("Monotonic timestamps require one host")
    ranks = sorted(ranks, key=lambda r: r["rank"])
    waves = []
    for rank in ranks:
        if rank["scenario"] != scenario.to_dict():
            raise ValueError("Worker scenario mismatch")
        steps = []
        previous_end = None
        for step_index, step in enumerate(rank["steps"]):
            if step.get("observation") != "native_output_consumption":
                raise ValueError(
                    "Requires naturally observed outputs, not old sync data"
                )
            if step["observed_ns"] <= step["start_ns"]:
                raise ValueError("Nonpositive observation interval")
            encoders = [s for s in step["stages"] if s["stage"] == "encoder"]
            if not encoders:
                continue
            if len(encoders) != 1:
                raise ValueError("Multi-forward step cannot qualify as extend")
            row = encoders[0]
            if row["phase"] != "prefill":
                continue
            if row["requests"] != 1 or row["prompts"] != [81920]:
                raise ValueError("Expected local C1 and 81920-token prompt")
            start, end = row["starts"][0], row["ends"][0]
            if previous_end is not None and previous_end != start:
                raise ValueError("Nonconsecutive context: missing or replayed chunk")
            if not 0 <= start < end <= 81920:
                raise ValueError("Invalid context")
            counts = token_counts([start], [end], 65536)
            if any(row[k] != v for k, v in counts.items()):
                raise ValueError("Token counters disagree with actual context")
            if row["rows"] != end - start or row["padded_rows"] < row["rows"]:
                raise ValueError("Invalid actual/padded rows")
            if previous_end is None:
                category = "encoder_first"
                if start > 65536 or end <= 65536:
                    raise ValueError(
                        "First chunk must cross the cached-prefix boundary"
                    )
            elif end == 81920:
                category = "encoder_tail"
            else:
                category = "encoder_middle"
                if (row["rows"], row["padded_rows"], row["replay_tokens"]) != (
                    2044,
                    2048,
                    0,
                ):
                    raise ValueError("Middle requires actual2044/pad2048 and no replay")
                if any(s["stage"] == "decoder" for s in step["stages"]):
                    raise ValueError("Decoder executed in a middle chunk")
            if steps and step_index != steps[-1]["step_index"] + 1:
                raise ValueError("Idle or mixed step interrupted the extend window")
            previous_end = end
            steps.append(
                {**step, "encoder": row, "category": category, "step_index": step_index}
            )
        if not steps or previous_end != 81920:
            raise ValueError("Incomplete extend wave")
        if sum(s["encoder"]["new_tokens"] for s in steps) != 16384:
            raise ValueError("Wave must account for exactly 16384 new tokens per rank")
        waves.append(steps)
    if len({len(w) for w in waves}) != 1:
        raise ValueError("Ranks executed different chunk counts")
    categories = {}
    previous = None
    for index, group in enumerate(zip(*waves, strict=True)):
        if (
            len(
                {
                    (
                        s["category"],
                        tuple(s["encoder"]["starts"]),
                        tuple(s["encoder"]["ends"]),
                    )
                    for s in group
                }
            )
            != 1
        ):
            raise ValueError("Ranks have different contexts or chunk categories")
        category = group[0]["category"]
        observed = max(s["observed_ns"] for s in group)
        # An observation-to-observation interval counts the CURRENT chunk only.
        # First chunk uses entry-to-observation and is reference-only.
        begin = min(s["start_ns"] for s in group) if previous is None else previous
        if observed <= begin:
            raise ValueError("Observations are not monotonically ordered")
        sample = {
            "chunk_index": index,
            "seconds": (observed - begin) / 1e9,
            "effective_tokens": sum(s["encoder"]["new_tokens"] for s in group),
            "start_ns": begin,
            "end_ns": observed,
            "rank_observed_ns": [s["observed_ns"] for s in group],
            "observation_skew_ms": (observed - min(s["observed_ns"] for s in group))
            / 1e6,
            "contexts": [s["encoder"]["starts"] for s in group],
            "rows": [s["encoder"]["rows"] for s in group],
            "padded_rows": [s["encoder"]["padded_rows"] for s in group],
            "observation_paths": [s["output_path"] for s in group],
        }
        categories.setdefault(category, []).append(sample)
        previous = observed
    if len(categories.get("encoder_middle", [])) < 2:
        raise ValueError("Need consecutive full-DP middle chunks")
    report = {
        "scenario": scenario.to_dict(),
        "timing": "natural_server_output_host_cadence",
        "device_seconds": None,
        "capacity_qualified": False,
        "limitation": (
            "Intermediate output may not wait for device completion; "
            "host cadence is not isolated device or D-node capacity."
        ),
        "categories": {},
    }
    for category, samples in categories.items():
        seconds = sum(s["seconds"] for s in samples)
        tokens = sum(s["effective_tokens"] for s in samples)
        rate = tokens / seconds
        report["categories"][category] = {
            "steps": len(samples),
            "median_step_ms": statistics.median(s["seconds"] for s in samples) * 1000,
            "total_seconds": seconds,
            "effective_tokens": tokens,
            "global_tokens_s": rate,
            "per_device_tokens_s": rate / 8,
            "per_physical_chip_tokens_s": rate / 4,
            "token_kind": "new_input",
            "samples": samples,
        }
    return report
