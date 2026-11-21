"""Host planning for decoder SWA bounded replay across scheduled chunks."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from vllm.distributed import get_dp_group


def prefix_replay_enabled(vllm_config) -> bool:
    """Whether prefix caching runs through SWA bounded replay.

    Upstream's contract: the SWA group leaves prefix caching, and the
    scheduler resumes each hit one window early so replay rebuilds it. This
    holds for CED and full execution alike. Replay requires unsharded token
    rows, so prefill context parallelism stays out.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return bool(
        getattr(cache_config, "enable_prefix_caching", False)
        and getattr(cache_config, "swa_bounded_replay", False)
        and getattr(parallel_config, "prefill_context_parallel_size", 1) == 1
    )


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int


def start_decoder_replay_coordination(
    plan: DecoderReplayPlan | None, num_tokens: int
) -> Callable[[], DecoderReplayPlan | None]:
    """Start agreeing on one decoder row bucket across EP participants.

    The returned call waits for the DP all-reduce and gives the coordinated
    plan, so the caller can enqueue other work while the replicas meet.
    """
    group = get_dp_group()
    if group.world_size == 1:
        return lambda: plan
    count = torch.tensor(
        [num_tokens if plan is None else plan.padded_num_tokens],
        dtype=torch.int64,
        device="cpu",
    )
    work = torch.distributed.all_reduce(
        count, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group, async_op=True
    )

    def finish() -> DecoderReplayPlan | None:
        work.wait()
        return replace(plan, padded_num_tokens=int(count.item())) if plan else None

    return finish


def decoder_buckets(num_tokens: int, compile_sizes) -> list[int]:
    """The decoder row buckets a plan for a ``num_tokens`` step can pick.

    Each stage dispatches by its own token bucket. Never introduce a second,
    independent symbolic token dimension into vLLM's guard-free full graph.
    """
    sizes = {int(n) for n in compile_sizes if 0 < int(n) <= num_tokens}
    return sorted({num_tokens, *sizes})


def plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    """Keep the same prompt-tail window across every chunk and later decode.

    The encoder publishes global KV for the complete chunk before this plan
    narrows the decoder batch. With prefix caching, the cache holds no SWA
    rows, so the scheduler resumes a hit at its replay start (SWA bounded
    replay): the decoder window starts no earlier than that.
    """
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    hits = getattr(metadata, "replay_starts_cpu", None) or (0,) * len(lengths or ())
    if bounds is None or lengths is None or prompts is None:
        return None  # Profiling/warmup executes both stages at the full bucket.
    if (
        window < 1
        or len(bounds) != len(lengths) + 1
        or len(prompts) != len(lengths)
        or len(hits) != len(lengths)
    ):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds[0] != 0 or bounds[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    indices, replay_starts, output_bounds = [], [], [0]
    for start, end, length, prompt, hit in zip(
        bounds, bounds[1:], lengths, prompts, hits
    ):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = max(0, prompt - window, hit)
        skip = min(count, max(0, replay_start - (length - count)))
        indices.extend(range(start + skip, end))
        replay_starts.append(replay_start)
        output_bounds.append(len(indices))
    kept = len(indices)
    padded = (
        next(n for n in decoder_buckets(num_tokens, compile_sizes) if n >= kept)
        if kept
        else 0
    )
    return DecoderReplayPlan(
        tuple(indices), tuple(output_bounds), tuple(replay_starts), padded
    )
