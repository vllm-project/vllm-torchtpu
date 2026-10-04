"""Decoder replay planning and its bounded SWA state for shared CSA."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int


def plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    """Keep the same prompt-tail window across every chunk and later decode.

    The encoder publishes global KV for the complete chunk before this plan
    narrows the decoder batch. Prefix caching must be disabled: a cached prefix
    may not contain the encoder outputs required to rebuild decoder SWA.
    """
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None  # Profiling/warmup executes both stages at the full bucket.
    if window < 1 or len(bounds) != len(lengths) + 1 or len(prompts) != len(lengths):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds[0] != 0 or bounds[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    indices, replay_starts, output_bounds = [], [], [0]
    for start, end, length, prompt in zip(bounds, bounds[1:], lengths, prompts):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = max(0, prompt - window)
        skip = min(count, max(0, replay_start - (length - count)))
        indices.extend(range(start + skip, end))
        replay_starts.append(replay_start)
        output_bounds.append(len(indices))
    kept = len(indices)
    # Each stage dispatches by its own token bucket. Never introduce a second,
    # independent symbolic token dimension into vLLM's guard-free full graph.
    buckets = sorted({num_tokens, *(int(n) for n in compile_sizes if int(n) > 0)})
    padded = next(n for n in buckets if n >= kept) if kept else 0
    return DecoderReplayPlan(
        tuple(indices), tuple(output_bounds), tuple(replay_starts), padded
    )
