import pytest

from vllm_torchtpu.core.pp_chunks import (StepCostModel, chunk_buckets,
                                          chunk_granularity, chunk_pairs,
                                          profile_points)

BUCKETS = [16, 4096, 8192, 12288, 16384]
ATTN = 1.5e-7  # ms per token of chunk per token of context


def _linear(tokens: int) -> float:
    """Prefix-free time of one request without its own causal attention."""
    return 12.0 + 0.0109 * tokens


def _measured(tokens: int, prefix: int, scale: float = 1.0) -> float:
    """What one stage reports: the ladder plus attention over the prefix
    and over the chunk itself."""
    return scale * (_linear(tokens) + ATTN *
                    (tokens * prefix + tokens * tokens / 2))


def _stage(scale=1.0):
    out = []
    for tokens in BUCKETS:
        out.append((tokens, 0, _measured(tokens, 0, scale)))
        for prefix in (16384, 32768):
            out.append((tokens, prefix, _measured(tokens, prefix, scale)))
    return out


def _model(slack=0.1, granularity=4096) -> StepCostModel:
    return StepCostModel.fit([_stage(1.0), _stage(0.9)], slack, granularity)


def test_granularity_is_an_eighth_step_in_whole_blocks_when_aligned():
    assert chunk_granularity(4096, 16384, True) == 4096
    assert chunk_granularity(4352, 16384, True) == 4352
    assert chunk_granularity(16, 16384, True) == 2048
    assert chunk_granularity(1024, 16384, True) == 2048
    assert chunk_granularity(4096, 8192, True) == 4096


def test_granularity_ignores_the_block_without_alignment_or_above_a_step():
    assert chunk_granularity(4352, 16384, False) == 2048
    assert chunk_granularity(1114112, 16384, True) == 2048
    assert chunk_granularity(8192, 4096, True) == 512


def test_buckets_cover_every_granule_multiple():
    assert chunk_buckets(16384, 4096) == [4096, 8192, 12288]
    assert chunk_buckets(16384, 4352) == [4352, 8704, 13056]
    assert chunk_buckets(4096,
                         512) == [512, 1024, 1536, 2048, 2560, 3072, 3584]
    assert chunk_buckets(4096, 4096) == []


def test_profile_points_cover_every_bucket_and_prefixes_for_large_ones():
    points = profile_points(BUCKETS, 16384, 65536)
    assert [p for p in points if p[1] == 0] == [(b, 0) for b in BUCKETS]
    assert (16, 16384) not in points
    assert (4096, 16384) in points and (4096, 32768) in points
    assert (16384, 16384) in points and (16384, 32768) in points
    # Every point stays a step short of the model length.
    assert max(t + p for t, p in points) <= 65536 - 16384
    clipped = profile_points(BUCKETS, 16384, 40960)
    assert (16384, 8192) in clipped and (16384, 16384) not in clipped


def test_profile_points_skip_prefixes_beyond_the_schedule_capacity():
    # 16K behind 32K needs 64 * 128 + 2080 = 10,272 pairs of 256x256 tiles,
    # 16K behind 64K needs 64 * 256 + 2080 = 18,464.
    points = profile_points(BUCKETS, 32768, 262144, (13436, 256, 256))
    assert [p for p in points if p[1] == 0] == [(b, 0) for b in BUCKETS]
    assert (16384, 32768) in points and (16384, 65536) not in points
    assert (8192, 32768) in points and (8192, 65536) in points
    assert chunk_pairs((13436, 256, 256), 16384, 32768) == 10272
    assert chunk_pairs(None, 16384, 32768) == 0


def test_fit_takes_the_slowest_stage_and_removes_the_causal_term():
    model = _model()
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)
    assert model.linear_ms[16384] == pytest.approx(_linear(16384), rel=1e-6)
    assert model.target_ms == pytest.approx(1.1 * _measured(16384, 0))
    assert model.granularity == 4096
    assert model.buckets == BUCKETS


def test_fit_uses_the_median_slope_and_ignores_a_faster_prefix_run():
    stage = _stage()
    # One bucket whose far prefix point ran faster than its near one.
    stage = [(t, p, ms if (t, p) != (16384, 32768) else 100.0)
             for t, p, ms in stage]
    model = StepCostModel.fit([stage], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)


def test_fit_slope_ignores_an_offset_on_the_prefix_free_points():
    # Every prefix-free point carries 14 ms that the prefix points lack.
    stage = [(t, p, ms + (14.0 if p == 0 else 0.0)) for t, p, ms in _stage()]
    model = StepCostModel.fit([stage], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)
    # A bucket with a single prefix point falls back to its prefix-free one.
    single = [(t, p, ms) for t, p, ms in _stage() if p != 32768]
    model = StepCostModel.fit([single], 0.1, 4096)
    assert model.attn_ms_per_token_pair == pytest.approx(ATTN, rel=1e-6)


def test_fit_rejects_negative_slack_empty_input_and_mismatched_ladders():
    with pytest.raises(ValueError, match="slack"):
        StepCostModel.fit([_stage()], -0.1, 4096)
    with pytest.raises(ValueError, match="no stage"):
        StepCostModel.fit([], 0.1, 4096)
    with pytest.raises(ValueError, match="different token buckets"):
        StepCostModel.fit([_stage(), [(16384, 0, 190.0)]], 0.1, 4096)


def test_step_time_adds_the_causal_term_of_each_chunk():
    model = _model()
    # Four 4K first chunks cost less than one 16K chunk in the same bucket.
    assert model.step_ms(16384, 0.0, 4 * 4096.0**2) < model.step_ms(
        16384, 0.0, 16384.0**2)


def test_first_chunk_fills_the_step():
    model = _model()
    assert model.chunk(0, 16384, 0, 0.0, 0.0) == 16384
    assert model.chunk(0, 8192, 8192, 0.0, 8192.0**2) == 8192


def test_chunk_behind_a_prefix_shrinks_to_a_bucket_that_fits():
    model = _model()
    # 16K behind 16K costs 41 ms over a full step: over the 10% slack.
    assert model.chunk(16384, 16384, 0, 0.0, 0.0) == 12288
    # Deep in a prompt only a granule fits.
    assert model.chunk(112 * 1024, 16384, 0, 0.0, 0.0) == 4096
    # The last chunk of a prompt need not be aligned.
    assert model.chunk(16384, 5000, 0, 0.0, 0.0) == 5000


def test_a_clipped_chunk_is_at_least_a_quarter_step():
    model = _model(granularity=2048)
    # Very deep in a prompt the time budget would allow under 2K tokens.
    assert model.chunk(200 * 1024, 16384, 0, 0.0, 0.0) == 4096
    # The floor never exceeds the room left in the step.
    assert model.chunk(200 * 1024, 16384, 14336, 0.0, 0.0) == 2048
    assert model.chunk(200 * 1024, 16384, 16384, 0.0, 0.0) == 0


def test_a_step_limit_holds_the_step_to_a_smaller_bucket():
    # Buckets at every granule multiple, as the runner compiles them.
    buckets = list(range(2048, 16385, 2048))
    linear = {b: _linear(b) for b in buckets}
    model = StepCostModel(buckets, linear, ATTN, 1.1 * _measured(16384, 0),
                          2048)
    # A first chunk fills the limited step; a short request is untouched.
    assert model.chunk(0, 65536, 0, 0.0, 0.0, step_limit=4096) == 4096
    assert model.chunk(0, 1000, 0, 0.0, 0.0, step_limit=4096) == 1000
    # The limit wins over the quarter-step floor deep in a prompt.
    assert model.chunk(200 * 1024, 16384, 0, 0.0, 0.0, step_limit=2048) == 2048
    # 37 decode tokens in the step: the total still lands on the limit.
    assert model.chunk(0, 65536, 37, 0.0, 37.0, step_limit=4096) == 4096 - 37
    # A second request finds the limited step full.
    assert model.chunk(0, 65536, 4096, 0.0, 4096.0**2, step_limit=4096) == 0


def test_chunk_rounds_the_step_total_to_a_granule():
    model = _model()
    # 37 decode tokens sit in the step: a first chunk lands the total on
    # 16384, a chunk behind 16K on 12288.
    assert model.chunk(0, 65536, 37, 0.0, 37.0) == 16384 - 37
    assert model.chunk(16384, 65536, 37, 37.0 * 16384, 37.0) == 12288 - 37


def test_later_requests_get_what_the_step_has_left():
    model = _model()
    # A 4K chunk behind 16K costs 10 ms; a first chunk still fills the step.
    assert model.chunk(0, 16384, 4096, 4096.0 * 16384, 4096.0**2) == 12288
    # A 12K chunk behind 16K costs 30 ms; a 4K first chunk on top would take
    # the step past its target, so it waits.
    assert model.chunk(0, 16384, 12288, 12288.0 * 16384, 12288.0**2) == 0
    assert model.chunk(0, 16384, 16384, 0.0, 0.0) == 0


def test_a_slow_odd_bucket_does_not_hide_a_larger_fitting_one():
    model = _model()
    model.linear_ms[12288] = model.linear_ms[16384] + 10.0
    assert model.chunk(0, 16384, 4096, 4096.0 * 16384, 4096.0**2) == 12288


def test_without_an_attention_term_chunks_fill_the_bucket():
    model = StepCostModel.fit([[(b, 0, _linear(b)) for b in BUCKETS]], 0.0,
                              4096)
    assert model.attn_ms_per_token_pair == 0.0
    assert model.chunk(65536, 16384, 0, 0.0, 0.0) == 16384


def test_pairs_count_the_kernel_schedule_of_a_chunk():
    model = _model()
    assert model.pairs(4096, 0) == 0  # no schedule capacity known
    model.schedule = (10000, 256, 256)
    # 16 query blocks; block i attends to (i + 1) KV blocks.
    assert model.pairs(4096, 0) == sum(range(1, 17))
    # Behind 8K tokens each block also attends to the 32 prefix blocks.
    assert model.pairs(4096, 8192) == sum(range(1, 17)) + 16 * 32
    assert model.pairs(0, 8192) == 0


def test_the_schedule_capacity_bounds_a_chunk_deep_in_a_prompt():
    model = _model(granularity=2048)
    model.schedule = (10000, 256, 256)
    # 16K behind 48K needs 64 * (192 + ~32) pairs: over 10000.
    got = model.chunk(49152, 16384, 0, 0.0, 0.0)
    assert got % 2048 == 0 and 0 < got < 16384
    assert model.pairs(got, 49152) <= 10000
    assert model.pairs(got + 2048, 49152) > 10000
    # Pairs already in the step leave less room.
    less = model.chunk(49152, 16384, 2048, 0.0, 2048.0**2, None, 6000)
    assert less < got and model.pairs(less, 49152) <= 4000
    # A first chunk with a short prefix is not bounded.
    assert model.chunk(0, 16384, 0, 0.0, 0.0) == 16384


def test_fit_keeps_the_schedule_capacity():
    model = StepCostModel.fit([_stage()], 0.1, 4096, (12345, 256, 512))
    assert model.schedule == (12345, 256, 512)
    assert "12345 attention pairs (256x512)" in model.describe()


def test_describe_names_target_buckets_and_granularity():
    text = _model().describe()
    assert "16384:" in text and "target" in text and "granularity 4096" in text
