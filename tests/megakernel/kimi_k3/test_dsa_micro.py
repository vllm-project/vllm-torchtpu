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
"""The Kimi-K3.1 DSA microbenchmark tools (CPU): the decode tool's slope
arithmetic, flags and dispatch, and the prefill stage tool's flags and
``select-topk`` case."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jax.sharding import PartitionSpec as P

from tools.megakernel import __main__ as tool_main
from tools.megakernel import bench, dsa_micro, k3_stage, stage_dsa_micro
from tools.megakernel import dsa_reference as ref
from vllm_torchtpu.megakernel import bridge

pytestmark = pytest.mark.cpu_test

# Fields every record already carries (the tool's context wins in
# `Records.add`): a case's `extra` must not use these names.
RECORD_CONTEXT = {
    "page_tokens",
    "kv_cache_dtype",
    "git",
    "jobid",
    "config",
    "time",
    "command",
    "synthetic_indexer",
}


def test_slowest_rank_median_takes_the_slowest_rank_of_each_repeat():
    n1, n2 = 8, 72

    def run(slope_us, launch_s):
        return (launch_s + n1 * slope_us * 1e-6, launch_s + n2 * slope_us * 1e-6)

    # Rank 1 is slowest in repeats 0 and 2, rank 0 in repeat 1 (a long launch
    # on rank 0 does not move its slope); the median of 4, 3, 5 us is 4.
    samples = [
        [run(2, 1e-3), run(3, 5e-3), run(1, 1e-3)],
        [run(4, 2e-3), run(2, 1e-3), run(5, 1e-3)],
    ]
    assert dsa_micro.slowest_rank_median(samples, n1, n2) == pytest.approx(4.0)
    assert dsa_micro.slopes(samples, n1, n2)[0][1] == pytest.approx(3.0)


def test_dsa_micro_flags_and_dispatch(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro"])
    assert (args.cases, args.rows, args.n1, args.n2, args.repeats) == (
        ["copy", "barrier"],
        [1, 8, 32, 64],
        8,
        0,
        5,
    )
    args = parse(["bench", "dsa-micro", "--cases", "copy", "--rows", "8,64"])
    assert (args.cases, args.rows) == (["copy"], [8, 64])
    with pytest.raises(SystemExit):
        parse(["bench", "dsa-micro", "--cases", "nope"])

    calls = []
    monkeypatch.setattr(dsa_micro, "run", lambda *a: calls.append(a))
    bench.run(args, "records")
    assert calls == [(args, "records")]

    # No weights: the tool gets as far as asking for an allocation.
    with pytest.raises(SystemExit, match="jobid"):
        tool_main.main(["bench", "dsa-micro"])
    with pytest.raises(SystemExit, match="presharded"):
        tool_main.main(["bench", "decode"])


def test_spread_flag_marks_a_slowest_rank_well_above_the_median_rank():
    # Median rank 3.0 us: the slowest rank's median is flagged from 0.05 * us above.
    ranks = [2.9] * 16 + [3.0] * 15
    assert not dsa_micro.Timing(3.2, [*ranks, 3.15]).spread_flag
    assert dsa_micro.Timing(3.2, [*ranks, 3.17]).spread_flag
    assert dsa_micro.Timing(0.8, [0.2] + [0.35] * 30 + [0.55]).spread_flag


def test_a_headline_above_the_slowest_ranks_own_median_is_flagged():
    def old_rule(timing):
        """The flag before the headline-against-slowest-rank clause."""
        gap = timing.rank_us[-1] - timing.rank_median
        return gap > dsa_micro.SPREAD_FRACTION * timing.us

    # Like the real outlier record: most ranks' median slopes lie near the
    # slowest rank's, while the headline (a median over repeats of the maximum
    # over ranks, with different ranks spiking in different repeats) is far above
    # all of them. Only the new clause sees it.
    ranks = [1.63] * 31 + [2.24]
    outlier = dsa_micro.Timing(31.6, ranks)
    assert not old_rule(outlier) and outlier.spread_flag
    # The clause fires from 5% above the slowest rank's own median.
    ranks = [2.2] * 31 + [2.24]
    above = dsa_micro.Timing(2.24 * 1.06, ranks)
    assert not old_rule(above) and above.spread_flag
    assert not dsa_micro.Timing(2.24 * 1.04, ranks).spread_flag


def test_the_long_loop_is_chosen_on_the_slowest_ranks_own_median(monkeypatch):
    calls = []

    def fake(name, body, operands, in_specs, out_specs, *, n1, n2, repeats):
        calls.append((n1, n2, repeats))
        if n2 == 72:
            return dsa_micro.Timing(31.6, [1.63] * 31 + [2.24])
        return dsa_micro.Timing(2.3, [1.63] * 31 + [2.24])

    monkeypatch.setattr(dsa_micro, "time_detail", fake)
    setup = dsa_micro.Setup(None, (), (), ())
    # The headline (a maximum over ranks of each repeat) is far above 20 us, but
    # the slowest rank's own median is not: the call is retimed with the long loop.
    n2, timing = dsa_micro.measure("x", setup, 8, 0, 5)
    assert (n2, timing.us, calls) == (264, 2.3, [(8, 72, 5), (8, 264, 5)])
    # A call that is slow at its slowest rank keeps the short loop.
    calls.clear()
    monkeypatch.setattr(
        dsa_micro,
        "time_detail",
        lambda *a, n1, n2, repeats: calls.append((n1, n2, repeats))
        or dsa_micro.Timing(40.0, [25.0] * 31 + [30.0]),
    )
    assert dsa_micro.measure("x", setup, 8, 0, 5)[0] == 72 and calls == [(8, 72, 5)]
    # An explicit --n2 is never replaced.
    calls.clear()
    monkeypatch.setattr(dsa_micro, "time_detail", fake)
    assert dsa_micro.measure("x", setup, 8, 100, 5)[0] == 100
    assert calls == [(8, 100, 5)]


def test_size_flags_must_be_positive_lists(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    for flag in ("--rows", "--tokens", "--selected", "--inflight", "--wait-group"):
        assert parse(["bench", "dsa-micro", flag, "4,16"])
        for value in ("0", "8,0", "-1", "", "x"):
            with pytest.raises(SystemExit):
                parse(["bench", "dsa-micro", flag, value])
    # --lengths keeps 0: every slot of every rank.
    assert parse(["bench", "dsa-micro", "--lengths", "0,7"]).lengths == [0, 7]


def test_loop_lengths_are_validated():
    for n1, n2, repeats in ((8, 8, 5), (0, 8, 5), (72, 8, 5), (8, 72, 0)):
        with pytest.raises(ValueError, match="n1 < n2"):
            dsa_micro.check_lengths(n1, n2, repeats)
    dsa_micro.check_lengths(8, 264, 1)
    for flag in ("--n1", "--repeats"):
        with pytest.raises(SystemExit):
            tool_main.parser().parse_args(["bench", "dsa-micro", flag, "0"])


def test_op_names_are_sanitised_and_never_reused_for_another_body(monkeypatch):
    registered = []
    monkeypatch.setattr(dsa_micro, "_OPS", {})
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 0))
    monkeypatch.setattr(
        bridge,
        "register_op",
        lambda name, mesh, body, in_specs, out_specs: registered.append(name) or name,
    )

    def body(x, iterations):
        return (x,)

    operands = (torch.zeros(1, 8, 640, dtype=torch.bfloat16),)
    specs = ((P("tp"),), (P("tp"),))
    op = dsa_micro._op("copy r8/a-b", body, operands, *specs, 8)
    assert registered == ["pallas::dsa_micro_copy_r8_a_b_n8"]
    assert dsa_micro._op("copy r8/a-b", body, operands, *specs, 8) == op
    assert dsa_micro._op("copy r8/a-b", body, operands, *specs, 72) != op
    assert len(registered) == 2
    other = (torch.zeros(1, 16, 640, dtype=torch.bfloat16),)
    with pytest.raises(ValueError, match="own name"):
        dsa_micro._op("copy r8/a-b", body, other, *specs, 8)
    with pytest.raises(ValueError, match="own name"):
        dsa_micro._op("copy r8/a-b", lambda x, iterations: (x,), operands, *specs, 8)


def test_run_checks_each_setup_and_records_what_it_measured(monkeypatch):
    logged, added, called = [], [], []

    class Records:
        def add(self, **fields):
            added.append(fields)

    def passes(outputs):
        return True

    def fails(outputs):
        return False

    def timed(*operands):
        return operands

    def checked(*operands):
        return operands

    def setup(variant, **fields):
        return dsa_micro.Setup(timed, ("a",), ("in",), ("out",), variant, **fields)

    setups = [
        setup("plain", check=passes),
        setup("split", check=passes, check_body=checked),
        setup("ablation", invalid=True, extra={"mode": "x"}),
        setup("bare"),
        setup("bad", check=fails, extra={"mode": "y"}),
    ]
    monkeypatch.setitem(
        dsa_micro.CASES, "fake", dsa_micro.CaseDef(lambda *a: setups, per_tokens=True)
    )
    monkeypatch.setitem(
        dsa_micro.CASES,
        "flat",
        dsa_micro.CaseDef(lambda *a: [setup("")], per_tokens=False),
    )
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 0))
    monkeypatch.setattr(dsa_micro.harness, "sync", lambda: None)
    monkeypatch.setattr(dsa_micro.harness, "log", logged.append)
    monkeypatch.setattr(
        dsa_micro,
        "measure",
        lambda name, setup, n1, n2, repeats: (
            72,
            dsa_micro.Timing(10.0, [9.9] * 31 + [10.0]),
        ),
    )
    monkeypatch.setattr(
        dsa_micro, "call_body", lambda *call: called.append(call) or ("out",)
    )
    monkeypatch.setattr(
        dsa_micro,
        "reduce_check",
        lambda check, outputs: {} if check is passes else {3: "check returned False"},
    )

    def args(**fields):
        base = dict(cases=["fake", "flat"], tokens=[256], rows=[8], n1=8, n2=0)
        return SimpleNamespace(**{**base, "repeats": 5, **fields})

    # Loop lengths are checked before anything runs.
    with pytest.raises(SystemExit):
        dsa_micro.run(args(n1=8, n2=4), Records())
    assert not added and not called

    # A failing check is recorded with the others, then fails the run.
    with pytest.raises(SystemExit) as stop:
        dsa_micro.run(args(), Records())
    assert stop.value.code == 1
    assert [(r["case"], r["variant"], r["tokens"]) for r in added] == [
        ("fake", "plain", 256),
        ("fake", "split", 256),
        ("fake", "ablation", 256),
        ("fake", "bare", 256),
        ("fake", "bad", 256),
        ("flat", "", 0),
    ]
    # `invalid` ablations are timing only; a case without a check records None.
    assert [r["check"] for r in added] == [True, True, "invalid", None, False, None]
    assert [r["failed_ranks"] for r in added] == [[], [], [], [], [3], []]
    # `extra` joins the record, with the harness's own fields.
    assert (added[2]["mode"], added[4]["mode"]) == ("x", "y")
    assert (added[0]["us"], added[0]["n1"], added[0]["n2"], added[0]["repeats"]) == (
        10.0,
        8,
        72,
        5,
    )
    assert (added[0]["rank_us_min"], added[0]["rank_us_median"]) == (9.9, 9.9)
    assert added[0]["rank_us_max"] == 10.0 and not added[0]["spread_flag"]
    # The check reads `check_body`'s outputs when set, after `n1` iterations.
    assert called == [
        ("fake_r8_t256_plain", timed, ("a",), ("in",), ("out",), 8),
        ("fake_r8_t256_split_check", checked, ("a",), ("in",), ("out",), 8),
        ("fake_r8_t256_bad", timed, ("a",), ("in",), ("out",), 8),
    ]
    assert logged[-1] == (
        "FAIL: check failed: fake R=8 T=256 on ranks {3: 'check returned False'}"
    )
    assert any("check invalid" in line for line in logged)


def test_a_check_is_reduced_over_every_rank(monkeypatch):
    sent = []

    def gather(out, mine):
        sent.append(mine)
        out[:] = [(0, None), (7, None), (2, "check returned False"), (3, "boom")]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 5))
    # Whatever this rank sees, the result names the failing ranks of all of them.
    assert dsa_micro.reduce_check(lambda outputs: True, "out") == {
        2: "check returned False",
        3: "boom",
    }
    dsa_micro.reduce_check(lambda outputs: False, "out")

    def raises(outputs):
        raise RuntimeError("no data")

    dsa_micro.reduce_check(raises, "out")
    assert sent == [
        (5, None),
        (5, "check returned False"),
        (5, "RuntimeError: no data"),
    ]


def test_select_keys_order_like_the_scores():
    import numpy as np

    scores = np.array(
        [-np.inf, -3e38, -2.5, -1.0, -1e-30, 1e-30, 0.5, 1.0, 7.0, 3e38, np.inf],
        np.float32,
    )
    keys = dsa_micro.order_keys(scores)
    assert keys.dtype == np.uint32 and (np.diff(keys.astype(np.int64)) > 0).all()
    assert dsa_micro.order_keys(np.float32([1.0, -1.0])).tolist() == [
        0xBF800000,
        0x407FFFFF,
    ]


def test_select_global_token_index_interleaves_pages():
    import numpy as np

    # Rank 3's local page 1 is global page 35.
    assert dsa_micro.global_token_index(256 + 7, 3, 256) == 35 * 256 + 7
    index = dsa_micro.global_token_index(
        np.arange(1024)[None], np.arange(32)[:, None], 256
    )
    assert sorted(index.ravel()) == list(range(32 * 1024))
    counts = dsa_micro.local_counts(3 * 32 * 256 + 300, 1024, 256)
    assert counts.sum() == 3 * 32 * 256 + 300
    assert counts[0] == 3 * 256 + 256 and counts[1] == 3 * 256 + 44
    assert counts[2] == 3 * 256


def test_select_round_digit_and_upper_count():
    # Counts at or above thresholds d = 0..3; k = 5: the digit is 2, with two
    # keys at or above threshold 3.
    assert dsa_micro.radix_digit([10, 7, 5, 2], 5, 0) == (2, 2)
    # The top digit keeps the previous round's count above.
    assert dsa_micro.radix_digit([10, 9, 8, 6], 5, 4) == (3, 4)
    assert dsa_micro.radix_digit([5, 0, 0, 0], 5, 0) == (0, 0)


def test_select_cumulative_counts_add_the_keys_above_the_prefix():
    # Digit counts 1, 0, 3, 2 under a prefix with 4 keys above it.
    assert dsa_micro.cumulative_counts([1, 0, 3, 2], 4).tolist() == [10, 9, 9, 6]


def test_select_tie_offsets_follow_global_token_order():
    import numpy as np

    rng = np.random.default_rng(0)
    ties = rng.integers(0, 3, (32, 4))
    # Global order is (page, rank): rank r's page j follows every rank's
    # earlier pages and the lower ranks' page j.
    order = [(j, r) for j in range(4) for r in range(32)]
    ahead = {
        key: sum(ties[r, j] for j, r in order[: order.index(key)]) for key in order
    }
    for rank in (0, 5, 31):
        own_before = np.cumsum(ties[rank]) - ties[rank]
        np.testing.assert_array_equal(
            dsa_micro.tie_offsets(ties, rank) + own_before,
            [ahead[(j, rank)] for j in range(4)],
        )


@pytest.mark.parametrize("histogram", ["cumulative", "packed"])
@pytest.mark.parametrize("bits", [4, 8])
def test_select_model_equals_the_reference_with_ties(bits, histogram):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tools.megakernel import dsa_reference as ref

    tokens, page, k = 64, 16, 300
    rng = np.random.default_rng(bits)
    cases = [
        (np.round(4 * rng.standard_normal((32, tokens))) / 4 + 0.125, 32 * tokens),
        (np.full((32, tokens), 0.75), 32 * tokens),
        (np.round(rng.standard_normal((32, tokens))) + 0.5, 1000),
        (rng.standard_normal((32, tokens)), 32 * tokens),
        (rng.standard_normal((32, tokens)), k),
    ]
    local = np.arange(tokens)
    for scores, length in cases:
        scores = scores.astype(np.float32)
        index = dsa_micro.global_token_index(local[None], np.arange(32)[:, None], page)
        flat = np.zeros(32 * tokens, np.float32)
        flat[index] = scores
        with jax.default_device(jax.devices("cpu")[0]):
            chosen = np.asarray(
                ref.select_topk(jnp.asarray(flat[None]), jnp.asarray([length]), k)
            )[0]
        wanted = np.zeros(32 * tokens, bool)
        wanted[chosen[chosen >= 0]] = True
        counts = dsa_micro.local_counts(length, tokens, page)
        got = dsa_micro.radix_select_model(
            dsa_micro.order_keys(scores), counts, k, bits, page, histogram
        )
        np.testing.assert_array_equal(got, wanted[index])


def test_select_layout_round_trips():
    import numpy as np

    for rows, tokens in ((1, 1024), (4, 256), (8, 512), (64, 1024)):
        groups = dsa_micro.virtual_groups(rows, tokens)
        values = np.arange(rows * tokens).reshape(rows, tokens)
        virtual = dsa_micro.to_virtual(values, groups)
        assert virtual.shape[1:] == (max(rows, 8), 128)
        np.testing.assert_array_equal(dsa_micro.from_virtual(virtual, rows), values)
        mask = values % 3 == 0
        np.testing.assert_array_equal(
            dsa_micro.unpack_mask(
                dsa_micro.pack_mask(mask, rows, tokens), rows, tokens
            ),
            mask,
        )
    with pytest.raises(ValueError, match="divide 8"):
        dsa_micro.virtual_groups(3, 1024)
    with pytest.raises(ValueError, match="lane rows"):
        dsa_micro.virtual_groups(1, 512)


def test_pack_mask_inverts_unpack_mask():
    import numpy as np

    rng = np.random.default_rng(0)
    for rows, tokens in ((1, 1024), (2, 512), (4, 256), (8, 512), (64, 1024)):
        mask = rng.random((rows, tokens)) < 0.3
        words = dsa_micro.pack_mask(mask, rows, tokens)
        assert words.dtype == np.int32
        np.testing.assert_array_equal(dsa_micro.unpack_mask(words, rows, tokens), mask)


def test_rows_split_into_tiles_and_pad_to_eight():
    # R: (rows sharing a score tile, select parts per row, padded rows). Odd
    # R (3, 5, 12) fit neither layout; gather pads any R.
    layouts = {
        1: (1, 8, 8),
        2: (2, 4, 8),
        3: (None, None, 8),
        4: (4, 2, 8),
        5: (None, None, 8),
        8: (8, 1, 8),
        12: (None, None, 16),
        16: (8, 1, 16),
        64: (8, 1, 64),
    }
    for rows, (group, parts, padded) in layouts.items():
        assert dsa_micro.padded_rows(rows) == padded
        if group is None:
            for layout in (
                lambda r: dsa_micro.tile_group_rows(r, "score"),
                lambda r: dsa_micro.virtual_groups(r, 1024),
            ):
                with pytest.raises(ValueError, match="divide 8"):
                    layout(rows)
        else:
            assert dsa_micro.tile_group_rows(rows, "score") == group
            assert dsa_micro.virtual_groups(rows, 1024) == parts


def test_select_pages_must_be_powers_of_two():
    # Whole pages, a power-of-two count of them, each a power-of-two number of
    # 128-token lane rows (the page shift and the packets' dense layout).
    assert dsa_micro.select_pages(1024, 256) == 4
    assert dsa_micro.select_pages(4096, 512) == 8
    assert dsa_micro.select_pages(32768, 256) == 128
    for tokens, page_tokens in ((1536, 384), (768, 256), (1024, 192), (512, 1024)):
        with pytest.raises(ValueError, match="power of two"):
            dsa_micro.select_pages(tokens, page_tokens)
    with pytest.raises(ValueError, match="at most 128"):
        dsa_micro.select_pages(256 * 256, 256)
    # `_select_setup` refuses before it builds anything.
    args = SimpleNamespace(page_tokens=384)
    with pytest.raises(ValueError, match="power of two"):
        dsa_micro._select_setup(8, 1536, "radix4", "flat", True, "cumulative", args)


def test_select_setups_record_their_settings(monkeypatch):
    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    args = SimpleNamespace(page_tokens=512, scores="normal", lengths=[])

    def setup(variant, exchange, ties, histogram, page_tokens=512):
        args.page_tokens = page_tokens
        return dsa_micro._select_setup(
            8, 2048, variant, exchange, ties, histogram, args
        )

    got = setup("radix4", "flat", True, "packed")
    assert got.variant == "radix4_flat_packed_p512" and got.check is not None
    assert got.extra == {
        "exchange": "flat",
        "histogram": "packed",
        "ties": "on",
        "select_page_tokens": 512,
        "tie_skip": "off",
        "row_sum": "after",
    }
    # The tool's own context (`Records.add` lets it win) names the serving page
    # size and cache dtype: an extra with those names would be overwritten.
    assert not set(got.extra) & RECORD_CONTEXT
    got = setup("radix8", "none", False, "cumulative", 256)
    assert got.variant == "radix8_none_noties" and got.check is None and got.invalid
    assert got.extra["ties"] == "off" and got.extra["select_page_tokens"] == 256


def test_select_packets_stay_dense_below_a_lane_row():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    shape = dsa_select._packet_shape
    # Counts as wide as a lane row or wider go as they are.
    assert shape(8, 128) == (8, 128) and shape(64, 256) == (64, 256)
    # Narrower ones share lane rows: row block f of the packet takes lanes
    # [f width, (f + 1) width).
    assert shape(64, 16) == (8, 128) and shape(64, 64) == (32, 128)
    assert shape(32, 4) == (8, 128) and shape(8, 16) == (8, 128)
    for vrows in (8, 16, 32, 64):
        for width in (1, 4, 16, 64):
            rows, lanes = shape(vrows, width)
            assert lanes == 128 and rows >= 8
            assert rows * lanes >= vrows * width
            # Dense unless the 8-row minimum pads it.
            assert rows == 8 or rows * lanes == vrows * width


def test_select_verdicts_need_the_lowest_index_ties():
    import numpy as np

    # One row of 32 x 4 tokens, k = 3: 9 is the top score, the 5s tie.
    values = np.zeros((1, 128), np.float32)
    values[0, [10, 20, 30, 40]] = [9, 5, 5, 5]
    wanted = np.zeros((1, 128), bool)
    wanted[0, [10, 20, 30]] = True
    selected = np.zeros(128, bool)

    def verdict(picked):
        selected[:] = False
        selected[picked] = True
        # As if one rank held every token (the other ranks select nothing).
        tie = values[0] == 5
        stats = [
            [
                (
                    bool((selected == wanted[0]).all()),
                    int(selected.sum()),
                    float(values[0][selected].min()),
                    float(values[0][~selected].max()),
                    int((selected & tie).sum()),
                    int(np.flatnonzero(selected & tie).max(initial=-1)),
                    int(np.flatnonzero(~selected & tie).min(initial=128)),
                )
            ]
        ]
        return dsa_micro.select_verdicts(stats, values, wanted, [128], k=3)[0]

    assert verdict([10, 20, 30])[0]
    passed, line = verdict([10, 20, 40])
    assert not passed and "lowest indices False" in line
    assert not verdict([10, 20])[0]


def test_select_check_gives_each_rank_its_own_verdict(monkeypatch):
    import numpy as np

    rows, tokens, page, name = 4, 256, 256, "normal"
    lengths = dsa_micro._select_lengths(rows, tokens, [])
    _, wanted, _ = dsa_micro._select_reference(rows, tokens, name, lengths, page)
    index = [
        dsa_micro.global_token_index(np.arange(tokens), r, page) for r in range(32)
    ]
    masks = [wanted[:, i].copy() for i in index]
    logged = []
    monkeypatch.setattr(dsa_micro.harness, "log", logged.append)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 32)

    def verdicts(masks, edit=lambda stats: None, broken=None):
        """Every rank's return value (an exception's text for a rank that
        raises), the ranks' statistics gathered as the group would. ``broken``:
        a rank whose outputs cannot be read."""
        stats = [
            dsa_micro._select_row_stats(m, r, rows, tokens, name, lengths, page)
            for r, m in enumerate(masks)
        ]
        edit(stats)
        if broken is not None:
            stats[broken] = "ValueError: no data"
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda out, mine: out.__setitem__(slice(None), enumerate(stats)),
        )
        check = dsa_micro._select_check(rows, tokens, name, lengths, page, "select")
        out = []
        for rank, mask in enumerate(masks):
            monkeypatch.setattr(
                bridge, "kernel_mesh", lambda axis, size, mesh=(None, rank): mesh
            )
            words = torch.from_numpy(dsa_micro.pack_mask(mask, rows, tokens))[None]
            try:
                out.append(check((words if rank != broken else None,)))
            except RuntimeError as exc:
                out.append(str(exc))
        return out

    assert all(verdicts(masks)) and not any("FAIL" in line for line in logged)
    # One rank misses a selected token: it fails, the others still hold.
    masks[5][0, np.flatnonzero(masks[5][0])[0]] = False
    got = verdicts(masks)
    assert [r for r, ok in enumerate(got) if not ok] == [5]
    assert any("row 0: FAIL" in line for line in logged)
    masks[5] = wanted[:, index[5]]

    # Every mask equals the reference but the group selected one token too many:
    # no rank is to blame, all fail.
    def one_more(stats):
        stats[3] = [(s[0], s[1] + 1, *s[2:]) for s in stats[3]]

    assert not any(verdicts(masks, one_more))
    # A rank that cannot read its outputs fails with its own error; the others
    # keep the verdicts of their own masks.
    got = verdicts(masks, broken=7)
    assert "TypeError" in got[7] and got[:7] + got[8:] == [True] * 31


def test_seeded_inputs_keep_their_streams(monkeypatch):
    import numpy as np

    # Records stay reproducible only while the seeds do not change.
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    np.testing.assert_array_equal(
        dsa_micro._select_scores(2, 256, 3, "normal"),
        np.random.default_rng([dsa_micro.SELECT_SEED, 2, 256, 3])
        .standard_normal((2, 256))
        .astype(np.float32),
    )
    np.testing.assert_array_equal(
        dsa_micro._rank_rng(7).integers(1 << 20, size=4),
        np.random.default_rng([7, 3]).integers(1 << 20, size=4),
    )
    chosen = dsa_micro._gather_selection(2, 1024, 16, "recent")
    rng = np.random.default_rng([dsa_micro.GATHER_SEED, 1024, 16, 1, 1])
    np.testing.assert_array_equal(
        chosen[1], dsa_micro.select_tokens(rng, 1024, 16, "recent")
    )
    # Compact selections differ per rank; mixed rows are S, none, one, all.
    assert dsa_micro.row_mix_counts(64, 1024, 5, False) == [64] * 5
    assert dsa_micro.row_mix_counts(64, 1024, 5, True) == [64, 0, 1, 1024, 64]
    chosen = dsa_micro._compact_selection(4, 1024, 64, "recent", True)
    rng = np.random.default_rng([dsa_micro.COMPACT_SEED, 1024, 64, 1, 0, 3])
    np.testing.assert_array_equal(
        chosen[0], dsa_micro.select_tokens(rng, 1024, 64, "recent")
    )
    assert [len(c) for c in chosen] == [64, 0, 1, 1024]
    np.testing.assert_array_equal(chosen[3], np.arange(1024))


def test_fp8_and_bf16_bit_patterns_upload_as_their_torch_dtype(monkeypatch):
    import ml_dtypes
    import numpy as np

    sent = []
    monkeypatch.setattr(
        torch.Tensor, "to", lambda self, device: sent.append(device) or self
    )
    values = np.array([[0.5, -1.0, 448.0, 0.0, 3.0]], np.float32)
    for dtype, torch_dtype, word in (
        (ml_dtypes.float8_e4m3fn, torch.float8_e4m3fn, np.uint8),
        (ml_dtypes.bfloat16, torch.bfloat16, np.uint16),
    ):
        tensor = dsa_micro._upload_bits(values.astype(dtype).view(word), torch_dtype)
        assert tensor.dtype == torch_dtype and tensor.shape == (1, 1, 5)
        assert tensor[0].float().tolist() == values.tolist()
    assert sent == ["tpu", "tpu"]


def test_gather_token_location_follows_the_pool_layout():
    import numpy as np

    bf16, fp8 = dsa_micro.GEOMETRY["bf16"], dsa_micro.GEOMETRY["fp8"]
    assert (bf16.block_tokens, fp8.block_tokens) == (256, 512)
    assert (bf16.token_bytes, fp8.token_bytes) == (1280, 640)
    assert bf16.row_bytes == fp8.row_bytes == 2560
    block, sublane, slot = dsa_micro.locate(np.array([0, 1, 2, 255, 256, 700]), bf16)
    assert block.tolist() == [0, 0, 0, 0, 1, 2]
    assert sublane.tolist() == [0, 0, 1, 127, 0, 94]
    assert slot.tolist() == [0, 1, 0, 1, 0, 0]
    block, sublane, slot = dsa_micro.locate(np.array([3, 4, 511, 512, 1000]), fp8)
    assert block.tolist() == [0, 0, 0, 1, 1]
    assert sublane.tolist() == [0, 1, 127, 0, 122]
    assert slot.tolist() == [3, 0, 3, 0, 0]
    # Every token of a block has its own (sublane, slot).
    every = dsa_micro.locate(np.arange(512), fp8)
    assert len(set(zip(every[1].tolist(), every[2].tolist()))) == 512


@pytest.mark.parametrize("pattern", ["uniform", "recent"])
@pytest.mark.parametrize("count", [16, 64, 256, 2048])
def test_gather_selection_is_distinct_sorted_and_seeded(pattern, count):
    import numpy as np

    tokens = 8192
    a = dsa_micro.select_tokens(np.random.default_rng(1), tokens, count, pattern)
    b = dsa_micro.select_tokens(np.random.default_rng(1), tokens, count, pattern)
    assert a.tolist() == b.tolist()
    assert len(a) == count and (np.diff(a) > 0).all()
    assert a.min() >= 0 and a.max() < tokens
    other = dsa_micro.select_tokens(np.random.default_rng(2), tokens, count, pattern)
    assert a.tolist() != other.tolist()


def test_gather_recent_pattern_puts_most_tokens_in_the_window():
    import numpy as np

    tokens, rng = 8192, np.random.default_rng(0)
    # The balanced share (64) is 48 recent, 3 sinks and 13 uniform.
    chosen = dsa_micro.select_tokens(rng, tokens, 64, "recent")
    assert (chosen >= tokens - 128).sum() >= 48
    assert (chosen < 128).sum() >= 3
    # A rank holding the whole selection owns every page: the window grows with S.
    assert dsa_micro.recent_window(16) == dsa_micro.recent_window(64) == 128
    assert dsa_micro.recent_window(256) == 512
    assert dsa_micro.recent_window(2048) == 4096
    chosen = dsa_micro.select_tokens(rng, tokens, 2048, "recent")
    assert (chosen >= tokens - 4096).sum() >= 1536
    assert (chosen < 128).sum() >= 102
    # Uniform spreads: far fewer than 3/4 land in the last 128 tokens.
    uniform = dsa_micro.select_tokens(rng, tokens, 64, "uniform")
    assert (uniform >= tokens - 128).sum() < 20
    with pytest.raises(ValueError, match="cannot select"):
        dsa_micro.select_tokens(rng, 8, 9, "uniform")
    with pytest.raises(ValueError, match="pattern"):
        dsa_micro.select_tokens(rng, 64, 8, "nope")
    # Tiny pools still give distinct tokens (the windows overlap).
    small = dsa_micro.select_tokens(rng, 300, 200, "recent")
    assert len(set(small.tolist())) == 200


def test_gather_units_deduplicate_tiles_and_pages_per_row():
    import numpy as np

    bf16 = dsa_micro.GEOMETRY["bf16"]
    # Tokens 0, 1 share sublane row 0; 2, 3 row 1; 15 is row 7 (tile 0); 16 is
    # tile 1; 256 is the second block.
    selected = np.array([0, 1, 2, 15, 16, 256, 300, 511])
    block, first = dsa_micro.gather_units(selected, "row", bf16)
    assert block.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert first.tolist() == [0, 0, 1, 7, 8, 0, 22, 127]
    block, first = dsa_micro.gather_units(selected, "tile", bf16)
    assert block.tolist() == [0, 0, 1, 1, 1]
    assert first.tolist() == [0, 8, 0, 16, 120]
    block, first = dsa_micro.gather_units(selected, "page", bf16)
    assert block.tolist() == [0, 1] and first.tolist() == [0, 0]
    # FP8 tiles hold 32 tokens: tokens 0..31 are one tile.
    fp8 = dsa_micro.GEOMETRY["fp8"]
    block, first = dsa_micro.gather_units(np.array([0, 5, 31, 32]), "tile", fp8)
    assert block.tolist() == [0, 0] and first.tolist() == [0, 8]
    with pytest.raises(ValueError, match="variant"):
        dsa_micro.gather_units(selected, "nope", bf16)


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_gather_units_cover_every_selected_token(kv_dtype):
    import numpy as np

    geometry = dsa_micro.GEOMETRY[kv_dtype]
    selected = dsa_micro.select_tokens(np.random.default_rng(3), 8192, 256, "recent")
    block, sublane, _ = dsa_micro.locate(selected, geometry)
    for variant in dsa_micro.GATHER_VARIANTS:
        blocks, first = dsa_micro.gather_units(selected, variant, geometry)
        rows = dsa_micro.GATHER_SLOT_ROWS[variant]
        covered = {
            (b, f + r)
            for b, f in zip(blocks.tolist(), first.tolist())
            for r in range(rows)
        }
        wanted = zip(block.tolist(), sublane.tolist())
        assert all(unit in covered for unit in wanted)
        # Only tiles and pages deduplicate: `row` has a DMA per selected token,
        # even for the two tokens of one sublane row.
        distinct = len(set(zip(blocks.tolist(), first.tolist())))
        assert (len(blocks) == distinct) == (variant != "row")
        assert first.max() + rows <= dsa_micro.POOL_ROWS
    assert len(dsa_micro.gather_units(selected, "row", geometry)[0]) == 256
    tiles = len(dsa_micro.gather_units(selected, "tile", geometry)[0])
    pages = len(dsa_micro.gather_units(selected, "page", geometry)[0])
    assert pages <= tiles < 256


def test_gather_dma_codes_pack_row_block_and_first_row():
    import numpy as np

    codes = dsa_micro.dma_codes(
        [(np.array([0, 3]), np.array([5, 0])), (np.array([31]), np.array([127]))]
    )
    assert codes.dtype == np.int32
    assert codes.tolist() == [5, 3 << 8, (1 << 16) | (31 << 8) | 127]
    with pytest.raises(ValueError, match="out of range"):
        dsa_micro.dma_codes([(np.array([256]), np.array([0]))])


def test_gather_ring_slots_are_powers_of_two_and_fit_the_list():
    assert dsa_micro.ring_slots(16, 64) == 16
    assert dsa_micro.ring_slots(17, 64) == 32
    assert dsa_micro.ring_slots(1, 64) == 1
    assert dsa_micro.ring_slots(4096, 64) == 64
    assert dsa_micro.ring_slots(4096, 8) == 8
    for bad in (0, 3, 96):
        with pytest.raises(ValueError, match="power of two"):
            dsa_micro.ring_slots(100, bad)


def test_gather_block_table_scatters_every_block_once():
    import numpy as np

    table = dsa_micro.block_table(8, 32, 5)
    assert table.shape == (8, 32) and table.dtype == np.int32
    assert sorted(table.ravel().tolist()) == list(range(256))
    assert table.tolist() == dsa_micro.block_table(8, 32, 5).tolist()
    assert table[:, 0].tolist() != sorted(table[:, 0].tolist())


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_gather_accumulators_sum_the_weighted_words_of_each_unit(kv_dtype):
    import numpy as np

    geometry = dsa_micro.GEOMETRY[kv_dtype]
    rng = np.random.default_rng(0)
    dtype = np.uint16 if kv_dtype == "bf16" else np.uint8
    bits = rng.integers(
        0, np.iinfo(dtype).max, (10, 128, geometry.packing, 640), dtype=dtype
    )
    table = np.array([[3, 0, 5, 7, 9], [1, 4, 2, 6, 8]], np.int32)
    selected = [np.array([0, 3, 600]), np.array([7, 1100])]
    for variant in dsa_micro.GATHER_VARIANTS:
        units = [dsa_micro.gather_units(s, variant, geometry) for s in selected]
        slot_rows = dsa_micro.GATHER_SLOT_ROWS[variant]
        got = dsa_micro.gather_accumulators(bits, table, units, slot_rows, 8, geometry)
        assert got.shape == (8, 640) and got.dtype == np.int32
        assert not got[2:].any()
        # Brute force: word = tokens of a sublane row, token 0 in the low bits;
        # weights 2 n + 1 for DMA n (from 0) and 2 j + 1 for sublane row j.
        wanted = np.zeros((8, 640), np.uint64)
        ordinal = 0
        for row, (blocks, first) in enumerate(units):
            for block, start in zip(blocks.tolist(), first.tolist()):
                ordinal += 1
                for sub in range(start, start + slot_rows):
                    word = np.zeros(640, np.uint64)
                    for p in range(geometry.packing):
                        token = bits[table[row][block], sub, p].astype(np.uint64)
                        word |= token << (8 * geometry.itemsize * p)
                    wanted[row] += (2 * ordinal - 1) * (2 * (sub - start) + 1) * word
        np.testing.assert_array_equal(
            got.view(np.uint32), (wanted % 2**32).astype(np.uint32)
        )


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_gather_accumulators_see_the_top_bit_of_every_dma_and_row_order(kv_dtype):
    import numpy as np

    geometry = dsa_micro.GEOMETRY[kv_dtype]
    dtype = np.uint16 if kv_dtype == "bf16" else np.uint8
    top = dtype(1 << (8 * geometry.itemsize - 1))
    bits = np.random.default_rng(1).integers(
        0, np.iinfo(dtype).max, (10, 128, geometry.packing, 640), dtype=dtype
    )
    table = np.array([[3, 0, 5, 7, 9], [1, 4, 2, 6, 8]], np.int32)
    selected = [np.array([0, 5, 600]), np.array([9, 1100])]
    for variant in dsa_micro.GATHER_VARIANTS:
        units = [dsa_micro.gather_units(s, variant, geometry) for s in selected]
        slot_rows = dsa_micro.GATHER_SLOT_ROWS[variant]

        def accumulators(units=units, slot_rows=slot_rows):
            return dsa_micro.gather_accumulators(
                bits, table, units, slot_rows, 8, geometry
            )

        base = accumulators()
        # The top bit of the last token of a word, in the first and the last
        # sublane row of a unit, of every DMA (even and odd numbers alike): a
        # weight with a factor of two would drop it.
        for row, (blocks, first) in enumerate(units):
            for block, start in zip(blocks.tolist(), first.tolist()):
                for sub in (start, start + slot_rows - 1):
                    at = (table[row][block], sub, geometry.packing - 1, 5)
                    bits[at] ^= top
                    assert (accumulators() != base).any(), (variant, row, block, sub)
                    bits[at] ^= top
        np.testing.assert_array_equal(accumulators(), base)
        if slot_rows > 1:
            # Two sublane rows of a unit swapped.
            row, block, start = 0, units[0][0][0], units[0][1][0]
            swapped = (table[row][block], slice(start, start + 2))
            saved = bits[swapped].copy()
            bits[swapped] = saved[::-1]
            assert (accumulators() != base).any()
            bits[swapped] = saved


def test_gather_flags_and_variant_split(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "gather"])
    assert args.variant == [] and args.inflight == [64] and args.wait_group == [1]
    assert args.pattern == ["uniform", "recent"] and args.kv_dtype == ["bf16", "fp8"]
    assert args.selected == [16, 64, 256, 2048] and args.verify == "on"
    args = parse(
        ["bench", "dsa-micro", "--variant", "tile,radix4", "--kv-dtype", "fp8",
         "--inflight", "8,64", "--wait-group", "1,4", "--verify", "off"]
    )  # fmt: skip
    assert args.variant == ["tile", "radix4"] and args.kv_dtype == ["fp8"]
    assert (args.inflight, args.wait_group, args.verify) == ([8, 64], [1, 4], "off")
    for flag, value in (
        ("--variant", "radix5"),
        ("--pattern", "random"),
        ("--kv-dtype", "int8"),
        ("--verify", "maybe"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])
    # Each case keeps its own variant names (and all of them if given none).
    assert [v for v in args.variant if v in dsa_micro.SELECT_BITS] == ["radix4"]
    assert [v for v in args.variant if v in dsa_micro.GATHER_VARIANTS] == ["tile"]


def test_distinct_units_deduplicate_rows_and_mark_tokens():
    import numpy as np

    bf16, fp8 = dsa_micro.GEOMETRY["bf16"], dsa_micro.GEOMETRY["fp8"]
    selected = np.array([0, 1, 2, 15, 16, 256, 300, 511])
    # Tokens 0 and 1 share sublane row 0 (bits 0 and 1); 2 is row 1's first
    # token, 15 row 7's second; 256..511 are the second block.
    wanted = {
        "row": ([0, 0, 0, 0, 1, 1, 1], [0, 1, 7, 8, 0, 22, 127], [3, 1, 2, 1, 1, 1, 2]),
        "tile": ([0, 0, 1, 1, 1], [0, 8, 0, 16, 120], [32775, 1, 1, 4096, 32768]),
        "page": ([0, 1], [0, 0], [0, 0]),
    }
    for variant, lists in wanted.items():
        got = dsa_micro.distinct_units(selected, variant, bf16)
        assert [x.tolist() for x in got] == list(lists), variant
    # Tiles and pages are the gather's units; rows deduplicate.
    for variant in ("tile", "page"):
        units = dsa_micro.gather_units(selected, variant, bf16)
        got = dsa_micro.distinct_units(selected, variant, bf16)
        assert [x.tolist() for x in got[:2]] == [x.tolist() for x in units]
    assert dsa_micro.pair_counts(selected, bf16) == (8, 7, 5, 2)
    # An FP8 tile holds 32 tokens: its bitmask fills the 32-bit word.
    _, _, bits = dsa_micro.distinct_units(np.array([0, 5, 31, 32]), "tile", fp8)
    assert bits.tolist() == [2**0 | 2**5 | 2**31, 1]
    assert dsa_micro._int32(bits)[0] < 0


def test_compact_expected_splits_parts_and_caps_lists():
    import numpy as np

    bf16 = dsa_micro.GEOMETRY["bf16"]
    # R = 1, T = 2048: eight parts of 256 tokens; the tokens lie in parts 0, 1,
    # 1, 5 and 7.
    selected = [np.array([3, 300, 301, 1500, 2047])]
    tiles = dsa_micro.compact_expected(selected, 1, 2048, "tile", bf16, 128)
    assert tiles.codes.shape == tiles.bitmasks.shape == (8, 128)
    assert tiles.counts.shape == (8, 4) and tiles.counts.dtype == np.int32
    assert tiles.counts[:, 0].tolist() == [1, 2, 0, 0, 0, 1, 0, 1]
    assert tiles.counts[:, 2].tolist() == [1, 1, 0, 0, 0, 1, 0, 1]
    # Tokens 300 and 301 share one tile: block 1, sublane rows 16..23.
    assert tiles.codes[1, 0] == (1 << 8) | 16
    assert tiles.bitmasks[1, 0] == (1 << 12) | (1 << 13)
    # Token 3 is block 0, sublane row 1: its tile's code is 0, a real unit, so a
    # list's length is its count, not its nonzero codes.
    assert tiles.counts[0, 2] == 1 and not tiles.codes[0].any()
    assert tiles.bitmasks[0, 0] == 1 << 3
    # Pages list per row, on its first part: the five tokens' pages 0, 1, 5, 7.
    pages = dsa_micro.compact_expected(selected, 1, 2048, "page", bf16, 128)
    assert pages.codes[0, :5].tolist() == [0, 1 << 8, 5 << 8, 7 << 8, 0]
    assert not pages.codes[1:].any() and not pages.bitmasks.any()
    assert pages.counts[:, 3].tolist() == [4, 0, 0, 0, 0, 0, 0, 0]
    # Every variant records the same counts.
    np.testing.assert_array_equal(pages.counts, tiles.counts)
    # R = 8, T = 8192: row 2's 200 distinct sublane rows overflow 128 codes.
    selections = [np.array([5 * r]) for r in range(8)]
    selections[2] = np.arange(0, 400, 2)
    rows = dsa_micro.compact_expected(selections, 8, 8192, "row", bf16, 128)
    assert rows.counts[2, :2].tolist() == [200, 200]
    block, first, _ = dsa_micro.distinct_units(selections[2], "row", bf16)
    np.testing.assert_array_equal(rows.codes[2], ((block << 8) | first)[:128])
    assert (rows.bitmasks[2] == 1).all()


def _selection_mask(selections, tokens):
    import numpy as np

    mask = np.zeros((len(selections), tokens), bool)
    for row, selected in enumerate(selections):
        mask[row, selected] = True
    return mask


@pytest.mark.parametrize("count, capacity", [(64, 128), (300, 256)])
@pytest.mark.parametrize("pattern", ["uniform", "recent"])
@pytest.mark.parametrize(
    "rows, tokens", [(1, 1024), (2, 1024), (4, 2048), (8, 2048), (16, 4096)]
)
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("variant", ["row", "tile", "page"])
def test_compact_model_equals_expected(
    monkeypatch, variant, kv_dtype, rows, tokens, pattern, count, capacity
):
    import numpy as np

    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    geometry = dsa_micro.GEOMETRY[kv_dtype]
    # Rows of S, no, one and every token: empty planes, empty parts and lists
    # longer than the capacity.
    selections = dsa_micro._compact_selection(rows, tokens, count, pattern, True)
    words = dsa_micro.pack_mask(_selection_mask(selections, tokens), rows, tokens)
    got = dsa_micro.compact_model(words, rows, tokens, variant, geometry, capacity)
    wanted = dsa_micro.compact_expected(
        selections, rows, tokens, variant, geometry, capacity
    )
    for name in ("codes", "bitmasks", "counts"):
        np.testing.assert_array_equal(
            getattr(got, name), getattr(wanted, name), err_msg=name
        )


def test_compact_flags_setups_and_records(monkeypatch):
    import numpy as np

    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "compact"])
    assert (args.capacity, args.smem_path, args.row_mix) == ([128], ["dma"], "off")
    args = parse(
        ["bench", "dsa-micro", "--cases", "compact", "--capacity", "128,256",
         "--smem-path", "dma,hbm", "--row-mix", "on"]
    )  # fmt: skip
    assert (args.capacity, args.smem_path) == ([128, 256], ["dma", "hbm"])
    assert args.row_mix == "on"
    for flag, value in (
        ("--capacity", "0"),
        ("--capacity", "128,0"),
        ("--capacity", "-128"),
        ("--smem-path", "vmem"),
        ("--row-mix", "maybe"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])
    assert dsa_micro.CASES["compact"].per_tokens

    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    built = []
    monkeypatch.setattr(
        dsa_micro, "_compact_kernel", lambda *a: built.append(a) or (None, [])
    )

    def setup(
        variant="tile",
        pattern="uniform",
        count=64,
        kv_dtype="bf16",
        capacity=128,
        smem_path="dma",
        row_mix="off",
        rows=8,
        tokens=1024,
    ):
        args = SimpleNamespace(row_mix=row_mix)
        return dsa_micro._compact_setup(
            rows, tokens, variant, pattern, count, kv_dtype, capacity, smem_path, args
        )

    got = setup()
    assert got.variant == "tile_uniform_s64_bf16"
    assert got.check is not None and got.check_body is not None
    # The timed kernel drains one block's SMEM chunk, the checked one all.
    assert [a[-1] for a in built] == [False, True]
    keys = {"granularity", "selected", "pattern", "kv_dtype", "capacity"}
    keys |= {"smem_path", "row_mix", "codes", "overflow"}
    assert set(got.extra) == keys and not keys & RECORD_CONTEXT
    selections = dsa_micro._compact_selection(8, 1024, 64, "uniform", False)
    wanted = dsa_micro.compact_expected(
        selections, 8, 1024, "tile", dsa_micro.GEOMETRY["bf16"], 128
    )
    assert got.extra["codes"] == wanted.counts[:, 2].sum()
    assert got.extra["overflow"] == 0 and len(got.operands) == 2
    got = setup("tile", "recent", 64, "fp8", 256, "hbm", "on")
    assert got.variant == "tile_recent_s64_fp8_c256_hbm_mix"
    assert got.extra["smem_path"] == "hbm" and got.extra["row_mix"] == "on"
    # The HBM path carries its bounce buffer as a second aliased output.
    assert len(got.operands) == 3 and len(got.out_specs) == 2
    # Set directly (not parsed), invalid sizes are refused before any kernel.
    built.clear()
    for bad in (
        dict(capacity=100),
        dict(capacity=640),
        dict(capacity=0),
        dict(capacity=-128),
        dict(count=1025),
        dict(count=0),
        dict(rows=3),
    ):
        with pytest.raises(ValueError):
            setup(**bad)
    assert not built

    # The check reassembles the [V / 8, 3, 8, C] blocks into the three lists.
    got = setup("row", row_mix="on", rows=16)
    selections = dsa_micro._compact_selection(16, 1024, 64, "uniform", True)
    wanted = dsa_micro.compact_expected(
        selections, 16, 1024, "row", dsa_micro.GEOMETRY["bf16"], 128
    )
    # Rows 3, 7, 11 and 15 select every token: 512 sublane rows each.
    assert got.extra["overflow"] == 4
    counts = np.zeros((16, 128), np.int32)
    counts[:, :4] = wanted.counts
    blocks = np.stack([wanted.codes, wanted.bitmasks, counts])
    blocks = np.ascontiguousarray(blocks.reshape(3, 2, 8, 128).transpose(1, 0, 2, 3))
    logged = []
    monkeypatch.setattr(dsa_micro.harness, "log", logged.append)

    def verdict():
        return got.check((torch.from_numpy(blocks.copy())[None],))

    assert verdict() and "row_uniform_s64_bf16_mix check PASS" in logged[-1]
    assert logged[-1].startswith("compact_r16_t1024_")
    blocks[1, 1, 3, 0] += 1  # virtual row 11's first bitmask
    assert not verdict() and "FAIL (bitmasks; virtual rows [11])" in logged[-1]
    blocks[1, 1, 3, 0] -= 1
    blocks[0, 2, 5, 4] = 1  # a count lane past the four fields
    assert not verdict() and "FAIL (counts; virtual rows [5])" in logged[-1]


@functools.cache
def _pool(kv_dtype, tokens, rows, rank):
    return dsa_micro._pool_bits(kv_dtype, tokens, rows, rank)


def test_pool_bits_keep_the_gather_stream(monkeypatch):
    import ml_dtypes
    import numpy as np

    # The gather records stay reproducible: the pool's bits and block table are
    # the stream the gather drew before they were factored out.
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    monkeypatch.setattr(dsa_micro, "_upload_bits", lambda bits, dtype: bits)
    monkeypatch.setattr(dsa_micro, "GATHER_POOLS", {})
    for kv_dtype, tokens in (("bf16", 512), ("fp8", 1024)):
        geometry = dsa_micro.GEOMETRY[kv_dtype]
        per_row = tokens // geometry.block_tokens
        shape = (2 * per_row, dsa_micro.POOL_ROWS, geometry.packing, 640)
        dtype, word = {
            "bf16": (ml_dtypes.bfloat16, np.uint16),
            "fp8": (ml_dtypes.float8_e4m3fn, np.uint8),
        }[kv_dtype]
        rng = np.random.default_rng([dsa_micro.GATHER_SEED + geometry.packing, 3])
        flat = np.empty(int(np.prod(shape)), word)
        for start in range(0, flat.size, 2**24):
            part = rng.standard_normal(min(2**24, flat.size - start), np.float32)
            flat[start : start + part.size] = part.astype(dtype).view(word)
        bits, table = dsa_micro._pool_bits(kv_dtype, tokens, 2, 3)
        assert bits.dtype == word and bits.shape == shape
        np.testing.assert_array_equal(bits, flat.reshape(shape))
        np.testing.assert_array_equal(
            table, dsa_micro.block_table(2, per_row, dsa_micro.GATHER_SEED)
        )
        # The gather uploads exactly these bits of its own rank.
        pool = dsa_micro._gather_pool(kv_dtype, tokens, 2)
        np.testing.assert_array_equal(pool.device, bits)
        np.testing.assert_array_equal(pool.table, table)
    with pytest.raises(ValueError, match="whole fp8 blocks"):
        dsa_micro._pool_bits("fp8", 256, 1, 0)


def test_fp8_query_packet_matches_the_decode_wire():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    rng = np.random.default_rng(5)
    query = rng.standard_normal((96, 640)) * 2.0 ** rng.integers(-6, 7, (96, 1))
    query = query.astype(np.float32)
    query[:, 576:] = 0
    query[7] = 0  # an all-zero head
    ours = dsa_micro.fp8_query_packet(query)
    with jax.default_device(jax.devices("cpu")[0]):
        theirs = np.asarray(jax.jit(mla_dcp._pack_fp8_query)(jnp.asarray(query)))
    assert ours.dtype == theirs.dtype == np.uint8
    assert ours.shape == theirs.shape == (96, 640)

    def scales(packet):
        # The FP32 scale's four bytes, repeated over lanes 576:640.
        assert (packet[:, 576:] == np.tile(packet[:, 576:580], 16)).all()
        return np.ascontiguousarray(packet[:, 576:580]).view(np.float32)[:, 0]

    def steps(packet):
        # E4M3 codes in magnitude order, signed.
        codes = packet[:, :576].astype(np.int64)
        return np.where(codes & 0x80, -(codes & 0x7F), codes & 0x7F)

    mine, wire = scales(ours), scales(theirs)
    ulps = np.abs(mine.view(np.int32).astype(np.int64) - wire.view(np.int32))
    assert ulps.max() <= 2
    assert np.abs(steps(ours) - steps(theirs)).max() <= 1
    assert mine[7] == wire[7] == 1.0
    assert not ours[7, :576].any() and not theirs[7, :576].any()


def test_absorbed_queries_give_the_reference_softmax():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tools.megakernel import dsa_reference as ref

    rng = np.random.default_rng(6)
    q_nope = rng.standard_normal((2, 96, 128)).astype(np.float32)
    q_rope = rng.standard_normal((2, 96, 64)).astype(np.float32)
    w_uk = (rng.standard_normal((96, 512, 128)) / np.sqrt(128)).astype(np.float32)
    rows = rng.standard_normal((64, 640)).astype(np.float32)
    query = dsa_micro.absorbed_queries(q_nope, q_rope, w_uk)
    assert query.shape == (2, 96, 640) and query.dtype == np.float32
    assert not query[..., 576:].any()
    for r in range(2):
        logits = query[r, :, :576].astype(np.float64) @ rows[:, :576].T.astype(
            np.float64
        )
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        mine = weights / weights.sum(axis=1, keepdims=True) @ rows[:, :512]
        with jax.default_device(jax.devices("cpu")[0]):
            wanted = np.asarray(
                ref.sparse_mla_attention(
                    jnp.asarray(q_nope[r]),
                    jnp.asarray(q_rope[r]),
                    jnp.asarray(w_uk),
                    jnp.asarray(rows),
                    jnp.arange(63, dtype=jnp.int32),
                    jnp.asarray(rows[63]),
                )
            )
        assert np.abs(mine - wanted).max() / np.abs(wanted).max() < 1e-5


def test_fold_reference_handles_one_and_no_tokens():
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    rng = np.random.default_rng(7)
    q_nope = rng.standard_normal((96, 128)).astype(np.float32)
    q_rope = rng.standard_normal((96, 64)).astype(np.float32)
    w_uk = (rng.standard_normal((96, 512, 128)) / np.sqrt(128)).astype(np.float32)
    latents = rng.standard_normal((16, 640)).astype(np.float32)
    o, lse = dsa_micro.fold_reference(q_nope, q_rope, w_uk, latents, np.array([5]))
    assert o.shape == (96, 512) and lse.shape == (96,)
    np.testing.assert_allclose(o, np.broadcast_to(latents[5, :512], (96, 512)))
    query = dsa_micro.absorbed_queries(q_nope[None], q_rope[None], w_uk)[0]
    np.testing.assert_allclose(lse, query[:, :576] @ latents[5, :576], atol=1e-4)
    o, lse = dsa_micro.fold_reference(q_nope, q_rope, w_uk, latents, np.zeros(0, int))
    assert o.shape == (96, 512) and not o.any()
    assert (lse == mla_dcp.NEG).all() and lse.shape == (96,)


def _fold_staged_tokens(stage, kv_dtype):
    """``[n, 640]``: the bits of every staged token, stream token ``i`` in word
    row ``i // packing`` from bit ``8 itemsize (i % packing)``."""
    import numpy as np

    geometry = dsa_micro.GEOMETRY[kv_dtype]
    width = 8 * geometry.itemsize
    words = stage.words.view(np.uint32)
    shifts = width * np.arange(geometry.packing, dtype=np.uint32)
    tokens = (words[:, None, :] >> shifts[None, :, None]) & ((1 << width) - 1)
    return tokens.reshape(-1, 640)


def _fold_mask_bits(stage, offset, step):
    import numpy as np

    words = stage.masks[offset : offset + step // 32].view(np.uint32)
    bits = (words[:, None] >> np.arange(32, dtype=np.uint32)) & 1
    return bits.reshape(-1).astype(bool)


FOLD_FORMS = [
    ("row", "none"),
    ("tile", "none"),
    ("tile", "onehot"),
    ("tile", "copy"),
    ("page", "none"),
    ("page", "onehot"),
    ("page", "copy"),
    ("dense", "none"),
    ("none", "none"),
]


@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize(
    "rows, tokens, count", [(1, 1024, 64), (3, 2048, 300), (8, 4096, 64)]
)
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("variant, compaction", FOLD_FORMS)
def test_fold_stage_streams_exactly_the_selected_tokens(
    variant, compaction, kv_dtype, rows, tokens, count, narrow
):
    import numpy as np

    m = dsa_micro
    geometry = m.GEOMETRY[kv_dtype]
    pages = variant == "page"
    step = (512 if narrow else 1024) if pages else (128 if narrow else 512)
    pattern = "recent" if rows == 3 else "uniform"
    bits, table = _pool(kv_dtype, tokens, rows, 3)
    stage = m.fold_stage(
        rows, tokens, count, pattern, kv_dtype, variant, step, compaction, True,
        bits, table,
    )  # fmt: skip
    meta = stage.meta.reshape(-1, 6)
    assert stage.meta.dtype == stage.masks.dtype == np.int32
    assert stage.copy_tokens.dtype == stage.words.dtype == np.int32
    assert len(meta) == stage.steps and stage.words.shape[1] == 640
    counts = m.row_mix_counts(count, tokens, rows, True)
    sizes = [len(s) for s in stage.selections]
    assert sizes == ([0] * rows if variant == "none" else counts)
    # The rank-free selection: row r's seed is (T, S, pattern, r).
    if variant != "none":
        rng = np.random.default_rng(
            [m.FOLD_SEED, tokens, count, m.GATHER_PATTERNS.index(pattern), 0]
        )
        np.testing.assert_array_equal(
            stage.selections[0], m.select_tokens(rng, tokens, count, pattern)
        )
    assert (np.diff(meta[:, 0]) >= 0).all()
    assert sorted(set(meta[:, 0].tolist())) == list(range(rows))
    staged = _fold_staged_tokens(stage, kv_dtype)
    nan = {"bf16": 0x7FC0, "fp8": 0x7F}[kv_dtype]
    size = 1 if variant in ("dense", "none") else m.unit_tokens(variant, geometry)
    offset = chunk = folded = 0
    for r in range(rows):
        mine = meta[meta[:, 0] == r]
        flags = mine[:, 1]
        assert flags[0] & m.STEP_FIRST and flags[-1] & m.STEP_LAST
        assert not (flags[1:] & m.STEP_FIRST).any()
        assert not (flags[:-1] & m.STEP_LAST).any()
        selected, stream = stage.selections[r], stage.streams[r]
        if not len(selected):
            assert len(mine) == 1 and len(stream) == 0
            assert flags[0] == m.STEP_FIRST | m.STEP_LAST | m.STEP_EMPTY
            assert mine[0, 2] == -1
            continue
        assert not (flags & m.STEP_EMPTY).any()
        # Whole units holding a selected token, ascending; padding (-1) only
        # in the last step.
        real = stream[stream >= 0]
        units = np.unique(selected // size)
        np.testing.assert_array_equal(
            real, (units[:, None] * size + np.arange(size)).ravel()
        )
        assert len(stream) % step == 0 and len(stream) - len(real) < step
        assert (stream[len(real) :] == -1).all()
        # The staged words decode to the stream's latent rows, NaN lanes
        # 576:640 on every token, padding zero.
        block, sublane, slot = m.locate(real, geometry)
        got = staged[offset : offset + len(real)]
        np.testing.assert_array_equal(
            got[:, :576], bits[table[r][block], sublane, slot][:, :576]
        )
        assert (got[:, 576:] == nan).all()
        assert not staged[offset + len(real) : offset + len(stream)].any()
        offset += len(stream)
        # The masked steps take the stream's chunks in order and their masks
        # are exactly the selection; a fold step always holds one.
        kind = m.STEP_FOLD if compaction == "none" else m.STEP_COMPACT
        masked = mine[(flags & (m.STEP_FOLD | m.STEP_COMPACT)) != 0]
        assert ((masked[:, 1] & (m.STEP_FOLD | m.STEP_COMPACT)) == kind).all()
        assert len(masked) == len(stream) // step
        picked = []
        for k, (_, flag, c, at, n, copy_at) in enumerate(masked):
            assert c == chunk
            chunk += 1
            bits_k = _fold_mask_bits(stage, at, step)
            part = stream[k * step : (k + 1) * step][bits_k]
            assert (part >= 0).all() and n == len(part)
            assert len(part) or kind == m.STEP_COMPACT
            if compaction == "copy":
                np.testing.assert_array_equal(
                    stage.copy_tokens[copy_at : copy_at + n], np.flatnonzero(bits_k)
                )
            picked.append(part)
        np.testing.assert_array_equal(np.concatenate(picked), selected)
        dense = mine[(flags & m.STEP_DENSE) != 0]
        if compaction == "none":
            assert len(dense) == 0
            folded += len(stream)
        else:
            # COMPACT steps first, then the dense buffer in whole steps.
            assert (flags[: len(masked)] & m.STEP_COMPACT).all()
            assert len(dense) == -(-len(selected) // step) == len(mine) - len(masked)
            assert dense[:, 2].tolist() == [-1] * len(dense)
            assert dense[:, 3].tolist() == list(range(len(dense)))
            assert dense[:, 4].sum() == len(selected) and (dense[:-1, 4] == step).all()
            folded += len(dense) * step
    assert chunk == stage.chunks and offset <= len(staged)
    assert not staged[offset:].any()
    assert stage.folded_tokens == folded


def test_fold_stage_refuses_points_the_kernel_cannot_take(monkeypatch):
    m = dsa_micro
    bf16, fp8 = _pool("bf16", 1024, 2, 3), _pool("fp8", 1024, 2, 3)

    def stage(variant="tile", step=128, compaction="none", pool=bf16, kv_dtype="bf16"):
        return m.fold_stage(
            2, 1024, 64, "uniform", kv_dtype, variant, step, compaction, False, *pool
        )

    stage()
    for bad in (
        dict(step=96),
        dict(step=0),
        dict(variant="page", step=128),
        dict(variant="page", step=256, pool=fp8, kv_dtype="fp8"),
        dict(variant="row", compaction="onehot"),
        dict(variant="dense", compaction="copy"),
    ):
        with pytest.raises(ValueError):
            stage(**bad)
    monkeypatch.setattr(m, "FOLD_SMEM_WORDS", 64)
    with pytest.raises(ValueError, match="SMEM"):
        stage()
    monkeypatch.setattr(m, "FOLD_SMEM_WORDS", 32768)
    monkeypatch.setattr(m, "FOLD_DENSE_TOKENS", 32)
    stage()
    with pytest.raises(ValueError, match="dense"):
        stage(compaction="onehot")


def test_fold_plan_is_the_same_on_every_rank():
    import numpy as np

    # One SPMD op: steps, chunks and table and operand shapes must not depend
    # on the rank; only the pool bits do.
    for variant, compaction in (("row", "none"), ("tile", "copy"), ("page", "onehot")):
        stages = [
            dsa_micro.fold_stage(
                8, 2048, 300, "uniform", "fp8", variant, 512, compaction, True,
                *_pool("fp8", 2048, 8, rank),
            )
            for rank in (0, 31)
        ]  # fmt: skip
        a, b = stages
        for name in ("meta", "masks", "copy_tokens"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert (a.steps, a.chunks, a.folded_tokens) == (
            b.steps,
            b.chunks,
            b.folded_tokens,
        )
        assert a.words.shape == b.words.shape and (a.words != b.words).any()


@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("count", [1, 64, 300])
@pytest.mark.parametrize("variant", ["row", "tile", "page", "dense"])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_fold_model_within_tolerance_and_with_teeth(kv_dtype, variant, count, narrow):
    import numpy as np

    m = dsa_micro
    step = (512 if narrow else 1024) if variant == "page" else (128 if narrow else 512)
    bits, table = _pool(kv_dtype, 2048, 2, 0)
    stage = m.fold_stage(
        2, 2048, count, "uniform", kv_dtype, variant, step, "none", False, bits, table
    )
    rng = np.random.default_rng([m.FOLD_SEED, count])
    w_uk = (rng.standard_normal((96, 512, 128)) / np.sqrt(128)).astype(np.float32)
    q_nope = rng.standard_normal((96, 128)).astype(np.float32)
    q_rope = rng.standard_normal((96, 64)).astype(np.float32)
    query = m.absorbed_queries(q_nope[None], q_rope[None], w_uk)[0]
    values, scale = m.fold_query_values(query, kv_dtype)
    selected = stage.selections[0]
    o_ref, lse_ref = m.fold_reference(
        q_nope, q_rope, w_uk,
        m.latent_values(bits, table, 0, selected, kv_dtype), np.arange(len(selected)),
    )  # fmt: skip
    bound_o, bound_lse = m.FOLD_TOLERANCE[kv_dtype]

    def errors(row, **fault):
        stream = stage.streams[row]
        latents = m.latent_values(bits, table, row, stream, kv_dtype)
        latents[:, 576:] = 0
        mask = np.isin(stream, stage.selections[row])
        got = m.fold_model(values, scale, latents, mask, step, kv_dtype, **fault)
        return m.fold_errors(*got, o_ref, lse_ref)

    o_error, lse_error = errors(0)
    assert o_error <= bound_o and lse_error <= bound_lse
    # Two gross faults fail the bound: another row's tokens, and (streams of two
    # or more steps; one step starts from O = 0, l = 0) no rescaling.
    assert errors(1)[0] > bound_o
    steps = len(stage.streams[0]) // step
    assert steps >= 2 or not (count == 300 and narrow)
    if steps >= 2:
        assert errors(0, rescale=False)[0] > bound_o


def test_fold_model_folds_one_step_as_the_softmax_does():
    import numpy as np

    rng = np.random.default_rng(8)
    query = rng.standard_normal((96, 640)).astype(np.float32)
    stream = rng.standard_normal((256, 640)).astype(np.float32)
    stream[:, 576:] = 0
    mask = rng.random(256) < 0.3
    o, lse = dsa_micro.fold_model(query, np.ones((96, 1)), stream, mask, 128, "bf16")
    logits = (query.astype(np.float64) @ stream.T.astype(np.float64))[:, mask]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    wanted = weights / weights.sum(axis=1, keepdims=True) @ stream[mask, :512]
    # BF16 probabilities: a few parts in a thousand.
    assert np.abs(o - wanted).max() / np.abs(wanted).max() < 1e-2
    top = logits.max(axis=1)
    np.testing.assert_allclose(
        lse, top + np.log(np.exp(logits - top[:, None]).sum(axis=1)), rtol=1e-5
    )
    # FP8: P as E4M3 of 256 p, and the per-head scale on the logits.
    scale = np.full((96, 1), 0.5, np.float32)
    o8, lse8 = dsa_micro.fold_model(query, scale, stream, mask, 128, "fp8")
    np.testing.assert_allclose(lse8, top / 2 + np.log(
        np.exp((logits - top[:, None]) / 2).sum(axis=1)), rtol=1e-5)  # fmt: skip
    weights = np.exp((logits - top[:, None]) / 2)
    wanted = weights / weights.sum(axis=1, keepdims=True) @ stream[mask, :512]
    # E4M3 probabilities: a few parts in a hundred.
    assert np.abs(o8 - wanted).max() / np.abs(wanted).max() < 0.1
    # No tokens at all: (0, NEG).
    empty = dsa_micro.fold_model(query, scale, stream[:0], mask[:0], 128, "fp8")
    assert not empty[0].any() and (empty[1] == -1e30).all()
    # fold_errors: per row, relative o and absolute lse errors.
    got = dsa_micro.fold_errors(o, lse, wanted, lse + 0.25)
    assert got[0] == pytest.approx(np.abs(o - wanted).max() / np.abs(wanted).max())
    assert got[1] == pytest.approx(0.25)


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_fold_exact_lse_gate_fails_dropped_tokens(kv_dtype):
    import numpy as np

    m = dsa_micro
    tokens, rows, count = 2048, 4, 512
    bits, table = _pool(kv_dtype, tokens, rows, 0)
    rng = np.random.default_rng([m.FOLD_SEED, 0])
    w_uk = (rng.standard_normal((96, 512, 128)) / np.sqrt(128)).astype(np.float32)
    bound = m.FOLD_EXACT_LSE_TOLERANCE[kv_dtype]
    for row in range(rows):
        q_nope = rng.standard_normal((96, 128)).astype(np.float32)
        q_rope = rng.standard_normal((96, 64)).astype(np.float32)
        query = m.absorbed_queries(q_nope[None], q_rope[None], w_uk)[0]
        values, scale = m.fold_query_values(query, kv_dtype)
        selected = m.select_tokens(
            np.random.default_rng([m.FOLD_SEED, tokens, count, 0, row]),
            tokens, count, "uniform",
        )  # fmt: skip
        exact = m.fold_exact_lse(
            values, scale, m.latent_values(bits, table, row, selected, kv_dtype)
        )
        for variant, step in (("dense", 128), ("tile", 256)):
            stream = m.fold_stream(selected, variant, m.GEOMETRY[kv_dtype], step)
            latents = m.latent_values(bits, table, row, stream, kv_dtype)
            latents[:, 576:] = 0
            mask = np.isin(stream, selected)
            one = mask.copy()
            one[np.flatnonzero(mask)[count // 2]] = False
            every_32nd = mask & (np.arange(len(mask)) % 32 != 31)
            assert every_32nd.sum() < count

            errors = [
                np.abs(
                    m.fold_model(values, scale, latents, folded, step, kv_dtype)[1]
                    - exact
                ).max()
                for folded in (mask, one, every_32nd)
            ]
            assert errors[0] <= bound
            assert min(errors[1:]) > bound, (row, variant, errors)


def test_fold_flags_tags_and_records(monkeypatch):
    import numpy as np

    m = dsa_micro
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "fold"])
    assert (args.fold_tokens, args.fold_compaction, args.fold_compute) == (
        [256],
        ["none"],
        ["on"],
    )
    args = parse(
        ["bench", "dsa-micro", "--cases", "fold", "--variant", "row,dense,none",
         "--fold-tokens", "128,512", "--fold-compaction", "none,onehot,copy",
         "--fold-compute", "on,off"]
    )  # fmt: skip
    assert args.variant == ["row", "dense", "none"] and args.fold_tokens == [128, 512]
    assert args.fold_compaction == ["none", "onehot", "copy"]
    assert args.fold_compute == ["on", "off"]
    for flag, value in (
        ("--fold-tokens", "0"),
        ("--fold-tokens", "128,-128"),
        ("--fold-compaction", "scatter"),
        ("--fold-compute", "maybe"),
        ("--variant", "sparse"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])
    assert m.CASES["fold"].per_tokens
    # Other cases keep their own variant names.
    assert [v for v in args.variant if v in m.GATHER_VARIANTS] == ["row"]

    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: bits)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    built = []
    monkeypatch.setattr(
        m, "_fold_kernel", lambda plan: built.append(plan) or (None, [])
    )

    def setup(
        variant="tile",
        pattern="uniform",
        count=64,
        kv_dtype="bf16",
        fold_tokens=256,
        compaction="none",
        compute="on",
        row_mix="off",
        rows=8,
        tokens=1024,
    ):
        args = SimpleNamespace(rows=[rows], row_mix=row_mix)
        return m._fold_setup(
            rows, tokens, variant, pattern, count, kv_dtype, fold_tokens, compaction,
            compute, args,
        )  # fmt: skip

    got = setup()
    assert got.variant == "tile_uniform_s64_bf16_k256" and not got.invalid
    assert got.check is not None and got.check_body is not None
    # The timed kernel writes the last row's result, the checked one every row's.
    assert [plan.full for plan in built] == [False, True]
    assert (
        built[0].steps == got.extra["steps"] and built[0].chunks == got.extra["chunks"]
    )
    keys = {"granularity", "selected", "pattern", "kv_dtype", "fold_tokens"}
    keys |= {"compaction", "compute", "row_mix", "steps", "chunks", "folded_tokens"}
    keys |= {"selected_tokens"}
    assert set(got.extra) == keys
    assert not (keys | {"max_error", "lse_error", "model_error"}) & RECORD_CONTEXT
    assert got.extra["selected_tokens"] == 8 * 64 and got.extra["compute"] == "on"
    assert len(got.operands) == 6 and len(got.in_specs) == 6
    meta, masks, copies, queries, words, out = got.operands
    assert queries.shape == (8, 96, 640) and str(queries.dtype) == "bfloat16"
    assert out.shape == (8, 96, 640) and not out.any()
    assert words.shape == (got.extra["folded_tokens"] // 2, 640)
    got = setup("tile", "recent", 64, "fp8", 512, "onehot", "on", "on")
    assert got.variant == "tile_recent_s64_fp8_k512_onehot_mix"
    assert got.operands[3].dtype == np.uint8 and built[-1].dense_tokens > 0
    assert setup("page", fold_tokens=512, compaction="copy").variant == (
        "page_uniform_s64_bf16_k512_copy"
    )
    built.clear()
    got = setup("row", compute="off")
    assert got.variant == "row_uniform_s64_bf16_k256_nocompute" and got.invalid
    assert got.check is None and got.check_body is None
    assert [plan.compute for plan in built] == [False]
    for kwargs in (dict(), dict(count=300, pattern="recent", fold_tokens=512)):
        got = setup("none", **kwargs)
        assert got.variant == "none_bf16" and got.extra["steps"] == 8
        assert got.extra["selected_tokens"] == 0 and got.extra["chunks"] == 0
    # Set directly (not parsed), invalid sizes are refused before any kernel.
    built.clear()
    for bad in (
        dict(fold_tokens=96),
        dict(fold_tokens=0),
        dict(fold_tokens=-128),
        dict(count=0),
        dict(count=1025),
        dict(variant="row", compaction="onehot"),
        dict(variant="page", fold_tokens=128),
        dict(variant="sparse"),
        dict(compaction="scatter"),
    ):
        with pytest.raises(ValueError):
            setup(**bad)
    assert not built

    # The grid: compactions only on tile and page, S above T skipped, `none`
    # once, and a point over the SMEM guard skipped with a log line.
    args = parse(
        ["bench", "dsa-micro", "--cases", "fold", "--variant", "row,tile,page,none",
         "--selected", "64,300,4096", "--kv-dtype", "bf16", "--pattern", "uniform",
         "--fold-tokens", "128,256", "--fold-compaction", "none,onehot",
         "--rows", "8"]
    )  # fmt: skip
    logged = []
    monkeypatch.setattr(m.harness, "log", logged.append)
    tags = [s.variant for s in m._build_fold(8, 1024, args)]
    assert len(tags) == len(set(tags)) and tags.count("none_bf16") == 1
    assert (
        "row_uniform_s64_bf16_k128" in tags
        and "tile_uniform_s300_bf16_k256_onehot" in tags
    )
    assert not [t for t in tags if t.startswith("row") and "onehot" in t]
    assert not [t for t in tags if "s4096" in t]
    # BF16 pages are 256 tokens: no page point at 128.
    assert "page_uniform_s64_bf16_k256" in tags and not [
        t for t in tags if "page" in t and "k128" in t
    ]
    assert len(tags) == 2 * 2 + 2 * 2 * 2 + 2 * 2 + 1
    assert not logged
    smem = m.fold_stage(8, 1024, 300, "uniform", "bf16", "tile", 128, "none", False,
                        *m._fold_pool("bf16", 1024, 8))  # fmt: skip
    monkeypatch.setattr(m, "FOLD_SMEM_WORDS", len(smem.masks) + 6 * smem.steps - 1)
    tags = [s.variant for s in m._build_fold(8, 1024, args)]
    assert (
        "tile_uniform_s300_bf16_k128" not in tags
        and "tile_uniform_s64_bf16_k128" in tags
    )
    assert any(
        "tile_uniform_s300_bf16_k128" in line and "SMEM" in line for line in logged
    )


def test_fold_check_passes_the_model_and_fails_faults(monkeypatch):
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    m = dsa_micro
    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: bits)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    monkeypatch.setattr(m, "_fold_kernel", lambda plan: (None, []))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, mine: out.__setitem__(0, mine),
    )
    logged = []
    monkeypatch.setattr(m.harness, "log", logged.append)
    args = SimpleNamespace(rows=[4], row_mix="on")
    for kv_dtype, variant, compaction in (
        ("fp8", "tile", "copy"),
        ("bf16", "row", "none"),
    ):
        got = m._fold_setup(
            4, 1024, variant, "uniform", 64, kv_dtype, 128, compaction, "on", args
        )
        q_nope, q_rope, w_uk, _ = m._fold_inputs(4)
        bits, table = m._fold_pool(kv_dtype, 1024, 4)
        _, _, _, queries = m._fold_inputs(4)
        stage = m.fold_stage(
            4, 1024, 64, "uniform", kv_dtype, variant, 128, compaction, True,
            bits, table,
        )  # fmt: skip
        out = np.zeros((4, 96, 640), np.float32)
        reference = np.zeros_like(out)
        # Rows of 64, no, one and every token: fold_model's (o, lse) over the
        # stream the kernel folds, and the reference's.
        for r, count in enumerate((64, 0, 1, 1024)):
            selected = stage.selections[r]
            assert len(selected) == count
            stream = selected if compaction != "none" else stage.streams[r]
            latents = m.latent_values(bits, table, r, stream, kv_dtype)
            latents[:, 576:] = 0
            values, scale = m.fold_query_values(queries[r], kv_dtype)
            o, lse = m.fold_model(
                values, scale, latents, np.isin(stream, selected), 128, kv_dtype
            )
            out[r, :, :512], out[r, :, 512] = o, lse
            o, lse = m.fold_reference(
                q_nope[r], q_rope[r], w_uk,
                m.latent_values(bits, table, r, selected, kv_dtype), np.arange(count),
            )  # fmt: skip
            reference[r, :, :512], reference[r, :, 512] = o, lse

        def verdict(values, setup=got):
            return setup.check((torch.from_numpy(values)[None],))

        assert verdict(out) and "check PASS" in logged[-1]
        assert logged[-1].startswith(f"fold_r4_t1024_{got.variant} ")
        assert got.extra["model_error"] == 0 and got.extra["exact_lse_error"] < 1e-5
        assert got.extra["max_error"] <= m.FOLD_TOLERANCE[kv_dtype][0]
        assert (out[1, :, 512] == np.float32(mla_dcp.NEG)).all()
        # The reference's lse is not that of the kernel's query and cache values.
        assert not verdict(reference) and got.extra["exact_lse_error"] > 1e-3

        def lse_shift(x):
            x[0, :, 512] += 1e-3

        def o_off(x):
            x[0, :, :512] += 1.0

        def lse_off(x):
            x[3, :, 512] += 10.0

        def empty_lse(x):
            x[1, :, 512] = 0.0

        def empty_o(x):
            x[1, :, 5] = 1e-30

        def stray_lane(x):
            x[2, :, 600] = 1.0

        def poisoned(x):
            x[0, 4, 7] = np.nan

        edits = (lse_shift, o_off, lse_off, empty_lse, empty_o, stray_lane, poisoned)
        for edit in edits:
            bad = out.copy()
            edit(bad)
            assert not verdict(bad) and "check FAIL" in logged[-1], edit.__name__


def test_fold_exact_lse_tolerance_follows_where_the_kernel_runs(monkeypatch):
    import jax

    m = dsa_micro
    for kv_dtype in ("bf16", "fp8"):
        assert m.fold_exact_lse_tolerance(kv_dtype) == 1e-4
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert m.fold_exact_lse_tolerance("bf16") == m.FOLD_EXACT_LSE_TOLERANCE_TPU["bf16"]
    assert m.fold_exact_lse_tolerance("fp8") == m.FOLD_EXACT_LSE_TOLERANCE_TPU["fp8"]
    assert m.FOLD_EXACT_LSE_TOLERANCE_TPU["bf16"] == 1e-4


def _drop_mask_token(stage, step_tokens):
    """Clear the mask bit of row 0's middle selected token (its step keeps
    another)."""
    import numpy as np

    m = dsa_micro
    stream, selected = stage.streams[0], stage.selections[0]
    chosen = np.isin(stream, selected)
    hits = np.flatnonzero(chosen)
    p = hits[len(hits) // 2]
    assert chosen[p // step_tokens * step_tokens :][:step_tokens].sum() > 1
    offset = stage.meta[m.META_FIELDS * (p // step_tokens) + 3]
    word = offset + p % step_tokens // m.WORD_TOKENS
    bits = stage.masks[word : word + 1].view(np.uint32)
    bits &= ~np.uint32(1 << (p % m.WORD_TOKENS))


def _swap_packed_slot(stage, step_tokens):
    """Row 0's middle selected token whose 4-token word holds an unselected
    token reads that token's bytes: one wrong packing slot."""
    import numpy as np

    stream, selected = stage.streams[0], stage.selections[0]
    chosen = np.isin(stream, selected)
    words = stage.words.view(np.uint8).reshape(len(stage.words), -1, 4)
    hits = [
        p
        for p in np.flatnonzero(chosen)
        if not chosen[p // 4 * 4 : p // 4 * 4 + 4].all()
    ]
    p = hits[len(hits) // 2]
    q = next(q for q in range(p // 4 * 4, p // 4 * 4 + 4) if not chosen[q])
    words[p // 4, :576, p % 4] = words[q // 4, :576, q % 4]


def _shift_copy_tile(stage, step_tokens):
    """The middle copied token is read from the next (or previous) 32-token
    tile of its chunk."""
    j = len(stage.copy_tokens) // 2
    t = int(stage.copy_tokens[j])
    stage.copy_tokens[j] = t + 32 if t + 32 < step_tokens else t - 32


def _short_onehot_count(stage, step_tokens):
    """The COMPACT step with the most selected tokens counts one fewer, so the
    one-hot window merges all but its last token."""
    import numpy as np

    m = dsa_micro
    meta = stage.meta.reshape(-1, m.META_FIELDS)
    compact = np.flatnonzero((meta[:, 1] & m.STEP_COMPACT) != 0)
    i = compact[np.argmax(meta[compact, 4])]
    assert meta[i, 4] >= 2
    meta[i, 4] -= 1


@pytest.mark.parametrize(
    "variant, compaction, fault",
    [
        ("row", "none", None),
        ("row", "none", _drop_mask_token),
        ("row", "none", _swap_packed_slot),
        ("tile", "copy", None),
        ("tile", "copy", _shift_copy_tile),
        ("tile", "onehot", None),
        ("tile", "onehot", _short_onehot_count),
    ],
)
def test_fold_kernel_interpreted_loses_no_fp8_token(
    monkeypatch, variant, compaction, fault
):
    """The kernel in Pallas interpret mode (exact dots) at FP8, S = 512: it
    passes the 1e-4 exact-lse gate, and one selected token lost through the
    mask, a packed word's slot, the copy tile or the one-hot window fails it."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    m = dsa_micro
    step_tokens = 256
    monkeypatch.setattr(bridge, "upload", lambda value: jnp.asarray(value)[None])
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: jnp.asarray(bits)[None])
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, mine: out.__setitem__(0, mine),
    )
    monkeypatch.setattr(m.harness, "log", lambda line: None)
    if fault is not None:
        stage_of = m.fold_stage

        def faulted(*args, **kwargs):
            stage = stage_of(*args, **kwargs)
            fault(stage, step_tokens)
            return stage

        monkeypatch.setattr(m, "fold_stage", faulted)
    setup = m._fold_setup(
        1, 2048, variant, "uniform", 512, "fp8", step_tokens, compaction, "on",
        SimpleNamespace(rows=[1], row_mix="off"),
    )  # fmt: skip
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        full = jax.block_until_ready(setup.check_body(*setup.operands, 1))
    ok = setup.check((torch.from_numpy(np.asarray(full[0])),))
    error = setup.extra["exact_lse_error"]
    assert m.fold_exact_lse_tolerance("fp8") == 1e-4
    if fault is None:
        assert ok and error <= 1e-5, error
    else:
        assert not ok and error > 1e-4, error


@pytest.mark.parametrize("block_tokens", [256, 512])
def test_score_index_blocks_round_trip_through_the_cache_layout(block_tokens):
    import jax
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    tokens = 2 * block_tokens
    keys = dsa_micro.score_keys(np.random.default_rng(0), tokens)
    with jax.default_device(jax.devices("cpu")[0]):
        key_bytes, exponent = (
            np.asarray(x) for x in dsa_cache.quantize_index_keys(keys)
        )
        wanted = np.asarray(dsa_cache.dequantize_index_keys(key_bytes, exponent))
        # The per-token magnitudes give many different scales.
        assert len(np.unique(exponent)) > 4
        blocks = dsa_micro.pack_index_blocks(key_bytes, exponent, block_tokens)
        assert blocks.shape == (2, *dsa_cache.index_block_shape(block_tokens))
        assert blocks.dtype == np.uint8
        # Token t of block b: key bytes in row t, its exponent at row
        # tokens_per_block + t // 128, lane t % 128; unused scale rows zero.
        t = np.arange(tokens) % block_tokens
        b = np.arange(tokens) // block_tokens
        got = dsa_cache.dequantize_index_keys(
            blocks[b, t], blocks[b, block_tokens + t // 128, t % 128]
        )
        np.testing.assert_array_equal(np.asarray(got), wanted)
    assert not blocks[:, block_tokens + block_tokens // 128 :].any()
    with pytest.raises(ValueError, match="whole"):
        dsa_micro.pack_index_blocks(key_bytes[:300], exponent[:300], block_tokens)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_score_head_sum_model_equals_the_reference(rows):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tools.megakernel import dsa_reference as ref

    rng = np.random.default_rng(rows)
    tokens = 256
    q = rng.standard_normal((rows, 32, 128)).astype(np.float32)
    w = rng.standard_normal((rows, 32)).astype(np.float32)
    keys = rng.standard_normal((tokens, 128)).astype(np.float32)
    index = dsa_micro.score_query_rows(rows)
    assert sorted(index.ravel().tolist()) == list(range(32 * rows))
    q_rows = np.zeros((32 * rows, 128), np.float32)
    w_rows = np.zeros(32 * rows, np.float32)
    q_rows[index], w_rows[index] = q, w
    got = dsa_micro.head_sum_model(q_rows @ keys.T, w_rows, rows)
    padded = dsa_micro.padded_rows(rows)
    assert got.shape == (padded, tokens) and not got[rows:].any()
    with jax.default_device(jax.devices("cpu")[0]):
        wanted = np.asarray(
            ref.scores(jnp.asarray(q), jnp.asarray(w), jnp.asarray(keys))
        )
    np.testing.assert_allclose(got[:rows], wanted, rtol=1e-4, atol=1e-3)


def test_score_query_rows_are_head_major_within_a_group():
    # Head h of row r of a group of eight is sublane r of the group's tile h.
    index = dsa_micro.score_query_rows(16)
    assert index[9, 3].item() == 256 + 3 * 8 + 1
    assert dsa_micro.score_query_rows(2)[1, 5].item() == 5 * 2 + 1


def test_score_error_and_recall():
    import numpy as np

    wanted = np.array([[1.0, -4.0, 2.0, 0.5], [4.0, 1.0, 2.0, 0.0]], np.float32)
    got = wanted + np.array([[0, 0.5, 0, -0.25], [0, 0, 0.0625, 0]], np.float32)
    np.testing.assert_array_equal(dsa_micro.score_errors(got, wanted), [1 / 8, 1 / 64])
    # Top 2: row 0 wants {2, 0} and gets {2, 0}; row 1 wants {0, 2}, gets {0, 2}.
    np.testing.assert_allclose(dsa_micro.topk_recall(got, wanted, 2), [1.0, 1.0])
    swapped = wanted.copy()
    swapped[0, 3] = 5.0
    np.testing.assert_allclose(dsa_micro.topk_recall(swapped, wanted, 2), [0.5, 1.0])
    # At most k tokens: every one is selected.
    np.testing.assert_allclose(dsa_micro.topk_recall(swapped, wanted, 4), [1.0, 1.0])


def test_score_check_gives_each_rank_its_own_verdict_and_records_the_group(
    monkeypatch,
):
    import numpy as np

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((2, 256)).astype(np.float32)
    inputs = dsa_micro.ScoreInputs(None, None, None, None, reference)
    peer = (np.full(2, 0.5), np.array([0.25, 1.0]), True, (reference * 0, reference))
    logged = []
    monkeypatch.setattr(dsa_micro.harness, "log", logged.append)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 32)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    # This rank is 3; the other 31 report the same bad scores.
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, mine: out.__setitem__(
            slice(None), [mine if r == 3 else (r, peer) for r in range(32)]
        ),
    )

    def verdict(got, extra):
        check = dsa_micro._score_check(2, inputs, 1e-5, extra, "score")
        return check((torch.from_numpy(got)[None],))

    exact = np.zeros((8, 256), np.float32)
    exact[:2] = reference
    extra = {}
    # The peers' errors do not fail this rank; they are in the record.
    assert verdict(exact, extra) is True
    assert extra["max_error"] == 0.5 and extra["recall_local"] == 0.25
    assert "recall_global" in extra and "(rank 0)" in logged[-1]
    # This rank's own errors do: a score off by more than the bound, or a
    # padded row that was written.
    off = exact.copy()
    off[1, 5] += 1.0
    assert verdict(off, {}) is False
    padded = exact.copy()
    padded[7, 0] = 1.0
    assert verdict(padded, {}) is False
    # A rank whose outputs cannot be read still joins the gather, then raises.
    with pytest.raises(RuntimeError, match="TypeError"):
        dsa_micro._score_check(2, inputs, 1e-5, {}, "score")((None,))


def test_score_flags(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "score"])
    assert args.cases == ["score"] and args.block_tokens == [256, 512]
    assert args.writeback == "slice"
    args = parse(
        ["bench", "dsa-micro", "--cases", "score,gather", "--variant", "fp8q,page",
         "--block-tokens", "512", "--writeback", "full"]
    )  # fmt: skip
    assert args.variant == ["fp8q", "page"] and args.block_tokens == [512]
    assert args.writeback == "full"
    assert [v for v in args.variant if v in dsa_micro.SCORE_VARIANTS] == ["fp8q"]
    for flag, value in (
        ("--variant", "fp16q"),
        ("--block-tokens", "128"),
        ("--writeback", "none"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])


def test_score_plans_cap_and_deduplicate(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--dot-tokens", "128,256,512",
         "--dot-groups", "1,8", "--prefetch", "1,4,64"]
    )  # fmt: skip
    assert args.prefetch == [1, 4, 64]
    plans = dsa_micro.score_plans(1, 2048, args)
    # One row group; 2048 tokens are 8 / 4 blocks; 512 keys do not fit a
    # 256-token block.
    assert {(p.block_tokens, p.dot_tokens) for p in plans} == {
        (256, 128),
        (256, 256),
        (512, 128),
        (512, 256),
        (512, 512),
    }
    assert {p.dot_groups for p in plans} == {1}
    assert {(p.block_tokens, p.prefetch) for p in plans} == {
        (256, 1),
        (256, 4),
        (256, 8),
        (512, 1),
        (512, 4),
    }
    assert len(plans) == len(set(plans))
    by_block = {p.block_tokens: p for p in plans if p.prefetch == 4}
    assert by_block[256].slots == 5 and by_block[512].slots == 4
    assert {p.dot_groups for p in dsa_micro.score_plans(64, 2048, args)} == {1, 8}
    assert not any(p.full for p in plans)
    with pytest.raises(ValueError, match="whole blocks"):
        dsa_micro.score_plans(1, 1280, args)


def test_score_decode_plans_take_the_decode_scorers_form(monkeypatch):
    """--score-pages decode is the decode scorer's own form (select, scores,
    one page a step, full work, its own ring, BF16-held queries); other forms
    and an asked-for fp8q are refused. Its record's prefetch is the ring's
    pages in flight. Its setup hands the scorer local tables, every row's
    position, the pool with a layer axis and the new token's handoff."""
    import ml_dtypes
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    m = dsa_micro
    _score_on_host(monkeypatch)
    base = dict(
        block_tokens=[512], dot_tokens=[128, 256], dot_groups=[1, 8], prefetch=[16],
        writeback="slice", layout=["select"], emit=["scores"], score_pages=["decode"],
        pages_per_step=[1], score_work=["full"],
    )  # fmt: skip
    (plan,) = m.score_plans(8, 1024, SimpleNamespace(**base))
    assert (plan.pages, plan.prefetch, plan.dot_tokens, plan.dot_groups) == (
        "decode",
        dsa_decode.RING_DEPTH,
        128,
        1,
    )
    for flag, value in (
        ("layout", ["padded"]),
        ("emit", ["keys"]),
        ("pages_per_step", [2]),
        ("score_work", ["compute-only"]),
    ):
        with pytest.raises(ValueError, match="score-pages decode times the decode"):
            m.score_plans(8, 1024, SimpleNamespace(**{**base, flag: value}))
    setup = m._score_setup(8, 1024, "bf16q", plan)
    assert setup.variant == "bf16q_b512_vsel_dec"
    assert (setup.extra["score_pages"], setup.extra["page_order"]) == (
        "decode",
        "decode",
    )
    assert setup.extra["prefetch"] == dsa_decode.RING_DEPTH
    assert not set(setup.extra) & RECORD_CONTEXT
    table, positions, pool, queries, weights, new_values, out = setup.operands
    assert positions.tolist() == [m.RANKS * 1024 - 1] * 8
    assert sorted(table.tolist()) == list(range(16)) and pool.shape == (16, 1, 544, 128)
    assert queries.shape == weights.shape == (8, 32, 128)
    assert out.shape == (8, 8, 128) and out.dtype == np.float32
    inputs = m._score_inputs(8, 1024, "per-row")
    last = inputs.key_bytes[:, -1].view(ml_dtypes.float8_e4m3fn)
    np.testing.assert_array_equal(new_values[:, 0], last.astype(ml_dtypes.bfloat16))
    np.testing.assert_array_equal(
        new_values[:, 1].astype(np.float32), np.repeat(inputs.exponent[:, -1:], 128, 1)
    )
    assert not new_values[:, 2].astype(np.float32).any()
    args = tool_main.parser().parse_args(
        ["bench", "dsa-micro", "--cases", "score", "--score-pages", "shared,decode",
         "--layout", "select", "--block-tokens", "512"]
    )  # fmt: skip
    assert [s.variant for s in m._build_score(8, 1024, args)] == [
        "bf16q_b512_vsel",
        "fp8q_b512_vsel",
        "bf16q_b512_vsel_dec",
    ]
    for variants in (["fp8q"], ["bf16q", "fp8q"]):
        args.variant = variants
        with pytest.raises(
            ValueError, match="decode times the decode scorer: --variant"
        ):
            m._build_score(8, 1024, args)


def test_score_knobs_must_be_positive_and_fit_a_block(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    for flag in ("--dot-tokens", "--dot-groups", "--prefetch"):
        for value in ("0", "-128", "128,0"):
            with pytest.raises(SystemExit):
                parse(["bench", "dsa-micro", "--cases", "score", flag, value])
    args = parse(["bench", "dsa-micro", "--cases", "score", "--dot-tokens", "384"])
    with pytest.raises(ValueError, match=r"\[384\]"):
        dsa_micro.score_plans(8, 2048, args)
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--block-tokens", "256",
         "--dot-tokens", "128,512"]
    )  # fmt: skip
    with pytest.raises(ValueError, match=r"\[512\]"):
        dsa_micro.score_plans(8, 2048, args)
    # Set directly (not parsed), a zero or negative knob is refused before any
    # division by it.
    for knob in ("dot_tokens", "dot_groups", "prefetch"):
        for value in (0, -128):
            args = parse(["bench", "dsa-micro", "--cases", "score"])
            setattr(args, knob, [value])
            with pytest.raises(ValueError, match="positive"):
                dsa_micro.score_plans(8, 2048, args)


def test_score_select_head_order():
    import numpy as np

    m = dsa_micro
    for rows in (1, 8, 16, 64):
        np.testing.assert_array_equal(
            m.score_query_rows(rows, "select"), m.score_query_rows(rows)
        )
    # Below eight rows tile t holds heads g t .. g t + g - 1 of every row, row
    # r on sublanes r g .. r g + g - 1 (g = 8 / R).
    row, head = np.meshgrid(np.arange(2), np.arange(32), indexing="ij")
    np.testing.assert_array_equal(
        m.score_query_rows(2, "select"), 8 * (head // 4) + 4 * row + head % 4
    )
    row, head = np.meshgrid(np.arange(4), np.arange(32), indexing="ij")
    np.testing.assert_array_equal(
        m.score_query_rows(4, "select"), 8 * (head // 2) + 2 * row + head % 2
    )
    for rows in (1, 2, 4, 8, 16):
        for layout in ("padded", "select"):
            index = m.score_query_rows(rows, layout).ravel()
            assert sorted(index.tolist()) == list(range(32 * rows))


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_head_sum_model_select_layout_equals_the_reference(rows):
    import numpy as np

    m = dsa_micro
    rng = np.random.default_rng(rows)
    tokens = 256
    q = rng.standard_normal((rows, 32, 128)).astype(np.float32)
    w = rng.standard_normal((rows, 32)).astype(np.float32)
    keys = rng.standard_normal((tokens, 128)).astype(np.float32)

    def model(layout):
        index = m.score_query_rows(rows, layout)
        q_rows = np.zeros((32 * rows, 128), np.float32)
        w_rows = np.zeros(32 * rows, np.float32)
        q_rows[index], w_rows[index] = q, w
        return m.head_sum_model(q_rows @ keys.T, w_rows, rows, layout)

    logits = np.einsum("rhd,td->rht", q, keys).astype(np.float32)
    wanted = (np.maximum(logits, 0).astype(np.float64) * w[..., None]).sum(axis=1)
    got = model("select")
    parts = 8 // rows if rows < 8 else 1
    assert got.shape == (m.padded_rows(rows), tokens)
    for r in range(rows):
        for j in range(parts):
            error = np.abs(got[r * parts + j] - wanted[r]).max()
            assert error <= 1e-6 * np.abs(wanted[r]).max(), (r, j, error)
    if rows == 1:
        # Every sublane holds the row's total (each summed in its own order).
        padded = model("padded")
        assert not padded[1:].any()
        error = np.abs(got - padded[0]).max()
        assert error <= 1e-6 * np.abs(padded[0]).max()


def test_signed_keys_order_and_invert():
    import numpy as np

    m = dsa_micro
    rng = np.random.default_rng(0)
    tiny = np.float32(1e-45)
    special = np.array(
        [0.0, -0.0, tiny, -tiny, 1e-38, -1e-38, 3e38, -3e38, np.inf, -np.inf],
        np.float32,
    )
    random = rng.standard_normal(4096) * 10.0 ** rng.integers(-30, 30, 4096)
    scores = np.concatenate([random.astype(np.float32), special])
    keys = m.signed_keys(scores)
    assert keys.dtype == np.int32
    # A higher score has a higher key (the map is one to one on the bits).
    assert (np.diff(scores[np.argsort(keys, kind="stable")]) >= 0).all()
    assert len(np.unique(keys)) == len(np.unique(scores.view(np.uint32)))
    np.testing.assert_array_equal(
        m.scores_of_keys(keys).view(np.uint32), scores.view(np.uint32)
    )
    np.testing.assert_array_equal(m.signed_keys(m.scores_of_keys(keys)), keys)
    # The kernel's key is the order key with its top bit flipped, as int32.
    np.testing.assert_array_equal(
        keys.view(np.uint32), m.order_keys(scores) ^ np.uint32(0x80000000)
    )
    assert (keys > m.INT32_MIN).all()
    assert m.signed_keys(np.float32([-np.inf]))[0] == np.uint32(0x807FFFFF).view(
        np.int32
    )


@pytest.mark.parametrize("rows, tokens", [(1, 1024), (2, 2048), (4, 4096)])
def test_virtual_place_matches_to_virtual(rows, tokens):
    import numpy as np

    m = dsa_micro
    values = np.random.default_rng(rows).standard_normal((rows, tokens))
    groups = m.virtual_groups(rows, tokens)
    planes = tokens // (groups * 128)
    got = np.zeros((planes, rows * groups, 128))
    for column in range(tokens // 128):
        plane, part = m.virtual_place(column, rows, tokens)
        for r in range(rows):
            got[plane, r * groups + part] = values[r, 128 * column : 128 * column + 128]
    np.testing.assert_array_equal(got, m.to_virtual(values, groups))


def _score_on_host(monkeypatch, rank=3):
    """Score setups built on the host: uploads stay host arrays, this rank is
    ``rank``."""
    monkeypatch.delenv("JOBID", raising=False)
    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(dsa_micro, "_upload_bits", lambda bits, dtype: bits)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, rank))


def test_score_plans_keep_the_default_tags(monkeypatch):
    import numpy as np

    m = dsa_micro
    _score_on_host(monkeypatch)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "score"])
    assert (args.layout, args.emit, args.score_pages) == (
        ["padded"],
        ["scores"],
        ["shared"],
    )
    plans = m.score_plans(8, 2048, args)
    # The default plans: the later fields at their defaults compare equal.
    assert plans == [m.ScorePlan(8, 2048, b, 128, 1, 1, False) for b in (256, 512)]
    setups = [m._score_setup(8, 2048, "bf16q", plan) for plan in plans]
    assert [s.variant for s in setups] == ["bf16q_b256", "bf16q_b512"]
    assert len(setups[0].operands) == 4 and setups[0].operands[3].shape == (8, 2048)
    args = parse(["bench", "dsa-micro", "--cases", "score", "--prefetch", "16"])
    plan = m.score_plans(8, 4096, args)[0]
    assert m._score_setup(8, 4096, "bf16q", plan).variant == "bf16q_b256_p16"
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--prefetch", "16", "--layout",
         "select", "--emit", "keys", "--block-tokens", "256"]
    )  # fmt: skip
    (plan,) = m.score_plans(8, 4096, args)
    assert (plan.layout, plan.emit, plan.pages) == ("select", "keys", "shared")
    got = m._score_setup(8, 4096, "bf16q", plan)
    assert got.variant == "bf16q_b256_p16_vsel_keys"
    assert {"layout", "emit", "score_pages"} <= set(got.extra)
    assert not set(got.extra) & RECORD_CONTEXT
    assert (got.extra["layout"], got.extra["emit"]) == ("select", "keys")
    # Keys add the rows' counts (SMEM); select is [planes, V, 128] int32.
    meta, queries, weights, pool, out = got.operands
    assert meta.dtype == np.int32 and meta.tolist() == [4096] * 8
    assert out.shape == (32, 8, 128) and out.dtype == np.int32
    with pytest.raises(ValueError, match="divide 8"):
        m.score_plans(3, 2048, args)
    with pytest.raises(ValueError, match="split"):
        m.score_plans(1, 512, args)
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--layout", "padded,select",
         "--emit", "scores,keys", "--score-pages", "shared,per-row"]
    )  # fmt: skip
    plans = m.score_plans(2, 2048, args)
    assert len(plans) == 2 * 2 * 2 * 2 == len(set(plans))
    tags = {m._score_setup(2, 2048, "bf16q", p).variant for p in plans}
    assert "bf16q_b512_vsel_keys_pr" in tags and "bf16q_b256_pr" in tags
    for flag, value in (
        ("--layout", "dense"),
        ("--emit", "bits"),
        ("--score-pages", "all"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])


def test_score_keys_mask_tokens_past_each_rows_context(monkeypatch):
    from dataclasses import replace

    m = dsa_micro
    _score_on_host(monkeypatch, rank=3)
    args = SimpleNamespace(
        block_tokens=[256], dot_tokens=[128], dot_groups=[1], prefetch=[1],
        writeback="slice", layout=["padded"], emit=["keys"], score_pages=["shared"],
        pages_per_step=[1], score_work=["full"],
    )  # fmt: skip
    (plan,) = m.score_plans(4, 1024, args)
    got = m._score_setup(4, 1024, "bf16q", plan, lengths=[0, 20000, 2048, 1])
    # Rank 3 of 32 holds global pages 3, 35, 67, 99 of 256 tokens.
    # SMEM tables hold at least eight entries (the kernel reads R of them).
    assert got.operands[0].tolist() == [1024, 768, 256, 0, 0, 0, 0, 0]
    full = m._score_setup(4, 1024, "bf16q", plan).operands[0].tolist()
    assert full == [1024] * 4 + [0] * 4
    # Scores take no counts.
    plan = replace(plan, emit="scores")
    assert len(m._score_setup(4, 1024, "bf16q", plan, lengths=[5]).operands) == 4


def test_score_per_row_pages_give_each_row_its_own_keys(monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    m = dsa_micro
    _score_on_host(monkeypatch)
    parse = tool_main.parser().parse_args
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--score-pages", "per-row",
         "--block-tokens", "512", "--prefetch", "4,64", "--dot-groups", "1,8"]
    )  # fmt: skip
    plans = m.score_plans(4, 2048, args)
    # The ring streams R x T / block pages: the prefetch caps there, and a
    # page's dot is one row's (no row groups).
    assert {(p.prefetch, p.slots, p.dot_groups) for p in plans} == {
        (4, 5, 1),
        (16, 16, 1),
    }
    plan = plans[0]
    got = m._score_setup(4, 2048, "bf16q", plan)
    assert got.variant == "bf16q_b512_p4_pr"
    assert (
        got.extra["score_pages"] == "per-row" and got.extra["page_order"] == "row-major"
    )
    table, queries, weights, pool, out = got.operands
    inputs = m._score_inputs(4, 2048, "per-row")
    assert inputs.key_bytes.shape == (4, 2048, 128) and inputs.reference.shape == (
        4,
        2048,
    )
    assert not np.array_equal(inputs.key_bytes[0], inputs.key_bytes[1])
    # Stream entry r * blocks + b is row r's local block b, wherever it sits.
    assert sorted(table.tolist()) == list(range(16)) and table.tolist() != list(
        range(16)
    )
    for r in range(4):
        blocks = m.pack_index_blocks(inputs.key_bytes[r], inputs.exponent[r], 512)
        for b in range(4):
            np.testing.assert_array_equal(pool[table[4 * r + b]], blocks[b])
    # Row r's heads are query rows 32 r .. 32 r + 31.
    np.testing.assert_array_equal(
        np.asarray(queries, np.float32).reshape(4, 32, 128), inputs.queries
    )
    np.testing.assert_array_equal(weights[:, 0].reshape(4, 32), inputs.weights)
    assert got.extra["index_bytes"] == pool.nbytes == 16 * 544 * 128
    # Each row's reference comes from its own keys.
    with jax.default_device(jax.devices("cpu")[0]):
        for r in (0, 3):
            keys = dsa_cache.dequantize_index_keys(
                inputs.key_bytes[r], inputs.exponent[r]
            )
            wanted = ref.scores(
                jnp.asarray(inputs.queries[r : r + 1]),
                jnp.asarray(inputs.weights[r : r + 1]),
                keys,
            )
            np.testing.assert_allclose(
                inputs.reference[r], np.asarray(wanted)[0], rtol=1e-6, atol=1e-6
            )


def _one_rank_gather(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, mine: out.__setitem__(0, mine),
    )
    monkeypatch.setattr(dsa_micro.harness, "log", lambda line: None)


def test_score_check_reads_the_select_layout_and_masked_keys(monkeypatch):
    import numpy as np

    m = dsa_micro
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 0))
    _one_rank_gather(monkeypatch)
    rows, tokens = 2, 2048
    reference = np.random.default_rng(1).standard_normal((rows, tokens))
    reference = reference.astype(np.float32)
    inputs = m.ScoreInputs(None, None, None, None, reference)
    counts = np.array([tokens, 300])
    groups = m.virtual_groups(rows, tokens)

    def verdict(got, layout, emit, counts=None):
        check = m._score_check(
            rows, inputs, 1e-5, {}, "score", layout=layout, emit=emit, counts=counts
        )
        return check((torch.from_numpy(np.ascontiguousarray(got))[None],))

    valid = np.arange(tokens)[None] < counts[:, None]
    keys = np.where(valid, m.signed_keys(reference), m.INT32_MIN).astype(np.int32)
    assert verdict(m.to_virtual(reference, groups), "select", "scores")
    assert verdict(m.to_virtual(keys, groups), "select", "keys", counts)
    # Two parts on each other's virtual rows fail.
    swapped = m.to_virtual(reference, groups)[:, [1, 0, 2, 3, 4, 5, 6, 7]]
    assert not verdict(swapped, "select", "scores")
    # A masked token holding its key fails, as does a valid key off the bound.
    leak = keys.copy()
    leak[1, 300] = m.signed_keys(reference[1, 300:301])[0]
    assert not verdict(m.to_virtual(leak, groups), "select", "keys", counts)
    off = keys.copy()
    off[0, 7] = m.signed_keys(reference[0, 7:8] + np.float32(0.5))[0]
    assert not verdict(m.to_virtual(off, groups), "select", "keys", counts)
    # A row with no tokens holds INT32_MIN only.
    empty = np.where(np.arange(rows)[:, None] == 1, m.INT32_MIN, keys)
    assert verdict(m.to_virtual(empty, groups), "select", "keys", np.array([2048, 0]))
    # Padded layout: rows past R are zero with scores, INT32_MIN with keys.
    padded = np.full((8, tokens), m.INT32_MIN, np.int32)
    padded[:rows] = keys
    assert verdict(padded, "padded", "keys", counts)
    padded[5, 0] = 0
    assert not verdict(padded, "padded", "keys", counts)


SCORE_FORMS = [
    # rows, layout, emit, pages
    (1, "select", "keys", "shared"),
    (2, "select", "keys", "shared"),
    (4, "select", "scores", "shared"),
    (2, "padded", "keys", "shared"),
    (8, "select", "keys", "shared"),
    (16, "select", "scores", "shared"),
    (1, "select", "keys", "per-row"),
    (2, "select", "keys", "per-row"),
    (4, "padded", "keys", "per-row"),
    (2, "padded", "scores", "per-row"),
    (8, "select", "scores", "per-row"),
    (16, "padded", "keys", "per-row"),
]


@pytest.mark.parametrize("rows, layout, emit, pages", SCORE_FORMS)
def test_score_kernel_interpreted_passes_its_check(
    monkeypatch, rows, layout, emit, pages
):
    """The kernel in Pallas interpret mode at T = 1024 (rank 3: contexts of
    1024, 768, 256 and no local tokens): its full write-back passes the check,
    and the timed body writes the same last plane or columns."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    m = dsa_micro
    monkeypatch.setattr(bridge, "upload", lambda value: jnp.asarray(value)[None])
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: jnp.asarray(bits)[None])
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    _one_rank_gather(monkeypatch)
    args = SimpleNamespace(
        block_tokens=[256], dot_tokens=[128], dot_groups=[8], prefetch=[2],
        writeback="slice", layout=[layout], emit=[emit], score_pages=[pages],
        pages_per_step=[1], score_work=["full"],
    )  # fmt: skip
    (plan,) = m.score_plans(rows, 1024, args)
    setup = m._score_setup(rows, 1024, "bf16q", plan, lengths=[0, 20000, 2048, 1])
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        full = np.array(jax.block_until_ready(setup.check_body(*setup.operands, 1))[0])
        timed = np.array(jax.block_until_ready(setup.body(*setup.operands, 2))[0])
    extra = setup.extra
    assert setup.check((torch.from_numpy(full),)), extra.get("max_error")
    assert extra["max_error"] <= 1e-5
    if layout == "select":
        np.testing.assert_array_equal(timed[0, -1], full[0, -1])
    else:
        np.testing.assert_array_equal(timed[0, :, -128:], full[0, :, -128:])


def test_score_pages_per_step_flags_plans_and_tags(monkeypatch):
    m = dsa_micro
    _score_on_host(monkeypatch)
    parse = tool_main.parser().parse_args
    args = parse(["bench", "dsa-micro", "--cases", "score"])
    assert (args.pages_per_step, args.score_work) == ([1], ["full"])
    for flag, value in (
        ("--pages-per-step", "3"),
        ("--pages-per-step", "16"),
        ("--score-work", "none"),
    ):
        with pytest.raises(SystemExit):
            parse(["bench", "dsa-micro", flag, value])
    args = parse(
        ["bench", "dsa-micro", "--cases", "score", "--block-tokens", "512",
         "--prefetch", "16", "--layout", "select", "--emit", "keys",
         "--score-pages", "shared,per-row", "--pages-per-step", "1,2,4,8"]
    )  # fmt: skip
    plans = m.score_plans(64, 8192, args)
    # Shared pages ignore the knob; per-row pages take every K.
    assert [(p.pages, p.pages_per_step) for p in plans] == [
        ("shared", 1),
        ("per-row", 1),
        ("per-row", 2),
        ("per-row", 4),
        ("per-row", 8),
    ]
    # D = ceil(prefetch / K) steps in flight, (D + 1) K slots.
    assert [(p.in_flight, p.slots) for p in plans[1:]] == [
        (16, 17),
        (8, 18),
        (4, 20),
        (2, 24),
    ]
    setups = [m._score_setup(64, 8192, "bf16q", p) for p in plans]
    assert [s.variant for s in setups] == [
        "bf16q_b512_p16_vsel_keys",
        "bf16q_b512_p16_vsel_keys_pr",
        "bf16q_b512_p16_vsel_keys_pr_k2",
        "bf16q_b512_p16_vsel_keys_pr_k4",
        "bf16q_b512_p16_vsel_keys_pr_k8",
    ]
    assert [s.extra["pages_per_step"] for s in setups] == [1, 1, 2, 4, 8]
    assert [s.extra.get("page_order") for s in setups] == [
        None,
        "row-major",
        "row-major",
        "row-major",
        "block-major",
    ]
    for setup in setups:
        assert setup.extra["score_work"] == "full" and not setup.invalid
        assert not set(setup.extra) & RECORD_CONTEXT
    # Below eight rows K = 8 takes eight consecutive blocks of one row; one
    # step streams them all, so the ring holds one step.
    (plan,) = [p for p in m.score_plans(1, 4096, args) if p.pages_per_step == 8]
    assert (plan.in_flight, plan.slots) == (1, 8)
    assert m._score_setup(1, 4096, "bf16q", plan).extra["page_order"] == "row-major"
    # Row-major steps need K to divide a row's blocks; block-major K = 8 does
    # not.
    with pytest.raises(ValueError, match="pages-per-step"):
        m.score_plans(4, 2048, args)
    assert {p.pages_per_step for p in m.score_plans(8, 2048, args)} == {1, 2, 4, 8}
    # Set directly (not parsed), a K or a work off the choices is refused.
    args.pages_per_step = [3]
    with pytest.raises(ValueError, match="pages-per-step"):
        m.score_plans(8, 4096, args)
    args.pages_per_step, args.score_work = [1], ["none"]
    with pytest.raises(ValueError, match="score-work"):
        m.score_plans(8, 4096, args)


def test_score_work_arms_are_timing_only(monkeypatch):
    m = dsa_micro
    _score_on_host(monkeypatch)
    args = tool_main.parser().parse_args(
        ["bench", "dsa-micro", "--cases", "score", "--block-tokens", "512",
         "--prefetch", "16", "--layout", "select", "--emit", "keys",
         "--score-pages", "shared,per-row", "--pages-per-step", "1,8",
         "--score-work", "full,compute-only,dma-only"]
    )  # fmt: skip
    plans = m.score_plans(8, 4096, args)
    # Shared pages take the full work only.
    assert sorted((p.pages, p.pages_per_step, p.work) for p in plans) == sorted(
        [("shared", 1, "full")]
        + [
            ("per-row", k, work)
            for k in (1, 8)
            for work in ("full", "compute-only", "dma-only")
        ]
    )
    setups = {
        (p.pages_per_step, p.work): m._score_setup(8, 4096, "bf16q", p) for p in plans
    }
    assert setups[1, "compute-only"].variant == "bf16q_b512_p16_vsel_keys_pr_conly"
    assert setups[8, "dma-only"].variant == "bf16q_b512_p16_vsel_keys_pr_k8_donly"
    for (_, work), setup in setups.items():
        assert setup.extra["score_work"] == work
        timing_only = work != "full"
        assert setup.invalid == timing_only
        assert (setup.check is None) == timing_only
        assert (setup.check_body is None) == timing_only


def test_score_stream_rows_and_blocks(monkeypatch):
    import numpy as np

    m = dsa_micro
    rows, blocks = m.score_stream(2, 4, 2)
    assert rows.tolist() == [0] * 4 + [1] * 4 and blocks.tolist() == [0, 1, 2, 3] * 2
    # K = 8 from eight rows: block-major within groups of eight rows.
    rows, blocks = m.score_stream(16, 2, 8)
    assert rows.tolist() == [*range(8)] * 2 + [*range(8, 16)] * 2
    assert blocks.tolist() == ([0] * 8 + [1] * 8) * 2
    rows, blocks = m.score_stream(4, 8, 8)
    assert rows.tolist() == np.repeat(np.arange(4), 8).tolist()
    assert blocks.tolist() == [*range(8)] * 4
    # The SMEM table lists each stream entry's pool block.
    _score_on_host(monkeypatch)
    args = SimpleNamespace(
        block_tokens=[512], dot_tokens=[128], dot_groups=[1], prefetch=[16],
        writeback="slice", layout=["select"], emit=["scores"], score_pages=["per-row"],
        pages_per_step=[8], score_work=["full"],
    )  # fmt: skip
    (plan,) = m.score_plans(16, 1024, args)
    table, queries, weights, pool, out = m._score_setup(
        16, 1024, "bf16q", plan
    ).operands
    inputs = m._score_inputs(16, 1024, "per-row")
    rows, blocks = m.score_stream(16, 2, 8)
    for entry, (r, b) in enumerate(zip(rows, blocks)):
        packed = m.pack_index_blocks(inputs.key_bytes[r], inputs.exponent[r], 512)
        np.testing.assert_array_equal(pool[table[entry]], packed[b])


@pytest.mark.parametrize("sign", [1, -1])
def test_combine_rows_puts_each_tiles_total_on_its_sublane(sign):
    import numpy as np

    tiles = list(np.random.default_rng(5).standard_normal((8, 8, 128)))
    got = dsa_micro.combine_rows(
        tiles,
        np.arange(8)[:, None],
        lambda x, shift: np.roll(x, sign * shift, 0),
        np.where,
    )
    np.testing.assert_allclose(got, np.stack([t.sum(0) for t in tiles]), rtol=1e-12)


def test_score_keys_bound_covers_one_token_rows(monkeypatch):
    import numpy as np

    m = dsa_micro
    _score_on_host(monkeypatch)
    _one_rank_gather(monkeypatch)
    args = SimpleNamespace(
        block_tokens=[256], dot_tokens=[128], dot_groups=[1], prefetch=[1],
        writeback="slice", layout=["padded"], emit=["scores", "keys"],
        score_pages=["shared"], pages_per_step=[1], score_work=["full"],
    )  # fmt: skip
    plans = m.score_plans(8, 1024, args)
    setups = {p.emit: m._score_setup(8, 1024, "bf16q", p) for p in plans}
    reference = m._score_inputs(8, 1024, "shared").reference
    off = (reference * np.float32(1 + 1.5e-5)).astype(np.float32)
    # The keys' bound is looser than the scores' (their error is relative to
    # the valid tokens only).
    assert setups["keys"].check((torch.from_numpy(m.signed_keys(off)[None]),))
    assert not setups["scores"].check((torch.from_numpy(off[None]),))


STEP_FORMS = [
    # rows, tokens, layout, emit, pages per step, prefetch
    (1, 1024, "select", "keys", 4, 2),
    (2, 1024, "select", "keys", 2, 2),
    (4, 1024, "select", "scores", 2, 3),
    (4, 1024, "padded", "scores", 4, 2),
    (2, 2048, "padded", "keys", 8, 2),
    (8, 1024, "select", "keys", 2, 2),
    (8, 1024, "select", "keys", 8, 2),
    (8, 1024, "padded", "keys", 8, 2),
    (16, 1024, "select", "scores", 8, 2),
    (16, 1024, "padded", "scores", 8, 3),
]


def _interpret_score(monkeypatch, rows, tokens, layout, emit, k, prefetch, work):
    import jax.numpy as jnp

    m = dsa_micro
    monkeypatch.setattr(bridge, "upload", lambda value: jnp.asarray(value)[None])
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: jnp.asarray(bits)[None])
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    _one_rank_gather(monkeypatch)
    args = SimpleNamespace(
        block_tokens=[256], dot_tokens=[128], dot_groups=[1], prefetch=[prefetch],
        writeback="slice", layout=[layout], emit=[emit], score_pages=["per-row"],
        pages_per_step=[k], score_work=[work],
    )  # fmt: skip
    (plan,) = m.score_plans(rows, tokens, args)
    setup = m._score_setup(rows, tokens, "bf16q", plan, lengths=[0, 20000, 2048, 1])
    assert (setup.extra["pages_per_step"], setup.extra["score_work"]) == (k, work)
    return setup


@pytest.mark.parametrize("rows, tokens, layout, emit, k, prefetch", STEP_FORMS)
def test_score_steps_interpreted_pass_their_check(
    monkeypatch, rows, tokens, layout, emit, k, prefetch
):
    """Per-row pages, K pages a step, in Pallas interpret mode (rank 3's
    contexts): the full write-back passes the check against each row's own
    reference, and the timed body writes the same last plane or columns."""
    import jax
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    setup = _interpret_score(
        monkeypatch, rows, tokens, layout, emit, k, prefetch, "full"
    )
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        full = np.array(jax.block_until_ready(setup.check_body(*setup.operands, 1))[0])
        timed = np.array(jax.block_until_ready(setup.body(*setup.operands, 2))[0])
    assert setup.check((torch.from_numpy(full),)), setup.extra.get("max_error")
    assert setup.extra["max_error"] <= 1e-5
    if layout == "select":
        np.testing.assert_array_equal(timed[0, -1], full[0, -1])
    else:
        np.testing.assert_array_equal(timed[0, :, -128:], full[0, :, -128:])


@pytest.mark.parametrize("work", ["compute-only", "dma-only"])
@pytest.mark.parametrize("rows, k", [(8, 8), (2, 1)])
def test_score_work_arms_interpreted_run(monkeypatch, work, rows, k):
    """The timing-only arms run to the end (every DMA they start is waited)."""
    import jax
    from jax.experimental.pallas import tpu as pltpu

    setup = _interpret_score(monkeypatch, rows, 1024, "select", "keys", k, 2, work)
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        jax.block_until_ready(setup.body(*setup.operands, 2))


@pytest.mark.parametrize("rows, rank", [(2, 31), (8, 3), (16, 31)])
def test_score_decode_form_is_the_decode_scorer(monkeypatch, rows, rank):
    """--score-pages decode runs dsa_decode.IndexScorer itself: in Pallas
    interpret mode its full write-back passes the check, equals
    dsa_decode.score_rows on the same inputs over each row's bound, and the
    timed body writes the same last plane."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    m = dsa_micro
    monkeypatch.setattr(bridge, "upload", lambda value: jnp.asarray(value)[None])
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: jnp.asarray(bits)[None])
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, rank))
    _one_rank_gather(monkeypatch)
    args = SimpleNamespace(
        block_tokens=[512], dot_tokens=[128], dot_groups=[1], prefetch=[16],
        writeback="slice", layout=["select"], emit=["scores"], score_pages=["decode"],
        pages_per_step=[1], score_work=["full"],
    )  # fmt: skip
    (plan,) = m.score_plans(rows, 1024, args)
    setup = m._score_setup(rows, 1024, "bf16q", plan)
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        full = np.array(jax.block_until_ready(setup.check_body(*setup.operands, 1))[0])
        timed = np.array(jax.block_until_ready(setup.body(*setup.operands, 2))[0])
        want = np.asarray(m.decode_reference_scores(setup, rank))
    assert setup.check((torch.from_numpy(full),)), setup.extra.get("max_error")
    np.testing.assert_array_equal(timed[0, -1], full[0, -1])  # [rank, planes, V, 128]
    length = int(dsa_decode.local_length(m.RANKS * 1024 - 1, rank, 512, m.RANKS))
    got = dsa_decode.scores_by_row(full[0], rows)[:, :length]
    np.testing.assert_array_equal(got, dsa_decode.scores_by_row(want, rows)[:, :length])


def _interpret_on_one_rank(monkeypatch, rank=0):
    """Setups built for Pallas interpret mode on the host: an upload becomes
    this rank's ``[1, ...]`` block (pool bits viewed as their dtype)."""
    import jax.numpy as jnp

    dtypes = {torch.bfloat16: jnp.bfloat16, torch.float8_e4m3fn: jnp.float8_e4m3fn}
    monkeypatch.setattr(bridge, "upload", lambda value: jnp.asarray(value)[None])
    monkeypatch.setattr(
        dsa_micro,
        "_upload_bits",
        lambda bits, dtype: jnp.asarray(bits).view(dtypes[dtype])[None],
    )
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, rank))
    monkeypatch.setattr(dsa_micro.harness, "log", lambda line: None)


def _interpreted(body, operands, iterations=1, outputs=1):
    """``body`` in Pallas interpret mode on one host device, as the only rank of
    a one-device ``tp`` mesh (kernels that read their rank can trace)."""
    import jax
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices("cpu")[:1]), (dsa_micro.AXIS,))
    call = jax.shard_map(
        lambda *ops: body(*ops, iterations),
        mesh=mesh,
        in_specs=(P(dsa_micro.AXIS),) * len(operands),
        out_specs=(P(dsa_micro.AXIS),) * outputs,
        check_vma=False,
    )
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    with pltpu.force_tpu_interpret_mode(params):
        return [np.asarray(o) for o in jax.block_until_ready(call(*operands))]


def test_smem_tables_have_eight_entries_at_least():
    import numpy as np

    m = dsa_micro
    assert m.smem_table(np.int32([5])).tolist() == [5, 0, 0, 0, 0, 0, 0, 0]
    meta = m.smem_table(np.int32([[3], [4]]))
    assert meta.dtype == np.int32 and meta.tolist() == [[3] + [0] * 7, [4] + [0] * 7]
    wide = np.arange(9, dtype=np.int32)
    np.testing.assert_array_equal(m.smem_table(wide), wide)


def test_select_meta_pads_without_changing_the_select(monkeypatch):
    """One row: the padded ``meta [2, 8]`` and the unpadded ``[2, 1]`` give the same
    mask (the local exact top-k without an exchange)."""
    import numpy as np

    m = dsa_micro
    _interpret_on_one_rank(monkeypatch)
    rows, tokens = 1, 4096
    args = SimpleNamespace(page_tokens=256, scores="normal", lengths=[])
    setup = m._select_setup(rows, tokens, "radix8", "none", True, "packed", args)
    meta, scores, mask = setup.operands
    assert meta.shape == (1, 2, 8) and not np.asarray(meta[0, :, rows:]).any()
    padded = _interpreted(setup.body, setup.operands)[0]
    plain = _interpreted(setup.body, (meta[..., :rows], scores, mask))[0]
    np.testing.assert_array_equal(padded, plain)
    values = m._select_scores(rows, tokens, 0, "normal")[0]
    wanted = np.zeros(tokens, bool)
    wanted[np.lexsort((np.arange(tokens), -values))[: m.SELECT_K]] = True
    np.testing.assert_array_equal(m.unpack_mask(padded[0], rows, tokens)[0], wanted)


def test_gather_smem_tables_pad_without_changing_the_gather(monkeypatch):
    """R = 1, FP8 T = 1024 pages: the block table (2 entries) and the DMA list
    padded to 8 trace the same kernel as the unpadded tables but for their shapes
    (Pallas interpret mode cannot run the gather's bitcast views)."""
    import re

    import jax
    import numpy as np

    m = dsa_micro
    _interpret_on_one_rank(monkeypatch)
    monkeypatch.setattr(m, "GATHER_POOLS", {})
    args = SimpleNamespace(rows=[1], verify="on")
    setup = m._gather_setup(1, 1024, "page", "uniform", 16, "fp8", 64, 1, args)
    table, codes, pool, carry = setup.operands
    entries = (m.GATHER_POOLS["fp8", 1024, 1].table.size, setup.extra["dmas"])
    assert entries[0] == 2 and entries[1] <= 2
    assert table.shape == (1, 8) and codes.shape == (1, 8)
    assert not np.asarray(table[0, 2:]).any() and not np.asarray(codes[0, 2:]).any()
    unpadded = (table[:, : entries[0]], codes[:, : entries[1]], pool, carry)
    # Only the two-dimensional operands are SMEM tables.
    two_d = (r"i32\[1,\d+\]", r"\(Blocked\(block_size=1\), Blocked\(block_size=\d+\)\)")
    two_d += (r"shape=\(1, \d+\), dtype=int32",)
    for body in (setup.body, setup.check_body):
        padded, plain = (
            str(jax.make_jaxpr(lambda *ops, body=body: body(*ops, 1))(*operands))
            for operands in (setup.operands, unpadded)
        )
        assert padded != plain
        for pattern in two_d:
            padded, plain = (re.sub(pattern, "SMEM", text) for text in (padded, plain))
        assert padded == plain


@pytest.mark.parametrize("variant, rows", [("row", 1), ("none", 1), ("tile", 2)])
def test_fold_smem_tables_pad_without_changing_the_fold(monkeypatch, variant, rows):
    """FP8 fold tables padded to 8 entries (``copy_tokens`` has one without the
    copy compaction; ``none`` has one 6-entry step and no masks) fold the same
    rows as the same tables unpadded."""
    import jax.numpy as jnp
    import numpy as np

    m = dsa_micro
    _interpret_on_one_rank(monkeypatch)
    monkeypatch.setattr(m, "_upload_bits", lambda bits, dtype: jnp.asarray(bits)[None])
    _one_rank_gather(monkeypatch)
    stages = []
    stage_of = m.fold_stage
    monkeypatch.setattr(
        m, "fold_stage", lambda *a, **k: stages.append(stage_of(*a, **k)) or stages[-1]
    )
    setup = m._fold_setup(
        rows, 2048, variant, "uniform", 64, "fp8", 256, "none", "on",
        SimpleNamespace(rows=[rows], row_mix="off"),
    )  # fmt: skip
    (stage,) = stages
    lengths = [len(stage.meta), len(stage.masks), len(stage.copy_tokens)]
    assert lengths[2] == 1 and min(lengths) < 8
    tables = setup.operands[:3]
    assert [t.shape[-1] for t in tables] == [max(n, 8) for n in lengths]
    padded = _interpreted(setup.check_body, setup.operands)[0]
    unpadded = (*(t[:, :n] for t, n in zip(tables, lengths)), *setup.operands[3:])
    np.testing.assert_array_equal(padded, _interpreted(setup.check_body, unpadded)[0])
    assert setup.check((torch.from_numpy(padded),))


def test_tie_round_needed():
    import numpy as np

    values = np.float32([[5, 4, 4, 3], [5, 4, 3, 3], [5, 4, 4, 4]])
    got = dsa_micro.tie_round_needed(values, [4, 4, 2], k=2)
    # The 2nd value ties the 3rd; it does not; the row is no longer than k.
    assert got.tolist() == [True, False, False]


def test_exact_tie_scores_fill_or_overflow_the_quota():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tools.megakernel import dsa_reference as ref

    m = dsa_micro
    rows, tokens, page = 4, 256, 256
    values = m.exact_tie_scores(rows, tokens, page)
    lengths = [32 * tokens] * rows
    assert values.shape == (rows, 32 * tokens)
    assert m.tie_round_needed(values, lengths).tolist() == [False, True, False, True]
    with jax.default_device(jax.devices("cpu")[0]):
        chosen = np.asarray(
            ref.select_topk(jnp.asarray(values), jnp.asarray(lengths), 2048)
        )
    assert ((chosen >= 0).sum(axis=1) == 2048).all()
    for r, (ties, extra) in enumerate(m.EXACT_TIES):
        tied = np.flatnonzero(values[r] == 1.0)
        assert len(tied) == ties + extra and (values[r] > 1).sum() == 2048 - ties
        assert len(set((tied // page) % 32)) >= 4  # the ties' ranks
        rest = values[r][values[r] < 1]
        assert len(np.unique(rest)) == len(rest)
    with pytest.raises(ValueError, match="exact tie"):
        m.exact_tie_scores(1, 64, 64)


@pytest.mark.parametrize("histogram", ["cumulative", "packed"])
@pytest.mark.parametrize("bits", [4, 8])
def test_radix_select_model_with_tie_skip_equals_the_reference(bits, histogram):
    import numpy as np

    m = dsa_micro
    rows, tokens, page = 4, 256, 256
    index = m.global_token_index(np.arange(tokens)[None], np.arange(32)[:, None], page)
    needs = set()
    for data in ("exact", "ties"):
        lengths = m._select_lengths(rows, tokens, [0, 0, 5000, 2048])
        values, wanted, length = m._select_reference(rows, tokens, data, lengths, page)
        need = m.tie_round_needed(values, length)
        for r in range(rows):
            keys = m.order_keys(values[r][index])
            counts = m.local_counts(length[r], tokens, page)
            plain = m.radix_select_model(keys, counts, 2048, bits, page, histogram)
            got, needed = m.radix_select_model(
                keys, counts, 2048, bits, page, histogram, tie_skip=True
            )
            np.testing.assert_array_equal(plain, wanted[r][index])
            np.testing.assert_array_equal(got, plain)
            assert needed == need[r]
            needs.add((data, needed))
    # Both data sets have rows that skip the round and rows that run it.
    assert needs == {(d, n) for d in ("exact", "ties") for n in (False, True)}


def test_row_packet_round_trips():
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    m = dsa_micro
    rng = np.random.default_rng(0)
    for rows, bins in ((1, 256), (2, 256), (4, 256), (1, 16), (4, 16)):
        counts = rng.integers(0, 2**20, (rows, bins))
        packet = m.row_packet(counts, rows, bins)
        assert packet.shape == (8, 128) and not packet.reshape(-1)[rows * bins :].any()
        np.testing.assert_array_equal(m.row_unpacket(packet, rows, bins), counts)
        # The kernel's pieces put each row's 128-lane chunks where the packet has them.
        for r, c, s, offset, width in dsa_select._row_pieces(rows, bins):
            np.testing.assert_array_equal(
                packet[s, offset : offset + width], counts[r, c * 128 : c * 128 + width]
            )
    with pytest.raises(ValueError, match="packet"):
        m.row_packet(np.zeros((8, 256)), 8, 256)
    with pytest.raises(ValueError, match="packet"):
        dsa_select._row_pieces(8, 256)


def _select_on_host(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    monkeypatch.setattr(bridge, "upload", lambda value: value)
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    monkeypatch.setattr(dsa_micro, "_host_order_checked", lambda: None)


def test_select_default_flags_keep_the_default_kernels(monkeypatch):
    m = dsa_micro
    _select_on_host(monkeypatch)
    built = []

    def kernel(rows, tokens, *_, **knobs):
        built.append(knobs)
        groups = m.virtual_groups(rows, tokens)
        planes = tokens // (groups * 128)
        return None, [], -(-planes // 4), rows * groups, planes

    monkeypatch.setattr(m, "_select_kernel", kernel)
    parse = tool_main.parser().parse_args
    base = ["bench", "dsa-micro", "--cases", "select", "--variant", "radix4"]
    base += ["--exchange", "flat", "--histogram", "packed", "--page-tokens", "512"]
    args = parse(base)
    assert (args.tie_skip, args.row_sum, args.scores) == (["off"], ["after"], "normal")
    (got,) = m._build_select(8, 2048, args)
    # The default tag, operands and specs; the extra adds the tie-skip and
    # row-sum settings.
    assert got.variant == "radix4_flat_packed_p512" and got.check is not None
    assert built[-1] == {"tie_skip": False, "row_sum_order": "after"}
    assert got.extra == {
        "exchange": "flat",
        "histogram": "packed",
        "ties": "on",
        "select_page_tokens": 512,
        "tie_skip": "off",
        "row_sum": "after",
    }
    assert len(got.operands) == len(got.in_specs) == 3 and len(got.out_specs) == 1
    plain, skip = m._build_select(8, 2048, parse(base + ["--tie-skip", "off,on"]))
    assert plain.variant == got.variant and skip.variant == got.variant + "_a"
    assert built[-1] == {"tie_skip": True, "row_sum_order": "after"}
    assert len(skip.operands) == len(skip.in_specs) == 4 and len(skip.out_specs) == 2
    assert skip.operands[3].shape == (8, 128) and skip.extra["tie_skip"] == "on"
    assert not (set(skip.extra) | {"tie_rows", "tie_round_ran"}) & RECORD_CONTEXT
    args = parse(base + ["--row-sum", "after,before"])
    one = m._build_select(1, 2048, args)
    assert [s.variant for s in one] == [got.variant, got.variant + "_e"]
    assert one[1].extra["row_sum"] == "before" and len(one[1].operands) == 3
    # From eight rows a row is one virtual row: before is the default setup.
    (eight,) = m._build_select(8, 2048, args)
    assert eight.variant == got.variant and eight.extra["row_sum"] == "after"
    assert built[-1] == {"tie_skip": False, "row_sum_order": "after"}
    with pytest.raises(ValueError, match="tie round"):
        m._build_select(8, 2048, parse(base + ["--tie-skip", "on", "--ties", "off"]))
    for data in ("exact", "indexer"):
        assert parse(base + ["--scores", data]).scores == data
    for flag, value in (
        ("--scores", "ties,exact"),
        ("--tie-skip", "maybe"),
        ("--row-sum", "during"),
    ):
        with pytest.raises(SystemExit):
            parse(base + [flag, value])
    # Setups built directly (a namespace without the later flags) keep the defaults.
    namespace = SimpleNamespace(page_tokens=512, scores="normal", lengths=[])
    direct = m._select_setup(8, 2048, "radix4", "flat", True, "packed", namespace)
    assert direct.variant == got.variant and direct.extra == got.extra


def _select_jaxpr(setup):
    """The traced body of a select setup built on the host (its kernel jaxpr
    included), on an abstract 32-rank mesh."""
    import jax
    from jax.sharding import AbstractMesh, NamedSharding

    mesh = AbstractMesh((32,), (dsa_micro.AXIS,))
    specs = [
        jax.ShapeDtypeStruct(
            (32, *o.shape[1:]), o.dtype, sharding=NamedSharding(mesh, P(dsa_micro.AXIS))
        )
        for o in setup.operands
    ]
    with jax.sharding.use_abstract_mesh(mesh):
        body = jax.shard_map(
            lambda *ops: setup.body(*ops, 2),
            mesh=mesh,
            in_specs=setup.in_specs,
            out_specs=setup.out_specs,
            check_vma=False,
        )
        return str(jax.make_jaxpr(body)(*specs))


def test_select_row_sum_before_changes_the_kernel_below_eight_rows(monkeypatch):
    import numpy as np

    m = dsa_micro
    monkeypatch.delenv("JOBID", raising=False)
    monkeypatch.setattr(bridge, "upload", lambda value: np.asarray(value)[None])
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, size: (None, 3))
    args = SimpleNamespace(page_tokens=256, scores="normal", lengths=[])

    def traced(rows, row_sum="after", tie_skip=False):
        setup = m._select_setup(
            rows, 1024, "radix8", "flat", True, "packed", args, tie_skip, row_sum
        )
        return _select_jaxpr(setup)

    one = traced(1)
    assert traced(1, "before") != one and traced(1, tie_skip=True) != one
    eight = traced(8)
    # The kernel itself takes before at eight rows as after.
    kernel_of = m._select_kernel
    with monkeypatch.context() as patch:
        patch.setattr(
            m,
            "_select_kernel",
            lambda *a, **k: kernel_of(*a, **{**k, "row_sum_order": "before"}),
        )
        assert traced(8) == eight


def test_select_exact_scores_follow_the_page_size(monkeypatch):
    import numpy as np

    m = dsa_micro
    rows, tokens = 4, 1024
    lengths = m._select_lengths(rows, tokens, [])
    for page in (256, 512):
        exact = m.exact_tie_scores(rows, tokens, page)
        values, wanted, _ = m._select_reference(rows, tokens, "exact", lengths, page)
        np.testing.assert_array_equal(values, exact)
        index = m.global_token_index(np.arange(tokens), 3, page)
        np.testing.assert_array_equal(
            m._select_scores(rows, tokens, 3, "exact", page), exact[:, index]
        )
        assert m.tie_round_needed(values, lengths).tolist() == [False, True] * 2
        assert (wanted.sum(axis=1) == 2048).all()
    assert not np.array_equal(
        m.exact_tie_scores(rows, tokens, 256), m.exact_tie_scores(rows, tokens, 512)
    )


def test_select_tie_stats_verdict(monkeypatch):
    import numpy as np

    m = dsa_micro
    rows, tokens, page = 4, 256, 256
    lengths = m._select_lengths(rows, tokens, [0, 0, 0, 2000])
    values, wanted, length = m._select_reference(rows, tokens, "exact", lengths, page)
    need = m.tie_round_needed(values, length)
    assert need.tolist() == [False, True, False, False]
    at = [2048, 2049, 2048, 0]  # count(>= tau); a row of at most k: any
    groups = 2
    stats = np.zeros((rows * groups, 128), np.int64)
    stats[:, 0] = np.repeat(need, groups)
    stats[:, 1] = 1
    stats[:, 2] = np.repeat(at, groups)
    passed, tie_rows, ran, line = m.select_tie_stats(
        stats, values, wanted, length, rows
    )
    assert passed and (tie_rows, ran) == (1, True), line
    # Rows (both virtual rows) with a wrong need, round or count, and one
    # virtual row unlike its row's other one.
    for lane, part, value in (
        (0, slice(2, 4), 0),
        (0, slice(4, 6), 1),
        (1, slice(None), 0),
        (2, slice(0, 2), 2049),
        (2, slice(2, 4), 2048),
        (2, slice(5, 6), 2047),
    ):
        wrong = stats.copy()
        wrong[part, lane] = value
        assert not m.select_tie_stats(wrong, values, wanted, length, rows)[0]
    # The count of a row of at most k tokens is not judged.
    stats[6:, 2] = 2**31 - 1
    assert m.select_tie_stats(stats, values, wanted, length, rows)[0]


def _local_tie_scores(rows, tokens, shift):
    """Local scores ``[R, T]`` whose local k-th value ties as
    ``EXACT_TIES[(r + shift) % 4]`` says (no exchange: the local top-k)."""
    import numpy as np

    rng = np.random.default_rng([5, rows, tokens, shift])
    out = np.empty((rows, tokens), np.float32)
    for r in range(rows):
        ties, extra = dsa_micro.EXACT_TIES[(r + shift) % 4]
        order = rng.permutation(tokens)
        above = dsa_micro.SELECT_K - ties
        out[r, order[:above]] = 2 + rng.permutation(above) / tokens
        out[r, order[above : above + ties + extra]] = 1.0
        rest = order[above + ties + extra :]
        out[r, rest] = -rng.permutation(len(rest)) / tokens
    return out


SELECT_FORMS = [
    # rows, variant, histogram, tie skip, row sum, tie pattern shift
    (1, "radix8", "packed", True, "after", 0),
    (1, "radix8", "packed", True, "before", 1),
    (2, "radix4", "cumulative", True, "before", 0),
    (4, "radix8", "cumulative", True, "before", 1),
    (4, "radix4", "packed", False, "before", 0),
    (8, "radix4", "packed", True, "after", 2),
]


@pytest.mark.parametrize(
    "rows, variant, histogram, tie_skip, row_sum, shift", SELECT_FORMS
)
def test_select_kernel_interpreted_skips_the_tie_round_exactly(
    monkeypatch, rows, variant, histogram, tie_skip, row_sum, shift
):
    """The kernel without an exchange in Pallas interpret mode at T = 4096 (the
    exact local top-k, local ties in local order): with the skip it selects
    the same tokens whether a row's ties fill the quota or overflow it, and
    its stats give each row's need, whether the round ran and count(>= tau);
    the counts summed before the (absent) exchange give the same selection.
    Two chained calls."""
    import numpy as np

    m = dsa_micro
    _interpret_on_one_rank(monkeypatch)
    tokens = 4096
    local = _local_tie_scores(rows, tokens, shift)
    monkeypatch.setattr(m, "_select_scores", lambda *a, **k: local)
    args = SimpleNamespace(page_tokens=256, scores="normal", lengths=[])
    setup = m._select_setup(
        rows, tokens, variant, "none", True, histogram, args, tie_skip, row_sum
    )
    outputs = _interpreted(setup.body, setup.operands, 2, len(setup.out_specs))
    got = m.unpack_mask(outputs[0][0], rows, tokens)
    for r in range(rows):
        wanted = np.zeros(tokens, bool)
        wanted[np.lexsort((np.arange(tokens), -local[r]))[: m.SELECT_K]] = True
        np.testing.assert_array_equal(got[r], wanted, err_msg=f"row {r}")
    if tie_skip:
        need = m.tie_round_needed(local, [tokens] * rows)
        groups = m.virtual_groups(rows, tokens)
        stats = outputs[1][0]
        np.testing.assert_array_equal(stats[:, 0], np.repeat(need, groups))
        assert (stats[:, 1] == need.any()).all()
        at = [(row >= np.sort(row)[::-1][m.SELECT_K - 1]).sum() for row in local]
        np.testing.assert_array_equal(stats[:, 2], np.repeat(at, groups))
        assert not stats[:, 3:].any()


def test_select_indexer_scores_come_from_one_score_call(monkeypatch):
    import numpy as np

    m = dsa_micro
    _select_on_host(monkeypatch)
    monkeypatch.setattr(m, "INDEXER_SCORES", {})
    monkeypatch.setattr(m.harness, "log", lambda line: None)
    plans, calls = [], []
    inputs = SimpleNamespace(reference=np.full((2, 1024), 3.0, np.float32))
    monkeypatch.setattr(m, "_score_inputs", lambda rows, tokens: inputs)

    def score_setup(rows, tokens, variant, plan):
        plans.append((rows, tokens, variant, plan))
        return m.Setup("body", ("operands",), ("in",), ("out",), variant="b512_full")

    def call(name, body, operands, in_specs, out_specs, iterations):
        calls.append((name, body, iterations))
        return (torch.full((1, 8, 1024), 3.0),)

    monkeypatch.setattr(m, "_score_setup", score_setup)
    monkeypatch.setattr(m, "call_body", call)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 32)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, mine: out.__setitem__(
            slice(None), [(r, mine[1] + r, mine[2]) for r in reversed(range(32))]
        ),
    )
    got = m._indexer_scores(2, 1024, 512)
    assert plans == [(2, 1024, "bf16q", m.ScorePlan(2, 1024, 512, 128, 1, 16, True))]
    assert [(c[1], c[2]) for c in calls] == [("body", 1)]
    assert got.shape == (32, 2, 1024) and (got[5] == 8).all()
    np.testing.assert_array_equal(m._select_scores(2, 1024, 5, "indexer", 512), got[5])
    assert m._indexer_scores(2, 1024, 512) is got and len(calls) == 1
    with pytest.raises(ValueError, match="index blocks"):
        m._indexer_scores(2, 1024, 384)
    with pytest.raises(ValueError, match="exceed"):
        m._indexer_scores(64, 8192, 512)
    # Scores off the reference stop the point on every rank.
    monkeypatch.setattr(m, "INDEXER_SCORES", {})
    inputs.reference = np.zeros((2, 1024), np.float32)
    with pytest.raises(RuntimeError, match="reference"):
        m._indexer_scores(2, 1024, 512)


def test_merge_candidates_are_every_token_or_the_local_top_k():
    import numpy as np

    m = dsa_micro
    page, rank, k = 256, 5, m.SELECT_K
    # Every valid token when the rank holds at most k, padded with key 0, id -1.
    values = m._select_scores(2, 1024, rank, "normal", page)
    lengths = m._select_lengths(2, 1024, [0, 20000])
    keys, ids = m.merge_candidates_model(values, lengths, rank, page_tokens=page)
    assert keys.shape == ids.shape == (2, 1024)
    assert (keys.dtype, ids.dtype) == (np.uint32, np.int32)
    index = m.global_token_index(np.arange(1024), rank, page)
    np.testing.assert_array_equal(ids[0], index)
    np.testing.assert_array_equal(keys[0], m.order_keys(values[0]))
    count = m.local_counts(20000, 1024, page)[rank]
    assert 0 < count < 1024
    np.testing.assert_array_equal(ids[1, :count], index[:count])
    np.testing.assert_array_equal(keys[1, :count], m.order_keys(values[1, :count]))
    assert (ids[1, count:] == -1).all() and (keys[1, count:] == 0).all()
    # Above k local tokens: the local top-k, in local order.
    values = m._select_scores(2, 4096, rank, "normal", page)
    lengths = m._select_lengths(2, 4096, [0, 20000])
    keys, ids = m.merge_candidates_model(values, lengths, rank, page_tokens=page)
    assert keys.shape == ids.shape == (2, k)
    index = m.global_token_index(np.arange(4096), rank, page)
    top = np.sort(np.argsort(-values[0])[:k])
    np.testing.assert_array_equal(ids[0], index[top])
    np.testing.assert_array_equal(keys[0], m.order_keys(values[0, top]))
    count = m.local_counts(20000, 4096, page)[rank]
    np.testing.assert_array_equal(ids[1, :count], index[:count])
    assert (ids[1, count:] == -1).all()
    # Ties at the local threshold go to the lowest local (so global) indices.
    for local in (_local_tie_scores(4, 4096, 1), m._select_scores(4, 4096, 3, "exact")):
        keys, ids = m.merge_candidates_model(local, [32 * 4096] * 4, 3)
        index = m.global_token_index(np.arange(4096), 3, 256)
        for r in range(4):
            top = np.sort(np.lexsort((np.arange(4096), -local[r]))[:k])
            np.testing.assert_array_equal(ids[r], index[top], err_msg=f"row {r}")


@pytest.mark.parametrize("data", ["normal", "ties", "exact"])
@pytest.mark.parametrize("rows, tokens", [(1, 1024), (1, 4096), (4, 1024), (4, 4096)])
def test_merge_select_model_equals_select_topk(data, rows, tokens):
    import numpy as np

    m = dsa_micro
    page = 256
    lengths = m._select_lengths(rows, tokens, [0, 0, 20000, 2048])
    gathered = [
        m.merge_candidates_model(
            m._select_scores(rows, tokens, rank, data, page),
            lengths,
            rank,
            page_tokens=page,
        )
        for rank in range(32)
    ]
    keys = np.stack([g[0] for g in gathered])
    ids = np.stack([g[1] for g in gathered])
    got = m.merge_select_model(keys, ids, lengths, size=32 * tokens)
    _, wanted, _ = m._select_reference(rows, tokens, data, lengths, page)
    assert got.dtype == np.int8 and got.shape == wanted.shape
    np.testing.assert_array_equal(got != 0, wanted)
    if tokens <= m.SELECT_K:  # every token a candidate: the default size
        assert m.merge_select_model(keys, ids, lengths).shape == wanted.shape


def test_select_merge_setups(monkeypatch):
    m = dsa_micro
    _select_on_host(monkeypatch)
    parse = tool_main.parser().parse_args
    base = ["bench", "dsa-micro", "--cases", "select", "--variant", "radix4"]
    base += ["--exchange", "merge", "--histogram", "packed", "--page-tokens", "512"]
    args = parse(base)
    for rows, tokens in ((1, 1024), (8, 1024), (2, 4096), (8, 32768)):
        (got,) = m._build_select(rows, tokens, args)
        width = min(tokens, m.SELECT_K)
        # `--exchange merge` marks the variant: no extra suffix.
        assert got.variant == "radix4_merge_packed_p512"
        assert got.check is not None and not got.invalid
        assert got.extra["exchange"] == "merge" and got.extra["tie_skip"] == "off"
        assert got.extra["merge_candidates"] == 32 * width
        # Received per rank per call: 31 peers' 8-byte (key, id) candidates.
        assert got.extra["merge_bytes"] == 31 * rows * width * 8
        assert not set(got.extra) & RECORD_CONTEXT
        assert len(got.operands) == len(got.in_specs) == 3 and len(got.out_specs) == 1
    with pytest.raises(ValueError, match="R <= 8"):
        m._build_select(16, 1024, args)
    with pytest.raises(ValueError, match="tie-skip"):
        m._build_select(8, 1024, parse(base + ["--tie-skip", "on"]))
    # The timing-only arm without the all-gather and the ids' select.
    (off,) = m._build_select(8, 1024, parse(base + ["--ties", "off"]))
    assert off.variant == "radix4_merge_noties_packed_p512"
    assert off.invalid and off.check is None and off.extra["ties"] == "off"
    assert off.extra["merge_bytes"] is None and off.extra["merge_candidates"] is None
    # No count exchange: a row sum before it is the default setup.
    (one,) = m._build_select(1, 4096, parse(base + ["--row-sum", "after,before"]))
    assert one.variant == "radix4_merge_packed_p512" and one.extra["row_sum"] == "after"
    # The other exchanges keep their setups and extra fields.
    flat = ["--exchange", "flat,merge", "--page-tokens", "512"]
    mixed = m._build_select(8, 1024, parse(base[:6] + flat))
    assert [s.variant for s in mixed] == ["radix4_flat_p512", "radix4_merge_p512"]
    assert "merge_bytes" not in mixed[0].extra


def _merge_compacted(rows, tokens, page, mask, keys):
    """``_merge_compact`` in Pallas interpret mode on one rank: the candidate
    slot ``[2 C, 8, 128]`` it fills from the select's int32 ``keys`` and the
    selection ``mask`` (both ``[R, T]``)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m = dsa_micro
    groups = m.virtual_groups(rows, tokens)
    planes = tokens // (groups * 128)
    cand = m.SELECT_K // (groups * 128)
    chosen = m.to_virtual(mask.astype(np.int32), groups)
    words = np.zeros((planes // 4, 8, 128), np.int32)
    for i in range(4):
        words |= chosen[i::4] << (8 * i)
    signed = m.to_virtual(keys, groups)

    def kernel(keys_in, words_in, out, slot, table, flush, lane_rows, sem):
        me = jax.lax.axis_index(m.AXIS)
        part = tokens // groups

        def token_id(s, at):
            return m._token_ids(s, at, me, groups, part, page)

        m._merge_compact(
            keys_in,
            words_in,
            slot,
            table,
            flush,
            lane_rows,
            sem.at[0],
            token_id,
            groups=groups,
            planes=planes,
            cand_planes=cand,
        )
        copy = pltpu.make_async_copy(slot, out, sem.at[1])
        copy.start()
        copy.wait()

    def body(keys_in, words_in, _):
        vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((2 * cand, 8, 128), jnp.int32),
            in_specs=[vmem, vmem],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2 * cand, 8, 128), jnp.int32),
                pltpu.VMEM((8, 128), jnp.int32),
                pltpu.SMEM((8, 128), jnp.int32),
                pltpu.SMEM((8,), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        )
        return (call(keys_in[0], words_in[0])[None],)

    operands = (jnp.asarray(signed)[None], jnp.asarray(words)[None])
    return _interpreted(body, operands)[0][0]


# A row's selected tokens: k at random, k in its first virtual row's part,
# none, a random spread, k - 1, few. Each R's shifts cover the first five.
MERGE_PICKS = [2048, "part", 0, 300, 2047, 5, 1, 2048 - 129]


@pytest.mark.parametrize(
    "rows, shift", [(1, s) for s in range(5)] + [(2, 0), (2, 2), (2, 4), (8, 0)]
)
def test_merge_compaction_moves_each_selected_token_once(monkeypatch, rows, shift):
    """Each row's selected tokens (``MERGE_PICKS[(r + shift) % 8]``) land in
    its own sublanes of the candidate slot once each, with their global ids;
    the rest is padding."""
    import numpy as np

    m = dsa_micro
    _interpret_on_one_rank(monkeypatch)
    tokens, page, k = 4096, 256, m.SELECT_K
    rng = np.random.default_rng([rows, shift])
    keys = (rng.permutation(2**20) - 2**19).astype(np.int32)[: rows * tokens]
    keys = keys.reshape(rows, tokens)
    keys[:, -3:] = m.INT32_MIN  # past the row's length
    mask = np.zeros((rows, tokens), bool)
    for r in range(rows):
        pick = MERGE_PICKS[(r + shift) % len(MERGE_PICKS)]
        if pick == "part":  # every selected token in the row's first part
            mask[r, :k] = True
        else:
            mask[r, rng.choice(tokens - 3, pick, replace=False)] = True
    slot = _merge_compacted(rows, tokens, page, mask, keys)
    groups = m.virtual_groups(rows, tokens)
    cand = k // (groups * 128)
    index = m.global_token_index(np.arange(tokens), 0, page)
    for r in range(rows):
        got_k = slot[:cand, r * groups : (r + 1) * groups].reshape(-1)
        got_i = slot[cand:, r * groups : (r + 1) * groups].reshape(-1)
        real = got_i >= 0
        assert (got_k[~real] == m.INT32_MIN).all(), f"row {r}"
        want = np.flatnonzero(mask[r])
        order = np.argsort(got_i[real])
        np.testing.assert_array_equal(
            got_i[real][order], index[want], err_msg=f"row {r}"
        )
        np.testing.assert_array_equal(
            got_k[real][order], keys[r, want], err_msg=f"row {r}"
        )


# rows, tokens, variant, histogram, scores, page tokens, lengths
MERGE_INTERPRET_FORMS = [
    # Every token a candidate (T <= k) at R 8: exact ties overflowing the quota
    # across ranks on rows 1, 3 and 5; row 6 a partial context, row 7 k tokens.
    (8, 512, "radix4", "cumulative", "exact", 256, [0, 0, 0, 0, 0, 0, 9000, 2048]),
    # The local select and compaction (T > k) at R 1: tie-heavy scores whose
    # ties overflow both the local and the global quota.
    (1, 4096, "radix4", "cumulative", "ties", 256, []),
]


def _merge_interpret32(forms, iterations=2):
    """The merge select kernel in Pallas interpret mode on 32 host devices (a
    process whose ``XLA_FLAGS`` gives 32): after ``iterations`` chained calls
    every rank's mask equals ``dsa_reference.select_topk``'s selection, for
    each of ``forms`` (:data:`MERGE_INTERPRET_FORMS`)."""
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    m = dsa_micro
    mesh = Mesh(np.array(jax.devices("cpu")[: m.RANKS]), (m.AXIS,))
    bridge.upload = lambda value: np.asarray(value)[None]
    failed = []
    for rows, tokens, variant, histogram, data, page, lengths in forms:
        start = time.time()
        args = SimpleNamespace(page_tokens=page, scores=data, lengths=lengths)
        setups = []
        for rank in range(m.RANKS):
            bridge.kernel_mesh = lambda axis, size=None, rank=rank: (None, rank)
            setups.append(
                m._select_setup(rows, tokens, variant, "merge", True, histogram, args)
            )
        setup = setups[0]
        operands = [
            jnp.asarray(np.concatenate([s.operands[i] for s in setups]))
            for i in range(len(setup.operands))
        ]
        call = jax.shard_map(
            lambda *ops, setup=setup: setup.body(*ops, iterations),
            mesh=mesh,
            in_specs=setup.in_specs,
            out_specs=setup.out_specs,
            check_vma=False,
        )
        params = pltpu.InterpretParams(dma_execution_mode="on_wait")
        with pltpu.force_tpu_interpret_mode(params):
            out = np.asarray(jax.block_until_ready(call(*operands))[0])
        full = m._select_lengths(rows, tokens, lengths)
        _, wanted, _ = m._select_reference(rows, tokens, data, full, page)
        local = np.arange(tokens)
        bad = [
            rank
            for rank in range(m.RANKS)
            if not np.array_equal(
                m.unpack_mask(out[rank], rows, tokens),
                wanted[:, m.global_token_index(local, rank, page)],
            )
        ]
        name = f"{setup.variant} R{rows} T{tokens} {data}"
        print(f"{name}: {'FAIL on ranks ' + str(bad) if bad else 'PASS'}", end="")
        print(f" ({time.time() - start:.0f} s)", flush=True)
        if bad:
            failed.append(name)
    assert not failed, f"masks differ from select_topk: {failed}"


@pytest.mark.slow
def test_merge_kernel_on_32_interpreted_ranks_equals_select_topk(request):
    """The whole merge kernel (candidates, the all-gather, the select over every
    rank's candidates, the tied ids' select and the marking) on 32 interpreted
    ranks, in a subprocess (the host device count is process-wide). Several
    minutes: it runs when ``-m`` selects ``slow``."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if "slow" not in (request.config.option.markexpr or ""):
        pytest.skip(
            "a 32-rank interpret run of several minutes: select it with -m slow"
        )
    root = Path(__file__).resolve().parents[3]
    path = [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=32",
        "PYTHONPATH": os.pathsep.join(path),
    }
    code = (
        "from tests.megakernel.kimi_k3.test_dsa_micro import _merge_interpret32, "
        "MERGE_INTERPRET_FORMS; _merge_interpret32(MERGE_INTERPRET_FORMS)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    print(done.stdout)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


def test_stage_dsa_micro_flags_and_dispatch(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    base = ["stage", "dsa-micro", "--prompts", "8192x1"]
    args = parse(base)
    assert (args.cases, args.rows, args.candidates, args.repeats) == (
        list(stage_dsa_micro.CASES),
        [1024, 2048, 8192],
        [8192, 16384, 32768, 147456],
        20,
    )
    args = parse(
        [*base, "--cases", "select-tc,exchange-bits", "--rows", "1024"]
        + ["--candidates", "8192,147456", "--repeats", "5"]
    )
    assert (args.cases, args.rows, args.candidates, args.repeats) == (
        ["select-tc", "exchange-bits"],
        [1024],
        [8192, 147456],
        5,
    )
    for flag, value in (("--cases", "nope"), ("--candidates", "100")):
        with pytest.raises(SystemExit):
            parse([*base, flag, value])

    calls = []
    monkeypatch.setattr(stage_dsa_micro, "run", lambda *a: calls.append(a))
    k3_stage.run(args, "records")
    assert calls == [(args, "records")]


def test_stage_dsa_micro_topk_select_is_the_reference_on_tied_rows():
    rows, candidates, k = 64, 4096, 2048
    _, _, _, lengths, scores = stage_dsa_micro.inputs(3, rows, candidates, k)
    tied = np.arange(0, rows, 4)
    scores = stage_dsa_micro.inject_ties(scores, lengths, tied, k)
    host, n = np.asarray(scores), np.asarray(lengths)
    for r in tied:
        best = np.sort(host[r, : n[r]])[::-1]
        assert best[k - 1] == best[k], r
    got = np.asarray(stage_dsa_micro.topk_mask(scores, lengths, k))
    want = np.asarray(ref.select_topk(scores, lengths, k))
    for r in range(rows):
        assert set(np.flatnonzero(got[r])) == set(want[r][want[r] >= 0]), r
    assert stage_dsa_micro.verdicts(got, want, host, n, k) == (0, 0)

    # The tie's other member instead: ties differ. A column below the k-th
    # score instead: a mismatch.
    r = int(tied[0])
    order = np.argsort(-host[r, : n[r]], kind="stable")
    swapped = got.copy()
    swapped[r, order[k - 1]], swapped[r, order[k]] = False, True
    assert stage_dsa_micro.verdicts(swapped, want, host, n, k) == (0, 1)
    swapped[r, order[k]], swapped[r, order[k + 1]] = False, True
    assert stage_dsa_micro.verdicts(swapped, want, host, n, k) == (1, 0)


def test_stage_threshold_forms_are_the_reference_threshold():
    """The ``threshold-*`` cases' three forms give the same ``(tau, quota)``
    as the reference's ``k``-th score on rows with a tie at the boundary,
    with zeros of both signs and with at most ``k`` valid keys."""
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    rows, candidates, k = 320, 512, 256
    _, _, _, lengths, scores = stage_dsa_micro.inputs(5, rows, candidates, k)
    scores = stage_dsa_micro.inject_ties(scores, lengths, np.arange(0, rows, 3), k)
    scores = scores.at[-1, : candidates // 2].set(0.0).at[-1, ::4].set(-0.0)
    valid = stage_dsa_micro.valid_of(lengths, candidates)
    keys = dsa_prefill.order_keys(scores)
    host, n = np.asarray(scores), np.asarray(lengths)
    reference = np.asarray(ref.select_topk(scores, lengths, k))
    assert (n <= k).any() and (n > k).any()
    for case in ("threshold-r8", "threshold-r4", "threshold-bisect"):
        tau, quota = stage_dsa_micro.threshold_form(case)(keys, valid, k)
        tau, quota = np.asarray(tau), np.asarray(quota)
        assert (
            stage_dsa_micro.threshold_mismatches(tau, quota, host, reference, n, k) == 0
        ), case
        if case == "threshold-r8":
            base = tau, quota
        np.testing.assert_array_equal(tau[n > k], base[0][n > k], case)
        np.testing.assert_array_equal(quota[n > k], base[1][n > k], case)
    np.testing.assert_array_equal(
        stage_dsa_micro.host_order_keys(host), np.asarray(keys)
    )
    tau = jnp.asarray(base[0]).at[-2].add(1)
    assert (
        stage_dsa_micro.threshold_mismatches(
            np.asarray(tau), base[1], host, reference, n, k
        )
        == 1
    )


def test_stage_dsa_check_and_bench_capacity_flags(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    top = ["--dsa-config", "DIR", "--synthetic-indexer", "random"]
    args = parse([*top, "stage", "dsa-check", "--prompts", "8192x1,6144x1@8192"])
    assert (args.stage_command, args.prompts, args.seed) == (
        "dsa-check",
        "8192x1,6144x1@8192",
        0,
    )
    calls = []
    monkeypatch.setattr(k3_stage, "run_dsa_check", lambda *a: calls.append(a))
    k3_stage.run(args, "records")
    assert calls == [(args, "records")]
    with pytest.raises(SystemExit):
        parse(["stage", "dsa-check"])  # --prompts is required

    bench_args = ["stage", "bench", "--prompts", "8192x1"]
    assert parse(bench_args).capacity == 0
    assert parse([*bench_args, "--capacity", "131072"]).capacity == 131072
    with pytest.raises(SystemExit):
        parse([*bench_args, "--capacity", "many"])


def test_stage_prefix_check_flags_and_helpers(monkeypatch):
    monkeypatch.delenv("JOBID", raising=False)
    parse = tool_main.parser().parse_args
    base = ["stage", "prefix-check", "--prompts", "41316x1@32768"]
    args = parse([*base, "--chain", "DIR"])
    assert (args.stage_command, args.chain, args.budget) == (
        "prefix-check",
        "DIR",
        8192,
    )
    assert (args.max_num_seqs, args.max_model_len, args.seed) == (4, 131072, 0)
    assert args.token_ids is None and not args.shuffle_blocks
    assert args.cold_steps is None and args.hit_steps is None
    assert k3_stage.prefix_runs(args.runs) == [
        ("cold", False, None),
        ("cold-again", False, None),
        ("hit", True, None),
        ("hit-stale", True, "stale"),
    ]
    assert [r[0] for r in k3_stage.prefix_runs("cold,stale,cold,copy,stale,cold")] == [
        "cold",
        "hit-stale",
        "cold-again",
        "hit-copy",
        "hit-stale-2",
        "cold-3",
    ]
    with pytest.raises(ValueError):
        k3_stage.prefix_runs("hit,cold")
    tokens = np.array([10, 11, 12])
    a = np.zeros((2, 3, 4), np.uint8)
    b = a.copy()
    b[1, 1:, 0] = 1
    assert k3_stage.layer_differences(a, b, tokens) == [(-1, 0), (11, 2)]
    assert k3_stage.merge_layer_differences(
        [[(-1, 0), (11, 2)], [(-1, 0), (9, 1)]]
    ) == [(-1, 0), (9, 3)]
    assert k3_stage.explicit_steps("4,2", 8, 14) == [(8, 4), (12, 2)]
    with pytest.raises(SystemExit):
        k3_stage.explicit_steps("4,2", 8, 15)
    assert k3_stage._cases(args.prompts) == [(41316, 1, 32768)]
    calls = []
    monkeypatch.setattr(k3_stage, "run_prefix_check", lambda *a: calls.append(a))
    k3_stage.run(args, "records")
    assert calls == [(args, "records")]
    with pytest.raises(SystemExit):
        parse(base)  # --chain is required

    # Arena column c is cached token c, window column capacity + w is token
    # start + w; the scores keep that order over the row's valid candidates.
    capacity, start, row = 64, 40, 5
    columns = [3, 39, capacity + 0, capacity + 5]
    words = np.zeros((capacity + 32) // 32, np.uint32)
    for c in columns:
        words[c // 32] |= np.uint32(1) << np.uint32(c % 32)
    np.testing.assert_array_equal(
        k3_stage.selected_tokens(words.view(np.int32), start, capacity),
        [3, 39, 40, 45],
    )
    scores = np.arange(capacity + 32, dtype=np.float32)
    got = k3_stage.candidate_scores(scores, start, row, capacity)
    np.testing.assert_array_equal(got[[3, 39, 40, 45]], [3, 39, 64, 69])
    assert got.size == start + row + 1
    one = np.float32(1.0)
    assert k3_stage.ulps([one], [np.nextafter(one, np.float32(2))]) == 1
    assert k3_stage.ulps([np.float32(0.0)], [np.float32(-0.0)]) == 0

    # A hit shares the cold run's MLA blocks before it and its KDA state after
    # them, and takes new blocks for the rest.
    ids = iter(range(1, 100))
    block, state, widths = 16, 4, (4, 16)
    cold, new = k3_stage.prefix_tables(
        40, 0, [(0, 16), (16, 16), (32, 8)], block, state, widths, None,
        lambda: next(ids),
    )  # fmt: skip
    assert cold[0].tolist() == [1, 2, 3, 0] and len(new[1]) == 9
    assert cold[1][[3, 7, 9]].tolist() == [4, 5, 6]
    warm, new = k3_stage.prefix_tables(
        40, 32, [(32, 8)], block, state, widths, cold, lambda: next(ids)
    )
    assert warm[0].tolist() == [1, 2, 13, 0] and new[0] == [13]
    assert [t[7] for t in warm[1:]] == [t[7] for t in cold[1:]]
    assert [int(t[9]) for t in warm[1:]] == new[1] == [14, 15, 16]

    a = np.zeros((2, 3, 4), np.uint8)
    b = a.copy()
    b[1, 2, 0] = 1
    assert k3_stage.equal_tokens(a, b) == 2
    assert k3_stage.equal_tokens(a[:, :0], b[:, :0]) == 0
