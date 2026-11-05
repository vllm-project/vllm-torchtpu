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
"""Kimi-K3.1's decode select (dsa_select) on interpreted ranks."""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test


def _rank_inputs(rows, page, group, positions, data, seed):
    """Global scores per row (normal, rounded ties, or exact ties at the
    threshold: on even rows across the quota, on odd rows filling it, the new
    token among them) and each member's local scores in the scorer's layout."""
    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = np.random.default_rng(seed)
    held = max(
        int(dsa_decode.local_pages(p + 1, o, page, group))
        for p in positions
        for o in range(group)
    )
    tokens = held * page
    local = [np.full((rows, tokens), np.nan, np.float32) for _ in range(group)]
    for r, position in enumerate(positions):
        values = rng.standard_normal(position + 1).astype(np.float32)
        if data == "ties":
            values = np.round(values * 2).astype(np.float32)
        if data == "exact" and position + 1 > 2048:
            order = np.argsort(-values, kind="stable")
            order = order[order != position]
            # With the new token: across the quota, or exactly filling it.
            ties = 20 if r % 2 == 0 else 7
            values[order[2040 : 2040 + ties]] = values[order[2040]]
            values[position] = values[order[2040]]
        for o in range(group):
            length = int(dsa_decode.local_length(position, o, page, group))
            local[o][r, :length] = values[
                ref.global_token(np.arange(length), o, page, group)
            ]
    return local, tokens


def _place(by_row, rows, tokens, fill=np.nan):
    """[rows, T] -> the scorer's [planes, V, 128] (dsa_decode.scores_by_row's
    inverse), ``fill`` padding."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    planes, vrows, _ = dsa_decode.score_shape(tokens, rows)
    parts = dsa_decode.virtual_rows(rows)[1]
    out = np.full((planes, vrows, 128), fill, by_row.dtype)
    padded = np.full((rows, parts * planes * 128), fill, by_row.dtype)
    padded[:, :tokens] = by_row
    out[:, : rows * parts] = (
        padded.reshape(rows, parts, planes, 128)
        .transpose(2, 0, 1, 3)
        .reshape(planes, -1, 128)
    )
    return out


SELECT_POINTS = [
    # rows, page tokens, positions (cycled), scores
    (1, 256, [9000], "normal"),
    (2, 512, [12000, 1500], "ties"),
    (4, 256, [5000, 2047, 2048, 9999], "exact"),
    (8, 512, [12287, 100, 4000, 2049, 7000, 12000, 3000, 2048], "normal"),
    (16, 256, [6000, 2000, 9000, 4097], "ties"),
    (1, 256, [20000], "ties"),  # the tie round over eight parts
    (32, 512, [9000, 3000, 30000, 2100], "ties"),
    (4, 256, [100, 2047, 0, 1500], "normal"),  # no row scores
]
# Eight rows (dsa_select_pack16): normal, rounded and threshold ties.
PACK16_POINTS = [
    (8, 512, [12287, 100, 4000, 2049, 7000, 12000, 3000, 2048], "normal"),
    (8, 256, [6000, 2000, 9000, 4097, 5000, 2047, 2048, 9999], "ties"),
    (8, 256, [5000, 2047, 2048, 9999, 6000, 2000, 9000, 4097], "exact"),
]


def _select_on_ranks(
    group, points, races=False, owners=None, form=None, stops=None, words=False
):
    """select_rows on ``group`` interpreted ranks (a shard_map over the host
    devices; rank ``d`` the page owner ``owners[d]``, default ``d``; ``form``
    the select form, default ``SELECT_FORMS[rows]``): every rank's mask and
    new-token flags against dsa_reference.decode_selection; a row of at most k
    tokens (its fold is dense) keeps an empty mask. ``stops`` (a list): each
    point's rank-0 stop flag is appended. With ``words`` (rows of one virtual
    row) the select marks bit words, checked with their summary and counts."""
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    mesh = Mesh(np.array(jax.devices("cpu")[:group]), ("tp",))
    owners = list(range(group)) if owners is None else list(owners)
    failed = []
    for rows, page, cycle, data in points:
        positions = [cycle[i % len(cycle)] for i in range(rows)]
        local, tokens = _rank_inputs(
            rows, page, group, positions, data, seed=rows + page
        )
        want, want_new = ref.decode_selection(
            local, positions, page_tokens=page, group=group
        )
        short = np.asarray(positions) + 1 <= 2048
        # Every rank's copy of the new token's score.
        new_scores = np.zeros(rows, np.float32)
        for r, position in enumerate(positions):
            owner = position // page % group
            length = int(np.sum(~np.isnan(local[owner][r])))
            new_scores[r] = local[owner][r, length - 1]
        scores = np.stack([_place(local[o], rows, tokens) for o in owners])
        interpret = pltpu.InterpretParams(
            dma_execution_mode="eager" if races else "on_wait",
            detect_races=races,
            uninitialized_memory="nan",
        )

        def body(s, p, n, rows=rows, page=page, interpret=interpret):
            return dsa_select.select_rows(
                s[0],
                p[0],
                n[0],
                page_tokens=page,
                group=group,
                form=form or dsa_select.SELECT_FORMS[rows],
                owners=owners,
                words=words,
                interpret=interpret,
                stats=True,
            )

        call = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp")),
            out_specs=((P("tp"),) * 3 if words else P("tp"), P("tp"), P("tp")),
            check_vma=False,
        )
        selection, flags, stopped = call(
            jnp.asarray(scores),
            jnp.asarray(np.tile(np.asarray(positions, np.int32), (group, 1))),
            jnp.asarray(np.tile(new_scores, (group, 1))),
        )
        if words:
            bits, summary, counts = (np.asarray(x) for x in selection)
            bits = bits.reshape(group, -1, *bits.shape[1:])
            summary = summary.reshape(group, -1, *summary.shape[1:])
            counts = counts.reshape(group, -1, counts.shape[-1])
        else:
            mask_words = np.asarray(selection)
            mask_words = mask_words.reshape(group, -1, *mask_words.shape[1:])
        if races:
            assert not interpret_pallas_call.races.races_found, (rows, page, data)
        flags = np.asarray(flags).reshape(group, rows)
        stopped = np.asarray(stopped).reshape(group)
        assert (stopped == stopped[0]).all(), (rows, page, data, stopped)
        if stops is not None:
            stops.append(int(stopped[0]))
        for d, o in enumerate(owners):
            expect = want[o] & ~short[:, None]
            if words:
                got = dsa_select.unpack_words(bits[d], rows, tokens)
                word_rows = bits[d].transpose(1, 0, 2).reshape(bits.shape[2], -1)[:rows]
                nonzero = (word_rows != 0)[:, : 32 * 128]
                marked = (
                    summary[d][0, :rows].view(np.uint32)[:, :, None]
                    >> np.arange(32, dtype=np.uint32)
                ) & 1
                marked = marked.reshape(rows, -1)[:, : nonzero.shape[1]] != 0
                sparse = ~short
                if not (
                    np.array_equal(counts[d][:rows, 0][sparse], got.sum(axis=1)[sparse])
                    and np.array_equal(marked[sparse], nonzero[sparse])
                ):
                    failed.append((rows, page, data, d, "summary or count"))
            else:
                got = dsa_select.unpack_mask(mask_words[d], rows, tokens)
            if not (
                np.array_equal(got, expect) and np.array_equal(flags[d] != 0, want_new)
            ):
                failed.append((rows, page, data, d))
    assert not failed, failed


@pytest.mark.parametrize("races", [False, True])
def test_select_on_8_interpreted_ranks_is_the_reference(races):
    _select_on_ranks(8, SELECT_POINTS, races=races)


WORD_POINTS = [
    # rows, page tokens, positions (cycled), scores: one virtual row a row
    (8, 512, [12287, 100, 4000, 2049, 7000, 12000, 3000, 2048], "normal"),
    (16, 256, [6000, 2000, 9000, 4097], "ties"),
    (32, 512, [9000, 3000, 30000, 2100], "ties"),
]


@pytest.mark.parametrize("races", [False, True])
def test_select_words_on_8_interpreted_ranks_are_the_reference(races):
    """The selection as lane-major bit words, with their summary and counts."""
    _select_on_ranks(8, WORD_POINTS, races=races, words=True)


def test_select_ties_follow_the_owner_table():
    """Ranks holding the page owners of a permuted table: the tie round's
    ranks still follow global token order (slots by owner, not by rank)."""
    _select_on_ranks(8, SELECT_POINTS[1:3], owners=[3, 0, 6, 1, 7, 2, 5, 4])


def _pack16_form():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    return dsa_select.SelectForm(8, "flat", "packed", "after", "pack16")


@pytest.mark.parametrize("races", [False, True])
def test_pack16_select_on_8_interpreted_ranks_is_the_reference(races):
    """Eight rows with 16-bit counts in a flat exchange (two counts a lane,
    every member's packet unpacked before the sum): exact, ties included."""
    _select_on_ranks(8, PACK16_POINTS, races=races, form=_pack16_form())


def test_pack16_form_needs_a_flat_exchange_of_packed_counts():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    for fields in (
        (8, "host", "packed", "after"),
        (4, "flat", "cumulative", "after"),
        (8, "flat", "packed", "before"),
    ):
        with pytest.raises(ValueError, match="16-bit counts"):
            dsa_select.SelectForm(*fields, "pack16")
    assert dsa_select.select_form(8, 4096) == dsa_select.SELECT_FORMS[8]


# Eight rows, narrower digits (dsa_select_digit_bits) and the stop (dsa_select_stop):
# normal scores (rows decided after 20-30 bits, so the rounds stop early), rounded
# scores and threshold ties (no row is decided before the last round; the tie round
# runs).
NARROW_POINTS = [
    (8, 512, [24575, 30000, 20000, 18000, 26000, 28000, 22000, 16000], "normal"),
    (8, 256, [6000, 2000, 9000, 4097, 5000, 2047, 2048, 9999], "ties"),
    (8, 256, [5000, 2047, 2048, 9999, 6000, 2000, 9000, 4097], "exact"),
]


def _forms(bits, stop):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    return [
        dsa_select.SelectForm(bits, "host", "packed", "after", "int32", stop),
        dsa_select.SelectForm(bits, "flat", "packed", "after", "pack16", stop),
    ]


@pytest.mark.parametrize("bits", [6, 7, 8])
def test_narrow_digits_and_stop_on_8_interpreted_ranks_are_the_reference(bits):
    """Eight rows at 6, 7 and 8 bits a digit (a last round of the remaining
    bits), int32 counts in the host exchange and 16-bit counts in the flat
    one, with and without the stop: exact, ties included; the stop ends the
    rounds early on the normal point and every member agrees."""
    for stop in ("off", "any"):
        for form in _forms(bits, stop):
            stops = []
            _select_on_ranks(8, NARROW_POINTS, form=form, stops=stops)
            if stop == "off":
                assert stops == [0, 0, 0], (form, stops)
            else:
                assert stops[0] == 1 and stops[2] == 0, (form, stops)


def test_stop_at_16_and_32_rows_is_the_reference():
    """4-bit digits at 16 and 32 rows with the stop after any round and after
    an even number of rounds (two passes a layer), and 8-bit digits below
    eight rows (virtual rows summed before the exchange)."""
    import dataclasses

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    points = SELECT_POINTS[:3] + SELECT_POINTS[4:7]
    for stop in ("any", "even"):
        for point in points:
            form = dataclasses.replace(dsa_select.SELECT_FORMS[point[0]], stop=stop)
            _select_on_ranks(8, [point], form=form)


# Below eight rows, eight, 16 and 32 rows (ties), and 16 rows at 300K tokens:
# over 255 planes a row, so two spans of 8-bit counters, in every block and with
# a block of short rows, whose second span holds no planes.
UNROLL_POINTS = [
    *SELECT_POINTS[:3],
    PACK16_POINTS[1],
    SELECT_POINTS[4],
    SELECT_POINTS[6],
    (16, 512, [300000, 280000, 9000, 2100], "normal"),
    (16, 512, [300000] * 8 + [9000] * 8, "normal"),
]


@pytest.mark.parametrize("unroll", [1, 2, 4])
def test_count_passes_at_each_unroll_are_the_reference(unroll):
    """KernelConfig.dsa_select_count_unroll and dsa_select_packed_unroll: the
    packed forms with the count pass ``unroll`` planes a trip (each block's odd
    planes in the tail), packed counts at 16 and 32 rows with and without the
    stop, and at eight rows the 6- and 8-bit forms with 16-bit counts: exact,
    ties included; at 300K tokens two spans of counters, each folded into
    16-bit lane pairs."""
    import dataclasses

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    for point in UNROLL_POINTS:
        rows = point[0]
        base = dsa_select.SELECT_FORMS[rows]
        forms = [dataclasses.replace(base, histogram="packed", unroll=unroll)]
        if rows < 16 and unroll == 1:
            forms = []
        if rows >= 16 and unroll > 1:
            forms += [dataclasses.replace(forms[0], stop=s) for s in ("any", "even")]
        if rows == 8 and unroll > 1:
            forms += [
                dataclasses.replace(_forms(bits, "off")[1], unroll=unroll)
                for bits in (6, 8)
            ]
        for form in forms:
            _select_on_ranks(8, [point], form=form)


@pytest.mark.parametrize("races", [False, True])
def test_narrow_pack16_stop_has_no_races(races):
    """The 6-bit 16-bit-count packet ([2, 128]: 1 KiB at eight rows) and the
    stop under the race detector."""
    _select_on_ranks(8, NARROW_POINTS[:1], races=races, form=_forms(6, "any")[1])


def test_narrow_forms_and_rounds():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    for bits, rounds, first in ((4, 8, 4), (6, 6, 3), (7, 5, 2), (8, 4, 2)):
        form = dsa_select.SelectForm(bits, "flat", "packed", "after", stop="any")
        assert (form.rounds, form.first_stop) == (rounds, first)
    assert (
        dsa_select.SelectForm(4, "flat", "cumulative", "after", stop="even").first_stop
        == 5
    )
    with pytest.raises(ValueError, match="unknown select form"):
        dsa_select.SelectForm(5, "flat", "packed", "after")
    with pytest.raises(ValueError, match="unknown select form"):
        dsa_select.SelectForm(8, "flat", "packed", "after", stop="yes")
    with pytest.raises(ValueError, match="16-bit counts"):
        dsa_select.SelectForm(4, "flat", "packed", "after", "pack16")
    with pytest.raises(ValueError, match="unknown select form"):
        dsa_select.SelectForm(4, "flat", "packed", "after", unroll=3)
    with pytest.raises(ValueError, match="unknown select form"):
        dsa_select.SelectForm(4, "flat", "cumulative", "after", unroll=2)
    assert dsa_select._histogram_packet(8, 1, _forms(6, "off")[1]) == (2, 128)
    assert dsa_select._histogram_packet(8, 1, _forms(7, "off")[1]) == (4, 128)
    assert dsa_select._histogram_packet(8, 1, _forms(8, "off")[1]) == (8, 128)
    for form in _forms(6, "any") + _forms(7, "any"):
        for group in (8, 32):
            entries = dsa_select.persistent_scratch(8, 64 * 512, 512, group, form)
            assert len(entries) == len(dsa_select.persistent_names(form, group))


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_select_refuses_inboxes_of_another_group(rows):
    """The histogram and tie exchanges index their inboxes by peer slot: wires
    sized for another group are refused when the select is built."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    form, local_tokens, page = dsa_select.SELECT_FORMS[rows], 2048, 256

    def select(group, wired):
        wires = dsa_select.persistent_scratch(rows, local_tokens, page, wired, form)
        return dsa_select.TopkSelect(
            dsa_select.SelectScratch(None, None, None, None, None, wires),
            batch=rows, local_tokens=local_tokens, position_ref=None, owner=0,
            member=0, base=0, group=group, page_tokens=page, form=form,
        )  # fmt: skip

    select(16, 16)
    for wired in (8, 32):
        with pytest.raises(ValueError, match="inboxes"):
            select(16, wired)


@pytest.mark.slow
def test_select_on_32_interpreted_ranks_is_the_reference(request):
    """The select on 32 interpreted ranks (the production group: radix8 host
    exchange in both stages at R 8), in a subprocess (the host device count is
    process-wide). Several minutes: it runs when ``-m`` selects ``slow``."""
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
        "from tests.megakernel.kimi_k3.test_dsa_select import _select_on_ranks, "
        "SELECT_POINTS, PACK16_POINTS, _pack16_form; "
        "_select_on_ranks(32, SELECT_POINTS[:2] + SELECT_POINTS[3:4]); "
        "_select_on_ranks(32, PACK16_POINTS, form=_pack16_form()); "
        "from tests.megakernel.kimi_k3.test_dsa_select import NARROW_POINTS, _forms; "
        "[_select_on_ranks(32, NARROW_POINTS, form=f) "
        "for f in _forms(6, 'any') + _forms(7, 'any')]"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    print(done.stdout)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


@pytest.mark.parametrize(
    "rows, local_tokens",
    [
        (1, 2048),  # eight parts of two planes: every 512-token call crosses parts
        (4, 1536),  # two parts of six planes, the last mask word half used
        (16, 1024),
    ],
)
def test_lanes_read_each_rows_selection(rows, local_tokens):
    """lanes(row, first, 512) at traced rows and starts against unpack_mask;
    calls running past the table read inside it (their tokens are unchecked:
    past the row's limit); a row of at most k tokens reads ones."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    page, width = 256, 512
    planes = dsa_decode.score_shape(local_tokens, rows)[0]
    rng = np.random.default_rng(rows)
    by_plane = _place(rng.random((rows, local_tokens)) < 0.5, rows, local_tokens, False)
    by_plane = np.concatenate(
        (by_plane, np.zeros((-planes % 4, *by_plane.shape[1:]), bool))
    ).astype(np.uint8)
    words = by_plane.reshape(-1, 4, *by_plane.shape[1:]).transpose(0, 2, 3, 1)
    words = np.ascontiguousarray(words).view(np.int32)[..., 0]
    want = dsa_select.unpack_mask(words, rows, local_tokens)
    positions = np.full(rows, 40000, np.int32)
    positions[-1] = 2047 if rows > 1 else positions[-1]
    cases = np.asarray(
        [(r, f) for r in range(rows) for f in range(0, local_tokens, 128)], np.int32
    ).T

    def kernel(position_ref, row_ref, first_ref, words_ref, out_ref):
        scratch = dsa_select.SelectScratch(words_ref, None, None, None, None, ())
        select = dsa_select.TopkSelect(
            scratch,
            batch=rows,
            local_tokens=local_tokens,
            position_ref=position_ref,
            owner=0,
            member=0,
            base=0,
            group=8,
            page_tokens=page,
            form=dsa_select.SELECT_FORMS[rows],
        )

        @pl.loop(0, cases.shape[1])
        def _(i):
            out_ref[pl.ds(i, 1), :] = select.lanes(row_ref[i], first_ref[i], width)

    got = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((cases.shape[1], width), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        interpret=pltpu.InterpretParams(),
    )(jnp.asarray(positions), jnp.asarray(cases[0]), jnp.asarray(cases[1]), words)
    got = np.asarray(got)
    for i, (r, first) in enumerate(cases.T):
        if positions[r] + 1 <= 2048:
            assert (got[i] == 1).all(), (r, first)
            continue
        inside = min(width, local_tokens - first)
        np.testing.assert_array_equal(
            got[i, :inside], want[r, first : first + inside], err_msg=f"{r}, {first}"
        )


@pytest.mark.parametrize(
    "rows, local_tokens, width", [(8, 8192, 512), (32, 5120, 1024), (16, 4096, 256)]
)
def test_word_lanes_read_each_rows_selection(rows, local_tokens, width):
    """lanes(row, first, width) from bit words (``first`` a multiple of
    ``width``, as the attention's fold steps call it) against unpack_words; a
    row of at most k tokens reads ones."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    planes, vrows, _ = dsa_decode.score_shape(local_tokens, rows)
    tiles = -(-planes // 32)
    rng = np.random.default_rng(rows)
    chosen = np.zeros((vrows, tiles * 32 * 128), bool)
    chosen[:rows, :local_tokens] = rng.random((rows, local_tokens)) < 0.5
    bits = (
        chosen.reshape(vrows, tiles, 32, 128).astype(np.uint32)
        << np.arange(32, dtype=np.uint32)[:, None]
    )
    words = np.ascontiguousarray(
        bits.sum(axis=2, dtype=np.uint32).transpose(1, 0, 2)
    ).view(np.int32)
    want = dsa_select.unpack_words(words, rows, local_tokens)
    assert np.array_equal(want, chosen[:rows, :local_tokens])
    positions = np.full(rows, 40000, np.int32)
    positions[-1] = 2047
    cases = np.asarray(
        [(r, f) for r in range(rows) for f in range(0, local_tokens, width)], np.int32
    ).T

    def kernel(position_ref, row_ref, first_ref, words_ref, out_ref):
        scratch = dsa_select.SelectScratch(
            None, None, None, None, None, (), words=words_ref
        )
        select = dsa_select.TopkSelect(
            scratch,
            batch=rows,
            local_tokens=local_tokens,
            position_ref=position_ref,
            owner=0,
            member=0,
            base=0,
            group=8,
            page_tokens=256,
            form=dsa_select.SELECT_FORMS[rows],
        )

        @pl.loop(0, cases.shape[1])
        def _(i):
            out_ref[pl.ds(i, 1), :] = select.lanes(row_ref[i], first_ref[i], width)

    got = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((cases.shape[1], width), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        interpret=pltpu.InterpretParams(),
    )(
        jnp.asarray(positions),
        jnp.asarray(cases[0]),
        jnp.asarray(cases[1]),
        jnp.asarray(words),
    )
    got = np.asarray(got)
    for i, (r, first) in enumerate(cases.T):
        if positions[r] + 1 <= 2048:
            assert (got[i] == 1).all(), (r, first)
            continue
        np.testing.assert_array_equal(
            got[i], want[r, first : first + width], err_msg=f"{r}, {first}"
        )


def test_tables_over_the_tie_packet_are_refused():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    assert dsa_select.tie_pages(5 * 512, 512) == 8
    with pytest.raises(ValueError, match="128"):
        dsa_select.tie_pages(129 * 256, 256)


def test_every_batch_class_has_a_form():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    assert set(dsa_select.SELECT_FORMS) == set(dsa_decode.BATCH_CLASSES)
    assert all(
        f.row_sum == "after" for b, f in dsa_select.SELECT_FORMS.items() if b >= 8
    )


def test_persistent_scratch_matches_its_names():
    """One name per persistent entry: the histogram stages of the form (the
    host exchange's second stage only across hosts), then the tie inbox."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    for batch, form in dsa_select.SELECT_FORMS.items():
        for group in (8, 32):
            entries = dsa_select.persistent_scratch(batch, 64 * 512, 512, group, form)
            names = dsa_select.persistent_names(form, group)
            assert len(entries) == len(names) and len(set(names)) == len(names)
    host = dsa_select.SELECT_FORMS[8]
    assert len(dsa_select.persistent_names(host, 32)) == 6
    assert len(dsa_select.persistent_names(host, 8)) == 4
    with pytest.raises(ValueError, match="power of two"):
        dsa_select.persistent_names(dsa_select.SELECT_FORMS[1], 24)
