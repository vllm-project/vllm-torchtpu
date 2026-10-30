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
"""The expert-placement generator on synthetic recordings: a valid table in
the loader's format (every expert once among the primaries, copies never on
their owner, slots_per_core = owned + extra), the hottest experts copied,
the LPT and EPLB primaries balanced, both input forms (json, log lines)
read, the validator's teeth, and the layout round-trips through json."""

import importlib.util
import json
import pathlib
import random

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/vllm/benchmark/inferencex/qwen3.5/make_expert_placement.py"
)
_spec = importlib.util.spec_from_file_location("make_expert_placement", _SCRIPT)
mep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mep)

SHARDS, EXPERTS, COPIES = 8, 512, 128


def _loads(layers=3, seed=0):
    rng = random.Random(seed)
    return {
        i: [float(int(rng.paretovariate(1.2) * 100)) for _ in range(EXPERTS)]
        for i in range(layers)
    }


def _recording(loads, prefill=None):
    return {
        "experts": EXPERTS,
        "top_k": 10,
        "layers": {
            str(i): {
                "decode": [int(v) for v in load],
                "prefill": [int(v) for v in (prefill or {}).get(i, [0] * EXPERTS)],
            }
            for i, load in loads.items()
        },
    }


def _check_table(table, loads, copies=COPIES):
    mep.validate(table)
    assert table["shards"] == SHARDS
    assert table["experts"] == EXPERTS
    assert table["owned_per_core"] == EXPERTS // SHARDS
    assert table["extra_slots"] == copies // SHARDS
    assert table["slots_per_core"] == EXPERTS // SHARDS + copies // SHARDS
    assert table["copies"] is bool(copies)
    assert set(table["layers"]) == {str(i) for i in loads}
    for i, load in loads.items():
        layer = table["layers"][str(i)]
        owned = table["owned_per_core"]
        prim = [e for c in layer["cores"] for e in c[:owned]]
        assert sorted(prim) == list(range(EXPERTS))
        assert all(len(c) == table["slots_per_core"] for c in layer["cores"])
        if copies:
            hottest = sorted(range(EXPERTS), key=lambda e: (-load[e], e))[:copies]
            assert list(map(int, layer["copies"])) == hottest  # hottest first
            for e, r in layer["copies"].items():
                assert int(e) not in layer["cores"][r][:owned]
                assert int(e) in layer["cores"][r][owned:]


def test_lpt_table_is_valid_and_copies_the_hottest():
    loads = _loads()
    table = mep.build(loads, SHARDS, EXPERTS, COPIES, algorithm="lpt")
    _check_table(table, loads)
    assert "longest-processing-time" in table["algorithm"]


def test_lpt_primaries_are_balanced():
    loads = _loads(layers=1)
    cores = mep.balanced_primaries_lpt(loads[0], SHARDS)
    per_core = [sum(loads[0][e] for e in c) for c in cores]
    assert all(len(c) == EXPERTS // SHARDS for c in cores)
    # LPT's bound: no core above the mean by more than one expert's load (a
    # Pareto draw can hold one expert heavier than everything else together)
    mean = sum(per_core) / SHARDS
    assert max(per_core) <= mean + max(loads[0]), per_core
    assert min(per_core) >= mean - max(loads[0]), per_core


def test_eplb_primaries_when_vllm_imports():
    pytest.importorskip("vllm.distributed.eplb.policy.default")
    loads = _loads()
    table = mep.build(loads, SHARDS, EXPERTS, COPIES, algorithm="eplb")
    _check_table(table, loads)
    assert "DefaultEplbPolicy" in table["algorithm"]
    cores = table["layers"]["0"]["cores"]
    owned = table["owned_per_core"]
    per_core = [sum(loads[0][e] for e in c[:owned]) for c in cores]
    mean = sum(per_core) / SHARDS
    assert max(per_core) <= mean + max(loads[0]), per_core


def test_auto_falls_back_to_lpt_without_vllm(monkeypatch):
    monkeypatch.setattr(mep, "balanced_primaries_eplb", lambda loads, shards: None)
    table = mep.build(_loads(layers=1), SHARDS, EXPERTS, COPIES)
    assert "longest-processing-time" in table["algorithm"]


def test_no_copies_table():
    loads = _loads(layers=1)
    table = mep.build(loads, SHARDS, EXPERTS, 0, algorithm="lpt")
    _check_table(table, loads, copies=0)
    assert "copies" not in table["layers"]["0"]


def test_copies_go_to_the_least_expected_load_with_a_free_slot():
    # 16 hot experts per owning core (128 in all), expert 0 far hotter
    load = [2.0 if e % 64 < 16 else 1.0 for e in range(EXPERTS)]
    load[0] = 1000.0
    cores = [list(range(r * 64, (r + 1) * 64)) for r in range(SHARDS)]
    placed = mep.place_copies(load, cores, extra_slots=16, copies=COPIES)
    assert len(placed) == COPIES
    assert placed[0] == 1  # not its owner; all other cores tie, lowest wins
    holders = list(placed.values())
    assert all(holders.count(r) == 16 for r in range(SHARDS))  # every slot filled
    assert all(e not in cores[r] for e, r in placed.items())
    assert set(placed) == {e for e in range(EXPERTS) if e % 64 < 16}


def test_copies_repair_moves_an_earlier_copy():
    """3 shards x 2 owned, one extra slot each, 3 copies. Experts 0 and 1
    (core 0) are heavy, so the greedy sends copy(2) to core 2 and copy(4) to
    core 1; copy(0) then finds a free slot only on its owner. The repair
    moves copy(4) (owned by core 2, held on core 1) onto core 0 and gives
    core 1's slot to expert 0."""
    cores = [[0, 1], [2, 3], [4, 5]]
    load = [8.0, 8.0, 10.0, 0.1, 9.0, 0.1]
    placed = mep.place_copies(load, cores, extra_slots=1, copies=3)
    assert placed == {2: 2, 4: 0, 0: 1}


def test_copies_infeasible_is_an_error():
    # both hot experts owned by core 0, only one non-owner slot
    with pytest.raises(SystemExit):
        mep.place_copies([10.0, 9.0, 1.0, 1.0], [[0, 1], [2, 3]], 1, 2)


def test_prefill_weight_changes_the_load():
    loads = {0: [1.0] * EXPERTS}
    prefill = {0: [0] * EXPERTS}
    prefill[0][7] = 1000
    doc = _recording(loads, prefill)
    assert mep.layer_loads([doc], 0.0)[0][7] == 1.0
    assert mep.layer_loads([doc], 0.5)[0][7] == 501.0
    assert mep.layer_loads([doc, doc], 0.0)[0][7] == 2.0  # summed


def test_reads_json_and_log_lines(tmp_path):
    import base64
    import zlib

    def _encode_counts(payload):
        return base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()

    loads = _loads(layers=2)
    doc = _recording(loads)
    path = tmp_path / "routing_counts_rank0.json"
    path.write_text(json.dumps(doc))
    assert mep.read_recording(str(path))["layers"].keys() == {"0", "1"}

    log = tmp_path / "server.log"
    lines = ["INFO something else\n", f"{mep.LOG_TAG} rank=0 meta eJwDAAAAAAE=\n"]
    for step in (2000, 4000):  # two dumps: the last one wins
        for key, entry in doc["layers"].items():
            payload = {
                "decode": [v * (step // 2000) for v in entry["decode"]],
                "prefill": entry["prefill"],
            }
            lines.append(
                f"{mep.LOG_TAG} rank=0 step={step} layer={key} "
                f"{_encode_counts(payload)}\n"
            )
    log.write_text("".join(lines))
    got = mep.read_recording(str(log))
    assert got["layers"]["1"]["decode"] == [v * 2 for v in doc["layers"]["1"]["decode"]]
    # two ranks' lines are summed
    log.write_text("".join(lines) + "".join(lines).replace("rank=0", "rank=1"))
    got = mep.read_recording(str(log))
    assert got["layers"]["1"]["decode"] == [v * 4 for v in doc["layers"]["1"]["decode"]]


def test_validator_rejects_broken_tables():
    table = mep.build(_loads(layers=1), SHARDS, EXPERTS, COPIES, algorithm="lpt")
    bad = json.loads(json.dumps(table))
    bad["layers"]["0"]["cores"][0][0] = bad["layers"]["0"]["cores"][0][1]
    with pytest.raises(ValueError, match="exactly once"):
        mep.validate(bad)
    bad = json.loads(json.dumps(table))
    c = bad["layers"]["0"]["cores"][0]
    c[64] = c[0]  # a copy on its owner
    with pytest.raises(ValueError, match="owner"):
        mep.validate(bad)
    bad = json.loads(json.dumps(table))
    bad["slots_per_core"] = 81
    with pytest.raises(ValueError):
        mep.validate(bad)
    bad = json.loads(json.dumps(table))
    bad["layers"]["0"]["copies"] = {}
    with pytest.raises(ValueError, match="copy order"):
        mep.validate(bad)


def test_layout_round_trips_one_core_per_line():
    table = mep.build(_loads(layers=2), SHARDS, EXPERTS, COPIES, algorithm="lpt")
    text = mep.dumps(table)
    assert json.loads(text) == table
    lines = text.splitlines()
    assert lines[0] == "{" and lines[1] == '  "shards": 8,'
    assert '  "layers": {' in lines
    assert len([ln for ln in lines if ln.startswith("        [")]) == 2 * SHARDS


def test_check_accepts_the_reference_copies_table():
    ref = pathlib.Path("/tmp/port06-ref-copies128.json")
    if not ref.exists():
        pytest.skip("reference table not on this host")
    assert mep.main(["--check", str(ref)]) == 0


def test_main_end_to_end(tmp_path):
    doc = _recording(_loads(layers=2))
    src = tmp_path / "routing_counts_rank0.json"
    src.write_text(json.dumps(doc))
    out = tmp_path / "placement.json"
    args = [str(src), "--shards", "8", "--experts", "512", "--copies", "128"]
    assert mep.main(args + ["--algorithm", "lpt", "-o", str(out)]) == 0
    table = json.loads(out.read_text())
    mep.validate(table)
    assert mep.main(["--check", str(out)]) == 0
