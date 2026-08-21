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
"""VLLM_XLA_CHECK_RECOMPILATION should describe the graphs it counted.

The counter alone says four programs appeared; it does not say which op keeps
producing fresh shapes, which is the only thing that makes a recompilation
actionable.

Identifying them is the hard part. Entries carry no id, and the list is not
append-only: on a live Kimi K3 run its tail held boot-time programs with
``read_count`` up to 92 while the counter claimed 90 were new. What an entry
does carry is ``last_read``/``read_count``, so "first read since the previous
report" is the available definition, and that is what these tests pin.
"""

import contextlib
import datetime as dt
import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

T0 = dt.datetime(2026, 8, 20, 17, 40, 0)


def entry(seconds_after_t0, read_count=1, compile_secs=0.0):
    return SimpleNamespace(
        last_read=T0 + dt.timedelta(seconds=seconds_after_t0),
        read_count=read_count,
        compilation_duration=dt.timedelta(seconds=compile_secs),
    )


def make_runner():
    # Deliberately does not set _xla_graphs_checked_at: the first report runs
    # from inside __init__, so it has to hold before anything assigns it.
    runner = object.__new__(TPUModelRunner)
    runner.check_recompilation = True
    runner.enforce_eager = False
    runner.num_xla_graphs = 0
    return runner


def set_cache(monkeypatch, entries):
    stats = SimpleNamespace(per_entry_stats=list(entries),
                            num_cache_reqs=len(entries),
                            num_cache_hits=0)
    monkeypatch.setattr(torch.tpu, "_get_cache_stats", lambda: stats)


@contextlib.contextmanager
def captured_report(logger_name: str = "vllm"):
    """Collect the INFO lines emitted under `logger_name`.

    Deliberately not pytest's `caplog`: vLLM's dictConfig sets
    `propagate=False` on the `vllm` logger, so these records never reach the
    root logger where `caplog` installs its handler.
    """
    messages: list[str] = []
    handler = logging.Handler(logging.INFO)
    handler.emit = lambda record: messages.append(record.getMessage())
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    # Another test may have raised the level or called logging.disable();
    # either drops the record before any handler runs.
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


def capture(runner, case):
    """The report one check emitted, as one block of text."""
    with captured_report() as messages:
        runner._update_num_xla_graphs(case)
    return "\n".join(messages)


def test_the_first_report_runs_before_anything_has_been_checked(monkeypatch):
    """The boot crash this report first shipped with.

    ``__init__`` calls the report before it could assign any state of its own,
    so the "when did I last check" marker has to already hold a value. It did
    not, and every process with VLLM_XLA_CHECK_RECOMPILATION=1 died in
    ``init_device`` with an AttributeError.
    """
    runner = make_runner()
    set_cache(monkeypatch, [entry(0), entry(1)])

    text = capture(runner, "init")

    assert "new: 2" in text
    assert runner.num_xla_graphs == 2


def test_a_runtime_compile_is_described(monkeypatch):
    runner = make_runner()
    boot = [entry(0, read_count=9), entry(1, read_count=4)]
    set_cache(monkeypatch, boot)
    capture(runner, "init")

    # One program appears once traffic is running, and the boot entries get
    # read again in the same step.
    boot[0].read_count = 10
    boot[0].last_read = T0 + dt.timedelta(seconds=90)
    set_cache(monkeypatch, boot + [entry(91, compile_secs=1.25)])
    text = capture(runner, "decoding_step")

    assert "new: 1" in text
    assert "1 first read since the last check, 1.250s compiling" in text
    assert "graph 1/1: " in text
    assert "read_count=1" in text
    assert runner.num_xla_graphs == 3


def test_a_reread_boot_graph_is_not_reported_as_fresh(monkeypatch):
    """The failure that made the first version of this report useless.

    The list is not append-only, so its tail is full of boot programs. Those
    have been read before, which is what rules them out.
    """
    runner = make_runner()
    boot = [entry(0, read_count=92), entry(1, read_count=92)]
    set_cache(monkeypatch, boot)
    capture(runner, "init")

    for e in boot:
        e.read_count += 1
        e.last_read = T0 + dt.timedelta(seconds=120)
    set_cache(monkeypatch, boot)
    text = capture(runner, "decoding_step")

    assert "No new compiled graphs" in text
    assert "graph 1/" not in text


def test_a_cache_hit_is_counted_but_costs_no_compile_time(monkeypatch):
    """An entry with a zero duration was loaded, not compiled here."""
    runner = make_runner()
    set_cache(monkeypatch, [entry(0, read_count=3)])
    capture(runner, "init")

    set_cache(monkeypatch,
              [entry(0, read_count=3),
               entry(60, compile_secs=0.0)])
    text = capture(runner, "decoding_step")

    assert "new: 1" in text
    assert "1 first read since the last check, 0.000s compiling" in text


def test_the_costliest_compiles_are_described_first(monkeypatch):
    runner = make_runner()
    set_cache(monkeypatch, [entry(0, read_count=2)])
    capture(runner, "init")

    set_cache(monkeypatch, [
        entry(0, read_count=2),
        entry(10, compile_secs=0.1),
        entry(11, compile_secs=9.5),
        entry(12, compile_secs=2.0),
    ])
    text = capture(runner, "decoding_step")

    order = [
        float(line.split("compilation_duration=0:00:0")[1].split()[0])
        for line in text.splitlines() if "compilation_duration=" in line
    ]
    assert order == sorted(order, reverse=True), order


def test_long_reports_are_capped_but_say_so(monkeypatch):
    runner = make_runner()
    count = TPUModelRunner._MAX_DESCRIBED_GRAPHS + 7
    set_cache(monkeypatch, [entry(i) for i in range(count)])
    text = capture(runner, "init")

    assert text.count("  graph ") == TPUModelRunner._MAX_DESCRIBED_GRAPHS
    assert "... 7 more not shown" in text


def test_a_shrinking_cache_resynchronizes_instead_of_going_negative(
        monkeypatch):
    runner = make_runner()
    set_cache(monkeypatch, [entry(i, read_count=2) for i in range(5)])
    capture(runner, "init")

    set_cache(monkeypatch, [entry(0, read_count=2), entry(1, read_count=2)])
    text = capture(runner, "decoding_step")

    assert "shrank" in text
    assert "graph 1/" not in text
    assert runner.num_xla_graphs == 2


def test_describe_survives_entries_it_cannot_read():

    class Hostile:

        @property
        def explodes(self):
            raise RuntimeError("needs a live device")

        @property
        def enormous(self):
            return "x" * 500

        @property
        def multiline(self):
            return "first\nsecond"

    described = TPUModelRunner._describe_xla_graph(Hostile())

    assert "explodes" not in described
    assert "\n" not in described
    assert "enormous=" + "x" * 117 + "..." in described
    assert "multiline=first second" in described


def test_describe_falls_back_to_repr_for_fieldless_entries():
    described = TPUModelRunner._describe_xla_graph(object())
    assert described.startswith("<object object at")


def test_compile_seconds_tolerate_a_missing_field():
    assert TPUModelRunner._xla_graph_compile_secs(SimpleNamespace()) == 0.0
    assert TPUModelRunner._xla_graph_compile_secs(
        SimpleNamespace(compilation_duration=None)) == 0.0


@pytest.mark.parametrize("check,eager", [(False, False), (True, True)])
def test_disabled_paths_do_not_touch_the_cache(check, eager, monkeypatch):
    runner = make_runner()
    runner.check_recompilation = check
    runner.enforce_eager = eager

    def boom():
        raise AssertionError("_get_cache_stats must not be called")

    monkeypatch.setattr(torch.tpu, "_get_cache_stats", boom)
    runner._update_num_xla_graphs("init")
