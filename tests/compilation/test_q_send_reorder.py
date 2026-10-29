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
"""CPU check of the DSv4 q-send post-scheduler pass on stand-in instructions.

The instructions mimic what the pass sees after XLA's scheduler: Pallas custom
calls named ``pallas_call.N`` whose kernel name is only in their op_name
metadata, fresh wrapper objects per ``operands()`` call, and get-tuple-elements
that carry the custom call's op_name.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

_MODULE = (
    pathlib.Path(__file__).parents[2]
    / "src/vllm_torchtpu/compilation/q_send_reorder.py"
)


@pytest.fixture
def qsr(monkeypatch):
    """Load the pass by path with a stand-in logger (no package imports)."""
    logger_module = types.ModuleType("vllm_torchtpu.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *a: None, warning=lambda *a: None
    )
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.logger", logger_module)
    spec = importlib.util.spec_from_file_location("q_send_reorder_under_test", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Inst:
    def __init__(self, name, operands=(), kernel=None, custom_call=False):
        self.name, self._operands = name, list(operands)
        self.kernel, self.custom_call = kernel, custom_call

    def operands(self):
        # A fresh wrapper per call, like the HLO API: identity must be by name.
        return [
            _Inst(o.name, o._operands, o.kernel, o.custom_call) for o in self._operands
        ]

    def to_string(self):
        op = "custom-call(" if self.custom_call else "get-tuple-element("
        meta = (
            f', metadata={{op_name="jit(fn)/{self.kernel}/pallas_call"}}'
            if self.kernel
            else ""
        )
        return f"%{self.name} = bf16[8] {op}...){meta}"


def _call(name, kernel, *operands):
    return _Inst(name, operands, kernel, custom_call=True)


def _layer(kind, tag, x):
    """One layer as XLA schedules it: KV kernels first, then the q path."""
    seq = []
    if kind == "csa":
        rq = _call(f"pallas_call.rq{tag}", "rope_quant", x)
        comp = _call(f"pallas_call.c{tag}", "proj_and_save_state", x)
        idx = _call(f"pallas_call.i{tag}", "StreamIdxTC-p-bq_7", rq)
        seq += [rq, comp, idx]
    elif kind == "hca":
        comp = _call(f"pallas_call.c{tag}", "proj_and_save_state", x)
        store = _call(f"pallas_call.s{tag}", "compress_norm_rope_store", comp)
        seq += [comp, store]
    gather = _Inst(f"all-gather.{tag}", [x])
    matmul = _Inst(f"fusion.{tag}", [gather])
    start = _call(f"pallas_call.st{tag}", "q_send_start", matmul)
    gte = _Inst(f"get-tuple-element.{tag}", [start], kernel="q_send_start")
    wait = _call(f"pallas_call.w{tag}", "q_send_wait", gte)
    seq += [gather, matmul, start, gte, wait]
    return seq


def _labels(qsr, seq):
    return [qsr._label(i) if i.kernel else i.name.split(".")[0] for i in seq]


def test_hca_start_moves_before_the_compressor(qsr):
    x = _Inst("x")
    new, moved, layers = qsr.reorder_sequence([x, *_layer("hca", 0, x)])
    assert (moved, layers) == (3, 1)  # gather, matmul, start
    assert _labels(qsr, new) == [
        "x",
        "all-gather",
        "fusion",
        "q_send_start",
        "proj_and_save_state",
        "compress_norm_rope_store",
        "q_send_start",
        "q_send_wait",
    ]


def test_csa_start_moves_before_the_indexer_only(qsr):
    x = _Inst("x")
    new, moved, layers = qsr.reorder_sequence([x, *_layer("csa", 0, x)])
    assert (moved, layers) == (3, 1)
    # rope_quant and the compressor stay ahead; the start lands before StreamIdx.
    assert _labels(qsr, new) == [
        "x",
        "rope_quant",
        "proj_and_save_state",
        "all-gather",
        "fusion",
        "q_send_start",
        "StreamIdxTC-p-bq_7",
        "q_send_start",
        "q_send_wait",
    ]


def test_swa_only_layer_is_left_alone(qsr):
    x = _Inst("x")
    seq = [x, *_layer("swa", 0, x)]
    assert qsr.reorder_sequence(seq) == (seq, 0, 0)


def test_start_that_depends_on_the_kernel_is_not_moved(qsr):
    x = _Inst("x")
    comp = _call("pallas_call.c", "proj_and_save_state", x)
    matmul = _Inst("fusion.0", [comp])  # q path reads the compressor's output
    start = _call("pallas_call.st", "q_send_start", matmul)
    wait = _call("pallas_call.w", "q_send_wait", start)
    seq = [x, comp, matmul, start, wait]
    assert qsr.reorder_sequence(seq) == (seq, 0, 0)


def test_each_layer_moves_within_its_own_window(qsr):
    x = _Inst("x")
    first = _layer("hca", 0, x)
    second = _layer("csa", 1, first[-1])  # next layer reads this layer's output
    new, moved, layers = qsr.reorder_sequence([x, *first, *second])
    assert layers == 2
    labels = _labels(qsr, new)
    waits = [i for i, n in enumerate(labels) if n == "q_send_wait"]
    starts = [i for i, n in enumerate(labels) if n == "q_send_start"][::2]
    # start -> that layer's KV kernels -> wait, for both layers.
    assert labels[starts[0] + 1] == "proj_and_save_state" and starts[0] < waits[0]
    assert (
        labels[starts[1] + 1] == "StreamIdxTC-p-bq_7"
        and waits[0] < starts[1] < waits[1]
    )


class _Log:
    def __init__(self):
        self.info, self.warning = [], []
        self.as_logger = types.SimpleNamespace(
            info=lambda *a: self.info.append(a),
            warning=lambda *a: self.warning.append(a),
        )


def test_count_starts_counts_one_per_layer(qsr):
    x = _Inst("param.0")
    seq = _layer("csa", 0, x) + _layer("hca", 1, x) + _layer("swa", 2, x)
    assert qsr.count_starts(seq) == 3


def test_warns_when_starts_are_found_but_none_moved(qsr, monkeypatch):
    log = _Log()
    monkeypatch.setattr(qsr, "logger", log.as_logger)
    qsr._report(found=3, moved=0, layers=0)
    assert len(log.warning) == 1 and "moved none" in log.warning[0][0]
    qsr._report(found=3, moved=8, layers=2)
    assert len(log.info) == 1 and len(log.warning) == 1
    qsr._report(found=0, moved=0, layers=0)  # no q send in this module: silent
    assert len(log.info) == 1 and len(log.warning) == 1


def test_register_once_survives_a_missing_pass_api(qsr, monkeypatch):
    log = _Log()
    monkeypatch.setattr(qsr, "logger", log.as_logger)
    # A jax without register_hlo_module_transformation.
    monkeypatch.setitem(sys.modules, "jax.extend.xla", types.ModuleType("xla"))
    qsr.register_once()  # must not raise
    assert len(log.warning) == 1 and "could not register" in log.warning[0][0]
    qsr.register_once()  # tried once per process: no second warning
    assert len(log.warning) == 1
