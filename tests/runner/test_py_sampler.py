# SPDX-License-Identifier: Apache-2.0
import threading
import time

from vllm_torchtpu.runner import py_sampler


def _busy_leaf(stop):
    while not stop.is_set():
        sum(range(1000))


def test_sampler_attributes_samples_to_the_busy_function(monkeypatch):
    logged = []
    monkeypatch.setattr(py_sampler.logger, "info", lambda msg, *a: logged.append(msg % a))
    stop = threading.Event()
    worker = threading.Thread(target=_busy_leaf, args=(stop,))
    worker.start()
    try:
        py_sampler._run(worker.ident, 0.3, 0.002, rank=0)
    finally:
        stop.set()
        worker.join()
    text = "\n".join(logged)
    assert "PYSAMPLER rank=0" in text and "_busy_leaf" in text
