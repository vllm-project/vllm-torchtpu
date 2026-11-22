# SPDX-License-Identifier: Apache-2.0
"""A new request built in the engine core's waits is the stock request.

See `vllm_torchtpu.core.preproc_in_wait` (q38 SV-EC PREPROC). With the key off
nothing is patched. With it on, a request built on the input thread has the
same block hashes and the same fields as vLLM's own build, it is built only
while the window is open (until its wait budget runs out), it reaches the
queue once, complete and in order, the main thread keeps the stock path, and
the window opens only in the main thread's long waits and while it blocks on
an empty input queue.
"""

import ast
import inspect
import pathlib
import queue
import random
import threading
import time
import types

import pytest
from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore, EngineCoreProc

from vllm_torchtpu import envs
from vllm_torchtpu.core import preproc_in_wait as piw

STOCK = inspect.unwrap(EngineCore.preprocess_add_request)
KEYS = ("VLLM_TPU_PREPROC_IN_WAIT", "VLLM_TPU_PREPROC_IN_WAIT_AB_MS")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 30)
    monkeypatch.setattr(piw, "AB_MS", 0)
    monkeypatch.setattr(piw, "stats", {k: (None if k == "ab_window" else 0) for k in piw.stats})
    piw.WINDOW.clear()
    yield
    piw.WINDOW.set()  # releases any thread a failing test left waiting
    time.sleep(0.05)
    piw.WINDOW.clear()


def _hasher(block_size: int):
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)
    return get_request_block_hasher(block_size, fn)


class _Counting:
    def __init__(self, real, on_call=None):
        self.real, self.calls, self.on_call = real, 0, on_call

    def __call__(self, request):
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        return self.real(request)


def _ecr(n: int, seed: int = 0, salt: str | None = None) -> EngineCoreRequest:
    rng = random.Random(seed)
    return EngineCoreRequest(
        request_id=f"r{seed}-{n}",
        prompt_token_ids=[rng.randrange(0, 151_000) for _ in range(n)],
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        pooling_params=None,
        arrival_time=1234.5,
        lora_request=None,
        cache_salt=salt,
        data_parallel_rank=None,
    )


def _engine(hasher):
    return types.SimpleNamespace(
        mm_receiver_cache=None, structured_output_manager=None, request_block_hasher=hasher
    )


def _same_request(a, b) -> None:
    assert type(a) is type(b)
    da, db = vars(a), vars(b)
    assert da.keys() == db.keys()
    for k in da:
        x, y = da[k], db[k]
        if x is y:
            continue
        if type(x).__name__ == "ConstantList":
            assert list(x) == list(y), k
            continue
        assert x == y, k


def _in_thread(fn, *args):
    out = {}

    def run():
        out["value"] = fn(*args)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_key_off_patches_nothing(fresh, monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert envs.VLLM_TPU_PREPROC_IN_WAIT == 0 and envs.VLLM_TPU_PREPROC_IN_WAIT_AB_MS == 0
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 0)
    before = (EngineCore.preprocess_add_request, SpinCondition.wait,
              SpinCondition.record_read, EngineCoreProc.__init__)
    piw.patch_preproc_in_wait()
    after = (EngineCore.preprocess_add_request, SpinCondition.wait,
             SpinCondition.record_read, EngineCoreProc.__init__)
    assert all(x is y for x, y in zip(before, after))


def test_key_on_installs_once(fresh, monkeypatch):
    for owner, name in ((EngineCore, "preprocess_add_request"), (SpinCondition, "wait"),
                        (SpinCondition, "record_read"), (EngineCoreProc, "__init__")):
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    orig = EngineCore.preprocess_add_request
    piw.patch_preproc_in_wait()
    piw.patch_preproc_in_wait()
    for fn in (EngineCore.preprocess_add_request, SpinCondition.wait,
               SpinCondition.record_read, EngineCoreProc.__init__):
        assert getattr(fn, piw._PATCH_FLAG, False)
    assert EngineCore.preprocess_add_request.__wrapped__ is orig


def test_refused_with_the_spinloop_extension(fresh, monkeypatch):
    monkeypatch.setattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", True, raising=False)
    monkeypatch.setattr(EngineCore, "preprocess_add_request", EngineCore.__dict__["preprocess_add_request"])
    before = EngineCore.preprocess_add_request
    piw.patch_preproc_in_wait()
    assert EngineCore.preprocess_add_request is before


@pytest.mark.parametrize("block_size", [16, 128, 512])
@pytest.mark.parametrize("n", [0, 1, 127, 128, 129, 2048, 4096, 4097, 100_003])
@pytest.mark.parametrize("salt", [None, "salt"])
def test_gated_build_equals_the_stock_build(fresh, block_size, n, salt):
    real = _hasher(block_size)
    ecr = _ecr(n, seed=n + block_size, salt=salt)
    stock, wave = STOCK(_engine(real), ecr)
    piw.WINDOW.set()
    t, out = _in_thread(piw.make_preprocess(STOCK), _engine(real), ecr)
    t.join(30)
    assert not t.is_alive()
    req, wave2 = out["value"]
    assert wave2 == wave
    assert len(req.block_hashes) == n // block_size
    assert req.block_hashes == stock.block_hashes
    assert req._block_hasher is real
    _same_request(req, stock)


def test_build_waits_for_the_window_and_pauses_when_it_closes(fresh, monkeypatch):
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 60_000)
    real = _hasher(128)
    n = 100_003
    ecr = _ecr(n, seed=7)
    stock, _ = STOCK(_engine(real), ecr)

    def close_after_third(calls):
        if calls == 3:
            piw.WINDOW.clear()

    counting = _Counting(real, close_after_third)
    t, out = _in_thread(piw.make_preprocess(STOCK), _engine(counting), ecr)
    time.sleep(0.3)
    assert t.is_alive() and counting.calls == 0  # waits before building
    piw.WINDOW.set()
    deadline = time.time() + 10
    while counting.calls < 3 and time.time() < deadline:
        time.sleep(0.001)
    time.sleep(0.3)
    assert t.is_alive() and counting.calls == 3  # paused between slices
    piw.WINDOW.set()
    t.join(30)
    assert not t.is_alive()
    req, _ = out["value"]
    assert counting.calls == -(-n // piw.CHUNK_TOKENS)
    assert req.block_hashes == stock.block_hashes
    assert req._block_hasher is counting
    assert piw.stats["gated"] == 1 and piw.stats["waits"] >= 2


def test_wait_budget_bounds_the_deferral(fresh, monkeypatch):
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 50)
    real = _hasher(128)
    ecr = _ecr(20_000, seed=3)
    stock, _ = STOCK(_engine(real), ecr)
    t0 = time.monotonic()
    t, out = _in_thread(piw.make_preprocess(STOCK), _engine(real), ecr)
    t.join(10)
    elapsed = time.monotonic() - t0
    assert not t.is_alive()
    assert 0.045 <= elapsed < 5
    req, _ = out["value"]
    assert req.block_hashes == stock.block_hashes
    assert piw.stats["budget_out"] == 1


def test_main_thread_keeps_the_stock_path(fresh, monkeypatch):
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 10_000)
    real = _hasher(128)
    counting = _Counting(real)
    ecr = _ecr(100_003, seed=5)
    t0 = time.monotonic()
    req, _ = piw.make_preprocess(STOCK)(_engine(counting), ecr)
    assert time.monotonic() - t0 < 2
    assert counting.calls == 1
    assert piw.stats["gated"] == 0
    assert req.block_hashes == STOCK(_engine(real), ecr)[0].block_hashes


def test_window_opens_in_long_main_thread_waits_and_closes_on_read(fresh, monkeypatch):
    t0 = 10_000_000  # monotonic_ns is never 0 (0 marks "no wait running")
    now = [t0]
    monkeypatch.setattr(piw.time, "monotonic_ns", lambda: now[0])
    cond = types.SimpleNamespace()
    piw.note_wait(cond)
    assert not piw.WINDOW.is_set()
    now[0] = t0 + 400_000
    piw.note_wait(cond)
    assert not piw.WINDOW.is_set()
    now[0] = t0 + 600_000
    piw.note_wait(cond)
    assert piw.WINDOW.is_set()
    piw.note_read(cond)
    assert not piw.WINDOW.is_set()
    now[0] = t0 + 700_000
    piw.note_wait(cond)
    now[0] = t0 + 1_000_000
    piw.note_wait(cond)
    assert not piw.WINDOW.is_set()  # a new wait starts its own clock
    other = types.SimpleNamespace()

    def off_main():
        piw.note_wait(other)
        now[0] = t0 + 9_000_000
        piw.note_wait(other)

    t = threading.Thread(target=off_main)
    t.start()
    t.join()
    assert not piw.WINDOW.is_set()
    piw.WINDOW.set()
    t = threading.Thread(target=piw.note_read, args=(other,))
    t.start()
    t.join()
    assert piw.WINDOW.is_set()  # only the main thread closes it


def test_spin_condition_wrappers_note_first(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(SpinCondition, "wait", lambda self, timeout_ms=None: seen.append(("wait", timeout_ms)))
    monkeypatch.setattr(SpinCondition, "record_read", lambda self: seen.append("read"))
    monkeypatch.setattr(EngineCore, "preprocess_add_request", EngineCore.__dict__["preprocess_add_request"])
    monkeypatch.setattr(EngineCoreProc, "__init__", EngineCoreProc.__dict__["__init__"])
    monkeypatch.setattr(piw, "note_wait", lambda c: seen.append("note_wait"))
    monkeypatch.setattr(piw, "note_read", lambda c: seen.append("note_read"))
    piw.patch_preproc_in_wait()
    SpinCondition.wait(object(), 5)
    SpinCondition.record_read(object())
    assert seen == ["note_wait", ("wait", 5), "note_read", "read"]


def test_empty_input_queue_opens_the_window(fresh):
    q = queue.Queue()
    piw.wrap_input_queue(q)
    piw.wrap_input_queue(q)  # idempotent
    seen = {}

    def producer():
        deadline = time.time() + 5
        while not piw.WINDOW.is_set() and time.time() < deadline:
            time.sleep(0.001)
        seen["open"] = piw.WINDOW.is_set()
        q.put("x")

    t = threading.Thread(target=producer)
    t.start()
    assert q.get() == "x"
    t.join()
    assert seen["open"] is True
    assert not piw.WINDOW.is_set()
    q.put("y")
    assert q.get_nowait() == "y" and not piw.WINDOW.is_set()


def test_requests_reach_the_queue_once_complete_and_in_order(fresh, monkeypatch):
    monkeypatch.setattr(piw, "MAX_DEFER_MS", 60_000)
    real = _hasher(128)
    sizes = [random.Random(i).choice([0, 5, 128, 3000, 40_000, 100_003]) for i in range(24)]
    ecrs = [_ecr(n, seed=i) for i, n in enumerate(sizes)]
    q = queue.Queue()
    pre = piw.make_preprocess(STOCK)
    eng = _engine(real)
    done = threading.Event()

    def input_thread():
        for e in ecrs:
            req, _ = pre(eng, e)
            q.put((req, len(req.block_hashes)))
        done.set()

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()
    deadline = time.time() + 120
    while not done.is_set() and time.time() < deadline:  # the main thread's waits open and close
        piw.WINDOW.set()
        time.sleep(0.002)
        piw.WINDOW.clear()
        time.sleep(0.001)
    piw.WINDOW.set()
    t.join(30)
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    assert [r.request_id for r, _ in got] == [e.request_id for e in ecrs]
    for (req, nh_at_put), e, n in zip(got, ecrs, sizes):
        assert nh_at_put == n // 128
        assert req.block_hashes == STOCK(_engine(real), e)[0].block_hashes
    assert piw.stats["gated"] == len(ecrs)


def test_ab_windows_follow_the_wall_clock(fresh, monkeypatch):
    monkeypatch.setattr(piw, "AB_MS", 1000)
    assert piw.ab_on(0) and not piw.ab_on(1_500_000_000) and piw.ab_on(2_100_000_000)
    monkeypatch.setattr(piw.time, "time_ns", lambda: 1_500_000_000)
    real = _hasher(128)
    counting = _Counting(real)
    ecr = _ecr(50_000, seed=11)
    piw.WINDOW.set()
    t, out = _in_thread(piw.make_preprocess(STOCK), _engine(counting), ecr)
    t.join(30)
    req, _ = out["value"]
    assert counting.calls == 1  # odd window: the stock build
    assert piw.stats["stock_ab"] == 1 and piw.stats["gated"] == 0
    assert req._block_hasher is counting


def test_keys_are_registered_and_out_of_the_compile_key():
    assert all(k in envs.environment_variables for k in KEYS)
    src = pathlib.Path(piw.__file__).parents[1] / "compilation" / "tpu_compiler.py"
    tree = ast.parse(src.read_text())
    ignored = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TPU_COMPILE_ENV_IGNORED" for t in node.targets
        ):
            ignored = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert ignored is not None and all(k in ignored for k in KEYS)


def test_registry_installs_it_last_at_engine_core():
    from vllm_torchtpu import patch_registry

    targets = [p.target for p in patch_registry.PATCHES]
    t = "vllm_torchtpu.core.preproc_in_wait:patch_preproc_in_wait"
    assert t in targets
    p = patch_registry.PATCHES[targets.index(t)]
    assert p.stages == ("engine_core",)
    assert targets.index(t) > targets.index(
        "vllm_torchtpu.core.prompt_token_array:patch_prompt_token_array"
    )
