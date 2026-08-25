# SPDX-License-Identifier: Apache-2.0
"""Per-token-bucket graph specialization (b/545443817).

vLLM's real compile pipeline (``@support_torch_compile`` -> ``VllmBackend`` ->
``PiecewiseBackend`` -> the real ``TpuCompilerAdaptor``) drives a fake torch_tpu
backend whose "executable" records the structure it was compiled from, so a test
can tell which trace's executable served a bucket.

Every case that compiles runs in a spawned child: driving torch.compile on a TPU
host makes PJRT claim the chip for the life of the process, and the pytest parent
holding it breaks every later test that starts a worker or an engine core. Same
reason, and same shape, as tests/distributed/tpu_test_utils.py.
"""

import dataclasses
import importlib
import multiprocessing
import os
import traceback
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy
import torch
import vllm
from torch import fx, nn
from torch.utils._sympy.value_ranges import ValueRanges
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationMode, DeviceConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.forward_context import set_forward_context

import vllm_torchtpu
import vllm_torchtpu.env_override
from vllm_torchtpu.compilation import shape_variants, tpu_compiler
from vllm_torchtpu.compilation.shape_variants import unsupported_reason

if not hasattr(vllm,
               "__version__"):  # namespace-package quirk in the CPU image
    import importlib.metadata
    vllm.__version__ = importlib.metadata.version("vllm")

HIDDEN = 8
MAX_TOKENS = 16384
BUCKETS = [16, 2048, 4096]

_lib = torch.library.Library("tpu_shape_variants_test", "FRAGMENT")
_lib.define("wave(Tensor x) -> Tensor")
_lib.impl("wave", lambda x: x + 1.0, "CPU")
torch.library.register_fake("tpu_shape_variants_test::wave",
                            lambda x: torch.empty_like(x))
WAVE = torch.ops.tpu_shape_variants_test.wave


@support_torch_compile(dynamic_arg_dims={"x": 0})
class WaveModel(nn.Module):
    """One opaque call per 2048 tokens -- the shape of commit 2e3e9a495."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves = (x.shape[0] + 2047) // 2048
        return torch.cat([WAVE(chunk) for chunk in torch.chunk(x, waves)])


@support_torch_compile(dynamic_arg_dims={"x": 0})
class TieredModel(nn.Module):
    """Three structures, so a graph that over-claims its buckets is caught."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        waves = 4 if tokens >= 8192 else (2 if tokens >= 4096 else 1)
        return torch.cat([WAVE(chunk) for chunk in torch.chunk(x, waves)])


@support_torch_compile(dynamic_arg_dims={"x": 0})
class FlatModel(nn.Module):
    """Same op, no dependence on the token count."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return WAVE(x)


@dataclasses.dataclass
class FakeExecutable:
    """Picklable stand-in for torch_tpu's compiled executable.

    A real one has its bucket's shapes baked in, so it is the giveaway when the
    wrong graph serves a bucket: it returns the rows it was compiled for.
    """

    rows: int
    waves: int

    def __call__(self, *args):
        CALLS.append((self.rows, self.waves))
        return (torch.zeros(self.rows, HIDDEN), )

    def serialize(self):
        """Stand in for the bundled AOTAutograd entry.

        A method rather than an attribute set per instance: the adaptor pickles
        whatever this returns, and a closure in the instance __dict__ would not
        survive that. The real entry bundles a wrapper around an executable;
        here the executable is the whole runnable, so it is its own entry.
        """
        return self


CALLS: list[tuple[int, int]] = []


class FakeTpuBackend:
    """torch_tpu's TpuBackend, minus torch_tpu. Everything above it is real.

    Mirrors the one part of the real contract the compiler adaptor depends on:
    the returned runnable carries ``serialize()``, which hands back a picklable
    bundled entry holding both the wrapper and the executable. The adaptor
    caches that entry and nothing else, so a fake without it silently disables
    the disk cache and every warm start recompiles.
    """

    def __init__(self) -> None:
        self.compiled: list[tuple[int, int]] = []  # (bucket, waves)

    def __call__(self, graph: fx.GraphModule, example_inputs):
        rows = max(t.shape[0] for t in example_inputs
                   if isinstance(t, torch.Tensor))
        waves = sum(1 for node in graph.graph.nodes
                    if "wave" in str(node.target))
        self.compiled.append((rows, waves))
        return FakeExecutable(rows, waves)


@dataclasses.dataclass
class Starts:
    """Traces vs artifact reloads.

    ``VllmBackend.__call__`` -- and its "Dynamo bytecode transform time" log
    line -- fire on a reload too, and vLLM's counters are zeroed by every
    ``reset_compile_wrapper``, so count the two entry points directly.
    """

    traced: int = 0
    loaded: int = 0


def count_starts() -> Starts:
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    counts = Starts()
    load, compile_ = (decorators._try_load_aot_compiled_fn,
                      TorchCompileWithNoGuardsWrapper.aot_compile)

    def counted_load(model, path):
        found = load(model, path)
        counts.loaded += found is not None
        return found

    def counted_compile(self, *args, **kwargs):
        counts.traced += 1
        return compile_(self, *args, **kwargs)

    decorators._try_load_aot_compiled_fn = counted_load
    TorchCompileWithNoGuardsWrapper.aot_compile = counted_compile
    return counts


def make_config(splitting_ops=None) -> VllmConfig:
    from vllm.platforms import current_platform

    # TpuPlatform.check_and_update_config() needs a fully built ModelConfig and
    # the cache-key hash reads it; this test only needs the compile pipeline.
    original = type(current_platform).check_and_update_config
    type(current_platform).check_and_update_config = classmethod(
        lambda cls, config: None)
    try:
        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(max_num_batched_tokens=MAX_TOKENS,
                                             max_model_len=MAX_TOKENS,
                                             is_encoder_decoder=False),
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend="",  # defers to TpuPlatform.get_compile_backend()
                compile_sizes=list(BUCKETS),
                splitting_ops=splitting_ops or [],
                cudagraph_capture_sizes=[],
            ),
        )
    finally:
        type(current_platform).check_and_update_config = original
    for field in dataclasses.fields(config.compilation_config.pass_config):
        if isinstance(
                getattr(config.compilation_config.pass_config, field.name),
                bool):
            setattr(config.compilation_config.pass_config, field.name, False)
    # The TPU platform patches this away: torch_tpu rejects SymInt shapes, so
    # only the exact buckets are ever compiled (vllm_torchtpu/__init__.py).
    type(config.compilation_config).get_compile_ranges = lambda self: []
    return config


def prepare(cache_root: str, splitting_ops=None):
    """What the pipeline needs, set up inside the child process."""
    os.environ.update(VLLM_CACHE_ROOT=cache_root,
                      VLLM_DISABLE_COMPILE_CACHE="0",
                      VLLM_USE_AOT_COMPILE="1",
                      VLLM_USE_MEGA_AOT_ARTIFACT="0")
    # The cache keys hash a ModelConfig these tests do not build.
    tpu_compiler.compute_tpu_compilation_hash = lambda config: "test"
    # TpuPlatform.check_and_update_config() installs this, and these tests skip
    # that hook; without it the AOT artifact path cannot tell traces apart.
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    backend = FakeTpuBackend()
    tpu_compiler._tpu_backend = backend
    # A FakeExecutable never went through torch, so it cannot come back through
    # torch's bundled-entry deserializer either; it is already the runnable.
    tpu_compiler._deserialize_entry = lambda entry: entry
    CALLS.clear()
    return make_config(splitting_ops), backend


# Keep a report under the pipe buffer: a larger one parks the queue's feeder
# thread until the parent reads it, and the parent is in join() by then.
MAX_REPORT = 32 * 1024


def _child(target, result):
    try:
        target()
    except BaseException:
        result.put(traceback.format_exc()[-MAX_REPORT:])
    else:
        result.put(None)


def in_child(target) -> None:
    """Run a compiling test body in a spawned process.

    Mirrors tests/distributed/tpu_test_utils.py, which is not importable from
    here (tests/ is not a package). ``target`` must be picklable.
    """
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_child, args=(target, result))
    process.start()
    try:
        # Half of CI's own pytest --timeout, so this cleanup still runs: its
        # alarm firing inside join() would leave the child alive and pytest
        # unable to exit. terminate() has to be reached on every path.
        process.join(300)
        if process.is_alive():
            raise AssertionError("isolated test timed out")
        if result.empty():
            raise AssertionError(
                f"isolated test crashed (exit {process.exitcode})")
        failure = result.get()
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    if failure is not None:
        raise AssertionError(f"isolated test failed:\n{failure}")


def build(cls, config: VllmConfig) -> nn.Module:
    with set_current_vllm_config(config):
        return cls(vllm_config=config, prefix="")


def run(model: nn.Module, config: VllmConfig, num_tokens: int):
    with set_forward_context({}, config):
        return model(torch.zeros(num_tokens, HIDDEN))


def warm(model, config, buckets=BUCKETS) -> int:
    """What TPUModelRunner._precompile_backbone does, minus the runner.

    Returns the number of extra traces the ladder needed.
    """
    retraces = 0
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        for num_tokens in buckets:
            if tracked.needs_retrace(num_tokens):
                shape_variants.retrace(model, config)
                retraces += 1
            run(model, config, num_tokens)
    return retraces


def _one_graph_when_shape_free(cache_root):
    config, backend = prepare(cache_root)
    model = build(FlatModel, config)
    assert warm(model, config) == 0
    assert backend.compiled == [(bucket, 1) for bucket in BUCKETS]


def test_one_trace_serves_every_bucket_when_shape_free(tmp_path):
    in_child(partial(_one_graph_when_shape_free, str(tmp_path)))


def _guards_say_which_buckets_a_trace_serves(cache_root):
    config, _ = prepare(cache_root)
    model = build(WaveModel, config)
    graphs = []
    original = tpu_compiler.TpuCompilerAdaptor.compile

    def capture(self, graph, *args, **kwargs):
        graphs.append(graph)
        return original(self, graph, *args, **kwargs)

    tpu_compiler.TpuCompilerAdaptor.compile = capture
    try:
        warm(model, config)
    finally:
        tpu_compiler.TpuCompilerAdaptor.compile = original
    env = shape_variants.trace_shape_env(graphs[0])
    assert [
        s for s in (16, 512, 2048, 2049, 4096)
        if unsupported_reason(env, s) is None
    ] == [16, 512, 2048]


def test_unsupported_reason_reads_the_traces_guards(tmp_path):
    in_child(partial(_guards_say_which_buckets_a_trace_serves, str(tmp_path)))


def _refused_outside_warmup(cache_root):
    config, _ = prepare(cache_root)
    model = build(WaveModel, config)
    try:
        with set_current_vllm_config(config):
            run(model, config, BUCKETS[0])
    except Exception as exc:  # Dynamo wraps whatever the backend raises
        assert "cannot be served by the FX graph" in str(exc)
    else:
        raise AssertionError("a bucket no graph can serve was compiled anyway")


def test_bucket_no_graph_can_serve_is_refused_outside_warmup(tmp_path):
    """A module nothing re-traces must fail loudly, not miscompile."""
    in_child(partial(_refused_outside_warmup, str(tmp_path)))


def _each_bucket_gets_its_own_graph(cache_root):
    config, backend = prepare(cache_root)
    model = build(WaveModel, config)
    warm(model, config)

    # Two traces, one executable per bucket, each with its own structure.
    assert backend.compiled == [(16, 1), (2048, 1), (4096, 2)]

    # The live graph is the last one traced; every bucket still runs the
    # executable built for it, through vLLM's own per-range dispatch.
    CALLS.clear()
    for num_tokens in (4096, 16, 2048, 4096):
        out = run(model, config, num_tokens)
        assert out.shape[0] == num_tokens
    assert CALLS == [(4096, 2), (16, 1), (2048, 1), (4096, 2)]


def test_each_bucket_gets_the_graph_traced_for_it(tmp_path):
    in_child(partial(_each_bucket_gets_its_own_graph, str(tmp_path)))


def _three_tiers_keep_their_structure(cache_root):
    buckets = [16, 2048, 4096, 8192]
    config, backend = prepare(cache_root)
    config.compilation_config.compile_sizes = buckets
    model = build(TieredModel, config)
    assert warm(model, config, buckets) == 2

    assert backend.compiled == [(16, 1), (2048, 1), (4096, 2), (8192, 4)]
    CALLS.clear()
    for num_tokens in buckets:
        assert run(model, config, num_tokens).shape[0] == num_tokens
    assert CALLS == [(16, 1), (2048, 1), (4096, 2), (8192, 4)]


def test_three_tiers_each_keep_their_own_structure(tmp_path):
    """A graph that over-claimed a bucket would serve it the wrong rows."""
    in_child(partial(_three_tiers_keep_their_structure, str(tmp_path)))


def _split_graph_is_refused(cache_root):
    config, _ = prepare(cache_root,
                        splitting_ops=["tpu_shape_variants_test::wave"])
    model = build(WaveModel, config)
    try:
        warm(model, config)
    except Exception as exc:
        assert "TPU_KERNEL_ITER_MODE" in str(exc)
    else:
        raise AssertionError("a split graph handed a bucket over anyway")


def test_split_graph_cannot_hand_buckets_between_traces(tmp_path):
    in_child(partial(_split_graph_is_refused, str(tmp_path)))


def _warm_start_reuses_the_cache(cache_root):
    config, backend = prepare(cache_root)
    starts = count_starts()
    assert warm(build(WaveModel, config), config) == 1
    assert backend.compiled == [(16, 1), (2048, 1), (4096, 2)]
    assert (starts.traced, starts.loaded) == (2, 0)  # one trace per group

    # Second start, same cache dir: nothing is compiled, and the ladder still
    # splits at the same bucket -- the first graph must refuse 4096 on the load
    # path too, or it would adopt an executable it cannot call (on TPU:
    # "Donation requested for invalid buffer", job j-14584b09).
    torch._dynamo.reset()
    backend.compiled.clear()
    model = build(WaveModel, config)
    starts.traced = starts.loaded = 0
    assert warm(model, config) == 1
    assert backend.compiled == []
    # Neither group traced: each came back from the artifact of its own ordinal.
    assert (starts.traced, starts.loaded) == (0, 2)
    CALLS.clear()
    for num_tokens in BUCKETS:
        assert run(model, config, num_tokens).shape[0] == num_tokens
    assert CALLS == [(16, 1), (2048, 1), (4096, 2)]


def test_warm_start_reuses_the_cache_without_compiling(tmp_path):
    in_child(partial(_warm_start_reuses_the_cache, str(tmp_path)))


def _session_does_not_outlive_warmup(cache_root):
    config, _ = prepare(cache_root)
    warm(build(FlatModel, config), config)
    assert shape_variants._session is None


def test_nothing_survives_warmup(tmp_path):
    in_child(partial(_session_does_not_outlive_warmup, str(tmp_path)))


def _refused_bucket_never_retraced_raises(cache_root):
    config, _ = prepare(cache_root)
    model = build(WaveModel, config)
    warm(model, config, buckets=[BUCKETS[0]])  # ladder stops before 4096
    with pytest.raises(shape_variants.ShapeSpecializationError,
                       match="never compiled"):
        run(model, config, BUCKETS[-1])  # the poisoned runnable is invoked


def test_a_refused_bucket_nobody_retraced_raises(tmp_path):
    """The closure left behind has to raise, not return something plausible."""
    in_child(partial(_refused_bucket_never_retraced_raises, str(tmp_path)))


def test_warmup_scope_survives_a_failure():
    """A failed warmup must not leave the session pinned for the next one."""
    with pytest.raises(RuntimeError):
        with shape_variants.warmup():
            raise RuntimeError("boom")
    assert shape_variants._session is None


def test_a_guard_the_range_cannot_express_still_refuses():
    """The guard list is read, not just var_to_range.

    Dynamo folds most guards into the ranges, so a test driving the real
    pipeline cannot tell the two apart; this one can.
    """
    symbol = sympy.Symbol("s0", positive=True, integer=True)
    env = SimpleNamespace(
        var_to_range={symbol: ValueRanges(2, 8192)},
        guards=[SimpleNamespace(expr=sympy.Eq(sympy.Mod(symbol, 3), 0))])

    assert unsupported_reason(env, 2049) is None  # 2049 % 3 == 0, holds
    assert unsupported_reason(env, 2048) is not None  # provably violated


def test_env_override_keeps_the_mega_artifact_off():
    """It is inductor-only on the load side; see the commit for why."""
    with patch.dict(os.environ, clear=False) as env:
        env.pop("VLLM_USE_MEGA_AOT_ARTIFACT", None)
        importlib.reload(vllm_torchtpu.env_override)

        assert os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"] == "0"
        # The write only means something while vLLM still reads this name.
        assert "VLLM_USE_MEGA_AOT_ARTIFACT" in vllm.envs.environment_variables
        assert vllm.envs.VLLM_USE_MEGA_AOT_ARTIFACT is False
