# Copyright 2025 Google LLC
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
"""Shared pytest configuration for all tests."""

import os
import sys

# Keep TP collectives compiled in-graph (no graph break) for fullgraph TP.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Forking after TorchTPU/JAX/PyTorch background threads have started can
# deadlock vLLM engine or worker subprocess startup. Set this before test
# modules import vLLM so every LLM test uses a clean spawned process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Disable vLLM background usage telemetry in tests to prevent background threads
# and IPC sockets from hanging multi-process worker teardown.
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

import pytest
import torch


def pytest_collection_modifyitems(items):
    """Run whole-slice worker-spawning tests before in-process TPU tests.

    Once the pytest process itself initializes the TPU runtime (any test that
    runs jax/torch_tpu in-process), a test that later spawns a fresh
    ``torch.distributed.run`` worker group can no longer bootstrap the slice:
    the workers fail with "PjRtClient is not initialized" and libtpu may tear
    down the whole session with SLICE_FAILURE_SW_INJECT_ERROR. libtpu exposes
    no release API, so the only safe ordering is spawn-first.

    Within each group, files run in path order, whatever order they were
    passed in. Tests that start an engine in a subprocess (tests/entrypoints)
    need the TPU free, so they must run before the in-process tests under
    tests/kernels and tests/layers, and a sharded run receives its files in
    the planner's order. The sort is stable, so tests within a file keep
    their order.
    """
    items.sort(
        key=lambda item: (
            0 if item.get_closest_marker("spawns_tpu_workers") else 1,
            item.nodeid.split("::")[0],
        )
    )


def pytest_addoption(parser):
    """Add --use-tpu command line option."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=True,
        help="Run tests on TPU device instead of CPU",
    )


@pytest.fixture(autouse=True)
def _reset_warning_once():
    """`warning_once` dedups by message for the life of the process, so without
    this a test passes or fails depending on what ran before it."""
    try:
        import vllm.logger
    except ImportError:
        yield
        return

    vllm.logger._print_warning_once.cache_clear()
    yield
    vllm.logger._print_warning_once.cache_clear()


@pytest.fixture(autouse=True)
def _empty_shared_mamba_ops():
    """Isolate the shared recurrent `pallas.jax_op` cache across tests."""
    mod_name = "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils"
    mod = sys.modules.get(mod_name)
    if mod is not None:
        mod._shared_mamba_ops.clear()
    yield
    mod_after = sys.modules.get(mod_name)
    if mod_after is not None:
        mod_after._shared_mamba_ops.clear()


@pytest.fixture(autouse=True)
def _reset_ep_device_placement():
    """Any compile hash marks the EP placement key as read for the rest of the
    process, and `ep_device_ids` then raises on a placement it has not seen.
    Start every test with no placement recorded and the key unread.

    Resets directly instead of through `monkeypatch`: an autouse fixture that
    requests `monkeypatch` sets it up before every module fixture, so its undo
    would run after their teardown, while their patches are still in place."""

    def reset():
        ep_mesh = sys.modules.get("vllm_torchtpu.distributed.ep_mesh")
        if ep_mesh is not None:
            ep_mesh._GATHERED_DEVICE_IDS.clear()
            ep_mesh._PLACEMENT_KEY_READ = False

    reset()
    yield
    reset()


@pytest.fixture
def raiden_pd_env(monkeypatch):
    """Isolate Raiden P/D env vars and enable GLM admission by default.

    Defaults to ``TPU_RAIDEN_GLM_ADMISSION="1"`` so minimal single-chip unit
    tests pass startup validation without needing a multi-chip Qwen3.5
    topology. Returns a helper ``configure(**overrides)`` that clears all
    Raiden P/D and removed legacy KV env vars (plus cached ``envs``
    attributes) and applies ``overrides`` on top of
    ``TPU_USE_RAIDEN_KV_CACHE_MANAGER="1"``. Pass ``None`` as a value to leave
    a variable unset.
    """
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.distributed.kv_transfer.raiden_pd_validation import (
        _REMOVED_KV_TRANSFER_ENVS,
        RAIDEN_PD_ADMISSION_FLAGS,
    )

    managed = (
        "TPU_KV_RESHARD_TRANSPORT",
        "TPU_USE_RAIDEN_KV_CACHE_MANAGER",
        "TPU_RAIDEN_POOL_TAGS_PER_LAYER",
        *RAIDEN_PD_ADMISSION_FLAGS,
        *_REMOVED_KV_TRANSFER_ENVS,
    )

    def _configure(**overrides: str | None) -> None:
        for name in managed:
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delitem(vars(tpu_envs), name, raising=False)
        env: dict[str, str | None] = {
            "TPU_USE_RAIDEN_KV_CACHE_MANAGER": "1",
            **overrides,
        }
        for name, value in env.items():
            monkeypatch.delitem(vars(tpu_envs), name, raising=False)
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    _configure(TPU_RAIDEN_GLM_ADMISSION="1")
    return _configure


@pytest.fixture
def vllm_config_context():
    """Run backend helpers with the same resolved layout as worker startup."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig()
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.fixture(scope="module")
def dist_ctx():
    """Single-rank gloo world and model-parallel groups for one test module.

    Destroys only the groups this fixture created, so later test modules
    start without a default process group. Groups that already existed when
    the fixture ran stay in place.

    Defined here rather than in tests/models/vllm/conftest.py: pytest 9.1 drops
    a subdirectory conftest's fixtures when the file arguments go back into that
    directory after a file directly under tests/, and sharded CI passes files in
    any order.
    """
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    owns_world = not dist.is_initialized()
    owns_model_parallel = False
    if owns_world:
        import portpicker

        port = portpicker.pick_unused_port()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    try:
        owns_model_parallel = not model_parallel_is_initialized()
        # GroupCoordinator construction consults the current vLLM config.
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)
        yield
    finally:
        if owns_model_parallel:
            destroy_model_parallel()
        if owns_world:
            destroy_distributed_environment()


@pytest.fixture
def cpu_model_tests(monkeypatch):
    """Keep CPU-only model tests from opening the TPU.

    `import torch` loads torch_tpu through its backend entry point, so
    `torch.tpu` can be registered even on a host without chips. Each stub
    guards one way these tests would otherwise open the TPU:

    - `torch.accelerator`: Dynamo queries the current device while it traces
      CPU code; report no accelerator so it does not open a TPU client.
    - `torch.tpu.manual_seed_all`: `torch.manual_seed` also seeds every
      registered device backend, and torch_tpu's `manual_seed_all` opens the
      TPU.
    """
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
        monkeypatch.setattr(torch.accelerator, "device_count", lambda: 0)


@pytest.fixture
def make_fp8_experts(monkeypatch):
    """CPU `RoutedExperts` under `quant`, holding TP rank `tp_rank` of the experts."""
    # Here rather than in tests/layers/adapter/quantization/: see dist_ctx.

    def make(quant, shard_size, *, tp_size=2, tp_rank=1):
        from types import SimpleNamespace

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            MoEActivation,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            ExpertMapManager,
        )

        parallel = FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=tp_rank,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        config = FusedMoEConfig(
            num_experts=2,
            experts_per_token=1,
            hidden_dim=256,
            intermediate_size=tp_size * shard_size,
            num_local_experts=2,
            num_logical_experts=2,
            activation=MoEActivation.SILU,
            device=torch.device("cpu"),
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
        )
        expert_map = ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=1,
            global_num_experts=2,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.quantization.fp8."
            "get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        monkeypatch.setattr(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        quant.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
        )
        with set_current_vllm_config(quant.vllm_config):
            return RoutedExperts(
                layer_name="model.layers.0.mlp.experts",
                params_dtype=torch.bfloat16,
                moe_config=config,
                quant_config=quant,
                expert_map_manager=expert_map,
            )

    return make


@pytest.fixture
def device(request):
    """Get the device to run tests on (CPU or TPU).

    Usage:
        pytest tests/ -v              # Run on CPU
        pytest tests/ -v --use-tpu    # Run on TPU
    """
    use_tpu = request.config.getoption("--use-tpu")
    if use_tpu:
        try:
            import torch_tpu  # noqa: F401

            return torch.device("tpu")
        except ImportError:
            pytest.skip("torch_tpu not available")
        except Exception as e:
            pytest.fail(f"TPU requested but failed to initialize: {e}")
    return torch.device("cpu")


@pytest.fixture
def fp8_moe_runner_kit(monkeypatch):
    """CPU harness for vLLM `MoERunner`s over `VllmFp8MoEMethodTPU`.

    It patches the MoERunner module's forward-context, layer-registry and
    all-reduce hooks, the DP group, and `get_tpu_info` (a tpu7x chip). It
    returns a kit that builds the MoE config, quant method, stand-in routed
    experts and runner. Shared by the fused MoE decode and MoE TP decode tests,
    so a vLLM `MoERunner` change needs one edit here.
    """
    from types import SimpleNamespace

    import vllm.distributed.parallel_state as parallel_state
    from jax.experimental.pallas import tpu as pltpu
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner import (
        moe_runner as moe_runner_mod,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    class _StandInRoutedExperts(torch.nn.Module):
        forward_monolithic = RoutedExperts.forward_monolithic

        def __init__(
            self,
            moe_config,
            quant_method,
            *,
            local_experts,
            global_experts,
            hidden,
            inter,
            scoring_func,
            use_grouped_topk,
            random_weights,
        ):
            super().__init__()
            self.moe_config = moe_config
            self.quant_method = quant_method
            self.global_num_experts = global_experts
            self.scoring_func = scoring_func
            self.renormalize = True
            self.use_grouped_topk = use_grouped_topk
            self.num_expert_group = 1
            self.topk_group = 1
            self.custom_routing_function = None
            self.routed_scaling_factor = 1.0
            self.expert_placement_strategy = "linear"
            f8 = torch.float8_e4m3fn

            def param(t):
                return torch.nn.Parameter(t, requires_grad=False)

            e = local_experts
            if random_weights:
                self.w13_weight = param(torch.randn(e, hidden, 2 * inter).to(f8))
                self.w2_weight = param(torch.randn(e, inter, hidden).to(f8))
                self.w13_weight_scale_inv = param(torch.rand(e, 1, 1, 2 * inter))
                self.w2_weight_scale_inv = param(torch.rand(e, 1, 1, hidden))
                self.e_score_correction_bias = param(torch.rand(global_experts) * 0.1)
            else:
                self.w13_weight = param(torch.zeros((e, hidden, 2 * inter), dtype=f8))
                self.w2_weight = param(torch.zeros((e, inter, hidden), dtype=f8))
                self.w13_weight_scale_inv = param(torch.ones((e, 1, 1, 2 * inter)))
                self.w2_weight_scale_inv = param(torch.ones((e, 1, 1, hidden)))
                self.e_score_correction_bias = None
            moe_routing.register_experts_start_buffer(self, device=torch.device("cpu"))

        def _ensure_moe_quant_config_init(self):
            pass

    class Kit:
        def __init__(self):
            self.layers = {}
            self.all_reduces = 0

        def all_reduce(self, t):
            self.all_reduces += 1
            return t

        @staticmethod
        def moe_config(
            *, use_ep, n_ranks, topk, hidden, in_dtype=None, skip_final_all_reduce=False
        ):
            par = SimpleNamespace(
                use_ep=use_ep,
                ep_size=n_ranks if use_ep else 1,
                ep_rank=0,
                tp_size=1 if use_ep else n_ranks,
                dp_size=1,
                pcp_size=1,
                enable_eplb=False,
                is_sequence_parallel=False,
                all2all_backend="allgather_reducescatter",
                use_fi_nvl_two_sided_kernels=False,
                use_all2all_kernels=False,
            )
            return SimpleNamespace(
                moe_parallel_config=par,
                experts_per_token=topk,
                in_dtype=in_dtype,
                is_sequence_parallel=False,
                skip_final_all_reduce=skip_final_all_reduce,
                tp_size=par.tp_size,
                ep_size=par.ep_size,
                ep_rank=0,
                dp_size=1,
                pcp_size=1,
                sp_size=1,
                hidden_dim=hidden,
                hidden_dim_unpadded=hidden,
            )

        @staticmethod
        def fp8_method(moe_config):
            qm = VllmFp8MoEMethodTPU.__new__(VllmFp8MoEMethodTPU)
            FusedMoEMethodBase.__init__(qm, moe_config)
            qm._tpu_activation_str = "silu"
            return qm

        @staticmethod
        def routed_experts(moe_config, quant_method, **kw):
            return _StandInRoutedExperts(moe_config, quant_method, **kw)

        def runner(
            self,
            name,
            moe_config,
            routed_experts,
            *,
            gate=None,
            shared_experts=None,
            routed_scaling_factor=1.0,
        ):
            runner = MoERunner(
                layer_name=name,
                moe_config=moe_config,
                router=SimpleNamespace(prepare_monolithic_logits=lambda logits: logits),
                routed_experts=routed_experts,
                gate=gate,
                shared_experts=shared_experts,
                routed_scaling_factor=routed_scaling_factor,
            )
            self.layers[name] = runner
            return runner

    kit = Kit()
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )
    monkeypatch.setattr(moe_runner_mod, "get_layer_from_name", kit.layers.__getitem__)
    monkeypatch.setattr(
        moe_runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(moe_runner_mod, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        moe_runner_mod, "tensor_model_parallel_all_reduce", kit.all_reduce
    )
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    return kit


def pytest_collection_finish(session):
    """Fail fast if collecting the suite claimed the accelerator.

    libtpu is process-exclusive. A PJRT client built in the pytest PARENT locks
    out every child, and vLLM's engine and workers run in spawned children (see
    VLLM_WORKER_MULTIPROC_METHOD above), so every engine-starting test in the
    run then dies with "pjrt client not initialized" -- far from the module
    that actually claimed the device.

    Collection is import-time, so anything evaluated at module scope counts:
    the usual culprit is a `pytest.mark.skipif(jax.device_count() < N)`, which
    builds a client to answer and does so even when the marker goes on to
    deselect every test in the file. Put that check in a fixture instead; a
    fixture runs at setup, after deselection.

    Checked here rather than left to CI because the failure is invisible to any
    run that does not collect the offending file and an engine test together.
    """
    if "jax" not in sys.modules:
        return
    from jax._src import xla_bridge

    if not xla_bridge.backends_are_initialized():
        return
    raise pytest.UsageError(
        "A JAX backend was initialized during test collection, which claims "
        "the TPU for this process and makes every spawned engine/worker child "
        "fail with 'pjrt client not initialized'. Some test module touches the "
        "accelerator at import time -- most often a module-level "
        "`jax.device_count()` in a `skipif`. Move it into a fixture. To find "
        "the module: bisect with `pytest --collect-only <subset>` and check "
        "`jax._src.xla_bridge.backends_are_initialized()` after each."
    )
