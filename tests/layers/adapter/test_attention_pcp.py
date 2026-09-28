import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.envs as vllm_envs
from jax.sharding import PartitionSpec
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

import vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter as pcp_adapter
from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout, flat_kernel_ids
from vllm_torchtpu.distributed import pallas_shapes
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl,
    _pallas_rpa_kernel_local,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config(pcp_size=2, interleave_size=4):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        ),
    )


def _impl(kv_cache_dtype="bfloat16", **kwargs):
    return PallasAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )


def _batched_impl(kv_cache_dtype="bfloat16", **kwargs):
    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )


def _layer(k_scale=0.0, v_scale=0.0):
    layer = MagicMock()
    layer._k_scale_float = k_scale
    layer._v_scale_float = v_scale
    return layer


def _tensors():
    query = torch.ones(3, 2, 128)
    key = torch.ones(3, 1, 128)
    value = torch.ones(3, 1, 128)
    kv_cache = torch.zeros(8, 16, 2, 1, 128)
    return query, key, value, kv_cache


def _metadata(*, pcp_streaming=False, block_tables=None):
    layout_kwargs = {}
    if pcp_streaming:
        layout_kwargs = {
            "sequence_layout_kind": SequenceLayoutKind.PARTIAL.value,
            "sequence_layout_protocol": "pcp_streaming",
        }
    return AttentionMetadata(
        input_positions=torch.arange(3, dtype=torch.int32),
        block_tables=(
            torch.zeros(2, 4, dtype=torch.int32)
            if block_tables is None
            else block_tables
        ),
        seq_lens=torch.tensor([3, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 3], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        **layout_kwargs,
    )


def _capture_kernel(monkeypatch):
    captured = {}

    def fake_build(
        self,
        q_scale,
        k_scale,
        v_scale,
        *,
        skip_kv_update=False,
        use_pcp_streaming,
        cp_kv_cache_interleave_size,
        block_major=False,
    ):
        captured["build"] = {
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "skip_kv_update": skip_kv_update,
            "use_pcp_streaming": use_pcp_streaming,
            "cp_kv_cache_interleave_size": cp_kv_cache_interleave_size,
            "block_major": block_major,
        }

        def fake_kernel(*args):
            captured["args"] = args
            return torch.full_like(args[1], 7.0)

        return fake_kernel

    monkeypatch.setattr(PallasAttentionBackendImpl, "_build_rpa_kernel", fake_build)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.synchronize_tensors",
        lambda *_args, **_kwargs: None,
    )
    return captured


def test_forward_routes_partial_layout_to_streaming_kernel(
    monkeypatch, vllm_config_context
):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        outputs = _impl().forward(_layer(), query, key, value, kv_cache, metadata)

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["cp_kv_cache_interleave_size"] == 4
    assert len(captured["args"]) == 9
    assert captured["args"][1].shape[0] == query.shape[0]
    assert captured["args"][8] is None
    assert torch.all(outputs == 7)


def test_forward_routes_fp8_partial_layout_with_kv_scales(
    monkeypatch, vllm_config_context
):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)
    layer = _layer(k_scale=0.125, v_scale=0.25)

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        outputs = _impl(kv_cache_dtype="fp8_e4m3").forward(
            layer, query, key, value, kv_cache, metadata
        )

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["q_scale"] is None
    assert captured["build"]["k_scale"] == 0.125
    assert captured["build"]["v_scale"] == 0.25
    assert captured["args"][2].dtype == torch.float8_e4m3fn
    assert captured["args"][3].dtype == torch.float8_e4m3fn
    assert torch.all(outputs == 7)


def test_forward_routes_pcp_decode_metadata_to_streaming_kernel(
    monkeypatch, vllm_config_context
):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        outputs = _impl().forward(_layer(), query, key, value, kv_cache, metadata)

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["cp_kv_cache_interleave_size"] == 4
    assert len(captured["args"]) == 9
    assert captured["args"][8] is None


@pytest.mark.parametrize(
    ("impl_kwargs", "expected_message"),
    [
        ({"sinks": torch.zeros(2)}, "attention sinks"),
        ({"logits_soft_cap": 30.0}, "logits soft cap"),
        ({"kv_sharing_target_layer_name": "shared"}, "skip_kv_update"),
    ],
)
def test_forward_rejects_unsupported_pcp_streaming_features(
    impl_kwargs, expected_message, vllm_config_context
):
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with (
        set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
        pytest.raises(NotImplementedError, match=expected_message),
    ):
        _impl(**impl_kwargs).forward(_layer(), query, key, value, kv_cache, metadata)


def test_initialize_kernel_rejects_unsupported_pcp_streaming_features(
    vllm_config_context,
):
    with (
        set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
        pytest.raises(NotImplementedError, match="logits soft cap"),
    ):
        _impl(logits_soft_cap=30.0).initialize_kernel(_layer())


def test_forward_keeps_non_pcp_batch_on_disabled_mode(monkeypatch, vllm_config_context):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata()

    with set_vllm_model_wrapper_context(
        mesh=_mesh(), vllm_config=_vllm_config(pcp_size=1)
    ):
        _impl().forward(_layer(), query, key, value, kv_cache, metadata)

    assert captured["build"]["use_pcp_streaming"] is False
    assert captured["build"]["cp_kv_cache_interleave_size"] == 0


def test_initialize_kernel_prebuilds_streaming_variant_for_native_pcp(
    monkeypatch, vllm_config_context
):
    calls = []

    def fake_build(
        self,
        q_scale,
        k_scale,
        v_scale,
        skip_kv_update=False,
        use_pcp_streaming=False,
        cp_kv_cache_interleave_size=0,
        block_major=False,
    ):
        calls.append((use_pcp_streaming, cp_kv_cache_interleave_size))
        return MagicMock()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_build_rpa_kernel", fake_build)

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        _impl().initialize_kernel(_layer())

    assert calls == [(True, 4)]


def test_initialize_kernel_marks_the_batched_layer_as_streaming_under_pcp(
    monkeypatch, vllm_config_context
):
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_build_rpa_kernel",
        lambda self, *args, **kwargs: MagicMock(),
    )
    impl = _batched_impl()

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        impl.initialize_kernel(_layer())

    assert not impl.runs_batched_rpa_schedule()


def test_no_kernel_reports_a_schedule_bound(vllm_config_context):
    """The batched kernel streams its schedule from an HBM table and exposes
    no capacity, so no layer asks the runner to bound its schedule."""
    assert not _batched_impl().runs_batched_rpa_schedule()
    assert not _impl().runs_batched_rpa_schedule()


def test_streaming_rpa_partition_specs_cover_pcp_and_tp_axes():
    assert (
        PartitionSpec("pcp", None, "tp"),
        PartitionSpec("pcp", "tp"),
        PartitionSpec("pcp", "tp"),
        PartitionSpec("pcp", "tp"),
        PartitionSpec(),
        PartitionSpec(),
        PartitionSpec(),
        PartitionSpec(),
    ) == pcp_adapter.PCP_STREAMING_RPA_INPUT_PARTITION_SPECS
    assert (
        PartitionSpec("pcp", None, "tp"),
        PartitionSpec("pcp", "tp"),
    ) == pcp_adapter.PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS


def test_streaming_rpa_mesh_requests_full_pcp_tp_grid(monkeypatch):
    captured = {}
    mesh = object()

    def fake_get_mesh(axis_name, *, tp_axis_name):
        captured["axis_names"] = (axis_name, tp_axis_name)
        return mesh

    monkeypatch.setattr(pcp_adapter, "get_or_create_pcp_mesh", fake_get_mesh)

    assert pcp_adapter.get_pcp_streaming_mesh() is mesh
    assert captured["axis_names"] == ("pcp", "tp")


def test_build_rpa_kernel_reuses_prebuilt_config_before_mesh_lookup(
    monkeypatch, vllm_config_context
):
    impl = _impl()
    fake_mesh = object()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})

    def fake_select_kernel_mesh(_default_mesh, _use_pcp_streaming):
        return fake_mesh, fake_mesh, None

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

        def __call__(self, kv_cache, query, *_args, **_kwargs):
            return kv_cache, query

    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(fake_select_kernel_mesh),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op",
        lambda *_args, **_kwargs: FakeOp(),
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        prebuilt = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
        )

    def fail_select_kernel_mesh(_default_mesh, _use_pcp_streaming):
        raise AssertionError("mesh lookup should not run on prebuilt cache hit")

    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(fail_select_kernel_mesh),
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        cached = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
        )

    assert cached is prebuilt


@pytest.mark.parametrize("block_major", [False, True])
def test_build_streaming_kernel_registers_eight_tensor_custom_op(
    monkeypatch, vllm_config_context, block_major
):
    impl = _impl()
    fake_mesh = object()
    captured = {}

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})

    def fake_select_kernel_mesh(_default_mesh, use_pcp_streaming):
        assert use_pcp_streaming is True
        return (
            fake_mesh,
            fake_mesh,
            pcp_adapter.PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        )

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

        def __call__(self, *args, **_kwargs):
            captured["op_args"] = args
            return args[0], args[1]

    def fake_jax_op(_name, fn, **kwargs):
        captured["signature"] = inspect.signature(fn, follow_wrapped=False)
        captured["input_partition_specs"] = kwargs["input_partition_specs"]
        captured["output_partition_specs"] = kwargs["output_partition_specs"]
        return FakeOp()

    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(fake_select_kernel_mesh),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op", fake_jax_op
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        kernel = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
            block_major=block_major,
        )

    params = list(captured["signature"].parameters.values())
    assert len(params) == 8
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
    assert len(captured["input_partition_specs"]) == 8
    mesh = SimpleNamespace(shape={"pcp": 4, "tp": 2})
    local_cache = (8, 3, 16, 2, 1, 128) if block_major else (8, 16, 2, 1, 128)
    global_cache = (32, 3, 16, 4, 1, 128) if block_major else (32, 16, 4, 1, 128)
    assert (
        pallas_shapes.get_global_shape(
            local_cache, mesh, captured["input_partition_specs"][0]
        )
        == global_cache
    )
    assert (
        pallas_shapes.get_local_shape(
            global_cache, mesh, captured["output_partition_specs"][0]
        )
        == local_cache
    )

    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    kernel(
        kv_cache,
        query,
        key,
        value,
        metadata.seq_lens,
        metadata.block_tables,
        metadata.query_start_loc,
        metadata.request_distribution,
        None,
    )

    assert len(captured["op_args"]) == 8
    assert captured["op_args"][7] is metadata.request_distribution


@pytest.fixture
def hnd_vllm_config_context(monkeypatch):
    vllm_envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    config = VllmConfig(attention_config=AttentionConfig(backend="CUSTOM"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield config


def test_pcp_hnd_layout_is_forwarded_to_streaming_kernel(
    monkeypatch, hnd_vllm_config_context
):
    impl = _batched_impl(kv_cache_dtype="fp8_e4m3")
    fake_mesh = object()
    captured = {}

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (fake_mesh, fake_mesh, None)),
    )

    def fake_make_kernel(**kwargs):
        captured.update(kwargs)
        return object()

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.make_pcp_streaming_rpa_kernel",
        fake_make_kernel,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op",
        lambda *_args, **_kwargs: FakeOp(),
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        impl._build_rpa_kernel(
            None,
            0.125,
            0.25,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=128,
        )

    assert captured["kv_layout"] == KVLayout.SEQ_ALONG_LANE


def test_pcp_hnd_layout_selects_sequence_cache_shape_and_page_sizes(
    hnd_vllm_config_context,
):
    shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        8, 256, 1, 128, torch.float8_e4m3fn
    )

    assert shape == (8, 2, 32, 4, 256)
    parallel_config = hnd_vllm_config_context.parallel_config
    parallel_config.prefill_context_parallel_size = 8
    parallel_config.cp_kv_cache_interleave_size = 256
    assert PallasBatchedRPAAttentionBackend.get_supported_kernel_block_sizes() == [
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]


def test_pcp_hnd_layout_requires_custom_backend_and_fp8(hnd_vllm_config_context):
    with pytest.raises(NotImplementedError, match="non-CUSTOM"):
        _impl(kv_cache_dtype="fp8_e4m3")._validate_pcp_streaming_support(False)
    with pytest.raises(NotImplementedError, match="without an FP8"):
        _batched_impl()._validate_pcp_streaming_support(False)

    _batched_impl(kv_cache_dtype="fp8_e4m3")._validate_pcp_streaming_support(False)


def test_adapter_make_streaming_kernel_has_eight_tensor_signature_and_calls_wrapper(
    monkeypatch,
):
    captured = {}
    new_cache = object()
    output = object()

    def fake_pcp_wrapper(**kwargs):
        captured.update(kwargs)
        return output, new_cache

    monkeypatch.setattr(
        pcp_adapter, "sharded_pcp_ragged_paged_attention", fake_pcp_wrapper
    )

    mesh = object()
    kernel = pcp_adapter.make_pcp_streaming_rpa_kernel(
        q_scale=None,
        k_scale=0.125,
        v_scale=0.25,
        mesh=mesh,
        sliding_window=None,
        sm_scale=1.0,
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=4,
    )

    params = list(inspect.signature(kernel).parameters.values())
    assert len(params) == 8
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)

    args = tuple(object() for _ in range(8))
    assert kernel(*args) == (new_cache, output)
    assert captured["mesh"] is mesh
    assert captured["q"] is args[1]
    assert captured["k"] is args[2]
    assert captured["v"] is args[3]
    assert captured["kv_cache"] is args[0]
    assert captured["kv_lens"] is args[4]
    assert captured["page_indices"] is args[5]
    assert captured["cu_q_lens"] is args[6]
    assert captured["distribution"] is args[7]
    assert captured["attention_sink"] is None
    assert captured["sm_scale"] == 1.0
    assert captured["attention_chunk_size"] is None
    assert captured["q_scale"] is None
    assert captured["k_scale"] == 0.125
    assert captured["v_scale"] == 0.25
    assert captured["update_kv_cache"] is True
    assert captured["cp_kv_cache_interleave_size"] == 4


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"soft_cap": 30.0, "skip_kv_update": False}, "logits soft cap"),
        ({"soft_cap": None, "skip_kv_update": True}, "skip_kv_update"),
    ],
)
def test_adapter_make_streaming_kernel_rejects_unsupported_features(
    kwargs, expected_message
):
    with pytest.raises(NotImplementedError, match=expected_message):
        pcp_adapter.make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=None,
            v_scale=None,
            mesh=object(),
            sliding_window=None,
            sm_scale=1.0,
            cp_kv_cache_interleave_size=4,
            **kwargs,
        )


def test_build_pcp_streaming_callable_returns_stock_jax_callable(monkeypatch):
    """The callable must be stock, not a subclass that overrides __call__.

    It used to be a local subclass so it could size shard_map outputs from
    ``lowered._out_named_shardings``; stock derived them from
    ``lowered.out_avals``, which jax.export records as replicated. Since
    cl/975923205 stock takes ``output_partition_specs`` directly, so the
    subclass is unnecessary -- and while it existed it silently dropped every
    argument stock's __call__ gained, ``mesh_device_ids`` included. Asserting
    the exact type is what keeps a subclass from creeping back in.
    """
    captured = {}

    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_verify_signature", lambda _signature: None
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_infer_static_argnums", lambda _signature: ()
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl,
        "_get_kernel_invocation_key",
        lambda *_args, **_kwargs: "trace_key",
    )

    # Record the kwargs but hand back a genuinely jitted function:
    # JaxCallable.__init__ runs jax.export.export on it, which rejects
    # anything that is not the result of jit.
    real_jit = pcp_adapter.jax.jit

    def recording_jit(fn, **kwargs):
        captured["jit_kwargs"] = kwargs
        return real_jit(fn, **kwargs)

    monkeypatch.setattr(pcp_adapter.jax, "jit", recording_jit)

    mesh = object()
    callable_ = pcp_adapter.build_pcp_streaming_callable(
        "pcp::op",
        lambda tensor: tensor,
        donate_argnums=(0,),
        mesh=mesh,
        input_partition_specs=("input_spec",),
        output_partition_specs=("out_spec",),
    )

    assert type(callable_) is pcp_adapter.pallas_impl.JaxCallable
    assert callable_.name == "pcp::op"
    assert callable_.mesh is mesh
    assert callable_.input_partition_specs == ("input_spec",)
    assert callable_.output_partition_specs == ("out_spec",)
    assert callable_.donate_argnums == (0,)
    # Stock does not constrain the jit's outputs, and neither should this: the
    # shard_map body already declares out_specs, so pinning them again only
    # adds an advisory result attribute to a module whose partitioner is a
    # no-op. Matching stock is what makes the two paths interchangeable.
    assert "out_shardings" not in captured["jit_kwargs"]


def test_build_pcp_streaming_callable_defaults_output_specs(monkeypatch):
    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_verify_signature", lambda _signature: None
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_infer_static_argnums", lambda _signature: ()
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl,
        "_get_kernel_invocation_key",
        lambda *_args, **_kwargs: "trace_key",
    )

    callable_ = pcp_adapter.build_pcp_streaming_callable(
        "pcp::op",
        lambda tensor: tensor,
        donate_argnums=None,
        mesh=object(),
        input_partition_specs=("input_spec",),
    )

    assert (
        callable_.output_partition_specs
        == pcp_adapter.PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS
    )


def test_adapter_converts_dimensions_sharded_over_pcp_and_tp():
    mesh = SimpleNamespace(shape={"pcp": 4, "tp": 2})
    spec = PartitionSpec(None, ("tp", "pcp"), None)

    global_shape = pallas_shapes._convert_partitioned_shape(
        (3, 8, 16), mesh, spec, to_global=True
    )
    local_shape = pallas_shapes._convert_partitioned_shape(
        global_shape, mesh, spec, to_global=False
    )

    assert global_shape == (3, 64, 16)
    assert local_shape == (3, 8, 16)


def test_adapter_rejects_nondivisible_compound_axis_output_shape():
    mesh = SimpleNamespace(shape={"pcp": 4, "tp": 2})
    with pytest.raises(ValueError, match="not divisible by mesh factor 8"):
        pallas_shapes._convert_partitioned_shape(
            (3, 63, 16),
            mesh,
            PartitionSpec(None, ("tp", "pcp"), None),
            to_global=False,
        )


def test_adapter_pcp_jax_op_uses_scoped_callable(monkeypatch):
    captured = {}

    class FakePcpCallable:
        def __init__(self, **kwargs):
            captured["callable_kwargs"] = kwargs

    class FakeCustomOp:
        def register_fake(self, fake_impl):
            captured["fake_impl"] = fake_impl

    def fake_custom_op(name, wrapped_fn, *, mutates_args):
        captured["custom_op"] = (name, wrapped_fn, mutates_args)
        return FakeCustomOp()

    monkeypatch.setattr(pcp_adapter.pallas_impl, "JaxCallable", FakePcpCallable)
    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_verify_signature", lambda _signature: None
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl, "_infer_static_argnums", lambda _signature: ()
    )
    monkeypatch.setattr(
        pcp_adapter.pallas_impl,
        "_get_kernel_invocation_key",
        lambda *_args, **_kwargs: "trace_key",
    )
    monkeypatch.setattr(
        pcp_adapter.jax, "jit", lambda fn, **kwargs: ("jit_fn", fn, kwargs)
    )
    monkeypatch.setattr(pcp_adapter.torch.library, "custom_op", fake_custom_op)

    def fn(tensor):
        return tensor

    result = pcp_adapter.pcp_streaming_jax_op(
        "pcp::op",
        fn,
        donate_argnums=(0,),
        mesh=object(),
        input_partition_specs=("input_spec",),
    )

    assert isinstance(result, FakeCustomOp)
    assert isinstance(captured["custom_op"][1], FakePcpCallable)
    assert captured["callable_kwargs"]["name"] == "pcp::op"
    assert captured["callable_kwargs"]["jit_fn"][0] == "jit_fn"
    assert captured["callable_kwargs"]["donate_argnums"] == (0,)
    assert (
        captured["callable_kwargs"]["output_partition_specs"]
        == pcp_adapter.PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS
    )
    assert captured["custom_op"][2] == ()


def test_adapter_invoke_streaming_op_trims_generic_tensor_args():
    captured = {}
    kv_cache = object()
    generic_args = (*tuple(object() for _ in range(7)), None)
    expected = object()

    def fake_op(*args):
        captured["op_args"] = args
        return expected

    actual = pcp_adapter.invoke_pcp_streaming_op(fake_op, kv_cache, generic_args, {})

    assert actual is expected
    assert captured["op_args"] == (kv_cache, *generic_args[:7])


def test_adapter_invoke_streaming_op_rejects_unsupported_sink():
    args = tuple(object() for _ in range(8))

    with pytest.raises(NotImplementedError, match="attention sinks"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), args, {})


def test_adapter_invoke_streaming_op_rejects_kwargs_and_wrong_arg_count():
    with pytest.raises(ValueError, match="does not accept keyword"):
        pcp_adapter.invoke_pcp_streaming_op(
            lambda *_args: None, object(), (), {"sink": object()}
        )

    with pytest.raises(ValueError, match="expects 8 tensor args"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), (), {})


@pytest.mark.parametrize(
    "rebind_local, expects_kv_layout",
    [(False, True), (True, False)],
    ids=["batched entry", "tp=1 draft rebound to local entry"],
)
def test_kv_layout_kwarg_follows_the_bound_kernel_entry(
    monkeypatch, vllm_config_context, rebind_local, expects_kv_layout
):
    """`tpu_runner._initialize_attention_kernels` rebinds `_kernel_entry` on the
    instance for a tp=1 draft, so the kwarg must be decided per instance.

    Binding `kv_layout` into `_pallas_rpa_kernel_local`, which has no such
    parameter, took down both disagg spec-decode suites with
    `TypeError: got an unexpected keyword argument 'kv_layout'`.
    """
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    impl = PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )
    if rebind_local:
        impl._kernel_entry = _pallas_rpa_kernel_local
        impl._kernel_op_prefix = "pallas::rpa_kernel_local"

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)),
    )

    captured = {}

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

        def __call__(self, kv_cache, query, *_args, **_kwargs):
            return kv_cache, query

    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured["fn"] = wrapped_fn
        return FakeOp()

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pallas.jax_op", fake_jax_op
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        impl._build_rpa_kernel(None, None, None)

    bound = captured["fn"]
    assert bound.func is impl._kernel_entry
    assert ("kv_layout" in bound.keywords) is expects_kv_layout
    # The partial must be callable with what it was bound to.
    inspect.signature(bound.func).bind_partial(**bound.keywords)


# --- block-major unified pool ---------------------------------------------------


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


_BM_LAYOUT = BlockMajorPoolLayout(
    num_blocks=8, num_pools=3, split=4, pool_index_by_layer={}
)
_BM_POOL_INDEX = 1


def _reset_kernel_registry(monkeypatch):
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)),
    )


@pytest.mark.parametrize("block_major", [True, False])
def test_forward_remaps_block_tables_for_the_block_major_pool(
    monkeypatch, cpu_vllm_config_context, block_major
):
    """Only a layer bound to a merged pool region hands the kernel remapped
    block tables; a per-region pool passes the scheduler's tables through."""
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    if block_major:
        kv_cache = torch.zeros(8, 3 * 4, 4, 2, 1, 128)
    block_tables = torch.arange(1, 9, dtype=torch.int32).reshape(2, 4)
    metadata = _metadata(block_tables=block_tables)
    impl = _impl()
    if block_major:
        impl._bm_kernel = True
        impl.set_block_major_pool(_BM_POOL_INDEX, _BM_LAYOUT)

    with set_vllm_model_wrapper_context(
        mesh=_mesh(), vllm_config=_vllm_config(pcp_size=1)
    ):
        impl.forward(_layer(), query, key, value, kv_cache, metadata)

    assert captured["build"]["block_major"] is block_major
    passed = captured["args"][5]
    if block_major:
        expected = flat_kernel_ids(block_tables, _BM_POOL_INDEX, 3, 4)
        assert passed.tolist() == expected.tolist()
        assert passed.tolist() != block_tables.tolist()
    else:
        assert passed is block_tables


def test_build_rpa_kernel_rejects_an_entry_without_block_major(
    monkeypatch, cpu_vllm_config_context
):
    impl = _impl()
    impl._kernel_entry = _pallas_rpa_kernel_local
    impl._kernel_op_prefix = "pallas::rpa_kernel_local"
    _reset_kernel_registry(monkeypatch)

    with (
        set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
        pytest.raises(NotImplementedError, match="block-major"),
    ):
        impl._build_rpa_kernel(None, None, None, block_major=True)


def test_build_rpa_kernel_keys_the_registry_on_block_major(
    monkeypatch, cpu_vllm_config_context
):
    impl = _impl()
    _reset_kernel_registry(monkeypatch)
    registered = []

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

        def __call__(self, kv_cache, query, *_args, **_kwargs):
            return kv_cache, query

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        registered.append((name, wrapped_fn))
        return FakeOp()

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pallas.jax_op", fake_jax_op
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
        layer_major = impl._build_rpa_kernel(None, None, None)
        block_major = impl._build_rpa_kernel(None, None, None, block_major=True)
        again = impl._build_rpa_kernel(None, None, None, block_major=True)

    assert layer_major is not block_major
    assert again is block_major
    assert len(PallasAttentionBackendImpl._kernel_registry) == 2
    (lm_name, lm_fn), (bm_name, bm_fn) = registered
    assert lm_name != bm_name
    assert "block_major" not in lm_fn.keywords
    assert bm_fn.keywords["block_major"] is True
    assert bm_fn.func is impl._kernel_entry
