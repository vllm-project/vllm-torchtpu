import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.envs as vllm_envs
from vllm.config import set_current_vllm_config

import vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter as pcp_adapter
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl, PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context


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


def _metadata(*, pcp_streaming=False):
    layout_kwargs = {}
    if pcp_streaming:
        layout_kwargs = {
            "sequence_layout_kind": SequenceLayoutKind.PARTIAL.value,
            "sequence_layout_protocol": "pcp_streaming",
        }
    return AttentionMetadata(
        input_positions=torch.arange(3, dtype=torch.int32),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.tensor([3, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 3], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        **layout_kwargs,
    )


def _capture_kernel(monkeypatch):
    captured = {}

    def fake_build(self,
                   q_scale,
                   k_scale,
                   v_scale,
                   *,
                   skip_kv_update=False,
                   use_pcp_streaming,
                   cp_kv_cache_interleave_size):
        captured["build"] = {
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "skip_kv_update": skip_kv_update,
            "use_pcp_streaming": use_pcp_streaming,
            "cp_kv_cache_interleave_size": cp_kv_cache_interleave_size,
        }

        def fake_kernel(*args):
            captured["args"] = args
            return torch.full_like(args[1], 7.0)

        return fake_kernel

    monkeypatch.setattr(PallasAttentionBackendImpl, "_build_rpa_kernel",
                        fake_build)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.synchronize_tensors",
        lambda *_args, **_kwargs: None,
    )
    return captured


def test_forward_routes_partial_layout_to_streaming_kernel(
        monkeypatch, vllm_config_context):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = _impl().forward(_layer(), query, key, value, kv_cache,
                                  metadata)

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["cp_kv_cache_interleave_size"] == 4
    assert len(captured["args"]) == 9
    assert captured["args"][1].shape[0] == query.shape[0]
    assert captured["args"][8] is None
    assert torch.all(outputs == 7)


def test_forward_routes_fp8_partial_layout_with_kv_scales(
        monkeypatch, vllm_config_context):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)
    layer = _layer(k_scale=0.125, v_scale=0.25)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = _impl(kv_cache_dtype="fp8_e4m3").forward(
            layer, query, key, value, kv_cache, metadata)

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["q_scale"] is None
    assert captured["build"]["k_scale"] == 0.125
    assert captured["build"]["v_scale"] == 0.25
    assert captured["args"][2].dtype == torch.float8_e4m3fn
    assert captured["args"][3].dtype == torch.float8_e4m3fn
    assert torch.all(outputs == 7)


def test_forward_routes_pcp_decode_metadata_to_streaming_kernel(
        monkeypatch, vllm_config_context):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        outputs = _impl().forward(_layer(), query, key, value, kv_cache,
                                  metadata)

    assert outputs.shape == query.shape
    assert captured["build"]["use_pcp_streaming"] is True
    assert captured["build"]["cp_kv_cache_interleave_size"] == 4
    assert len(captured["args"]) == 9
    assert captured["args"][8] is None


@pytest.mark.parametrize(
    ("impl_kwargs", "expected_message"),
    [
        ({
            "sinks": torch.zeros(2)
        }, "attention sinks"),
        ({
            "logits_soft_cap": 30.0
        }, "logits soft cap"),
        ({
            "kv_sharing_target_layer_name": "shared"
        }, "skip_kv_update"),
    ],
)
def test_forward_rejects_unsupported_pcp_streaming_features(
        impl_kwargs, expected_message, vllm_config_context):
    query, key, value, kv_cache = _tensors()
    metadata = _metadata(pcp_streaming=True)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match=expected_message):
            _impl(**impl_kwargs).forward(_layer(), query, key, value, kv_cache,
                                         metadata)


def test_initialize_kernel_rejects_unsupported_pcp_streaming_features(
        vllm_config_context):
    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        with pytest.raises(NotImplementedError, match="logits soft cap"):
            _impl(logits_soft_cap=30.0).initialize_kernel(_layer())


def test_forward_keeps_non_pcp_batch_on_disabled_mode(monkeypatch,
                                                      vllm_config_context):
    captured = _capture_kernel(monkeypatch)
    query, key, value, kv_cache = _tensors()
    metadata = _metadata()

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config(pcp_size=1)):
        _impl().forward(_layer(), query, key, value, kv_cache, metadata)

    assert captured["build"]["use_pcp_streaming"] is False
    assert captured["build"]["cp_kv_cache_interleave_size"] == 0


def test_initialize_kernel_prebuilds_streaming_variant_for_native_pcp(
        monkeypatch, vllm_config_context):
    calls = []

    def fake_build(self,
                   q_scale,
                   k_scale,
                   v_scale,
                   skip_kv_update=False,
                   use_pcp_streaming=False,
                   cp_kv_cache_interleave_size=0):
        calls.append((use_pcp_streaming, cp_kv_cache_interleave_size))
        return MagicMock()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_build_rpa_kernel",
                        fake_build)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        _impl().initialize_kernel(_layer())

    assert calls == [(True, 4)]


def test_build_rpa_kernel_reuses_prebuilt_config_before_mesh_lookup(
        monkeypatch, vllm_config_context):
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

    monkeypatch.setattr(PallasAttentionBackendImpl, "_select_kernel_mesh",
                        staticmethod(fake_select_kernel_mesh))
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op",
        lambda *_args, **_kwargs: FakeOp())

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        prebuilt = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
        )

    def fail_select_kernel_mesh(_default_mesh, _use_pcp_streaming):
        raise AssertionError(
            "mesh lookup should not run on prebuilt cache hit")

    monkeypatch.setattr(PallasAttentionBackendImpl, "_select_kernel_mesh",
                        staticmethod(fail_select_kernel_mesh))

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        cached = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
        )

    assert cached is prebuilt


def test_build_streaming_kernel_registers_eight_tensor_custom_op(
        monkeypatch, vllm_config_context):
    impl = _impl()
    fake_mesh = object()
    captured = {}

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})

    def fake_select_kernel_mesh(_default_mesh, use_pcp_streaming):
        assert use_pcp_streaming is True
        return (fake_mesh, fake_mesh,
                pcp_adapter.PCP_STREAMING_RPA_INPUT_PARTITION_SPECS)

    class FakeOp:

        def register_fake(self, _fake_impl):
            pass

        def __call__(self, *args, **_kwargs):
            captured["op_args"] = args
            return args[0], args[1]

    def fake_jax_op(_name, fn, **kwargs):
        captured["signature"] = inspect.signature(fn, follow_wrapped=False)
        captured["input_partition_specs"] = kwargs["input_partition_specs"]
        return FakeOp()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_select_kernel_mesh",
                        staticmethod(fake_select_kernel_mesh))
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op",
        fake_jax_op)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        kernel = impl._build_rpa_kernel(
            None,
            None,
            None,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=4,
        )

    params = list(captured["signature"].parameters.values())
    assert len(params) == 8
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in params)
    assert len(captured["input_partition_specs"]) == 8

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


def test_pcp_hnd_layout_is_forwarded_to_streaming_kernel(monkeypatch):
    vllm_envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
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
        "vllm_torchtpu.layers.adapter.attention."
        "make_pcp_streaming_rpa_kernel",
        fake_make_kernel,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pcp_streaming_jax_op",
        lambda *_args, **_kwargs: FakeOp(),
    )

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        impl._build_rpa_kernel(
            None,
            0.125,
            0.25,
            use_pcp_streaming=True,
            cp_kv_cache_interleave_size=128,
        )

    assert captured["kv_layout"] == KVLayout.SEQ_ALONG_LANE


def test_pcp_hnd_layout_selects_sequence_cache_shape_and_page_sizes(
        monkeypatch):
    vllm_envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    shape = PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        8, 256, 1, 128, torch.float8_e4m3fn)

    assert shape == (8, 2, 32, 4, 256)
    with set_current_vllm_config(_vllm_config(pcp_size=8,
                                              interleave_size=256)):
        assert (PallasBatchedRPAAttentionBackend.
                get_supported_kernel_block_sizes() == [
                    128, 256, 512, 1024, 2048, 4096
                ])


def test_pcp_hnd_layout_requires_custom_backend_and_fp8(monkeypatch):
    vllm_envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")

    with pytest.raises(NotImplementedError, match="non-CUSTOM"):
        _impl(kv_cache_dtype="fp8_e4m3")._validate_pcp_streaming_support(False)
    with pytest.raises(NotImplementedError, match="without an FP8"):
        _batched_impl()._validate_pcp_streaming_support(False)

    _batched_impl(
        kv_cache_dtype="fp8_e4m3")._validate_pcp_streaming_support(False)


def test_adapter_make_streaming_kernel_has_eight_tensor_signature_and_calls_wrapper(
        monkeypatch):
    captured = {}
    new_cache = object()
    output = object()

    def fake_pcp_wrapper(**kwargs):
        captured.update(kwargs)
        return output, new_cache

    monkeypatch.setattr(pcp_adapter, "sharded_pcp_ragged_paged_attention",
                        fake_pcp_wrapper)

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
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in params)

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
        ({
            "soft_cap": 30.0,
            "skip_kv_update": False
        }, "logits soft cap"),
        ({
            "soft_cap": None,
            "skip_kv_update": True
        }, "skip_kv_update"),
    ],
)
def test_adapter_make_streaming_kernel_rejects_unsupported_features(
        kwargs, expected_message):
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


def test_adapter_scoped_callable_uses_lowered_out_shardings(monkeypatch):
    captured = {}
    original_call = pcp_adapter.pallas_impl.JaxCallable.__call__

    class FakeOutTree:

        def unflatten(self, values):
            return tuple(values)

    class FakeLowered:
        out_avals = ("aval0", "aval1")
        _out_named_shardings = ("sharding0", "sharding1")
        out_tree = FakeOutTree()
        mlir_module_serialized = b"mlir"

        def mlir_module(self):
            return "module {}"

    class FakePcpCallable(pcp_adapter._PcpStreamingJaxCallable):

        def __init__(self):
            self.trace_key = "trace"
            self.static_argnums = ()
            self.output_shapes = {}
            self.kernel_key_to_mlir_fingerprint = {}
            self.mesh = "mesh"
            self.input_partition_specs = ("input_spec", )
            self.name = "op"
            self.exported = lambda *args, **kwargs: FakeLowered()
            self.donate_argnums = ()
            self.input_output_aliases = {}

        def _validate_args(self, *args):
            captured["validated_args"] = args

    class FakePallasRuntime:

        def lookup_custom_kernel(self, name, key):
            captured["lookup"] = (name, key)
            return False

        def register_custom_kernel(self, name, key, *, serialized_mlir_module):
            captured["registered"] = (name, key, serialized_mlir_module)

        def call_custom_kernel(self, name, key, *, inputs, output_shapes,
                               donate_argnums):
            captured["call"] = {
                "name": name,
                "key": key,
                "inputs": inputs,
                "output_shapes": output_shapes,
                "donate_argnums": donate_argnums,
            }
            return ("result0", "result1")

    monkeypatch.setattr(pcp_adapter.pallas_impl, "tpu_torch_pallas",
                        FakePallasRuntime())
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_get_kernel_invocation_key",
                        lambda *_args, **_kwargs: "kernel_key")
    monkeypatch.setattr(pcp_adapter.pallas_impl, "jax_placeholders",
                        lambda *_args, **_kwargs: ("jax_arg", ))
    monkeypatch.setattr(
        pcp_adapter, "_torch_placeholder_with_sharding",
        lambda aval, sharding, mesh: f"{aval}:{sharding}:{mesh}")

    result = FakePcpCallable()(torch.tensor(1.0))

    assert result == ("result0", "result1")
    assert captured["call"]["output_shapes"] == [
        "aval0:sharding0:mesh", "aval1:sharding1:mesh"
    ]
    assert pcp_adapter.pallas_impl.JaxCallable.__call__ is original_call


def test_adapter_pcp_jax_op_uses_scoped_callable(monkeypatch):
    captured = {}
    original_call = pcp_adapter.pallas_impl.JaxCallable.__call__

    class FakePcpCallable:

        def __init__(self, **kwargs):
            captured["callable_kwargs"] = kwargs

    class FakeCustomOp:

        def register_fake(self, fake_impl):
            captured["fake_impl"] = fake_impl

    def fake_custom_op(name, wrapped_fn, *, mutates_args):
        captured["custom_op"] = (name, wrapped_fn, mutates_args)
        return FakeCustomOp()

    monkeypatch.setattr(pcp_adapter, "_PcpStreamingJaxCallable",
                        FakePcpCallable)
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_verify_signature",
                        lambda _signature: None)
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_infer_static_argnums",
                        lambda _signature: ())
    monkeypatch.setattr(pcp_adapter.pallas_impl, "_get_kernel_invocation_key",
                        lambda *_args, **_kwargs: "trace_key")
    monkeypatch.setattr(pcp_adapter, "_named_shardings",
                        lambda *_args, **_kwargs: ("out0", "out1"))
    monkeypatch.setattr(pcp_adapter.jax, "jit", lambda fn, **kwargs:
                        ("jit_fn", fn, kwargs))
    monkeypatch.setattr(pcp_adapter.torch.library, "custom_op", fake_custom_op)

    def fn(tensor):
        return tensor

    result = pcp_adapter.pcp_streaming_jax_op(
        "pcp::op",
        fn,
        donate_argnums=(0, ),
        mesh=object(),
        input_partition_specs=("input_spec", ),
    )

    assert isinstance(result, FakeCustomOp)
    assert isinstance(captured["custom_op"][1], FakePcpCallable)
    assert captured["callable_kwargs"]["name"] == "pcp::op"
    assert captured["callable_kwargs"]["jit_fn"][0] == "jit_fn"
    assert captured["callable_kwargs"]["donate_argnums"] == (0, )
    assert captured["custom_op"][2] == ()
    assert pcp_adapter.pallas_impl.JaxCallable.__call__ is original_call


def test_adapter_invoke_streaming_op_trims_generic_tensor_args():
    captured = {}
    kv_cache = object()
    generic_args = (*tuple(object() for _ in range(7)), None)
    expected = object()

    def fake_op(*args):
        captured["op_args"] = args
        return expected

    actual = pcp_adapter.invoke_pcp_streaming_op(fake_op, kv_cache,
                                                 generic_args, {})

    assert actual is expected
    assert captured["op_args"] == (kv_cache, *generic_args[:7])


def test_adapter_invoke_streaming_op_rejects_unsupported_sink():
    args = tuple(object() for _ in range(8))

    with pytest.raises(NotImplementedError, match="attention sinks"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(),
                                            args, {})


def test_adapter_invoke_streaming_op_rejects_kwargs_and_wrong_arg_count():
    with pytest.raises(ValueError, match="does not accept keyword"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), (),
                                            {"sink": object()})

    with pytest.raises(ValueError, match="expects 8 tensor args"):
        pcp_adapter.invoke_pcp_streaming_op(lambda *_args: None, object(), (),
                                            {})


@pytest.mark.parametrize(
    "rebind_local, expects_kv_layout",
    [(False, True), (True, False)],
    ids=["batched entry", "tp=1 draft rebound to local entry"],
)
def test_kv_layout_kwarg_follows_the_bound_kernel_entry(
        monkeypatch, vllm_config_context, rebind_local, expects_kv_layout):
    """`tpu_runner._initialize_attention_kernels` rebinds `_kernel_entry` on the
    instance for a tp=1 draft, so the kwarg must be decided per instance.

    Binding `kv_layout` into `_pallas_rpa_kernel_local`, which has no such
    parameter, took down both disagg spec-decode suites with
    `TypeError: got an unexpected keyword argument 'kv_layout'`.
    """
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl, _pallas_rpa_kernel_local)

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
        PallasAttentionBackendImpl, "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)))

    captured = {}

    class FakeOp:

        def register_fake(self, _fake_impl):
            pass

        def __call__(self, kv_cache, query, *_args, **_kwargs):
            return kv_cache, query

    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured["fn"] = wrapped_fn
        return FakeOp()

    monkeypatch.setattr("vllm_torchtpu.layers.adapter.attention.pallas.jax_op",
                        fake_jax_op)

    with set_vllm_model_wrapper_context(mesh=_mesh(),
                                        vllm_config=_vllm_config()):
        impl._build_rpa_kernel(None, None, None)

    bound = captured["fn"]
    assert bound.func is impl._kernel_entry
    assert ("kv_layout" in bound.keywords) is expects_kv_layout
    # The partial must be callable with what it was bound to.
    inspect.signature(bound.func).bind_partial(**bound.keywords)
