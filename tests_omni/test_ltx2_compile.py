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
import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from vllm_omni.diffusion import compile as diffusion_compile
from vllm_omni.diffusion.attention.parallel import ulysses as ulysses_mod
from vllm_omni.diffusion.distributed import comm, parallel_state
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_guidance,
    ltx2_transformer,
    pipeline_ltx2,
    pipeline_ltx2_audio,
    pipeline_ltx2_two_stage,
)
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_sequence_parallel as ltx2_sp_mod,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.worker import diffusion_model_runner

from vllm_torchtpu.omni import ltx2_compile
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = pytest.mark.omni

_UPSTREAM_BUILD_PERTURBATION = ltx2_guidance.build_perturbation_kwargs
_UPSTREAM_PREDICT_PARALLEL = ltx2_guidance.LTXGuidanceExecutor.predict_parallel_guidance
_UPSTREAM_SEQ_A2A = comm.SeqAllToAll4D
_UPSTREAM_V2A_PRE = ltx2_sp_mod.LTX2VideoToAudioParallelAttention.pre_attention
_UPSTREAM_V2A_POST = ltx2_sp_mod.LTX2VideoToAudioParallelAttention.post_attention
_UPSTREAM_LTX2_FORWARD = ltx2_transformer.LTX2VideoTransformer3DModel.forward
_UPSTREAM_ATTN_PROC_CALL = ltx2_transformer.LTX2AudioVideoAttnProcessor.__call__


@contextmanager
def _temp_process_group(backend: str, world_size: int):
    if dist.is_initialized():
        pytest.skip("torch.distributed process group already initialized")
    store = FakeStore() if backend == "fake" else dist.HashStore()
    dist.init_process_group(
        backend=backend,
        rank=0,
        world_size=world_size,
        store=store,
    )
    try:
        yield dist.group.WORLD
    finally:
        torch._dynamo.reset()
        dist.destroy_process_group()


@pytest.fixture
def sync_log(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ltx2_compile,
        "synchronize_tensors",
        lambda tensors, wait=False: calls.append((tensors, wait)),
    )
    return calls


def test_functional_all_to_all_4d_matches_upstream(monkeypatch):
    world_size = 4
    monkeypatch.setattr(
        ltx2_compile.dist, "get_world_size", lambda group=None: world_size
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: world_size
    )

    def _run_world_funcol(rank_inputs, scatter_idx, gather_idx):
        sends = []
        monkeypatch.setattr(
            ltx2_compile.funcol,
            "all_to_all_single",
            lambda x, out_s, in_s, group: (
                sends.append(x.clone()) or torch.zeros_like(x)
            ),
        )
        for r in range(world_size):
            ltx2_compile._functional_all_to_all_4d(
                rank_inputs[r], scatter_idx, gather_idx, group="pg"
            )
        chunked = torch.stack(
            [torch.stack(s.chunk(world_size, dim=0), dim=0) for s in sends], dim=0
        )
        transposed = chunked.transpose(0, 1).contiguous()
        outputs = []
        for r in range(world_size):
            recv_flat = transposed[r].reshape_as(sends[r])
            monkeypatch.setattr(
                ltx2_compile.funcol,
                "all_to_all_single",
                lambda x, out_s, in_s, group, rf=recv_flat: rf.clone(),
            )
            outputs.append(
                ltx2_compile._functional_all_to_all_4d(
                    rank_inputs[r], scatter_idx, gather_idx, group="pg"
                )
            )
        return outputs

    def _run_world_eager(rank_inputs, scatter_idx, gather_idx):
        sends = []

        def collect(recv_buf, send_buf, group=None):
            del group
            sends.append(send_buf.clone())
            recv_buf.zero_()

        monkeypatch.setattr(torch.distributed, "all_to_all_single", collect)
        for r in range(world_size):
            comm.all_to_all_4D(rank_inputs[r], scatter_idx, gather_idx, group="pg")
        transposed = torch.stack(sends, dim=0).transpose(0, 1).contiguous()
        outputs = []
        for r in range(world_size):
            r_recv = transposed[r]
            monkeypatch.setattr(
                torch.distributed,
                "all_to_all_single",
                lambda recv_buf, send_buf, group=None, rr=r_recv: recv_buf.copy_(rr),
            )
            outputs.append(
                comm.all_to_all_4D(rank_inputs[r], scatter_idx, gather_idx, group="pg")
            )
        return outputs

    torch.manual_seed(0)
    in_21 = [torch.randn(2, 3, 8, 4) for _ in range(world_size)]
    for g, w in zip(
        _run_world_funcol(in_21, 2, 1), _run_world_eager(in_21, 2, 1), strict=True
    ):
        torch.testing.assert_close(g, w, rtol=0, atol=0)

    in_12 = [torch.randn(2, 12, 2, 4) for _ in range(world_size)]
    for g, w in zip(
        _run_world_funcol(in_12, 1, 2), _run_world_eager(in_12, 1, 2), strict=True
    ):
        torch.testing.assert_close(g, w, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="Invalid"):
        ltx2_compile._functional_all_to_all_4d(in_21[0], 0, 3, group="pg")


def test_traceable_collectives_compile_fullgraph(monkeypatch):
    monkeypatch.setattr(comm, "SeqAllToAll4D", _UPSTREAM_SEQ_A2A)
    monkeypatch.setattr(ulysses_mod, "SeqAllToAll4D", _UPSTREAM_SEQ_A2A)
    monkeypatch.setattr(ltx2_sp_mod, "SeqAllToAll4D", _UPSTREAM_SEQ_A2A)
    monkeypatch.setattr(
        ltx2_sp_mod.LTX2VideoToAudioParallelAttention,
        "pre_attention",
        _UPSTREAM_V2A_PRE,
    )
    monkeypatch.setattr(
        ltx2_sp_mod.LTX2VideoToAudioParallelAttention,
        "post_attention",
        _UPSTREAM_V2A_POST,
    )
    ltx2_compile._patch_ltx2_traceable_collectives()
    ltx2_compile._patch_ltx2_traceable_collectives()  # idempotent

    # 1. Single-rank gloo group: compiled V2A pre/attn/post and SeqAllToAll4D
    # trace with fullgraph=True and match eager upstream bit-for-bit.
    with _temp_process_group("gloo", world_size=1) as pg:
        sp_group = SimpleNamespace(
            ring_world_size=1,
            ulysses_world_size=1,
            ulysses_rank=0,
            ulysses_group=pg,
        )
        v2a = ltx2_sp_mod.LTX2VideoToAudioParallelAttention(sp_group)
        v2a._tpu_head_idx = torch.arange(4, dtype=torch.int32)

        torch.manual_seed(1)
        q = torch.randn(1, 3, 4, 2)
        k = torch.randn(1, 5, 4, 2)
        v = torch.randn(1, 5, 4, 2)
        x = torch.randn(1, 3, 4, 2)

        def _v2a_and_a2a(qt, kt, vt, xt):
            ql, kf, vf, _, ctx = v2a.pre_attention(qt, kt, vt, None)
            attn_out = ql + kf[:, : ql.shape[1]] + vf[:, : ql.shape[1]]
            out_v2a = v2a.post_attention(attn_out, ctx)
            out_a2a = comm.SeqAllToAll4D.apply(pg, xt, 2, 1, False)
            return out_v2a, out_a2a

        eager_v2a, eager_a2a = _v2a_and_a2a(q, k, v, x)
        compiled_fn = torch.compile(
            _v2a_and_a2a, backend="eager", fullgraph=True, dynamic=False
        )
        comp_v2a, comp_a2a = compiled_fn(q, k, v, x)
        torch.testing.assert_close(comp_v2a, eager_v2a, rtol=0, atol=0)
        torch.testing.assert_close(comp_a2a, eager_a2a, rtol=0, atol=0)

        # Missing or wrong-shape _tpu_head_idx fails loudly under compile.
        del v2a._tpu_head_idx
        compiled_missing = torch.compile(
            lambda qt, kt, vt: v2a.pre_attention(qt, kt, vt, None),
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        with pytest.raises(
            (RuntimeError, torch._dynamo.exc.Unsupported),
            match="_tpu_head_idx must be initialized",
        ):
            compiled_missing(q, k, v)

        v2a._tpu_head_idx = torch.arange(2, dtype=torch.int32)
        compiled_bad_shape = torch.compile(
            lambda qt, kt, vt: v2a.pre_attention(qt, kt, vt, None),
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        with pytest.raises(
            (ValueError, torch._dynamo.exc.Unsupported), match="Expected 4 heads"
        ):
            compiled_bad_shape(q, k, v)

    # 2. Multi-rank fake group: exercises the (2, 1) and (1, 2) funcol layout
    # transforms, index_select head slicing, and funcol.all_gather_tensor
    # under fullgraph=True.
    with _temp_process_group("fake", world_size=2) as pg:
        sp_group = SimpleNamespace(
            ring_world_size=1,
            ulysses_world_size=2,
            ulysses_rank=0,
            ulysses_group=pg,
        )
        v2a = ltx2_sp_mod.LTX2VideoToAudioParallelAttention(sp_group)
        v2a._tpu_head_idx = torch.arange(2, dtype=torch.int32)

        def _multi_rank_step(qt, kt, vt, x21, x12):
            ql, kf, vf, _, ctx = v2a.pre_attention(qt, kt, vt, None)
            attn_out = ql + kf[:, : ql.shape[1]] + vf[:, : ql.shape[1]]
            out_v2a = v2a.post_attention(attn_out, ctx)
            out_21 = comm.SeqAllToAll4D.apply(pg, x21, 2, 1, False)
            out_12 = comm.SeqAllToAll4D.apply(pg, x12, 1, 2, False)
            return out_v2a, out_21, out_12

        compiled_multi = torch.compile(
            _multi_rank_step, backend="eager", fullgraph=True, dynamic=False
        )
        out_v2a, out_21, out_12 = compiled_multi(
            torch.randn(1, 3, 4, 2),
            torch.randn(1, 5, 4, 2),
            torch.randn(1, 5, 4, 2),
            torch.randn(1, 3, 4, 2),
            torch.randn(1, 6, 2, 2),
        )
        assert out_v2a.shape == (1, 3, 4, 2)
        assert out_21.shape == (1, 6, 2, 2)
        assert out_12.shape == (1, 3, 4, 2)


@pytest.mark.parametrize(
    ("local_names", "expect_injected"),
    [
        (("uncond", "cond"), True),
        (("ptb", "mod"), False),
    ],
)
def test_symmetric_perturbation_masks_across_cfg_ranks(
    monkeypatch, sync_log, local_names, expect_injected
):
    monkeypatch.setattr(
        ltx2_guidance, "build_perturbation_kwargs", _UPSTREAM_BUILD_PERTURBATION
    )
    captured_kwargs = []
    monkeypatch.setattr(
        ltx2_guidance.LTXGuidanceExecutor,
        "predict_parallel_guidance",
        lambda self, *args, **kwargs: (
            captured_kwargs.append(
                ltx2_guidance.build_perturbation_kwargs(
                    SimpleNamespace(
                        names=local_names,
                        passes=(1, 2),
                        spec=args[1].spec,
                    ),
                    1,
                    torch.zeros(2, 4, 8),
                )
            )
            or (torch.zeros(1), torch.zeros(1))
        ),
    )

    ltx2_compile._patch_ltx2_symmetric_perturbation_masks()
    plan = SimpleNamespace(
        names=("uncond", "cond", "ptb", "mod"),
        spec=SimpleNamespace(
            video=SimpleNamespace(do_stg=True, stg_blocks=[28]),
            audio=SimpleNamespace(do_stg=True, stg_blocks=[28]),
        ),
    )
    executor = ltx2_guidance.LTXGuidanceExecutor()
    executor.predict_parallel_guidance(
        None, plan, 0, torch.tensor(1.0), None, None, None
    )
    kw = captured_kwargs[0]
    assert kw["video_self_attention_blocks"] == [28]
    assert kw["audio_self_attention_blocks"] == [28]
    for key in (
        "video_self_attention_mask",
        "audio_self_attention_mask",
        "a2v_cross_attention_mask",
        "v2a_cross_attention_mask",
    ):
        assert kw[key].shape == (2, 1, 1)
    if expect_injected:
        for key in (
            "video_self_attention_mask",
            "audio_self_attention_mask",
            "a2v_cross_attention_mask",
            "v2a_cross_attention_mask",
        ):
            assert torch.all(kw[key] == 1.0)
        assert len(sync_log) == 1
        assert sync_log[0][1] is False
    else:
        # Upstream builds real 0/1 masks for ptb (index 0) and mod (index 1).
        torch.testing.assert_close(
            kw["video_self_attention_mask"].reshape(-1),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            kw["v2a_cross_attention_mask"].reshape(-1),
            torch.tensor([1.0, 0.0]),
        )
        assert sync_log == []


def test_ensure_zero_offset_and_rope_cache(sync_log):
    full = torch.randn(2, 1, 8, 16)
    assert ltx2_compile._ensure_zero_offset(full) is full
    assert len(sync_log) == 0

    cache = {}
    view0 = full[:, :, 0:4, :]
    view1 = full[:, :, 4:8, :]
    m0 = ltx2_compile._ensure_zero_offset(view0, cache)
    assert m0.is_contiguous() and m0.storage_offset() == 0
    torch.testing.assert_close(m0, view0, rtol=0, atol=0)
    assert ltx2_compile._ensure_zero_offset(full[:, :, 0:4, :], cache) is m0
    assert len(sync_log) == 1

    m1 = ltx2_compile._ensure_zero_offset(view1, cache)
    assert m1.storage_offset() == 0
    torch.testing.assert_close(m1, view1, rtol=0, atol=0)
    assert len(sync_log) == 2


def test_rope_cache_keeps_only_the_current_request_tables(sync_log):
    # Each request builds new tables; their shard copies replace the previous
    # request's, which are then released, and cos and sin (same layout) keep
    # separate entries.
    cache = {}
    old = None
    for _ in range(5):
        tables = (torch.randn(2, 1, 8, 16), torch.randn(2, 1, 8, 16))
        for _block in range(3):
            views = tuple(t[:, :, 4:8, :] for t in tables)
            got = ltx2_compile._ensure_zero_offset(views, cache, "video_rotary_emb")
            for g, v in zip(got, views, strict=True):
                torch.testing.assert_close(g, v, rtol=0, atol=0)
        assert len(cache) == 2
        assert len(sync_log) == 2  # one copy of cos and of sin per request
        assert old is None or old() is None
        old = weakref.ref(tables[0])
        del tables, views, got
        sync_log.clear()  # the fixture's record holds the copies too
        gc.collect()


def test_shared_block_compile_binds_one_graph_across_blocks(monkeypatch, sync_log):
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 1)

    real_compile = torch.compile
    compile_calls = []
    compiled_graphs = []

    def _counting_eager(gm, example_inputs):
        del example_inputs
        compiled_graphs.append(gm)
        return gm

    def _eager_compile(fn=None, *, backend="inductor", fullgraph=False, dynamic=None):
        compile_calls.append((backend, fullgraph, dynamic))
        if fn is None:
            return lambda m: real_compile(
                m, backend=_counting_eager, fullgraph=fullgraph, dynamic=dynamic
            )
        return real_compile(
            fn, backend=_counting_eager, fullgraph=fullgraph, dynamic=dynamic
        )

    monkeypatch.setattr(torch, "compile", _eager_compile)
    torch._dynamo.reset()

    class _ToyBlock(torch.nn.Module):
        def __init__(self, scale: float, bias_val: float) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            self.register_buffer("bias_buf", torch.tensor(bias_val))
            with torch.no_grad():
                self.lin.weight.copy_(torch.eye(4) * scale)

        def forward(self, hs, ahs, ehs, aehs, **kwargs):
            del ehs, aehs
            mask = kwargs["video_self_attention_mask"]
            return (self.lin(hs) + self.bias_buf) * mask, ahs + 1.0

    class _ToyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=2,
                attention_head_dim=4,
                audio_num_attention_heads=2,
                audio_attention_head_dim=4,
            )
            self.transformer_blocks = torch.nn.ModuleList(
                [_ToyBlock(2.0, 1.0), _ToyBlock(3.0, 2.0), _ToyBlock(5.0, 3.0)]
            )

    try:
        model = _ToyTransformer()
        ltx2_compile._install_shared_block_compile(model)
        ltx2_compile._install_shared_block_compile(model)  # idempotent
        assert compile_calls == [("tpu", True, False)]

        mask = torch.ones(1, 1, 1)
        for _step in range(2):
            x = torch.ones(1, 2, 4)
            a = torch.zeros(1, 2, 4)
            e = torch.zeros(1, 2, 4)
            for blk in model.transformer_blocks:
                x, a = blk(x, a, e, e, video_self_attention_mask=mask)
            # (((1*2 + 1)*3 + 2)*5 + 3) = 58
            torch.testing.assert_close(x, torch.full((1, 2, 4), 58.0))
            torch.testing.assert_close(a, torch.full((1, 2, 4), 3.0))

        # All 3 blocks across both steps share a single fullgraph=True graph.
        assert len(compiled_graphs) == 1
    finally:
        torch._dynamo.reset()


def test_shared_block_compile_skips_heterogeneous_buffers(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 1)
    compile_calls = []
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *args, **kwargs: compile_calls.append((args, kwargs)) or (lambda f: f),
    )

    class _Block(torch.nn.Module):
        def __init__(self, with_adapter: bool) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(2, 2, bias=False)
            if with_adapter:
                self.register_buffer("lora_a", torch.zeros(2, 2))

    class _Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([_Block(True), _Block(False)])

    model = _Transformer()
    ltx2_compile._install_shared_block_compile(model)
    assert compile_calls == []
    assert not getattr(model, "_tpu_shared_blocks_compiled", False)


def test_shared_block_compile_initializes_sp_v2a_head_idx(monkeypatch, sync_log):
    sp_group = SimpleNamespace(
        ring_world_size=1,
        ulysses_world_size=2,
        ulysses_rank=1,
        ulysses_group="fake_pg",
    )
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_sp_group", lambda: sp_group)
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: (lambda fn: fn))

    class _SpBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
            inner = SimpleNamespace(
                ltx_sp_mode="video_to_audio",
                use_sync=False,
                _video_to_audio_strategy=None,
            )
            self.video_to_audio_attn = SimpleNamespace(attn=inner)

    class _SpTransformer(torch.nn.Module):
        def __init__(self, audio_heads: int) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=4,
                attention_head_dim=4,
                audio_num_attention_heads=audio_heads,
                audio_attention_head_dim=4,
            )
            self.transformer_blocks = torch.nn.ModuleList([_SpBlock(), _SpBlock()])

    model = _SpTransformer(audio_heads=6)
    ltx2_compile._install_shared_block_compile(model)
    expected = torch.tensor([3, 4, 5], dtype=torch.int32)
    for blk in model.transformer_blocks:
        strat = blk.video_to_audio_attn.attn._video_to_audio_strategy
        assert isinstance(strat, ltx2_sp_mod.LTX2VideoToAudioParallelAttention)
        torch.testing.assert_close(strat._tpu_head_idx, expected, rtol=0, atol=0)
    assert len(sync_log) == 1
    assert sync_log[0][1] is False

    bad_model = _SpTransformer(audio_heads=5)
    with pytest.raises(ValueError, match="divisible by ulysses_world_size"):
        ltx2_compile._install_shared_block_compile(bad_model)


def test_fresh_ltx2_forward_flushes_outputs(monkeypatch, sync_log):
    monkeypatch.setattr(ltx2_compile, "_patch_ltx2_traceable_collectives", lambda: None)
    monkeypatch.setattr(
        ltx2_compile, "_patch_ltx2_symmetric_perturbation_masks", lambda: None
    )
    monkeypatch.setattr(
        ltx2_transformer.LTX2VideoTransformer3DModel,
        "forward",
        lambda self, *args, **kwargs: (
            torch.ones(1, 4),
            torch.ones(1, 2),
        ),
    )

    ltx2_compile._activate_ltx2_compile_runtime_hooks()
    dummy = object.__new__(ltx2_transformer.LTX2VideoTransformer3DModel)
    got = ltx2_transformer.LTX2VideoTransformer3DModel.forward(
        dummy,
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([500.0]),
        torch.tensor([500.0]),
        sigma=torch.tensor([0.5]),
    )
    assert len(got) == 2
    assert len(sync_log) == 2
    assert len(sync_log[0][0]) == 3 and sync_log[0][1] is False
    assert len(sync_log[1][0]) == 2 and sync_log[1][1] is False


@pytest.mark.parametrize(
    ("model_cls", "should_compile"),
    [
        (FluxTransformer2DModel, True),
        (QwenImageTransformer2DModel, True),
        (WanTransformer3DModel, False),
        (torch.nn.Module, False),
    ],
)
def test_tpu_diffusion_compile_hook_gates_to_verified_families(
    monkeypatch, model_cls, should_compile
):
    offload_calls = []

    def _fake_enable_offload(od_config, pipeline, device=None):
        offload_calls.append((od_config, pipeline, device))
        return pipeline, "offload-token"

    monkeypatch.setattr(
        diffusion_model_runner, "enable_offload_backend", _fake_enable_offload
    )
    regional_calls = []
    sentinel = object()

    def _fake_regionally_compile(model, **kwargs):
        regional_calls.append((model, kwargs))
        return sentinel

    monkeypatch.setattr(
        diffusion_compile, "regionally_compile", _fake_regionally_compile
    )
    skip_logs = []
    monkeypatch.setattr(
        ltx2_compile.logger,
        "info_once",
        lambda msg, *args: skip_logs.append(msg % args if args else msg),
    )

    ltx2_compile.apply_ltx2_compile_patches()
    patched = diffusion_model_runner.enable_offload_backend
    ltx2_compile.apply_ltx2_compile_patches()  # idempotent
    assert diffusion_model_runner.enable_offload_backend is patched

    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    pipeline = SimpleNamespace(transformer=model)
    od_config = SimpleNamespace(
        enforce_eager=False,
        diffusion_compile_granularity="regional",
    )

    out_pipeline, token = diffusion_model_runner.enable_offload_backend(
        od_config, pipeline, device="tpu"
    )
    assert out_pipeline is pipeline
    assert token == "offload-token"
    assert offload_calls == [(od_config, pipeline, "tpu")]

    if should_compile:
        assert regional_calls == [(model, {"backend": "tpu", "dynamic": False})]
        assert pipeline.transformer is sentinel
        assert skip_logs == []
    else:
        assert regional_calls == []
        assert pipeline.transformer is model
        assert len(skip_logs) == 1
        assert model_cls.__name__ in skip_logs[0]


def test_tpu_diffusion_compile_hook_ltx2_and_inductor_isolation(monkeypatch):
    # Platform flag stays False so non-diffusion models (Qwen3 code predictor,
    # VoxCPM2 talker) and upstream Inductor setup_compile() hooks stay eager.
    assert OmniTpuPlatform.supports_torch_inductor() is False

    monkeypatch.setattr(
        diffusion_model_runner,
        "enable_offload_backend",
        lambda od_config, pipeline, device=None: (pipeline, None),
    )
    hooks_activated = []
    shared_compiled = []
    monkeypatch.setattr(
        ltx2_compile,
        "_activate_ltx2_compile_runtime_hooks",
        lambda: hooks_activated.append(True),
    )
    monkeypatch.setattr(
        ltx2_compile,
        "_install_shared_block_compile",
        lambda tr: shared_compiled.append(tr),
    )
    regional_calls = []
    monkeypatch.setattr(
        diffusion_compile,
        "regionally_compile",
        lambda model, **kwargs: regional_calls.append((model, kwargs)) or model,
    )

    ltx2_compile.apply_ltx2_compile_patches()

    # 1. LTX2Pipeline compiles via shared-block compile when enforce_eager=False.
    ltx_pipe = object.__new__(pipeline_ltx2.LTX2Pipeline)
    torch.nn.Module.__init__(ltx_pipe)
    tr1 = torch.nn.Module()
    ltx_pipe.transformer = tr1

    eager_cfg = SimpleNamespace(
        enforce_eager=True, diffusion_compile_granularity="regional"
    )
    compile_cfg = SimpleNamespace(
        enforce_eager=False, diffusion_compile_granularity="regional"
    )

    diffusion_model_runner.enable_offload_backend(eager_cfg, ltx_pipe)
    assert hooks_activated == []
    assert shared_compiled == []

    diffusion_model_runner.enable_offload_backend(compile_cfg, ltx_pipe)
    assert hooks_activated == [True]
    assert shared_compiled == [tr1]
    assert regional_calls == []

    # 1b. The two-stage pipeline (not an LTX2Pipeline subclass) uses the same
    # shared-block compile instead of falling back to eager.
    two_stage_pipe = object.__new__(pipeline_ltx2_two_stage.LTX2TwoStagePipeline)
    torch.nn.Module.__init__(two_stage_pipe)
    tr2 = torch.nn.Module()
    two_stage_pipe.transformer = tr2
    assert not isinstance(two_stage_pipe, pipeline_ltx2.LTX2Pipeline)
    diffusion_model_runner.enable_offload_backend(eager_cfg, two_stage_pipe)
    assert shared_compiled == [tr1]
    diffusion_model_runner.enable_offload_backend(compile_cfg, two_stage_pipe)
    assert shared_compiled == [tr1, tr2]
    assert regional_calls == []
    shared_compiled.pop()

    # 2. Upstream custom setup_compile() pipelines (e.g. LTX2TextToAudioPipeline)
    # are neither compiled as LTX2Pipeline nor allowed to invoke Inductor setup_compile.
    audio_pipe = object.__new__(pipeline_ltx2_audio.LTX2TextToAudioPipeline)
    torch.nn.Module.__init__(audio_pipe)
    audio_pipe.transformer = torch.nn.Module()
    upstream_setup_calls = []
    monkeypatch.setattr(
        audio_pipe,
        "setup_compile",
        lambda: upstream_setup_calls.append(True),
    )
    diffusion_model_runner.enable_offload_backend(compile_cfg, audio_pipe)
    assert upstream_setup_calls == []
    assert len(shared_compiled) == 1
    assert regional_calls == []

    # 3. Verified non-LTX2 models also honor enforce_eager=True.
    flux_model = object.__new__(FluxTransformer2DModel)
    torch.nn.Module.__init__(flux_model)
    flux_pipe = SimpleNamespace(transformer=flux_model)
    diffusion_model_runner.enable_offload_backend(eager_cfg, flux_pipe)
    assert regional_calls == []


def test_tpu_diffusion_compile_hook_full_granularity_and_missing_attr(monkeypatch):
    monkeypatch.setattr(
        diffusion_model_runner,
        "enable_offload_backend",
        lambda od_config, pipeline, device=None: (pipeline, None),
    )
    regional_calls = []
    monkeypatch.setattr(
        diffusion_compile,
        "regionally_compile",
        lambda model, **kwargs: regional_calls.append((model, kwargs)) or model,
    )

    ltx2_compile.apply_ltx2_compile_patches()

    flux_model = object.__new__(FluxTransformer2DModel)
    torch.nn.Module.__init__(flux_model)
    full_calls = []
    monkeypatch.setattr(
        flux_model, "compile", lambda **kwargs: full_calls.append(kwargs)
    )
    wan_model = object.__new__(WanTransformer3DModel)
    torch.nn.Module.__init__(wan_model)
    monkeypatch.setattr(
        wan_model, "compile", lambda **kwargs: full_calls.append(("wan", kwargs))
    )

    od_config = SimpleNamespace(
        enforce_eager=False, diffusion_compile_granularity="full"
    )
    pipeline = SimpleNamespace(transformer=flux_model, transformer_2=None)
    diffusion_model_runner.enable_offload_backend(od_config, pipeline)
    assert full_calls == [{"backend": "tpu", "dynamic": False}]
    assert regional_calls == []

    # Full granularity is also skipped for unverified families like Wan.
    pipeline.transformer = wan_model
    diffusion_model_runner.enable_offload_backend(od_config, pipeline)
    assert full_calls == [{"backend": "tpu", "dynamic": False}]
    assert regional_calls == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_gates", [True, False])
def test_mxu_gated_attention_matches_upstream(monkeypatch, dtype, use_gates):
    monkeypatch.setattr(
        ltx2_transformer.LTX2AudioVideoAttnProcessor,
        "__call__",
        _UPSTREAM_ATTN_PROC_CALL,
    )
    ltx2_compile._patch_ltx2_mxu_gated_attention()
    ltx2_compile._patch_ltx2_mxu_gated_attention()  # idempotent

    torch.manual_seed(0)
    heads, head_dim = 4, 8
    dim = heads * head_dim
    gate_proj = torch.nn.Linear(dim, heads).to(dtype=dtype) if use_gates else None
    q_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    k_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    v_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)
    out_proj = torch.nn.Linear(dim, dim, bias=False).to(dtype=dtype)

    class _StubAttn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = head_dim
            self.rope_type = "split"
            self.to_gate_logits = gate_proj
            self.norm_q = torch.nn.Identity()
            self.norm_k = torch.nn.Identity()
            self.to_out = torch.nn.ModuleList([out_proj, torch.nn.Identity()])
            self.attn = lambda q, k, v, meta=None: (q + k + v) * 0.25

    proc = ltx2_transformer.LTX2AudioVideoAttnProcessor()
    monkeypatch.setattr(
        proc,
        "_prepare_attention_mask",
        lambda **kw: kw["attention_mask"],
    )
    monkeypatch.setattr(
        proc,
        "_project_qkv",
        lambda attn, hidden_states, encoder_hidden_states, is_self_attention: (
            q_proj(hidden_states),
            k_proj(encoder_hidden_states),
            v_proj(encoder_hidden_states),
        ),
    )

    attn_mod = _StubAttn()
    hs = torch.randn(2, 6, dim, dtype=dtype)
    pmask = torch.tensor([[[1.0]], [[0.0]]], dtype=dtype)

    got = proc(attn_mod, hs, perturbation_mask=pmask)
    want = _UPSTREAM_ATTN_PROC_CALL(proc, attn_mod, hs, perturbation_mask=pmask)
    assert got.dtype == dtype
    # One 1.0 entry per column in expand_mat makes bf16 indicator matmul bit-exact.
    torch.testing.assert_close(got, want, rtol=0, atol=0)
