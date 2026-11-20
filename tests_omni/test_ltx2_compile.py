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
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from vllm_omni.diffusion.attention.parallel import ulysses as ulysses_mod
from vllm_omni.diffusion.distributed import comm
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_guidance,
)
from vllm_omni.diffusion.models.ltx2 import (
    ltx2_sequence_parallel as ltx2_sp_mod,
)

from vllm_torchtpu.omni import ltx2_compile

pytestmark = pytest.mark.omni

_UPSTREAM_BUILD_PERTURBATION = ltx2_guidance.build_perturbation_kwargs
_UPSTREAM_PREDICT_PARALLEL = ltx2_guidance.LTXGuidanceExecutor.predict_parallel_guidance
_UPSTREAM_SEQ_A2A = comm.SeqAllToAll4D
_UPSTREAM_V2A_PRE = ltx2_sp_mod.LTX2VideoToAudioParallelAttention.pre_attention
_UPSTREAM_V2A_POST = ltx2_sp_mod.LTX2VideoToAudioParallelAttention.post_attention


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
