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
"""DCP draft pair merge (TPU_DRAFT_DCP_PAIR_MERGE): `forward_with_dcp`'s
dispatch, `arm_pair_merge`'s wiring and refusals, and the torch op's trace
(one graph for every draft bucket). The exchange kernel itself:
tests/kernels/collectives/test_pair_lse_merge.py."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_torchtpu.layers.adapter import cp_draft_attention as cda  # noqa: E402

H, D = 8, 128
# Draft buckets (tokens) and a prefill-sized forward past the kernel's cap.
TOKENS = (16, 32, 64, 128, 256, 512, 4096)


def _recording_ops(calls):
    def attend(q_all, kv_cache, seq_lens, tables, starts, rank):
        calls.append(("attend", tuple(q_all.shape)))
        t = q_all.shape[0]
        return torch.zeros(t, 2 * H, D), torch.zeros(t, 2 * H)

    def pair_merge(out, lse, meta):
        calls.append(("pair_merge", tuple(out.shape), tuple(meta.shape)))
        return out[:, :H], lse[:, :H]

    def block(q, key, value, seq_lens, starts, ctx_out, ctx_lse):
        calls.append(("block", tuple(ctx_out.shape)))
        return q

    return dict(attend=attend, pair_merge=pair_merge, block=block)


def _forward(t, ops, pair_meta, head_group):
    md = SimpleNamespace(
        seq_lens=torch.zeros(4, dtype=torch.int32),
        block_tables=torch.zeros(4, 8, dtype=torch.int32),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
    )
    return cda.forward_with_dcp(
        query=torch.zeros(t, H, D, dtype=torch.bfloat16),
        key=torch.zeros(t, 1, D, dtype=torch.bfloat16),
        value=torch.zeros(t, 1, D, dtype=torch.bfloat16),
        kv_cache=torch.zeros(1, dtype=torch.bfloat16),
        attn_metadata=md,
        dcp_rank=torch.zeros(1, dtype=torch.int32),
        dcp=4,
        kv_head_replicas=2,
        head_group=head_group,
        k_scale=None,
        v_scale=None,
        ops=ops,
        pair_meta=pair_meta,
    )


class _Group:
    world_size = 2
    rank_in_group = 0
    ranks = (0, 1)

    def all_gather(self, x, dim):
        return torch.cat([x, x], dim=dim)


def test_forward_dispatch_same_ops_every_bucket(monkeypatch):
    """With ``pair_meta`` the partials take ``ops["pair_merge"]`` (never
    cp_lse_ag_out_rs); without it cp_lse_ag_out_rs, as before. Either way the
    op sequence is one for every bucket: the kernel / permute choice is made
    inside the op on its static shape, so the traced torch program does not
    branch on the token count."""
    from vllm_torchtpu.distributed import dcp_comm

    seen = []

    def fake_ag_rs(out, lse, group, return_lse=False):
        seen.append(tuple(out.shape))
        return out[:, :H], lse[:, :H]

    monkeypatch.setattr(dcp_comm, "cp_lse_ag_out_rs", fake_ag_rs)
    meta = torch.tensor([[1, 0]], dtype=torch.int32)
    patterns = {True: set(), False: set()}
    for t in TOKENS:
        for armed in (True, False):
            calls = []
            _forward(t, _recording_ops(calls), meta if armed else None, _Group())
            names = tuple(c[0] for c in calls)
            patterns[armed].add(names)
            if armed:
                assert calls[1] == ("pair_merge", (t, 2 * H, D), (1, 2))
    assert patterns[True] == {("attend", "pair_merge", "block")}
    assert patterns[False] == {("attend", "block")}
    # cp_lse_ag_out_rs ran only unarmed, once per bucket.
    assert seen == [(t, 2 * H, D) for t in TOKENS]


def test_pair_perm():
    assert cda._pair_perm([(10, 11), (12, 13), (10, 11)], [10, 11, 12, 13]) == (
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
    )
    assert cda._pair_perm([(0, 4), (1, 5)], list(range(8)))[:2] == ((0, 4), (4, 0))


class _Layer(torch.nn.Module):
    def __init__(self, head_group):
        super().__init__()
        self.impl = SimpleNamespace(
            sharded_draft_dcp=dict(head_group=head_group, ops=dict(attend=1)),
            num_heads=H,
            head_size=D,
        )
        self.register_buffer("draft_dcp_rank", torch.zeros(1, dtype=torch.int32))


def _arm_env(monkeypatch, armed, rank_in_group=1):
    import vllm.distributed

    from vllm_torchtpu.distributed import pallas_all_reduce

    tp = SimpleNamespace(world_size=2, ranks=[0, 1], rank_in_group=rank_in_group,
                         cpu_group=None)
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: tp)
    monkeypatch.setattr(pallas_all_reduce, "armed_for", lambda ranks: armed)

    def gather(out, obj, group=None):
        out[:] = [obj] * len(out)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)


def test_arm_refuses_without_pallas_all_reduce(monkeypatch):
    group = _Group()
    model = torch.nn.Sequential(_Layer(group), _Layer(group))
    for armed in (None, SimpleNamespace(mesh=None, rank_ordered=True),
                  SimpleNamespace(mesh="m", rank_ordered=False)):
        _arm_env(monkeypatch, armed)
        assert cda.arm_pair_merge(model) == 0
        for layer in model:
            assert "pair_merge" not in layer.impl.sharded_draft_dcp["ops"]
            assert not hasattr(layer, "draft_pair_meta")


def test_arm_refuses_other_head_groups(monkeypatch):
    group = _Group()
    group.world_size = 4
    model = torch.nn.Sequential(_Layer(group))
    _arm_env(monkeypatch, SimpleNamespace(mesh="m", rank_ordered=True))
    assert cda.arm_pair_merge(model) == 0


@pytest.mark.parametrize("selftest", [True, False])
def test_arm_wires_layers(monkeypatch, selftest):
    group = _Group()
    group.rank_in_group = 1
    model = torch.nn.Sequential(_Layer(group), _Layer(group))
    _arm_env(monkeypatch, SimpleNamespace(mesh="mesh", rank_ordered=True))
    built = []
    monkeypatch.setattr(
        cda, "_pair_merge_op", lambda mesh, perm: built.append((mesh, perm)) or "op"
    )
    monkeypatch.setattr(cda, "_pair_selftest", lambda *a: selftest)
    n = cda.arm_pair_merge(model)
    assert built == [("mesh", ((0, 1), (1, 0)))]
    if not selftest:
        assert n == 0
        assert "pair_merge" not in model[0].impl.sharded_draft_dcp["ops"]
        return
    assert n == 2
    for layer in model:
        ops = layer.impl.sharded_draft_dcp["ops"]
        assert ops == dict(attend=1, pair_merge="op")
        # (peer's mesh index, this rank's block in the head group)
        assert layer.draft_pair_meta.tolist() == [[0, 1]]


def _cpu_mesh():
    jax = pytest.importorskip("jax")
    pytest.importorskip("torch_tpu")
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices (XLA_FLAGS host device count)")
    from jax.sharding import Mesh

    return Mesh(np.array(jax.devices()[:8]), ("x",))


def test_pair_merge_op_one_dynamo_graph_for_all_buckets():
    """The constructed torch op (pallas.jax_op) traces with a dynamic token
    dim under torch.compile(fullgraph=True) over meta tensors: one graph and
    the [T, H, D] / [T, H] outputs for every draft bucket and a large T."""
    mesh = _cpu_mesh()
    cda._pair_merge_op.cache_clear()
    op = cda._pair_merge_op(mesh, ((0, 1), (1, 0)))
    frames = []

    def backend(gm, example_inputs):
        frames.append(gm)
        return gm.forward

    torch._dynamo.reset()
    fn = torch.compile(lambda o, ls, m: op(o, ls, m), backend=backend, fullgraph=True)
    meta = torch.empty((1, 2), dtype=torch.int32, device="meta")
    for t in TOKENS:
        out = torch.empty((t, 2 * H, D), dtype=torch.bfloat16, device="meta")
        lse = torch.empty((t, 2 * H), dtype=torch.float32, device="meta")
        torch._dynamo.mark_dynamic(out, 0)
        torch._dynamo.mark_dynamic(lse, 0)
        o, ol = fn(out, lse, meta)
        assert tuple(o.shape) == (t, H, D) and o.dtype == torch.bfloat16
        assert tuple(ol.shape) == (t, H) and ol.dtype == torch.float32
    assert len(frames) == 1, len(frames)


def test_armed_for(monkeypatch):
    from vllm_torchtpu.distributed import pallas_all_reduce as par

    rank = torch.zeros((1, 1), dtype=torch.int32)
    # Records built without the new fields keep their defaults.
    assert par._Armed(ranks=(0, 1), op=None, rank=rank).mesh is None
    rec = par._Armed(ranks=(0, 1), op=None, rank=rank, mesh="m", max_bytes=8)
    monkeypatch.setattr(par, "_ARMED", rec)
    assert par.armed_for([0, 1]) is rec
    assert par.armed_for([0, 2]) is None
    monkeypatch.setattr(par, "_ARMED", None)
    assert par.armed_for([0, 1]) is None
