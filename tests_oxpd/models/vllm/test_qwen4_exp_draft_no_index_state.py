# SPDX-License-Identifier: Apache-2.0
"""q38 X-D14-DRAFT-1 (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE): the torch side.

- ``Qwen4ExpQSA.bind_pool`` builds the no-index-state shared ops for the MTP draft's
  layers only, and only with VLLM_TPU_QSA_DRAFT_SHARED_SELECTION (replicated and the
  VLLM_TPU_QSA_MTP_KV_DCP draft ops); flag off builds exactly what it built before.
- ``attend_with_indices`` over an 8-pass draft (step 0 with rows, 7 all-token passes):
  flag on calls the ``_noidx`` ops, never runs ``index_qk_proj`` and passes ``hidden``
  as the index operand; flag off runs it every pass. Every other operand, the shared
  selection each pass attends and the carried selection (``_selection_rows``) are
  identical between the two. A draft whose shared selection is NOT bound keeps its
  own indexer and the full index projection even with the flag on.
- ``qsa_op``: the builders register the ``_noidx`` jax ops on the ``_noidx`` steps, and
  the flag-off builders the unchanged ops on the unchanged steps.

CPU only: the custom ops are recorded, not run (the kernels are proven in
tests/kernels/qwen4_exp/dcp2/test_dcp2_draft_no_index_state.py).
"""

from types import SimpleNamespace as NS

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config as attention_config

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.layers.adapter.custom_ops import qsa_op
from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_module
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata, Qwen4ExpQSA

DRAFT = "mtp.layers.2.self_attn"
TARGET = "model.layers.1.self_attn"
FLAGS = ("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE",
         "VLLM_TPU_QSA_MAIN_KV_DCP", "VLLM_TPU_QSA_MTP_KV_DCP", "VLLM_TPU_QSA_COMPRESSED_DCP",
         "VLLM_TPU_MEGAKERNEL", "VLLM_TPU_QSA_INDEXER_SHARD", "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD",
         "VLLM_TPU_QSA_DECODE_REQUEST_SHARD")


@pytest.fixture(autouse=True)
def cpu_vllm_config_context(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (("get_tensor_model_parallel_rank", 0),
                        ("get_tensor_model_parallel_world_size", 1)):
        monkeypatch.setattr(f"vllm.model_executor.parameter.{name}", lambda value=value: value)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _env(monkeypatch, *, shared, no_index, dcp=0):
    monkeypatch.setenv("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "1" if shared else "0")
    if no_index is None:
        monkeypatch.delenv("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", "1" if no_index else "0")
    if dcp:
        monkeypatch.setenv("VLLM_TPU_QSA_MAIN_KV_DCP", str(dcp))
        monkeypatch.setenv("VLLM_TPU_QSA_MTP_KV_DCP", "1")


class _Op:
    """A recorded custom op: logs its operands, returns shaped outputs."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, *args, **kwargs):
        self.log.append(NS(name=self.name, args=args, kwargs=kwargs))
        qgate = args[0]
        shared_op = "shared" in self.name
        rows = args[16] if "rows" in self.name else None
        selected = args[-1] if shared_op else None
        count = len(rows) if rows is not None else len(qgate)
        out = torch.full((count, qgate.shape[1] // 2), float(len(self.log)), dtype=qgate.dtype)
        if selected is None:
            selected = torch.full((count, 135), -1, dtype=torch.int32)
        return out, selected


def _record_builders(monkeypatch):
    """Replace the qsa_op builders bind_pool imports with recorders."""
    log, built = [], []

    def builder(kind):
        def build(*args, **kwargs):
            built.append((kind, args, kwargs))
            label, suffix = kind, ""
            if kind == "shared":
                suffix = "_rows" if args[0] else ""
            elif kind == "draft_dcp":
                suffix = "_rows" if kwargs["with_rows"] else ""
                label = "shared_dcp" if kwargs["shared"] else "rows_dcp"
            if kwargs.get("index_state") is False:
                suffix += "_noidx"
            return _Op(label + suffix, log)

        return build

    for name, kind in (("build_pooled_qsa_op", "plain"), ("build_pooled_qsa_rows_op", "rows"),
                       ("build_pooled_qsa_shared_op", "shared"),
                       ("build_pooled_qsa_dcp_op", "dcp"),
                       ("build_pooled_qsa_draft_dcp_op", "draft_dcp")):
        monkeypatch.setattr(qsa_op, name, builder(kind))
    monkeypatch.setattr(qsa_module, "main_kv_dcp_group", lambda size: NS(size=size))
    return log, built


def _layer(prefix, *, tp_size=1, seed=0):
    config = attention_config(
        hidden_size=128, num_attention_heads=8, num_key_value_heads=4, head_dim=64,
        indexer_n_heads=2, indexer_head_dim=32, indexer_budget=128, indexer_compress_ratio=8,
        rms_norm_eps=0.01, partial_rotary_factor=0.5, rope_parameters={"rope_theta": 12345.0},
        max_position_embeddings=256,
    )
    with torch.device("cpu"):
        layer = Qwen4ExpQSA(config=config, tp_size=tp_size, tp_rank=0, max_model_len=256,
                            prefix=prefix, pooled=True, max_query_len=4)
    gen = torch.Generator().manual_seed(seed)
    # Deterministic stand-ins for the projections (the linear layers' dispatch is not
    # under test); index_qk_proj counts its calls.
    weights = {name: torch.randn(width, 128, generator=gen).to(torch.bfloat16) * 0.1
               for name, width in (("q", 1024), ("k", 256), ("v", 256), ("i", 96))}

    class Proj(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight_t, self.calls = weight, 0

        def forward(self, x):
            self.calls += 1
            return torch.nn.functional.linear(x, self.weight_t)

    layer.q_proj, layer.k_proj, layer.v_proj = (Proj(weights[n]) for n in "qkv")
    layer.indexer.index_qk_proj = Proj(weights["i"])
    with torch.no_grad():
        for norm in (layer.q_norm, layer.k_norm, layer.indexer.q_layernorm,
                     layer.indexer.k_layernorm):
            norm.weight.copy_(torch.randn(norm.weight.shape, generator=gen).to(norm.weight.dtype))
    return layer


def _bind_selection(layer, max_tokens=16, seed=1):
    """What Qwen4ExpMTP.bind_shared_selection registers on a draft layer."""
    gen = torch.Generator().manual_seed(seed)
    width = layer.max_selected_tokens
    sink = torch.randint(-1, 500, (max_tokens, width), generator=gen, dtype=torch.int32)
    layer.register_buffer("_selection_source", sink, persistent=False)
    layer.register_buffer("_selection_rows", torch.full_like(sink, -1), persistent=False)


def _forward_context(monkeypatch, layer):
    table = torch.zeros(4, dtype=torch.int32)
    metadata = {name: NS(block_tables=table) for name in (
        layer.prefix + ".raw_ring", layer.main_cache_name, layer.compressed_cache_name)}
    monkeypatch.setattr(qsa_module, "get_forward_context", lambda: NS(attn_metadata=metadata))
    monkeypatch.setattr(qsa_module, "update_rows_",
                        lambda buf, rows: buf[: rows.shape[0]].copy_(rows))


def _draft_passes(layer, steps=8, seed=2):
    """Step 0: a 3-request verify chunk attended at each request's last row; then
    one-token passes (8 padded rows) reusing the carried selection."""
    gen = torch.Generator().manual_seed(seed)
    metadata = QSAMetadata(torch.tensor([0, 3, 7, 8]), torch.tensor([40, 90, 7]),
                           torch.zeros(3, dtype=torch.int32), torch.zeros(3, 2, dtype=torch.int32),
                           torch.zeros(3, 2, dtype=torch.int32))
    outputs = []
    for step in range(steps):
        hidden = torch.randn(8, 128, generator=gen).to(torch.bfloat16)
        positions = torch.arange(8, dtype=torch.int64) + step
        rows = torch.tensor([2, 6, 7]) if step == 0 else None
        outputs.append(layer.attend_with_indices(hidden, positions, metadata, rows=rows))
        outputs[-1] = (hidden, *outputs[-1])
    return outputs


def _run(monkeypatch, *, no_index, bind=True, prefix=DRAFT):
    _env(monkeypatch, shared=True, no_index=no_index)
    log, built = _record_builders(monkeypatch)
    layer = _layer(prefix)
    layer.bind_pool(torch.zeros(4, 8))
    if bind:
        _bind_selection(layer)
    _forward_context(monkeypatch, layer)
    passes = _draft_passes(layer)
    return layer, log, built, passes


@pytest.mark.parametrize("no_index", [None, False])
def test_flag_off_builds_and_calls_the_unchanged_shared_ops(monkeypatch, no_index):
    """Unset and 0: the shared ops are built as before and get index_qk_proj(hidden)."""
    layer, log, built, passes = _run(monkeypatch, no_index=no_index)
    assert layer._shared_no_index_state is False
    assert [b for b in built if b[0] == "shared"] == [("shared", (False,), {}), ("shared", (True,), {})]
    assert [entry.name for entry in log] == ["shared_rows"] + ["shared"] * 7
    assert layer.indexer.index_qk_proj.calls == 8
    for entry, (hidden, _, _) in zip(log, passes):
        want = torch.nn.functional.linear(hidden, layer.indexer.index_qk_proj.weight_t)
        assert torch.equal(entry.args[3], want)


def test_flag_on_draft_takes_noidx_ops_without_the_index_projection(monkeypatch):
    on, on_log, on_built, on_passes = _run(monkeypatch, no_index=True)
    assert on._shared_no_index_state is True
    assert [b for b in on_built if b[0] == "shared"] == [
        ("shared", (False,), {"index_state": False}), ("shared", (True,), {"index_state": False})]
    assert [entry.name for entry in on_log] == ["shared_rows_noidx"] + ["shared_noidx"] * 7
    assert on.indexer.index_qk_proj.calls == 0
    for entry, (hidden, _, _) in zip(on_log, on_passes):
        assert entry.args[3] is hidden
    # The same draft with the flag off: every other operand, the selection each pass
    # attends, the outputs and the carried selection are identical.
    off, off_log, _, off_passes = _run(monkeypatch, no_index=False)
    assert len(on_log) == len(off_log) == 8
    for step, (a, b) in enumerate(zip(on_log, off_log)):
        assert len(a.args) == len(b.args), step
        for i, (x, y) in enumerate(zip(a.args, b.args)):
            if i == 3:
                continue
            assert (torch.equal(x, y) if isinstance(x, torch.Tensor) else x is y), (step, i)
        assert a.kwargs == b.kwargs
    for (_, out_a, sel_a), (_, out_b, sel_b) in zip(on_passes, off_passes):
        assert torch.equal(out_a, out_b) and torch.equal(sel_a, sel_b)
    assert torch.equal(on._selection_rows, off._selection_rows)
    source = on._selection_source
    # Step 0 attended the source rows of each request's last token; later passes the
    # carried copy of them (then -1 rows).
    assert torch.equal(on_log[0].args[-1], source[torch.tensor([2, 6, 7])])
    assert torch.equal(on._selection_rows[:3], source[torch.tensor([2, 6, 7])])
    assert (on._selection_rows[3:] == -1).all()
    for entry in on_log[1:]:
        assert torch.equal(entry.args[-1], on._selection_rows[:8])


def test_flag_on_without_a_bound_selection_keeps_the_drafts_own_indexer(monkeypatch):
    """bind_shared_selection did not bind: the regular ops, with the full index operand."""
    layer, log, _, passes = _run(monkeypatch, no_index=True, bind=False)
    assert [entry.name for entry in log] == ["rows"] + ["plain"] * 7
    assert layer.indexer.index_qk_proj.calls == 8
    for entry, (hidden, _, _) in zip(log, passes):
        assert entry.args[3] is not hidden
        assert torch.equal(entry.args[3],
                           torch.nn.functional.linear(hidden, layer.indexer.index_qk_proj.weight_t))


def test_flag_on_leaves_target_layers_unchanged(monkeypatch):
    _env(monkeypatch, shared=True, no_index=True)
    _, built = _record_builders(monkeypatch)
    layer = _layer(TARGET)
    layer.bind_pool(torch.zeros(4, 8))
    assert layer._shared_no_index_state is False
    assert [b for b in built if b[0] == "shared"] == [("shared", (False,), {}), ("shared", (True,), {})]


def test_flag_needs_shared_selection(monkeypatch):
    _env(monkeypatch, shared=False, no_index=True)
    _, built = _record_builders(monkeypatch)
    layer = _layer(DRAFT)
    layer.bind_pool(torch.zeros(4, 8))
    assert layer._shared_no_index_state is False and layer._shared_ops is None
    assert not [b for b in built if b[0] == "shared"]


@pytest.mark.parametrize("no_index", [False, True])
def test_dcp_draft_ops_take_index_state_for_the_shared_ops_only(monkeypatch, no_index):
    """VLLM_TPU_QSA_MTP_KV_DCP: the draft's rows op keeps its indexer; its shared ops
    are the no-index-state ones with the flag on."""
    _env(monkeypatch, shared=True, no_index=no_index, dcp=2)
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: NS(world_size=2, rank_in_group=0))
    log, built = _record_builders(monkeypatch)
    layer = _layer(DRAFT, tp_size=2)
    assert layer.main_kv_dcp == 2
    layer.bind_pool(torch.zeros(4, 8))
    drafts = [(kw["shared"], kw["with_rows"], kw.get("index_state", True))
              for kind, _, kw in built if kind == "draft_dcp"]
    want_state = not no_index
    assert drafts == [(False, True, True), (True, False, want_state), (True, True, want_state)]
    suffix = "_noidx" if no_index else ""
    assert [op.func.name for op in layer._shared_ops] == [f"shared_dcp{suffix}",
                                                          f"shared_dcp_rows{suffix}"]
    assert layer._rows_op.func.name == "rows_dcp_rows"


# ----------------------------------------------------------------------------
# qsa_op builders


class _JaxOp:
    def __init__(self, name, step, log):
        self.name, self.step, self.log = name, step, log

    def register_fake(self, fn):
        return fn

    def __call__(self, *args, **kwargs):
        self.log.append(self.name)
        pool, qgate, nq = args[10], args[0], args[4]
        rows = args[16].shape[0] if "rows" in self.name else qgate.shape[0]
        heads = qgate.shape[1] // (2 * nq.shape[0])
        return (torch.empty_like(pool), torch.zeros(rows, heads, nq.shape[0]),
                torch.zeros(rows, heads), torch.zeros(rows, 7, dtype=torch.int32))


@pytest.fixture
def recorded_jax_ops(monkeypatch):
    """qsa_op's jax_op replaced by a recorder; its cached builders rebuilt around it."""
    created, calls = {}, []

    def jax_op(name, step, **_):
        created[name] = step
        return _JaxOp(name, step, calls)

    caches = (qsa_op.build_pooled_qsa_shared_op, qsa_op._dcp_draft_ops,
              qsa_op._dcp_draft_noidx_ops, qsa_op._dcp_ops)
    for cache in caches:
        cache.cache_clear()
    monkeypatch.setattr(qsa_op, "pallas", NS(jax_op=jax_op))
    monkeypatch.setattr(qsa_op, "_pair_gather_qgate", lambda qgate, *_: qgate)
    monkeypatch.setattr(qsa_op, "_dcp_finish", lambda out, *_: out)
    yield created, calls
    for cache in caches:
        cache.cache_clear()


def test_shared_op_builders_register_the_noidx_steps(recorded_jax_ops):
    created, _ = recorded_jax_ops
    qsa_op.build_pooled_qsa_shared_op(False)
    qsa_op.build_pooled_qsa_shared_op(True)
    assert created == {
        "qwen4_exp::pooled_qsa_shared": pooled_qsa.pooled_qsa_shared_step,
        "qwen4_exp::pooled_qsa_shared_rows": pooled_qsa.pooled_qsa_shared_rows_step,
    }
    created.clear()
    qsa_op.build_pooled_qsa_shared_op(False, index_state=False)
    qsa_op.build_pooled_qsa_shared_op(True, index_state=False)
    assert created == {
        "qwen4_exp::pooled_qsa_shared_noidx": pooled_qsa.pooled_qsa_shared_noidx_step,
        "qwen4_exp::pooled_qsa_shared_rows_noidx": pooled_qsa.pooled_qsa_shared_rows_noidx_step,
    }


def _dcp_args(with_rows):
    t = 4
    args = [torch.zeros(t, 2 * 6 * 8), torch.zeros(t, 8), torch.zeros(t, 8), torch.zeros(t, 8),
            torch.zeros(8), torch.zeros(8), torch.zeros(8), torch.zeros(8),
            torch.zeros(t, dtype=torch.int64), torch.zeros(16, 8), torch.zeros(3, 4),
            torch.zeros(3, dtype=torch.int64), torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64),
            torch.zeros(2, 2, dtype=torch.int64)]
    if with_rows:
        args.append(torch.tensor([1, 3]))
    args.append(torch.zeros(2 if with_rows else t, 7, dtype=torch.int32))
    return args


@pytest.mark.parametrize("with_rows", [False, True])
@pytest.mark.parametrize("index_state", [True, False])
def test_draft_dcp_op_runs_the_noidx_step(recorded_jax_ops, with_rows, index_state):
    created, calls = recorded_jax_ops
    run = qsa_op.build_pooled_qsa_draft_dcp_op(
        None, 2, shared=True, with_rows=with_rows, **({} if index_state else {"index_state": False}))
    run(*_dcp_args(with_rows), dcp_rank=torch.tensor([0], dtype=torch.int32),
        compress_ratio=4, token_topk=4, norm_eps=1e-6, rotary_dim=4)
    name = "qwen4_exp::pooled_qsa_shared" + ("_rows" if with_rows else "") + "_dcp"
    if not index_state:
        name += "_noidx"
    assert calls == [name]
    want = {
        "qwen4_exp::pooled_qsa_shared_dcp": pooled_qsa.pooled_qsa_shared_dcp_step,
        "qwen4_exp::pooled_qsa_shared_rows_dcp": pooled_qsa.pooled_qsa_shared_rows_dcp_step,
        "qwen4_exp::pooled_qsa_shared_dcp_noidx": pooled_qsa.pooled_qsa_shared_dcp_noidx_step,
        "qwen4_exp::pooled_qsa_shared_rows_dcp_noidx":
            pooled_qsa.pooled_qsa_shared_rows_dcp_noidx_step,
    }
    assert created[name] is want[name]


def test_draft_dcp_rows_op_refuses_no_index_state(recorded_jax_ops):
    with pytest.raises(AssertionError, match="needs the shared selection"):
        qsa_op.build_pooled_qsa_draft_dcp_op(None, 2, shared=False, with_rows=True,
                                             index_state=False)


# ----------------------------------------------------------------------------
# Engagement marker (STACK-D v3 entry rule): one INFO line per process (TP rank).

MARKER = "q38 DRAFT-1 no draft index state: on (TP rank 0)"


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args, **_):
        self.lines.append(msg % args if args else msg)

    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.mark.parametrize(
    "no_index,bind,prefix,want",
    [
        (True, True, DRAFT, [MARKER]),  # on: an mtp layer bound to the target's selection
        (False, True, DRAFT, []),  # flag 0
        (None, True, DRAFT, []),  # flag unset
        (True, False, DRAFT, []),  # flag on, selection never bound: the draft keeps its indexer
        (True, True, TARGET, []),  # flag on, a target layer
    ],
)
def test_engagement_marker_once_per_process_only_when_engaged(monkeypatch, no_index, bind,
                                                              prefix, want):
    _env(monkeypatch, shared=True, no_index=no_index)
    _record_builders(monkeypatch)
    log = _Logger()
    monkeypatch.setattr(qsa_module, "logger", log)
    monkeypatch.setattr(qsa_module, "_Q38_DRAFT1_LOGGED", False)
    layer = _layer(prefix)
    if bind:
        _bind_selection(layer)  # load time: before the pool is bound
    layer.bind_pool(torch.zeros(4, 8))
    layer.bind_pool(torch.zeros(4, 8))  # the final pool after the profiling scratch pool
    other = _layer(prefix)  # a second layer in the same process
    if bind:
        _bind_selection(other)
    other.bind_pool(torch.zeros(4, 8))
    got = [line for line in log.lines if "DRAFT-1" in line]
    print("DRAFT1_MARKER", no_index, bind, prefix, got)
    assert got == want


# ----------------------------------------------------------------------------
# Producer guard (L2 review, audit item 7): a KV producer would ship the draft's
# unwritten index state to its consumer, so the flag is refused there at load.


@pytest.mark.parametrize(
    "role,shared,no_index,refused",
    [
        ("kv_producer", True, True, True),
        ("kv_both", True, True, True),
        ("kv_producer", False, True, True),  # decode-only: refused even where it is inert
        ("kv_consumer", True, True, False),
        (None, True, True, False),  # no KV transfer at all
        ("kv_producer", True, False, False),  # flag 0
        ("kv_producer", True, None, False),  # flag unset
    ],
)
def test_producer_refuses_the_flag_at_load(monkeypatch, role, shared, no_index, refused):
    from test_qwen4_exp_shared_selection_bind import TOKENS, _models

    from vllm_torchtpu.models.vllm.qwen4_exp import mtp as mtp_module

    _env(monkeypatch, shared=shared, no_index=no_index)
    target, draft = _models()
    config = NS(kv_transfer_config=None if role is None else NS(kv_role=role))
    draft.kv_producer = mtp_module.draft_kv_producer(config)
    assert draft.kv_producer is (role in ("kv_producer", "kv_both"))
    if refused:
        with pytest.raises(ValueError, match="on a KV producer"):
            draft.bind_shared_selection(target, TOKENS)
        assert not hasattr(target.model.layers[2].self_attn, "_selection_sink")
    else:
        assert draft.bind_shared_selection(target, TOKENS) is shared
