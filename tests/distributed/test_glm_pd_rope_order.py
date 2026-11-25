# SPDX-License-Identifier: Apache-2.0
"""P/D: the cached rope channel order must enter the GLM layout fingerprint.

An engine with TPU_MLA_ROPE_NEOX_PERMUTE (and TPU_INDEXER_ROPE_NEOX_PERMUTE for
the indexer key; both on by default) permutes its q and k_pe rope projection
columns into half-split order after load, so its MLA cache holds permuted
k_pe. A peer without the permute caches k_pe in the checkpoint's interleaved
order. The pages have the same geometry, so both legs used to fingerprint
alike, and the decode read the producer's rope bytes in the wrong channel
order: every loaded token's rope score was wrong while its latent (nope) bytes
were right. No TPU runtime or transport is involved here.
"""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.model_executor.custom_op as custom_op_mod
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

from .raiden_test_utils import glm_named_kv_caches
from .test_raiden_layout_fingerprint import _glm_layout_getter, _glm_versions
from .test_tpu_connector import _FakeAdmissionRaidenEngine, _make_raiden_worker

_MOD = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
PAGE = 1024
SAL_PAGE = 256
ROPE = 64
GLM_ROPE_THETA = 8_000_000.0  # GLM-5.2 config rope_parameters (rope_type default)
# test_raiden_layout_fingerprint.py's GLM golden (same layout getter and
# versions, imported from there): the interleaved order declares nothing, so
# it must not move.
GOLDEN_GLM_FINGERPRINT = (
    "bfea2b664510bfb11f87b5092431492fddaf535f0f321f01d59a79e139c3bc2b"
)
HALF = rlf.GLM_ROPE_ORDER_HALF_SPLIT


def _manifest(path, layers=(0, 1)):
    """The manifest each admission path builds: class tags (TP), per-layer
    tags for a pipeline stage holding ``layers``, or the DCP identity's
    SEQ_ALONG_LANE indexer page."""
    if path == "dcp_identity":
        named = glm_named_kv_caches(
            block_size_tokens=SAL_PAGE, idx_shape=(16, 33, 4, SAL_PAGE)
        )
        return rpm.build_glm_mla_pool_manifest(
            named_kv_caches=named,
            raw_tensors=(),
            block_size_tokens=SAL_PAGE,
            indexer_seq_along_lane=True,
        )
    named = {
        name: cache
        for name, cache in glm_named_kv_caches().items()
        if rpm.layer_index_from_name(name) in layers
    }
    return rpm.build_glm_mla_pool_manifest(
        named_kv_caches=named,
        raw_tensors=(),
        block_size_tokens=PAGE,
        per_layer_tags=path == "pp",
    )


def _fingerprint(path, layers=(0, 1), **rope_order):
    # Each path's arguments as _measure_raiden_glm_layout passes them.
    return rlf.measured_glm_layout_fingerprint(
        _manifest(path, layers),
        page_tokens=SAL_PAGE if path == "dcp_identity" else PAGE,
        layout_getter=_glm_layout_getter,
        package_version=_glm_versions,
        hash_indexer=path != "pp",
        indexer_seq_along_lane=path == "dcp_identity",
        **rope_order,
    )


def test_interleaved_rope_keeps_the_golden():
    # A peer caching the interleaved order declares nothing, so a peer from a
    # tree without the permute keeps matching it.
    for path in ("tp", "pp", "dcp_identity"):
        before = _fingerprint(path)
        after = _fingerprint(
            path, mla_rope_half_split=False, indexer_rope_half_split=False
        )
        assert after == before
        assert not {"mla_rope_order", "indexer_rope_order"} & set(after[1])
    assert _fingerprint("tp")[0] == GOLDEN_GLM_FINGERPRINT


@pytest.mark.parametrize("path", ["tp", "pp", "dcp_identity"])
@pytest.mark.parametrize(
    ("mla_half", "indexer_half"), [(True, True), (True, False), (False, True)]
)
def test_half_split_decode_never_pairs_with_an_interleaved_producer(
    path, mla_half, indexer_half
):
    # The measured failure: both legs logged one fingerprint, the decode read
    # the producer's rope bytes as half-split k_pe, and its continuation was
    # wrong. Holds for a pipeline stage (only its own layers) and for the DCP
    # identity transfer alike.
    producer, _ = _fingerprint(path, layers=(1,) if path == "pp" else (0, 1))
    decode, payload = _fingerprint(
        path, mla_rope_half_split=mla_half, indexer_rope_half_split=indexer_half
    )
    assert decode != producer
    assert payload.get("mla_rope_order") == (HALF if mla_half else None)
    assert payload.get("indexer_rope_order") == (HALF if indexer_half else None)


@pytest.mark.parametrize("path", ["tp", "pp", "dcp_identity"])
def test_half_split_legs_still_pair(path):
    # Both legs permuting (the default) keep pairing with each other,
    # including a pipeline stage that holds no indexer layer.
    order = {"mla_rope_half_split": True, "indexer_rope_half_split": True}
    producer = _fingerprint(path, layers=(1,) if path == "pp" else (0, 1), **order)
    assert producer == _fingerprint(path, **order)


class _OwnForwardRope(RotaryEmbedding):
    """A rope with its own forward_native, as a platform rope adapter has: the
    permute cannot reorder it, so a flag left at its default skips it."""

    def forward_native(self, *args, **kwargs):
        return super().forward_native(*args, **kwargs)


def _glm_rope(cls=RotaryEmbedding):
    """GLM-5.2's MLA rope as vLLM builds it: GPT-J style."""
    cfg = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["none"])
    with patch.object(custom_op_mod, "get_cached_compilation_config", lambda: cfg):
        return cls(
            head_size=ROPE,
            rotary_dim=ROPE,
            max_position_embeddings=4096,
            base=GLM_ROPE_THETA,
            is_neox_style=False,
            dtype=torch.float32,
        )


class _FakeIndexer(torch.nn.Module):
    pass


class _FakeMLA(torch.nn.Module):
    pass


@pytest.fixture(autouse=True)
def _fake_layer_classes(monkeypatch):
    """The reader recognises layers by class; stand in for the indexer and the
    MLA wrapper with bare modules carrying the attributes it reads."""
    monkeypatch.setattr(mla_op, "VllmTPUIndexer", _FakeIndexer)
    monkeypatch.setattr(mla_op, "VllmTPUMultiHeadLatentAttentionWrapper", _FakeMLA)


def _model(
    *, permuted_mla=False, permuted_indexer=False, indexer=True, rope=RotaryEmbedding
):
    """One layer as the reader walks it: an MLA layer that holds
    ``_rope_neox_permuted`` from construction on, its rope, and optionally an
    indexer, which holds it from rebind on."""
    model = torch.nn.Module()
    model.attn = _FakeMLA()
    model.attn._rope_neox_permuted = permuted_mla
    model.attn.qk_rope_head_dim = ROPE
    model.attn.rotary_emb = _glm_rope(rope)
    if indexer:
        model.attn.indexer = _FakeIndexer()
        model.attn.indexer._rope_neox_permuted = permuted_indexer
    return model


def _flags(mla, indexer):
    # The env module the reader (mla_attention_op) reads the flags from.
    return (
        patch.object(mla_op.envs, "TPU_MLA_ROPE_NEOX_PERMUTE", mla),
        patch.object(mla_op.envs, "TPU_INDEXER_ROPE_NEOX_PERMUTE", indexer),
    )


def test_rope_order_is_read_from_the_permuted_layers():
    # The connector reads the order the layers actually cache (the flag a
    # layer sets once its post-load permute ran), not the env's intent.
    read = mla_op.glm_rope_half_split
    mla_flag, indexer_flag = _flags(True, True)
    with mla_flag, indexer_flag:
        assert read(_model()) == (False, False)
        assert read(_model(permuted_mla=True)) == (True, False)
        assert read(_model(permuted_mla=True, permuted_indexer=True)) == (True, True)
        # No layers to read while a flag is on: refuse to guess.
        with pytest.raises(RuntimeError, match="cached rope order"):
            read(None)
    mla_flag, indexer_flag = _flags(False, False)
    with mla_flag, indexer_flag:
        assert read(None) == (False, False)


@pytest.mark.parametrize(
    ("mla_flag", "indexer_flag"), [(True, True), (True, False), (False, True)]
)
def test_model_without_mla_layers_refuses_to_guess(mla_flag, indexer_flag):
    # Every admitted GLM stage holds an MLA layer. A model whose modules()
    # yields none (a wrapper hiding the layers, a mock) was not read, so it
    # must not declare the interleaved order while a permute flag is on.
    hidden = torch.nn.Module()
    hidden.layer = torch.nn.Linear(2, 2)
    only_indexer = torch.nn.Module()
    only_indexer.indexer = _FakeIndexer()
    only_indexer.indexer._rope_neox_permuted = True
    for model in (hidden, only_indexer, MagicMock()):
        mla_patch, indexer_patch = _flags(mla_flag, indexer_flag)
        with (
            mla_patch,
            indexer_patch,
            pytest.raises(RuntimeError, match="holds no"),
        ):
            mla_op.glm_rope_half_split(model, SimpleNamespace())
    # Both flags off: nothing was permuted, so there is nothing to read.
    mla_patch, indexer_patch = _flags(False, False)
    with mla_patch, indexer_patch:
        assert mla_op.glm_rope_half_split(hidden) == (False, False)


# A pipeline stage with no indexer layer ships no indexer key; it declares the
# order an indexer layer of its own would cache its keys in. vLLM builds that
# rope GPT-J style only under indexer_rope_interleave (GLM-5.2 sets it), NeoX
# otherwise. The permute reorders only a GPT-J rope into half-split; a NeoX
# rope is cached in the checkpoint's own order on every engine, so it is never
# declared (both legs cache the same bytes).
@pytest.mark.parametrize(
    ("config", "rope", "indexer_flag", "expected"),
    [
        pytest.param(
            {"indexer_rope_interleave": True},
            RotaryEmbedding,
            True,
            True,
            id="gptj_indexer_rope_permuted_declares_half_split",
        ),
        pytest.param(
            {"indexer_rope_interleave": False},
            RotaryEmbedding,
            True,
            False,
            id="neox_indexer_rope_unpermuted_declares_nothing",
        ),
        pytest.param(
            {},
            RotaryEmbedding,
            True,
            False,
            id="config_key_absent_neox_unpermuted_declares_nothing",
        ),
        pytest.param(
            {"indexer_rope_interleave": True},
            _OwnForwardRope,
            True,
            False,
            id="rope_class_permute_skips_unpermuted_declares_nothing",
        ),
        pytest.param(
            {"indexer_rope_interleave": True},
            RotaryEmbedding,
            False,
            False,
            id="indexer_flag_off_unpermuted_declares_nothing",
        ),
    ],
)
def test_no_indexer_stage_declares_the_order_its_indexer_would_cache(
    config, rope, indexer_flag, expected
):
    permuted_mla = rope is RotaryEmbedding
    stage = _model(permuted_mla=permuted_mla, indexer=False, rope=rope)
    mla_flag, idx_flag = _flags(True, indexer_flag)
    with mla_flag, idx_flag:
        assert mla_op.glm_rope_half_split(stage, SimpleNamespace(**config)) == (
            permuted_mla,
            expected,
        )


def test_no_indexer_stage_without_model_config_refuses_to_guess():
    stage = _model(permuted_mla=True, indexer=False)
    mla_flag, idx_flag = _flags(True, True)
    with (
        mla_flag,
        idx_flag,
        pytest.raises(RuntimeError, match="indexer_rope_interleave"),
    ):
        mla_op.glm_rope_half_split(stage)
    # With the indexer flag off there is nothing to read.
    mla_flag, idx_flag = _flags(True, False)
    with mla_flag, idx_flag:
        assert mla_op.glm_rope_half_split(stage) == (True, False)


def test_flag_on_but_permute_skipped_declares_interleaved():
    # With the flags at their default, a rope the permute cannot reorder keeps
    # the interleaved order: the layers never set the flag, so the engine
    # declares nothing, with or without an indexer layer.
    assert not mla_op.rope_half_split_supported(_glm_rope(_OwnForwardRope), ROPE)
    assert mla_op.rope_half_split_supported(_glm_rope(), ROPE)
    mla_flag, indexer_flag = _flags(True, True)
    with mla_flag, indexer_flag:
        glm = SimpleNamespace(indexer_rope_interleave=True)
        for indexer in (True, False):
            model = _model(indexer=indexer, rope=_OwnForwardRope)
            assert mla_op.glm_rope_half_split(model, glm) == (False, False)


class _ComparingController:
    """The reshard service as the workers see it: it keeps each unit's
    registered layout fingerprint and refuses a transfer whose source and
    destination units registered different ones (the native service's
    "Layout fingerprint mismatch between source and destination")."""

    def __init__(self):
        self.fingerprints = {}
        self.start_transfer_calls = 0

    @staticmethod
    def _key(unit):
        return tuple(sorted(vars(unit).items()))

    def register_work_unit(self, *, unit, layout_fingerprint, **_):
        self.fingerprints[self._key(unit)] = layout_fingerprint

    def start_transfer(self, *, src_units, dst_units, **_):
        self.start_transfer_calls += 1
        seen = {self.fingerprints[self._key(u)] for u in [*src_units, *dst_units]}
        if len(seen) != 1:
            raise RuntimeError(
                "Layout fingerprint mismatch between source and destination"
            )
        return True


def _register(controller, *, is_producer, model, path):
    """Runs one worker's real admission measure and work-unit registration
    against ``controller``; returns the registered unit."""
    worker = _make_raiden_worker(
        is_producer=is_producer, tp_rank=0, tp_size=1, block_size=PAGE
    )
    worker.runner = SimpleNamespace(model=model)
    worker._new_raiden_controller_facade = MagicMock(return_value=controller)
    worker._new_raiden_id = MagicMock(
        side_effect=lambda fields: SimpleNamespace(**fields)
    )
    worker._local_raiden_transfer_rank = MagicMock(return_value=0)
    worker._validate_stage3_transfer_parallelism = MagicMock()
    with (
        patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER", True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_GLM_ADMISSION", True),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM", 1),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_RESHARD_IMPL", "controller"),
        patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER", False),
        # Both permute flags on (the default) on every leg: what a leg caches
        # is read from its layers.
        patch(f"{_MOD}.tpu_envs.TPU_MLA_ROPE_NEOX_PERMUTE", True),
        patch(f"{_MOD}.tpu_envs.TPU_INDEXER_ROPE_NEOX_PERMUTE", True),
        patch(f"{_MOD}._raiden_glm_dcp_identity", return_value=path == "identity"),
        patch(f"{_MOD}._glm_indexer_seq_along_lane", return_value=False),
    ):
        registration = worker._register_raiden_stage3_work_unit(
            engine=_FakeAdmissionRaidenEngine(),
            manifest=_manifest("tp"),
            controller_address="controller.test:27000",
        )
    unit = worker._raiden_work_unit
    fingerprint = registration["layout_fingerprint"]
    assert controller.fingerprints[controller._key(unit)] == fingerprint
    return unit, registration["layout_fingerprint_payload"]


# How a leg's layers ended up: permuted, never permuted, or flag on with a
# rope the permute skipped.
_ENGINES = {
    "interleaved": {},
    "half_split": {"permuted_mla": True, "permuted_indexer": True},
    "skipped": {"rope": _OwnForwardRope},
}


@pytest.mark.parametrize("path", ["tp", "identity"])
@pytest.mark.parametrize(
    ("producer", "decode", "refused"),
    [
        ("interleaved", "half_split", True),
        ("half_split", "interleaved", True),
        ("half_split", "half_split", False),
        ("interleaved", "interleaved", False),
        ("skipped", "interleaved", False),
        ("interleaved", "skipped", False),
        ("skipped", "half_split", True),
        ("half_split", "skipped", True),
    ],
)
def test_mixed_rope_pairing_is_refused_at_transfer_start(
    monkeypatch, path, producer, decode, refused
):
    # End to end on the CPU side: each leg measures its own layers, registers
    # its fingerprint, and the decode's transfer start goes through the same
    # retry wrapper as a real load. A mixed pairing is refused and not
    # retried; a matched pairing (either order) is accepted.
    monkeypatch.setattr(
        rlf,
        "measured_glm_layout_fingerprint",
        functools.partial(
            rlf.measured_glm_layout_fingerprint,
            layout_getter=_glm_layout_getter,
            package_version=_glm_versions,
        ),
    )
    controller = _ComparingController()
    src, src_payload = _register(
        controller, is_producer=True, model=_model(**_ENGINES[producer]), path=path
    )
    dst, dst_payload = _register(
        controller, is_producer=False, model=_model(**_ENGINES[decode]), path=path
    )
    for kind, payload in ((producer, src_payload), (decode, dst_payload)):
        declared = {payload.get("mla_rope_order"), payload.get("indexer_rope_order")}
        assert declared == ({HALF} if kind == "half_split" else {None})
    start = functools.partial(
        tc.TPURaidenConnectorWorker._start_stage3_transfer_with_d5_retry,
        controller,
        src_units=[src],
        dst_units=[dst],
        req_id="probe",
        uuid=1,
    )
    with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30.0):
        if refused:
            with pytest.raises(RuntimeError, match="Layout fingerprint mismatch"):
                start()
        else:
            assert start() is True
    assert controller.start_transfer_calls == 1


def test_permuted_decode_misreads_interleaved_rope_bytes():
    # Why the declaration is needed. q at the decode's first computed position
    # (4), keys at 0..3 (the loaded tokens).
    rope = _glm_rope()
    assert mla_op.rope_half_split_supported(rope, ROPE)  # the decode permutes GLM
    perm = mla_op.mla_rope_half_split_perm(ROPE)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(4, 1, ROPE, generator=gen)
    k = torch.randn(4, 1, ROPE, generator=gen)
    q_pos = torch.full((4,), 4)
    k_pos = torch.arange(4)

    def gptj(pos, x):
        return rope.forward_native(pos, x.clone(), x.clone())[1]

    def neox(pos, x):
        return mla_op.rope_half_split(rope, pos, x.clone(), x.clone())[1]

    reference = (gptj(q_pos, q) * gptj(k_pos, k)).sum(-1)
    producer_k = gptj(k_pos, k)  # interleaved, as the producer caches it
    decode_q = neox(q_pos, q[..., perm])  # permuted projections, NeoX rope
    decode_native_k = neox(k_pos, k[..., perm])  # permuted k_pe, cached by the decode
    torch.testing.assert_close((decode_q * decode_native_k).sum(-1), reference)
    torch.testing.assert_close((decode_q * producer_k[..., perm]).sum(-1), reference)
    cross = (decode_q * producer_k).sum(-1)
    assert (cross - reference).abs().max() > 0.5 * reference.abs().max()
