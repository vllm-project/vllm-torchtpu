# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 DiT step-time TPU patches.

The vLLM-Omni modules under test are stand-ins exposing the attributes the
patches replace, registered in ``sys.modules`` under the upstream names. The
AdaLN cache check also loads vLLM-Omni's model-independent
exact_projection_cache.py from sys.path by file location (skipped if absent).
TPU dispatch is exercised by forcing ``minimax_h3._is_tpu``; the math stays on
CPU.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.omni import minimax_h3

pytestmark = pytest.mark.omni

_TPU_DEVICE = SimpleNamespace(type="tpu")


def _install(monkeypatch, module: types.ModuleType) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _force_tpu(monkeypatch) -> None:
    """Dispatch every tensor as a torch_tpu tensor (the math stays on CPU)."""
    monkeypatch.setattr(minimax_h3, "_is_tpu", lambda value: True)


def _fake_pipeline(monkeypatch) -> types.ModuleType:
    """Minimal pipeline_minimax_h3 stand-in so apply_minimax_h3_patches runs."""
    pipeline = types.ModuleType(minimax_h3.PIPELINE_MODULE)
    pipeline.dist = torch.distributed

    def _broadcast_tensor(tensor, *, dtype, device):
        raise AssertionError("unpatched _broadcast_tensor")

    class MiniMaxH3Pipeline:
        def _encoder_group_broadcast_tensor(self, tensor, *, dtype, device):
            raise AssertionError("unpatched")

        def _distribute_encode_inputs(self, ids, vision_kwargs):
            raise AssertionError("unpatched")

    pipeline._broadcast_tensor = _broadcast_tensor
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    monkeypatch.setitem(sys.modules, minimax_h3.INDEXED_MODULATION_MODULE, None)
    return _install(monkeypatch, pipeline)


def test_is_tpu():
    assert minimax_h3._is_tpu(SimpleNamespace(device=_TPU_DEVICE))
    assert minimax_h3._is_tpu(_TPU_DEVICE)
    assert minimax_h3._is_tpu("tpu") and minimax_h3._is_tpu("tpu:3")
    assert not minimax_h3._is_tpu(torch.zeros(1))
    assert not minimax_h3._is_tpu(torch.device("cpu"))
    assert not minimax_h3._is_tpu(torch.device("meta"))
    assert not minimax_h3._is_tpu("cpu")
    assert not minimax_h3._is_tpu(None)


# -----------------------------------------------------------------------------
# (a) AdaLN projection cache
# -----------------------------------------------------------------------------


def _fake_adaln_module(monkeypatch, base: type = object):
    """adaln_cache stand-in; ``prepare`` mirrors adaln_cache.py:70-82."""
    module = types.ModuleType(minimax_h3.ADALN_CACHE_MODULE)
    calls = []

    class MiniMaxH3RuntimeAdalnCache(base):
        def prepare(self, embedding):
            """Upstream prepare."""
            calls.append(embedding)
            self._sidecar_plan = None
            if base is object:
                self._input = embedding
                self._key = "digest"
            else:
                super().prepare(embedding)
            self._sidecar_plan = "plan"

    module.MiniMaxH3RuntimeAdalnCache = MiniMaxH3RuntimeAdalnCache
    return _install(monkeypatch, module), calls


def test_adaln_cache_prepare_dispatch(monkeypatch):
    module, calls = _fake_adaln_module(monkeypatch)
    cls = module.MiniMaxH3RuntimeAdalnCache
    original = cls.prepare
    assert minimax_h3.patch_adaln_projection_cache()
    assert cls._tpu_orig_prepare is original
    assert cls.prepare.__wrapped__.__wrapped__ is original
    assert cls.prepare.__doc__ == "Upstream prepare."

    # CPU (and CUDA) tensors arm the cache exactly as upstream does.
    cache = cls()
    embedding = torch.randn(2, 4)
    cache.prepare(embedding)
    assert calls == [embedding]
    assert cache._key == "digest" and cache._input is embedding
    assert cache._sidecar_plan == "plan"

    # TPU tensors leave it unarmed without running the host digest.
    _force_tpu(monkeypatch)
    cache.prepare(torch.randn(2, 4))
    assert calls == [embedding]
    assert cache._key is None and cache._input is None
    assert cache._sidecar_plan is None


def test_adaln_cache_patch_idempotent_and_noop(monkeypatch):
    module, _ = _fake_adaln_module(monkeypatch)
    assert minimax_h3.patch_adaln_projection_cache()
    patched = module.MiniMaxH3RuntimeAdalnCache.prepare
    assert not minimax_h3.patch_adaln_projection_cache()
    assert module.MiniMaxH3RuntimeAdalnCache.prepare is patched

    monkeypatch.setitem(sys.modules, minimax_h3.ADALN_CACHE_MODULE, None)
    assert minimax_h3.patch_adaln_projection_cache() is False
    monkeypatch.delitem(sys.modules, minimax_h3.ADALN_CACHE_MODULE)
    assert minimax_h3.patch_adaln_projection_cache() is False
    assert minimax_h3.ADALN_CACHE_MODULE not in sys.modules


@pytest.fixture(scope="module")
def upstream_exact_projection_cache():
    """vLLM-Omni's exact_projection_cache.py, loaded by path under a private name.

    tests/omni/mock_omni.py may replace ``vllm_omni`` in ``sys.modules``; the
    file only imports torch (and vllm.distributed inside ``project``).
    """
    relative = os.path.join(
        "vllm_omni", "diffusion", "cache", "exact_projection_cache.py"
    )
    for entry in sys.path:
        path = os.path.join(entry or os.getcwd(), relative)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "_upstream_exact_projection_cache", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip(f"{relative} not found on sys.path")


def test_adaln_cache_tpu_projects_without_tp_vote(
    monkeypatch, upstream_exact_projection_cache
):
    distributed = pytest.importorskip("vllm.distributed")

    def no_vote():
        raise AssertionError("TP hit vote")

    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(distributed, "get_tp_group", no_vote)
    module, _ = _fake_adaln_module(
        monkeypatch, upstream_exact_projection_cache.ExactProjectionCache
    )
    cls = module.MiniMaxH3RuntimeAdalnCache
    linear = torch.nn.Linear(4, 6)
    embedding = torch.randn(3, 4)
    computed = []

    def compute():
        computed.append(1)
        return linear(embedding)

    with torch.inference_mode():
        # Upstream: an armed cache votes over the TP group before projecting.
        cache = cls(max_bytes=1 << 20)
        cache.prepare(embedding)
        assert cache._key is not None
        with pytest.raises(AssertionError, match="TP hit vote"):
            cache.project("blocks.0.adaln_proj.linear", linear, embedding, compute)

        assert minimax_h3.patch_adaln_projection_cache()
        _force_tpu(monkeypatch)
        cache = cls(max_bytes=1 << 20)
        cache.prepare(embedding)
        expected = linear(embedding)
        for _ in range(2):
            out = cache.project(
                "blocks.0.adaln_proj.linear", linear, embedding, compute
            )
            assert torch.equal(out, expected)
    assert computed == [1, 1]
    assert cache.hits == 0 and cache.misses == 0


def test_apply_installs_adaln_cache_patch(monkeypatch):
    _fake_pipeline(monkeypatch)
    module, _ = _fake_adaln_module(monkeypatch)
    assert minimax_h3.apply_minimax_h3_patches()
    assert "_tpu_orig_prepare" in module.MiniMaxH3RuntimeAdalnCache.__dict__
    patched = module.MiniMaxH3RuntimeAdalnCache.prepare
    assert not minimax_h3.apply_minimax_h3_patches()
    assert module.MiniMaxH3RuntimeAdalnCache.prepare is patched


# -----------------------------------------------------------------------------
# (b) fixed-width timestep table
# -----------------------------------------------------------------------------

# Row groups of a packed request: text/padding (video timestep), video target,
# image condition, audio target, audio reference (denoise_loop.py:235-265).
_GROUPS = {
    "t2va": [0, 0, 1, 1, 1, 3, 3, 0],
    "fl2va": [0, 2, 2, 1, 1, 3, 3, 0],
    "ref2va-mixed": [0, 2, 1, 1, 3, 4, 3, 0],
}


def _fake_denoise_loop_module(monkeypatch) -> types.ModuleType:
    module = types.ModuleType(minimax_h3.DENOISE_LOOP_MODULE)

    class MiniMaxH3DenoiseBranch:
        def __init__(self, groups):
            self.groups = torch.tensor(groups)
            self.static_kwargs = {"token_tags": self.groups % 3}

        def forward_kwargs(
            self,
            *,
            video_rows=None,
            audio_rows=None,
            t_video,
            t_audio,
            imgvid_cond_timestep,
            audio_ref_cond_timestep,
            video_target_timesteps=None,
            audio_target_timesteps=None,
        ):
            """Upstream forward_kwargs (denoise_loop.py:182-215)."""
            values = torch.tensor(
                [
                    t_video,
                    t_video,
                    imgvid_cond_timestep,
                    t_audio,
                    audio_ref_cond_timestep,
                ],
                dtype=torch.float32,
            )
            timesteps = values[self.groups]
            if video_target_timesteps is not None:
                timesteps = timesteps.clone()
                timesteps[self.groups == 1] = video_target_timesteps
            unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
            return {
                **self.static_kwargs,
                "timesteps": timesteps,
                "unique_timesteps": unique,
                "inverse_indices": inverse,
            }

    module.MiniMaxH3DenoiseBranch = MiniMaxH3DenoiseBranch
    return _install(monkeypatch, module)


def _step_kwargs(sigma_video, sigma_audio):
    t_v, t_a = 1.0 - sigma_video, 1.0 - sigma_audio
    return {
        "t_video": t_v,
        "t_audio": t_a,
        "imgvid_cond_timestep": max(t_v, 0.999),
        "audio_ref_cond_timestep": max(t_a, 1.0),
    }


def test_pad_unique_timesteps():
    pad = minimax_h3.pad_unique_timesteps
    one = torch.tensor([0.0])
    assert pad(one, 4).tolist() == [0.0, 0.0, 0.0, 0.0]
    two = torch.tensor([0.25, 0.5], dtype=torch.float32)
    padded = pad(two, 4)
    assert padded.dtype == torch.float32
    assert padded.tolist() == [0.25, 0.5, 0.5, 0.5]
    for unchanged in (
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.arange(6, dtype=torch.float32),
        torch.empty(0),
        torch.zeros(2, 2),
    ):
        assert pad(unchanged, 4) is unchanged


def test_timestep_table_padded_on_tpu(monkeypatch):
    module = _fake_denoise_loop_module(monkeypatch)
    cls = module.MiniMaxH3DenoiseBranch
    original = cls.forward_kwargs
    assert minimax_h3.patch_denoise_branch_timestep_rows()
    assert cls._tpu_orig_forward_kwargs is original
    assert cls.forward_kwargs.__wrapped__ is original
    rows = minimax_h3.H3_UNIQUE_TIMESTEP_ROWS
    schedule = [(1.0, 1.0), (0.9, 0.7), (0.5, 0.25), (0.05, 0.01)]

    # CPU: the upstream table, whose width changes with the step.
    branch = cls(_GROUPS["t2va"])
    widths = {
        branch.forward_kwargs(**_step_kwargs(*s))["unique_timesteps"].shape[0]
        for s in schedule
    }
    assert widths == {1, 2}

    _force_tpu(monkeypatch)
    for task, groups in _GROUPS.items():
        branch = cls(groups)
        for sigmas in schedule:
            kwargs = _step_kwargs(*sigmas)
            expected = original(branch, **kwargs)
            out = branch.forward_kwargs(**kwargs)
            unique = out["unique_timesteps"]
            exact = expected["unique_timesteps"]
            count = exact.shape[0]
            assert unique.shape == (rows,), (task, sigmas)
            assert torch.equal(unique[:count], exact)
            assert torch.equal(unique[count:], exact[-1:].expand(rows - count))
            assert torch.equal(out["inverse_indices"], expected["inverse_indices"])
            assert int(out["inverse_indices"].max()) < count
            # Every row still reads its own timestep.
            assert torch.equal(unique[out["inverse_indices"]], out["timesteps"])
            assert out["token_tags"] is branch.static_kwargs["token_tags"]

    # Latent-edit schedules wider than the pad keep the upstream table.
    branch = cls(_GROUPS["t2va"])
    kwargs = _step_kwargs(0.5, 0.25)
    edit = torch.tensor([0.1, 0.2, 0.3])
    out = branch.forward_kwargs(**kwargs, video_target_timesteps=edit)
    assert out["unique_timesteps"].shape[0] == 5


def test_padded_timestep_table_feeds_the_same_rows(monkeypatch):
    """The DiT's consumers (time embedder, AdaLN, gathers) are row-independent."""
    torch.manual_seed(0)
    module = _fake_denoise_loop_module(monkeypatch)
    cls = module.MiniMaxH3DenoiseBranch
    assert minimax_h3.patch_denoise_branch_timestep_rows()
    _force_tpu(monkeypatch)
    proj_in = torch.nn.Linear(16, 32)
    adaln = torch.nn.Linear(32, 3 * 2 * 8)
    modality_num = 3

    def consume(kwargs):
        # minimax_h3_transformer.py:368-381, :898-906, :1815.
        t = kwargs["unique_timesteps"]
        freqs = torch.exp(-torch.arange(8, dtype=torch.float32) / 8)
        args = t[:, None] * freqs[None]
        t_emb = proj_in(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))
        x = adaln(torch.nn.functional.silu(t_emb))
        x = x.view(x.shape[0] * modality_num, -1)
        combined = kwargs["inverse_indices"] * modality_num + kwargs["token_tags"]
        return x.index_select(0, combined)

    branch = cls(_GROUPS["fl2va"])
    for sigmas in [(1.0, 1.0), (0.6, 0.3)]:
        kwargs = _step_kwargs(*sigmas)
        exact = cls._tpu_orig_forward_kwargs(branch, **kwargs)
        padded = branch.forward_kwargs(**kwargs)
        assert padded["unique_timesteps"].shape[0] == 4
        torch.testing.assert_close(consume(padded), consume(exact))


def test_timestep_table_patch_idempotent_and_noop(monkeypatch):
    module = _fake_denoise_loop_module(monkeypatch)
    assert minimax_h3.patch_denoise_branch_timestep_rows()
    patched = module.MiniMaxH3DenoiseBranch.forward_kwargs
    assert not minimax_h3.patch_denoise_branch_timestep_rows()
    assert module.MiniMaxH3DenoiseBranch.forward_kwargs is patched

    monkeypatch.setitem(sys.modules, minimax_h3.DENOISE_LOOP_MODULE, None)
    assert minimax_h3.patch_denoise_branch_timestep_rows() is False


def test_apply_installs_timestep_table_patch(monkeypatch):
    _fake_pipeline(monkeypatch)
    module = _fake_denoise_loop_module(monkeypatch)
    assert minimax_h3.apply_minimax_h3_patches()
    cls = module.MiniMaxH3DenoiseBranch
    assert "_tpu_orig_forward_kwargs" in cls.__dict__
    patched = cls.forward_kwargs
    assert not minimax_h3.apply_minimax_h3_patches()
    assert cls.forward_kwargs is patched


# -----------------------------------------------------------------------------
# (c) per-request RoPE table
# -----------------------------------------------------------------------------


class _FakeRopeModel:
    def __init__(self):
        self.calls = []

    def prepare_rope_table(self, img_position_ids, *, seq_len):
        self.calls.append((img_position_ids, seq_len))
        return torch.full((seq_len, 6), float(len(self.calls)))


class _FakeBranch:
    seq_len = 5

    def __init__(self):
        self.static_kwargs = {"img_position_ids": torch.zeros(1, 5, 3)}

    def prepare_rope_table(self, model):
        """denoise_loop.py:166-180."""
        prepare = getattr(model, "prepare_rope_table", None)
        if not callable(prepare):
            return
        self.static_kwargs["rope_table"] = prepare(
            self.static_kwargs["img_position_ids"], seq_len=self.seq_len
        )


def _fake_loop_modules(monkeypatch):
    module = types.ModuleType(minimax_h3.DENOISE_LOOP_MODULE)
    seen = []

    def minimax_h3_denoise_loop(*, model, positive, device, sigmas_video=(1.0, 0.0)):
        """Upstream loop: the model sees static_kwargs at every step."""
        for _ in range(len(sigmas_video) - 1):
            seen.append(positive.static_kwargs.get("rope_table"))
        return "rows"

    module.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    pipeline = types.ModuleType(minimax_h3.PIPELINE_MODULE)
    pipeline.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    unrelated = types.ModuleType("vllm_omni.fake_unrelated")
    unrelated.minimax_h3_denoise_loop = "unrelated"
    for loaded in (module, pipeline, unrelated):
        _install(monkeypatch, loaded)
    return module, pipeline, unrelated, seen


def test_rope_table_prepared_once_per_run_on_tpu(monkeypatch):
    module, pipeline, unrelated, seen = _fake_loop_modules(monkeypatch)
    original = module.minimax_h3_denoise_loop
    assert minimax_h3.patch_denoise_loop_rope_table()
    wrapper = module.minimax_h3_denoise_loop
    assert wrapper is not original and wrapper.__wrapped__ is original
    assert module._tpu_orig_minimax_h3_denoise_loop is original
    assert pipeline.minimax_h3_denoise_loop is wrapper
    assert unrelated.minimax_h3_denoise_loop == "unrelated"

    # TPU: one table per denoise run, reused by every step.
    model, branch = _FakeRopeModel(), _FakeBranch()
    sigmas = (1.0, 0.75, 0.5, 0.0)
    out = pipeline.minimax_h3_denoise_loop(
        model=model, positive=branch, device=_TPU_DEVICE, sigmas_video=sigmas
    )
    assert out == "rows"
    assert len(model.calls) == 1
    assert model.calls[0][1] == branch.seq_len
    table = branch.static_kwargs["rope_table"]
    assert seen == [table, table, table]
    # The next run (e.g. a refine pass) rebuilds it, like the NPU path.
    pipeline.minimax_h3_denoise_loop(
        model=model, positive=branch, device="tpu:0", sigmas_video=sigmas
    )
    assert len(model.calls) == 2

    # Other devices stay on the per-forward reference path.
    seen.clear()
    model, branch = _FakeRopeModel(), _FakeBranch()
    pipeline.minimax_h3_denoise_loop(
        model=model, positive=branch, device=torch.device("cpu"), sigmas_video=sigmas
    )
    assert model.calls == [] and "rope_table" not in branch.static_kwargs
    assert seen == [None, None, None]


def test_rope_table_patch_idempotent_and_noop(monkeypatch):
    module, pipeline, _, _ = _fake_loop_modules(monkeypatch)
    assert minimax_h3.patch_denoise_loop_rope_table()
    wrapper = module.minimax_h3_denoise_loop
    assert not minimax_h3.patch_denoise_loop_rope_table()
    assert module.minimax_h3_denoise_loop is wrapper
    assert pipeline.minimax_h3_denoise_loop is wrapper

    # A module that bound the original later is rebound, without re-wrapping.
    late = types.ModuleType("vllm_omni.fake_late_loop_importer")
    late.minimax_h3_denoise_loop = module._tpu_orig_minimax_h3_denoise_loop
    _install(monkeypatch, late)
    assert minimax_h3.patch_denoise_loop_rope_table()
    assert late.minimax_h3_denoise_loop is wrapper

    monkeypatch.setitem(sys.modules, minimax_h3.DENOISE_LOOP_MODULE, None)
    assert minimax_h3.patch_denoise_loop_rope_table() is False
    not_a_function = types.ModuleType(minimax_h3.DENOISE_LOOP_MODULE)
    not_a_function.minimax_h3_denoise_loop = object()
    _install(monkeypatch, not_a_function)
    assert minimax_h3.patch_denoise_loop_rope_table() is False


def test_apply_installs_rope_table_patch(monkeypatch):
    pipeline = _fake_pipeline(monkeypatch)
    module = types.ModuleType(minimax_h3.DENOISE_LOOP_MODULE)

    def minimax_h3_denoise_loop(*, model, positive, device):
        return "rows"

    module.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    pipeline.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    _install(monkeypatch, module)
    assert minimax_h3.apply_minimax_h3_patches()
    wrapper = module.minimax_h3_denoise_loop
    assert wrapper.__wrapped__ is minimax_h3_denoise_loop
    assert pipeline.minimax_h3_denoise_loop is wrapper
    assert not minimax_h3.apply_minimax_h3_patches()
    assert module.minimax_h3_denoise_loop is wrapper
