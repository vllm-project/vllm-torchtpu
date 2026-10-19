# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the sync-free MiniMax-H3 denoise loop TPU patches.

vLLM-Omni's own denoise_loop.py, latent_mask.py, sampling.py and
scheduling_minimax_h3_euler_ancestral.py are imported from sys.path under
their real module names (skipped if absent), with small stand-ins for the
rest of vllm_omni (tests/omni/mock_omni.py may replace the package). TPU
dispatch is exercised by forcing ``minimax_h3._is_tpu``; the math stays on
CPU, where ``_flush_stream`` is a no-op.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.omni import minimax_h3

pytestmark = pytest.mark.omni

_PKG = "vllm_omni.diffusion.models.minimax_h3"
_SUBMODULES = (
    "scheduling_minimax_h3_euler_ancestral",
    "sampling",
    "latent_mask",
    "denoise_loop",
)
_CPU = torch.device("cpu")
_STEPS = 8


class _Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def upstream_dir() -> str | None:
    relative = os.path.join("vllm_omni", "diffusion", "models", "minimax_h3")
    for entry in sys.path:
        path = os.path.join(entry or os.getcwd(), relative)
        if os.path.isfile(os.path.join(path, "denoise_loop.py")):
            return path
    return None


def load_upstream_minimax_h3(directory: str, setitem) -> SimpleNamespace:
    """Import upstream's denoise loop files with stand-ins for their imports.

    ``setitem(name, module)`` registers a stand-in in ``sys.modules``.
    """
    progress = []
    abstract = types.ModuleType("vllm_omni.diffusion.attention.backends.abstract")
    abstract.VideoTokenLayout = _Record
    abstract.VideoTokenSpan = _Record
    cancellation = types.ModuleType("vllm_omni.diffusion.cancellation")
    cancellation.check_request_cancellation = lambda *, synchronize=False: None
    forward_context = types.ModuleType("vllm_omni.diffusion.forward_context")
    for name in ("step_idx", "timestep", "total_steps"):
        setattr(
            forward_context,
            f"set_forward_context_denoise_{name}",
            lambda value, name=name: progress.append((name, value)),
        )
    platforms = types.ModuleType("vllm_omni.platforms")
    platforms.current_omni_platform = SimpleNamespace(is_npu=lambda: False)
    package = types.ModuleType(_PKG)
    package.__path__ = [directory]
    for module in (abstract, cancellation, forward_context, platforms, package):
        setitem(module.__name__, module)
    loop = importlib.import_module(f"{_PKG}.denoise_loop")
    return SimpleNamespace(
        loop=loop,
        scheduling=sys.modules[f"{_PKG}.scheduling_minimax_h3_euler_ancestral"],
        sampling=sys.modules[f"{_PKG}.sampling"],
        progress=progress,
    )


def upstream_functions(upstream: SimpleNamespace) -> dict:
    """The upstream functions pinned in minimax_h3._UPSTREAM_SOURCE_SHA256."""
    branch = upstream.loop.MiniMaxH3DenoiseBranch
    return {
        "denoise_loop.minimax_h3_denoise_loop": (upstream.loop.minimax_h3_denoise_loop),
        "denoise_loop.MiniMaxH3DenoiseBranch.forward_kwargs": branch.forward_kwargs,
        "denoise_loop.MiniMaxH3DenoiseBranch.fill_timesteps": branch.fill_timesteps,
        "scheduling_minimax_h3_euler_ancestral._require_finite_tensor": (
            upstream.scheduling._require_finite_tensor
        ),
        "scheduling_minimax_h3_euler_ancestral._validate_unit_timestep": (
            upstream.scheduling._validate_unit_timestep
        ),
    }


@pytest.fixture
def upstream(monkeypatch):
    directory = upstream_dir()
    if directory is None:
        pytest.skip("vllm_omni/diffusion/models/minimax_h3 not found on sys.path")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for name in _SUBMODULES:
        # Undone last-in first-out: whatever a test or the import below puts
        # under these names, teardown restores the state before the test.
        monkeypatch.setitem(sys.modules, f"{_PKG}.{name}", None)
        monkeypatch.delitem(sys.modules, f"{_PKG}.{name}")
    return load_upstream_minimax_h3(
        directory, lambda name, module: monkeypatch.setitem(sys.modules, name, module)
    )


# Packed layouts: video update mask, audio update mask, locked audio rows,
# audio rows packed ahead of the video rows (ref2va).
_LAYOUTS = {
    "t2va": ([True] * 12, [True] * 6, False, False),
    "fl2va-first": ([False] * 4 + [True] * 12, [True] * 6, False, False),
    "fl2va-last": ([True] * 12 + [False] * 4, [True] * 6, False, False),
    "fl2va-first-last": (
        [False] * 4 + [True] * 12 + [False] * 4,
        [True] * 6,
        False,
        False,
    ),
    "ref2va-image": ([False] * 4 + [True] * 12, [True] * 6, False, True),
    "ref2va-audio": ([True] * 12, [False] * 4 + [True] * 6, False, True),
    "ref2va-mixed": ([False] * 4 + [True] * 12, [False] * 4 + [True] * 6, False, True),
    "t2va-locked-audio": ([True] * 12, [True] * 6, True, False),
}


def _branch(upstream, layout: str):
    video_update, audio_update, locked, audio_first = _LAYOUTS[layout]
    generator = torch.Generator().manual_seed(0)
    text_len, pad = 5, 3
    n_img, n_audio = len(video_update), len(audio_update)
    first, second = (n_audio, n_img) if audio_first else (n_img, n_audio)
    first_pos = torch.arange(text_len, text_len + first)
    second_pos = torch.arange(text_len + first, text_len + first + second)
    img_pos, audio_pos = (
        (second_pos, first_pos) if audio_first else (first_pos, second_pos)
    )
    used = text_len + n_img + n_audio
    seq_len = used + pad
    packed = {
        "seq_len": torch.tensor(seq_len),
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "update_mask": torch.tensor(video_update),
        "audio_update_mask": torch.tensor(audio_update),
        "text_pos": torch.arange(text_len),
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
        "img_position_ids": torch.zeros(seq_len, 3),
        "latent_grid": torch.tensor([3, 2, 2]),
        "video_row_start": torch.tensor(text_len),
    }
    branch = upstream.loop.MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(text_len, 8, generator=generator),
        token_tags=torch.zeros(seq_len, dtype=torch.long),
        device=_CPU,
    )
    if locked:
        branch.locked_audio_rows = torch.randn(
            sum(audio_update), 32, generator=generator
        )
    return branch


def _loop_inputs(branch) -> dict:
    generator = torch.Generator().manual_seed(2)
    n_cond = int((~branch.update_mask).sum())
    n_ref = int((~branch.audio_update_mask).sum())
    return {
        "initial_video_rows": torch.randn(
            branch.img_pos.shape[0], 96, generator=generator
        ),
        "initial_audio_rows": torch.randn(
            branch.audio_pos.shape[0], 32, generator=generator
        ),
        "keyframe_cond_rows": (
            torch.randn(n_cond, 96, generator=generator) if n_cond else None
        ),
        "audio_ref_rows": torch.randn(n_ref, 32, generator=generator)
        if n_ref
        else None,
    }


def _schedule(shift: float) -> list[float]:
    """8-step shifted rectified-flow sigmas, 1 -> 0."""
    sigmas = [1.0 - index / _STEPS for index in range(_STEPS + 1)]
    return [shift * s / (1.0 + (shift - 1.0) * s) for s in sigmas]


_VIDEO_SIGMAS = _schedule(12.0)
_AUDIO_SIGMAS = _schedule(3.0)


def _step_kwargs(step: int) -> dict:
    t_v, t_a = 1.0 - _VIDEO_SIGMAS[step], 1.0 - _AUDIO_SIGMAS[step]
    return {
        "t_video": t_v,
        "t_audio": t_a,
        "imgvid_cond_timestep": max(t_v, 0.999),
        "audio_ref_cond_timestep": max(t_a, 1.0),
    }


class _VelocityModel:
    """Random velocities that also depend on every DiT input the loop builds."""

    def __init__(self, branch, nan_step: int | None = None):
        self.generator = torch.Generator().manual_seed(1)
        self.img_pos = branch.img_pos
        self.audio_pos = branch.audio_pos
        self.nan_step = nan_step
        self.inputs = []

    def __call__(self, **kwargs):
        step = len(self.inputs)
        timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        x = kwargs["x"][0].index_select(0, self.img_pos)
        audio_x = kwargs["audio_x"][0].index_select(0, self.audio_pos)
        self.inputs.append(
            (x.clone(), audio_x.clone(), timesteps.clone(), kwargs["inverse_indices"])
        )
        v_video = (
            torch.randn(x.shape, generator=self.generator)
            + 0.5 * x
            - timesteps.index_select(0, self.img_pos)[:, None]
        )
        v_audio = (
            torch.randn(audio_x.shape, generator=self.generator)
            - 0.25 * audio_x
            + timesteps.index_select(0, self.audio_pos)[:, None]
        )

        if step == self.nan_step:
            v_video[v_video.shape[0] // 2, 0] = float("nan")
        return v_video.to(torch.bfloat16), v_audio.to(torch.bfloat16)


def _denoise(upstream, branch, model, sampler: str = "euler", **extra):
    """Run upstream's loop on ``branch`` with the shared schedules."""
    return upstream.loop.minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        **_loop_inputs(branch),
        sigmas_video=_VIDEO_SIGMAS,
        sigmas_audio=_AUDIO_SIGMAS,
        device=_CPU,
        sampler=sampler,
        **extra,
    )


def _run(upstream, layout: str, sampler: str = "euler", nan_step=None):
    branch = _branch(upstream, layout)
    model = _VelocityModel(branch, nan_step=nan_step)
    steps = []
    rows = _denoise(
        upstream,
        branch,
        model,
        sampler,
        on_step=lambda step, video, audio: steps.append(step),
    )
    return rows, model, branch, steps


def _patch_tpu(monkeypatch) -> list:
    """Install the loop patches, force TPU dispatch, count stream flushes."""
    assert minimax_h3.patch_denoise_branch_timestep_rows()
    assert minimax_h3.patch_scheduler_finite_checks()
    assert minimax_h3.patch_denoise_loop_rope_table()
    monkeypatch.setattr(minimax_h3, "_is_tpu", lambda value: True)
    flushes = []
    monkeypatch.setattr(minimax_h3, "_flush_stream", flushes.append)
    return flushes


def _record_warnings(monkeypatch) -> list:
    warnings = []
    monkeypatch.setattr(
        minimax_h3.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    return warnings


# -----------------------------------------------------------------------------
# Source pins
# -----------------------------------------------------------------------------


def test_upstream_sources_match_pins(upstream):
    functions = upstream_functions(upstream)
    assert functions.keys() == minimax_h3._UPSTREAM_SOURCE_SHA256.keys()
    for key, func in functions.items():
        assert (
            minimax_h3._source_sha256(func) == (minimax_h3._UPSTREAM_SOURCE_SHA256[key])
        ), key


# -----------------------------------------------------------------------------
# Host-built timestep table
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_host_timestep_table_matches_torch_unique(upstream, monkeypatch, layout):
    cls = upstream.loop.MiniMaxH3DenoiseBranch
    original = cls.forward_kwargs
    flushes = _patch_tpu(monkeypatch)
    rows = minimax_h3.H3_UNIQUE_TIMESTEP_ROWS
    branch = _branch(upstream, layout)
    generator = torch.Generator().manual_seed(3)
    for step in range(_STEPS):
        kwargs = {
            "video_rows": torch.randn(branch.img_pos.shape[0], 96, generator=generator),
            "audio_rows": torch.randn(
                branch.audio_pos.shape[0], 32, generator=generator
            ),
            **_step_kwargs(step),
        }
        expected = original(branch, **kwargs)
        flushes.clear()
        out = branch.forward_kwargs(**kwargs)
        assert flushes == [_CPU]
        # Upstream's table: torch.unique(return_inverse=True) of fill_timesteps.
        timesteps = torch.empty(branch.seq_len, dtype=torch.float32)
        branch.fill_timesteps(
            timesteps,
            **{
                k: v for k, v in kwargs.items() if k not in ("video_rows", "audio_rows")
            },
        )
        unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
        assert torch.equal(expected["unique_timesteps"], unique)
        count = unique.shape[0]
        assert out["unique_timesteps"].dtype == torch.float32
        assert out["unique_timesteps"].shape == (rows,), (layout, step)
        assert torch.equal(out["unique_timesteps"][:count], unique), (layout, step)
        assert torch.equal(
            out["unique_timesteps"][count:], unique[-1:].expand(rows - count)
        )
        assert out["inverse_indices"].dtype == torch.int64
        assert torch.equal(out["inverse_indices"], inverse), (layout, step)
        assert torch.equal(out["x"], expected["x"])
        assert torch.equal(out["audio_x"], expected["audio_x"])
        assert out.keys() == expected.keys()
        for key, value in branch.static_kwargs.items():
            assert out[key] is value


def test_host_timestep_table_skips_latent_edits(upstream, monkeypatch):
    cls = upstream.loop.MiniMaxH3DenoiseBranch
    original = cls.forward_kwargs
    flushes = _patch_tpu(monkeypatch)
    branch = _branch(upstream, "t2va")
    kwargs = {
        "video_rows": torch.zeros(12, 96),
        "audio_rows": torch.zeros(6, 32),
        **_step_kwargs(3),
    }
    edit = torch.linspace(0.1, 0.9, 12)
    expected = original(branch, **kwargs, video_target_timesteps=edit)
    out = branch.forward_kwargs(**kwargs, video_target_timesteps=edit)
    assert flushes == []
    assert "_tpu_timestep_plan" not in branch.__dict__
    assert torch.equal(out["inverse_indices"], expected["inverse_indices"])
    assert torch.equal(out["unique_timesteps"], expected["unique_timesteps"])


def test_host_table_cutoff_between_measured_shapes():
    # The host table helps up to 30,592 packed rows and costs 8% per step at
    # 42,304 packed rows (768x768x10s, TP8).
    assert 30_592 < minimax_h3._HOST_TABLE_MAX_SEQ_LEN <= 42_304


# seq_len just below / at / above the cut-off.
@pytest.mark.parametrize(("offset", "host"), [(1, True), (0, False), (-1, False)])
def test_host_timestep_table_only_below_cutoff(upstream, monkeypatch, offset, host):
    cls = upstream.loop.MiniMaxH3DenoiseBranch
    original = cls.forward_kwargs
    flushes = _patch_tpu(monkeypatch)
    branch = _branch(upstream, "fl2va-first")
    monkeypatch.setattr(minimax_h3, "_HOST_TABLE_MAX_SEQ_LEN", branch.seq_len + offset)
    rows = minimax_h3.H3_UNIQUE_TIMESTEP_ROWS
    kwargs = {
        "video_rows": torch.zeros(16, 96),
        "audio_rows": torch.zeros(6, 32),
        **_step_kwargs(3),
    }
    expected = original(branch, **kwargs)
    out = branch.forward_kwargs(**kwargs)
    assert flushes == ([_CPU] if host else [])
    assert ("_tpu_timestep_plan" in branch.__dict__) is host
    # Either way: upstream's table padded to the fixed width.
    count = expected["unique_timesteps"].shape[0]
    assert out["unique_timesteps"].shape == (rows,)
    assert torch.equal(out["unique_timesteps"][:count], expected["unique_timesteps"])
    assert torch.equal(out["inverse_indices"], expected["inverse_indices"])
    assert torch.equal(out["x"], expected["x"])
    assert torch.equal(out["audio_x"], expected["audio_x"])


# -----------------------------------------------------------------------------
# Sync-free TPU loop
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_tpu_loop_bit_identical(upstream, monkeypatch, layout, sampler):
    (video_ref, audio_ref), model_ref, _, steps_ref = _run(upstream, layout, sampler)
    progress_ref = list(upstream.progress)
    upstream.progress.clear()

    flushes = _patch_tpu(monkeypatch)
    (video, audio), model, branch, steps = _run(upstream, layout, sampler)

    assert "_tpu_row_indices" in branch.__dict__  # the TPU copy ran
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)
    assert steps == steps_ref == list(range(_STEPS))
    assert upstream.progress == progress_ref
    assert len(model.inputs) == len(model_ref.inputs) == _STEPS
    for (x, audio_x, t, inverse), (x_ref, audio_x_ref, t_ref, inverse_ref) in zip(
        model.inputs, model_ref.inputs, strict=True
    ):
        assert torch.equal(x, x_ref)
        assert torch.equal(audio_x, audio_x_ref)
        assert torch.equal(t, t_ref)
        assert torch.equal(inverse, inverse_ref)
    if sampler == "euler":
        # One flush per upstream sync: forward_kwargs (1), the two velocity
        # gathers (2) and, per stream, 3 row gathers / scatters + 5 rf_v_to_x0
        # + 3 euler checks (11 x 2), plus one per condition-anchor reset.
        video_update, audio_update, _, _ = _LAYOUTS[layout]
        per_step = 25 + (not all(video_update)) + (not all(audio_update))
        assert len(flushes) == per_step * _STEPS
        assert set(flushes) == {_CPU}


def test_tpu_loop_above_host_table_cutoff(upstream, monkeypatch):
    (video_ref, audio_ref), model_ref, _, _ = _run(upstream, "fl2va-first")
    flushes = _patch_tpu(monkeypatch)
    monkeypatch.setattr(minimax_h3, "_HOST_TABLE_MAX_SEQ_LEN", 0)
    (video, audio), model, branch, _ = _run(upstream, "fl2va-first")

    assert "_tpu_row_indices" in branch.__dict__  # the TPU copy ran
    assert "_tpu_timestep_plan" not in branch.__dict__  # upstream's table
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)
    for inputs, inputs_ref in zip(model.inputs, model_ref.inputs, strict=True):
        for value, value_ref in zip(inputs, inputs_ref, strict=True):
            assert torch.equal(value, value_ref)
    # As in test_tpu_loop_bit_identical (26 per step for this layout), minus
    # the forward_kwargs flush: upstream's forward_kwargs syncs by itself.
    assert len(flushes) == 25 * _STEPS


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
def test_tpu_loop_nan_raises_upstream_error(upstream, monkeypatch, sampler):
    with pytest.raises(ValueError, match=r"^v must be finite$"):
        _run(upstream, "fl2va-first", sampler, nan_step=3)

    _patch_tpu(monkeypatch)
    branch = _branch(upstream, "fl2va-first")
    model = _VelocityModel(branch, nan_step=3)
    with pytest.raises(ValueError, match=r"^v must be finite$"):
        _denoise(upstream, branch, model, sampler)
    assert "_tpu_row_indices" in branch.__dict__
    # Euler reads the checks once after the loop; res_multistep stops at the
    # first inline solver check (step 3) and raises the earlier deferred one.
    assert len(model.inputs) == (_STEPS if sampler == "euler" else 4)
    assert minimax_h3._DEFERRED_CHECK_SCOPES == []


def test_deferred_checks_order():
    checks = minimax_h3._DeferredChecks()
    checks.record("a must be finite", torch.tensor(True), fails_if=False)
    checks.record("t must be in [0, 1]", torch.tensor(False), fails_if=True)
    checks.raise_first_failure()
    checks.record("b must be finite", torch.tensor(False), fails_if=False)
    checks.record("t must be in [0, 1]", torch.tensor(True), fails_if=True)
    with pytest.raises(ValueError, match=r"^b must be finite$"):
        checks.raise_first_failure()
    minimax_h3._DeferredChecks().raise_first_failure()


def test_finite_checks_outside_the_loop_unchanged(upstream, monkeypatch):
    _patch_tpu(monkeypatch)
    scheduling = upstream.scheduling
    assert scheduling._require_finite_tensor.__wrapped__ is (
        scheduling._tpu_orig__require_finite_tensor
    )
    with pytest.raises(ValueError, match=r"^x must be finite$"):
        scheduling._require_finite_tensor(torch.tensor([float("nan")]), "x")
    with pytest.raises(ValueError, match=r"^t must be in \[0, 1\]$"):
        scheduling._validate_unit_timestep(torch.tensor(2.0), "t")
    with pytest.raises(ValueError, match=r"^t must be a torch.Tensor$"):
        scheduling._validate_unit_timestep(0.5, "t")
    scheduling.minimax_h3_rf_v_to_x0(
        torch.zeros(2, 3), torch.ones(2, 3), torch.tensor(0.5)
    )


def test_scheduler_patch_idempotent_and_noop(upstream, monkeypatch):
    scheduling = upstream.scheduling
    importer = types.ModuleType("vllm_omni.fake_finite_importer")
    importer._require_finite_tensor = scheduling._require_finite_tensor
    monkeypatch.setitem(sys.modules, importer.__name__, importer)
    assert minimax_h3.patch_scheduler_finite_checks()
    wrapper = scheduling._require_finite_tensor
    assert importer._require_finite_tensor is wrapper
    assert not minimax_h3.patch_scheduler_finite_checks()
    assert scheduling._require_finite_tensor is wrapper

    monkeypatch.setitem(sys.modules, minimax_h3.SCHEDULER_MODULE, None)
    assert minimax_h3.patch_scheduler_finite_checks() is False


# -----------------------------------------------------------------------------
# Fallbacks
# -----------------------------------------------------------------------------


def test_modified_upstream_loop_falls_back(upstream, monkeypatch, tmp_path):
    (video_ref, audio_ref), _, _, _ = _run(upstream, "fl2va-first-last")
    source = Path(upstream.loop.__file__).read_text()
    marker = "    num_steps = len(sigmas_video) - 1\n"
    assert source.count(marker) == 1
    path = tmp_path / "denoise_loop.py"
    path.write_text(source.replace(marker, marker + "    # locally modified\n"))
    spec = importlib.util.spec_from_file_location(minimax_h3.DENOISE_LOOP_MODULE, path)
    modified = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, minimax_h3.DENOISE_LOOP_MODULE, modified)
    spec.loader.exec_module(modified)
    upstream.loop = modified
    warnings = _record_warnings(monkeypatch)

    _patch_tpu(monkeypatch)
    (video, audio), _, branch, _ = _run(upstream, "fl2va-first-last")
    assert "_tpu_row_indices" not in branch.__dict__
    assert "_tpu_timestep_plan" in branch.__dict__  # forward_kwargs is unchanged
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)
    assert [args[1] for args in warnings] == ["denoise_loop.minimax_h3_denoise_loop"]


def test_modified_upstream_finite_check_is_not_patched(upstream, monkeypatch):
    key = "scheduling_minimax_h3_euler_ancestral._require_finite_tensor"
    monkeypatch.setitem(minimax_h3._UPSTREAM_SOURCE_SHA256, key, "0" * 64)
    warnings = _record_warnings(monkeypatch)
    original = upstream.scheduling._require_finite_tensor
    assert minimax_h3.patch_scheduler_finite_checks()
    assert upstream.scheduling._require_finite_tensor is original
    assert "_tpu_orig__validate_unit_timestep" in vars(upstream.scheduling)
    assert [args[1] for args in warnings] == [key]


def test_tpu_loop_skips_latent_edits_and_other_devices(upstream):
    original = upstream.loop.minimax_h3_denoise_loop
    covered = {
        "model": object(),
        "positive": object(),
        "initial_video_rows": None,
        "initial_audio_rows": None,
        "keyframe_cond_rows": None,
        "sigmas_video": [1.0, 0.0],
        "sigmas_audio": [1.0, 0.0],
        "device": "tpu:0",
    }
    arguments = minimax_h3._tpu_denoise_loop_arguments(original, covered)
    assert arguments["sampler"] == "euler"
    assert arguments["imgvid_cond_noise_aug_for_inference"] == (
        upstream.loop.MINIMAX_H3_IMGVID_COND_TIMESTEP
    )
    assert arguments["audio_ref_rows"] is None and arguments["on_step"] is None
    for extra in (
        {"video_edit": object()},
        {"audio_edit": object()},
        {"sampler": "unknown"},
        {"unexpected": 1},
    ):
        assert (
            minimax_h3._tpu_denoise_loop_arguments(original, {**covered, **extra})
            is None
        ), extra
    assert minimax_h3._tpu_denoise_loop_arguments(original, {"model": 1}) is None
    # CPU runs upstream's loop (no row-index cache on the branch).
    assert minimax_h3.patch_denoise_loop_rope_table()
    _, _, branch, _ = _run(upstream, "t2va")
    assert "_tpu_row_indices" not in branch.__dict__


def test_flush_stream_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        minimax_h3, "_synchronize_tensors", lambda: lambda **kw: calls.append(kw)
    )
    minimax_h3._flush_stream(_CPU)
    minimax_h3._flush_stream(torch.zeros(1))
    minimax_h3._flush_stream(None)
    assert calls == []
    minimax_h3._flush_stream(SimpleNamespace(type="tpu"))
    minimax_h3._flush_stream("tpu:1")
    assert calls == [{"wait": False}, {"wait": False}]


def test_tpu_loop_deferred_failure_obscured_by_runtime_error(upstream, monkeypatch):
    """A deferred NaN check wins over a later inline RuntimeError."""
    _patch_tpu(monkeypatch)
    branch = _branch(upstream, "fl2va-first")

    class ModelWithRuntimeError(_VelocityModel):
        def __call__(self, **kwargs):
            out = super().__call__(**kwargs)
            if len(self.inputs) == 4:  # step 3
                raise RuntimeError("something else broke inline")
            return out

    model = ModelWithRuntimeError(branch, nan_step=2)
    with pytest.raises(ValueError, match=r"^v must be finite$"):
        _denoise(upstream, branch, model)
    assert minimax_h3._DEFERRED_CHECK_SCOPES == []


def test_tpu_loop_plain_runtime_error_propagates(upstream, monkeypatch):
    _patch_tpu(monkeypatch)
    branch = _branch(upstream, "fl2va-first")

    class ModelWithPlainRuntimeError(_VelocityModel):
        def __call__(self, **kwargs):
            if len(self.inputs) == 3:  # step 3
                raise RuntimeError("plain runtime error")
            return super().__call__(**kwargs)

    model = ModelWithPlainRuntimeError(branch, nan_step=None)
    with pytest.raises(RuntimeError, match=r"^plain runtime error$"):
        _denoise(upstream, branch, model)
    assert minimax_h3._DEFERRED_CHECK_SCOPES == []


def test_tpu_loop_deferred_materialization_raises(upstream, monkeypatch):
    """If reading the deferred checks itself fails, the inline error is kept."""
    _patch_tpu(monkeypatch)
    branch = _branch(upstream, "fl2va-first")

    class ModelCorruptingDeferredChecks(_VelocityModel):
        def __call__(self, **kwargs):
            if len(self.inputs) == 3:  # step 3
                checks = minimax_h3._active_deferred_checks()._checks
                checks[0] = (checks[0][0], "not a tensor", False)
                raise RuntimeError("original inline error")
            return super().__call__(**kwargs)

    model = ModelCorruptingDeferredChecks(branch, nan_step=2)
    with pytest.raises(RuntimeError, match=r"^original inline error$"):
        _denoise(upstream, branch, model)
    assert minimax_h3._DEFERRED_CHECK_SCOPES == []
