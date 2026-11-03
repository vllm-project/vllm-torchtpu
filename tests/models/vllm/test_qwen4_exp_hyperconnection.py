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
"""Packed-mix state contracts: owner binding, shapes, reload, caller entry."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HcPackedMixState,
    HyperConnectionConfig,
    combine_then_mix_packed,
    combine_then_mix_terminal,
)

TPU = pytest.mark.tpu_test


@pytest.fixture
def backbone_config_context():
    """Backbone construction and eager dispatch need a current vLLM config.

    ``Qwen4ExpBackbone`` carries the production ``@support_torch_compile``
    wrapper, whose ``__init__`` resolves the active config exactly like worker
    startup; constructing it without one fails before ``forward`` is traced.
    Yields the config so callers can also use the production eager dispatch.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield config


@pytest.fixture(autouse=True)
def cpu_default(request):
    # Repository device fixture defaults to TPU; CPU cases intentionally never
    # request it. Explicit factories also protect against a caller default.
    if request.node.get_closest_marker("tpu_test"):
        # Independent module cases must not exhaust shared wrapper variants.
        torch.compiler.reset()
        # Worker initialization precedes HC construction in native execution.
        torch.ones(1, device="tpu").cpu()
        yield
    else:
        with torch.device("cpu"):
            yield


def owner_fixture(tokens=3, seed=383, full_tile=128):
    torch.random.default_generator.manual_seed(seed)
    config = HyperConnectionConfig(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        hc_per_branch_norm=True,
    )
    module = GatedResidual(config, prefill_full_tile=full_tile)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_((torch.randn_like(parameter) * 0.05).bfloat16())
    x = (torch.randn(tokens, 4 * 2560) * 0.5).bfloat16()
    block = (torch.randn(tokens, 2560) * 0.5).bfloat16()
    return module, x, block


def _weight(parameter):
    """Reference math reads a host copy; the module may live on TPU."""
    return parameter.detach().cpu()


def two_owner_fixture(tokens=3, seed=471, full_tile=128):
    """Two same-geometry owners as the boundary fusion's caller pair."""
    torch.random.default_generator.manual_seed(seed)
    config = HyperConnectionConfig(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        hc_per_branch_norm=True,
    )
    prior = GatedResidual(config, role="attn", prefill_full_tile=full_tile)
    following = GatedResidual(config, role="mlp", prefill_full_tile=full_tile)
    with torch.no_grad():
        for module in (prior, following):
            for parameter in module.parameters():
                parameter.copy_((torch.randn_like(parameter) * 0.05).bfloat16())
    x = (torch.randn(tokens, 4 * 2560) * 0.5).bfloat16()
    block = (torch.randn(tokens, 2560) * 0.5).bfloat16()
    return prior, following, x, block


def _reference_norm(module, flat):
    """Per-branch RMS norm with the checkpoint BF16 store boundary."""
    branches, hidden = module.hc_count, module.hidden_size
    eps = module.hc_norm.variance_epsilon
    values = flat.reshape(-1, branches, hidden).float()
    weight = _weight(module.hc_norm.weight).float().reshape(-1, hidden)
    return (
        (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            * (1 + weight)
        )
        .bfloat16()
        .flatten(1)
    )


def _reference_mix(module, flat):
    """Mix with the NVIDIA FP32 sigmoid over the BF16 up-linear output."""
    branches, hidden = module.hc_count, module.hidden_size
    norm = _reference_norm(module, flat)
    down = F.linear(norm, _weight(module.input_mix_weight_down.weight))
    silu = F.silu(down.float() / branches).bfloat16()
    gate = torch.sigmoid(
        F.linear(silu, _weight(module.input_mix_weight_up.weight)).float()
    ).reshape(-1, branches, hidden)
    mixed = (gate * norm.float().reshape(-1, branches, hidden)).mean(1).bfloat16()
    return mixed, norm


def _reference_combine(module, flat, norm, block):
    """Combine with FP32 sigmoid over the BF16 injection projection."""
    branches, hidden = module.hc_count, module.hidden_size
    inject = 2 * torch.sigmoid(
        F.linear(norm, _weight(module.block_inject_weight.weight)).float() / branches
    )
    return (
        (
            flat.float().reshape(-1, branches, hidden)
            + block.float()[:, None] * inject[:, :, None]
        )
        .bfloat16()
        .flatten(1)
    )


def _np(tensor):
    return tensor.detach().float().cpu().numpy()


def _words(tensor):
    """Exact BF16 storage words: signed zeros and payloads included."""
    return tensor.detach().cpu().view(torch.uint16).numpy()


def record_fused_dispatch(registry):
    """Record actual dispatches through a registered fused-op registry.

    Every entry is replaced by a wrapper that delegates to the real registered
    kernel and appends the tile, so a recorded entry is the production call
    itself: eager runs append once per call and compiled runs append from the
    traced graph's replayed side effect. A populated registry only proves
    admission; this proves the op ran. Raw output words are not dispatch
    evidence in either direction (a fused kernel may round exactly like the
    unfused sequence over the same stores), so routing is asserted here and
    never through output word differences.
    """
    calls = []
    for tile, op in list(registry.items()):

        def wrapped(*args, _op=op, _tile=tile, **kwargs):
            calls.append(_tile)
            return _op(*args, **kwargs)

        registry[tile] = wrapped
    return calls


def assert_metrics(actual, expected, threshold=0.01):
    assert np.isfinite(actual).all()
    rel = np.linalg.norm((actual - expected).ravel()) / np.linalg.norm(expected.ravel())
    cos = np.vdot(actual.ravel(), expected.ravel()) / (
        np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel()) + 1e-30
    )
    assert rel <= threshold
    assert cos >= 0.999


def test_mix_packed_outside_admitted_layout_returns_public_mix():
    module, x, _ = owner_fixture()
    packed_mixed, packed_residual = module.mix_packed(x)
    public_mixed, public_residual = module.mix(x)
    torch.testing.assert_close(packed_mixed, public_mixed, rtol=0, atol=0)
    # The fallback keeps the public tuple carrier, not a packed state.
    assert isinstance(packed_residual, tuple) and not isinstance(
        packed_residual, HcPackedMixState
    )
    for first, second in zip(packed_residual, public_residual, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_mix_packed_broadcast_outside_admitted_layout_returns_public_mix():
    """The broadcast entry materializes the expansion when not admitted.

    Host tensors are outside the device-bound admission, so the entry must
    reproduce the public ``mix_packed`` result over the explicitly expanded
    embedding exactly, with the public tuple carrier.
    """
    module, x, _ = owner_fixture()
    embedding = x[:, : module.hidden_size]
    broadcast_mixed, broadcast_residual = module.mix_packed_broadcast(embedding)
    public_mixed, public_residual = module.mix_packed(
        embedding.repeat(1, module.hc_count)
    )
    torch.testing.assert_close(broadcast_mixed, public_mixed, rtol=0, atol=0)
    assert isinstance(broadcast_residual, tuple) and not isinstance(
        broadcast_residual, HcPackedMixState
    )
    for first, second in zip(broadcast_residual, public_residual, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_broadcast_state_combine_expands_first_residual():
    """A broadcast state's combine runs over the genuine expanded words.

    The state carries the un-expanded first residual; ``combine`` must expand
    it branchwise before the public arithmetic, so the result equals the
    reference combine over the explicitly expanded plane.
    """
    module, x, block = owner_fixture()
    embedding = x[:, : module.hidden_size]
    expanded = embedding.repeat(1, module.hc_count)
    norm = _reference_norm(module, expanded)
    # The packed kernel stores the standalone linear's BF16 logits; the state
    # carries them with the un-expanded residual.
    logits = F.linear(norm, module.block_inject_weight.weight)
    state = HcPackedMixState(embedding, logits, module._hc_owner_id, broadcast=True)
    expected = _reference_combine(module, expanded, norm, block)
    actual = module.combine(block, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combine_then_mix_fallback_with_broadcast_state_returns_unfused_sequence():
    """A broadcast state outside admission keeps the unfused public sequence.

    Host tensors are outside the device-bound admission, so the boundary entry
    must fall back to the prior owner's expanding combine followed by the
    following owner's public packed mix, exactly like a non-broadcast state.
    """
    prior, following, x, block = two_owner_fixture(tokens=6)
    embedding = x[:, : prior.hidden_size]
    expanded = embedding.repeat(1, prior.hc_count)
    norm = _reference_norm(prior, expanded)
    logits = F.linear(norm, prior.block_inject_weight.weight)
    state = HcPackedMixState(embedding, logits, prior._hc_owner_id, broadcast=True)
    combined, mixed, residual = combine_then_mix_packed(prior, following, block, state)
    expected_combined = prior.combine(block, state)
    expected_mixed, expected_state = following.mix_packed(expected_combined)
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)
    assert isinstance(residual, tuple) and not isinstance(residual, HcPackedMixState)
    for first, second in zip(residual, expected_state, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_packed_state_combine_keeps_original_gate_boundary():
    module, x, block = owner_fixture()
    norm = _reference_norm(module, x)
    # The packed kernel stores the standalone linear's BF16 logits; combine
    # only applies the FP32 sigmoid. The owner id is the module's own token.
    logits = F.linear(norm, module.block_inject_weight.weight)
    state = HcPackedMixState(x, logits, module._hc_owner_id)
    expected = _reference_combine(module, x, norm, block)
    actual = module.combine(block, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combine_rejects_foreign_packed_state():
    module, x, block = owner_fixture()
    other, _, _ = owner_fixture(seed=977)
    norm = _reference_norm(module, x)
    logits = F.linear(norm, module.block_inject_weight.weight)
    state = HcPackedMixState(x, logits, other._hc_owner_id)
    with pytest.raises(RuntimeError, match="another hyper-connection"):
        module.combine(block, state)


@pytest.mark.parametrize(
    "modify",
    [
        lambda logits: logits[..., :2],
        lambda logits: torch.cat((logits, logits[..., :2]), dim=-1),
        lambda logits: logits[:-1],
    ],
    ids=["narrow", "wide", "short-rows"],
)
def test_combine_rejects_mismatched_packed_logits_shape(modify):
    module, x, block = owner_fixture(tokens=5)
    norm = _reference_norm(module, x)
    logits = F.linear(norm, module.block_inject_weight.weight)
    state = HcPackedMixState(x, modify(logits), module._hc_owner_id)
    with pytest.raises(RuntimeError, match="shape does not match"):
        module.combine(block, state)


def test_combine_then_mix_fallback_returns_unfused_sequence():
    # Host tensors are outside the device-bound admission, so the single
    # boundary entry point must reproduce the unfused public calls exactly.
    prior, following, x, block = two_owner_fixture(tokens=6)
    mixed_prior, residual = prior.mix_packed(x)
    combined, mixed, state = combine_then_mix_packed(prior, following, block, residual)
    expected_combined = prior.combine(block, residual)
    expected_mixed, expected_state = following.mix_packed(expected_combined)
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)
    assert isinstance(state, tuple) and not isinstance(state, HcPackedMixState)
    for first, second in zip(state, expected_state, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_combine_then_mix_rejects_foreign_packed_state():
    prior, following, x, block = two_owner_fixture(tokens=6)
    other, _, _ = owner_fixture(seed=977)
    norm = _reference_norm(prior, x)
    logits = F.linear(norm, prior.block_inject_weight.weight)
    state = HcPackedMixState(x, logits, other._hc_owner_id)
    with pytest.raises(RuntimeError, match="another hyper-connection"):
        combine_then_mix_packed(prior, following, block, state)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 256, 384])
def test_hc_tpu_packed_mix_and_combine(compiled, tokens):
    with torch.device("cpu"):
        module, x, block = owner_fixture(tokens=tokens)
    assert module._prefill_inject_ops
    module = module.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")

    def chain(hyper_input, block_output):
        mixed, state = module.mix_packed(hyper_input)
        return mixed, module.combine(block_output, state)

    # Exact storage words of every read-only operand, captured before the
    # first tested call: float equality would hide signed zeros and BF16
    # payload differences, and post-call snapshots could not observe a call
    # that overwrote its operands.
    readonly = (
        x_tpu,
        block_tpu,
        module.hc_norm.weight,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        module.block_inject_weight.weight,
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]

    run = (
        torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else chain
    )
    with torch.inference_mode():
        eager_mixed, eager_state = module.mix_packed(x_tpu)
        assert isinstance(eager_state, HcPackedMixState)
        actual = run(x_tpu, block_tpu)
        repeated = run(x_tpu, block_tpu)
    # Repeat determinism and eager/compiled agreement at raw BF16 word level.
    for first, second in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    np.testing.assert_array_equal(_words(actual[0].cpu()), _words(eager_mixed.cpu()))
    mixed_ref, norm_ref = _reference_mix(module, x)
    expected_out = _reference_combine(module, x, norm_ref, block)
    expected_logits = F.linear(norm_ref, _weight(module.block_inject_weight.weight))
    assert_metrics(_np(actual[0]), _np(mixed_ref))
    assert_metrics(_np(actual[1]), _np(expected_out))
    # The carried logits keep the standalone linear's BF16 projection boundary.
    assert_metrics(_np(eager_state.inject_logits), _np(expected_logits))
    # The retained inputs and parameters still hold their exact storage words.
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 256, 384])
def test_hc_tpu_boundary_mix_matches_reference(compiled, tokens):
    with torch.device("cpu"):
        prior, following, x, block = two_owner_fixture(tokens=tokens)
    assert following._prefill_boundary_ops
    prior = prior.to("tpu")
    following = following.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")

    def boundary(hyper_input, block_output):
        mixed, state = prior.mix_packed(hyper_input)
        return combine_then_mix_packed(prior, following, block_output, state)

    # Exact storage words of every read-only operand, captured before the
    # first tested call: float equality would hide signed zeros and BF16
    # payload differences, and post-call snapshots could not observe a call
    # that overwrote its operands.
    readonly = (
        x_tpu,
        block_tpu,
        *(
            parameter
            for module in (prior, following)
            for parameter in (
                module.hc_norm.weight,
                module.input_mix_weight_down.weight,
                module.input_mix_weight_up.weight,
                module.block_inject_weight.weight,
            )
        ),
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]

    run = (
        torch.compile(boundary, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else boundary
    )
    with torch.inference_mode():
        eager = boundary(x_tpu, block_tpu)
        actual = run(x_tpu, block_tpu)
        repeated = run(x_tpu, block_tpu)

    def boundary_outputs(result):
        combined, mixed, state = result
        assert isinstance(state, HcPackedMixState)
        return combined, mixed, state.inject_logits

    for result in (actual, repeated):
        # Repeat determinism and eager/compiled agreement at raw word level.
        for first, second in zip(
            boundary_outputs(eager), boundary_outputs(result), strict=True
        ):
            np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    for first, second in zip(
        boundary_outputs(actual), boundary_outputs(repeated), strict=True
    ):
        np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    norm_a = _reference_norm(prior, x)
    expected_combined = _reference_combine(prior, x, norm_a, block)
    expected_mixed, expected_norm_b = _reference_mix(following, expected_combined)
    expected_logits = F.linear(
        expected_norm_b, _weight(following.block_inject_weight.weight)
    )
    assert_metrics(_np(actual[0]), _np(expected_combined))
    assert_metrics(_np(actual[1]), _np(expected_mixed))
    # The fused state carries this owner's token and the BF16 projection
    # boundary of a standalone linear over the stored combined value.
    assert isinstance(actual[2], HcPackedMixState)
    assert actual[2].owner_id == following._hc_owner_id
    np.testing.assert_array_equal(
        _words(actual[2].hyper_input.cpu()), _words(actual[0].cpu())
    )
    assert_metrics(_np(actual[2].inject_logits), _np(expected_logits))
    # The returned state remains a live operand of the following combine.
    with torch.inference_mode():
        consumed = following.combine(block_tpu, actual[2])
    expected_consumed = _reference_combine(
        following, expected_combined, expected_norm_b, block
    )
    assert_metrics(_np(consumed.cpu()), _np(expected_consumed))
    # The retained inputs and parameters still hold their exact storage words.
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 256])
def test_hc_tpu_initial_broadcast_mix_and_boundary(compiled, tokens):
    """The chain's first mix broadcasts the un-expanded residual in kernel.

    ``mix_packed_broadcast`` admits the compact embedding and returns a
    broadcast state whose hyper input is that un-expanded value, and the
    following boundary fuses through the broadcast kernel while the plain
    boundary and inject registries stay undispatched. Every output is checked
    against the reference over the explicitly expanded plane through the
    original L2/cosine oracles; the state's own combine expands the compact
    residual first and keeps the public path.
    """
    with torch.device("cpu"):
        prior, following, x, block = two_owner_fixture(tokens=tokens)
    assert prior._prefill_inject_broadcast_ops
    assert following._prefill_boundary_broadcast_ops
    prior = prior.to("tpu")
    following = following.to("tpu")
    embedding = x[:, : prior.hidden_size]
    embed_tpu, block_tpu = embedding.to("tpu"), block.to("tpu")
    expanded = embedding.repeat(1, prior.hc_count)

    def initial_chain(unexpanded, block_output):
        mixed, state = prior.mix_packed_broadcast(unexpanded)
        return combine_then_mix_packed(prior, following, block_output, state)

    readonly = (
        embed_tpu,
        block_tpu,
        *(
            parameter
            for module in (prior, following)
            for parameter in (
                module.hc_norm.weight,
                module.input_mix_weight_down.weight,
                module.input_mix_weight_up.weight,
                module.block_inject_weight.weight,
            )
        ),
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]
    broadcast_inject_calls = record_fused_dispatch(prior._prefill_inject_broadcast_ops)
    broadcast_boundary_calls = record_fused_dispatch(
        following._prefill_boundary_broadcast_ops
    )
    plain_inject_calls = record_fused_dispatch(prior._prefill_inject_ops)
    plain_boundary_calls = record_fused_dispatch(following._prefill_boundary_ops)

    run = (
        torch.compile(initial_chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else initial_chain
    )
    with torch.inference_mode():
        eager_mixed, eager_state = prior.mix_packed_broadcast(embed_tpu)
        assert isinstance(eager_state, HcPackedMixState)
        assert eager_state.broadcast
        # The broadcast state carries the un-expanded residual's exact words.
        np.testing.assert_array_equal(
            _words(eager_state.hyper_input.cpu()), _words(embed_tpu.cpu())
        )
        # The broadcast state's own combine expands it first: the public path
        # over the genuine expanded residual.
        consumed = prior.combine(block_tpu, eager_state)
        eager = initial_chain(embed_tpu, block_tpu)
        actual = run(embed_tpu, block_tpu)
        repeated = run(embed_tpu, block_tpu)

    def boundary_outputs(result):
        combined, mixed, state = result
        assert isinstance(state, HcPackedMixState)
        assert not state.broadcast
        return combined, mixed, state.inject_logits

    for result in (actual, repeated):
        # Repeat determinism and eager/compiled agreement at raw word level.
        for first, second in zip(
            boundary_outputs(eager), boundary_outputs(result), strict=True
        ):
            np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    for first, second in zip(
        boundary_outputs(actual), boundary_outputs(repeated), strict=True
    ):
        np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    norm_a = _reference_norm(prior, expanded)
    expected_mixed, _ = _reference_mix(prior, expanded)
    expected_logits_a = F.linear(norm_a, _weight(prior.block_inject_weight.weight))
    expected_combined = _reference_combine(prior, expanded, norm_a, block)
    expected_mixed_b, expected_norm_b = _reference_mix(following, expected_combined)
    expected_logits_b = F.linear(
        expected_norm_b, _weight(following.block_inject_weight.weight)
    )
    assert_metrics(_np(eager_mixed), _np(expected_mixed))
    assert_metrics(_np(eager_state.inject_logits), _np(expected_logits_a))
    assert_metrics(_np(consumed.cpu()), _np(expected_combined))
    assert_metrics(_np(actual[0]), _np(expected_combined))
    assert_metrics(_np(actual[1]), _np(expected_mixed_b))
    assert isinstance(actual[2], HcPackedMixState)
    assert actual[2].owner_id == following._hc_owner_id
    assert_metrics(_np(actual[2].inject_logits), _np(expected_logits_b))
    # The fused state's hyper input is exactly the stored combined value.
    np.testing.assert_array_equal(
        _words(actual[2].hyper_input.cpu()), _words(actual[0].cpu())
    )
    # Routing: the broadcast registries dispatched, and neither plain registry
    # (the expanded-input kernels) ever ran.
    assert broadcast_inject_calls
    assert broadcast_boundary_calls
    assert not plain_inject_calls
    assert not plain_boundary_calls
    # The retained inputs and parameters still hold their exact storage words.
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 512])
def test_hc_tpu_boundary_mix_auto_admits_compact_whole_tiles(compiled, tokens):
    """Auto policy dispatches the fused kernels at compact whole-tile counts.

    The staged tile is the kernel's unit of work, so the production auto
    admission covers the actual runtime token count whenever it is whole
    tiles: 128 tokens dispatch the tile-128 op and 512 the tile-256 op on the
    resource-admitted geometry, with no token floor in between. The fallback
    for ragged counts stays covered by the forced-tile 130-token case.
    """
    with torch.device("cpu"):
        prior, following, x, block = two_owner_fixture(tokens=tokens, full_tile=None)
    assert following.prefill_full_tile == "auto"
    assert following._prefill_boundary_ops
    prior = prior.to("tpu")
    following = following.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")

    def boundary(hyper_input, block_output):
        mixed, state = prior.mix_packed(hyper_input)
        return combine_then_mix_packed(prior, following, block_output, state)

    readonly = (
        x_tpu,
        block_tpu,
        *(
            parameter
            for module in (prior, following)
            for parameter in (
                module.hc_norm.weight,
                module.input_mix_weight_down.weight,
                module.input_mix_weight_up.weight,
                module.block_inject_weight.weight,
            )
        ),
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]

    run = (
        torch.compile(boundary, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else boundary
    )
    with torch.inference_mode():
        # The packed and boundary kernels dispatch at these compact counts;
        # a fallback would return the public tuple carrier instead.
        _, eager_state = prior.mix_packed(x_tpu)
        assert isinstance(eager_state, HcPackedMixState)
        eager = boundary(x_tpu, block_tpu)
        assert isinstance(eager[2], HcPackedMixState)
        actual = run(x_tpu, block_tpu)
        repeated = run(x_tpu, block_tpu)

    for result in (actual, repeated):
        combined, mixed, state = result
        assert isinstance(state, HcPackedMixState)
        assert state.owner_id == following._hc_owner_id
        for first, second in zip(
            (combined, mixed, state.inject_logits),
            (eager[0], eager[1], eager[2].inject_logits),
            strict=True,
        ):
            np.testing.assert_array_equal(_words(first.cpu()), _words(second.cpu()))
    norm_a = _reference_norm(prior, x)
    expected_combined = _reference_combine(prior, x, norm_a, block)
    expected_mixed, expected_norm_b = _reference_mix(following, expected_combined)
    expected_logits = F.linear(
        expected_norm_b, _weight(following.block_inject_weight.weight)
    )
    assert_metrics(_np(actual[0]), _np(expected_combined))
    assert_metrics(_np(actual[1]), _np(expected_mixed))
    assert_metrics(_np(actual[2].inject_logits), _np(expected_logits))
    # The state's hyper input is exactly the stored combined value.
    np.testing.assert_array_equal(
        _words(actual[2].hyper_input.cpu()), _words(actual[0].cpu())
    )
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)


@TPU
def test_hc_tpu_boundary_falls_back_below_admitted_tokens():
    with torch.device("cpu"):
        prior, following, x, block = two_owner_fixture(tokens=130)
    assert following._prefill_boundary_ops
    prior = prior.to("tpu")
    following = following.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")
    # 130 tokens are not a full tile, so the boundary entry point keeps the
    # unfused public sequence with the tuple carrier.
    with torch.inference_mode():
        _, residual = prior.mix_packed(x_tpu)
        combined, mixed, state = combine_then_mix_packed(
            prior, following, block_tpu, residual
        )
    assert isinstance(state, tuple) and not isinstance(state, HcPackedMixState)
    with torch.inference_mode():
        expected_combined = prior.combine(block_tpu, residual)
        expected_mixed, _ = following.mix_packed(expected_combined)
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)


@TPU
def test_hc_tpu_packed_mix_rejects_foreign_owner():
    with torch.device("cpu"):
        module, x, block = owner_fixture(tokens=128)
        other, _, _ = owner_fixture(tokens=128, seed=977)
    module = module.to("tpu")
    other = other.to("tpu")
    with torch.inference_mode():
        _, state = module.mix_packed(x.to("tpu"))
        assert isinstance(state, HcPackedMixState)
        with pytest.raises(RuntimeError, match="another hyper-connection"):
            other.combine(block.to("tpu"), state)


@TPU
def test_hc_tpu_packed_mix_tracks_reloaded_weights():
    with torch.device("cpu"):
        module, x, block = owner_fixture(tokens=128)
    module = module.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")
    with torch.inference_mode():
        _, first_state = module.mix_packed(x_tpu)
        first = module.combine(block_tpu, first_state)
    # Reload the injection weight after a packed run; the projection must
    # read the live parameter, never a stale packed copy.
    torch.random.default_generator.manual_seed(642)
    replacement = torch.randn(module.block_inject_weight.weight.shape) * 0.05
    with torch.no_grad():
        module.block_inject_weight.weight.copy_(replacement.bfloat16())
    with torch.inference_mode():
        _, second_state = module.mix_packed(x_tpu)
        second = module.combine(block_tpu, second_state)
    norm = _reference_norm(module, x)
    expected = _reference_combine(module, x, norm, block)
    assert_metrics(_np(second.cpu()), _np(expected))
    assert not torch.equal(first.cpu(), second.cpu())


@TPU
def test_hc_tpu_packed_owner_roundtrip_keeps_fallback():
    with torch.device("cpu"):
        module, x, _ = owner_fixture(tokens=128)
    assert module._prefill_inject_ops
    module = module.to("tpu")
    with torch.inference_mode():
        _, state = module.mix_packed(x.to("tpu"))
        assert isinstance(state, HcPackedMixState)
    # Back on the host the owner is outside the device-bound admission; the
    # single entry point returns the public tuple carrier with correct math.
    module = module.to("cpu")
    with torch.inference_mode():
        mixed, residual = module.mix_packed(x)
        assert isinstance(residual, tuple) and not isinstance(
            residual, HcPackedMixState
        )
    expected_mixed, _ = _reference_mix(module, x)
    assert_metrics(_np(mixed), _np(expected_mixed))


class _StandInBlock(nn.Module):
    """Synthetic attention/MoE stand-in: one BF16 linear plus sigmoid."""

    def __init__(self, seed, hidden):
        super().__init__()
        torch.random.default_generator.manual_seed(seed)
        self.weight = nn.Parameter(
            (torch.randn(hidden, hidden) * hidden**-0.5).bfloat16()
        )

    def forward(self, x):
        return torch.sigmoid(F.linear(x, self.weight))


@pytest.mark.parametrize("state_kind", ["tuple", "packed", "broadcast"])
@torch.inference_mode()
def test_indexed_decoder_combines_attention_once_for_each_mix_state(state_kind):
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.projected = []

        def can_prefetch_output(self, hidden):
            return True

        def forward(
            self, mixed, positions, metadata, residual=None, hc_gate=None, **kw
        ):
            self.projected.append(residual is not None)
            output = torch.tanh(mixed)
            if residual is None:
                return output
            return torch.cat(
                [
                    residual[:, branch * 8 : (branch + 1) * 8].float()
                    + output.float() * hc_gate[:, branch : branch + 1]
                    for branch in range(2)
                ],
                dim=-1,
            ).bfloat16()

    torch.manual_seed(758)
    config = HyperConnectionConfig(hidden_size=8, hc_count=2, hc_lowrank=4)
    layer = qwen4_model.Qwen4ExpDecoderLayer.__new__(qwen4_model.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = "indexed_attention"
    layer.self_attn = Attention()
    layer.attn_hyper_connection = GatedResidual(config, prefill_full_tile=0)
    layer.mlp_hyper_connection = GatedResidual(config, prefill_full_tile=0)
    layer.mlp = nn.Identity()
    x = torch.randn(3, 8 if state_kind == "broadcast" else 16).bfloat16()
    expanded = x.repeat(1, 2) if state_kind == "broadcast" else x
    mixed, (hyper, norm) = layer.attn_hyper_connection.mix(expanded)
    attention_mix = None
    if state_kind != "tuple":
        state = HcPackedMixState(
            x,
            F.linear(norm, layer.attn_hyper_connection.block_inject_weight.weight),
            layer.attn_hyper_connection._hc_owner_id,
            broadcast=state_kind == "broadcast",
        )
        attention_mix = qwen4_model.HcCrossLayerMix(expanded, mixed, state)
    valid = torch.tensor([True, False, True])
    combined = layer.attn_hyper_connection.combine(torch.tanh(mixed), (hyper, norm))
    mlp_input, mlp_state = layer.mlp_hyper_connection.mix(combined)
    expected = layer.mlp_hyper_connection.combine(
        torch.where(valid[:, None], mlp_input, 0), mlp_state
    )
    actual = layer.forward(
        expanded,
        torch.arange(3),
        torch.zeros(3, dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.empty(0, 2, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        valid,
        attention_mix=attention_mix,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layer.self_attn.projected == [state_kind == "tuple"]


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_decoder_forward_packed_boundary(monkeypatch, compiled):
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    hidden_size, branches = 2560, 4
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", "128")
    monkeypatch.setattr(
        qwen4_model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        qwen4_model,
        "QwenGatedDeltaNetAttention",
        lambda t, vllm_config, prefix, gqa_interleaved_layout: _StandInBlock(
            11, hidden_size
        ),
    )
    monkeypatch.setattr(
        qwen4_model,
        "Qwen3NextSparseMoeBlock",
        lambda vllm_config, prefix: _StandInBlock(12, hidden_size),
    )
    text = q4exp_text_config(
        hidden_size=hidden_size,
        hc_count=branches,
        hc_lowrank=320,
        hc_per_branch_norm=True,
        vocab_size=32,
        num_hidden_layers=1,
        ple_layer_ids=[],
        layer_types=["linear_attention"],
    )
    cfg = NS(model_config=NS(hf_text_config=text), additional_config={})
    with torch.device("cpu"):
        layer = qwen4_model.Qwen4ExpDecoderLayer(cfg, 0, "model.layers.0")
    assert layer.attn_hyper_connection._prefill_inject_ops
    assert layer.mlp_hyper_connection._prefill_inject_ops
    # The production caller fuses the attention combine into the MLP mix.
    assert layer.mlp_hyper_connection._prefill_boundary_ops
    layer = layer.to("tpu")

    torch.random.default_generator.manual_seed(95)
    tokens = 256
    x = (torch.randn(tokens, branches * hidden_size) * 0.5).bfloat16()
    generator = torch.Generator().manual_seed(96)
    valid = torch.ones(tokens, dtype=torch.bool)
    valid[torch.randperm(tokens, generator=generator)[:9]] = False
    positions = torch.zeros(tokens, dtype=torch.int32)
    input_ids = torch.zeros(tokens, dtype=torch.int64)
    query_start_loc = torch.zeros(1, dtype=torch.int32)
    ngram_context = torch.empty(0, 2, dtype=torch.int32)
    context_lens = torch.zeros(1, dtype=torch.int32)

    def entry(hidden, valid_tokens):
        return layer.forward(
            hidden,
            positions,
            input_ids,
            query_start_loc,
            ngram_context,
            context_lens,
            valid_tokens,
        )

    run = (
        torch.compile(entry, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else entry
    )
    with torch.inference_mode():
        actual = run(x.to("tpu"), valid.to("tpu"))
        repeated = run(x.to("tpu"), valid.to("tpu"))
    torch.testing.assert_close(actual.cpu(), repeated.cpu(), rtol=0, atol=0)

    def decoder_reference(layer, hidden, valid_tokens):
        hc = hidden
        owners = (
            layer.attn_hyper_connection,
            layer.mlp_hyper_connection,
        )
        blocks = (layer.linear_attn, layer.mlp)
        for index, (owner, block) in enumerate(zip(owners, blocks, strict=True)):
            mixed, norm = _reference_mix(owner, hc)
            if index == 1:
                # Real caller masks the mlp mix output before the block.
                mixed = torch.where(valid_tokens[:, None], mixed, 0)
            block_out = torch.sigmoid(F.linear(mixed, _weight(block.weight)))
            hc = _reference_combine(owner, hc, norm, block_out)
        return hc

    expected = decoder_reference(layer, x, valid)
    assert_metrics(_np(actual), _np(expected))


def terminal_pair_fixture(tokens=3, seed=583, full_tile=128):
    """A combine-owning prior and a mix-only terminal owner as the fusion pair."""
    torch.random.default_generator.manual_seed(seed)
    config = HyperConnectionConfig(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        hc_per_branch_norm=True,
    )
    prior = GatedResidual(config, role="attn", prefill_full_tile=full_tile)
    terminal = GatedResidual(
        config, role="final", use_combine=False, prefill_full_tile=full_tile
    )
    with torch.no_grad():
        for module in (prior, terminal):
            for parameter in module.parameters():
                parameter.copy_((torch.randn_like(parameter) * 0.05).bfloat16())
    x = (torch.randn(tokens, 4 * 2560) * 0.5).bfloat16()
    block = (torch.randn(tokens, 2560) * 0.5).bfloat16()
    return prior, terminal, x, block


def test_mix_only_owner_mix_packed_keeps_public_tuple():
    # The mix-only owner has no combine to feed a packed state, so the single
    # entry point returns exactly the public mix tuple carrier.
    terminal, x, _ = terminal_pair_fixture()[1:]
    packed_mixed, packed_residual = terminal.mix_packed(x)
    public_mixed, public_residual = terminal.mix(x)
    torch.testing.assert_close(packed_mixed, public_mixed, rtol=0, atol=0)
    assert isinstance(packed_residual, tuple) and not isinstance(
        packed_residual, HcPackedMixState
    )
    for first, second in zip(packed_residual, public_residual, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_combine_then_mix_terminal_fallback_returns_unfused_sequence():
    # Host tensors are outside the device-bound admission, so the terminal
    # entry point must reproduce the unfused public calls exactly, including
    # the genuine combined tensor the fallback computes.
    prior, terminal, x, block = terminal_pair_fixture(tokens=6)
    _, residual = prior.mix_packed(x)
    actual = combine_then_mix_terminal(prior, terminal, block, residual)
    expected_combined = prior.combine(block, residual)
    expected = terminal.mix(expected_combined)[0]
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (6, 2560)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combine_then_mix_terminal_rejects_foreign_packed_state():
    prior, terminal, x, block = terminal_pair_fixture(tokens=6)
    other, _, _, _ = terminal_pair_fixture(tokens=6, seed=977)
    norm = _reference_norm(prior, x)
    logits = F.linear(norm, prior.block_inject_weight.weight)
    state = HcPackedMixState(x, logits, other._hc_owner_id)
    with pytest.raises(RuntimeError, match="another hyper-connection"):
        combine_then_mix_terminal(prior, terminal, block, state)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 256, 384])
def test_hc_tpu_terminal_mix_matches_reference(compiled, tokens):
    with torch.device("cpu"):
        prior, terminal, x, block = terminal_pair_fixture(tokens=tokens)
    assert terminal._prefill_terminal_ops
    dispatches = record_fused_dispatch(terminal._prefill_terminal_ops)
    prior = prior.to("tpu")
    terminal = terminal.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")

    def terminal_chain(hyper_input, block_output):
        _, state = prior.mix_packed(hyper_input)
        return combine_then_mix_terminal(prior, terminal, block_output, state)

    # Exact storage words of every read-only operand, captured before the
    # first tested call: float equality would hide signed zeros and BF16
    # payload differences, and post-call snapshots could not observe a call
    # that overwrote its operands.
    readonly = (
        x_tpu,
        block_tpu,
        *(
            parameter
            for module in (prior, terminal)
            for parameter in (
                module.hc_norm.weight,
                module.input_mix_weight_down.weight,
                module.input_mix_weight_up.weight,
            )
        ),
        prior.block_inject_weight.weight,
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]

    run = (
        torch.compile(terminal_chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else terminal_chain
    )
    with torch.inference_mode():
        eager = terminal_chain(x_tpu, block_tpu)
        actual = run(x_tpu, block_tpu)
        repeated = run(x_tpu, block_tpu)
    # Repeat determinism and eager/compiled agreement at raw BF16 word level.
    for first in (eager, repeated):
        np.testing.assert_array_equal(_words(actual.cpu()), _words(first.cpu()))
    norm_a = _reference_norm(prior, x)
    expected_combined = _reference_combine(prior, x, norm_a, block)
    expected_mixed, _ = _reference_mix(terminal, expected_combined)
    assert_metrics(_np(actual), _np(expected_mixed))
    # Routing evidence: every tested run above called the actual registered
    # terminal kernel, and the unfused fallback below must route around it.
    tested_dispatches = len(dispatches)
    assert tested_dispatches > 0
    with torch.inference_mode():
        _, state = prior.mix_packed(x_tpu)
        fallback = terminal.mix(prior.combine(block_tpu, state))[0]
    assert len(dispatches) == tested_dispatches
    assert_metrics(_np(fallback), _np(expected_mixed))
    # The retained inputs and parameters still hold their exact storage words.
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 512])
def test_hc_tpu_terminal_mix_auto_admits_compact_whole_tiles(compiled, tokens):
    """Auto policy dispatches the fused terminal kernel at whole-tile counts.

    The staged tile is the kernel's unit of work, so the production auto
    admission covers the actual runtime token count whenever it is whole
    tiles: 128 tokens dispatch the tile-128 op and 512 the tile-256 op on the
    resource-admitted geometry, with no token floor in between. The fallback
    for ragged counts stays covered by the forced-tile 130-token case.
    """
    with torch.device("cpu"):
        prior, terminal, x, block = terminal_pair_fixture(tokens=tokens, full_tile=None)
    assert terminal.prefill_full_tile == "auto"
    assert terminal._prefill_terminal_ops
    dispatches = record_fused_dispatch(terminal._prefill_terminal_ops)
    prior = prior.to("tpu")
    terminal = terminal.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")

    def terminal_chain(hyper_input, block_output):
        _, state = prior.mix_packed(hyper_input)
        return combine_then_mix_terminal(prior, terminal, block_output, state)

    run = (
        torch.compile(terminal_chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else terminal_chain
    )
    with torch.inference_mode():
        # The packed mix and terminal kernels dispatch at these compact
        # counts; a fallback would consume the public tuple carrier instead.
        _, eager_state = prior.mix_packed(x_tpu)
        assert isinstance(eager_state, HcPackedMixState)
        eager = terminal_chain(x_tpu, block_tpu)
        actual = run(x_tpu, block_tpu)
        repeated = run(x_tpu, block_tpu)
    np.testing.assert_array_equal(_words(actual.cpu()), _words(eager.cpu()))
    np.testing.assert_array_equal(_words(actual.cpu()), _words(repeated.cpu()))
    norm_a = _reference_norm(prior, x)
    expected_combined = _reference_combine(prior, x, norm_a, block)
    expected_mixed, _ = _reference_mix(terminal, expected_combined)
    assert_metrics(_np(actual), _np(expected_mixed))
    # Routing evidence: the auto policy dispatched the actual registered
    # terminal kernel at this whole-tile count, while the unfused fallback
    # sequence routes around it.
    tested_dispatches = len(dispatches)
    assert tested_dispatches > 0
    with torch.inference_mode():
        _, state = prior.mix_packed(x_tpu)
        fallback = terminal.mix(prior.combine(block_tpu, state))[0]
    assert len(dispatches) == tested_dispatches
    assert_metrics(_np(fallback), _np(expected_mixed))


@TPU
def test_hc_tpu_terminal_falls_back_below_admitted_tokens():
    with torch.device("cpu"):
        prior, terminal, x, block = terminal_pair_fixture(tokens=130)
    assert terminal._prefill_terminal_ops
    prior = prior.to("tpu")
    terminal = terminal.to("tpu")
    x_tpu, block_tpu = x.to("tpu"), block.to("tpu")
    # 130 tokens are not a full tile, so the terminal entry point keeps the
    # unfused public sequence over the genuine combined tensor.
    with torch.inference_mode():
        _, residual = prior.mix_packed(x_tpu)
        actual = combine_then_mix_terminal(prior, terminal, block_tpu, residual)
        expected = terminal.mix(prior.combine(block_tpu, residual))[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_backbone_forward_terminal_mix(
    monkeypatch, compiled, backbone_config_context
):
    """The production backbone caller fuses the final combine into the mixer.

    A tiny two-layer backbone over stand-in blocks exercises the actual
    production wiring: only the final layer receives the mix-only terminal
    hyper-connection, its combined output never materializes, and the
    backbone returns the terminal mixed tensor.
    """
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    hidden_size, branches = 2560, 4
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", "128")
    monkeypatch.setattr(
        qwen4_model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        qwen4_model,
        "QwenGatedDeltaNetAttention",
        lambda t, vllm_config, prefix, gqa_interleaved_layout: _StandInBlock(
            11, hidden_size
        ),
    )
    monkeypatch.setattr(
        qwen4_model,
        "Qwen3NextSparseMoeBlock",
        lambda vllm_config, prefix: _StandInBlock(12, hidden_size),
    )
    monkeypatch.setattr(qwen4_model, "VocabParallelEmbedding", nn.Embedding)
    text = q4exp_text_config(
        hidden_size=hidden_size,
        hc_count=branches,
        hc_lowrank=320,
        hc_per_branch_norm=True,
        vocab_size=32,
        num_hidden_layers=2,
        ple_layer_ids=[],
        layer_types=["linear_attention", "linear_attention"],
    )
    cfg = NS(
        model_config=NS(hf_text_config=text, max_model_len=128), additional_config={}
    )
    torch.random.default_generator.manual_seed(95)
    # The production runner creates every parameter in BF16, including the
    # embedding (params_dtype); the stand-in VocabParallelEmbedding must match
    # or the first mix falls back over an FP32 norm against BF16 weights.
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
        backbone = qwen4_model.Qwen4ExpBackbone(cfg)
    # The production caller admits the fused terminal mix on the mix-only
    # backbone mixer, and every intermediate owner keeps its packed ops.
    assert backbone.hyper_connection_mixer._prefill_terminal_ops
    for layer in backbone.layers:
        assert layer.attn_hyper_connection._prefill_inject_ops
        assert layer.mlp_hyper_connection._prefill_boundary_ops
    terminal_dispatches = record_fused_dispatch(
        backbone.hyper_connection_mixer._prefill_terminal_ops
    )
    backbone = backbone.to("tpu")

    torch.random.default_generator.manual_seed(96)
    generator = torch.Generator().manual_seed(96)
    tokens = 256
    input_ids = torch.randint(
        0, text.vocab_size, (tokens,), generator=generator, dtype=torch.int64
    )
    valid = torch.ones(tokens, dtype=torch.bool)
    valid[torch.randperm(tokens, generator=generator)[:9]] = False
    positions = torch.zeros(tokens, dtype=torch.int32)
    query_start_loc = torch.zeros(1, dtype=torch.int32)
    ngram_context = torch.empty(0, 2, dtype=torch.int32)
    context_lens = torch.zeros(1, dtype=torch.int32)

    def entry(ids, valid_tokens):
        return backbone(
            ids,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid_tokens,
        )

    run = (
        torch.compile(entry, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else entry
    )
    # The decorated backbone's __call__ dispatches through vLLM's compile
    # machinery by default; the production eager path is the forward context
    # with skip_compiled, which __call__ itself honors before any AOT hashing.
    # While tracing (compiled=True) dynamo short-circuits on
    # torch.compiler.is_compiling() and never consults this context.
    from vllm.forward_context import set_forward_context

    with (
        torch.inference_mode(),
        set_forward_context(None, backbone_config_context, skip_compiled=True),
    ):
        actual = run(input_ids.to("tpu"), valid.to("tpu"))
        repeated = run(input_ids.to("tpu"), valid.to("tpu"))
    torch.testing.assert_close(actual.cpu(), repeated.cpu(), rtol=0, atol=0)
    assert actual.shape == (tokens, hidden_size)

    def backbone_reference(backbone, ids, valid_tokens):
        embed = backbone.embed_tokens.weight.detach().cpu()
        hc = embed[ids.cpu()].repeat(1, branches)
        for layer in backbone.layers:
            owners = (
                layer.attn_hyper_connection,
                layer.mlp_hyper_connection,
            )
            blocks = (layer.linear_attn, layer.mlp)
            for index, (owner, block) in enumerate(zip(owners, blocks, strict=True)):
                mixed, norm = _reference_mix(owner, hc)
                if index == 1:
                    # Real caller masks the mlp mix output before the block.
                    mixed = torch.where(valid_tokens[:, None], mixed, 0)
                block_out = torch.sigmoid(F.linear(mixed, _weight(block.weight)))
                hc = _reference_combine(owner, hc, norm, block_out)
        return _reference_mix(backbone.hyper_connection_mixer, hc)[0]

    expected = backbone_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))

    def unfused_backbone(ids, valid_tokens):
        # The same production caller with the pre-fusion tail: the final layer
        # returns its genuine combined tensor and the public mixer mix runs
        # outside the layer.
        hidden = backbone.embed_tokens(ids).repeat(1, branches)
        for layer in backbone.layers:
            hidden = layer.forward(
                hidden,
                positions.to(torch.int64),
                ids.to(torch.int64),
                query_start_loc,
                ngram_context,
                context_lens,
                valid_tokens,
                None,
            )
        return backbone.hyper_connection_mixer.mix(hidden)[0]

    # Routing evidence: the production caller dispatched the actual registered
    # terminal kernel for the final layer, while the unfused tail below — the
    # same caller with the pre-fusion sequence — must route around it. The
    # fused kernel may round exactly like the unfused sequence over the same
    # BF16 stores, so output word differences are never dispatch proof here.
    tested_dispatches = len(terminal_dispatches)
    assert tested_dispatches > 0
    with torch.inference_mode():
        fallback = unfused_backbone(input_ids.to("tpu"), valid.to("tpu"))
    assert len(terminal_dispatches) == tested_dispatches
    assert_metrics(_np(fallback), _np(expected))


def _tiny_backbone(monkeypatch, *, num_layers, ple_layer_ids, seed=95):
    """Build the real production backbone over deterministic stand-in blocks.

    Returns ``(backbone, text)`` with every projection in BF16, exactly like
    the production runner: attention and MoE are linear stand-ins, the
    embedding is plain, and PLE (when requested) is a deterministic reader of
    the hidden state. Construction runs under the current vLLM config the
    ``@support_torch_compile`` wrapper resolves.
    """
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    hidden_size, branches = 2560, 4
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", "128")
    monkeypatch.setattr(
        qwen4_model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    # The PLE construction reads the DP group's size for the parallel wrapper
    # even though the stand-in never runs a collective.
    monkeypatch.setattr(
        qwen4_model, "get_dp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        qwen4_model,
        "QwenGatedDeltaNetAttention",
        lambda t, vllm_config, prefix, gqa_interleaved_layout: _StandInBlock(
            11, hidden_size
        ),
    )
    monkeypatch.setattr(
        qwen4_model,
        "Qwen3NextSparseMoeBlock",
        lambda vllm_config, prefix: _StandInBlock(12, hidden_size),
    )
    monkeypatch.setattr(qwen4_model, "VocabParallelEmbedding", nn.Embedding)
    if ple_layer_ids:
        monkeypatch.setattr(
            qwen4_model,
            "Qwen4ExpPooledPLE",
            _stand_in_ple_factory(branches * hidden_size),
        )
    text = q4exp_text_config(
        hidden_size=hidden_size,
        hc_count=branches,
        hc_lowrank=320,
        hc_per_branch_norm=True,
        vocab_size=32,
        num_hidden_layers=num_layers,
        ple_layer_ids=ple_layer_ids,
        layer_types=["linear_attention"] * num_layers,
    )
    cfg = NS(
        model_config=NS(hf_text_config=text, max_model_len=128),
        additional_config={},
        quant_config=None,
    )
    torch.random.default_generator.manual_seed(seed)
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
        backbone = qwen4_model.Qwen4ExpBackbone(cfg)
    return backbone, text


def _stand_in_ple_factory(width):
    """A PLE stand-in that reads and adds to the incoming hidden state.

    The production PLE consumes the combined value before the layer's mix;
    the stand-in keeps exactly that ownership shape (same call signature,
    ``cpu_offload`` flag the backbone checks, and a deterministic BF16
    function of the hidden state) without the full pooling machinery.
    """

    def factory(**kwargs):
        return _StandInPLE(kwargs["prefix"], 13, width)

    return factory


class _StandInPLE(nn.Module):
    def __init__(self, prefix, seed, width):
        super().__init__()
        self.prefix = prefix
        self.cpu_offload = False
        torch.random.default_generator.manual_seed(seed)
        self.weight = nn.Parameter((torch.randn(width) * 0.05).bfloat16())

    def forward_with_history(
        self,
        hidden_states,
        inputs,
        state_indices,
        has_initial_states,
        valid_tokens,
        raw_embeddings=None,
    ):
        return torch.sigmoid(hidden_states.float() * self.weight.float()).to(
            hidden_states.dtype
        )


def _backbone_inputs(text, tokens, seed=96):
    torch.random.default_generator.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, text.vocab_size, (tokens,), generator=generator, dtype=torch.int64
    )
    valid = torch.ones(tokens, dtype=torch.bool)
    valid[torch.randperm(tokens, generator=generator)[:9]] = False
    positions = torch.zeros(tokens, dtype=torch.int32)
    query_start_loc = torch.zeros(1, dtype=torch.int32)
    ngram_context = torch.empty(0, 2, dtype=torch.int32)
    context_lens = torch.zeros(1, dtype=torch.int32)
    return input_ids, valid, positions, query_start_loc, ngram_context, context_lens


def _backbone_hc_parameters(backbone):
    modules = [backbone.hyper_connection_mixer]
    for layer in backbone.layers:
        modules.append(layer.attn_hyper_connection)
        modules.append(layer.mlp_hyper_connection)
    return [
        parameter
        for module in modules
        for parameter in (
            module.hc_norm.weight,
            module.input_mix_weight_down.weight,
            module.input_mix_weight_up.weight,
            *((module.block_inject_weight.weight,) if module.use_combine else ()),
        )
    ]


def _backbone_block_parameters(backbone):
    return [
        parameter
        for layer in backbone.layers
        for parameter in (
            layer.linear_attn.weight,
            layer.mlp.weight,
        )
    ] + [backbone.embed_tokens.weight]


def _pre_cross_layer_wiring(
    backbone, positions, query_start_loc, ngram_context, context_lens
):
    """The pre-cross-layer production caller over the same backbone modules.

    Every layer runs with its default arguments (the public tensor flow),
    keeps the within-layer boundary fusion, and the final layer still fuses
    the terminal mixer: exactly the production wiring before the cross-layer
    fusion, so the only difference from the new backbone wiring is the
    MLP-combine -> next-attention-mix boundary.
    """

    def caller(ids, valid_tokens):
        hidden = backbone.embed_tokens(ids).repeat(1, backbone.hc_count)
        last = len(backbone.layers) - 1
        for index, layer in enumerate(backbone.layers):
            hidden = layer.forward(
                hidden,
                positions.to(torch.int64),
                ids.to(torch.int64),
                query_start_loc,
                ngram_context,
                context_lens,
                valid_tokens,
                None,
                terminal_mixer=(
                    backbone.hyper_connection_mixer if index == last else None
                ),
            )
        return hidden

    return caller


def _cross_layer_reference(backbone, ids, valid_tokens):
    """Independent CPU reference including any PLE stand-in reads."""
    ple_weights = {
        layer.ple.prefix: layer.ple.weight.detach().cpu()
        for layer in backbone.layers
        if layer.ple is not None
    }
    embed = backbone.embed_tokens.weight.detach().cpu()
    hc = embed[ids.cpu()].repeat(1, backbone.hc_count)
    for layer in backbone.layers:
        if layer.ple is not None:
            weight = ple_weights[layer.ple.prefix].float()
            hc = (hc.float() + torch.sigmoid(hc.float() * weight)).bfloat16()
        owners = (
            layer.attn_hyper_connection,
            layer.mlp_hyper_connection,
        )
        blocks = (layer.linear_attn, layer.mlp)
        for index, (owner, block) in enumerate(zip(owners, blocks, strict=True)):
            mixed, norm = _reference_mix(owner, hc)
            if index == 1:
                # Real caller masks the mlp mix output before the block.
                mixed = torch.where(valid_tokens[:, None], mixed, 0)
            block_out = torch.sigmoid(F.linear(mixed, _weight(block.weight)))
            hc = _reference_combine(owner, hc, norm, block_out)
    return _reference_mix(backbone.hyper_connection_mixer, hc)[0]


def test_backbone_cross_layer_mix_matches_public_flow(
    monkeypatch, backbone_config_context
):
    """The cross-layer carrier reproduces the public tensor flow exactly.

    On the host every fused admission falls back, so the new backbone wiring
    and the pre-cross-layer production loop run the identical unfused public
    sequence over the same modules: bit-equal outputs are the strongest form
    of the default-contract check.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[])
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=257)
    )
    from vllm.forward_context import set_forward_context

    old = _pre_cross_layer_wiring(
        backbone, positions, query_start_loc, ngram_context, context_lens
    )

    def new(ids, valid_tokens):
        return backbone(
            ids,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid_tokens,
        )

    with (
        torch.inference_mode(),
        set_forward_context(None, backbone_config_context, skip_compiled=True),
    ):
        expected = old(input_ids, valid)
        actual = new(input_ids, valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_backbone_forward_cross_layer_mix(
    monkeypatch, compiled, backbone_config_context
):
    """The production backbone fuses each MLP combine into the next mix.

    A tiny two-layer backbone over stand-in blocks exercises the actual
    production wiring: the first layer's MLP combine fuses into the second
    layer's attention mix (no PLE intervenes), the second layer consumes the
    cross-layer carrier, and the final layer keeps the terminal fusion. The
    boundary kernel that runs is the following owner's registry, which the
    pre-cross-layer wiring never dispatches.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[])
    assert backbone.hyper_connection_mixer._prefill_terminal_ops
    for layer in backbone.layers:
        assert layer.attn_hyper_connection._prefill_inject_ops
        assert layer.mlp_hyper_connection._prefill_boundary_ops
    # Only the cross-layer boundary dispatches the following attention
    # owner's registry: the within-layer boundaries use the MLP owners.
    cross_layer = record_fused_dispatch(
        backbone.layers[1].attn_hyper_connection._prefill_boundary_ops
    )
    backbone = backbone.to("tpu")
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=256)
    )
    ids_tpu, valid_tpu = input_ids.to("tpu"), valid.to("tpu")
    readonly = (
        input_ids,
        valid,
        *_backbone_hc_parameters(backbone),
        *_backbone_block_parameters(backbone),
    )
    snapshots = [tensor.view(torch.uint16).cpu().numpy().copy() for tensor in readonly]
    # int64/bool inputs carry exact storage in their own dtypes.
    id_snapshot = input_ids.numpy().tobytes()
    valid_snapshot = valid.numpy().tobytes()

    def entry(ids, valid_tokens):
        return backbone(
            ids,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid_tokens,
        )

    run = (
        torch.compile(entry, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else entry
    )
    from vllm.forward_context import set_forward_context

    with (
        torch.inference_mode(),
        set_forward_context(None, backbone_config_context, skip_compiled=True),
    ):
        actual = run(ids_tpu, valid_tpu)
        repeated = run(ids_tpu, valid_tpu)
    torch.testing.assert_close(actual.cpu(), repeated.cpu(), rtol=0, atol=0)
    assert actual.shape == (256, 2560)

    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))

    # Routing evidence: the production caller dispatched the actual registered
    # boundary kernel for the first layer's combine into the second layer's
    # attention mix, while the pre-cross-layer production wiring below — the
    # same modules with the public tensor flow — must route around it.
    tested_dispatches = len(cross_layer)
    assert tested_dispatches > 0
    old = _pre_cross_layer_wiring(
        backbone, positions, query_start_loc, ngram_context, context_lens
    )
    with (
        torch.inference_mode(),
        set_forward_context(None, backbone_config_context, skip_compiled=True),
    ):
        fallback = old(ids_tpu, valid_tpu)
    assert len(cross_layer) == tested_dispatches
    assert_metrics(_np(fallback), _np(expected))
    # The retained inputs and parameters still hold their exact storage.
    for tensor, snapshot in zip(readonly, snapshots, strict=True):
        np.testing.assert_array_equal(tensor.view(torch.uint16).cpu().numpy(), snapshot)
    assert input_ids.numpy().tobytes() == id_snapshot
    assert valid.numpy().tobytes() == valid_snapshot


@TPU
def test_hc_tpu_backbone_ple_boundary_keeps_public_flow(
    monkeypatch, backbone_config_context
):
    """A PLE-owning next layer keeps the public tensor boundary.

    The production PLE reads and modifies the combined value before the next
    layer's mix, so the backbone must not fuse the producing layer's combine
    into that mix: the second layer (PLE at decoder index 1) receives the
    genuine combined tensor, the PLE stand-in consumes it first, and the
    following attention owner's boundary registry is never dispatched.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[2])
    assert backbone.layers[1].ple is not None
    cross_layer = record_fused_dispatch(
        backbone.layers[1].attn_hyper_connection._prefill_boundary_ops
    )
    backbone = backbone.to("tpu")
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=256)
    )
    ids_tpu, valid_tpu = input_ids.to("tpu"), valid.to("tpu")

    def entry(ids, valid_tokens):
        return backbone(
            ids,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid_tokens,
        )

    from vllm.forward_context import set_forward_context

    metadata = {
        backbone.layers[1].ple.prefix: NS(
            mamba_state_indices=torch.empty(0, dtype=torch.int32)
        )
    }
    with (
        torch.inference_mode(),
        set_forward_context(metadata, backbone_config_context, skip_compiled=True),
    ):
        actual = entry(ids_tpu, valid_tpu)
        repeated = entry(ids_tpu, valid_tpu)
    torch.testing.assert_close(actual.cpu(), repeated.cpu(), rtol=0, atol=0)

    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))
    # The PLE read blocks the cross-layer fusion: the following attention
    # owner's boundary kernel never runs over this boundary.
    assert len(cross_layer) == 0


def _run_compiled_entry(run, config, ids, valid):
    """Run an already-built backbone entry callable under the eager context."""
    from vllm.forward_context import set_forward_context

    with torch.inference_mode(), set_forward_context(None, config, skip_compiled=True):
        return run(ids, valid)


def _run_backbone(
    backbone,
    config,
    ids,
    valid,
    positions,
    query_start_loc,
    ngram_context,
    context_lens,
    metadata=None,
):
    """The production backbone entry under the eager skip_compiled context."""
    from vllm.forward_context import set_forward_context

    with (
        torch.inference_mode(),
        set_forward_context(metadata, config, skip_compiled=True),
    ):
        return backbone(
            ids,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid,
        )


def test_backbone_local_hooks_keep_public_layer_contract(
    monkeypatch, backbone_config_context
):
    """Hook-bearing boundaries take the public tensor flow.

    A forward hook on the producing layer reads that layer's return value and
    a pre-hook on the consuming layer reads its arguments; both contracts are
    the public combined tensor, so the backbone must never hand the
    cross-layer carrier across an observed boundary even though the host
    fallback would compute the same values. The hooks fire and observe only
    public tensors, and the output still matches the independent reference.
    """
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    backbone, text = _tiny_backbone(monkeypatch, num_layers=3, ple_layer_ids=[])
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=257)
    )
    producer_outputs = []
    consumer_calls = []

    def producer_hook(module, args, output):
        producer_outputs.append(output)
        return output

    def consumer_hook(module, args, kwargs):
        consumer_calls.append((args, kwargs))
        return None

    producer_handle = backbone.layers[0].register_forward_hook(producer_hook)
    consumer_handle = backbone.layers[1].register_forward_pre_hook(
        consumer_hook, with_kwargs=True
    )
    try:
        actual = _run_backbone(
            backbone,
            backbone_config_context,
            input_ids,
            valid,
            positions,
            query_start_loc,
            ngram_context,
            context_lens,
        )
    finally:
        producer_handle.remove()
        consumer_handle.remove()
    # Both hooks fired; neither saw the internal carrier, only the public
    # combined tensor the pre-fusion production loop passes and returns.
    assert len(producer_outputs) == 1
    assert len(consumer_calls) == 1
    assert isinstance(producer_outputs[0], torch.Tensor)
    assert not isinstance(producer_outputs[0], qwen4_model.HcCrossLayerMix)
    assert producer_outputs[0].shape == (257, 4 * 2560)
    consumer_args, consumer_kwargs = consumer_calls[0]
    assert isinstance(consumer_args[0], torch.Tensor)
    assert consumer_args[0].shape == (257, 4 * 2560)
    assert not any(
        isinstance(value, qwen4_model.HcCrossLayerMix)
        for value in consumer_kwargs.values()
    )
    assert consumer_kwargs["attention_mix"] is None
    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))


def test_backbone_global_module_hook_keeps_public_layer_contract(
    monkeypatch, backbone_config_context
):
    """A global module hook keeps every decoder layer on the public contract.

    ``register_module_forward_hook`` observes every module call, so no
    boundary and no terminal fusion may run behind it: every decoder layer
    returns its genuine combined tensor, the terminal mix runs in the
    backbone, and the output matches the independent reference.
    """
    from torch.nn.modules import module as nn_module_registry

    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[])
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=257)
    )
    observed = []

    def global_hook(module, args, output):
        observed.append((module, output))
        return output

    handle = nn_module_registry.register_module_forward_hook(global_hook)
    try:
        actual = _run_backbone(
            backbone,
            backbone_config_context,
            input_ids,
            valid,
            positions,
            query_start_loc,
            ngram_context,
            context_lens,
        )
    finally:
        handle.remove()
    layer_outputs = [
        output
        for module, output in observed
        if isinstance(module, qwen4_model.Qwen4ExpDecoderLayer)
    ]
    assert len(layer_outputs) == 2
    assert all(
        isinstance(output, torch.Tensor) and output.shape == (257, 4 * 2560)
        for output in layer_outputs
    )
    assert not any(
        isinstance(output, qwen4_model.HcCrossLayerMix) for _, output in observed
    )
    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))


def test_backbone_last_layer_hook_keeps_public_terminal_tensor(
    monkeypatch, backbone_config_context
):
    """An observed final layer returns its public combined tensor.

    The terminal fusion makes the final layer return the terminal mixed
    [tokens, hidden] tensor; a forward hook observing that layer must instead
    read the genuine public combined [tokens, HC*hidden] tensor, with the
    terminal mix running in the backbone exactly like the pre-fusion
    production loop, and the backbone still returns the terminal mixed
    tensor.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[])
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=257)
    )
    final_outputs = []
    handle = backbone.layers[1].register_forward_hook(
        lambda module, args, output: (final_outputs.append(output), output)[1]
    )
    try:
        actual = _run_backbone(
            backbone,
            backbone_config_context,
            input_ids,
            valid,
            positions,
            query_start_loc,
            ngram_context,
            context_lens,
        )
    finally:
        handle.remove()
    assert len(final_outputs) == 1
    assert isinstance(final_outputs[0], torch.Tensor)
    assert final_outputs[0].shape == (257, 4 * 2560)
    assert actual.shape == (257, 2560)
    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))


def test_backbone_ple_layer_hook_keeps_ple_public_contract(
    monkeypatch, backbone_config_context
):
    """An observed PLE layer still reads the combined value before its mix.

    The PLE read is the structural reason the boundary stays public; a hook
    on the PLE-owning layer observes that same public tensor flow, the PLE
    still consumes the genuine combined value, and the output matches the
    independent reference that includes the PLE read.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[2])
    assert backbone.layers[1].ple is not None
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=257)
    )
    ple_outputs = []
    handle = backbone.layers[1].register_forward_hook(
        lambda module, args, output: (ple_outputs.append((args, output)), output)[1]
    )
    metadata = {
        backbone.layers[1].ple.prefix: NS(
            mamba_state_indices=torch.empty(0, dtype=torch.int32)
        )
    }
    try:
        actual = _run_backbone(
            backbone,
            backbone_config_context,
            input_ids,
            valid,
            positions,
            query_start_loc,
            ngram_context,
            context_lens,
            metadata,
        )
    finally:
        handle.remove()
    assert len(ple_outputs) == 1
    args, output = ple_outputs[0]
    # The hook read the genuine public combined tensor in and out.
    assert isinstance(args[0], torch.Tensor)
    assert args[0].shape == (257, 4 * 2560)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (257, 4 * 2560)
    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "hook_layer", [0, 1], ids=["producer-hook", "consumer-last-hook"]
)
def test_hc_tpu_backbone_hook_boundary_keeps_public_flow(
    monkeypatch, hook_layer, compiled, backbone_config_context
):
    """An observed boundary routes around the fused kernels on TPU.

    Eager routing evidence on the real device: a registered forward hook on
    the producing layer keeps the following attention owner's boundary kernel
    from ever dispatching, and a hook on the final layer additionally keeps
    the terminal kernel from dispatching (the backbone runs the public
    terminal mix itself). A producer-side hook leaves the unobserved terminal
    boundary fused. The hook observes the public combined tensor, and the
    output still matches the independent reference.

    Compiled routing evidence: the hook is registered before tracing, so the
    traced wiring takes the public flow at trace time and the compiled graph
    never dispatches the observed boundary's kernel either. Whatever the
    compiled graph replays for the hook on this build, any observation it
    records is the public combined tensor; the routing claim holds in both
    modes.
    """
    backbone, text = _tiny_backbone(monkeypatch, num_layers=2, ple_layer_ids=[])
    assert backbone.hyper_connection_mixer._prefill_terminal_ops
    for layer in backbone.layers:
        assert layer.attn_hyper_connection._prefill_inject_ops
        assert layer.mlp_hyper_connection._prefill_boundary_ops
    cross_layer = record_fused_dispatch(
        backbone.layers[1].attn_hyper_connection._prefill_boundary_ops
    )
    terminal_dispatches = record_fused_dispatch(
        backbone.hyper_connection_mixer._prefill_terminal_ops
    )
    backbone = backbone.to("tpu")
    input_ids, valid, positions, query_start_loc, ngram_context, context_lens = (
        _backbone_inputs(text, tokens=256)
    )
    ids_tpu, valid_tpu = input_ids.to("tpu"), valid.to("tpu")

    def entry(ids_in, valid_in):
        return backbone(
            ids_in,
            positions,
            None,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            context_lens=context_lens,
            valid_tokens=valid_in,
        )

    captured = []
    handle = backbone.layers[hook_layer].register_forward_hook(
        lambda module, args, output: (captured.append(output), output)[1]
    )
    try:
        run = (
            torch.compile(entry, backend="tpu", fullgraph=True, dynamic=False)
            if compiled
            else entry
        )
        actual = _run_compiled_entry(run, backbone_config_context, ids_tpu, valid_tpu)
        repeated = _run_compiled_entry(run, backbone_config_context, ids_tpu, valid_tpu)
    finally:
        handle.remove()
    torch.testing.assert_close(actual.cpu(), repeated.cpu(), rtol=0, atol=0)
    # The observed boundary never dispatches the fused cross-layer kernel.
    assert len(cross_layer) == 0
    if hook_layer == 1:
        # The final layer is observed, so the terminal fusion is off too and
        # the public terminal mix runs in the backbone.
        assert len(terminal_dispatches) == 0
    else:
        # The unobserved final layer still fuses its terminal mix.
        assert len(terminal_dispatches) > 0
    # Every hook observation is the public combined tensor, in whichever mode
    # the hook executes: one per call eagerly, and whatever the compiled
    # graph replays on this build.
    assert all(
        isinstance(output, torch.Tensor) and output.shape == (256, 4 * 2560)
        for output in captured
    )
    if not compiled:
        # The eager hook fires once per entry call (two calls above) and
        # observed the public tensor both times.
        assert len(captured) == 2
    assert actual.shape == (256, 2560)
    expected = _cross_layer_reference(backbone, input_ids, valid)
    assert_metrics(_np(actual), _np(expected))
