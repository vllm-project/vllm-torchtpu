# SPDX-License-Identifier: Apache-2.0
"""Independent PLE hash, checkpoint bytes and request-owned history contracts."""
import os

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.qwen4_exp.ple import (compute_ngram_ids,
                                                 resident_lookup, short_conv)
from vllm_torchtpu.models.vllm.qwen4_exp.ple import ResidentPLEEmbedding

TPU = pytest.mark.skipif(os.environ.get("Q38_TEST_TPU") != "1",
                         reason="Requires serialized TPU window")
EOS = 248044
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
SIZES = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
    20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
    20000161, 20000171
]
OFFSETS = np.cumsum([0] + SIZES[:-1]).tolist()


@pytest.fixture(autouse=True)
def cpu_default():
    with torch.device("cpu"):
        yield


def cpu_reference_ids(tokens,
                      starts,
                      context,
                      multipliers=MULTIPLIERS,
                      sizes=SIZES,
                      offsets=OFFSETS):
    """Python integers, explicit signed wrap, backward EOS walk; no kernel calls."""

    def signed(x):
        return (x + (1 << 63)) % (1 << 64) - (1 << 63)

    result = np.zeros((len(tokens), 16), np.int64)
    for request, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        sequence = list(context[request]) + list(tokens[start:end])
        for position in range(2, len(sequence)):
            previous = []
            stopped = False
            for distance in (1, 2):
                token = int(sequence[position - distance])
                stopped |= token == EOS
                previous.append(EOS if stopped else token)
            words = [int(sequence[position]), *previous]
            for head in range(16):
                mixed = 0
                for token, multiplier in zip(words[:2 + head // 8],
                                             multipliers):
                    mixed = signed(mixed ^ signed(token * multiplier))
                result[start + position - 2,
                       head] = mixed % sizes[head] + offsets[head]
    return torch.from_numpy(result)


def hash_inputs(tokens, starts, context):
    return [
        torch.tensor(x, dtype=torch.int64)
        for x in (tokens, starts, context, MULTIPLIERS, SIZES, OFFSETS)
    ]


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_hash_exact_eos_chunk_boundaries_and_reorder(chunk):
    sequences = [[EOS, EOS, 248319, 0, 9, EOS, 21], [248319, 18, 5, EOS]]
    context = [[EOS, EOS], [31, 47]]
    flat = sum(sequences, [])
    starts = [0, len(sequences[0]), len(flat)]
    expected = cpu_reference_ids(flat, starts, context)
    torch.testing.assert_close(compute_ngram_ids(*hash_inputs(
        flat, starts, context),
                                                 eos_token_id=EOS),
                               expected,
                               rtol=0,
                               atol=0)
    outputs = [[], []]
    for pos in range(0, 7, chunk):
        order = [1, 0] if pos % 2 else [0, 1]
        parts = [sequences[r][pos:pos + chunk] for r in order]
        history = [(context[r] + sequences[r][:pos])[-2:] for r in order]
        packed = sum(parts, [])
        loc = [0, len(parts[0]), len(packed)]
        actual = compute_ngram_ids(*hash_inputs(packed, loc, history),
                                   eos_token_id=EOS)
        for index, request in enumerate(order):
            outputs[request].append(actual[loc[index]:loc[index + 1]])
    torch.testing.assert_close(torch.cat([torch.cat(x) for x in outputs]),
                               expected,
                               rtol=0,
                               atol=0)
    # Stress signed overflow separately from the legal vocabulary contract.
    extreme = [2**62, 2**63 - 1, 248319]
    torch.testing.assert_close(compute_ngram_ids(*hash_inputs(
        extreme, [0, 3], [[EOS, EOS]]),
                                                 eos_token_id=EOS),
                               cpu_reference_ids(extreme, [0, 3],
                                                 [[EOS, EOS]]),
                               rtol=0,
                               atol=0)


def test_hash_empty_and_padded_tokens():
    args = hash_inputs([], [0], [])
    args[2] = args[2].reshape(0, 2)
    assert compute_ngram_ids(*args, eos_token_id=EOS).shape == (0, 16)
    args = hash_inputs([17, 19, 777], [0, 0, 2, 2], [[EOS, EOS]] * 3)
    actual = compute_ngram_ids(*args, eos_token_id=EOS)
    torch.testing.assert_close(actual,
                               cpu_reference_ids([17, 19, 777], [0, 0, 2, 2],
                                                 [[EOS, EOS]] * 3),
                               rtol=0,
                               atol=0)


def shard_weights(rows=48, shards=5):
    bits = torch.arange(rows * 160).remainder(112).to(torch.uint8).reshape(
        rows, 160)
    size = (rows + shards - 1) // shards
    weights = [
        (f"shard_{i}.weight",
         bits[i * size:min((i + 1) * size, rows)].view(torch.float8_e4m3fn))
        for i in range(shards)
    ]
    return bits, weights


@pytest.mark.parametrize("rank", range(4))
def test_loader_shard_intersection_raw_bytes_and_scale(rank):
    bits, weights = shard_weights()
    embedding = ResidentPLEEmbedding(tp_rank=rank,
                                     tp_size=4,
                                     total_rows=48,
                                     checkpoint_shards=5,
                                     device="cpu")
    loaded = embedding.load_weights(reversed(weights))
    assert loaded == {name for name, _ in weights}
    scale = torch.tensor([.00019931793212890625], dtype=torch.bfloat16)
    embedding.load_weights([("weight_scale", scale), ("weight_scale", scale)])
    embedding.finalize_weights()
    assert embedding._staging is None
    assert embedding.table.shape == (15, 128)
    assert embedding.table.dtype == torch.uint8
    torch.testing.assert_close(embedding.table.reshape(12, 160),
                               bits[rank * 12:(rank + 1) * 12],
                               rtol=0,
                               atol=0)
    ids = np.resize(np.array([0, 9, 10, 11, 12, 23, 24, 35, 36, 47]), (9, 16))
    actual = resident_lookup(jnp.asarray(embedding.table.numpy()),
                             jnp.asarray(ids),
                             jnp.asarray(scale.float().numpy()),
                             vocab_start=rank * 12,
                             local_rows=12)
    owned = (ids >= rank * 12) & (ids < (rank + 1) * 12)
    decoded = bits.view(torch.float8_e4m3fn).to(
        torch.bfloat16)[torch.tensor(ids)] * scale
    expected = torch.where(torch.tensor(owned[..., None]), decoded,
                           0).flatten(1)
    np.testing.assert_array_equal(
        np.asarray(actual).astype(np.float32),
        expected.float().numpy())


def test_loader_missing_conflicting_and_invalid_weights_fail():
    _, weights = shard_weights()

    def new():
        return ResidentPLEEmbedding(tp_rank=0,
                                    tp_size=4,
                                    total_rows=48,
                                    checkpoint_shards=5,
                                    device="cpu")

    m = new()
    m.load_weights(weights)
    with pytest.raises(ValueError, match="scale"):
        m.finalize_weights()
    m.load_weights([("weight_scale", torch.tensor([1.],
                                                  dtype=torch.bfloat16))])
    with pytest.raises(ValueError, match="conflicting"):
        m.load_weights([("weight_scale",
                         torch.tensor([2.], dtype=torch.bfloat16))])
    for bad in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="scale"):
            new().load_weights([("weight_scale",
                                 torch.tensor([bad], dtype=torch.bfloat16))])
    m = new()
    m.load_weights(weights[:-1] +
                   [("weight_scale", torch.ones(1, dtype=torch.bfloat16))])
    # Shard 4 is not locally owned; global omission must still fail.
    with pytest.raises(ValueError, match="shard"):
        m.finalize_weights()
    with pytest.raises(ValueError, match="FP8"):
        new().load_weights([(weights[0][0], weights[0][1].float())])


def conv_reference(x, weight, state, starts, slots, initial):
    output = torch.zeros_like(x)
    result = state.clone()
    for r, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if slots[r] < 0 or start == end:
            continue
        history = state[slots[r]] if initial[r] else torch.zeros_like(
            state[slots[r]])
        joined = torch.cat([history, x[start:end]])
        accum = torch.zeros_like(x[start:end], dtype=torch.float32)
        for tap in range(4):
            accum += joined[tap * 3:tap * 3 + end -
                            start].float() * weight[:, tap].float()
        output[start:end] = torch.nn.functional.silu(accum.bfloat16())
        result[slots[r]] = joined[-9:]
    return output, result


def conv_fixture():
    gen = torch.Generator().manual_seed(389)
    return tuple(
        torch.randn(shape, generator=gen).bfloat16()
        for shape in ((19, 128), (128, 4), (5, 9, 128)))


def jax_conv(x, w, state, starts, slots, initial):

    def convert(a):
        return jnp.asarray(a.float().numpy(), dtype=jnp.bfloat16)

    out, history = short_conv(convert(x), convert(w), convert(state),
                              jnp.array(starts), jnp.array(slots),
                              jnp.array(initial))
    return tuple(
        torch.from_numpy(np.asarray(a).astype(np.float32)).bfloat16()
        for a in (out, history))


def test_varlen_conv_mixed_requests_reset_padding_and_chunks():
    x, w, state = conv_fixture()
    starts, slots, initial = [0, 1, 15, 15,
                              18], [3, 1, -1, 4], [True, False, False, True]
    expected = conv_reference(x, w, state, starts, slots, initial)
    actual = jax_conv(x, w, state, starts, slots, initial)
    torch.testing.assert_close(actual[0], expected[0], rtol=.01, atol=.02)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    # Process request 1 in chunks, reorder it with the single-token decode.
    first_x = torch.cat([x[1:4], x[:1]])
    first_y, mid = jax_conv(first_x, w, state, [0, 3, 4], [1, 3],
                            [False, True])
    second_y, final = jax_conv(torch.cat([x[15:18], x[4:15]]), w, mid,
                               [0, 3, 14], [4, 1], [True, True])
    combined = torch.cat([
        first_y[3:], first_y[:3], second_y[3:], second_y[:3],
        torch.zeros_like(x[:1])
    ])
    torch.testing.assert_close(combined, actual[0], rtol=0, atol=0)
    torch.testing.assert_close(final, actual[1], rtol=0, atol=0)
    empty, unchanged = jax_conv(x[:0], w, state, [0], [], [])
    assert empty.shape == (0, 128)
    torch.testing.assert_close(unchanged, state, rtol=0, atol=0)


def values_fixture():
    gen = torch.Generator().manual_seed(390)
    shapes = ((5, 2560), (5, 10240), (10240, 2560), (2560, 2560), (10240, ),
              (10240, ), (10240, ))
    scales = (1., 1., .02, .02, .15, .15, .15)
    values = [(torch.randn(shape, generator=gen) * scale).bfloat16()
              for shape, scale in zip(shapes, scales)]
    values[1][0] = 0
    return values


def values_reference(embedding, query, key_weight, value_weight, nk, nq, nc):
    # CPU source operations with all original BF16 boundaries, independent of
    # JAX kernel/normalization helpers. FP32 linear accumulation then BF16.
    def norm(x, weight):
        x = x.float().reshape(-1, 4, 2560)
        variance = (x * x).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + 1e-6) *
                (1 + weight.float().reshape(4, 2560))).bfloat16()

    key = (embedding.float() @ key_weight.float().T).bfloat16()
    value = (embedding.float() @ value_weight.float().T).bfloat16()
    key, query = norm(key, nk), norm(query, nq)
    score = (key * query).sum(-1, keepdim=True) / np.sqrt(2560)
    gate = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
    gated = gate * value[:, None, :]
    return gated.flatten(1), norm(gated, nc).flatten(1)


def assert_metrics(actual, expected):
    a, b = actual.float().numpy().astype(
        np.float64), expected.float().numpy().astype(np.float64)
    nrmse = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    cosine = np.vdot(a.ravel(), b.ravel()) / max(
        np.linalg.norm(a) * np.linalg.norm(b), 1e-30)
    assert nrmse <= .01, nrmse
    assert cosine >= .999, cosine


def test_full_values_preserve_source_bf16_gate_boundaries():
    from vllm_torchtpu.kernels.qwen4_exp.ple import ple_values
    args = values_fixture()
    expected = values_reference(*args)
    actual = ple_values(
        *[jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16) for x in args])
    for a, b in zip(actual, expected):
        assert_metrics(torch.from_numpy(np.asarray(a).astype(np.float32)), b)


@TPU
def test_hash_tpu_compiled_exact():
    tokens = [EOS, EOS, 248319, 0, 2**62, 2**63 - 1, 17, 777]
    starts = [0, 0, 3, 7, 7]
    context = [[EOS, EOS], [EOS, EOS], [31, 47], [EOS, EOS]]
    args = hash_inputs(tokens, starts, context)

    def run(*args):
        return compute_ngram_ids(*args, eos_token_id=EOS)

    compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(*[x.to("tpu") for x in args]).cpu()
    torch.testing.assert_close(actual,
                               cpu_reference_ids(tokens, starts, context),
                               rtol=0,
                               atol=0)


@TPU
def test_lookup_tpu_compact_chunk8_owner_mask():
    from vllm_torchtpu.kernels.qwen4_exp.ple import build_lookup_op
    bits, _ = shard_weights()
    bits[12] = 127  # NaN row0: a masked non-owner must still return zero.
    scale = torch.tensor([.00019931793212890625], dtype=torch.bfloat16)
    ids = torch.tensor(
        np.resize(np.array([0, 11, 12, 13, 14, 15, 22, 23, 24, 47]), (19, 16)))
    op = torch.compile(build_lookup_op(12, 12),
                       backend="tpu",
                       fullgraph=True,
                       dynamic=False)
    actual = op(bits[12:24].reshape(-1, 128).to("tpu"), ids.to("tpu"),
                scale.to("tpu")).cpu()
    decoded = bits.view(torch.float8_e4m3fn).to(torch.bfloat16)[ids] * scale
    expected = torch.where(((ids >= 12) & (ids < 24))[..., None], decoded,
                           0).flatten(1)
    torch.testing.assert_close(actual,
                               expected,
                               rtol=0,
                               atol=0,
                               equal_nan=True)


@TPU
def test_short_conv_tpu_two_updates_keep_bound_owner():
    from vllm_torchtpu.kernels.qwen4_exp.ple import build_short_conv_op
    x, w, initial = conv_fixture()
    # Use the production width, including history-major physical padding.
    x, w, initial = x.repeat(1, 80), w.repeat(80, 1), initial.repeat(1, 1, 80)
    history = initial.to("tpu")
    owner = history
    op = torch.compile(build_short_conv_op(),
                       backend="tpu",
                       fullgraph=True,
                       dynamic=False)
    for starts, slots, flags in (([0, 1, 15, 15,
                                   18], [3, 1, -1,
                                         4], [True, False, False, True]),
                                 ([0, 3, 4, 18,
                                   18], [4, 3, 1,
                                         -1], [True, True, True, False])):
        expected, initial = conv_reference(x, w, initial, starts, slots, flags)
        args = [
            x, w, history,
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(slots, dtype=torch.int32),
            torch.tensor(flags)
        ]
        actual = op(*[a if a is history else a.to("tpu") for a in args]).cpu()
        assert history is owner
        torch.testing.assert_close(history.cpu(), initial, rtol=0, atol=0)
        assert_metrics(actual, expected)


@TPU
def test_full_values_tpu_source_math_and_normalized_history():
    from vllm_torchtpu.kernels.qwen4_exp.ple import (build_short_conv_op,
                                                     build_values_op)
    args = values_fixture()
    expected = values_reference(*args)
    values = torch.compile(build_values_op(),
                           backend="tpu",
                           fullgraph=True,
                           dynamic=False)
    gated, normalized = values(*[x.to("tpu") for x in args])
    for a, b in zip((gated, normalized), expected):
        assert_metrics(a.cpu(), b)
    gen = torch.Generator().manual_seed(392)
    w = (torch.randn(10240, 4, generator=gen) * .1).bfloat16()
    history = torch.randn(4, 9, 10240, generator=gen).bfloat16()
    starts, slots, flags = [0, 1, 4, 5], [2, 0, 3], [True, False, True]
    cpu_normalized = normalized.cpu()
    expected_conv, expected_history = conv_reference(cpu_normalized, w,
                                                     history, starts, slots,
                                                     flags)
    state = history.to("tpu")
    op = torch.compile(build_short_conv_op(),
                       backend="tpu",
                       fullgraph=True,
                       dynamic=False)
    convolution = op(normalized, w.to("tpu"), state,
                     torch.tensor(starts, dtype=torch.int32).to("tpu"),
                     torch.tensor(slots, dtype=torch.int32).to("tpu"),
                     torch.tensor(flags).to("tpu"))
    torch.testing.assert_close(state.cpu(), expected_history, rtol=0, atol=0)
    assert_metrics((gated + convolution).cpu(), expected[0] + expected_conv)


@TPU
@pytest.mark.skipif(os.environ.get("Q38_FULL_PLE_TABLE") != "1",
                    reason="Allocates the deployment's 6.4GB resident carrier")
def test_full_resident_tpu_carrier_high_addresses():
    from vllm_torchtpu.kernels.qwen4_exp.ple import build_lookup_op
    rows = 40000192
    start = rows * 7
    ids = torch.tensor(
        np.resize(
            np.array([
                start, start + 1, start + 2, start + 3, start + 13421772,
                start + 16777216, start + rows - 2, start + rows - 1, 0,
                start - 1
            ]), (129, 16)))
    # Match production loader's one CPU staging allocation and one H2D.
    # Distinct address sentinels catch wrapped indices; all other rows are zero.
    # Generating this static 6.4GB grid inside a Torch custom op triggers a
    # pathological compile-time initializer fold, unrelated to lookup.
    staging = torch.zeros((rows, 160), dtype=torch.uint8)
    owned_ids = ids[(ids >= start) & (ids < start + rows)].unique()
    columns = torch.arange(160, dtype=torch.int64)
    tags = ((((owned_ids[:, None] >>
               ((columns % 5) * 6)) & 63) + columns * 7) % 112).to(torch.uint8)
    staging[owned_ids - start] = tags
    table = staging.reshape(-1, 128).to("tpu")
    del staging
    assert table.nbytes == 6400030720
    columns = torch.arange(160, dtype=torch.int64)
    bits = ((((ids[..., None] >>
               ((columns % 5) * 6)) & 63) + columns * 7) % 112).to(torch.uint8)
    scale = torch.tensor([.00019931793212890625], dtype=torch.bfloat16)
    expected = bits.view(torch.float8_e4m3fn).bfloat16() * scale
    expected = torch.where(((ids >= start) & (ids < start + rows))[..., None],
                           expected, 0).flatten(1)
    lookup = torch.compile(build_lookup_op(start, rows),
                           backend="tpu",
                           fullgraph=True,
                           dynamic=False)
    actual = lookup(table, ids.to("tpu"), scale.to("tpu")).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_module_checkpoint_names_missing_hash_and_history_binding():
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import Qwen4ExpPLE
    module = Qwen4ExpPLE(tp_rank=0, device="cpu")
    assert module.get_state_shape() == ((9, 10240), )
    assert module.get_state_dtype() == (torch.bfloat16, )
    state = torch.zeros(3, 9, 10240, dtype=torch.bfloat16)
    module.bind_history(state)
    assert module.kv_cache[0] is state
    with pytest.raises(ValueError, match="history"):
        module.bind_history(state.transpose(1, 2))
    weights = [(name, torch.zeros_like(value))
               for name, value in module.named_parameters()]
    assert {name
            for name, _ in weights} == {
                "key_proj.weight", "value_proj.weight", "norm_key.weight",
                "norm_query.weight", "norm_conv.weight", "conv1d.weight"
            }
    assert module.load_weights(weights) == {name for name, _ in weights}
    with pytest.raises(ValueError, match="layer_multipliers"):
        module.finalize_weights()
    hash_weights = [("ple_embedding." + name,
                     torch.tensor(values, dtype=torch.int64))
                    for name, values in (("layer_multipliers", MULTIPLIERS),
                                         ("ngram_heads_vocab_sizes", SIZES),
                                         ("ngram_heads_offsets", OFFSETS))]
    assert module.load_weights(hash_weights) == {
        name
        for name, _ in hash_weights
    }
    for name, expected in hash_weights:
        torch.testing.assert_close(dict(module.named_buffers())[name],
                                   expected,
                                   rtol=0,
                                   atol=0)
    with pytest.raises(ValueError, match="shards"):
        module.finalize_weights()
