# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_toy_proxy_server = _load_toy_proxy_server()
_AsciiSafeStreamEncoder = _toy_proxy_server._AsciiSafeStreamEncoder
_replace_prompt_with_rendered_token_ids = (
    _toy_proxy_server._replace_prompt_with_rendered_token_ids
)


def test_ascii_safe_stream_encoder_handles_split_utf8() -> None:
    encoder = _AsciiSafeStreamEncoder()
    text = "\u044f\U0001f600"
    message = f'data: {{"choices": [{{"text": "{text}"}}]}}\n\n'
    encoded = message.encode("utf-8")
    split_at = encoded.index(text.encode("utf-8")) + 1

    output = encoder.encode(encoded[:split_at])
    output += encoder.encode(encoded[split_at:])
    output += encoder.encode(final=True)

    assert output.isascii()
    assert b"\\u044f" in output
    assert b"\\ud83d\\ude00" in output

    payload = output.decode("ascii").removeprefix("data: ").strip()
    assert json.loads(payload)["choices"][0]["text"] == text


def test_ascii_safe_stream_encoder_leaves_ascii_unchanged() -> None:
    encoder = _AsciiSafeStreamEncoder()
    message = b'data: {"choices": [{"text": "ok"}]}\n\n'

    assert encoder.encode(message) == message
    assert encoder.encode(final=True) == b""


def test_replace_prompt_with_rendered_token_ids_single_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
        "prompt_embeds": b"unused-after-render",
    }
    rendered = [{"token_ids": [1, 2, 3]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [1, 2, 3]
    assert "prompt_embeds" not in converted
    assert request["prompt"] == "hello"


def test_replace_prompt_with_rendered_token_ids_multi_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": ["hello", "world"],
    }
    rendered = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [[1, 2, 3], [4, 5]]


def test_replace_prompt_with_rendered_token_ids_rejects_bad_response() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
    }

    try:
        _replace_prompt_with_rendered_token_ids(request, [{"token_ids": ["1"]}])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("malformed render response should fail")


def _decode_budgets(n: int, cap: int):
    import asyncio

    cond = asyncio.Condition()
    return [
        _toy_proxy_server._AdmissionBudget(f"decode[{i}]", cap, 0, cond)
        for i in range(n)
    ]


def test_acquire_any_budget_prefers_the_routed_decode_while_it_has_room() -> None:
    import asyncio

    async def run():
        budgets = _decode_budgets(2, 1)
        idx, _ = await _toy_proxy_server._acquire_any_budget(budgets, 1, 100)
        assert idx == 1 and [b.count for b in budgets] == [0, 1]

    asyncio.run(run())


def test_acquire_any_budget_takes_the_decode_that_frees_first() -> None:
    import asyncio

    async def run():
        budgets = _decode_budgets(2, 1)
        # Both decodes busy; the request was routed to decode 0 at arrival.
        for budget in budgets:
            await budget.acquire(100)
        waiter = asyncio.create_task(
            _toy_proxy_server._acquire_any_budget(budgets, 0, 100)
        )
        await asyncio.sleep(0)
        assert not waiter.done() and budgets[0].waiting == 1
        await budgets[1].release(100)
        idx, _ = await asyncio.wait_for(waiter, timeout=1)
        assert idx == 1 and [b.count for b in budgets] == [1, 1]
        assert budgets[0].waiting == 0

    asyncio.run(run())


def test_acquire_any_budget_picks_the_least_loaded_decode_with_room() -> None:
    import asyncio

    async def run():
        budgets = _decode_budgets(3, 2)
        await budgets[0].acquire(100)
        await budgets[0].acquire(100)
        await budgets[1].acquire(100)
        idx, _ = await _toy_proxy_server._acquire_any_budget(budgets, 0, 100)
        assert idx == 2

    asyncio.run(run())


def test_router_rebind_moves_inflight_to_the_admitted_decode() -> None:
    router = _toy_proxy_server._PrefixAffinityRouter(n=2, load_metric="running")
    rank, _ = router.route([b"a", b"b"])
    other = 1 - rank
    router.rebind(rank, other)
    assert router.inflight[rank] == 0 and router.inflight[other] == 1
    assert router.stats()["rebound"] == 1
    router.start_running(other)
    router.done(other, was_running=True)
    assert router.inflight == [0, 0] and router.running == [0, 0]
