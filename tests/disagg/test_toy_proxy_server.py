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

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx


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


def _post_through_proxy(path: str, body: dict, backend_status: int, backend_body: dict):
    """POST `body` to the proxy's `path` with a fake prefill instance that answers
    `backend_status` / `backend_body`. Returns the proxy's response and the
    requests the prefill instance received."""
    received: list[httpx.Request] = []

    def prefill(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(backend_status, json=backend_body)

    async def run() -> httpx.Response:
        app = _toy_proxy_server.app
        backend = httpx.AsyncClient(
            transport=httpx.MockTransport(prefill), base_url="http://prefill"
        )
        app.state.prefill_clients = [{"client": backend, "id": 0}]
        try:
            # ASGITransport skips the lifespan, which needs the CLI arguments.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy"
            ) as client:
                return await client.post(path, json=body)
        finally:
            await backend.aclose()

    return asyncio.run(run()), received


def test_tokenize_is_forwarded_to_prefill() -> None:
    body = {"model": "m", "prompt": "hello world", "add_special_tokens": False}
    reply = {"count": 2, "max_model_len": 8192, "tokens": [9, 10]}
    response, received = _post_through_proxy("/tokenize", body, 200, reply)

    assert response.status_code == 200
    assert response.json() == reply
    (request,) = received
    assert request.url.path == "/tokenize"
    assert json.loads(request.content) == body


def test_detokenize_is_forwarded_to_prefill() -> None:
    body = {"model": "m", "tokens": [9, 10]}
    response, received = _post_through_proxy(
        "/detokenize", body, 200, {"prompt": "hello world"}
    )

    assert response.status_code == 200
    assert response.json() == {"prompt": "hello world"}
    (request,) = received
    assert request.url.path == "/detokenize"
    assert json.loads(request.content) == body


def test_tokenize_passes_backend_errors_through() -> None:
    error = {"error": {"message": "model not found", "code": 404}}
    response, _ = _post_through_proxy("/tokenize", {"model": "x"}, 404, error)

    assert response.status_code == 404
    assert response.json() == error
