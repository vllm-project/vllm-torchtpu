#!/usr/bin/env python3
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
"""Endpoint URL normalization and `/v1/models` lookup for benchmark scripts."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse


def _invalid_base_url(raw: str, hint: str = "") -> ValueError:
    # A value that may hold credentials (user:password@host) is not echoed:
    # the message goes to logs.
    shown = "<not shown: it contains '@'>" if "@" in raw else raw
    return ValueError(f"Invalid --base-url: {shown}{hint}")


def normalize_base_url(raw: str) -> str:
    """Validate and normalize an OpenAI-compatible API root URL.

    Accepts a bare host (defaulting to ``http://<host>:8000``) or an
    ``http://`` / ``https://`` URL with an optional port and path prefix.
    Strips a trailing ``/`` and a trailing ``/v1`` from the path only, so
    clients can append their own ``/v1/...`` routes without losing a host
    named ``v1`` or doubling an endpoint route.
    """
    text = raw.strip()
    if not text or any(ch.isspace() for ch in text):
        raise _invalid_base_url(raw)

    bare = "://" not in text
    try:
        url = urlparse("http://" + text if bare else text)
        port = url.port
    except ValueError as err:
        raise _invalid_base_url(raw) from err

    # Credentials in the URL would be echoed into logs with every command.
    if url.username is not None or url.password is not None:
        raise _invalid_base_url(
            raw, " (credentials in the URL are not supported; set OPENAI_API_KEY)"
        )
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.query
        or url.fragment
    ):
        raise _invalid_base_url(raw)

    path = url.path.rstrip("/")
    if path == "/v1" or path.endswith("/v1"):
        path = path[: -len("/v1")].rstrip("/")
    if "/v1/" in f"{path}/":
        raise _invalid_base_url(raw, " (pass the API root, not a /v1/... route)")

    netloc = url.netloc
    # "host:" has no port to keep, and "http:/host" parses as the host "http".
    if netloc.endswith(":"):
        raise _invalid_base_url(raw)
    if bare and port is None:
        netloc = f"{netloc}:8000"
    return f"{url.scheme}://{netloc}{path}"


class ModelLookupError(RuntimeError):
    """``/v1/models`` could not be read, so the served models are unknown."""


class ModelNotServedError(RuntimeError):
    """The model name a run would send is not one the endpoint lists."""


def list_served_models(
    base_url: str, *, api_key: str | None = None, timeout: float = 30.0
) -> list[str]:
    """Return the model ids advertised at ``<base_url>/v1/models``.

    Raises:
        ModelLookupError: The endpoint is unreachable, answers with an HTTP
            error (failed authentication included), or does not answer with
            a JSON model list.
    """
    models_url = f"{base_url}/v1/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.load(resp)
    # URLError covers HTTP errors, OSError timeouts and resets, HTTPException
    # a broken HTTP reply, and ValueError a body that is not JSON.
    except (
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        ValueError,
    ) as err:
        raise ModelLookupError(
            f"Could not list the served models at {models_url}: {err}"
        ) from err
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ModelLookupError(
            f"Could not list the served models at {models_url}: "
            "the response is not a JSON model list"
        )
    return [
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def choose_served_model(
    listed: list[str],
    *,
    config_model: str,
    served_model_name: str | None = None,
) -> str:
    """Return the model name to send to an endpoint that lists ``listed``.

    The name is ``served_model_name`` when given, otherwise ``config_model``.

    Raises:
        ModelNotServedError: The chosen name is not in ``listed``. Every
            request for it would fail, and ``vllm bench serve`` would still
            exit 0 with a result file of zeros.
    """
    if served_model_name:
        name, source = served_model_name, "--served-model-name"
    else:
        name, source = config_model, "the config's MODEL"
    if name not in listed:
        served = ", ".join(listed) if listed else "no models"
        raise ModelNotServedError(
            f"{source} '{name}' is not a model the endpoint serves (it lists: {served})"
        )
    return name


def _resolve_model(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="endpoint_url.py resolve-model")
    parser.add_argument("base_url")
    parser.add_argument("--config-model", required=True)
    parser.add_argument("--served-model-name")
    parsed = parser.parse_args(args)
    try:
        listed = list_served_models(
            parsed.base_url, api_key=os.environ.get("OPENAI_API_KEY") or None
        )
        name = choose_served_model(
            listed,
            config_model=parsed.config_model,
            served_model_name=parsed.served_model_name,
        )
    except ModelLookupError as err:
        print(
            f"ERROR: {err}. The run sends no request before it has checked "
            "the model name against this list.",
            file=sys.stderr,
        )
        return 1
    except ModelNotServedError as err:
        print(
            f"ERROR: {parsed.base_url}/v1/models: {err}. Pass "
            "--served-model-name with a name the endpoint serves.",
            file=sys.stderr,
        )
        return 1
    print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["resolve-model"]:
        return _resolve_model(args[1:])
    if len(args) != 2 or args[0] != "normalize":
        print(
            "Usage: endpoint_url.py normalize <url>\n"
            "       endpoint_url.py resolve-model <url> --config-model MODEL "
            "[--served-model-name NAME]",
            file=sys.stderr,
        )
        return 2
    try:
        print(normalize_base_url(args[1]))
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
