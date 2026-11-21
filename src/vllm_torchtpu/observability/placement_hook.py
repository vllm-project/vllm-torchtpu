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
"""Run the placement emitter in the API server's lifespan startup.

vLLM's launcher calls ``build_app`` after every engine is initialised and
before ``serve_http``; the app's lifespan startup is therefore the one point
that is after model load and before the first request. The patch-registry
callback below wraps ``build_app`` wherever the launcher has bound it.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.observability.placement_emitter import publish_placement

logger = init_logger(__name__)

# ``from ..app import build_app`` binds the name in each importer, so every
# module that may be on the call path is wrapped; only one of them runs.
_BUILD_APP_MODULES: tuple[str, ...] = (
    "vllm.entrypoints.launchers.api_server.entry",
    "vllm.entrypoints.launchers.app",
    "vllm.entrypoints.openai.api_server",
)
_HOOK_MARK = "_rank_placement_hook"


def wrap_build_app(build_app: Callable[..., FastAPI]) -> Callable[..., FastAPI]:
    """Wrap ``build_app(args, supported_tasks, model_config)``.

    The hook is installed only on engine-backed apps, which the API server
    builds with a ``model_config``. The render-only server (``vllm launch``)
    builds its app with ``("render",)`` and no model config, and its
    ``init_render_app_state`` sets neither ``engine_client`` nor
    ``vllm_config``, so there is nothing to publish there.
    """
    if getattr(build_app, _HOOK_MARK, False):
        return build_app

    def build_app_with_placement(*args: Any, **kwargs: Any) -> FastAPI:
        app = build_app(*args, **kwargs)
        model_config = kwargs.get("model_config", args[2] if len(args) > 2 else None)
        if model_config is not None:
            install_lifespan_hook(app)
        return app

    setattr(build_app_with_placement, _HOOK_MARK, True)
    build_app_with_placement.__wrapped__ = build_app  # type: ignore[attr-defined]
    return build_app_with_placement


def install_lifespan_hook(app: FastAPI) -> None:
    """Publish at startup, then enter vLLM's own lifespan unchanged.

    ``init_app_state`` populates ``app.state.vllm_config`` between
    ``build_app`` and the server start. Installing twice on one app is a
    no-op, so two wrapped ``build_app`` bindings on one call path still
    publish once.
    """
    inner = app.router.lifespan_context
    if getattr(inner, _HOOK_MARK, False):
        return

    @contextlib.asynccontextmanager
    async def lifespan_with_placement(app_: FastAPI):
        publish_placement(app_.state.vllm_config)
        async with inner(app_):
            yield

    setattr(lifespan_with_placement, _HOOK_MARK, True)
    app.router.lifespan_context = lifespan_with_placement


def _patch_api_server_placement_hook() -> bool | None:
    """Patch-registry callback.

    Returns ``False`` while no launcher module is imported (worker and
    engine-core processes, or activation before the launcher) so the registry
    retries at the next stage instead of marking the patch complete.
    """
    if not envs.VLLM_TPU_EMIT_PLACEMENT:
        return None
    patched = []
    for module_name in _BUILD_APP_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        build_app = getattr(module, "build_app", None)
        if build_app is None or getattr(build_app, _HOOK_MARK, False):
            continue
        setattr(module, "build_app", wrap_build_app(build_app))  # noqa: B010
        patched.append(module_name)
    if not patched:
        return False
    logger.info("Applied TPU patch: placement emitter on %s.", ", ".join(patched))
    return None
