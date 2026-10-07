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
"""The server side of a scenario: what runs, on which TPUs, with which flags."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field

from scenario.model.common import (
    Accelerator,
    ArgMap,
    ArgValue,
    ByteSize,
    Duration,
    EnvMap,
    ImageRef,
    Model,
    PosInt,
    RoleArgMap,
    RoleEnvMap,
    RoleName,
    ServedModelName,
    Topology,
)
from scenario.paths import KeyPath

OBJECT_STORAGE_SCHEMES = ("gs://", "s3://")

# What one role runs on, in GKE's terms; on server for one role, else per role.
HARDWARE_FIELDS = ("accelerator", "topology", "hosts", "chips_per_host")
_ACCELERATOR = (
    "TPU type: the cloud.google.com/gke-tpu-accelerator label, e.g. tpu7x "
    "(tpu-v6e-slice for v6e)."
)
_TOPOLOGY = (
    "TPU topology: the cloud.google.com/gke-tpu-topology label, e.g. 2x2x2. "
    "Its chips (the product of its dimensions) must equal hosts x chips_per_host."
)
_HOSTS = "Hosts (pods) of the role; more than one needs server.multihost."
_CHIPS_PER_HOST = "TPU chips per host: the pod's google.com/tpu, e.g. 4."
_ONE_ROLE = " For the single role; with roles, each role sets its own."


class ModelSpec(Model):
    """The model a server loads and the name clients call it by."""

    name: ServedModelName = Field(
        description="What clients call the model; always passed as --served-model-name."
    )
    path: str | None = Field(
        default=None,
        description="What the server loads (--model): a Hugging Face id, a local "
        "path, or a gs:// or s3:// URI. Defaults to name.",
    )

    @property
    def load_path(self) -> str:
        return self.path or self.name

    @property
    def in_object_storage(self) -> bool:
        return self.load_path.startswith(OBJECT_STORAGE_SCHEMES)


class Kv(Model):
    """The KV transfer side of one engine in a disaggregated scenario."""

    connector: Literal["TPURaidenConnector"] = Field(
        description="KV connector; the tool adds its module path and port wiring."
    )
    role: Literal["producer", "consumer", "both"] = Field(
        description="KV role of this engine: kv_producer, kv_consumer or kv_both."
    )


class Role(Model):
    """One engine of a multi-role scenario, e.g. prefill or decode."""

    accelerator: Accelerator = Field(description=_ACCELERATOR)
    topology: Topology = Field(description=_TOPOLOGY)
    hosts: PosInt = Field(description=_HOSTS)
    chips_per_host: PosInt = Field(description=_CHIPS_PER_HOST)
    args: RoleArgMap = Field(
        default_factory=dict,
        description="vllm serve flags over server.args for this role; null "
        "removes an inherited flag.",
    )
    env: RoleEnvMap = Field(
        default_factory=dict,
        description="Env over server.env for this role; null removes an "
        "inherited entry.",
    )
    kv: Kv | None = Field(default=None, description="KV transfer role of this engine.")


class Router(Model):
    """The proxy in front of a multi-role scenario's engines."""

    type: Literal["toy-proxy"] = Field(
        description="Selects the wiring flags the tool adds (hosts and ports of "
        "the producer and consumer roles)."
    )
    command: Annotated[list[str], Field(min_length=1)] = Field(
        description="Argv the wiring flags are appended to, e.g. [python3, "
        "examples/disagg/toy_proxy_server.py]."
    )
    args: ArgMap = Field(
        default_factory=dict,
        description="Extra router flags, rendered like server.args.",
    )


class Server(Model):
    """What runs: image, model, hardware or roles, flags and env."""

    image: ImageRef = Field(
        description="ci for this build's CI image, otherwise a full image URI "
        "with a tag or digest."
    )
    model: ModelSpec = Field(description="Served name and load path.")
    accelerator: Accelerator | None = Field(
        default=None, description=_ACCELERATOR + _ONE_ROLE
    )
    topology: Topology | None = Field(default=None, description=_TOPOLOGY + _ONE_ROLE)
    hosts: PosInt | None = Field(default=None, description=_HOSTS + _ONE_ROLE)
    chips_per_host: PosInt | None = Field(
        default=None, description=_CHIPS_PER_HOST + _ONE_ROLE
    )
    roles: dict[RoleName, Role] | None = Field(
        default=None,
        description="Two or more roles, e.g. prefill and decode, in run order, "
        "each with its own hardware. Not with the server-level hardware fields.",
    )
    multihost: Literal["mp", "ray"] | None = Field(
        default=None,
        description="How a role on several hosts runs one engine: vLLM's own "
        "multi-node DP (mp) or Ray. Required exactly when a role has more "
        "than one host.",
    )
    startup_timeout: Duration = Field(
        description="Engine startup budget: sets VLLM_ENGINE_READY_TIMEOUT_S, "
        "and the readiness wait is this plus a margin."
    )
    args: ArgMap = Field(
        default_factory=dict,
        description="vllm serve flags for every role: true -> --k, false -> "
        "--no-k, a scalar -> --k v, a list or mapping -> --k <compact JSON>.",
    )
    env: EnvMap = Field(
        default_factory=dict, description="Env for every server process of every role."
    )
    router: Router | None = Field(
        default=None, description="Proxy in front of the roles; only with roles."
    )
    scratch: ByteSize | None = Field(
        default=None,
        description="Temporary space every server host needs, e.g. 64Gi; the "
        "target decides where it comes from.",
    )

    def role_views(self) -> tuple["RoleView", ...]:
        """Every role with server.args and server.env applied under its own.

        A scenario without roles has one role named `server`. Empty when
        neither roles nor all four hardware fields are set.
        """
        if self.roles:
            return tuple(
                RoleView(
                    name=name,
                    path=("server", "roles", name),
                    hardware=hardware_of(role),
                    args=combine(self.args, role.args),
                    env=combine(self.env, role.env),
                    kv=role.kv,
                    own_args=frozenset(role.args),
                    own_env=frozenset(role.env),
                )
                for name, role in self.roles.items()
            )
        hardware = hardware_of(self)
        if hardware is not None:
            return (
                RoleView(
                    name="server",
                    path=("server",),
                    hardware=hardware,
                    args=dict(self.args),
                    env=dict(self.env),
                    kv=None,
                    own_args=frozenset(),
                    own_env=frozenset(),
                ),
            )
        return ()


@dataclass(frozen=True)
class Hardware:
    """What one role runs on, as written; the tool looks nothing up."""

    accelerator: str
    topology: str
    hosts: int
    chips_per_host: int

    @property
    def chips(self) -> int:
        return self.hosts * self.chips_per_host


def hardware_of(source: Role | Server) -> Hardware | None:
    """The four hardware fields of a role or server, or None when one is unset."""
    values = {name: getattr(source, name) for name in HARDWARE_FIELDS}
    if None in values.values():
        return None
    return Hardware(**values)


@dataclass(frozen=True)
class RoleView:
    """A role with the server-level args and env applied."""

    name: str
    path: KeyPath
    hardware: Hardware
    args: Mapping[str, ArgValue]
    env: Mapping[str, str]
    kv: Kv | None
    own_args: frozenset[str]
    own_env: frozenset[str]

    def arg_path(self, key: str) -> KeyPath:
        """Where the effective value of arg `key` is written."""
        if key in self.own_args:
            return (*self.path, "args", key)
        return ("server", "args", key)

    def env_path(self, name: str) -> KeyPath:
        if name in self.own_env:
            return (*self.path, "env", name)
        return ("server", "env", name)

    @property
    def args_path(self) -> KeyPath:
        return (*self.path, "args") if self.own_args else ("server", "args")


def combine(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> dict[str, Any]:
    """`upper` over `lower`: mappings merge per key, null removes, others replace."""
    out = dict(lower)
    for key, value in upper.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict):
            below = out.get(key)
            out[key] = combine(below if isinstance(below, dict) else {}, value)
        else:
            out[key] = value
    return out
