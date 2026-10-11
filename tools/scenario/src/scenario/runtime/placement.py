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
"""Where this host runs, and the values that fill the Plan's Refs.

Placement comes only from the SCENARIO_* env a target sets, or from the
equivalent `scenario run` flags, which win. The runtime never reads a pod
IP, a Job completion index or a naming convention of its own, so one Plan
runs the same on VMs and under any target that sets those names.
"""

import json
import os
import pathlib
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from scenario.plan import codec
from scenario.plan.types import CLIENT_ROLE, Concat, EnvItems, Plan, Ref, Value
from scenario.runtime import net
from scenario.runtime.exits import USAGE, RunError
from scenario.target import TargetData

ENV_PLAN = "SCENARIO_PLAN"
ENV_RESOLVED = "SCENARIO_RESOLVED"
ENV_HOST_INDEX = "SCENARIO_HOST_INDEX"
ENV_HEAD_ADDRESS = "SCENARIO_HEAD_ADDRESS"
ENV_ADVERTISE_IP = "SCENARIO_ADVERTISE_IP"
ENV_ROLE_HOSTS = "SCENARIO_ROLE_HOSTS"
ENV_IMAGE = "SCENARIO_IMAGE"
ENV_TARGET_DATA = "SCENARIO_TARGET_DATA"

LOOPBACK = "127.0.0.1"
# Its libtpu cannot be imported; first on a chipless driver's PYTHONPATH.
LIBTPU_BLOCK = pathlib.Path(__file__).resolve().parent / "libtpu_block"
# Read only to be extended: the libtpu block goes in front of it.
ENV_PYTHONPATH = "PYTHONPATH"


@dataclass(frozen=True)
class PlacementFlags:
    """`scenario run` flags; None where a flag was not given."""

    role: str | None = None
    host_index: int | None = None
    head_address: str | None = None
    advertise_ip: str | None = None
    # JSON {role: [hosts]}, or role=a,b;role=c.
    role_hosts: str | None = None


@dataclass(frozen=True)
class Placement:
    """This host: a role of the Plan (or the chipless client) and its index."""

    role: str
    host_index: int
    head_address: str | None = None
    advertise_ip: str | None = None
    # Role -> its hosts' names or addresses, in index order.
    role_hosts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def part(self) -> str:
        return f"{self.role}/{self.host_index}"

    def to_data(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "host_index": self.host_index,
            "head_address": self.head_address,
            "advertise_ip": self.advertise_ip,
            "role_hosts": {role: list(h) for role, h in self.role_hosts.items()},
        }


def parse_role_hosts(text: str) -> dict[str, tuple[str, ...]]:
    """`{"prefill": ["a", "b"]}` or `prefill=a,b;decode=c`.

    Raises:
        RunError: exit 2, the text is neither form.
    """
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError as error:
            raise RunError(USAGE, f"role hosts are not JSON: {error}") from None
        if not isinstance(data, dict) or not all(
            isinstance(hosts, list) and all(isinstance(h, str) for h in hosts)
            for hosts in data.values()
        ):
            raise RunError(USAGE, f"role hosts map roles to lists of hosts: {text}")
        return {str(role): tuple(hosts) for role, hosts in data.items()}
    found: dict[str, tuple[str, ...]] = {}
    for item in text.split(";"):
        role, sep, hosts = item.partition("=")
        if not sep or not role.strip() or not hosts.strip():
            raise RunError(USAGE, f"role hosts are role=host,host;...: {item!r}")
        found[role.strip()] = tuple(h.strip() for h in hosts.split(","))
    return found


def _roles(plan: Plan) -> list[str]:
    names = [role.name for role in plan.roles]
    if plan.driver.chipless:
        names.append(CLIENT_ROLE)
    return names


def _hosts(plan: Plan, role: str) -> int:
    return 1 if role == CLIENT_ROLE else plan.role(role).hosts


def place(plan: Plan, flags: PlacementFlags, env: Mapping[str, str]) -> Placement:
    """This host's placement: flags first, then SCENARIO_* env.

    Raises:
        RunError: exit 2, a missing or bad role, index or role host list.
    """
    roles = _roles(plan)
    role = flags.role
    if role is None:
        if len(roles) > 1:
            raise RunError(USAGE, f"pass --role; roles: {', '.join(roles)}")
        role = roles[0]
    if role not in roles:
        raise RunError(USAGE, f"no role {role}; roles: {', '.join(roles)}")
    index = flags.host_index
    if index is None:
        text = env.get(ENV_HOST_INDEX, "").strip() or "0"
        if not text.isdigit():
            raise RunError(USAGE, f"{ENV_HOST_INDEX} is not a host index: {text!r}")
        index = int(text)
    hosts = _hosts(plan, role)
    if not 0 <= index < hosts:
        raise RunError(
            USAGE, f"role {role} has hosts 0..{hosts - 1}; this host is {index}"
        )
    role_hosts = parse_role_hosts(
        flags.role_hosts
        if flags.role_hosts is not None
        else env.get(ENV_ROLE_HOSTS, "")
    )
    for name, listed in role_hosts.items():
        if name not in roles:
            raise RunError(
                USAGE, f"role hosts name {name}, not a role: {', '.join(roles)}"
            )
        if len(listed) != _hosts(plan, name):
            raise RunError(
                USAGE,
                f"role hosts list {len(listed)} hosts for {name}, which has "
                f"{_hosts(plan, name)}",
            )
    return Placement(
        role=role,
        host_index=index,
        head_address=flags.head_address or env.get(ENV_HEAD_ADDRESS) or None,
        advertise_ip=flags.advertise_ip or env.get(ENV_ADVERTISE_IP) or None,
        role_hosts=role_hosts,
    )


def decode_plan(text: str) -> Plan:
    """The Plan in a SCENARIO_PLAN value.

    Raises:
        RunError: exit 2, not an encoded Plan of this tool's format.
    """
    try:
        return codec.decode(text)
    except codec.PlanFormatError as error:
        raise RunError(USAGE, f"{ENV_PLAN}: {error}") from None


def decode_target_data(text: str) -> TargetData:
    """The TargetData in a SCENARIO_TARGET_DATA value.

    Raises:
        RunError: exit 2, the value has another shape.
    """
    try:
        return TargetData.from_data(codec.decode_data(text.strip()))
    except (codec.PlanFormatError, ValueError) as error:
        raise RunError(USAGE, f"{ENV_TARGET_DATA}: {error}") from None


def decode_resolved(text: str) -> dict[str, Any]:
    """The resolved scenario in a SCENARIO_RESOLVED value: {point, scenario}.

    Raises:
        RunError: exit 2, the value has another shape.
    """
    try:
        data = codec.decode_data(text.strip())
    except codec.PlanFormatError as error:
        raise RunError(USAGE, f"{ENV_RESOLVED}: {error}") from None
    if not isinstance(data, dict) or not isinstance(data.get("scenario"), dict):
        raise RunError(USAGE, f"{ENV_RESOLVED} holds no resolved scenario")
    return data


# Where vLLM's runai loader pulls an object-storage model; computed by the
# installed vLLM so the client reads exactly the directory the server fills.
PULLED_DIR_SNIPPET = (
    "import sys; from vllm.transformers_utils.runai_utils import "
    "ObjectStorageModel; print(ObjectStorageModel(url=sys.argv[1]).dir)"
)


def pulled_model_dir(model_path: str, env: Mapping[str, str]) -> str | None:
    """The directory the server pulls `model_path` into, or None when unknown."""
    try:
        done = subprocess.run(
            ["python3", "-c", PULLED_DIR_SNIPPET, model_path],
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = done.stdout.strip().splitlines()
    return lines[-1] if done.returncode == 0 and lines else None


Resolve = Callable[[str], str]


class Refs:
    """Fills the Plan's Refs for one host.

    Each value is computed when first needed, so a host whose commands use
    no Ref of a kind needs none of its inputs. Names resolve with a bounded
    retry, and a failure is a bring-up failure (exit 3); a missing input is
    a placement error (exit 2).
    """

    def __init__(
        self,
        plan: Plan,
        placement: Placement,
        *,
        run_dir: pathlib.Path,
        server_log: pathlib.Path,
        env: Mapping[str, str],
        resolve: Resolve = net.resolve_ipv4,
        pulled_dir: Callable[[], str | None] | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self.plan = plan
        self.placement = placement
        self.run_dir = run_dir
        self.server_log = server_log
        self._resolve = resolve
        self._image = env.get(ENV_IMAGE) or plan.image or ""
        self._pythonpath = env.get(ENV_PYTHONPATH, "")
        self._pulled_dir = pulled_dir or (
            lambda: pulled_model_dir(plan.model_path, env)
        )
        self._warn = warn or (lambda message: None)
        self._pulled: str | None = None
        self._pulled_known = False
        self._resolved: dict[str, str] = {}

    def value(self, value: Value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Concat):
            return "".join(self.value(part) for part in value.parts)
        return self.ref(value)

    def argv(self, argv: Iterable[Value]) -> list[str]:
        return [self.value(item) for item in argv]

    def env(self, items: EnvItems) -> dict[str, str]:
        return {name: self.value(value) for name, value in items}

    def ref(self, ref: Ref) -> str:
        at = self.placement
        if ref.kind == "self_ip":
            return self._self_ip()
        if ref.kind == "head_ip":
            return self._head_ip()
        if ref.kind == "host_index":
            return str(at.host_index)
        if ref.kind == "role_head":
            role = self._role_of(ref)
            if role == at.role and at.host_index == 0:
                return LOOPBACK
            return self._role_hosts(role)[0]
        if ref.kind == "role_ips":
            role = self._role_of(ref)
            if (
                role == at.role
                and role not in at.role_hosts
                and _hosts(self.plan, role) == 1
            ):
                return self._self_ip()
            ips = [
                self._self_ip()
                if role == at.role and index == at.host_index
                else self._ipv4(host)
                for index, host in enumerate(self._role_hosts(role))
            ]
            return ref.sep.join(ips)
        if ref.kind == "results_dir":
            return str(self.run_dir.resolve())
        if ref.kind == "server_log":
            return str(self.server_log.resolve())
        if ref.kind == "pulled_model_dir":
            return self._pulled_model_dir(ref)
        if ref.kind == "image":
            return self._image
        if ref.kind == "libtpu_block":
            parts = [str(LIBTPU_BLOCK), self._pythonpath]
            return os.pathsep.join(part for part in parts if part)
        raise RunError(USAGE, f"this runtime cannot fill a {ref.kind} Ref")

    def _head_ip(self) -> str:
        """The role's first host as IPv4: on that host, its own address.

        Elsewhere the head address, or the role's first listed host, resolved.
        """
        at = self.placement
        if at.host_index == 0 and at.advertise_ip is not None:
            return at.advertise_ip
        listed = at.role_hosts.get(at.role)
        name = at.head_address or (listed[0] if listed else None)
        if name is None:
            raise RunError(
                USAGE,
                f"{at.part} needs its role's head address: set "
                f"{ENV_HEAD_ADDRESS} or pass --head-address",
            )
        return self._ipv4(name)

    def _role_of(self, ref: Ref) -> str:
        if ref.role is None:
            raise RunError(USAGE, f"a {ref.kind} Ref names no role")
        return ref.role

    def _self_ip(self) -> str:
        if self.placement.advertise_ip is None:
            raise RunError(
                USAGE,
                f"{self.placement.part} needs its own address: set "
                f"{ENV_ADVERTISE_IP} or pass --advertise-ip",
            )
        return self.placement.advertise_ip

    def _role_hosts(self, role: str) -> tuple[str, ...]:
        hosts = self.placement.role_hosts.get(role)
        if not hosts:
            raise RunError(
                USAGE,
                f"{self.placement.part} needs the hosts of role {role}: set "
                f"{ENV_ROLE_HOSTS} or pass --role-hosts",
            )
        return hosts

    def _ipv4(self, name: str) -> str:
        if name not in self._resolved:
            self._resolved[name] = self._resolve(name)
        return self._resolved[name]

    def _pulled_model_dir(self, ref: Ref) -> str:
        """The pulled directory once the server has filled it, else the fallback.

        The directory is computed once; whether it holds config.json is
        checked at every use, since the server fills it while starting.
        """
        if not self._pulled_known:
            self._pulled = self._pulled_dir()
            self._pulled_known = True
        pulled = self._pulled
        if pulled is not None and pathlib.Path(pulled, "config.json").is_file():
            return pulled
        if ref.fallback is None:
            if pulled is None:
                raise RunError(
                    USAGE, f"cannot tell where {self.plan.model_path} is pulled"
                )
            return pulled
        self._warn(
            f"WARNING: pulled dir {pulled} for {self.plan.model_path} has no "
            f"config.json; the client resolves {ref.fallback} instead"
        )
        return ref.fallback
