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
"""Placement from flags and SCENARIO_* env, Plan decoding, and Ref filling (R-RUN-2)."""

import ast
import json
import os
import pathlib
import re
import sys
from typing import Any

import pytest
from plan_fixtures import Data, build
from scenario.plan import codec
from scenario.plan.types import Concat, Plan, Ref
from scenario.runtime import exits, placement
from scenario.runtime.exits import RunError
from scenario.runtime.placement import Placement, PlacementFlags, Refs, place
from scenario.target import AfterTask, TargetData, TargetInfo

RUNTIME = pathlib.Path(__file__).resolve().parents[1] / "src" / "scenario" / "runtime"


@pytest.fixture
def k3_pd(variants: dict[str, Data]) -> Plan:
    return build(variants["k3_pd"])


@pytest.fixture
def single_host(variants: dict[str, Data]) -> Plan:
    return build(variants["single_host"])


def usage_error(call: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(RunError) as raised:
        call(*args, **kwargs)
    assert raised.value.code == exits.USAGE
    return raised.value.message


def test_one_role_needs_no_flags(single_host: Plan) -> None:
    at = place(single_host, PlacementFlags(), {})
    assert at == Placement(role="server", host_index=0)
    assert at.part == "server/0"


def test_flags_win_over_env(k3_pd: Plan) -> None:
    env = {
        "SCENARIO_HOST_INDEX": "2",
        "SCENARIO_HEAD_ADDRESS": "wl-decode-0-0.wl",
        "SCENARIO_ADVERTISE_IP": "10.0.0.9",
        "SCENARIO_ROLE_HOSTS": json.dumps(
            {"prefill": ["p0", "p1", "p2", "p3"], "decode": ["d0", "d1", "d2", "d3"]}
        ),
    }
    from_env = place(k3_pd, PlacementFlags(role="decode"), env)
    assert from_env.host_index == 2
    assert from_env.head_address == "wl-decode-0-0.wl"
    assert from_env.advertise_ip == "10.0.0.9"
    assert from_env.role_hosts["decode"] == ("d0", "d1", "d2", "d3")
    flags = PlacementFlags(
        role="decode",
        host_index=1,
        head_address="10.0.0.1",
        advertise_ip="10.0.0.2",
        role_hosts="prefill=a,b,c,d;decode=e,f,g,h",
    )
    from_flags = place(k3_pd, flags, env)
    assert (from_flags.host_index, from_flags.head_address) == (1, "10.0.0.1")
    assert from_flags.advertise_ip == "10.0.0.2"
    assert from_flags.role_hosts["prefill"] == ("a", "b", "c", "d")


@pytest.mark.parametrize(
    ("flags", "env", "message"),
    [
        (PlacementFlags(), {}, "pass --role; roles: prefill, decode"),
        (PlacementFlags(role="client"), {}, "no role client; roles: prefill, decode"),
        (PlacementFlags(role="decode", host_index=4), {}, "has hosts 0..3"),
        (
            PlacementFlags(role="decode"),
            {"SCENARIO_HOST_INDEX": "x"},
            "SCENARIO_HOST_INDEX is not a host index",
        ),
        (
            PlacementFlags(role="decode", role_hosts="decode=a,b"),
            {},
            "list 2 hosts for decode, which has 4",
        ),
        (
            PlacementFlags(role="decode", role_hosts="router=a"),
            {},
            "name router, not a role",
        ),
        (PlacementFlags(role="decode", role_hosts="decode"), {}, "role=host,host"),
        (PlacementFlags(role="decode", role_hosts="{nope"), {}, "not JSON"),
    ],
)
def test_bad_placement(
    k3_pd: Plan, flags: PlacementFlags, env: dict[str, str], message: str
) -> None:
    assert message in usage_error(place, k3_pd, flags, env)


def test_a_chipless_driver_is_the_client_role(variants: dict[str, Data]) -> None:
    pd = build(variants["pd"])
    assert place(pd, PlacementFlags(role="client"), {}).role == "client"


def test_plans_and_target_data_decode(single_host: Plan) -> None:
    assert placement.decode_plan(codec.encode(single_host)) == single_host
    assert "not an encoded Plan" in usage_error(placement.decode_plan, "nope")
    data = codec.to_data(single_host) | {"format": 99}
    assert "plan format 99" in usage_error(
        placement.decode_plan, codec.encode_data(data)
    )
    target = TargetData(
        TargetInfo("fleet", "abc123", "results"),
        (
            AfterTask("upload", ("up", "{file}"), when="result"),
            AfterTask(
                "rows",
                ("rows", "{run_dir}"),
                env=(("RUN_TYPE", "DAILY"),),
                require="commands_ok",
                fatal=False,
                skip_on_experiment=True,
                timeout_s=60,
            ),
        ),
        lm_eval_cache_root="/cache/lm_eval",
    )
    encoded = codec.encode_data(target.to_data())
    assert placement.decode_target_data(encoded) == target
    resolved = {"point": None, "scenario": {"name": "x"}}
    assert placement.decode_resolved(codec.encode_data(resolved)) == resolved
    assert "no resolved scenario" in usage_error(
        placement.decode_resolved, codec.encode_data([1])
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "target data is a mapping"),
        ({"info": {}}, "info needs a name"),
        ({"info": {"name": "t"}, "extra": 1}, "field extra is unknown"),
        ({"info": {"name": "t"}, "results_root": 3}, "results_root is a string"),
        ({"info": {"name": "t"}, "after_tasks": [{"argv": []}]}, "without name"),
        (
            {"info": {"name": "t"}, "after_tasks": [{"name": "a", "argv": [1]}]},
            "holds a non-string",
        ),
        (
            {
                "info": {"name": "t"},
                "after_tasks": [{"name": "a", "argv": [], "when": "later"}],
            },
            "when 'later' is not one of result, end",
        ),
        (
            {
                "info": {"name": "t"},
                "after_tasks": [{"name": "a", "argv": [], "timeout_s": True}],
            },
            "timeout_s True is not a int",
        ),
        (
            {
                "info": {"name": "t"},
                "after_tasks": [{"name": "a", "argv": [], "env": [["A"]]}],
            },
            "env entries are [name, value]",
        ),
    ],
)
def test_bad_target_data(data: Any, message: str) -> None:
    text = codec.encode_data(data)
    assert message in usage_error(placement.decode_target_data, text)


class Resolver:
    def __init__(self, table: dict[str, str]) -> None:
        self.table = table
        self.calls: list[str] = []

    def __call__(self, name: str) -> str:
        self.calls.append(name)
        if name not in self.table:
            raise RunError(exits.BRINGUP, f"{name} did not resolve")
        return self.table[name]


HOSTS = {
    "prefill": ("p0.wl", "p1.wl", "p2.wl", "p3.wl"),
    "decode": ("d0.wl", "d1.wl", "d2.wl", "d3.wl"),
}
ADDRESSES = {
    name: f"10.0.{i}.{j}"
    for i, role in enumerate(HOSTS.values())
    for j, name in enumerate(role)
}


def refs_for(
    plan: Plan,
    at: Placement,
    tmp_path: pathlib.Path,
    pulled: str | None = None,
    warnings: list[str] | None = None,
) -> tuple[Refs, Resolver]:
    resolver = Resolver(ADDRESSES)
    refs = Refs(
        plan,
        at,
        run_dir=tmp_path / "run",
        server_log=tmp_path / "run" / "server.log",
        env={},
        resolve=resolver,
        pulled_dir=lambda: pulled,
        warn=(warnings if warnings is not None else []).append,
    )
    return refs, resolver


def test_every_ref_kind(k3_pd: Plan, tmp_path: pathlib.Path) -> None:
    at = Placement("decode", 2, "d0.wl", "10.0.1.2", HOSTS)
    refs, resolver = refs_for(k3_pd, at, tmp_path)
    assert refs.ref(Ref("self_ip")) == "10.0.1.2"
    assert refs.ref(Ref("head_ip")) == "10.0.1.0"
    assert refs.ref(Ref("host_index")) == "2"
    assert refs.ref(Ref("role_head", role="prefill")) == "p0.wl"
    assert refs.ref(Ref("role_head", role="decode")) == "d0.wl"
    # Its own entry is its own address, never a lookup of its own name.
    assert (
        refs.ref(Ref("role_ips", role="decode"))
        == "10.0.1.0,10.0.1.1,10.0.1.2,10.0.1.3"
    )
    assert (
        refs.ref(Ref("role_ips", role="prefill", sep=" "))
        == "10.0.0.0 10.0.0.1 10.0.0.2 10.0.0.3"
    )
    assert "d2.wl" not in resolver.calls
    assert resolver.calls.count("d0.wl") == 1
    assert refs.ref(Ref("results_dir")) == str((tmp_path / "run").resolve())
    assert refs.ref(Ref("server_log")) == str(
        (tmp_path / "run" / "server.log").resolve()
    )
    concat = Concat(("http://", Ref("role_head", role="prefill"), ":8000"))
    assert refs.value(concat) == "http://p0.wl:8000"


def test_the_head_is_loopback_to_itself(k3_pd: Plan, tmp_path: pathlib.Path) -> None:
    refs, _ = refs_for(k3_pd, Placement("prefill", 0, None, None, {}), tmp_path)
    assert refs.ref(Ref("role_head", role="prefill")) == "127.0.0.1"


def test_a_single_host_role_lists_itself(
    variants: dict[str, Data], tmp_path: pathlib.Path
) -> None:
    pd = build(variants["pd"])
    refs, resolver = refs_for(
        pd, Placement("prefill", 0, None, "10.9.9.9", {}), tmp_path
    )
    assert refs.ref(Ref("role_ips", role="prefill")) == "10.9.9.9"
    assert resolver.calls == []


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        (Ref("self_ip"), "set SCENARIO_ADVERTISE_IP or pass --advertise-ip"),
        (Ref("head_ip"), "set SCENARIO_HEAD_ADDRESS or pass --head-address"),
        (
            Ref("role_head", role="prefill"),
            "set SCENARIO_ROLE_HOSTS or pass --role-hosts",
        ),
        (Ref("role_ips", role="prefill"), "needs the hosts of role prefill"),
        (Ref("role_head"), "names no role"),
    ],
)
def test_missing_inputs_are_placement_errors(
    k3_pd: Plan, tmp_path: pathlib.Path, ref: Ref, message: str
) -> None:
    refs, _ = refs_for(k3_pd, Placement("decode", 1), tmp_path)
    assert message in usage_error(refs.ref, ref)


def test_an_unresolved_name_is_a_bring_up_failure(
    k3_pd: Plan, tmp_path: pathlib.Path
) -> None:
    refs, _ = refs_for(
        k3_pd, Placement("decode", 1, "missing.wl", "10.0.1.1"), tmp_path
    )
    with pytest.raises(RunError) as raised:
        refs.ref(Ref("head_ip"))
    assert raised.value.code == exits.BRINGUP


def test_the_pulled_dir_counts_once_it_holds_a_config(
    single_host: Plan, tmp_path: pathlib.Path
) -> None:
    pulled = tmp_path / "pulled"
    pulled.mkdir()
    calls: list[int] = []
    warnings: list[str] = []

    def lookup() -> str:
        calls.append(1)
        return str(pulled)

    refs = Refs(
        single_host,
        Placement("server", 0),
        run_dir=tmp_path,
        server_log=tmp_path / "server.log",
        env={},
        pulled_dir=lookup,
        warn=warnings.append,
    )
    tokenizer = Ref("pulled_model_dir", fallback="zai-org/GLM-5.2-FP8")
    assert refs.ref(tokenizer) == "zai-org/GLM-5.2-FP8"
    assert "has no config.json" in warnings[0]
    (pulled / "config.json").write_text("{}")
    assert refs.ref(tokenizer) == str(pulled)
    assert refs.ref(Ref("pulled_model_dir")) == str(pulled)
    assert calls == [1]


def test_the_pulled_dir_comes_from_vllm(tmp_path: pathlib.Path) -> None:
    package = tmp_path / "site" / "vllm" / "transformers_utils"
    package.mkdir(parents=True)
    for directory in (package.parent, package):
        (directory / "__init__.py").write_text("")
    (package / "runai_utils.py").write_text(
        "class ObjectStorageModel:\n"
        "    def __init__(self, url):\n"
        "        self.dir = '/tmp/pulled/' + url.rsplit('/', 1)[-1]\n"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site")}
    assert (
        placement.pulled_model_dir("gs://bucket/model-x", env) == "/tmp/pulled/model-x"
    )
    bare = {**os.environ, "PYTHONPATH": str(tmp_path / "nothing")}
    if sys.modules.get("vllm") is None:
        assert placement.pulled_model_dir("gs://bucket/m", bare) is None


def test_every_command_of_a_host_fills(k3_pd: Plan, tmp_path: pathlib.Path) -> None:
    for role in k3_pd.roles:
        for host in role.host_plans:
            at = Placement(
                role.name,
                host.index,
                HOSTS[role.name][0],
                ADDRESSES[HOSTS[role.name][host.index]],
                HOSTS,
            )
            refs, _ = refs_for(k3_pd, at, tmp_path, pulled=str(tmp_path))
            for command in (*host.pre, *([host.serve] if host.serve else [])):
                assert all(isinstance(word, str) for word in refs.argv(command.argv))
            filled = refs.env(host.tool_env)
            assert filled["VLLM_HOST_IP"] == at.advertise_ip
    driver = Placement("prefill", 0, "p0.wl", "10.0.0.0", HOSTS)
    refs, _ = refs_for(k3_pd, driver, tmp_path, pulled=str(tmp_path))
    for task in k3_pd.driver.tasks:
        env = refs.env(task.command.env)  # type: ignore[union-attr]
        assert env["DECODE_URL"] == "http://d0.wl:8000"
        assert env["PREFILL_URL"] == "http://127.0.0.1:8000"


# R-RUN-2: placement comes from SCENARIO_* only.
BANNED = (
    "POD_IP",
    "JOB_COMPLETION_INDEX",
    "job-completion-index",
    "cluster.local",
    "gethostname",
    "getfqdn",
    # The base env comes in as a parameter, so tests and targets choose it.
    "os.environ",
    "getenv",
)


def runtime_sources() -> list[pathlib.Path]:
    return sorted(RUNTIME.rglob("*.py"))


def test_the_runtime_reads_no_placement_of_its_own() -> None:
    assert runtime_sources()
    found = {
        str(path.relative_to(RUNTIME)): [
            word for word in BANNED if word in path.read_text()
        ]
        for path in runtime_sources()
    }
    assert {name: words for name, words in found.items() if words} == {}


ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]+")


def env_names_read() -> list[str]:
    """Env-like names the runtime names in ENV_ constants, .get() or [...]."""
    found = []
    for path in runtime_sources():
        for node in ast.walk(ast.parse(path.read_text())):
            key = None
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("ENV_")
                for t in node.targets
            ):
                key = node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
            ):
                key = node.args[0]
            elif isinstance(node, ast.Subscript):
                key = node.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and ENV_NAME.fullmatch(key.value)
            ):
                found.append(key.value)
    return found


def test_the_runtime_reads_only_scenario_env() -> None:
    names = env_names_read()
    assert "SCENARIO_PLAN" in names
    assert [name for name in names if not name.startswith("SCENARIO_")] == []
