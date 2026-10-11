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
"""A stand-in kubectl, run as a program from a directory on PATH.

It acts out a script in $FAKE_KUBECTL/state.json and appends every call to
calls.jsonl there. Script keys: `jobset_api` (bool), `exists` (the JobSet
is there before create), `create_fails` (stderr), `pods` (one pod list per
`get pods`, the last repeating; a list of {"signal": "TERM"} sends that
signal to the caller first), `jobset` (what `get jobset` returns after
create) and `logs`.
"""

import json
import os
import pathlib
import signal
import sys
import time


def main() -> int:
    root = pathlib.Path(os.environ["FAKE_KUBECTL"])
    state_path = root / "state.json"
    state = json.loads(state_path.read_text())
    args = sys.argv[1:]
    stdin = sys.stdin.read() if args[:3] == ["create", "-f", "-"] else None
    with (root / "calls.jsonl").open("a") as calls:
        calls.write(json.dumps(args) + "\n")
    if args[0] == "api-resources":
        print("jobsets.jobset.x-k8s.io" if state.get("jobset_api", True) else "")
        return 0
    if args[:2] == ["create", "-f"]:
        if state.get("create_fails"):
            print(state["create_fails"], file=sys.stderr)
            return 1
        (root / "created.yaml").write_text(stdin or "")
        state["created"] = True
        state_path.write_text(json.dumps(state))
        print("jobset.jobset.x-k8s.io/x created")
        return 0
    if args[:2] == ["get", "jobset"]:
        if not (state.get("created") or state.get("exists")):
            print(
                f'Error from server (NotFound): jobsets "{args[2]}" not found',
                file=sys.stderr,
            )
            return 1
        print(json.dumps(state.get("jobset", {"status": {}})))
        return 0
    if args[:2] == ["get", "pods"]:
        steps = state.get("pods", [[]])
        step = state.get("pod_step", 0)
        state["pod_step"] = step + 1
        state_path.write_text(json.dumps(state))
        pods = steps[min(step, len(steps) - 1)]
        if pods and "signal" in pods[0]:
            os.kill(os.getppid(), getattr(signal, f"SIG{pods[0]['signal']}"))
            time.sleep(5)
            pods = []
        print(json.dumps({"items": pods}))
        return 0
    if args[:2] == ["logs", "-f"]:
        print(state.get("logs", ""), end="")
        return 0
    if args[:2] == ["delete", "jobset"]:
        return 0
    print(f"fake kubectl: unexpected {args}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
