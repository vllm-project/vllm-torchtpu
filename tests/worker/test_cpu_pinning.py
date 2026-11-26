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
from vllm_torchtpu.worker import cpu_pinning


def test_cpulist_parses_ranges_and_singles():
    assert cpu_pinning._cpulist("0-2,5,112-113") == [0, 1, 2, 5, 112, 113]


def _node0(monkeypatch, cores=56):
    monkeypatch.setattr(
        cpu_pinning, "physical_cores", lambda node: [(c, c + 112) for c in range(cores)]
    )


def test_numa_mode_gives_the_whole_node(monkeypatch):
    _node0(monkeypatch)
    assert len(cpu_pinning.plan("numa", 0, 1, 4, reserve=2)) == 112


def test_slices_are_disjoint_whole_cores_and_keep_the_reserve(monkeypatch):
    _node0(monkeypatch)
    slices = [cpu_pinning.plan("slice", 0, slot, 4, reserve=2) for slot in range(4)]
    assert all(len(s) == 26 for s in slices)  # 13 cores with their SMT siblings
    union = set().union(*slices)
    assert len(union) == 4 * 26
    assert all((c + 112 in s) == (c in s) for s in slices for c in range(56))
    assert not union & {54, 55, 166, 167}  # the reserved last two cores
