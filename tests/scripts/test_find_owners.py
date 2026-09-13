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
"""Unit tests for scripts/find_owners.py.

The pattern matcher and the set cover are pure functions with no TPU or
network dependency, so they are cheap to pin down here. Everything that needs
`git` or `gh` is left to manual use of the tool.
"""

import importlib.util
import pathlib
import tempfile

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
           "find_owners.py")
_spec = importlib.util.spec_from_file_location("find_owners", _SCRIPT)
find_owners = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(find_owners)


def _matches(pattern, path):
    return find_owners.compile_pattern(pattern).match(path) is not None


# --------------------------------------------------------------------------
# pattern matching
# --------------------------------------------------------------------------
def test_bare_star_matches_at_any_depth():
    assert _matches("*", "README.md")
    assert _matches("*", "src/vllm_torchtpu/kernels/flash.py")


def test_leading_slash_anchors_to_the_repo_root():
    assert _matches("/docs/", "docs/index.md")
    assert not _matches("/docs/", "src/docs/index.md")
    assert _matches("/README.md", "README.md")
    assert not _matches("/README.md", "docs/README.md")


def test_directory_rule_owns_everything_beneath_it():
    assert _matches("/docs/", "docs/a/b/c.md")
    # A directory rule written without the trailing slash behaves the same.
    assert _matches("/docs", "docs/a/b/c.md")
    assert _matches("/docs/", "docs")


def test_trailing_star_owns_direct_children_only():
    # The example from GitHub's "About code owners", and the one place
    # CODEOWNERS departs from gitignore.
    assert _matches("docs/*", "docs/getting-started.md")
    assert not _matches("docs/*", "docs/build-app/troubleshooting.md")
    # A directory rule still owns the whole subtree, as does `**`.
    assert _matches("docs/", "docs/build-app/troubleshooting.md")
    assert _matches("docs/**", "docs/build-app/troubleshooting.md")


def test_unanchored_pattern_matches_at_any_level():
    assert _matches("docs/", "src/docs/a.md")
    assert _matches("*.py", "src/vllm_torchtpu/a.py")
    assert not _matches("*.py", "src/vllm_torchtpu/a.txt")


def test_single_star_stops_at_a_separator():
    assert _matches("/src/*/x.py", "src/a/x.py")
    assert not _matches("/src/*/x.py", "src/a/b/x.py")


def test_double_star_spans_separators():
    assert _matches("/src/**/tests/", "src/a/b/tests/t.py")
    assert _matches("/src/**/tests/", "src/tests/t.py")


def test_question_mark_matches_one_character():
    assert _matches("/a?.py", "ab.py")
    assert not _matches("/a?.py", "abc.py")


def test_dots_are_literal_not_wildcards():
    assert not _matches("/README.md", "READMExmd")


# --------------------------------------------------------------------------
# CODEOWNERS parsing
# --------------------------------------------------------------------------
def test_comments_and_blank_lines_are_ignored():
    rules = find_owners.parse_codeowners("""
        # a comment
        * @alice

        /docs/ @bob  # trailing comment
    """)
    assert [(r.pattern, r.owners) for r in rules] == [
        ("*", ["@alice"]),
        ("/docs/", ["@bob"]),
    ]


def test_team_handles_are_kept():
    rules = find_owners.parse_codeowners("/src/ @org/team @alice")
    assert rules[0].owners == ["@org/team", "@alice"]


def test_pattern_without_owners_un_owns_the_path():
    # GitHub lets a later ownerless rule strip ownership granted earlier.
    rules = find_owners.parse_codeowners("/docs/ @alice\n/docs/generated/\n")
    groups, unowned = find_owners.group_by_rule(
        ["docs/index.md", "docs/generated/api.md"], rules)
    assert list(groups) == ["/docs/"]
    assert groups["/docs/"]["files"] == ["docs/index.md"]
    assert unowned == ["docs/generated/api.md"]


def test_codeowners_override_is_read_instead_of_the_checkout():
    # The approval gate uses this to enforce the base branch's ownership while
    # running the PR's copy of the resolver.
    with tempfile.TemporaryDirectory() as d:
        override = pathlib.Path(d) / "CODEOWNERS"
        override.write_text("/docs/ @alice\n")
        rules, path = find_owners.load_rules(d, str(override))
    assert [(r.pattern, r.owners) for r in rules] == [("/docs/", ["@alice"])]
    assert path == str(override)


def test_empty_codeowners_yields_no_rules_rather_than_exiting():
    # An empty base-branch CODEOWNERS must degrade to "nothing to gate".
    assert find_owners.parse_codeowners("") == []


def test_last_matching_rule_wins():
    rules = find_owners.parse_codeowners("* @alice\n/docs/ @bob\n")
    assert find_owners.owners_for("docs/a.md", rules).owners == ["@bob"]
    assert find_owners.owners_for("main.py", rules).owners == ["@alice"]


# --------------------------------------------------------------------------
# grouping and set cover
# --------------------------------------------------------------------------
def _groups(mapping):
    return {
        pattern: {
            "owners": owners,
            "files": ["f"]
        }
        for pattern, owners in mapping.items()
    }


def test_files_are_grouped_under_their_effective_rule():
    rules = find_owners.parse_codeowners("* @alice\n/docs/ @bob\n")
    groups, unowned = find_owners.group_by_rule(["docs/a.md", "main.py"],
                                                rules)
    assert groups["/docs/"]["files"] == ["docs/a.md"]
    assert groups["*"]["files"] == ["main.py"]
    assert unowned == []


def test_cover_prefers_the_owner_spanning_the_most_areas():
    groups = _groups({
        "/a/": ["@alice", "@carol"],
        "/b/": ["@bob", "@carol"],
        "/c/": ["@carol"],
    })
    assert find_owners.minimal_cover(groups) == ["@carol"]


def test_cover_falls_back_to_several_reviewers_when_needed():
    groups = _groups({"/a/": ["@alice"], "/b/": ["@bob"]})
    assert sorted(find_owners.minimal_cover(groups)) == ["@alice", "@bob"]


def test_author_is_never_suggested_as_their_own_reviewer():
    groups = _groups({"/a/": ["@alice", "@bob"]})
    assert find_owners.minimal_cover(groups, author="alice") == ["@bob"]


def test_author_match_ignores_case_and_the_at_sign():
    groups = _groups({"/a/": ["@Alice", "@bob"]})
    assert find_owners.minimal_cover(groups, author="ALICE") == ["@bob"]


def test_area_owned_solely_by_the_author_yields_no_reviewer():
    groups = _groups({"/a/": ["@alice"]})
    assert find_owners.minimal_cover(groups, author="alice") == []
    assert find_owners.self_owned_areas(groups, author="alice") == ["/a/"]


def test_self_owned_area_does_not_suppress_the_rest_of_the_cover():
    groups = _groups({"/a/": ["@alice"], "/b/": ["@bob"]})
    assert find_owners.minimal_cover(groups, author="alice") == ["@bob"]
    assert find_owners.self_owned_areas(groups, author="alice") == ["/a/"]


def test_no_groups_means_no_reviewers():
    assert find_owners.minimal_cover({}) == []
    assert find_owners.self_owned_areas({}, author="alice") == []
