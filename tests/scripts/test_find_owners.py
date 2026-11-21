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

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "find_owners.py"
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
        ["docs/index.md", "docs/generated/api.md"], rules
    )
    assert list(groups) == ["/docs/"]
    assert groups["/docs/"]["files"] == ["docs/index.md"]
    assert unowned == ["docs/generated/api.md"]


def test_codeowners_override_is_read_instead_of_the_checkout():
    # --codeowners: check a PR against another copy of CODEOWNERS, such as
    # the base branch's, while running this checkout's resolver.
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
        pattern: {"owners": owners, "files": ["f"]}
        for pattern, owners in mapping.items()
    }


def test_files_are_grouped_under_their_effective_rule():
    rules = find_owners.parse_codeowners("* @alice\n/docs/ @bob\n")
    groups, unowned = find_owners.group_by_rule(["docs/a.md", "main.py"], rules)
    assert groups["/docs/"]["files"] == ["docs/a.md"]
    assert groups["*"]["files"] == ["main.py"]
    assert unowned == []


def test_cover_prefers_the_owner_spanning_the_most_areas():
    groups = _groups(
        {
            "/a/": ["@alice", "@carol"],
            "/b/": ["@bob", "@carol"],
            "/c/": ["@carol"],
        }
    )
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


# --------------------------------------------------------------------------
# balancing: one least-loaded owner per area
# --------------------------------------------------------------------------
def test_balanced_picks_takes_the_least_loaded_owner_per_area():
    groups = _groups({"/a/": ["@alice", "@bob"], "/b/": ["@carol", "@dave"]})
    loads = {"alice": 9, "bob": 2, "carol": 0, "dave": 5}
    assert find_owners.balanced_picks(groups, loads) == ["@bob", "@carol"]


def test_balanced_picks_visits_narrow_areas_first_and_reuses_the_pick():
    # carol owns both areas and is the lightest in the narrow one, so she is
    # picked there and the broad area needs nobody else.
    groups = _groups(
        {"*": ["@alice", "@bob", "@carol"], "/kernels/": ["@carol", "@dave"]}
    )
    loads = {"alice": 1, "bob": 1, "carol": 3, "dave": 8}
    assert find_owners.balanced_picks(groups, loads) == ["@carol"]
    # When the narrow pick does not own the broad area, the broad area gets
    # its own lightest owner.
    loads = {"alice": 1, "bob": 4, "carol": 9, "dave": 2}
    assert find_owners.balanced_picks(groups, loads) == ["@dave", "@alice"]


def test_balanced_picks_never_picks_the_author_and_matches_case():
    groups = _groups({"/a/": ["@Alice", "@bob"]})
    loads = {"alice": 0, "bob": 50}
    assert find_owners.balanced_picks(groups, loads, author="ALICE") == ["@bob"]
    assert find_owners.balanced_picks(_groups({"/a/": ["@alice"]}), {}, "alice") == []


def test_balanced_picks_tiebreak_is_deterministic_per_seed():
    groups = _groups({"/a/": ["@alice", "@bob", "@carol"]})
    loads = {"alice": 3, "bob": 3, "carol": 3}
    first = find_owners.balanced_picks(groups, loads, seed="123")
    assert first == find_owners.balanced_picks(groups, loads, seed="123")
    assert first[0] in {"@alice", "@bob", "@carol"}
    # Different PRs spread ties around rather than always taking one name.
    picks = {
        find_owners.balanced_picks(groups, loads, seed=str(n))[0] for n in range(40)
    }
    assert len(picks) > 1


def test_unknown_loads_count_as_zero():
    groups = _groups({"/a/": ["@alice", "@bob"]})
    assert find_owners.balanced_picks(groups, {"alice": 1}) == ["@bob"]


def test_uncovered_areas_skip_what_a_previous_pick_or_reviewer_covers():
    groups = _groups(
        {
            "*": ["@alice", "@bob"],
            "/a/": ["@bob", "@carol"],
            "/b/": ["@dave"],
            "/mine/": ["@me"],
        }
    )
    # Nothing covered: every area with an eligible owner, narrowest first.
    assert find_owners.uncovered_areas(groups, "me", set()) == ["/b/", "*", "/a/"]
    # bob (a previous pick) covers * and /a/; Dave reviewed, in any case.
    covered = {"bob", "DAVE"}
    assert find_owners.uncovered_areas(groups, "me", covered) == []


def test_prune_candidates_touches_only_pending_owners_of_balanced_areas():
    groups = _groups({"*": ["@alice", "@bob", "@carol"], "/x/": ["@dave", "@erin"]})
    pending = ["Alice", "bob", "carol", "dave", "frank"]
    # Only `*` is balanced this run; bob is its pick, carol already reviewed.
    pruned = find_owners.prune_candidates(
        groups, ["*"], pending, keep=["@bob", "carol"], author=None
    )
    # alice goes (spelled as GitHub returned her); dave owns an area that was
    # not balanced and frank owns nothing here, so a person asked them.
    assert pruned == ["Alice"]


def test_marker_round_trip_and_garbage_tolerance():
    state = {"areas": ["*"], "picks": ["bob"]}
    body = "text\n" + find_owners.format_marker(state) + "\nmore"
    assert find_owners.parse_marker(body) == state
    assert find_owners.parse_marker("no marker here") is None
    assert find_owners.parse_marker("<!-- reviewer-balancer not json -->") is None
    assert find_owners.parse_marker("<!-- reviewer-balancer [1] -->") is None


def test_balancer_comment_mentions_nobody_and_carries_the_state():
    body = find_owners.balancer_comment(
        {"*": ("@bob", 4)}, ["alice"], {"areas": ["*"], "picks": ["bob"]}
    )
    assert "@" not in body.replace("<!-- reviewer-balancer", "")
    assert "`bob` (4 pending)" in body
    assert "`alice`" in body
    assert find_owners.OPT_OUT_LABEL in body
    assert find_owners.parse_marker(body) == {"areas": ["*"], "picks": ["bob"]}


def test_loads_query_has_one_search_per_login():
    query = find_owners.loads_query(["alice", "bob-b"], "org/repo")
    assert query.count("search(") == 2
    assert 'review-requested:alice"' in query
    assert "repo:org/repo is:pr is:open review-requested:bob-b" in query
    assert "u0:" in query and "u1:" in query


class _FakeGitHub:
    """Stands in for the `gh` calls of balance_pr."""

    def __init__(self, monkeypatch, *, pending, reviewed=(), labels=(), state=None):
        self.pending = list(pending)
        self.reviewed = list(reviewed)
        self.labels = list(labels)
        self.comment = (7, state) if state is not None else (None, {})
        self.loads = {}
        self.pinned = set()  # logins GitHub refuses to withdraw
        self.requested = []
        self.pruned = []
        self.written = []

        def remove(logins, pr, repo, style):
            kept = [login for login in logins if login in self.pinned]
            withdrawn = [login for login in logins if login not in self.pinned]
            self.pruned.extend(withdrawn)
            return withdrawn, kept

        m = monkeypatch
        m.setattr(
            find_owners,
            "pr_review_state",
            lambda pr, repo: (self.pending, self.reviewed, self.labels),
        )
        m.setattr(find_owners, "read_marker", lambda pr, repo: self.comment)
        m.setattr(find_owners, "pending_review_loads", lambda c, repo: self.loads)
        m.setattr(
            find_owners,
            "request_reviewers",
            lambda handles, pr, repo, style: self.requested.extend(handles),
        )
        m.setattr(find_owners, "remove_reviewers", remove)
        m.setattr(
            find_owners,
            "write_marker",
            lambda pr, repo, cid, body: self.written.append((cid, body)),
        )


_DEFAULTS = ["@p", "@q", "@r", "@s"]
_BLAST_GROUPS = {
    "*": {"owners": _DEFAULTS, "files": ["runner.py"]},
    "/kernels/": {"owners": ["@k", "@q"], "files": ["k.py"]},
}


def _balance(gh, **kw):
    kw.setdefault("request", True)
    kw.setdefault("prune", True)
    kw.setdefault("comment", True)
    return find_owners.balance_pr(
        _BLAST_GROUPS, "author", "42", "org/repo", find_owners.Style(False), **kw
    )


def test_first_run_keeps_one_owner_per_area_and_withdraws_the_rest(monkeypatch):
    # GitHub asked everyone; r is the lightest default, k the lightest kernel
    # owner (q owns both but is the busiest).
    gh = _FakeGitHub(monkeypatch, pending=["p", "q", "r", "s", "k"])
    gh.loads = {"p": 5, "q": 9, "r": 1, "s": 5, "k": 3}
    result = _balance(gh)
    assert result["areas"] == ["/kernels/", "*"]
    assert result["picks"] == ["@k", "@r"]
    assert gh.requested == []  # both were already pending
    assert gh.pruned == ["p", "q", "s"]
    ((comment_id, body),) = gh.written
    assert comment_id is None
    assert find_owners.parse_marker(body) == {
        "areas": ["*", "/kernels/"],
        "picks": ["k", "r"],
    }


def test_rerun_leaves_covered_areas_and_hand_added_reviewers_alone(monkeypatch):
    # After the first run someone also asked s by hand; the recorded picks
    # still cover both areas, so nothing moves and no comment is rewritten.
    gh = _FakeGitHub(
        monkeypatch,
        pending=["k", "r", "s"],
        state={"areas": ["*", "/kernels/"], "picks": ["k", "r"]},
    )
    result = _balance(gh)
    assert result["areas"] == []
    assert gh.requested == gh.pruned == gh.written == []


def test_a_withdrawn_pick_is_not_re_requested(monkeypatch):
    # A person removed r (the `*` pick) without replacing them: the area
    # stays covered by the record, so the balancer does not fight them.
    gh = _FakeGitHub(
        monkeypatch,
        pending=["k"],
        state={"areas": ["*", "/kernels/"], "picks": ["k", "r"]},
    )
    assert _balance(gh)["areas"] == []
    assert gh.requested == []


def test_a_submitted_review_covers_its_areas(monkeypatch):
    # q reviewed before the balancer ran; q owns both areas, so both count
    # as covered and the balancer neither asks nor withdraws anyone.
    gh = _FakeGitHub(monkeypatch, pending=["p", "r", "s", "k"], reviewed=["q"])
    result = _balance(gh)
    assert result["areas"] == []
    assert gh.requested == gh.pruned == []


def test_new_area_on_push_is_balanced_without_touching_the_rest(monkeypatch):
    # The record covers `*` only; a push added kernel files and GitHub asked
    # both kernel owners. Only the kernel area is balanced and pruned.
    gh = _FakeGitHub(
        monkeypatch,
        pending=["r", "s", "k", "q"],
        state={"areas": ["*"], "picks": ["r"]},
    )
    gh.loads = {"k": 3, "q": 9}
    result = _balance(gh)
    assert result["areas"] == ["/kernels/"]
    assert result["picks"] == ["@k"]
    assert gh.pruned == ["q"]  # s owns only `*`, which was not rebalanced
    ((comment_id, body),) = gh.written
    assert comment_id == 7
    assert find_owners.parse_marker(body)["picks"] == ["k", "r"]


def test_pick_not_yet_pending_is_requested(monkeypatch):
    gh = _FakeGitHub(monkeypatch, pending=["p"])
    gh.loads = {"p": 5, "q": 9, "r": 1, "s": 5, "k": 3}
    _balance(gh)
    assert gh.requested == ["k", "r"]
    assert gh.pruned == ["p"]


def test_opt_out_label_stops_everything(monkeypatch):
    gh = _FakeGitHub(
        monkeypatch, pending=["p", "q"], labels=[find_owners.OPT_OUT_LABEL]
    )
    result = _balance(gh)
    assert result["skipped"].endswith(find_owners.OPT_OUT_LABEL)
    assert gh.requested == gh.pruned == gh.written == []


def test_dry_balance_reports_without_acting(monkeypatch):
    gh = _FakeGitHub(monkeypatch, pending=["p", "q", "r", "s", "k"])
    gh.loads = {"p": 5, "q": 9, "r": 1, "s": 5, "k": 3}
    result = _balance(gh, request=False, prune=False, comment=False)
    assert result["picks"] == ["@k", "@r"]
    assert result["pruned"] == []
    assert gh.requested == gh.pruned == gh.written == []


def test_pinned_requests_are_reported_not_claimed(monkeypatch):
    # The ruleset still requires code-owner review, so GitHub keeps q and s.
    gh = _FakeGitHub(monkeypatch, pending=["p", "q", "r", "s", "k"])
    gh.loads = {"p": 5, "q": 9, "r": 1, "s": 5, "k": 3}
    gh.pinned = {"q", "s"}
    result = _balance(gh)
    assert result["pruned"] == ["p"]
    assert result["pinned"] == ["q", "s"]
    ((_, body),) = gh.written
    assert "Withdrew the other code-owner requests: `p`." in body
    assert "GitHub kept the requests of `q`, `s`" in body
