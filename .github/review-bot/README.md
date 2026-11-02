# Review bot settings

The review bot, `cbs-oss-review-bot`, reviews pull requests in this repo with
Claude. It posts a GitHub review with a summary and inline comments. It never
approves a PR or blocks a merge. Repo owners control it with the files in this
folder.

The bot reads these files from `main`, so an edit takes effect once its PR
merges. A PR that edits them is still reviewed with the settings on `main`.

## Files

| File | What it does |
| --- | --- |
| `config.yaml` | Settings, listed below. A missing file or setting means the default. |
| `REVIEW.md` | Optional. Your own review instructions. See [REVIEW.md](#reviewmd). |
| `AGENTS.md` (repo root and folders) | Code rules. The bot always checks changes against them. |

Put code rules in `AGENTS.md`, because people and coding agents read it too.
Put instructions that are only for reviews in `REVIEW.md`. Where the two
disagree about a code rule, the bot follows `AGENTS.md`.

## Settings in config.yaml

The values in `config.yaml` today are the defaults.

| Setting | Possible values | Default | What it does |
| --- | --- | --- | --- |
| `nits` | `summary`, `inline`, `off` | `summary` | Where nits go. A nit is optional polish: naming, wording, a small cleanup. `summary` lists them in the review text, `inline` posts each on its line, `off` drops them. At most 10 per review. |
| `questions` | `summary`, `inline`, `off` | `summary` | Where questions go. A question is something the bot suspects but could not confirm in the code. Same choices as `nits`. At most 5 per review. |
| `max_inline_comments` | Whole number, 0 to 30 | `15` | Most inline comments in one review, counting problems, nits and questions. Problems come first. Problems past the limit go to the summary under "More problems". `0` puts everything in the summary. |
| `inline_labels` | `blocking, should-fix` or `blocking` | `blocking, should-fix` | Which problems get inline comments. A problem is labeled `blocking` (fix before merging) or `should-fix`. With only `blocking`, should-fix problems go to the summary under "More problems". |
| `ignore_paths` | Comma-separated glob patterns, or empty | empty | Files the bot does not review. It leaves them out of the diff it reviews and drops nits and questions about them. A PR that changes only ignored files gets no automatic review. |
| `cooldown_minutes` | Whole number, 15 to 1440 | `60` | Shortest time between two reviews of one PR. A request that comes sooner waits until the time is up. |
| `automatic_reviews` | `on`, `off` | `on` | `on`: the bot reviews a PR once without being asked, when the PR is ready (see below). `off`: it reviews only when someone asks. |
| `title_keywords_to_skip` | Comma-separated words, or empty | empty | More words that stop an automatic review when they appear anywhere in a PR title, ignoring case. They add to the built-in markers below. Asking for a review still works. |

### Format

`config.yaml` is a short list of `setting: value` lines, not full YAML:

- One setting per line.
- A list goes on one line, separated by commas: `docs/**, **/*.md`. A YAML
  list (`- item` lines or `[a, b]`) does not work.
- An empty value means an empty list.
- `#` starts a comment.

The bot skips a line it cannot read, and that setting keeps its default. The
next review then ends with a note to the repo owners that names the line.

### Glob patterns for ignore_paths

Patterns start at the repo root.

- `*` matches any name inside one folder.
- `**` matches across folders.

| Pattern | Matches |
| --- | --- |
| `docs/**` | Everything under `docs/` |
| `**/*.md` | Markdown files in any folder |
| `*.md` | Markdown files at the repo root only |
| `tests/**/*.json` | JSON files anywhere under `tests/` |

### Example

```yaml
nits: inline
inline_labels: blocking
max_inline_comments: 10
ignore_paths: docs/**, **/*.md
cooldown_minutes: 120
title_keywords_to_skip: experimental, on hold
```

## When the bot reviews on its own

With `automatic_reviews: on`, the bot reviews a PR once when all of these are
true:

- The PR is not a draft.
- Its title has no built-in "not ready" marker and none of the words in
  `title_keywords_to_skip`. The built-in markers are "WIP" or "draft" at the
  start or in brackets (`[WIP]`, `(draft)`), and "do not review", "do not
  merge" or "do not submit" anywhere.
- It does not have the `skip-review-bot` label.
- A bot did not open it.
- A human reviewer is requested, and no human has reviewed it yet.

Later reviews happen only on request.

## Other controls

- **Ask for a review:** comment `@cbs-oss-review-bot` in the PR's
  Conversation tab. Repo members, collaborators and allowlisted Google
  accounts can ask.
- **Skip a PR:** add the `skip-review-bot` label. The bot skips the PR, even
  when someone asks.

## REVIEW.md

Commit `REVIEW.md` in this folder to give the bot your own review
instructions: what to focus on, what to skip, how to word comments, and what
the review should look like. It replaces the bot's built-in review guide. Its
instructions win over the bot's own rules, except that inline comments still go
only on lines the PR changes. `AGENTS.md` still wins on code rules.

The bot then follows the review format in `REVIEW.md` and no longer writes its
own "Looks good." line. If you want a verdict, say in `REVIEW.md` how to state
it.

Questions, or a copy of the bot's built-in guide to start from: ask yunyao-gg.
