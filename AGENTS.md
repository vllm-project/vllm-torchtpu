# Agent Instructions for vLLM TPU

## Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm-torchtpu --comments
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

## Development workflow

### Running pre-commit

Install both hook types in the active project virtual environment:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Run the relevant checks before proposing a PR:

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff --all-files
```

### DCO sign-off

Every commit must include a Developer Certificate of Origin sign-off. Create
signed commits with `git commit --signoff` (or `git commit -s`) and verify that
the commit message contains this trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```
