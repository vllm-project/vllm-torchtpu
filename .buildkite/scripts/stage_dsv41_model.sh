#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
#   stage_dsv41_model.sh <hf-repo> <gs://bucket/prefix>
#
# Copies a public Hugging Face checkpoint into GCS once, for the dsv41 lane's
# serving benchmarks to stream with runai_streamer. The fleet cannot read the
# bucket the VM lane used, and every pod can reach huggingface.co.
#
# Each file is piped from Hugging Face into `gcloud storage cp -`, so nothing
# lands on the pod's disk. <prefix>/.staged is written last, after every
# object's size matches the revision's file list; a later run that finds it
# does nothing, and a run that died part way re-copies only the files whose
# object is missing or the wrong size.
set -euo pipefail

repo="${1:?Hugging Face repo is required, e.g. deepseek-ai/DeepSeek-V4.1-Flash}"
dest="${2:?destination gs:// prefix is required}"
dest="${dest%/}"
if [[ "$dest" != gs://*/* ]]; then
  echo "Expected gs://bucket/prefix, got: $dest" >&2
  exit 2
fi
marker="$dest/.staged"
jobs="${STAGE_PARALLELISM:-4}"

if gcloud storage cat "$marker" 2>/dev/null; then
  echo
  echo "--- $dest is already staged"
  exit 0
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

echo "--- Listing ${repo} on Hugging Face"
# huggingface_hub comes with transformers in the CI image. HF_TOKEN, when the
# pod has one, only raises the rate limit; the repo is public.
python3 - "$repo" "$scratch/files.tsv" "$scratch/revision" <<'PY'
import sys

from huggingface_hub import HfApi

repo, files, revision = sys.argv[1:]
info = HfApi().model_info(repo, files_metadata=True)
with open(files, "w") as out:
    for sibling in info.siblings:
        if sibling.size is None:
            raise SystemExit(f"{sibling.rfilename} has no size in the file list")
        out.write(f"{sibling.rfilename}\t{sibling.size}\n")
with open(revision, "w") as out:
    out.write(info.sha)
PY
revision="$(cat "$scratch/revision")"
count="$(wc -l < "$scratch/files.tsv" | tr -d ' ')"
total="$(awk -F'\t' '{s += $2} END {printf "%.1f", s / 2^30}' "$scratch/files.tsv")"
echo "${count} files, ${total} GiB at revision ${revision}"

# Sizes of what is already there, so a resumed run skips finished files.
gcloud storage ls -l -r "$dest/**" 2>/dev/null \
  | awk -v p="$dest/" '$1 ~ /^[0-9]+$/ && index($3, p) == 1 {
      print substr($3, length(p) + 1) "\t" $1 }' > "$scratch/existing.tsv" || true

# FILENAME, not NR == FNR: the listing is empty on a first run, and NR == FNR
# would then hold for every line of the file list too.
awk -F'\t' -v have_file="$scratch/existing.tsv" \
  'FILENAME == have_file { have[$1] = $2; next } have[$1] != $2 { print $1 }' \
  "$scratch/existing.tsv" "$scratch/files.tsv" > "$scratch/todo.txt"
echo "--- Copying $(wc -l < "$scratch/todo.txt" | tr -d ' ') files to ${dest}"

export repo revision dest
# One file per line; names in a Hugging Face repo hold no newlines. The body
# is single-quoted on purpose: the child shell expands the exported names.
# shellcheck disable=SC2016
xargs -P "$jobs" -I{} bash -c '
  set -o pipefail
  file="$1"
  url="https://huggingface.co/${repo}/resolve/${revision}/${file}"
  auth=()
  [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  for attempt in 1 2 3; do
    if curl -fsSL --retry 3 "${auth[@]}" "$url" \
        | gcloud storage cp --quiet - "${dest}/${file}"; then
      echo "copied ${file}"
      exit 0
    fi
    echo "WARNING: ${file} attempt ${attempt} failed" >&2
    sleep $((attempt * 10))
  done
  echo "ERROR: could not copy ${file}" >&2
  exit 1
' _ {} < "$scratch/todo.txt"

echo "--- Verifying object sizes"
# An empty prefix makes `ls` fail; the size check below reports it instead.
gcloud storage ls -l -r "$dest/**" 2>/dev/null \
  | awk -v p="$dest/" '$1 ~ /^[0-9]+$/ && index($3, p) == 1 {
      print substr($3, length(p) + 1) "\t" $1 }' > "$scratch/existing.tsv" || true
missing="$(awk -F'\t' -v have_file="$scratch/existing.tsv" \
  'FILENAME == have_file { have[$1] = $2; next }
  have[$1] != $2 { print $1 " (want " $2 ", have " (have[$1] == "" ? "none" : have[$1]) ")" }' \
  "$scratch/existing.tsv" "$scratch/files.tsv")"
if [[ -n "$missing" ]]; then
  echo "ERROR: objects missing or the wrong size:" >&2
  echo "$missing" >&2
  exit 1
fi

printf '{"repo": "%s", "revision": "%s", "files": %s}\n' \
  "$repo" "$revision" "$count" | gcloud storage cp --quiet - "$marker"
echo "--- Staged ${repo}@${revision} at ${dest}"
