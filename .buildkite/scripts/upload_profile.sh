#!/bin/bash
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

# Copy a benchmark run's xprof traces to the GCS dir named in its config.json,
# then drop the local copy so artifact_paths doesn't upload them again.
#
# Usage: upload_profile.sh <results_dir>
set -euo pipefail

results_dir=${1:?usage: upload_profile.sh <results_dir>}
profile_dir=$results_dir/profile

if [ ! -d "$profile_dir" ]; then
  echo "::error::No $profile_dir; the profile run never ran, see $results_dir/benchmark.log."
  exit 1
fi

for phase in "$profile_dir"/*/; do
  [ -d "$phase" ] || continue
  echo "phase $(basename "$phase"): $(find "$phase" -name '*.xplane.pb' | wc -l) xplane files"
done

trace_count=$(find "$profile_dir" -name '*.xplane.pb' | wc -l)
if [ "$trace_count" -eq 0 ]; then
  echo "::error::No xplane traces under $profile_dir; the profiler was armed but wrote nothing xprof can open."
  exit 1
fi

dest=$(jq -r '.profile_gcs_dir // ""' "$results_dir/config.json")
if [ -z "$dest" ]; then
  echo "::error::config.json carries no profile_gcs_dir, so nothing states where these profiles belong."
  exit 1
fi

echo "Uploading $trace_count traces to $dest"
gcloud storage cp -r "$profile_dir"/* "$dest/"
rm -rf "$profile_dir"
