#!/usr/bin/env bash
#
# Guard the code-only boundary of the release branch.
#
# This branch ships the method, its configuration and the training, inference
# and evaluation entry points; it does not ship the outcomes those entry points
# produce. The check fails when a tracked path or a tracked page breaks that
# boundary:
#
#   * results/ and outputs/ hold generated per-item scores and run artifacts
#   * the release documentation ends up carrying the paper outcome tables
#
# Usage:
#   bash scripts/check_release_hygiene.sh
#
# Run it from anywhere; it works on the repository of its own location.
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

status=0

if git ls-files -- 'results/*' 'results/**' | grep -q .; then
    echo "check_release_hygiene.sh: results/ is tracked; paper outcome tables do not belong to the code release" >&2
    status=1
fi

if git ls-files -- 'outputs/*' 'outputs/**' | grep -q .; then
    echo "check_release_hygiene.sh: outputs/ is tracked; run artifacts stay untracked" >&2
    status=1
fi

if git grep -nE '## Outcome|Expected numbers|Headline result' -- 'docs/**' 'experiments/**'; then
    echo "check_release_hygiene.sh: the release documentation carries paper outcomes" >&2
    status=1
fi

if [ "${status}" -ne 0 ]; then
    exit 1
fi

echo "release hygiene checks passed"
