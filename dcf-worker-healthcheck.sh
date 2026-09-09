#!/usr/bin/env bash
set -euo pipefail

python --version >/dev/null
git --version >/dev/null
curl --version >/dev/null
sha256sum --version >/dev/null
s5cmd --version >/dev/null

if [[ -d "${DCF_WORKSPACE:-/workspace}" ]]; then
    test -r "${DCF_WORKSPACE:-/workspace}"
fi

exit 0
