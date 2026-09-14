#!/bin/sh
# Resolve before uv runs: even --no-project may prepend the checkout's .venv
# to PATH, which would otherwise turn a runtime-parity test into an old-lock test.
set -eu
koder_test_command=$(command -v koder) || {
    echo "koder is not on PATH; use run_runtime_tests.py --runtime-python explicitly" >&2
    exit 2
}
koder_test_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --no-project --no-env-file python \
    "$koder_test_script_dir/run_runtime_tests.py" --koder "$koder_test_command" "$@"
