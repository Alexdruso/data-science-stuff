#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs the project (with development extras) into a local .venv so that ruff,
# mypy, pytest, and the competition training scripts work out of the box in a
# fresh remote container. Mirrors `make dev-setup`. Safe to run repeatedly.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment; local machines
# already have their own .venv from `make dev-setup`.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Resolve repo root (CLAUDE_PROJECT_DIR is set during hook execution; fall back
# to the script location so manual runs also work).
cd "${CLAUDE_PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}"

VENV_DIR=".venv"

# Ensure uv is available (prefer a pre-installed uv; otherwise bootstrap it).
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --quiet uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create the virtualenv if missing (idempotent).
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi

# Install the package with development extras into the venv (uv caches across
# sessions, so warm containers reinstall quickly).
VIRTUAL_ENV="$PWD/$VENV_DIR" uv pip install -e ".[development]"

# Persist the venv on PATH for the rest of the session so `ruff`, `mypy`,
# `pytest`, and `python` resolve to the project environment.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$PWD/$VENV_DIR\""
    echo "export PATH=\"$PWD/$VENV_DIR/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: dependencies installed into $PWD/$VENV_DIR"
