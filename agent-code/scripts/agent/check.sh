#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PROJECT_PYTHON:-}" ]]; then
  PROJECT_PYTHON="${PROJECT_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PROJECT_PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python" ]]; then
  PROJECT_PYTHON="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python"
else
  PROJECT_PYTHON="${PYTHON:-python}"
fi
PYTEST_BIN="${PYTEST_BIN:-pytest}"

usage() {
  cat <<'MSG'
用法: agent-code/scripts/agent/check.sh <scope>

可用 scope:
  python
  scripts
  docs
  agent-harness
  all
MSG
}

run() {
  echo "+ $*"
  "$@"
}

run_pytest() {
  echo "+ ${PYTEST_BIN} $*"
  "${PYTEST_BIN}" "$@"
}

check_scripts_syntax() {
  while IFS= read -r script; do
    run bash -n "${script}"
  done < <(find agent-code/scripts -type f -name '*.sh' -not -path '*/__pycache__/*' | sort)
}

check_agent_harness_syntax() {
  run bash -n agent-code/scripts/codex-hook.sh
  run bash -n agent-code/scripts/codex-post-tool-memory.sh
  run bash -n agent-code/scripts/agent/preflight.sh
  run bash -n agent-code/scripts/agent/impact.sh
  run bash -n agent-code/scripts/agent/check.sh
}

scope="${1:-}"
case "${scope}" in
  -h|--help|"")
    usage
    exit 0
    ;;
  python)
    if [[ -d tests && -d agent-code/tests ]]; then
      run_pytest tests agent-code/tests -q
    elif [[ -d tests ]]; then
      run_pytest tests -q
    elif [[ -d agent-code/tests ]]; then
      run_pytest agent-code/tests -q
    else
      echo "尚无 tests/ 或 agent-code/tests/ 目录。"
    fi
    ;;
  scripts)
    check_scripts_syntax
    if [[ -d agent-code/tests/agent ]]; then
      run_pytest agent-code/tests/agent -q
    fi
    ;;
  docs)
    if [[ -d agent-code/tests/agent ]]; then
      run_pytest agent-code/tests/agent -q
    fi
    run env -u LD_LIBRARY_PATH git diff --check
    ;;
  agent-harness)
    run_pytest agent-code/tests/agent -q
    check_agent_harness_syntax
    ;;
  all)
    "$0" agent-harness
    "$0" scripts
    "$0" docs
    "$0" python
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
