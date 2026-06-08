#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

tmp_file="$(mktemp)"
trap 'rm -f "${tmp_file}"' EXIT

git diff --name-only --diff-filter=ACMRTUXB > "${tmp_file}" || true
git diff --cached --name-only --diff-filter=ACMRTUXB >> "${tmp_file}" || true
git ls-files --others --exclude-standard >> "${tmp_file}" || true
sort -u "${tmp_file}" -o "${tmp_file}"

echo "变更文件"
if [[ -s "${tmp_file}" ]]; then
  sed 's/^/- /' "${tmp_file}"
else
  echo "- <无>"
fi

declare -A checks=()
while IFS= read -r path; do
  case "${path}" in
    AGENTS.md|README.md|docs/*|.codex/*)
      checks["docs"]=1
      ;;
  esac
  case "${path}" in
    agent-code/scripts/*)
      checks["scripts"]=1
      ;;
  esac
  case "${path}" in
    .codex/*|AGENTS.md|docs/agent*|docs/agent/*|docs/memory/*|agent-code/scripts/codex-hook.sh|agent-code/scripts/codex-post-tool-memory.sh|agent-code/scripts/agent/*|agent-code/tests/agent/*)
      checks["agent-harness"]=1
      ;;
  esac
  case "${path}" in
    src/*|tests/*|agent-code/tests/*|*.py|pyproject.toml|requirements*.txt)
      checks["python"]=1
      ;;
  esac
done < "${tmp_file}"

echo "建议检查"
if [[ ${#checks[@]} -eq 0 ]]; then
  echo "- agent-code/scripts/agent/check.sh docs"
  exit 0
fi

for scope in agent-harness python scripts docs; do
  if [[ -n "${checks[${scope}]:-}" ]]; then
    echo "- agent-code/scripts/agent/check.sh ${scope}"
  fi
done
