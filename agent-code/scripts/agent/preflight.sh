#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -n "${PROJECT_PYTHON:-}" ]]; then
  project_python="${PROJECT_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  project_python="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python" ]]; then
  project_python="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python"
else
  project_python="${PYTHON:-python}"
fi

PYTEST_BIN="${PYTEST_BIN:-$(command -v pytest || true)}"

echo "Codex 代理预检"
echo "仓库: ${ROOT_DIR}"
echo "项目 Python: ${project_python}"
"${project_python}" --version 2>&1 | sed 's/^/项目 Python 版本: /' || true

if [[ -n "${PYTEST_BIN}" ]]; then
  echo "pytest: ${PYTEST_BIN}"
  "${PYTEST_BIN}" --version 2>&1 | head -n 1 | sed 's/^/pytest 版本: /' || true
else
  echo "pytest: <未找到>"
fi

if command -v rg >/dev/null 2>&1; then
  echo "搜索工具: rg 可用"
else
  echo "搜索工具: 未找到 rg；使用 find/grep 兜底"
fi

echo "git 状态:"
git -C "${ROOT_DIR}" status --short | sed -n '1,40p' || true

echo "建议命令:"
echo "- agent-code/scripts/agent/impact.sh"
echo "- agent-code/scripts/agent/check.sh --help"
