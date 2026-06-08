#!/bin/sh
set -eu

event="${1:-Unknown}"
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
memory_dir="$repo_root/docs/memory"
log_file="$memory_dir/hook-events.log"
message_file="$memory_dir/hook-reminders.log"
progress_file="$memory_dir/progress.md"
current_file="$memory_dir/current.md"

mkdir -p "$memory_dir"

timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf '%s %s\n' "$timestamp" "$event" >> "$log_file"

ensure_progress_file() {
  if [ ! -f "$progress_file" ]; then
    cat > "$progress_file" <<'MSG'
# 进展

本文件是当前工作的轻量交接记录。保持简洁，并在活跃任务、下一步或阻塞项变化时更新。

## 当前工作

- 尚未记录活跃工作。

## 下一步

- 尚未记录下一步。

## 阻塞项

- 未记录。

## Hook 活动

hook 事件可能在下方追加简短提醒。Codex 仍负责把这些提醒转化为有用的当前工作和下一步条目。
MSG
  fi
}

ensure_current_file() {
  if [ ! -f "$current_file" ]; then
    cat > "$current_file" <<'MSG'
# 当前上下文

本文件是新 Codex 会话的短活跃上下文。保持简洁。

## 当前工作

- 尚未记录活跃工作。

## 稳定决策

- 存在 `AGENTS.md` 和 `docs/agent-start.md` 时，优先读取它们。
- 面向人阅读的文档、agent harness 说明、hook 提醒、脚本输出和未来 harness 设定默认使用中文。

## 下一步

- 尚未记录下一步。

## 阻塞项

- 无。
MSG
  fi
}

record_progress_event() {
  ensure_progress_file
  ensure_current_file
  case "$event" in
    SessionStart|UserPromptSubmit|SubagentStart|PreCompact|PostCompact|Stop)
      printf '\n- %s `%s`: 如果任务状态变化，刷新当前工作、下一步和阻塞项。\n' "$timestamp" "$event" >> "$progress_file"
      ;;
  esac
}

record_progress_event

case "$event" in
  SessionStart)
    cat >> "$message_file" <<'MSG'
[repo-hook] SessionStart:
立即读取 AGENTS.md。然后在存在时读取 docs/agent-start.md、docs/memory/current.md、
docs/README.md；代码任务还要读取 docs/code-index.md。
只有当短上下文说明需要更多历史时，才读取 docs/memory/progress.md 或 inbox.md。
使用 docs/ 作为项目记忆根目录，并按任务需要创建合适的文档结构。
把持久需求、设计、决策、设置记录和交接上下文记录到 docs/ 下。
保持 docs/memory/current.md 包含简洁的当前工作、下一步和阻塞项。
较长历史移到 docs/memory/progress.md。
代码布局、入口点、配置或资源边界变化时，保持 docs/code-index.md 最新。
使用 agent-code/scripts/agent/preflight.sh、agent-code/scripts/agent/impact.sh 和 agent-code/scripts/agent/check.sh，避免重复猜测设置和验证命令。
保存记忆时，选择现有文档、新文件或新目录，并清理过期记录。
MSG
    ;;
  UserPromptSubmit)
    cat >> "$message_file" <<'MSG'
[repo-hook] UserPromptSubmit:
如果本次提示包含持久需求、设计选择、决策，或 “remember/记录/记一下” 请求，
本轮就把简洁摘要写入 docs/。不要存储 secrets。
如果活跃任务变化，更新 docs/memory/current.md 中的当前工作和下一步。
选择正确的文档位置，并清理过期或重复记忆。
MSG
    ;;
  PreToolUse)
    cat >> "$message_file" <<'MSG'
[repo-hook] PreToolUse:
修改文件前，判断这次变更是否也需要更新 docs/ 下的文档。
MSG
    ;;
  PostToolUse)
    cat >> "$message_file" <<'MSG'
[repo-hook] PostToolUse:
如果工具改变了行为、设置、架构、接口或项目知识，
现在就更新 docs/，让未来 Codex 会话可以恢复上下文。
如果工具改变了任务进展，简短更新 docs/memory/progress.md。
如果工具改变了代码布局、入口点、配置或资源边界，更新 docs/code-index.md。
必要时创建新的 docs 路径；移除、合并或标记已废弃的过期记忆。
MSG
    ;;
  PermissionRequest)
    cat >> "$message_file" <<'MSG'
[repo-hook] PermissionRequest:
如果这次权限请求暴露了设置约束或操作知识，把脱敏记录写到 docs/ 下。
MSG
    ;;
  SubagentStart)
    cat >> "$message_file" <<'MSG'
[repo-hook] SubagentStart:
告诉子代理遵循 AGENTS.md，并使用 docs/ 保存持久项目记忆。
MSG
    ;;
  PreCompact)
    cat >> "$message_file" <<'MSG'
[repo-hook] PreCompact:
压缩前，把未解决任务状态、决策和下一步写入 docs/memory/current.md 及其他相关 docs/。
依赖压缩前，整理记忆并清理过时记录。
MSG
    ;;
  PostCompact)
    cat >> "$message_file" <<'MSG'
[repo-hook] PostCompact:
压缩后，继续工作前重新读取 AGENTS.md、docs/agent-start.md、docs/memory/current.md 和相关 docs/ 文件。
MSG
    ;;
  Stop)
    cat >> "$message_file" <<'MSG'
[repo-hook] Stop:
结束前，确保 docs/memory/current.md 和其他 docs/ 包含本轮有用的交接、决策和开放问题。
保持 docs/ 有序：更新现有文件，必要时创建新文件或目录，并清理过期记忆。
MSG
    ;;
  *)
    printf '[repo-hook] %s: 遵循 AGENTS.md，并把持久项目记忆保存在 docs/ 下。\n' "$event" >> "$message_file"
    ;;
esac

printf '{}\n'
exit 0
