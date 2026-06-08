from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_agent_startup_docs_are_indexed():
    expected = [
        ROOT / "docs/agent-start.md",
        ROOT / "docs/code-index.md",
        ROOT / "docs/memory/current.md",
        ROOT / "docs/memory/progress.md",
        ROOT / "docs/memory/maintenance.md",
        ROOT / "docs/memory/codex-hooks.md",
        ROOT / "docs/agent/README.md",
        ROOT / "docs/agent/commands.md",
    ]

    for path in expected:
        assert path.exists(), f"缺少 agent harness 文档: {path.relative_to(ROOT)}"

    docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    for path in expected:
        assert str(path.relative_to(ROOT / "docs")) in docs_index or path.name in docs_index

    agent_guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "docs/agent-start.md" in agent_guide
    assert "docs/memory/current.md" in agent_guide
    assert "docs/" in agent_guide


def test_codex_hooks_reference_repo_hook():
    hooks = json.loads((ROOT / ".codex/hooks.json").read_text(encoding="utf-8"))
    configured_events = set(hooks["hooks"])
    expected_events = {
        "SessionStart",
        "UserPromptSubmit",
        "PermissionRequest",
        "SubagentStart",
        "PreCompact",
        "PostCompact",
        "Stop",
    }
    assert expected_events <= configured_events

    commands = []
    for event_hooks in hooks["hooks"].values():
        for matcher in event_hooks:
            for hook in matcher["hooks"]:
                commands.append(hook["command"])
    assert all(command.startswith("./agent-code/scripts/codex-hook.sh ") for command in commands)


def test_agent_scripts_are_indexed_and_executable():
    scripts = [
        ROOT / "agent-code/scripts/codex-hook.sh",
        ROOT / "agent-code/scripts/codex-post-tool-memory.sh",
        ROOT / "agent-code/scripts/agent/preflight.sh",
        ROOT / "agent-code/scripts/agent/check.sh",
        ROOT / "agent-code/scripts/agent/impact.sh",
    ]

    scripts_readme = (ROOT / "agent-code/scripts/README.md").read_text(encoding="utf-8")
    for script in scripts:
        assert script.exists(), f"缺少 agent 脚本: {script.relative_to(ROOT)}"
        assert script.stat().st_mode & 0o111, f"agent 脚本不可执行: {script.relative_to(ROOT)}"
        assert str(script.relative_to(ROOT)) in scripts_readme or script.name in scripts_readme
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_agent_preflight_and_check_help_are_usable():
    preflight = subprocess.run(
        ["bash", str(ROOT / "agent-code/scripts/agent/preflight.sh")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "Codex 代理预检" in preflight.stdout
    assert "项目 Python" in preflight.stdout
    assert "pytest" in preflight.stdout

    check_help = subprocess.run(
        ["bash", str(ROOT / "agent-code/scripts/agent/check.sh"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "agent-harness" in check_help.stdout
    assert "scripts" in check_help.stdout
    assert "docs" in check_help.stdout


def test_agent_impact_reports_changed_scope():
    impact = subprocess.run(
        ["bash", str(ROOT / "agent-code/scripts/agent/impact.sh")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "变更文件" in impact.stdout
    assert "建议检查" in impact.stdout
