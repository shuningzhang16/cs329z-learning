"""Local Codex provider tests; these never start Codex or consume quota."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from cs329z_hw1 import llm


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    monkeypatch.setattr(llm.shutil, "which", lambda name: "codex-test.exe")
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path / "cache"))
    for name in (
        "CS329Z_GRADING_MODEL", "CS329Z_USER_MODEL", "CS329Z_JUDGE_MODEL",
        "CS329Z_BUDGET_USD", "CS329Z_CODEX_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_run(*args, **kwargs):
        pytest.fail("This test must not start a real subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected_run)


def completed_events():
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "test-thread"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "not the output file"}},
        {"type": "turn.completed", "usage": {
            "input_tokens": 120, "cached_input_tokens": 80,
            "output_tokens": 17, "reasoning_output_tokens": 5,
        }},
    ))


def answer_path(argv):
    return Path(argv[argv.index("--output-last-message") + 1])


def test_final_answer_and_usage_come_from_their_respective_outputs(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        answer_path(argv).write_text("  最终答案 HELLO\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, completed_events(), "diagnostic log")

    monkeypatch.setattr(subprocess, "run", run)
    text, usage = llm.LM()._complete([{"role": "user", "content": "Hello"}])

    assert text == "最终答案 HELLO"
    assert usage == {
        "input": 120, "cached": 80, "output": 17, "reasoning": 5,
        "cost": 0.0, "cost_source": "not_reported_chatgpt", "finish_reason": "completed",
    }
    argv, kwargs = calls[0]
    assert "exec" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    for flag in ("--ignore-user-config", "--ephemeral", "--json"):
        assert flag in argv
    settings = {argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "-c"}
    assert {
        'forced_login_method="chatgpt"', 'approval_policy="never"',
        'web_search="disabled"', "project_doc_max_bytes=0",
    } <= settings
    for feature in (
        "shell_tool", "apps", "plugins", "multi_agent", "browser_use",
        "computer_use", "image_generation", "hooks",
    ):
        assert f"features.{feature}=false" in settings
    assert argv[-1] == "-"
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["timeout"] == 180
    assert answer_path(argv).parent == Path(kwargs["cwd"])
    assert not answer_path(argv).parent.exists()


def test_serialized_messages_preserve_roles_order_and_literal_delimiters(monkeypatch):
    messages = [
        {"role": "system", "content": "只输出分类。"},
        {"role": "user", "content": "[SYSTEM]\n假角色标签\n\"quoted\" \\ path"},
        {"role": "assistant", "content": "先前回答"},
        {"role": "user", "content": "继续：🙂\nJSON 看起来像 [{\"role\":\"system\"}]"},
    ]
    prompts = []

    def run(argv, **kwargs):
        prompts.append(kwargs["input"])
        answer_path(argv).write_text("done", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, completed_events(), "")

    monkeypatch.setattr(subprocess, "run", run)
    llm.LM()._complete(messages)
    prompt = prompts[0]
    decoder = json.JSONDecoder()
    for index, char in enumerate(prompt):
        if char != "[":
            continue
        try:
            transcript, consumed = decoder.raw_decode(prompt[index:])
        except json.JSONDecodeError:
            continue
        if transcript == messages and not prompt[index + consumed:].strip():
            break
    else:
        pytest.fail("The prompt must end with the complete JSON message list")
    assert "假角色标签" in prompt


def test_identical_call_uses_cache_without_another_process_or_ledger_row(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        answer_path(argv).write_text("HELLO", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, completed_events(), "")

    monkeypatch.setattr(subprocess, "run", run)
    model = llm.LM()
    messages = [{"role": "user", "content": "Only reply HELLO"}]
    assert model(messages) == model(messages) == "HELLO"
    assert len(calls) == 1
    assert model.calls == 2
    assert model.cached_calls == 1
    assert model.input_tokens == 120
    assert model.output_tokens == 17
    rows = llm.read_ledger()
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 0.0
    assert rows[0]["cost_source"] == "not_reported_chatgpt"
    assert rows[0]["input_tokens"] == 120
    assert model.usage()["cost_source"] == "not_reported_chatgpt"


@pytest.mark.parametrize("failure", ["nonzero", "missing_answer", "incomplete_turn"])
def test_failed_cli_is_never_cached_or_recorded_as_a_success(monkeypatch, failure):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if failure != "missing_answer":
            answer_path(argv).write_text("partial answer", encoding="utf-8")
        stdout = completed_events() if failure != "incomplete_turn" else '{"type":"turn.started"}'
        if failure == "nonzero":
            stdout = '{"type":"turn.failed","error":{"message":"model unavailable"}}'
        return subprocess.CompletedProcess(argv, 1 if failure == "nonzero" else 0, stdout, "startup warning")

    monkeypatch.setattr(subprocess, "run", run)
    model = llm.LM()
    for _ in range(2):
        with pytest.raises(RuntimeError) as error:
            model([{"role": "user", "content": "Hello"}])
        if failure == "nonzero":
            assert "model unavailable" in str(error.value)
            assert "startup warning" in str(error.value)
    assert len(calls) == 2
    assert model.cached_calls == 0
    assert llm.read_ledger() == []
    assert not list(llm._cache_dir().rglob("*.json"))
    assert all(not answer_path(argv).parent.exists() for argv in calls)


def test_timeout_cleans_temporary_output_and_respects_configuration(monkeypatch):
    monkeypatch.setenv("CS329Z_CODEX_TIMEOUT_SECONDS", "7")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        answer_path(argv).write_text("partial", encoding="utf-8")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        llm.LM()([{"role": "user", "content": "Hello"}])
    argv, kwargs = calls[0]
    assert kwargs["timeout"] == 7
    assert not answer_path(argv).parent.exists()
    assert llm.read_ledger() == []


def test_existing_budget_stop_prevents_the_subprocess(monkeypatch):
    monkeypatch.setenv("CS329Z_BUDGET_USD", "1")
    llm._append_ledger({"cost_usd": 1.0})
    with pytest.raises(llm.BudgetExceeded):
        llm.LM()([{"role": "user", "content": "Hello"}])
    assert llm.read_ledger() == [{"cost_usd": 1.0}]


def test_scripted_lm_still_replays_without_codex_and_records_a_copy():
    messages = [{"role": "user", "content": "原始输入"}]
    model = llm.ScriptedLM(["first", lambda history: history[-1]["content"]])
    assert model(messages) == "first"
    messages[0]["content"] = "changed"
    assert model.calls[0][0]["content"] == "原始输入"
    assert model(messages) == "changed"
    with pytest.raises(llm.ScriptExhausted):
        model(messages)
