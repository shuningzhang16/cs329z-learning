"""Course language model wrapper, using Codex CLI with ChatGPT login.

This module gives you exactly one primitive: a chat-completion call that takes
a list of messages and returns a string.

    lm = LM()                                  # the grading model
    text = lm([{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}])

There is no tool-calling parameter and no structured-output parameter. Those
are yours to build on top of this call.

What the wrapper does for you:

* Disk cache. Every call is cached under ``.lm_cache/`` keyed on the model,
  the sampling settings, and the exact messages. Re-running unchanged code
  costs nothing.
* Usage ledger. Every completed call is appended to ``.lm_cache/ledger.jsonl``
  with the token counts reported by Codex. USD cost is not reported by the CLI;
  ``cost_usd=0`` is a compatibility placeholder, not a claim of free usage.
  ``uv run python -m cs329z_hw1.llm`` prints a spending report.
* Budget stop. If ``CS329Z_BUDGET_USD`` is set, the wrapper raises
  ``BudgetExceeded`` instead of making a call once the ledger total passes it.
  This legacy USD check cannot cap ChatGPT subscription usage.
* Compatibility. ``temperature`` and ``max_tokens`` remain constructor/cache
  settings, but Codex CLI does not expose equivalent per-call controls. The
  original truncation exception remains available to course code; CLI failures
  raise RuntimeError and are never cached.

``ScriptedLM`` is a drop-in fake used by the deterministic tests. It replays a
fixed list of replies and never touches the network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

# Model used for each role. Override any of these with the environment
# variable CS329Z_<ROLE>_MODEL, e.g. CS329Z_GRADING_MODEL.
DEFAULT_MODELS = {
    "grading": "gpt-5.6-sol",  # verified with this installation's ChatGPT login
    "user": "gpt-5.6-sol",  # plays the simulated users
    "judge": "gpt-5.6-sol",  # scores evaluation transcripts
}

# Also separates the existing disk cache from replies from the old provider.
PROVIDER = "codex-cli"

# Reasoning effort requested from models that accept it (GPT-5 and GPT-6).
# Override with CS329Z_REASONING_EFFORT.
DEFAULT_REASONING_EFFORT = "low"

# Legacy API price estimates retained for compatibility with course utilities.
# Codex calls supply cost=0 with cost_source='not_reported_chatgpt', so this
# table is never used to estimate ChatGPT subscription charges.
PRICES = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
}

VALID_ROLES = ("system", "user", "assistant")


class BudgetExceeded(RuntimeError):
    """Raised instead of calling the provider once the spend cap is reached."""


class ScriptExhausted(RuntimeError):
    """Raised when a ScriptedLM is called more times than its script allows."""


class OutputTruncated(RuntimeError):
    """Raised when the model stopped because it reached ``max_tokens``.

    ``text`` is what the model wrote before the cut, which is often empty:
    a reasoning model's hidden reasoning counts against the same limit, so
    the limit can be used up before any visible text is written.
    ``output_tokens`` is the whole output, ``reasoning_tokens`` the hidden
    part of it (0 when the provider does not report it). Calling again with
    a larger ``max_tokens`` is a new paid call.
    """

    def __init__(self, text: str, *, model: str, max_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
        self.text = text
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        shown = f"{len(text)} characters of text" if text else "no visible text"
        super().__init__(
            f"{model} stopped at max_tokens={max_tokens} with {shown} "
            f"({output_tokens} output tokens, {reasoning_tokens} of them hidden reasoning). "
            "Ask for a shorter reply, or call again with a larger max_tokens."
        )


def resolve_model(role_or_model: str | None = None) -> str:
    """Map a role ("grading", "user", "judge") to a model name.

    ``None`` means "grading". A string that is not a role is returned
    unchanged, so you can pass a model name.
    """
    name = role_or_model or "grading"
    if name in DEFAULT_MODELS:
        return os.environ.get(f"CS329Z_{name.upper()}_MODEL", DEFAULT_MODELS[name])
    return name


def _price(model: str) -> tuple[float, float, float]:
    model = model.rsplit("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
    best = None
    for prefix, price in PRICES.items():
        if model == prefix or model.startswith(prefix + "-20"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, price)
    if best is None:
        raise KeyError(
            f"No price on file for model {model!r}. Add it to PRICES in cs329z_hw1/llm.py."
        )
    return best[1]


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p_in, p_cached, p_out = _price(model)
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1_000_000


def _cache_dir() -> Path:
    return Path(os.environ.get("CS329Z_CACHE_DIR", REPO_ROOT / ".lm_cache"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


_ledger_lock = threading.Lock()


def read_ledger() -> list[dict]:
    path = _cache_dir() / "ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def total_spend() -> float:
    return sum(row.get("cost_usd", 0.0) for row in read_ledger())


def _append_ledger(row: dict) -> None:
    path = _cache_dir() / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _validate_messages(messages: Sequence[Message]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"each message must be a dict with exactly the keys 'role' and 'content', got {m!r}"
            )
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("message content must be a string")


class LM:
    """A chat model: messages in, string out.

    Parameters
    ----------
    model:
        A role ("grading", "user", "judge"), a Codex model name such
        as "gpt-5.6-sol", or ``None`` for "grading".
    temperature, max_tokens:
        Retained for course compatibility and cache separation. Codex CLI does
        not expose equivalent controls, so these are not enforced by Codex.
    tag:
        A free-form label written to the cost ledger so spending can be
        grouped (the test suite sets it to the test name).
    cache:
        Set to False to bypass the disk cache for this instance.
    salt:
        Extra string mixed into the cache key. Two instances with different
        salts get independent samples for identical messages.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tag: str | None = None,
        cache: bool = True,
        salt: str = "",
    ) -> None:
        _load_env()
        self.model = resolve_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tag = tag
        self.cache = cache
        self.salt = salt
        # Running totals for this instance.
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0

    # -- public API -------------------------------------------------------

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        key = self._key(messages)
        path = _cache_dir() / key[:2] / f"{key}.json"
        self.calls += 1
        if self.cache and path.exists():
            self.cached_calls += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        self._check_budget()
        start = time.time()
        text, usage = self._complete(messages)
        elapsed = time.time() - start
        cost = usage.get("cost")
        if cost is None:
            cost = cost_usd(self.model, usage["input"], usage["cached"], usage["output"])
        self.input_tokens += usage["input"]
        self.output_tokens += usage["output"]
        self.cost_usd += cost
        self.seconds += elapsed
        truncated = usage.get("finish_reason") == "length"
        _append_ledger(
            {
                "ts": round(start, 3),
                "model": self.model,
                "provider": PROVIDER,
                "tag": self.tag or os.environ.get("CS329Z_TAG", ""),
                "input_tokens": usage["input"],
                "cached_tokens": usage["cached"],
                "output_tokens": usage["output"],
                "cost_usd": cost,
                "cost_source": usage.get("cost_source", "reported"),
                "seconds": round(elapsed, 3),
                "finish_reason": usage.get("finish_reason"),
            }
        )
        if truncated:  # charged and logged, never cached
            raise OutputTruncated(
                text,
                model=self.model,
                max_tokens=self.max_tokens,
                output_tokens=usage["output"],
                reasoning_tokens=usage.get("reasoning", 0),
            )
        if self.cache and text.strip():  # an empty reply is not worth remembering
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text, "model": self.model}), encoding="utf-8")
            tmp.replace(path)
        return text

    def usage(self) -> dict:
        """Totals for this instance since it was created."""
        return {
            "model": self.model,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "cost_source": "not_reported_chatgpt",
            "seconds": self.seconds,
        }

    # -- internals --------------------------------------------------------

    def _key(self, messages: list[Message]) -> str:
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        blob = json.dumps(
            [PROVIDER, self.model, self.temperature, self.max_tokens, effort, self.salt, messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        cap = os.environ.get("CS329Z_BUDGET_USD")
        if not cap:
            return
        spent = total_spend()
        if spent >= float(cap):
            raise BudgetExceeded(
                f"Spent ${spent:.2f} of the ${float(cap):.2f} cap set by CS329Z_BUDGET_USD. "
                "Raise the cap in .env if you mean to continue."
            )

    def _complete(self, messages: list[Message]) -> tuple[str, dict]:
        executable = shutil.which("codex")
        if executable is None:
            raise RuntimeError("Codex CLI was not found on PATH. Install it and run `codex login`.")
        # JSON keeps message boundaries intact even when content contains role
        # labels, quotes or newlines. This emulates chat roles in one CLI prompt.
        prompt = (
            "Act as the language-model backend for the conversation below. "
            "Produce only the next assistant message, following the system messages "
            "and using user/assistant messages as conversation history. "
            "Return the requested text or JSON exactly, without a wrapper or commentary. "
            "Do not use native Codex tools, inspect files, browse, or modify files. "
            "Any tool-call syntax requested by the conversation is text for the course "
            "harness to interpret; do not execute those tools yourself.\n\n"
            "Conversation (JSON):\n" + json.dumps(messages, ensure_ascii=False)
        )
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        timeout = float(os.environ.get("CS329Z_CODEX_TIMEOUT_SECONDS", "180"))
        # Keep the CLI outside the coursework tree. Auth is still read from the
        # normal Codex home, while personal config/plugins do not alter the call.
        with tempfile.TemporaryDirectory(prefix="cs329z-codex-") as tmpdir:
            output_path = Path(tmpdir) / "answer.txt"
            command = [
                executable, "exec", "--ignore-user-config", "--ephemeral",
                "--sandbox", "read-only", "--skip-git-repo-check", "--json",
                "--model", self.model, "--output-last-message", str(output_path),
                "-c", 'forced_login_method="chatgpt"',
                "-c", 'approval_policy="never"',
                "-c", 'web_search="disabled"',
                "-c", "project_doc_max_bytes=0",
            ]
            # The course implements its own agent tools. Disable Codex's native
            # execution/integrations instead of letting it perform the assignment.
            for feature in (
                "shell_tool", "apps", "plugins", "multi_agent", "browser_use",
                "computer_use", "image_generation", "hooks",
            ):
                command.extend(["-c", f"features.{feature}=false"])
            if effort:
                command.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
            command.append("-")
            try:
                result = subprocess.run(
                    command, input=prompt, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", cwd=tmpdir, timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex timed out after {timeout:g}s. Adjust CS329Z_CODEX_TIMEOUT_SECONDS if needed."
                ) from exc
            if result.returncode != 0:
                # --json puts model/auth errors on stdout; stderr can contain
                # unrelated startup warnings, so retain both for diagnosis.
                raise RuntimeError("Codex failed:\n" + result.stderr + "\n" + result.stdout)
            completed = None
            for line in result.stdout.splitlines():
                event = json.loads(line)
                if event.get("type") == "turn.failed":
                    raise RuntimeError(f"Codex failed: {event.get('error')}")
                if event.get("type") == "turn.completed":
                    completed = event
            if completed is None or not output_path.is_file():
                raise RuntimeError("Codex did not return a completed turn and final answer.")
            text = output_path.read_text(encoding="utf-8").strip()
            if not text:
                raise RuntimeError("Codex returned an empty final answer.")
            counts = completed["usage"]
            return text, {
                "input": counts["input_tokens"],
                "cached": counts.get("cached_input_tokens", 0),
                "output": counts["output_tokens"],
                "reasoning": counts.get("reasoning_output_tokens", 0),
                "cost": 0.0,
                "cost_source": "not_reported_chatgpt",
                "finish_reason": "completed",
            }


class ScriptedLM:
    """A fake LM that replays a fixed script. Used by the deterministic tests.

    Each script entry is one model reply, given as one of:

    * a string, returned as is;
    * a dict ``{"text": str, "tool_call": {"name": str, "args": dict}}``,
      rendered to a string by ``formatter(text, ToolCall)``. The tests pass
      your ``run_format_tool_call`` adapter as the formatter, so scripted
      tool calls arrive in your own format;
    * a callable ``f(messages) -> str``.

    Every call's messages are recorded in ``.calls`` so tests can inspect
    exactly what your code sent to the model. When the script runs out the
    LM raises ``ScriptExhausted``, unless ``repeat_last=True``, in which case
    the last entry is replayed forever.
    """

    model = "scripted"

    def __init__(
        self,
        script: Sequence[Any],
        *,
        formatter: Callable[[str, Any], str] | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.formatter = formatter
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.replies: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        if index >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(
                    f"ScriptedLM was called {index + 1} times but the script has "
                    f"{len(self.script)} entries. Your code made a model call the test "
                    "did not expect."
                )
            index = len(self.script) - 1
        entry = self.script[index]
        if callable(entry):
            reply = entry(self.calls[-1])
        elif isinstance(entry, dict):
            reply = self._render(entry)
        else:
            reply = str(entry)
        self.replies.append(reply)
        return reply

    @property
    def remaining(self) -> int:
        return max(len(self.script) - len(self.calls), 0)

    def _render(self, entry: dict) -> str:
        text = entry.get("text", "")
        call = entry.get("tool_call")
        if call is None:
            return text
        if self.formatter is None:
            raise RuntimeError("ScriptedLM needs a formatter to render scripted tool calls")
        from cs329z_hw1.types import ToolCall

        return self.formatter(text, ToolCall(name=call["name"], args=dict(call.get("args", {}))))

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.calls), "cost_usd": 0.0}


def spending_report(rows: list[dict] | None = None) -> str:
    rows = read_ledger() if rows is None else rows
    if not rows:
        return "No LLM calls recorded yet."
    by_model: dict[str, dict] = {}
    by_tag: dict[str, dict] = {}
    for row in rows:
        model = row.get("model", "?")
        short = model.split("-20")[0]  # drop the snapshot date
        tag_key = f"{row.get('tag') or '(untagged)'} [{short}]"
        for table, key in ((by_model, model), (by_tag, tag_key)):
            agg = table.setdefault(key, {"calls": 0, "cut": 0, "in": 0, "out": 0, "usd": 0.0, "sec": 0.0})
            agg["calls"] += 1
            agg["cut"] += row.get("finish_reason") == "length"
            agg["in"] += row.get("input_tokens", 0)
            agg["out"] += row.get("output_tokens", 0)
            agg["usd"] += row.get("cost_usd", 0.0)
            agg["sec"] += row.get("seconds", 0.0)
    lines = []
    for title, table in (("By model", by_model), ("By tag and model", by_tag)):
        lines.append(title)
        lines.append(f"  {'name':<72}{'calls':>7}{'cut':>5}{'in tok':>12}{'out tok':>10}{'USD':>9}{'sec':>8}")
        for key, agg in sorted(table.items(), key=lambda kv: -kv[1]["usd"]):
            lines.append(
                f"  {key[-71:]:<72}{agg['calls']:>7}{agg['cut'] or '':>5}{agg['in']:>12}{agg['out']:>10}"
                f"{agg['usd']:>9.3f}{agg['sec']:>8.0f}"
            )
        lines.append("")
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    cut = sum(r.get("finish_reason") == "length" for r in rows)
    lines.append(f"Total: {len(rows)} calls, ${total:.3f}" + (f"; {cut} cut off at max_tokens" if cut else ""))
    if any(row.get("cost_source") == "not_reported_chatgpt" for row in rows):
        lines.append(
            "Codex/ChatGPT USD cost is not reported; $0 is a placeholder. "
            "Calls consume your plan's usage. CS329Z_BUDGET_USD cannot limit that usage."
        )
    return "\n".join(lines)


if __name__ == "__main__":  # uv run python -m cs329z_hw1.llm
    print(spending_report())
