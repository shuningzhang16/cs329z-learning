"""One JSON tool call per assistant reply, delimited by explicit tags."""

from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall


_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"


def render_tools(schemas: list[dict]) -> str:
    return (
        "Available tools (JSON Schemas describe their arguments):\n"
        + json.dumps(schemas, ensure_ascii=False)
        + '\n\nTo use a tool, emit exactly one call in this format:\n'
        '<tool_call>{"name":"tool_name","args":{"parameter":"value"}}</tool_call>\n'
        "Use the exact registered name and argument names. Include all required "
        "arguments, obey their types and constraints, and do not add extra arguments. "
        "Arguments must be valid JSON: quote strings and escape embedded quotes, "
        "backslashes, and newlines; use numbers, true, false, and null with their "
        "JSON types. For a tool without arguments use an empty object {}. "
        "You may write a brief explanation before the call. Never put multiple "
        "calls in one reply; wait for each tool result before the next call. "
        "When no tool is needed, give a plain-text final answer without tool tags."
    )


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps(
        {"name": call.name, "args": call.args}, ensure_ascii=False, allow_nan=False
    )
    return f"{text.strip()}\n{_OPEN}{payload}{_CLOSE}".strip()


def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def parse_response(text: str) -> ParsedResponse:
    start = text.find(_OPEN)
    if start < 0:
        return ParsedResponse(text.strip())
    try:
        # Decode before searching for the closing tag: a string argument can
        # contain a complete tagged call, which is data rather than another call.
        payload_start = start + len(_OPEN)
        while payload_start < len(text) and text[payload_start].isspace():
            payload_start += 1
        payload, end = _DECODER.raw_decode(text, payload_start)
        if not isinstance(payload, dict):
            raise ValueError("The tool call must be a JSON object.")
        if not isinstance(payload.get("name"), str) or not isinstance(payload.get("args"), dict):
            raise ValueError("A tool call needs a string name and an args object.")
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith(_CLOSE, end):
            raise ValueError(f"The tool call is missing {_CLOSE}.")
        end += len(_CLOSE)
        if _OPEN in text[end:]:
            raise ValueError("Only one tool call is allowed per reply.")
        remaining = (text[:start] + text[end:]).strip()
        return ParsedResponse(remaining, ToolCall(payload["name"], payload["args"]))
    except Exception as exc:
        return ParsedResponse(text.strip(), error=f"Malformed tool call: {exc}")
