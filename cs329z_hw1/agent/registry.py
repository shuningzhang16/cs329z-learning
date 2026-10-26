"""Ordered tool registration, argument validation, and exception containment."""

from __future__ import annotations

from copy import deepcopy

from jsonschema.validators import validator_for

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self._tools = {tool.name: tool for tool in tools}
        self._schemas = [
            {"name": tool.name, "description": tool.description,
             "parameters": deepcopy(tool.parameters)}
            for tool in tools
        ]
        self._validators = {
            tool.name: validator_for(tool.parameters)(tool.parameters)
            for tool in tools
        }

    def schemas(self) -> list[dict]:
        return deepcopy(self._schemas)

    def validate(self, name: str, args: dict) -> ToolResult | None:
        if not isinstance(name, str) or name not in self._tools:
            return ToolResult("unknown_tool", f"Unknown tool: {name!r}.")
        if not isinstance(args, dict):
            return ToolResult("invalid_args", "Tool arguments must be a JSON object.")
        try:
            self._validators[name].validate(args)
        except Exception as exc:
            message = getattr(exc, "message", str(exc))
            return ToolResult("invalid_args", f"Invalid arguments for {name}: {message}")
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        error = self.validate(name, args)
        if error is not None:
            return error
        try:
            value = self._tools[name].fn(**args)
            return ToolResult("ok", str(value))
        except Exception as exc:
            return ToolResult("tool_error", f"{type(exc).__name__}: {exc}")
