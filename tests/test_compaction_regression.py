"""Regression for an auxiliary model continuing a tool conversation."""

from cs329z_hw1 import adapters
from cs329z_hw1.tokens import count_message_tokens
from cs329z_hw1.types import AgentConfig
from tests.fixture_tools import make_fixture_tools


def test_auxiliary_tool_call_does_not_replace_summary_or_execute(scripted):
    prior_summary = "Retained summary: the original user requirements still apply."
    unexpected = {"text": "", "tool_call": {"name": "big_output", "args": {"n_lines": 42}}}
    aux = scripted([prior_summary, unexpected], repeat_last=True)
    requested = {"text": "", "tool_call": {"name": "big_output", "args": {"n_lines": 3000}}}
    lm = scripted(["Noted."] + [requested] * 6 + ["Finished.", "Your constraints are retained."])
    fixture = make_fixture_tools()
    session = adapters.run_agent_session(
        lm, fixture.tools, AgentConfig(context_budget=4000, aux_lm=aux)
    )

    session.send("The offsite budget cap is 18,450 dollars and the city must be Boulder.")
    session.send("Run the large-output check six times.")
    result = session.send("What were my original offsite requirements?")

    assert result.status == "done"
    assert len(aux.calls) >= 2  # Exercise both a valid summary and the erroneous call.
    assert fixture.count("big_output") == 6
    final_context = "\n".join(message["content"] for message in lm.calls[-1])
    assert prior_summary in final_context
    assert "18,450" in final_context and "Boulder" in final_context
    assert all(count_message_tokens(messages) <= 4000 for messages in lm.calls + aux.calls)
