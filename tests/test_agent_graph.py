"""
The graph's control flow, without calling a model.

    venv/bin/python tests/test_agent_graph.py

The model and the tools are stubbed on purpose. What is being tested is the
part LangGraph is responsible for — that `decide` leads to `act`, that a
tool result loops back for another decision, that `final_answer` ends the
run, and that the step budget stops a model which never decides to finish.

None of that needs a real model, and involving one would make the test slow,
non-deterministic and dependent on a quota.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent_graph as G  # noqa: E402


def build(decisions, search_result=("retrieved text", [{"title": "T", "url": "u"}])):
    """A graph whose model returns `decisions` in order and whose tools are fake."""
    seq = iter(decisions)
    G.call_llm = lambda prompt: next(seq)
    G.tool_search_web = lambda query: search_result
    G.tool_query_knowledge_base = lambda question: ("from memory", [])
    G.GRAPH = G.build_graph()
    return G


SEARCH = '{"thought":"need sources","tool":"search_web","args":{"query":"x"}}'
FINISH = '{"thought":"enough","tool":"final_answer","args":{"answer":"Done."}}'


def test_search_then_answer():
    g = build([SEARCH, FINISH])
    steps = list(g.run_agent("q"))

    kinds = [s["type"] for s in steps]
    assert kinds == ["step", "result", "step", "final"], kinds
    assert steps[-1]["answer"] == "Done."
    # Sources found during the run survive to the final answer, which is the
    # thing the shared state exists for.
    assert steps[-1]["sources"] == [{"title": "T", "url": "u"}]
    print("PASS  search, loop back, answer, sources carried")


def test_answers_immediately():
    g = build([FINISH])
    steps = list(g.run_agent("q"))
    assert [s["type"] for s in steps] == ["step", "final"], steps
    print("PASS  answers without using a tool")


def test_step_budget_stops_it():
    # A model that only ever searches. Without a budget this runs forever and
    # the bill is real.
    g = build([SEARCH] * 50)
    steps = list(g.run_agent("q"))
    assert steps[-1]["type"] == "error", steps[-1]
    assert "without reaching an answer" in steps[-1]["message"]

    decisions = [s for s in steps if s["type"] == "step"]
    assert len(decisions) <= G.MAX_STEPS, f"{len(decisions)} exceeds the budget"
    print(f"PASS  stopped after {len(decisions)} steps rather than looping")


def test_unknown_tool_is_reported():
    g = build(['{"thought":"?","tool":"teleport","args":{}}'])
    steps = list(g.run_agent("q"))
    assert steps[-1]["type"] == "error"
    assert "teleport" in steps[-1]["message"]
    print("PASS  an invented tool name is an error, not a crash")


def test_a_failing_tool_does_not_end_the_run():
    # The model should get told the tool failed and be free to choose again.
    def explode(query):
        raise RuntimeError("network down")

    g = build([SEARCH, FINISH])
    g.tool_search_web = explode
    g.GRAPH = g.build_graph()

    steps = list(g.run_agent("q"))
    assert steps[-1]["type"] == "final", steps
    assert any("failed" in str(s.get("content", "")) for s in steps), steps
    print("PASS  a broken tool is information, not a crash")


def test_malformed_json_is_an_error_not_an_exception():
    g = build(["not json at all"])
    steps = list(g.run_agent("q"))
    assert steps[-1]["type"] == "error", steps
    print("PASS  unreadable model output surfaces as an error")


if __name__ == "__main__":
    for fn in [
        test_search_then_answer,
        test_answers_immediately,
        test_step_budget_stops_it,
        test_unknown_tool_is_reported,
        test_a_failing_tool_does_not_end_the_run,
        test_malformed_json_is_an_error_not_an_exception,
    ]:
        fn()
    print("\nall control-flow tests passed")
