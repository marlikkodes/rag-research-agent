"""
agent_graph.py — the same ReAct loop, as a LangGraph state machine
==================================================================

`agent_manual.py` does exactly this in one function. Both are kept so they can
be read against each other.

--- what changes ---

The hand-written version is a `for` loop with the decision, the tool call, the
step budget and the transcript all interleaved. Everything is visible in the
order it happens, which is genuinely easier to follow — and it also means those
four concerns cannot be changed independently.

Here they are separate nodes over a shared state:

        decide ──► act ──► decide ──► ... ──► finish

`decide` asks the model what to do next. `act` runs the chosen tool. A
conditional edge looks at the state and sends control back to `decide` or on to
the end. The step budget is a field in the state rather than a loop counter, so
nothing has to remember to decrement it.

--- what it costs ---

More files, more concepts, and a dependency that moves quickly. For a loop this
small the framework is not obviously worth it; the reason to reach for it is
what comes next — persistence between runs, resuming a half-finished research
session, a human approval step before an expensive tool, or two agents sharing
state. Those are hard to bolt onto a `for` loop and close to free here.

--- what it does not change ---

The interface. `run_agent(question)` is still a generator yielding the same
step dictionaries, so the Streamlit UI does not know which implementation it is
talking to.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent_manual import (
    SYSTEM_PROMPT,
    tool_query_knowledge_base,
    tool_search_web,
)
from providers import call_llm

MAX_STEPS = 8


def _keep_last(_: Any, new: Any) -> Any:
    """Later writes win. The default for a plain field, stated explicitly."""
    return new


def _extend(existing: list, new: list) -> list:
    """Sources accumulate across steps rather than replacing each other."""
    seen = list(existing)
    for item in new:
        if item not in seen:
            seen.append(item)
    return seen


class AgentState(TypedDict):
    """
    Everything the agent knows, in one place.

    This is the real difference from the hand-written version. There, the same
    information lived in local variables inside one function and could only be
    reached from inside it. Here it is a value that can be inspected, logged,
    saved between runs or handed to another node.
    """
    question: str
    transcript: Annotated[str, _keep_last]
    steps_left: Annotated[int, _keep_last]

    # What the model just decided to do.
    thought: Annotated[str, _keep_last]
    tool: Annotated[str, _keep_last]
    args: Annotated[dict, _keep_last]

    # What came back from the tool.
    observation: Annotated[str, _keep_last]

    sources: Annotated[list, _extend]
    answer: Annotated[str, _keep_last]
    error: Annotated[str, _keep_last]


def _parse(raw: str) -> dict:
    """
    The model is asked for JSON and usually complies.

    Usually is not always: it fences the block, or adds a sentence before it.
    Rather than fail the run, find the outermost braces. This is the same
    forgiveness the hand-written version has, and it is needed just as much
    here — a framework does not make a model more obedient.
    """
    clean = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}") + 1
        if start == -1 or end <= start:
            raise
        return json.loads(clean[start:end])


# ── nodes ──────────────────────────────────────────────────────────────

def decide(state: AgentState) -> dict:
    """Ask the model what to do next."""
    left = state["steps_left"]
    urgency = (
        " YOU MUST call final_answer NOW with what you have."
        if left <= 2
        else f" ({left} steps remaining — call final_answer soon after querying.)"
    )
    prompt = f"{SYSTEM_PROMPT}\n\n{state['transcript']}\nWhat do you do next?{urgency}"

    try:
        decision = _parse(call_llm(prompt))
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not read the model's decision: {e}"}

    return {
        "thought": decision.get("thought", ""),
        "tool": decision.get("tool", ""),
        "args": decision.get("args", {}) or {},
        "steps_left": left - 1,
    }


def act(state: AgentState) -> dict:
    """Run whichever tool was chosen."""
    tool, args = state["tool"], state["args"]

    if tool == "final_answer":
        return {"answer": args.get("answer", "")}

    runner = {
        "search_web": tool_search_web,
        "query_knowledge_base": tool_query_knowledge_base,
    }.get(tool)

    if runner is None:
        return {"error": f"Unknown tool: {tool}"}

    try:
        observation, sources = runner(**args)
    except Exception as e:  # noqa: BLE001
        # A failing tool is information, not a crash. The model gets told and
        # can choose differently on the next turn — which is exactly the sort
        # of recovery a fixed script cannot do.
        return {"observation": f"Tool {tool} failed: {e}", "sources": []}

    return {"observation": observation, "sources": sources}


def record(state: AgentState) -> dict:
    """Append this turn to the transcript the model reads next time."""
    entry = (
        f"\nStep: thought={state['thought']!r} tool={state['tool']!r} "
        f"args={state['args']}\nResult: {state['observation'][:1500]}\n"
    )
    return {"transcript": state["transcript"] + entry}


# ── edges ──────────────────────────────────────────────────────────────

def next_move(state: AgentState) -> Literal["record", "__end__"]:
    """
    Carry on, or stop.

    Three ways to stop: an answer, an error, or running out of steps. The last
    one matters — without a budget a model that keeps deciding to search will
    search forever, and the bill is real.
    """
    if state.get("error") or state.get("answer"):
        return END
    if state["steps_left"] <= 0:
        return END
    return "record"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("decide", decide)
    g.add_node("act", act)
    g.add_node("record", record)

    g.set_entry_point("decide")
    g.add_edge("decide", "act")
    g.add_conditional_edges("act", next_move, {"record": "record", END: END})
    g.add_edge("record", "decide")

    return g.compile()


GRAPH = build_graph()


# ── the same interface the UI already speaks ───────────────────────────

def run_agent(question: str):
    """
    Generator yielding the same step dictionaries as the manual version.

    LangGraph streams state updates keyed by the node that produced them;
    this translates those into the shape the Streamlit UI already renders, so
    swapping implementations changes nothing above this line.
    """
    state: AgentState = {
        "question": question,
        "transcript": f"User question: {question}\n\nSteps taken so far:\n",
        "steps_left": MAX_STEPS,
        "thought": "", "tool": "", "args": {},
        "observation": "", "sources": [], "answer": "", "error": "",
    }

    step = 0
    sources: list[dict] = []

    for update in GRAPH.stream(state, {"recursion_limit": MAX_STEPS * 4}):
        for node, delta in update.items():
            if node == "decide":
                if delta.get("error"):
                    yield {"type": "error", "message": delta["error"]}
                    return
                step += 1
                yield {
                    "type": "step",
                    "step": step,
                    "thought": delta.get("thought", ""),
                    "tool": delta.get("tool", ""),
                    "args": delta.get("args", {}),
                }

            elif node == "act":
                if delta.get("error"):
                    yield {"type": "error", "message": delta["error"]}
                    return

                for s in delta.get("sources", []):
                    if s not in sources:
                        sources.append(s)

                if delta.get("answer"):
                    yield {"type": "final", "answer": delta["answer"], "sources": sources}
                    return

                if delta.get("observation"):
                    yield {"type": "result", "content": delta["observation"]}

    # Fell out of the loop without an answer — the step budget ran out.
    yield {
        "type": "error",
        "message": f"Stopped after {MAX_STEPS} steps without reaching an answer.",
    }
