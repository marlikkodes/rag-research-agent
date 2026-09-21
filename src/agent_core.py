"""
agent_core.py — picks an implementation.

Two exist and both work:

    agent_manual.py   the ReAct loop written by hand, one function
    agent_graph.py    the same loop as a LangGraph state machine

They yield identical step dictionaries, so everything above this line — the
Streamlit UI included — cannot tell them apart. That is the point: the
comparison is real because only the implementation differs.

    AGENT_IMPL=manual streamlit run app.py     # the hand-written loop
    AGENT_IMPL=graph  streamlit run app.py     # LangGraph (default)
"""

import os

IMPL = os.getenv("AGENT_IMPL", "graph").lower()

if IMPL == "manual":
    from agent_manual import run_agent  # noqa: F401
else:
    from agent_graph import run_agent  # noqa: F401

__all__ = ["run_agent", "IMPL"]
