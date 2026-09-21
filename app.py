"""
app.py — RAG Research Agent — Streamlit UI
===========================================
Run with:
  venv/bin/python -m streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from agent_core import run_agent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 RAG Research Agent")
    st.markdown("""
    An AI agent that researches any topic by:
    1. **Searching** the web for real articles
    2. **Storing** them in a vector database
    3. **Retrieving** the most relevant chunks
    4. **Answering** with citations

    Built with **ChromaDB · Mistral · Tavily** — no agent framework
    """)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    - `search_web` → Tavily finds live articles
    - `query_knowledge_base` → ChromaDB finds relevant chunks via semantic search
    - `final_answer` → Gemini answers using only retrieved content
    """)
    st.divider()
    st.caption("Abbas Aamir · BSc CS, BSBI Berlin")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("RAG Research Agent")
st.markdown("Ask any research question. The agent searches the web, builds a knowledge base, and answers with cited sources.")

# Example questions
st.markdown("**Try one of these:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🤖 What is RAG and how does it work?"):
        st.session_state["question"] = "What is RAG and how does it work?"
with col2:
    if st.button("🧠 How do transformer models work?"):
        st.session_state["question"] = "How do transformer models work?"
with col3:
    if st.button("🏙️ What is the Berlin tech startup scene like?"):
        st.session_state["question"] = "What is the Berlin tech startup scene like?"

st.divider()

# Question input
question = st.text_input(
    "Or type your own research question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the main differences between RAG and fine-tuning?",
    key="question_input",
)

run = st.button("🚀 Research", type="primary", disabled=not question.strip())

# ─────────────────────────────────────────────
# AGENT RUN
# ─────────────────────────────────────────────

if run and question.strip():

    st.divider()

    # Live agent thinking panel
    with st.status("🤖 Agent is researching...", expanded=True) as status:

        final_answer = None
        sources      = []

        for update in run_agent(question.strip()):

            if update["type"] == "step":
                step    = update["step"]
                thought = update["thought"]
                tool    = update["tool"]
                args    = update["args"]

                # Tool icon mapping
                icons = {
                    "search_web":           "🌐",
                    "query_knowledge_base":  "🔎",
                    "final_answer":          "✅",
                }
                icon = icons.get(tool, "🔧")

                st.markdown(f"**Step {step}**")
                st.markdown(f"💭 *{thought}*")

                if tool == "search_web":
                    st.markdown(f'{icon} **Searching web for:** `{args.get("query", "")}`')
                elif tool == "query_knowledge_base":
                    st.markdown(f'{icon} **Querying knowledge base:** `{args.get("question", "")}`')
                elif tool == "final_answer":
                    st.markdown(f'{icon} **Writing final answer...**')

            elif update["type"] == "result":
                with st.expander("📋 Tool result (click to expand)", expanded=False):
                    st.text(update["content"][:600] + ("..." if len(update["content"]) > 600 else ""))

            elif update["type"] == "final":
                final_answer = update["answer"]
                sources      = update["sources"]
                status.update(
                    label="✅ Research complete!",
                    state="complete",
                    expanded=False,
                )

            elif update["type"] == "error":
                status.update(label="❌ Error", state="error", expanded=True)
                st.error(update["message"])

    # ─────────────────────────────────────────
    # FINAL ANSWER
    # ─────────────────────────────────────────

    if final_answer:
        st.markdown("## Answer")
        st.markdown(final_answer)

        # Sources
        if sources:
            st.divider()
            st.markdown("### Sources")
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")
