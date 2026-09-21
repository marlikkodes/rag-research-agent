"""
agent_manual.py — the ReAct loop, written by hand
==================================================
The original implementation, kept deliberately.

agent_graph.py does the same job with LangGraph. Both are here so the two can
be read side by side: this file is the whole loop in one function, and that is
its advantage — everything that happens is visible in the order it happens.
Its disadvantage is that the loop, the retry, the step budget and the
transcript are all tangled together in that same function.

It YIELDS each step as a dictionary so the Streamlit UI can display them live.

Yielded step types:
  {"type": "step",    "step": int, "thought": str, "tool": str, "args": dict}
  {"type": "result",  "content": str}
  {"type": "final",   "answer": str, "sources": list}
  {"type": "error",   "message": str}
"""

import os
import json
import requests
import chromadb
from dotenv import load_dotenv
from tavily import TavilyClient

from providers import EMBED_MODEL, call_llm, embed_text

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

db_client = chromadb.EphemeralClient()
# The embedding model is part of the collection name on purpose. Vectors from
# two different models have different lengths and different meanings, and
# mixing them does not raise — it silently retrieves the wrong chunks. Changing
# the model gives a fresh collection rather than a corrupted one.
collection = db_client.get_or_create_collection(name=f"app_kb_{EMBED_MODEL}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def split_into_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def tool_search_web(query: str) -> tuple[str, list[dict]]:
    """Returns (summary_string, list of source dicts)."""
    results = tavily.search(query=query, max_results=4, search_depth="basic")
    sources, stored = [], 0
    for i, result in enumerate(results["results"]):
        content = result.get("content", "")
        title   = result.get("title", "Unknown")
        url     = result.get("url", "")
        if not content.strip():
            continue
        sources.append({"title": title, "url": url})
        for j, chunk in enumerate(split_into_chunks(content)):
            chunk_id = f"{query[:15]}_{i}_{j}".replace(" ", "_")
            if not collection.get(ids=[chunk_id])["ids"]:
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embed_text(chunk)],
                    documents=[chunk],
                    metadatas=[{"source": title, "url": url}],
                )
                stored += 1
    summary = f"Found {len(results['results'])} articles, stored {stored} new chunks."
    return summary, sources


def tool_query_knowledge_base(question: str) -> tuple[str, list[dict]]:
    """Returns (context_string, list of source dicts)."""
    if collection.count() == 0:
        return "Knowledge base is empty. Search the web first.", []
    results = collection.query(
        # A question, not a stored chunk — NVIDIA's retriever is trained
        # asymmetrically and gives worse results if told the wrong one.
        query_embeddings=[embed_text(question, kind="query")],
        n_results=4,
        include=["documents", "metadatas"],
    )
    context, sources = "", []
    seen_urls = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\n[Source: {meta['source']}]\n{doc}\n"
        if meta["url"] not in seen_urls:
            sources.append({"title": meta["source"], "url": meta["url"]})
            seen_urls.add(meta["url"])
    return context.strip(), sources


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research agent. You answer questions by searching
the web and querying your knowledge base.

At each step respond with ONLY a JSON object — no extra text, no markdown:
{"thought": "...", "tool": "tool_name", "args": {"param": "value"}}

Tools:
  search_web(query: str)
    → Search the web and store articles in the knowledge base.

  query_knowledge_base(question: str)
    → Retrieve relevant chunks from stored articles.

  final_answer(answer: str)
    → Write the final answer with [Source: title] citations.
    → Call this after your first query_knowledge_base. Do not keep looping.

Rules:
  - Search once or twice at most, then query, then answer.
  - After your FIRST query_knowledge_base, call final_answer immediately.
  - Output ONLY the JSON object — nothing else."""


# ─────────────────────────────────────────────
# AGENT GENERATOR
# ─────────────────────────────────────────────

def run_agent(question: str):
    """
    Generator that runs the ReAct agent loop and yields
    step updates for the Streamlit UI to display.
    """
    history  = f"User question: {question}\n\nSteps taken so far:\n"
    all_sources: list[dict] = []

    for step in range(1, 9):
        steps_left = 8 - step
        urgency = (
            " YOU MUST call final_answer NOW with what you have."
            if steps_left <= 2
            else f" ({steps_left} steps remaining — call final_answer soon after querying.)"
        )
        prompt = f"{SYSTEM_PROMPT}\n\n{history}\nWhat do you do next?{urgency}"

        raw = call_llm(prompt)
        clean = raw.strip().strip("```json").strip("```").strip()

        try:
            decision = json.loads(clean)
        except json.JSONDecodeError:
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            decision = json.loads(clean[start:end])

        thought   = decision.get("thought", "")
        tool_name = decision.get("tool", "")
        args      = decision.get("args", {})

        # Yield the step so the UI can display it
        yield {
            "type":    "step",
            "step":    step,
            "thought": thought,
            "tool":    tool_name,
            "args":    args,
        }

        if tool_name == "search_web":
            result_text, sources = tool_search_web(**args)
            all_sources.extend(s for s in sources if s not in all_sources)

        elif tool_name == "query_knowledge_base":
            result_text, sources = tool_query_knowledge_base(**args)
            all_sources.extend(s for s in sources if s not in all_sources)

        elif tool_name == "final_answer":
            yield {
                "type":    "final",
                "answer":  args.get("answer", ""),
                "sources": all_sources,
            }
            return

        else:
            yield {"type": "error", "message": f"Unknown tool: {tool_name}"}
            return

        # Yield the result summary so the UI can show it
        yield {"type": "result", "content": result_text}

        result_summary = result_text[:400] + "..." if len(result_text) > 400 else result_text
        history += (
            f"\nStep {step}:\n"
            f"  Thought: {thought}\n"
            f"  Tool: {tool_name}({args})\n"
            f"  Result: {result_summary}\n"
        )

    yield {"type": "error", "message": "Agent reached maximum steps without an answer."}
