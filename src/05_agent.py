"""
STEP 5 — The Research Agent
=============================
What this file does:
  - Gives Gemini a set of TOOLS it can call
  - Runs a ReAct loop: Gemini thinks → picks a tool
    → we run it → Gemini sees the result → thinks again
  - Stops when Gemini decides it has enough to answer
  - Returns a cited answer grounded in real web content

The three tools the agent has:
  1. search_web(query)         — search Tavily, store results in ChromaDB
  2. query_knowledge_base(q)   — retrieve relevant chunks from ChromaDB
  3. final_answer(answer)      — agent is done, this is the answer

Run it with:
  venv/bin/python src/05_agent.py
"""

import os
import json
import requests
import chromadb
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ChromaDB — persistent knowledge base
db_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_agent")
db_client = chromadb.PersistentClient(path=db_path)
collection = db_client.get_or_create_collection(name="agent_knowledge_base")


# ─────────────────────────────────────────────
# HELPERS (same as before)
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    )
    r = requests.post(url, json={"content": {"parts": [{"text": text}]}}, timeout=15)
    r.raise_for_status()
    return r.json()["embedding"]["values"]


def call_gemini(prompt: str) -> str:
    models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        )
        r = requests.post(
            url,
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            timeout=90,
        )
        if r.status_code == 503:
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("All Gemini models temporarily unavailable.")


def split_into_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ─────────────────────────────────────────────
# THE THREE TOOLS
# ─────────────────────────────────────────────

def tool_search_web(query: str) -> str:
    """Search the web and store results in ChromaDB."""
    print(f'     🌐 Searching: "{query}"')
    results = tavily.search(query=query, max_results=4, search_depth="basic")
    stored = 0
    for i, result in enumerate(results["results"]):
        content = result.get("content", "")
        title   = result.get("title", "Unknown")
        url     = result.get("url", "")
        if not content.strip():
            continue
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
    return f"Stored {stored} new chunks from {len(results['results'])} articles."


def tool_query_knowledge_base(question: str) -> str:
    """Find the most relevant chunks in ChromaDB for a question."""
    print(f'     🔎 Querying knowledge base: "{question}"')
    if collection.count() == 0:
        return "Knowledge base is empty. Search the web first."
    results = collection.query(
        query_embeddings=[embed_text(question)],
        n_results=4,
        include=["documents", "metadatas"],
    )
    output = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output += f"\n[Source: {meta['source']} | {meta['url']}]\n{doc}\n"
    return output.strip() if output.strip() else "No relevant content found."


def tool_final_answer(answer: str) -> str:
    """The agent is done. Return the final answer."""
    return answer


# Map tool names to actual functions
TOOLS = {
    "search_web":           tool_search_web,
    "query_knowledge_base": tool_query_knowledge_base,
    "final_answer":         tool_final_answer,
}


# ─────────────────────────────────────────────
# THE AGENT LOOP
# ─────────────────────────────────────────────
# This is the ReAct loop:
#   1. Build a prompt describing the tools + conversation so far
#   2. Ask Gemini what to do next (returns JSON)
#   3. Parse the JSON → run the chosen tool
#   4. Add the result to the conversation history
#   5. Repeat until Gemini calls "final_answer"
#
# Max 6 iterations prevents infinite loops.

SYSTEM_PROMPT = """You are a research agent. You answer questions by searching
the web and querying your knowledge base.

You have exactly three tools. At each step you must respond with ONLY a JSON
object — no extra text, no markdown fences — like this:
{"thought": "...", "tool": "tool_name", "args": {"param": "value"}}

Tools:
  search_web(query: str)
    → Search the web for a topic and store results in the knowledge base.
    → Use this first when you need information you don't have yet.

  query_knowledge_base(question: str)
    → Retrieve the most relevant content from stored articles.
    → Use this after searching, to find specific chunks to answer with.

  final_answer(answer: str)
    → Give the final cited answer to the user. Include [Source: ...] citations.
    → YOU MUST call this tool once you have queried the knowledge base.
    → Do NOT keep searching if you already have retrieved content.

Rules:
  - Search once or twice at most, then query, then answer.
  - After your FIRST query_knowledge_base call, synthesise what you have
    and call final_answer — do not keep looping.
  - Keep answers clear, specific, and cited.
  - Output ONLY the JSON object — nothing else."""


def run_agent(user_question: str) -> str:
    print("\n" + "=" * 60)
    print(f"  AGENT STARTING")
    print(f"  Question: {user_question}")
    print("=" * 60)

    # conversation history — grows with each step
    history = f"User question: {user_question}\n\nSteps taken so far:\n"

    for step in range(1, 9):
        print(f"\n  ── Step {step} ──")

        steps_left = 8 - step
        urgency = (
            " YOU MUST call final_answer NOW with what you have."
            if steps_left <= 2 else
            f" ({steps_left} steps remaining — plan to call final_answer soon.)"
        )
        prompt = f"{SYSTEM_PROMPT}\n\n{history}\nWhat do you do next?{urgency}"

        # Ask Gemini what to do
        raw_response = call_gemini(prompt)

        # Gemini should return pure JSON — strip any accidental formatting
        clean = raw_response.strip().strip("```json").strip("```").strip()

        try:
            decision = json.loads(clean)
        except json.JSONDecodeError:
            # If Gemini didn't return valid JSON, try to extract it
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            decision = json.loads(clean[start:end])

        thought    = decision.get("thought", "")
        tool_name  = decision.get("tool", "")
        args       = decision.get("args", {})

        print(f"     💭 Thought: {thought}")
        print(f"     🔧 Tool:    {tool_name}")

        if tool_name not in TOOLS:
            print(f"     ⚠️  Unknown tool: {tool_name}")
            break

        # Run the chosen tool
        result = TOOLS[tool_name](**args)

        # If agent chose final_answer, we're done
        if tool_name == "final_answer":
            print("\n" + "=" * 60)
            print("  FINAL ANSWER")
            print("=" * 60)
            print(f"\n{result}\n")
            return result

        print(f"     📋 Result: {result[:120]}...")

        # Add this step to history so Gemini remembers it.
        # Trim the result to 400 chars — Gemini only needs to know
        # what happened, not the full article content in the history.
        result_summary = result[:400] + "..." if len(result) > 400 else result
        history += (
            f"\nStep {step}:\n"
            f"  Thought: {thought}\n"
            f"  Tool: {tool_name}({args})\n"
            f"  Result: {result_summary}\n"
        )

    return "Agent reached maximum steps without a final answer."


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent("What are the main differences between RAG and fine-tuning an LLM, and when should you use each?")
