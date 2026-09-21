"""
STEP 3 — The Full RAG Pipeline
================================
What this file does:
  - Stores 6 chunks in ChromaDB (same as Step 2)
  - Takes a question from the user
  - Retrieves the 3 most relevant chunks
  - Sends question + chunks to Gemini
  - Gemini answers using ONLY those chunks, with citations

This is Retrieval-Augmented Generation in its complete form.
R = ChromaDB finds relevant chunks
A = we augment the question with those chunks
G = Gemini generates an answer from them

Run it with:
  venv/bin/python src/03_rag_pipeline.py
"""

import os
import json
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ─────────────────────────────────────────────
# HELPER: embed text (same as before)
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    )
    response = requests.post(
        url,
        json={"content": {"parts": [{"text": text}]}},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


# ─────────────────────────────────────────────
# HELPER: ask Gemini a question
# ─────────────────────────────────────────────
# We call the Gemini generate API directly (same
# approach that worked in Wayfare).
# We pass a SYSTEM PROMPT and a USER PROMPT.
#
# System prompt = the rules Gemini must follow
# User prompt   = the actual question + the chunks

def ask_gemini(system_prompt: str, user_prompt: str) -> str:
    models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 503:
            continue   # model overloaded, try next
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("All Gemini models temporarily unavailable.")


# ─────────────────────────────────────────────
# SET UP CHROMADB
# ─────────────────────────────────────────────

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_rag")
client = chromadb.PersistentClient(path=db_path)

try:
    client.delete_collection("rag_demo")
except Exception:
    pass

collection = client.create_collection(name="rag_demo")


# ─────────────────────────────────────────────
# KNOWLEDGE BASE: 6 chunks on different topics
# ─────────────────────────────────────────────

chunks = [
    {
        "id": "c1",
        "text": "Transformer models use self-attention to weigh the importance "
                "of each word relative to all others in a sequence. This allows "
                "them to capture long-range dependencies that RNNs struggle with.",
        "source": "Vaswani et al. (2017)",
    },
    {
        "id": "c2",
        "text": "Large language models like GPT-4 are trained on trillions of "
                "tokens of text from the internet. The training process uses "
                "next-token prediction as the objective function.",
        "source": "OpenAI Technical Report (2023)",
    },
    {
        "id": "c3",
        "text": "Retrieval-Augmented Generation (RAG) combines a retrieval "
                "system with a language model. The retrieval system finds "
                "relevant documents, and the LLM generates an answer grounded "
                "in those documents rather than relying on memorised knowledge.",
        "source": "Lewis et al. (2020) — RAG Paper",
    },
    {
        "id": "c4",
        "text": "Vector databases store high-dimensional embeddings and support "
                "approximate nearest-neighbour search. ChromaDB, Pinecone, and "
                "Weaviate are popular choices. They are the storage layer in "
                "most RAG systems.",
        "source": "AI Engineering Handbook (2024)",
    },
    {
        "id": "c5",
        "text": "The Amazon rainforest produces about 20% of the world's oxygen "
                "and is home to 10% of all species on Earth. Deforestation "
                "threatens both biodiversity and global climate stability.",
        "source": "WWF Environmental Report (2023)",
    },
    {
        "id": "c6",
        "text": "Berlin has one of the fastest-growing tech startup ecosystems "
                "in Europe. Companies like Zalando, Delivery Hero, and N26 "
                "were all founded there. The city attracts engineering talent "
                "from across the EU.",
        "source": "Startup Genome Report (2024)",
    },
]

print("\n" + "=" * 60)
print("  STEP 3: FULL RAG PIPELINE")
print("=" * 60)
print("\n📥 Building knowledge base (embedding + storing chunks)...\n")

for chunk in chunks:
    embedding = embed_text(chunk["text"])
    collection.add(
        ids=[chunk["id"]],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}],
    )
    print(f'  ✅ Stored: "{chunk["text"][:55]}..."')


# ─────────────────────────────────────────────
# THE RAG FUNCTION
# ─────────────────────────────────────────────
# This is the complete pipeline:
#   1. Embed the question
#   2. Retrieve top 3 chunks from ChromaDB
#   3. Build a prompt that includes those chunks
#   4. Ask Gemini to answer using only those chunks

def rag_answer(question: str) -> None:
    print("\n" + "─" * 60)
    print(f"❓ Question: {question}")
    print("─" * 60)

    # STEP R — Retrieve
    question_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas"],
    )
    retrieved_chunks  = results["documents"][0]
    retrieved_sources = results["metadatas"][0]

    print("\n📚 Retrieved chunks:")
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_sources), 1):
        print(f"  [{i}] {meta['source']}: \"{chunk[:70]}...\"")

    # STEP A — Augment: build the prompt with context
    # We number each source so Gemini can reference them in its answer
    context_block = ""
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_sources), 1):
        context_block += f"\n[Source {i}: {meta['source']}]\n{chunk}\n"

    system_prompt = (
        "You are a precise research assistant. "
        "Answer the user's question using ONLY the provided sources. "
        "Cite sources by their label, e.g. [Source 1]. "
        "If the answer is not in the sources, say: "
        "'I could not find this in the available documents.'"
    )

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    # STEP G — Generate
    print("\n🤖 Gemini's answer (grounded in retrieved chunks):\n")
    answer = ask_gemini(system_prompt, user_prompt)
    print(f"  {answer}")


# ─────────────────────────────────────────────
# RUN THREE QUESTIONS
# ─────────────────────────────────────────────
# Question 1: answerable from the knowledge base
# Question 2: also answerable
# Question 3: NOT in the knowledge base — should get "I don't know"

rag_answer("What is RAG and why is it useful?")
rag_answer("What kind of tech companies are based in Berlin?")
rag_answer("Who won the 2022 FIFA World Cup?")   # not in our knowledge base

print("\n" + "=" * 60)
print("  WHAT YOU JUST BUILT")
print("=" * 60)
print("""
  Question 1 → Gemini answered using chunk c3 and c4
  Question 2 → Gemini answered using chunk c6
  Question 3 → Gemini said it couldn't find the answer
               (because we never stored that fact)

  This is the hallucination prevention working.
  The model is not allowed to answer from memory.
  It can only use what was retrieved.

  Next step: instead of us manually typing these chunks,
  the AGENT will search the web and fill the database
  automatically based on any topic you give it.
""")
