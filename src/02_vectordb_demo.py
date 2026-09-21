"""
STEP 2 — Vector Database Demo (ChromaDB)
=========================================
What this file does:
  - Takes 5 fake "research article" chunks
  - Embeds each one and stores it in ChromaDB
  - Then asks two different questions
  - ChromaDB finds and returns the most relevant chunks

This is the RETRIEVAL part of RAG.
The LLM hasn't appeared yet — this is purely about
finding the right information before we answer anything.

Run it with:
  venv/bin/python src/02_vectordb_demo.py
"""

import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ─────────────────────────────────────────────
# SAME embed_text function from Step 1
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Convert a piece of text into a list of numbers (embedding)."""
    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    )
    payload = {"content": {"parts": [{"text": text}]}}
    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()["embedding"]["values"]


# ─────────────────────────────────────────────
# SET UP CHROMADB
# ─────────────────────────────────────────────
# ChromaDB is a local database that runs on your machine.
# "persistent" means it saves to disk — if you restart
# your script, the data is still there.
#
# A "collection" is like a table in a normal database.
# We're creating one called "research_demo".

print("\n" + "=" * 55)
print("  STEP 2: VECTOR DATABASE DEMO")
print("=" * 55)

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_demo")
client = chromadb.PersistentClient(path=db_path)

# Delete old collection if it exists (so we start fresh each run)
try:
    client.delete_collection("research_demo")
except Exception:
    pass

collection = client.create_collection(name="research_demo")
print("\n✅ ChromaDB collection created\n")


# ─────────────────────────────────────────────
# THE "DOCUMENTS" WE'RE STORING
# ─────────────────────────────────────────────
# In the real app these will be paragraphs from
# web articles found by Tavily. Here we fake them.
#
# Each chunk has:
#   - an id     (unique label)
#   - text      (the actual content)
#   - metadata  (where it came from — useful for citations)

chunks = [
    {
        "id": "chunk_1",
        "text": "Transformer neural networks use self-attention mechanisms "
                "to process sequences. Unlike RNNs, they handle all tokens "
                "simultaneously, making them much faster to train.",
        "source": "Vaswani et al. (2017) — Attention Is All You Need",
    },
    {
        "id": "chunk_2",
        "text": "BERT is a transformer model pre-trained on masked language "
                "modelling. It learns bidirectional context, meaning it reads "
                "text from both left and right at the same time.",
        "source": "Devlin et al. (2019) — BERT",
    },
    {
        "id": "chunk_3",
        "text": "GPT models are trained to predict the next token in a sequence. "
                "They use a decoder-only transformer architecture and are "
                "particularly good at generating coherent long-form text.",
        "source": "Radford et al. (2018) — Improving Language Understanding",
    },
    {
        "id": "chunk_4",
        "text": "Photosynthesis is the process by which plants convert sunlight "
                "into glucose using carbon dioxide and water. Chlorophyll in "
                "leaves absorbs light energy to drive this reaction.",
        "source": "Biology Textbook — Chapter 5",
    },
    {
        "id": "chunk_5",
        "text": "The FIFA World Cup is held every four years. The 2026 edition "
                "will be hosted across the United States, Canada, and Mexico "
                "with 48 teams competing for the first time.",
        "source": "FIFA Official Website",
    },
]

# ─────────────────────────────────────────────
# EMBED AND STORE EACH CHUNK
# ─────────────────────────────────────────────
# For each chunk we:
#   1. Call Gemini to get its 3072-number fingerprint
#   2. Store the fingerprint + original text + metadata in ChromaDB

print("📡 Embedding and storing 5 chunks...\n")

for chunk in chunks:
    print(f'  Embedding: "{chunk["text"][:60]}..."')
    embedding = embed_text(chunk["text"])

    collection.add(
        ids=[chunk["id"]],
        embeddings=[embedding],          # the 3072 numbers
        documents=[chunk["text"]],       # the original text
        metadatas=[{"source": chunk["source"]}],  # where it came from
    )

print(f"\n✅ {len(chunks)} chunks stored in ChromaDB\n")


# ─────────────────────────────────────────────
# NOW QUERY — ask a question and find relevant chunks
# ─────────────────────────────────────────────
# Steps:
#   1. Embed the question (same way as the chunks)
#   2. ChromaDB compares the question's fingerprint
#      against all stored fingerprints
#   3. Returns the top N most similar chunks

def query_collection(question: str, n_results: int = 2):
    print(f'\n🔍 Question: "{question}"')
    print("-" * 55)

    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # distances: lower = more similar (ChromaDB uses L2 distance by default)
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        similarity = round(1 - dist / 2, 4)   # rough conversion to 0–1 scale
        print(f"\n  Result #{rank}  (similarity ≈ {similarity})")
        print(f"  Source: {meta['source']}")
        print(f"  Text:   {doc[:120]}...")

    print()


# Run two very different questions
query_collection("How do transformer models process text sequences?")
query_collection("Tell me about football tournaments")

print("=" * 55)
print("  WHAT JUST HAPPENED")
print("=" * 55)
print("""
  For question 1 (transformers), ChromaDB should have
  returned chunks 1 and 2 — even though the question
  doesn't use the same words as the chunks.

  For question 2 (football), it should return chunk 5.

  Chunk 4 (photosynthesis) should never appear —
  it's semantically far from both questions.

  In the real app, these "results" get passed to Gemini
  with your original question, and Gemini answers using
  ONLY these chunks — with citations.
""")
