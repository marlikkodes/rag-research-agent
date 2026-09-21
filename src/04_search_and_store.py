"""
STEP 4 — Web Search + Store in ChromaDB
=========================================
What this file does:
  - Takes a research topic from you
  - Searches the web using Tavily (same API used in Wayfare)
  - Takes the article content from the search results
  - Splits long articles into smaller chunks
  - Embeds each chunk and stores it in ChromaDB

After running this, you have a knowledge base built
from real, live web content — not fake text we typed.

In the final agent, this whole process will run
automatically when you give it a research topic.

Run it with:
  venv/bin/python src/04_search_and_store.py
"""

import os
import requests
import chromadb
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)


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
# HELPER: split text into chunks
# ─────────────────────────────────────────────
# Why do we split?
#
# Imagine a 10-page article. If we embed the whole thing
# as one vector, that vector is a blurry average of all
# 10 pages. When someone asks about paragraph 3,
# the embedding won't specifically represent it.
#
# Instead we split into ~500 character chunks with
# 100 character overlap. The overlap ensures that a
# sentence split across two chunks doesn't lose context.
#
# Example of overlap:
#   Chunk 1: "...the model uses attention. This allows it to..."
#   Chunk 2: "This allows it to process long sequences..."
#             ↑ repeated from chunk 1 — keeps context intact

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():           # skip empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap   # step forward, keeping some overlap
    return chunks


# ─────────────────────────────────────────────
# SET UP CHROMADB
# ─────────────────────────────────────────────

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_main")
db_client = chromadb.PersistentClient(path=db_path)

# We use get_or_create so the collection persists
# across multiple runs — new searches ADD to it,
# they don't overwrite it
collection = db_client.get_or_create_collection(name="research_knowledge_base")


# ─────────────────────────────────────────────
# MAIN FUNCTION: search → chunk → embed → store
# ─────────────────────────────────────────────

def search_and_store(topic: str, num_results: int = 5) -> int:
    """
    Search the web for a topic, split the results into chunks,
    embed each chunk, and store it in ChromaDB.
    Returns the number of chunks stored.
    """
    print(f'\n🔍 Searching the web for: "{topic}"')
    print("─" * 55)

    # Tavily returns clean article text — no HTML tags,
    # no nav menus, just the actual content
    results = tavily.search(
        query=topic,
        max_results=num_results,
        search_depth="basic",
    )

    total_chunks_stored = 0

    for i, result in enumerate(results["results"]):
        title   = result.get("title", "No title")
        url     = result.get("url", "")
        content = result.get("content", "")

        if not content.strip():
            continue

        print(f"\n  📄 Article {i+1}: {title}")
        print(f"     URL: {url}")
        print(f"     Content length: {len(content)} characters")

        # Split this article into chunks
        chunks = split_into_chunks(content, chunk_size=500, overlap=100)
        print(f"     Split into {len(chunks)} chunks")

        # Embed and store each chunk
        for j, chunk_text in enumerate(chunks):
            chunk_id = f"{topic[:20]}_{i}_{j}".replace(" ", "_")

            # Check if this chunk ID already exists (avoid duplicates)
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            embedding = embed_text(chunk_text)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    "source": title,
                    "url": url,
                    "topic": topic,
                }],
            )
            total_chunks_stored += 1

        print(f"     ✅ {len(chunks)} chunks stored")

    return total_chunks_stored


# ─────────────────────────────────────────────
# QUERY FUNCTION (same as Step 3 but using
# the real knowledge base)
# ─────────────────────────────────────────────

def query_knowledge_base(question: str, n_results: int = 3) -> None:
    print(f'\n❓ Question: "{question}"')
    print("─" * 55)

    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not docs:
        print("  No relevant chunks found.")
        return

    print("\n  Top chunks retrieved:")
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        print(f"\n  [{i}] Source: {meta['source']}")
        print(f"       URL: {meta['url']}")
        print(f"       Text: \"{doc[:120]}...\"")


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 4: WEB SEARCH + STORE IN CHROMADB")
print("=" * 55)

topic = "how does RAG retrieval augmented generation work"

total = search_and_store(topic, num_results=4)

print(f"\n\n✅ Done — {total} new chunks stored in ChromaDB")
print(f"   Total chunks in knowledge base: {collection.count()}")

# Now query it with a real question
print("\n" + "=" * 55)
print("  QUERYING THE REAL KNOWLEDGE BASE")
print("=" * 55)

query_knowledge_base("What is the difference between RAG and fine-tuning?")
query_knowledge_base("What vector databases are commonly used in RAG systems?")

print("""
Next step: instead of hardcoding the topic,
the AGENT will decide what to search for based
on your question — and answer using what it finds.
""")
