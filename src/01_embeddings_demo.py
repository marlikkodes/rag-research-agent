"""
STEP 1 — Embeddings Demo
========================
What this file does:
  - Takes 4 sentences
  - Converts each one into a list of numbers (an embedding)
  - Measures how similar those sentences are to each other
  - Prints the results so you can SEE the concept working

Run it with:
  venv/bin/python src/01_embeddings_demo.py
"""

import os
import json
import math
import requests
from dotenv import load_dotenv

# Load the API keys from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ─────────────────────────────────────────────
# FUNCTION 1: embed_text
# ─────────────────────────────────────────────
# This calls the Gemini API and asks it to convert
# one sentence into a list of numbers.
# The model we use is "text-embedding-004" — a model
# trained specifically to produce embeddings.
# It outputs 3072 numbers per sentence.

def embed_text(text: str) -> list[float]:
    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()["embedding"]["values"]


# ─────────────────────────────────────────────
# FUNCTION 2: cosine_similarity
# ─────────────────────────────────────────────
# Two embeddings are lists of numbers (vectors).
# Cosine similarity measures the ANGLE between them.
#
# Score = 1.0  → identical meaning
# Score = 0.8+ → very similar
# Score = 0.5  → loosely related
# Score = 0.0  → completely unrelated
#
# The formula: dot(A, B) / (magnitude(A) * magnitude(B))
# You don't need to memorise this — it's a standard
# maths tool for comparing vectors.

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────
# THE SENTENCES WE'RE TESTING
# ─────────────────────────────────────────────
# Pair A: sentence 0 and 1 mean basically the same thing
#         (dog/puppy, chased/ran after, cat/kitten)
# Pair B: sentence 2 is completely unrelated (finance)
# Pair C: sentence 3 is about ML — related to what we're building

sentences = [
    "The dog chased the cat up the tree",       # 0
    "A puppy ran after a kitten",               # 1 — similar to 0
    "Interest rates rose sharply in Q3",        # 2 — unrelated
    "Machine learning models learn from data",  # 3
]


# ─────────────────────────────────────────────
# MAIN: embed everything, then compare
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 1: EMBEDDINGS DEMO")
print("=" * 55)

print("\n📡 Calling Gemini to embed 4 sentences...\n")

embeddings = []
for i, sentence in enumerate(sentences):
    vec = embed_text(sentence)
    embeddings.append(vec)
    # Show only the first 6 numbers — each embedding is 768 numbers long
    preview = [round(n, 4) for n in vec[:6]]
    print(f'  [{i}] "{sentence}"')
    print(f'       → [{", ".join(str(n) for n in preview)}, ...]')
    print(f'       → Total numbers in this vector: {len(vec)}\n')

print("=" * 55)
print("  SIMILARITY SCORES")
print("=" * 55)
print()
print("  (1.0 = identical meaning, 0.0 = completely unrelated)")
print()

# Compare every pair of sentences
pairs = [
    (0, 1, "dog/cat  vs  puppy/kitten  — same meaning, different words"),
    (0, 2, "dog/cat  vs  interest rates — completely unrelated"),
    (0, 3, "dog/cat  vs  machine learning — unrelated"),
    (1, 3, "puppy/kitten  vs  machine learning — unrelated"),
]

for i, j, label in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    bar = "█" * int(score * 20)
    print(f"  Sentences {i} & {j}: {score:.4f}  {bar}")
    print(f"  {label}")
    print()

print("=" * 55)
print("  WHAT THIS PROVES")
print("=" * 55)
print("""
  Sentences 0 and 1 should have the HIGHEST score
  even though they share zero words in common.

  This is how ChromaDB will later find relevant
  document chunks — not by matching keywords,
  but by comparing these vectors.

  When you ask "how do neural networks work?",
  ChromaDB finds chunks about "deep learning" and
  "gradient descent" because their vectors are close.
""")
