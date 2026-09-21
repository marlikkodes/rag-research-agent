"""
providers.py — where the model calls actually go.



This project ran entirely on Gemini until its prepaid balance emptied, at
which point every request returned RESOURCE_EXHAUSTED and the app stopped
working. Nothing about the agent was wrong; it just had exactly one provider.

So there are two now, tried in order, each skipped when its key is absent.
Adding a third is an environment variable rather than a code change.

--- chat and embeddings are not the same problem ---

Swapping the chat model changes the wording of an answer. Swapping the
EMBEDDING model invalidates every vector already stored: the numbers mean
different things and are usually a different length. Querying a store built
with one model using vectors from another does not raise an error — it
quietly returns the wrong chunks, and the answer on top of them looks fine.

Which is why the collection name carries the embedding model. Change the
model and you get a new, empty collection instead of a corrupted old one.
"""

from __future__ import annotations

import json
import os

import requests


def _key(name: str) -> str | None:
    """A key from the environment, or from Streamlit's secret store.

    Reading os.getenv alone is not enough. On Streamlit Cloud, secrets live in
    st.secrets, and relying on them also appearing as environment variables is
    an assumption — one that produced "No provider configured" on a deployment
    where the key was set correctly the whole time.

    streamlit is imported lazily and inside a try, because this module is also
    used by scripts that have nothing to do with Streamlit and should not
    require it installed.
    """
    # Stripped, always. A key pasted into a secrets box or an .env file
    # routinely carries a trailing newline or space, and it goes straight into
    # an Authorization header — where "Bearer key\n" is rejected as 403
    # Forbidden, indistinguishable from a genuinely invalid key. The character
    # doing the damage is invisible in every UI that shows it back to you.
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st
        secret = st.secrets.get(name)
        return secret.strip() if secret and secret.strip() else None
    except Exception:
        return None


NVIDIA_API_KEY = _key("NVIDIA_API_KEY")
MISTRAL_API_KEY = _key("MISTRAL_API_KEY")
GEMINI_API_KEY = _key("GEMINI_API_KEY")

# Sent to the outside world as part of the collection name, so it must stay
# short and filesystem-safe.
if NVIDIA_API_KEY:
    EMBED_MODEL = "nv-embedqa-e5-v5"
elif MISTRAL_API_KEY:
    EMBED_MODEL = "mistral-embed"
else:
    EMBED_MODEL = "gemini-embedding-2"


class NoProviderConfigured(RuntimeError):
    """Raised when no key was found anywhere.

    The message names what was looked for and where, because the version that
    just said "no provider configured" sent someone checking a secret that was
    already set correctly. It reports presence only — never a key, never a
    prefix, never a length.
    """

    def __init__(self) -> None:
        looked = {
            "NVIDIA_API_KEY": bool(NVIDIA_API_KEY),
            "MISTRAL_API_KEY": bool(MISTRAL_API_KEY),
            "GEMINI_API_KEY": bool(GEMINI_API_KEY),
        }
        found = ", ".join(k for k, v in looked.items() if v) or "none"
        super().__init__(
            "No model provider configured.\n"
            f"  checked (env, then st.secrets): {', '.join(looked)}\n"
            f"  found: {found}\n"
            "  Names are case-sensitive. Set NVIDIA_API_KEY or MISTRAL_API_KEY "
            "— both are free and need no card."
        )


# ─────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────

def _nvidia_chat(prompt: str) -> str:
    """NVIDIA NIM, OpenAI-compatible, streamed.

    Streaming is not a nicety here — it is the fix. A non-streamed request
    returns the whole completion in one response, so the read timeout has to
    cover the entire generation. On a busy free endpoint that regularly exceeds
    90 seconds, and the failure arrives as "Read timed out" with no clue that
    the request was fine and merely slow.

    Streamed, the read timeout only has to cover the gap between chunks, which
    is a fraction of a second once generation starts. The same call in another
    project of mine has streamed against this endpoint for weeks without a
    timeout; this one did not, and that was the whole difference.

    Models are tried largest first and fall through on timeout, so a slow
    endpoint degrades to a faster, smaller model rather than to nothing.
    """
    last = ""
    for model in (
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "meta/llama-3.1-8b-instruct",
    ):
        try:
            with requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.01,
                    "max_tokens": 1024,
                    "stream": True,
                },
                # (connect, read). The read budget now covers the pause between
                # chunks rather than the whole generation.
                timeout=(10, 60),
                stream=True,
            ) as r:
                if r.status_code != 200:
                    last = f"{model} -> HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code in (401, 403):
                        break
                    continue

                parts: list[str] = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue    # a partial frame; the next chunk completes it
                    if delta.get("content"):
                        parts.append(delta["content"])

                text = "".join(parts).strip()
                if text:
                    return text
                last = f"{model} -> empty stream"
        except requests.exceptions.Timeout:
            # Slow, not broken. The next model is smaller and faster.
            last = f"{model} -> timed out between chunks"
        except requests.exceptions.RequestException as e:
            last = f"{model} -> {type(e).__name__}: {str(e)[:120]}"

    raise RuntimeError(last or "nvidia: no response")


def _mistral_chat(prompt: str) -> str:
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _gemini_chat(prompt: str) -> str:
    # Several model names because Google retires them without much notice, and
    # a 404 on one should not take the agent down.
    for model in ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"):
        r = requests.post(
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            timeout=90,
        )
        # 404 is a retired model, 503 is a busy one. Both are worth trying the
        # next name for. Anything else is about this request, not this model.
        if r.status_code in (404, 503):
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("no Gemini model answered")


def call_llm(prompt: str) -> str:
    """First provider with a key that answers. Errors only when none do."""
    errors: list[str] = []

    for name, key, fn in (
        ("nvidia", NVIDIA_API_KEY, _nvidia_chat),
        ("mistral", MISTRAL_API_KEY, _mistral_chat),
        ("gemini", GEMINI_API_KEY, _gemini_chat),
    ):
        if not key:
            continue
        try:
            return fn(prompt)
        except Exception as e:  # noqa: BLE001 — the next provider is the handler
            errors.append(f"{name}: {e}")

    if not errors:
        raise NoProviderConfigured()
    raise RuntimeError("Every provider failed — " + "; ".join(errors))


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────

def embed_text(text: str, kind: str = "passage") -> list[float]:
    """
    One vector for one piece of text.

    `kind` is "passage" for text being stored and "query" for a question being
    asked. NVIDIA's retrieval embedders are trained asymmetrically and expect
    to be told which — passing the wrong one costs real retrieval quality and
    reports no error at all. The other providers ignore it.

    No failover here, deliberately. Falling back to a different embedding
    model mid-collection would put vectors of two different shapes and
    meanings into the same store, and the resulting nonsense would be silent.
    Better to fail loudly than to retrieve confidently wrong chunks.
    """
    if NVIDIA_API_KEY:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json={
                "model": "nvidia/nv-embedqa-e5-v5",
                "input": [text],
                "input_type": kind,
                "encoding_format": "float",
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    if MISTRAL_API_KEY:
        r = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-embed", "input": [text]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    if GEMINI_API_KEY:
        r = requests.post(
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}",
            json={"content": {"parts": [{"text": text}]}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]["values"]

    raise NoProviderConfigured()
