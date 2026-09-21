## Ownership
- Owner: Marvin Marlik (telegram @solidity pope)
- Email: marlikkodes@gmail.com

## Ownership

# RAG Research Agent

**An agent that decides its own next step: search the web, query what it has already
stored, or stop and answer.**

Not a chatbot with retrieval bolted on. The model is given three tools and a running
history, and at each step it chooses what to do next. The interesting part isn't
getting it to act — it's getting it to **stop**.

![Stack](https://img.shields.io/badge/Python-3.11+-blue) ![Stack](https://img.shields.io/badge/Mistral-API-orange) ![Stack](https://img.shields.io/badge/ChromaDB-vector%20store-green) ![Stack](https://img.shields.io/badge/Tavily-web%20search-purple)

---

## The loop

Every step, the agent gets the system prompt plus everything it has done so far, and
must reply with a single JSON object:

```json
{"thought": "I need current sources on this", "tool": "search_web", "args": {"query": "..."}}
```

The runner parses that, executes the named tool, appends the result to the history,
and asks again. It repeats until the model calls `final_answer`.

| Tool | What it does |
|---|---|
| `search_web` | Tavily fetches live articles, chunks them, embeds them, stores them in ChromaDB |
| `query_knowledge_base` | Semantic search over what has been stored, returns the most relevant chunks |
| `final_answer` | Terminates the loop and returns a cited answer |

## Stopping is the hard part

An agent that can search will happily search forever. Three things prevent that:

**A hard step ceiling.** The loop runs at most 8 iterations. If it exits without a
final answer, it says so rather than pretending.

**Escalating pressure in the prompt.** The agent is told how many steps remain, and
the wording sharpens as they run out:

```python
urgency = (
    " YOU MUST call final_answer NOW with what you have."
    if steps_left <= 2 else
    f" ({steps_left} steps remaining — plan to call final_answer soon.)"
)
```

**An explicit stop rule.** The system prompt states that after the first
`query_knowledge_base` call the agent must synthesise and answer, not keep looping.

Without these it reliably burned every step re-searching variations of the same query.
For anything automated, that bounded behaviour matters more than adding capability.

## Other things that survive contact with reality

**JSON parsing that expects failure.** Models emit markdown fences around JSON even
when told not to. The parser strips fences, and if that still fails it extracts the
substring between the first `{` and last `}` rather than crashing the run.

**Model fallback.** Requests try `gemini-3.5-flash`, then `gemini-3.1-flash-lite`,
then `gemini-flash-latest`, so a single unavailable model doesn't kill the run.

**Unknown tools exit cleanly.** If the model invents a tool name, the loop breaks
instead of raising.

**History is trimmed.** Tool results are cut to 400 characters before going into the
history — the agent needs to know *what happened*, not re-read whole articles every
step. Without this the context grows until the run fails.

## The UI

`app.py` streams the agent's reasoning live rather than showing a spinner: each step's
thought, the tool chosen, and the collapsible raw result. You watch it decide.

`run_agent()` in `agent_core.py` is a generator yielding typed updates
(`step` / `result` / `final` / `error`), which is what makes that possible. The CLI
version in `src/05_agent.py` is the same logic printing to stdout.

## Layout

The numbered files are a build-up, each runnable on its own:

```
src/01_embeddings_demo.py    what an embedding is, cosine similarity by hand
src/02_vectordb_demo.py      storing and querying vectors in ChromaDB
src/03_rag_pipeline.py       retrieval + generation end to end
src/04_search_and_store.py   live web search into the vector store
src/05_agent.py              the agent loop, CLI version

src/agent_core.py            streaming version used by the UI
app.py                       Streamlit interface
```

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your keys
streamlit run app.py
```

Needs one model key and a [Tavily key](https://tavily.com). Either works, and
both are free without a card:

| | |
|---|---|
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) — tried first |
| `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai) — phone verification |

`GEMINI_API_KEY` still works as a fallback if you have one. The agent uses the
first provider with a key present and only fails when none answer — this
project ran on Gemini alone until its balance emptied, at which point every
request returned RESOURCE_EXHAUSTED and nothing worked. Nothing about the
agent was wrong; it just had exactly one provider.

## Writing the loop, then rebuilding it

The hand-written version is a `for` loop with the decision, the tool call, the
step budget and the transcript interleaved. Everything is visible in the order
it happens, which is genuinely easier to read — and it also means those four
concerns cannot be changed independently.

The LangGraph version separates them into nodes over a shared state:

```
decide ──► act ──► record ──► decide ──► ... ──► end
```

`decide` asks the model what to do next, `act` runs the chosen tool, and a
conditional edge sends control back or ends the run. The step budget is a
field in the state rather than a loop counter, so nothing has to remember to
decrement it.

**Is the framework worth it here? For a loop this small, not obviously.** It
is more files, more concepts and a dependency that moves quickly, in exchange
for structure the eighty-line version did not need.

What it buys is what comes next. The agent's whole memory is now a value
rather than local variables inside one function — so it can be inspected,
logged, saved between runs, or handed to a second agent. Persistence,
resuming a half-finished research session, and a human approval step before an
expensive tool are all hard to bolt onto a `for` loop and close to free on a
graph.

Both are in the repository because the comparison is the interesting part.

The graph has tests the loop never had, and they run without a model:

```bash
venv/bin/python tests/test_agent_graph.py
```

Six checks on control flow — that a tool result loops back for another
decision, that `final_answer` ends the run, that an invented tool name is an
error rather than a crash, that a failing tool is reported to the model
instead of ending the run, and that the step budget stops an agent which never
decides to finish. Stubbing the model is what makes them fast, deterministic
and free.

## Known limits

- Retrieval quality isn't measured. There's no ground-truth set here, so treat answer
  quality as unverified. (I built [RAG Bench](https://github.com/aamirabbas858/rag-bench)
  separately to score exactly that.)
- The knowledge base persists between runs, so a later question can be answered from
  an earlier question's articles. Useful in practice, but it means runs aren't isolated.
- One agent, three tools, no parallel execution and no human approval step.
- Chunking is fixed-size with overlap, not semantic.
- **Switching embedding model invalidates the store.** Vectors from two models
  have different lengths and different meanings, and querying across them does
  not error — it silently returns the wrong chunks and the answer on top looks
  fine. The collection name carries the model for that reason, so changing it
  gives a fresh collection rather than a corrupted one.

## Stack

Python · LangGraph · NVIDIA NIM (`llama-3.3-70b` + `nv-embedqa-e5-v5`),
Mistral and Gemini as fallbacks · ChromaDB · Tavily · Streamlit

Two implementations of the same agent, kept side by side:

| | |
|---|---|
| `src/agent_manual.py` | the ReAct loop written by hand, one function |
| `src/agent_graph.py` | the same loop as a LangGraph state machine |

They yield identical step dictionaries, so the UI cannot tell them apart.
Switch with `AGENT_IMPL=manual` or `AGENT_IMPL=graph` (the default).

---

Built by [Abbas Aamir](https://www.linkedin.com/in/abbas-aamir-474969353/) — CS undergraduate in Berlin.
