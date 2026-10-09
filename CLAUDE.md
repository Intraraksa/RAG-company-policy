# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RAG microservice backend built with FastAPI, using AWS Bedrock for both the LLM (via the Bedrock
Runtime `converse` API) and document retrieval (via a Bedrock Knowledge Base, using the Bedrock
Agent Runtime `retrieve` API). The retrieve → generate pipeline is implemented as a LangGraph
graph. This is currently a single FastAPI service — splitting into separate services (e.g.
ingestion vs. query) is a possible future step, not the current layout.

## Agents

Defined in `.claude/agents/`:

- **ai-backend-engineer** (`.claude/agents/ai-backend-engineer.md`) — implements RAG/LangGraph
  pipeline changes, FastAPI endpoints, Bedrock integrations, and tests. Uses the `fastapi` and
  `langgraph-fundamentals` skills and the Context7 MCP for current library docs.
- **code-review-agent** (`.claude/agents/code-review-agent.md`) — read-only reviewer; reports
  ranked findings and runs tests/lint. Uses the `code-review` skill.
- **supervisor-agent** (`.claude/agents/supervisor-agent.md`) — coordinates multi-step features by
  breaking them into tasks, delegating implementation to `ai-backend-engineer` and review to
  `code-review-agent`, and iterating until everything is done and approved. Use for larger changes
  that need build-and-review coordination rather than a single pass.

## Skills

Used by the agents above (see each agent's frontmatter):

- **fastapi** — FastAPI/Pydantic conventions: endpoints, dependencies, streaming/SSE.
- **langgraph-fundamentals** — StateGraph, state schemas, nodes/edges, Command/Send, streaming,
  error handling.
- **code-review** — standards + spec review process followed by `code-review-agent`.
- **context7-mcp** — fetch current docs for libraries/frameworks via the Context7 MCP instead of
  relying on training data.

## Tech stack

- **FastAPI** — HTTP layer / API framework (`app/main.py`, `app/api/routes/`)
- **LangGraph** — orchestrates the RAG pipeline as a graph of nodes (`app/services/rag.py`)
- **AWS Bedrock** (via `boto3`) — model inference (Bedrock Runtime `converse`) and retrieval
  (Bedrock Agent Runtime `retrieve` against a Knowledge Base)
- **Pydantic / pydantic-settings** — request/response schemas (`app/models/schemas.py`) and
  env-based config (`app/core/config.py`)
- **uv** — dependency management, virtualenv, and task running (`uv sync`, `uv run ...`)
- **pytest** — test suite, with service-boundary mocking (see Testing conventions below)
- **ruff** — linting
- **Docker** — containerized runtime (see `Dockerfile`), built on the `uv`-managed environment
- **Jupyter** (`ipykernel`, dev only) — runs the architecture notebook in `notebooks/`

## Commands

This project uses `uv` for dependency management and running everything (venv, deps, lockfile).

```bash
uv sync                      # install/update dependencies (reads pyproject.toml / uv.lock)
uv run uvicorn app.main:app --reload   # run the API locally at http://127.0.0.1:8000
uv run pytest                # run the full test suite
uv run pytest tests/test_rag_service.py::test_answer_query  # run a single test
uv run ruff check .           # lint
uv run ruff check . --fix     # lint and auto-fix
```

Add a new dependency with `uv add <package>` (or `uv add --dev <package>` for dev/test-only tools)
rather than editing `pyproject.toml` by hand, so the lockfile stays in sync.

Configuration is loaded from environment variables / a local `.env` file (see `.env.example`) via
`app/core/config.py` (`pydantic-settings`; unknown env vars are ignored). Settings:
`AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_KNOWLEDGE_BASE_ID` (required for real retrieval;
empty is allowed at startup but queries then fail with 503), `BEDROCK_KB_SEARCH_TYPE`
(`vector` | `managed`; managed KBs reject `vectorSearchConfiguration`), `BEDROCK_KB_RETRIEVAL_RESULTS`,
`BEDROCK_MAX_TOKENS`, `BEDROCK_TEMPERATURE`, `SYSTEM_PROMPT` (defaults to
`DEFAULT_SYSTEM_PROMPT`), `HISTORY_MAX_TURNS`, `ASSISTANT_SCOPE` (what the assistant helps
with; defaults to `DEFAULT_ASSISTANT_SCOPE`, support for employees who resigned or left the
company), `CHAT_NUDGE_AFTER` (3) and `CHAT_MAX_CONSECUTIVE` (6; both ≥ 1, max ≥ nudge),
`CHAT_LIMIT_MESSAGE` (the capped-tier reply; defaults to `DEFAULT_CHAT_LIMIT_MESSAGE`, is
stripped and must not be blank; keep it in line with `ASSISTANT_SCOPE`),
`LOG_LEVEL`. AWS credentials are picked up
through the standard boto3 credential chain (env vars, shared credentials file, or an IAM role) —
they are not configured in this app's settings.

## Architecture

### Request flow

Both endpoints live in `app/api/routes/query.py` and validate the body against `QueryRequest`
(`app/models/schemas.py`): `question` is whitespace-stripped and must be 1–4000 chars;
`session_id` is optional, ≤128 chars of `[A-Za-z0-9_-]` (it is echoed and logged). Invalid
bodies get a 422.

- `POST /query` → `rag.answer_query` → runs `_rag_graph` to completion and returns a
  `QueryResponse` (`answer`, `citations`, `session_id`).
- `POST /query/stream` → `rag.stream_query` → runs `_rag_stream_graph` and yields typed events
  that the route turns into Server-Sent Events (`fastapi.sse.EventSourceResponse`, sync
  generator so blocking Bedrock calls run in the threadpool). Event order: one `citations`
  (list of citation objects, possibly empty), zero or more `token` (`{"text": ...}`), then one
  `done` (`{"session_id": ...}`). `done` is sent only after the turn has been checkpointed.
  A chat turn (no retrieval) sends `citations` as `[]` right after `classify_intent`.

### The LangGraph pipeline (`app/services/rag.py`)

`RagState` holds per-turn fields (`question`, `intent`, `chat_tier`, `standalone_question`,
`rewritten`, `citations`, `answer`; reset on each invoke by `turn_input`) plus two fields that persist
across turns: `history: list[ChatTurn]` and `chat_streak: int` (completed chat turns in a row
before this one). Seven nodes, with one conditional edge after `classify_intent`:

```
START → classify_intent ─┬─ knowledge ─→ rewrite_query → retrieve → generate ─┬→ update_history → END
                         ├─ chat ──────→ chat ────────────────────────────────┤
                         └─ chat, capped → chat_limit ────────────────────────┘
```

1. `classify_intent` → `bedrock_llm.classify_intent(question, history)` sets `intent`
   (`"knowledge"` | `"chat"`) and, for chat turns, `chat_tier` from `chat_streak`
   (`_chat_tier(streak)`): below `CHAT_NUDGE_AFTER` → `"normal"`; below
   `CHAT_MAX_CONSECUTIVE` → `"nudge"`; otherwise `"capped"`. Knowledge turns get
   `chat_tier=None`. The tier is computed only here; later nodes read `state["chat_tier"]`.
   `route_by_intent` is a pure read of `intent` / `chat_tier` (no side effects, no logging):
   knowledge turns go to `rewrite_query`; greetings, thanks, small talk and off-scope requests
   go to `chat`, or to `chat_limit` when the tier is capped.
2. `rewrite_query` — with no history, passes `question` through unchanged; otherwise calls
   `bedrock_llm.rewrite_question` to turn the follow-up into a standalone question
   (`rewritten=True`). Only retrieval uses the rewritten question.
3. `retrieve` → `bedrock_kb.retrieve(standalone_question)` — Bedrock Agent Runtime `retrieve`
   against the Knowledge Base; returns `Citation`s, skipping non-text results. An empty
   `BEDROCK_KNOWLEDGE_BASE_ID` raises a 503 without calling AWS.
4. `generate` → `bedrock_llm.generate_answer` (Bedrock Runtime `converse`) with the original
   question, citations and prior history. In the streaming graph this node is
   `generate_stream_node`, still registered as `"generate"`: it calls `bedrock_llm.stream_answer`
   (`converse_stream`) and `_emit_tokens` writes each delta on LangGraph's custom stream as
   `{"token": text}`. `stream_query` consumes `stream_mode=["updates", "custom"]` and treats
   only custom payloads with a `token` key as tokens.
5. `chat` — reply without retrieval (`citations` stays `[]`), normal or nudge tier only:
   `bedrock_llm.chat_reply(..., nudge=state["chat_tier"] == "nudge")` (a nudge steers the user
   back to the scope). The streaming variant `chat_stream_node` (still `"chat"`) uses
   `bedrock_llm.stream_chat_reply` and `_emit_tokens`.
6. `chat_limit` — capped tier: returns the fixed `settings.chat_limit_message`
   (`CHAT_LIMIT_MESSAGE`, read at call time) with no model call. One node serves both graphs:
   it also writes the message as a single token.
   `_stream_writer()` falls back to a no-op outside a graph run, so `chat_limit_node` (like
   the streaming nodes) can be called by hand; under `invoke` the write is ignored.
7. `update_history` — appends the plain user question and assistant answer (no context) and
   keeps the last `HISTORY_MAX_TURNS` turns. It also sets `chat_streak` to +1 after a chat
   turn, or to 0 after a knowledge turn. Capped turns (`chat_tier == "capped"`) count toward
   `chat_streak` but are **not** added to history, so repeated limit messages can't push
   earlier Q&A out of the window.

`_build_graph(checkpointer, generate=, chat=)` takes the `generate` / `chat` node
implementations, so both graphs share one topology. `build_graphs` compiles the blocking
graph and the streaming graph (`generate_stream_node` / `chat_stream_node`) over one
shared `InMemorySaver` checkpointer (msgpack allowlist: LangGraph's safe types plus
`Citation`), so `/query` and
`/query/stream` share session history and `chat_streak`. The thread id is the request's
`session_id`, or a fresh `uuid4` (returned to the caller) when missing. A failed or abandoned
turn never reaches `update_history`, so history and `chat_streak` are unchanged.
**Session memory is process-local and unbounded**
(only turns per session are capped): it is lost on restart, not shared across workers, and
the number of sessions grows without eviction.

### Errors

`app/services/errors.py` defines `ServiceError` with `ServiceUnavailableError` (503) and
`UpstreamServiceError` (502). `from_aws_error` maps botocore errors: throttling / service
unavailable / model not ready, and missing or partial credentials → 503; any other
`ClientError` or `BotoCoreError` → 502. In-stream `converse_stream` exceptions arrive as
`EventStreamError` (lower-camel code, normalized by `aws_error_code`) and follow the same rules.
An empty model response is a 502 (except from `classify_intent`, which then falls back to
knowledge). Messages carry only the AWS error code, never raw AWS
messages. `app/main.py` registers a `ServiceError` handler returning
`{"detail": message}` with the error's status. For `/query/stream` the 200 has already been
sent, so the route instead ends the stream with one `error` event
(`{"detail", "status_code"}`) and no `done`; unexpected exceptions are logged and sent as a
generic 500 `error` event.

### Prompting and logging

`bedrock_llm` wraps each retrieved chunk in `<passage id="n" source="...">` tags (n is the `[n]`
citation number) and neutralizes any passage tags inside chunk text; the source URI is
HTML-escaped. The rewrite and intent-router prompts share `build_transcript_prompt`, which uses
`<conversation>` / `<follow_up>` tags the same way. These system prompts tell the model to treat
tagged text as data. `SYSTEM_PROMPT` can override the answer prompt.

The router prompt (`router_system_prompt`) and the chat prompt (`chat_system_prompt`, plus
`CHAT_NUDGE_INSTRUCTION` when nudging) are built from `settings.assistant_scope` at call time.
The router asks for one word; `classify_intent` accepts `chat` / `knowledge` and falls back to
`"knowledge"` on an empty, unrecognized or hedged reply (AWS errors still propagate). The chat
prompt forbids citations and made-up policies, amounts, dates or entitlements.

Logging is configured in `app/main.py` from `LOG_LEVEL`. Log only metadata (ids, counts,
codes, stop reasons, token usage, timings, `purpose`, `intent`, whether the router label was
`recognized`, `chat_tier` / `chat_streak_after`), never question, history, context or answer text.
`answer_query` and `stream_query` each emit one summary line per completed turn
(`RAG query answered ...` / `RAG query streamed ...`) with `session_id`, `intent`, `chat_tier`
(the tier applied to this turn; `None` for knowledge turns) and `chat_streak_after` (the
streak saved by `update_history`, i.e. after this turn). Routing functions do not log.

### Conventions

Routes stay thin (HTTP concerns only); pipeline orchestration lives in the LangGraph graph built
in `app/services/rag.py`; each AWS Bedrock API (Knowledge Base retrieval vs. LLM invocation) has
its own module (`bedrock_kb.py` / `bedrock_llm.py`) rather than a single combined "bedrock
client". When extending the pipeline (e.g. adding re-ranking), add it as a new node/edge in the
graph rather than inlining logic into the existing nodes, and keep both graphs' topology in
sync via `_build_graph`. That keeps each step independently testable/mockable and keeps
conditional branching (e.g. retries, fallback models) expressible via LangGraph's edges instead
of ad hoc `if`s.

Each `bedrock_*.py` service module creates its boto3 client at import time using
`settings.aws_region`. When adding new AWS-backed services, follow the same pattern rather than
constructing clients inside request handlers.

## Project structure

```
app/
  main.py                 # FastAPI app, logging config, ServiceError handler, routers
  api/
    routes/
      health.py            # GET /health
      query.py             # POST /query, POST /query/stream (SSE)
  core/
    config.py              # pydantic-settings Settings, DEFAULT_SYSTEM_PROMPT,
                           #   DEFAULT_ASSISTANT_SCOPE, DEFAULT_CHAT_LIMIT_MESSAGE
  models/
    schemas.py             # QueryRequest / QueryResponse / Citation (pydantic), ChatTurn
  services/
    rag.py                  # LangGraph graphs: RagState, 7 nodes + intent/chat-tier routing,
                            #   answer_query / stream_query
    bedrock_kb.py           # Bedrock Agent Runtime: Knowledge Base retrieval
    bedrock_llm.py          # Bedrock Runtime: prompts, converse (classify/rewrite/answer/chat),
                            #   converse_stream (answer/chat)
    errors.py               # ServiceError (502/503) + AWS error mapping
tests/
  conftest.py               # autouse: fresh checkpointer + graphs; classify_intent → "knowledge"
  test_config.py            # Settings defaults, bounds, log level parsing
  test_health.py
  test_bedrock_kb.py        # patches bedrock_kb._client; result mapping, error mapping
  test_bedrock_llm.py       # patches bedrock_llm._client; prompts, converse, streaming, errors
  test_rag_service.py       # mocks bedrock_kb/bedrock_llm; graph, routing, chat tiers, history
  test_query_route.py       # POST /query via TestClient, incl. chat path and capped tier
  test_query_stream_route.py  # POST /query/stream SSE events, chat path, errors, shared sessions
notebooks/
  architecture_walkthrough.ipynb  # offline, step-by-step tour of the pipeline (see below)
pyproject.toml              # deps + tool config (uv, ruff, pytest)
Dockerfile                  # uv-based container build
.env.example                # every setting with its default; no credentials
```

New AWS-backed capabilities belong in `app/services/` as their own module (mirroring
`bedrock_kb.py` / `bedrock_llm.py`); new pipeline steps belong as graph nodes in `rag.py`; new
HTTP endpoints belong in `app/api/routes/` and get registered in `app/main.py`.

## Testing conventions

Pipeline and route tests mock the service-layer boundary (`bedrock_kb.retrieve`,
`bedrock_llm.classify_intent`, `bedrock_llm.rewrite_question`, `bedrock_llm.generate_answer`,
`bedrock_llm.stream_answer`, `bedrock_llm.chat_reply`, `bedrock_llm.stream_chat_reply`)
rather than mocking boto3 directly — see `tests/test_rag_service.py` and
`tests/test_query_stream_route.py` for the pattern. Tests of the service modules themselves
(`test_bedrock_kb.py`, `test_bedrock_llm.py`) patch the module-level `_client`; streaming
errors are simulated by raising botocore exceptions (e.g. `EventStreamError`) from the event
iterator, as real botocore does. No test makes real AWS calls. `moto` is available as a dev
dependency if a test needs to exercise actual boto3 client behavior against a simulated AWS
backend.

The graphs are module-level singletons with in-memory session history, so
`tests/conftest.py` has an autouse fixture that builds fresh graphs over a new checkpointer via
`rag.build_graphs(rag.new_checkpointer())` and patches `rag._rag_graph` / `rag._rag_stream_graph`
for each test. Use distinct `session_id`s within a test, and inspect history and `chat_streak`
with `rag._rag_graph.get_state({"configurable": {"thread_id": ...}})`. Logging tests should
assert that question/answer text does not appear in `caplog`.

A second autouse fixture, `classify_intent`, patches `bedrock_llm.classify_intent` to return
`"knowledge"`, so existing tests take the RAG path without a router call. It yields the mock:
request the fixture by name and set `return_value = "chat"` (or `side_effect`) to exercise the
chat path. `test_bedrock_llm.py`, which tests the real `classify_intent` against a mocked
`_client`, defines its own same-name `classify_intent` fixture that yields `None`. Pytest uses
the closest definition of a fixture name, so the module fixture overrides the conftest mock. To
reach the nudge/capped tiers quickly, patch
`settings.chat_nudge_after` / `settings.chat_max_consecutive` with `patch.object`.

## Architecture notebook

`notebooks/architecture_walkthrough.ipynb` is a learning aid that walks through the pipeline
(validation, `RagState`, each node by hand, intent routing, prompts, multi-turn rewrite, the
chat path and its streak tiers, streaming, routes via `TestClient`, error mapping). It runs
offline: by default (`USE_REAL_AWS = False`) it reassigns the same seven service-boundary
functions the tests mock to small fakes (a keyword-based `classify_intent`; retrieval over three
in-memory documents); everything else is real project code. It needs `ipykernel` (a dev
dependency) and `os.chdir`s to the project root so `app` imports and `.env` resolve.

It calls internals directly (`rag.turn_input`, the `*_node` functions, `rag.route_by_intent`,
`rag._rag_graph`, `bedrock_llm.build_prompt`, `bedrock_llm.router_system_prompt`,
`bedrock_llm.chat_system_prompt`, `settings.system_prompt`, `settings.chat_limit_message`, and
patches `settings.chat_nudge_after` / `chat_max_consecutive`). When you rename or change
the signature of these, or add/remove a pipeline node, update the notebook's cells and its
markdown diagrams to match, and re-run it top to bottom.
