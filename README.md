# RAG AWS Service

A retrieval-augmented generation (RAG) API built on FastAPI and AWS Bedrock. You ask a question;
the service retrieves relevant chunks from a Bedrock Knowledge Base, asks a Bedrock model to
answer using only those chunks, and returns the answer with its sources. By default it is set up
as an assistant for employees who have resigned or left the company (benefits, final pay,
offboarding steps, rights); change `ASSISTANT_SCOPE` to use it for another domain.

- **`POST /query`**: answer as one JSON response
- **`POST /query/stream`**: answer streamed token by token over Server-Sent Events (SSE)
- **Multi-turn sessions**: pass a `session_id` and follow-up questions ("what about the second
  one?") are rewritten into standalone questions before retrieval
- **Citations**: every answer comes with the retrieved passages, their source URI and score
- **Small talk without retrieval**: greetings, thanks and off-topic messages get a short
  conversational reply with no Knowledge Base call. Long runs of chit-chat are steered back to
  the assistant's topics, then capped with a fixed reply

The pipeline is a [LangGraph](https://langchain-ai.github.io/langgraph/) graph:

```
classify_intent ─┬─ knowledge → rewrite_query → retrieve → generate ─┬→ update_history
                 ├─ chat ─────→ chat ─────────────────────────────────┤
                 └─ capped ───→ chat_limit ───────────────────────────┘
```

| Step | What it does | AWS API |
|---|---|---|
| `classify_intent` | Labels the message `knowledge` or `chat` (when unsure: `knowledge`) | Bedrock Runtime `converse` |
| `rewrite_query` | Makes a follow-up question standalone (skipped on the first turn) | Bedrock Runtime `converse` |
| `retrieve` | Finds the top-N chunks in the Knowledge Base | Bedrock Agent Runtime `retrieve` |
| `generate` | Answers from the numbered chunks, citing them as `[1]`, `[2]` | Bedrock Runtime `converse` / `converse_stream` |
| `chat` | Replies to small talk without retrieval or citations; after several chat turns in a row, also suggests on-topic questions | Bedrock Runtime `converse` / `converse_stream` |
| `chat_limit` | Sends `CHAT_LIMIT_MESSAGE` once the chat streak hits the cap | none |
| `update_history` | Saves the question and answer to the session and updates the chat streak | none |

### Chit-chat limits

Each session counts consecutive chat messages. With the defaults:

| Chat messages in a row | Reply |
|---|---|
| 1–3 | Normal friendly reply |
| 4–6 (`CHAT_NUDGE_AFTER`) | Short reply that points the user back to the assistant's topics with example questions |
| 7+ (`CHAT_MAX_CONSECUTIVE`) | `CHAT_LIMIT_MESSAGE`, with no model call |

Any on-topic question resets the count. Capped replies are not saved to the session history, so
earlier questions and answers stay available for follow-ups.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An AWS account with:
  - a **Bedrock Knowledge Base** with at least one synced data source;
  - **model access** enabled in the Bedrock console for the model you want to use;
  - an IAM identity with these permissions:

    | Action | Resource |
    |---|---|
    | `bedrock:Retrieve` | your Knowledge Base |
    | `bedrock:InvokeModel` | your model |
    | `bedrock:InvokeModelWithResponseStream` | your model |

## Quick start

1. **Install dependencies**

   ```bash
   uv sync
   ```

2. **Configure AWS credentials.** The app uses boto3's standard credential chain and does
   **not** read access keys from `.env`. Locally, the simplest option is:

   ```bash
   aws configure      # writes ~/.aws/credentials and ~/.aws/config
   ```

   When deployed, use an IAM role instead of access keys.

3. **Create `.env`**

   ```bash
   cp .env.example .env
   ```

   Then set at least:

   ```dotenv
   AWS_REGION=us-east-1
   BEDROCK_MODEL_ID=amazon.nova-micro-v1:0     # any model you have access to in this region
   BEDROCK_KNOWLEDGE_BASE_ID=XXXXXXXXXX        # from the Bedrock console
   BEDROCK_KB_SEARCH_TYPE=managed              # see "Choosing the search type" below
   ```

4. **Run**

   ```bash
   uv run uvicorn app.main:app --reload
   ```

   The API is now at http://127.0.0.1:8000, with interactive docs at
   http://127.0.0.1:8000/docs.

### Choosing the search type

Bedrock Knowledge Bases come in two kinds, and the retrieve call is different for each:

| Your Knowledge Base | `BEDROCK_KB_SEARCH_TYPE` |
|---|---|
| Bedrock-managed (e.g. created with the console **quick start**) | `managed` |
| Backed by your own vector store (OpenSearch, Aurora, Pinecone, ...) | `vector` (default) |

If this is wrong, every query fails with
`502 {"detail": "Knowledge Base retrieval failed (ValidationException)"}`.

## Configuration

All settings are read from environment variables or `.env`. Unknown variables are ignored.

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region of your Knowledge Base and model |
| `BEDROCK_MODEL_ID` | `anthropic.claude-sonnet-5` | Bedrock model ID or inference profile ID |
| `BEDROCK_KNOWLEDGE_BASE_ID` | *(empty)* | **Required.** If empty, the app still starts but every query returns 503 |
| `BEDROCK_KB_SEARCH_TYPE` | `vector` | `vector` or `managed` (see above) |
| `BEDROCK_KB_RETRIEVAL_RESULTS` | `5` | Chunks retrieved per question |
| `BEDROCK_MAX_TOKENS` | `1024` | Maximum tokens per answer (≥ 1) |
| `BEDROCK_TEMPERATURE` | `0.0` | Sampling temperature (0.0–1.0) |
| `HISTORY_MAX_TURNS` | `5` | Question/answer turns of history the model sees per session |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `SYSTEM_PROMPT` | built-in | Overrides the answer system prompt (see `app/core/config.py`) |
| `ASSISTANT_SCOPE` | offboarding support | What the assistant helps with. Used to decide whether a message needs the Knowledge Base and to word chat replies |
| `CHAT_NUDGE_AFTER` | `3` | Chat messages in a row before replies start steering back to the topic (≥ 1) |
| `CHAT_MAX_CONSECUTIVE` | `6` | Chat messages in a row before the fixed `CHAT_LIMIT_MESSAGE` is sent (≥ `CHAT_NUDGE_AFTER`) |
| `CHAT_LIMIT_MESSAGE` | built-in | Fixed reply at the chat cap; must not be blank. Keep it in line with `ASSISTANT_SCOPE` |

## API

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### `POST /query`

Request body:

| Field | Type | Rules |
|---|---|---|
| `question` | string | Required. 1–4000 characters after trimming whitespace |
| `session_id` | string | Optional. Up to 128 characters of `A-Z a-z 0-9 _ -`. Omit it to start a new session |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period?"}'
```

```json
{
  "answer": "The notice period is 30 days [1].",
  "citations": [
    {
      "content": "…either party may terminate with 30 days' notice…",
      "source_uri": "s3://my-bucket/contract.pdf",
      "score": 0.61
    }
  ],
  "session_id": "bc5fd8d4-9ceb-498b-af72-7a59e05f4313"
}
```

To ask a follow-up, send the returned `session_id` back:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can it be shortened?", "session_id": "bc5fd8d4-9ceb-498b-af72-7a59e05f4313"}'
```

Small talk is answered without retrieval, so `citations` is empty:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi, how are you?"}'
# {"answer": "Hi! I'm doing well, thanks. ...", "citations": [], "session_id": "..."}
```

### `POST /query/stream`

This endpoint takes the same request body as `/query`. The response is an SSE stream with these
events, in order:

| Event | `data` | When |
|---|---|---|
| `citations` | list of citation objects | once, after retrieval (an empty list right away for small talk) |
| `token` | `{"text": "..."}` | for each piece of the answer |
| `done` | `{"session_id": "..."}` | once, after the turn is saved to the session |
| `error` | `{"detail": "...", "status_code": 502}` | instead of `done`, if something fails |

```bash
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period?"}'
```

> The stream always starts with HTTP 200, so errors arrive as an `error` event rather than as
> a status code. Clients must handle that event.

`/query` and `/query/stream` share sessions (history and chat streak), so a conversation can mix
the two.

### Errors

| Status | Meaning |
|---|---|
| `422` | Invalid request body (empty or too long question, bad `session_id`) |
| `502` | Bedrock returned an error (access denied, validation, model not found, ...) |
| `503` | Not configured (no Knowledge Base ID or no AWS credentials) or temporarily unavailable (throttling) |

Error responses look like `{"detail": "Knowledge Base retrieval failed (AccessDeniedException)"}`.
They include only the AWS error code. The full AWS error message is written to the server log.

### Troubleshooting

| You see | Likely cause |
|---|---|
| `UnrecognizedClientException` / `InvalidClientTokenId` | boto3 found credentials that AWS rejects. Check `~/.aws/credentials` and any `AWS_*` environment variables. Keys in `.env` are **not** used |
| `ValidationException` from retrieval | `BEDROCK_KB_SEARCH_TYPE` doesn't match your Knowledge Base type |
| `AccessDeniedException` | The IAM identity lacks a permission from [Prerequisites](#prerequisites), or model access isn't enabled |
| `ResourceNotFoundException` | Wrong Knowledge Base ID or model ID, or a different region |
| `503 Knowledge Base is not configured` | `BEDROCK_KNOWLEDGE_BASE_ID` is empty |

## Running with Docker

```bash
docker build -t rag-aws .
docker run --rm -p 8000:8000 --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  rag-aws
```

Mounting `~/.aws` gives the container your local credentials. On AWS (ECS, EKS, EC2), attach an
IAM role instead and leave out the mount.

## Development

```bash
uv run pytest                 # run the test suite
uv run ruff check .           # lint
uv run ruff check . --fix     # lint and auto-fix
uv add <package>              # add a dependency (uv add --dev for dev-only)
```

Tests never call AWS. They mock the service boundary (`bedrock_kb`, `bedrock_llm`) or the
boto3 client, and pass without credentials or network access.

Project layout:

```
app/
  main.py                 # FastAPI app, logging, error handler
  api/routes/             # health.py, query.py (/query and /query/stream)
  core/config.py          # settings
  models/schemas.py       # request/response models
  services/
    rag.py                # LangGraph pipeline and session handling
    bedrock_kb.py         # Knowledge Base retrieval
    bedrock_llm.py        # prompts and model calls
    errors.py             # AWS error → HTTP status mapping
tests/
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture notes and contribution conventions.

## Limitations

- **Sessions live in memory.** They are lost on restart and not shared between workers or
  containers. Run a single worker, or replace the `InMemorySaver` checkpointer in
  `app/services/rag.py` with a persistent one.
- **No authentication.** Anyone who can reach the API can query your documents, and anyone who
  knows a `session_id` can continue that conversation. Put the service behind an authenticating
  gateway before exposing it.
- **Concurrent requests on one session** can overwrite each other's history.
- **Disconnecting from `/query/stream` doesn't stop generation.** The model runs to the end and
  those tokens are still billed.
- **No document ingestion.** Add and sync documents through the Bedrock console or API.
- **Every message costs one extra model call** (the intent classifier, a few output tokens). If
  it labels a real question as small talk, the user gets a chat reply instead of a cited answer;
  asking again more specifically usually fixes it.
