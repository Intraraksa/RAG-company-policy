import logging
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamWriter

from app.core.config import settings
from app.models.schemas import ChatTurn, Citation, QueryRequest, QueryResponse
from app.services import bedrock_kb, bedrock_llm

logger = logging.getLogger(__name__)

ChatTier = Literal["normal", "nudge", "capped"]


class RagState(TypedDict):
    # Per-turn fields: reset on every invoke (see `turn_input`).
    question: str
    intent: bedrock_llm.Intent
    # Set by `classify_intent` for chat turns; None for knowledge turns.
    chat_tier: ChatTier | None
    standalone_question: str
    rewritten: bool
    citations: list[Citation]
    answer: str
    # Persisted across turns by the checkpointer; absent until a thread's first turn completes.
    # User/assistant pairs, oldest first.
    history: NotRequired[list[ChatTurn]]
    # Consecutive completed chat turns immediately before this one (0 after a knowledge turn).
    chat_streak: NotRequired[int]


# Events yielded by `stream_query`, in order: one "citations", zero or more "token", one "done".
StreamEvent = (
    tuple[Literal["citations"], list[Citation]]
    | tuple[Literal["token"], str]
    | tuple[Literal["done"], dict[str, str]]
)


def turn_input(question: str) -> dict:
    """Graph input for one turn. Omits `history` / `chat_streak` so checkpointed values are kept."""
    return {
        "question": question,
        "intent": "knowledge",
        "chat_tier": None,
        "standalone_question": "",
        "rewritten": False,
        "citations": [],
        "answer": "",
    }


def _chat_tier(streak: int) -> ChatTier:
    """Pick the chat reply tier from the number of chat turns in a row before this one."""
    if streak >= settings.chat_max_consecutive:
        return "capped"
    if streak >= settings.chat_nudge_after:
        return "nudge"
    return "normal"


def classify_intent_node(state: RagState) -> dict:
    """Set `intent`, and for chat turns the `chat_tier` derived from the current streak."""
    intent = bedrock_llm.classify_intent(state["question"], state.get("history") or [])
    chat_tier = _chat_tier(state.get("chat_streak", 0)) if intent == "chat" else None
    return {"intent": intent, "chat_tier": chat_tier}


def route_by_intent(state: RagState) -> Literal["rewrite_query", "chat", "chat_limit"]:
    """Knowledge questions go through retrieval; greetings and small talk skip it.

    Chat turns go to `chat_limit` (fixed reply, no model call) once the session is capped.
    """
    if state["intent"] != "chat":
        return "rewrite_query"
    return "chat_limit" if state["chat_tier"] == "capped" else "chat"


def rewrite_query_node(state: RagState) -> dict:
    history = state.get("history") or []
    if not history:
        return {"standalone_question": state["question"], "rewritten": False}
    return {
        "standalone_question": bedrock_llm.rewrite_question(state["question"], history),
        "rewritten": True,
    }


def retrieve_node(state: RagState) -> dict:
    return {"citations": bedrock_kb.retrieve(state["standalone_question"])}


def generate_node(state: RagState) -> dict:
    return {
        "answer": bedrock_llm.generate_answer(
            state["question"], state["citations"], history=state.get("history") or []
        )
    }


def _stream_writer() -> StreamWriter:
    """LangGraph's custom stream writer, or a no-op when a node is called outside a graph run."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _chunk: None


def _emit_tokens(deltas: Iterator[str]) -> str:
    """Write each delta on the custom stream as {"token": text} and return the joined text."""
    writer = _stream_writer()
    parts: list[str] = []
    for delta in deltas:
        parts.append(delta)
        writer({"token": delta})
    return "".join(parts)


def generate_stream_node(state: RagState) -> dict:
    """Streaming `generate`: emits each text delta on the custom stream as {"token": text}."""
    deltas = bedrock_llm.stream_answer(
        state["question"], state["citations"], history=state.get("history") or []
    )
    return {"answer": _emit_tokens(deltas)}


def chat_node(state: RagState) -> dict:
    """Conversational reply without retrieval (normal or nudge tier; capped turns skip it)."""
    answer = bedrock_llm.chat_reply(
        state["question"],
        history=state.get("history") or [],
        nudge=state["chat_tier"] == "nudge",
    )
    return {"answer": answer}


def chat_stream_node(state: RagState) -> dict:
    """Streaming `chat`: emits each text delta on the custom stream as {"token": text}."""
    deltas = bedrock_llm.stream_chat_reply(
        state["question"],
        history=state.get("history") or [],
        nudge=state["chat_tier"] == "nudge",
    )
    return {"answer": _emit_tokens(deltas)}


def chat_limit_node(state: RagState) -> dict:
    """Capped chat turn: the fixed `settings.chat_limit_message`, with no model call.

    The setting is read at call time (so it can be patched). Shared by both graphs: the
    message is one token under `stream`, and the stream writer is a no-op under `invoke` or
    when the node is called by hand.
    """
    message = settings.chat_limit_message
    _stream_writer()({"token": message})
    return {"answer": message}


def update_history_node(state: RagState) -> dict:
    """Append this turn's plain question/answer and keep the last `history_max_turns` turns.

    Also advances `chat_streak` (reset to 0 by a knowledge turn). Capped chat turns count
    toward the streak but are not added to history, so repeated limit messages cannot push
    earlier Q&A out of the window. Failed turns never get here.
    """
    chat_streak = state.get("chat_streak", 0) + 1 if state["intent"] == "chat" else 0
    history: list[ChatTurn] = list(state.get("history") or [])
    if state["chat_tier"] != "capped":
        history = [
            *history,
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state["answer"]},
        ][-2 * settings.history_max_turns :]
    return {"history": history, "chat_streak": chat_streak}


def _build_graph(
    checkpointer: InMemorySaver | None = None,
    generate: Callable[[RagState], dict] = generate_node,
    chat: Callable[[RagState], dict] = chat_node,
) -> CompiledStateGraph:
    # Nodes are always named "generate" / "chat" (only their implementations vary), so the
    # blocking and streaming graphs have identical topology and can share checkpoints per thread.
    graph = StateGraph(RagState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate)
    graph.add_node("chat", chat)
    graph.add_node("chat_limit", chat_limit_node)
    graph.add_node("update_history", update_history_node)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route_by_intent, ["rewrite_query", "chat", "chat_limit"]
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "update_history")
    graph.add_edge("chat", "update_history")
    graph.add_edge("chat_limit", "update_history")
    graph.add_edge("update_history", END)
    return graph.compile(checkpointer=checkpointer)


def new_checkpointer() -> InMemorySaver:
    # Strict msgpack allowlist: only LangGraph's built-in safe types plus `Citation` (stored
    # in the per-turn `citations` field) may be deserialized from a checkpoint.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(Citation.__module__, Citation.__name__)])
    return InMemorySaver(serde=serde)


def build_graphs(checkpointer: InMemorySaver) -> tuple[CompiledStateGraph, CompiledStateGraph]:
    """Blocking and streaming graphs over one checkpointer, so they share session history."""
    return _build_graph(checkpointer), _build_graph(
        checkpointer, generate=generate_stream_node, chat=chat_stream_node
    )


# Process-local, unbounded session memory: lost on restart and not shared across workers.
_rag_graph, _rag_stream_graph = build_graphs(new_checkpointer())


def answer_query(request: QueryRequest) -> QueryResponse:
    session_id = request.session_id or str(uuid.uuid4())
    start = time.perf_counter()
    result = _rag_graph.invoke(
        turn_input(request.question), {"configurable": {"thread_id": session_id}}
    )
    # chat_tier applied to this turn; chat_streak_after is the streak saved by update_history.
    logger.info(
        "RAG query answered session_id=%s new_session=%s intent=%s chat_tier=%s "
        "chat_streak_after=%d history_messages=%d rewritten=%s question_chars=%d citations=%d "
        "elapsed_ms=%.0f",
        session_id,
        request.session_id is None,
        result["intent"],
        result["chat_tier"],
        result["chat_streak"],
        len(result.get("history") or []),
        result["rewritten"],
        len(request.question),
        len(result["citations"]),
        (time.perf_counter() - start) * 1000,
    )
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        session_id=session_id,
    )


def stream_query(request: QueryRequest) -> Iterator[StreamEvent]:
    """Run one turn on the streaming graph, yielding `StreamEvent`s as they happen.

    "citations" is yielded after `retrieve`, or as an empty list as soon as the turn is
    classified as chat (no retrieval). "done" is yielded only after `update_history` has
    checkpointed the turn. Errors from any node propagate to the caller; a failed or abandoned
    stream never reaches `update_history`, so the session history is left unchanged.
    """
    session_id = request.session_id or str(uuid.uuid4())
    start = time.perf_counter()
    intent: bedrock_llm.Intent = "knowledge"
    chat_tier: ChatTier | None = None
    chat_streak_after = 0
    rewritten = False
    citations = 0
    history_messages = 0
    tokens = 0
    for part in _rag_stream_graph.stream(
        turn_input(request.question),
        {"configurable": {"thread_id": session_id}},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        if part["type"] == "custom":
            # Only the answer nodes' {"token": ...} payloads (see `_emit_tokens` and
            # `chat_limit_node`) are tokens; ignore others.
            data = part["data"]
            if isinstance(data, dict) and "token" in data:
                tokens += 1
                yield "token", data["token"]
        elif part["type"] == "updates":
            updates = part["data"]
            if "classify_intent" in updates:
                intent = updates["classify_intent"]["intent"]
                chat_tier = updates["classify_intent"]["chat_tier"]
                if intent == "chat":
                    yield "citations", []
            elif "rewrite_query" in updates:
                rewritten = updates["rewrite_query"]["rewritten"]
            elif "retrieve" in updates:
                citations = len(updates["retrieve"]["citations"])
                yield "citations", updates["retrieve"]["citations"]
            elif "update_history" in updates:
                history_messages = len(updates["update_history"]["history"])
                chat_streak_after = updates["update_history"]["chat_streak"]
    logger.info(
        "RAG query streamed session_id=%s new_session=%s intent=%s chat_tier=%s "
        "chat_streak_after=%d history_messages=%d rewritten=%s question_chars=%d citations=%d "
        "tokens=%d elapsed_ms=%.0f",
        session_id,
        request.session_id is None,
        intent,
        chat_tier,
        chat_streak_after,
        history_messages,
        rewritten,
        len(request.question),
        citations,
        tokens,
        (time.perf_counter() - start) * 1000,
    )
    yield "done", {"session_id": session_id}
