import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langgraph.config import get_stream_writer

from app.core.config import settings
from app.main import app
from app.models.schemas import Citation
from app.services import rag
from app.services.errors import ServiceUnavailableError, UpstreamServiceError

client = TestClient(app)

CITATION = Citation(content="The answer is 42.", source_uri="s3://bucket/doc.txt", score=0.7)
DELTAS = ["The answer", " is\n42", " [1]."]


def _parse_sse(body: str) -> list[tuple[str, object]]:
    """Return (event, decoded JSON data) pairs, skipping keep-alive comments."""
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        event, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].lstrip(" "))
        if event is not None:
            events.append((event, json.loads("\n".join(data))))
    return events


def _stream(body: dict) -> list[tuple[str, object]]:
    response = client.post("/query/stream", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return _parse_sse(response.text)


def _deltas(*deltas: str):
    def stream_answer(question, citations, history=None):
        yield from deltas

    return stream_answer


def _chat_deltas(*deltas: str):
    def stream_chat_reply(question, history=None, nudge=False):
        yield from deltas

    return stream_chat_reply


@pytest.fixture
def mocks():
    with (
        patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION]) as retrieve,
        patch(
            "app.services.rag.bedrock_llm.stream_answer", side_effect=_deltas(*DELTAS)
        ) as stream_answer,
        patch(
            "app.services.rag.bedrock_llm.rewrite_question", return_value="standalone?"
        ) as rewrite,
    ):
        yield retrieve, stream_answer, rewrite


def _state(session_id: str) -> dict:
    return rag._rag_graph.get_state({"configurable": {"thread_id": session_id}}).values


def _history(session_id: str) -> list[dict] | None:
    return _state(session_id).get("history")


def test_stream_happy_path_event_order_and_payloads(mocks) -> None:
    retrieve, stream_answer, rewrite = mocks

    events = _stream({"question": "  What is it?  ", "session_id": "s1"})

    assert events == [
        ("citations", [CITATION.model_dump()]),
        *[("token", {"text": delta}) for delta in DELTAS],
        ("done", {"session_id": "s1"}),
    ]
    retrieve.assert_called_once_with("What is it?")
    stream_answer.assert_called_once_with("What is it?", [CITATION], history=[])
    rewrite.assert_not_called()


def test_stream_token_newlines_do_not_break_framing(mocks) -> None:
    response = client.post("/query/stream", json={"question": "q"})

    assert "data: {\"text\": \" is\\n42\"}" in response.text


def test_stream_sets_sse_headers(mocks) -> None:
    response = client.post("/query/stream", json={"question": "q"})

    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_stream_without_session_id_returns_generated_uuid(mocks) -> None:
    events = _stream({"question": "What?"})

    kind, payload = events[-1]
    assert kind == "done"
    session_id = payload["session_id"]
    assert str(uuid.UUID(session_id)) == session_id


def test_stream_with_no_citations_still_streams(mocks) -> None:
    retrieve, _, _ = mocks
    retrieve.return_value = []

    events = _stream({"question": "q", "session_id": "empty"})

    assert events[0] == ("citations", [])
    assert events[-1] == ("done", {"session_id": "empty"})


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "q", "session_id": "has space"},
    ],
)
def test_stream_invalid_body_returns_422(mocks, body) -> None:
    retrieve, stream_answer, _ = mocks

    response = client.post("/query/stream", json=body)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    retrieve.assert_not_called()
    stream_answer.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (UpstreamServiceError("Knowledge Base retrieval failed (X)"), 502),
        (ServiceUnavailableError("Knowledge Base is not configured"), 503),
    ],
)
def test_stream_retrieve_error_sends_error_event(mocks, error, status_code) -> None:
    retrieve, stream_answer, _ = mocks
    retrieve.side_effect = error

    events = _stream({"question": "q", "session_id": "kb-fail"})

    assert events == [("error", {"detail": error.message, "status_code": status_code})]
    stream_answer.assert_not_called()


def test_stream_error_midway_sends_tokens_then_error(mocks) -> None:
    _, stream_answer, _ = mocks

    def failing(question, citations, history=None):
        yield "partial "
        yield "answer"
        raise ServiceUnavailableError("Model invocation is temporarily unavailable (X)")

    stream_answer.side_effect = failing

    events = _stream({"question": "q", "session_id": "mid-fail"})

    assert events == [
        ("citations", [CITATION.model_dump()]),
        ("token", {"text": "partial "}),
        ("token", {"text": "answer"}),
        (
            "error",
            {"detail": "Model invocation is temporarily unavailable (X)", "status_code": 503},
        ),
    ]


def test_stream_follow_up_rewrite_error_sends_only_error_event(mocks) -> None:
    retrieve, stream_answer, rewrite = mocks
    stream_answer.side_effect = _deltas("a1")

    _stream({"question": "q1", "session_id": "rewrite-fail"})
    turn_one = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    rewrite.side_effect = UpstreamServiceError("Model invocation failed (X)")

    events = _stream({"question": "q2", "session_id": "rewrite-fail"})

    assert events == [("error", {"detail": "Model invocation failed (X)", "status_code": 502})]
    rewrite.assert_called_once_with("q2", turn_one)
    retrieve.assert_called_once_with("q1")
    stream_answer.assert_called_once()
    assert _history("rewrite-fail") == turn_one


def test_stream_ignores_non_token_custom_payloads(mocks) -> None:
    _, stream_answer, _ = mocks

    def with_progress(question, citations, history=None):
        get_stream_writer()({"progress": "halfway"})
        yield "only token"

    stream_answer.side_effect = with_progress

    events = _stream({"question": "q", "session_id": "custom"})

    assert events == [
        ("citations", [CITATION.model_dump()]),
        ("token", {"text": "only token"}),
        ("done", {"session_id": "custom"}),
    ]


def test_stream_unexpected_error_sends_generic_error(mocks, caplog) -> None:
    retrieve, _, _ = mocks
    retrieve.side_effect = RuntimeError("internal secret detail")

    events = _stream({"question": "q", "session_id": "boom"})

    assert events == [("error", {"detail": "Internal server error", "status_code": 500})]
    assert "Unexpected error while streaming a query" in caplog.text


def test_stream_turn_is_visible_to_follow_up_stream(mocks) -> None:
    retrieve, stream_answer, rewrite = mocks
    stream_answer.side_effect = [_deltas("It is ", "42.")(None, None), _deltas("Why")(None, None)]

    _stream({"question": "What is it?", "session_id": "shared"})
    events = _stream({"question": "Why?", "session_id": "shared"})

    turn_one = [
        {"role": "user", "content": "What is it?"},
        {"role": "assistant", "content": "It is 42."},
    ]
    rewrite.assert_called_once_with("Why?", turn_one)
    retrieve.assert_called_with("standalone?")
    assert stream_answer.call_args.kwargs == {"history": turn_one}
    assert events[-1] == ("done", {"session_id": "shared"})


def test_stream_turn_is_visible_to_follow_up_blocking_query(mocks) -> None:
    _, _, rewrite = mocks

    _stream({"question": "What is it?", "session_id": "mixed"})
    with patch("app.services.rag.bedrock_llm.generate_answer", return_value="Because.") as gen:
        response = client.post("/query", json={"question": "Why?", "session_id": "mixed"})

    assert response.status_code == 200
    turn_one = [
        {"role": "user", "content": "What is it?"},
        {"role": "assistant", "content": "".join(DELTAS)},
    ]
    rewrite.assert_called_once_with("Why?", turn_one)
    assert gen.call_args.kwargs == {"history": turn_one}
    assert _history("mixed") == [
        *turn_one,
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "Because."},
    ]


def test_blocking_query_turn_is_visible_to_follow_up_stream(mocks) -> None:
    _, stream_answer, rewrite = mocks

    with patch("app.services.rag.bedrock_llm.generate_answer", return_value="It is 42."):
        client.post("/query", json={"question": "What is it?", "session_id": "mixed2"})
    _stream({"question": "Why?", "session_id": "mixed2"})

    turn_one = [
        {"role": "user", "content": "What is it?"},
        {"role": "assistant", "content": "It is 42."},
    ]
    rewrite.assert_called_once_with("Why?", turn_one)
    assert stream_answer.call_args.kwargs == {"history": turn_one}


def test_failed_stream_does_not_change_history(mocks) -> None:
    _, stream_answer, _ = mocks

    def failing(question, citations, history=None):
        yield "partial"
        raise UpstreamServiceError("Model invocation failed (X)")

    stream_answer.side_effect = [
        _deltas("a1")(None, None),
        failing(None, None),
        _deltas("a3")(None, None),
    ]

    _stream({"question": "q1", "session_id": "fail"})
    failed = _stream({"question": "q2", "session_id": "fail"})
    assert failed[-1][0] == "error"
    assert "done" not in [kind for kind, _ in failed]

    turn_one = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    assert _history("fail") == turn_one

    _stream({"question": "q3", "session_id": "fail"})

    assert stream_answer.call_args.kwargs == {"history": turn_one}
    assert _history("fail") == [
        *turn_one,
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_stream_chat_path_sends_empty_citations_then_tokens(mocks, classify_intent) -> None:
    retrieve, stream_answer, rewrite = mocks
    classify_intent.return_value = "chat"

    with patch(
        "app.services.rag.bedrock_llm.stream_chat_reply", side_effect=_chat_deltas("Hi", " there!")
    ) as stream_chat:
        events = _stream({"question": "  Hello!  ", "session_id": "chat"})

    assert events == [
        ("citations", []),
        ("token", {"text": "Hi"}),
        ("token", {"text": " there!"}),
        ("done", {"session_id": "chat"}),
    ]
    stream_chat.assert_called_once_with("Hello!", history=[], nudge=False)
    retrieve.assert_not_called()
    rewrite.assert_not_called()
    stream_answer.assert_not_called()


def test_stream_chat_error_sends_error_event_without_done(mocks, classify_intent) -> None:
    retrieve, _, _ = mocks
    classify_intent.return_value = "chat"

    def failing(question, history=None, nudge=False):
        yield "Hi"
        raise ServiceUnavailableError("Model invocation is temporarily unavailable (X)")

    with patch("app.services.rag.bedrock_llm.stream_chat_reply", side_effect=failing):
        events = _stream({"question": "Hello!", "session_id": "chat-fail"})

    assert events == [
        ("citations", []),
        ("token", {"text": "Hi"}),
        (
            "error",
            {"detail": "Model invocation is temporarily unavailable (X)", "status_code": 503},
        ),
    ]
    retrieve.assert_not_called()
    # The failed turn never reached `update_history`: no history and no chat streak recorded.
    assert "history" not in _state("chat-fail")
    assert "chat_streak" not in _state("chat-fail")


def test_query_and_stream_share_chat_streak(mocks, classify_intent) -> None:
    retrieve, _, _ = mocks
    classify_intent.return_value = "chat"
    body = {"question": "Hi", "session_id": "chat-mixed"}

    with (
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 2),
        patch("app.services.rag.bedrock_llm.chat_reply", return_value="Hello.") as chat,
        patch(
            "app.services.rag.bedrock_llm.stream_chat_reply", side_effect=_chat_deltas("Hey.")
        ) as stream_chat,
    ):
        client.post("/query", json=body)  # streak 0: normal
        _stream(body)  # streak 1 (from /query): nudge
        capped = client.post("/query", json=body).json()  # streak 2 (from the stream): capped
        events = _stream(body)  # still capped

    assert chat.call_args.kwargs["nudge"] is False
    assert stream_chat.call_args.kwargs["nudge"] is True
    chat.assert_called_once()
    stream_chat.assert_called_once()
    assert capped["answer"] == settings.chat_limit_message
    assert events == [
        ("citations", []),
        ("token", {"text": settings.chat_limit_message}),
        ("done", {"session_id": "chat-mixed"}),
    ]
    assert _state("chat-mixed")["chat_streak"] == 4
    retrieve.assert_not_called()
