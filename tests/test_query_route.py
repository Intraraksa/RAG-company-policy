import re
import uuid
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import MAX_QUESTION_LENGTH, SESSION_ID_PATTERN, Citation
from app.services import bedrock_kb, rag
from app.services.errors import UpstreamServiceError

client = TestClient(app)

CITATION = Citation(content="The answer is 42.", source_uri="s3://bucket/doc.txt", score=0.7)


@patch("app.services.rag.bedrock_llm.generate_answer", return_value="42 [1]")
@patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION])
def test_query_happy_path(mock_retrieve, mock_generate) -> None:
    response = client.post("/query", json={"question": "  What is it?  ", "session_id": "s1"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "42 [1]",
        "citations": [CITATION.model_dump()],
        "session_id": "s1",
    }
    mock_retrieve.assert_called_once_with("What is it?")
    mock_generate.assert_called_once_with("What is it?", [CITATION], history=[])


@pytest.mark.parametrize("question", ["", "   ", "\n\t", "x" * (MAX_QUESTION_LENGTH + 1)])
@patch("app.services.rag.bedrock_kb.retrieve")
def test_query_invalid_question_returns_422(mock_retrieve, question) -> None:
    response = client.post("/query", json={"question": question})

    assert response.status_code == 422
    mock_retrieve.assert_not_called()


@pytest.mark.parametrize(
    "session_id",
    ["", "has space", "line\nbreak", "semi;colon", "a" * 129],
)
@patch("app.services.rag.bedrock_kb.retrieve")
def test_query_invalid_session_id_returns_422(mock_retrieve, session_id) -> None:
    response = client.post("/query", json={"question": "What?", "session_id": session_id})

    assert response.status_code == 422
    mock_retrieve.assert_not_called()


@pytest.mark.parametrize(
    "session_id", ["3f2b8c1e-9d4a-4e2b-8f6a-1c2d3e4f5a6b", "user_42", "a" * 128]
)
@patch("app.services.rag.bedrock_llm.generate_answer", return_value="ok")
@patch("app.services.rag.bedrock_kb.retrieve", return_value=[])
def test_query_valid_session_id_accepted(mock_retrieve, mock_generate, session_id) -> None:
    response = client.post("/query", json={"question": "What?", "session_id": session_id})

    assert response.status_code == 200
    assert response.json()["session_id"] == session_id


@pytest.mark.parametrize("body", [{"question": "What?"}, {"question": "What?", "session_id": None}])
@patch("app.services.rag.bedrock_llm.generate_answer", return_value="ok")
@patch("app.services.rag.bedrock_kb.retrieve", return_value=[])
def test_query_without_session_id_returns_generated_uuid(mock_retrieve, mock_generate, body) -> None:
    response = client.post("/query", json=body)

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert str(uuid.UUID(session_id)) == session_id
    assert re.fullmatch(SESSION_ID_PATTERN, session_id)


@patch("app.services.rag.bedrock_llm.rewrite_question", return_value="Why is 42 the answer?")
@patch("app.services.rag.bedrock_llm.generate_answer", side_effect=["42", "Deep Thought"])
@patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION])
def test_query_follow_up_uses_returned_session(mock_retrieve, mock_generate, mock_rewrite) -> None:
    first = client.post("/query", json={"question": "What is it?"}).json()
    second = client.post(
        "/query", json={"question": "Why?", "session_id": first["session_id"]}
    ).json()

    assert second["session_id"] == first["session_id"]
    assert second["answer"] == "Deep Thought"
    mock_rewrite.assert_called_once()
    mock_retrieve.assert_called_with("Why is 42 the answer?")


@patch("app.services.rag.bedrock_llm.chat_reply", return_value="Hi! How can I help?")
@patch("app.services.rag.bedrock_llm.generate_answer")
@patch("app.services.rag.bedrock_kb.retrieve")
def test_query_chat_path_skips_retrieval(
    mock_retrieve, mock_generate, mock_chat, classify_intent
) -> None:
    classify_intent.return_value = "chat"

    response = client.post("/query", json={"question": "  Hello!  ", "session_id": "chat-1"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Hi! How can I help?",
        "citations": [],
        "session_id": "chat-1",
    }
    classify_intent.assert_called_once_with("Hello!", [])
    mock_chat.assert_called_once_with("Hello!", history=[], nudge=False)
    mock_retrieve.assert_not_called()
    mock_generate.assert_not_called()


@patch("app.services.rag.bedrock_llm.chat_reply", return_value="Hello again.")
@patch("app.services.rag.bedrock_kb.retrieve")
def test_query_chat_streak_nudges_then_caps(mock_retrieve, mock_chat, classify_intent) -> None:
    classify_intent.return_value = "chat"
    body = {"question": "Hi", "session_id": "chatty"}

    for _ in range(settings.chat_max_consecutive):
        assert client.post("/query", json=body).json()["answer"] == "Hello again."
    capped = client.post("/query", json=body)

    assert capped.status_code == 200
    assert capped.json() == {
        "answer": settings.chat_limit_message,
        "citations": [],
        "session_id": "chatty",
    }
    # Normal replies until `chat_nudge_after`, nudging replies up to the cap, then no model call.
    assert [call.kwargs["nudge"] for call in mock_chat.call_args_list] == [
        *[False] * settings.chat_nudge_after,
        *[True] * (settings.chat_max_consecutive - settings.chat_nudge_after),
    ]
    mock_retrieve.assert_not_called()
    # The capped turn counts toward the streak but is not stored in history.
    state = rag._rag_graph.get_state({"configurable": {"thread_id": "chatty"}}).values
    assert state["chat_streak"] == settings.chat_max_consecutive + 1
    assert settings.chat_limit_message not in [turn["content"] for turn in state["history"]]


def test_query_missing_question_returns_422() -> None:
    assert client.post("/query", json={}).status_code == 422


@patch("app.services.rag.bedrock_llm.generate_answer")
@patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION])
def test_query_upstream_error_returns_502(mock_retrieve, mock_generate) -> None:
    mock_generate.side_effect = UpstreamServiceError("Model invocation failed (X)")

    response = client.post("/query", json={"question": "What is it?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Model invocation failed (X)"}


def test_query_kb_client_error_is_wrapped_as_502() -> None:
    error = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "secret detail"}}, "Retrieve"
    )
    with (
        patch.object(settings, "bedrock_knowledge_base_id", "kb-test"),
        patch.object(bedrock_kb, "_client") as kb_client,
    ):
        kb_client.retrieve.side_effect = error
        response = client.post("/query", json={"question": "What is it?"})

    assert response.status_code == 502
    assert "AccessDeniedException" in response.json()["detail"]
    assert "secret detail" not in response.text


@patch("app.services.rag.bedrock_llm.generate_answer")
def test_query_missing_kb_id_returns_503(mock_generate) -> None:
    with (
        patch.object(settings, "bedrock_knowledge_base_id", ""),
        patch.object(bedrock_kb, "_client") as kb_client,
    ):
        response = client.post("/query", json={"question": "What is it?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Knowledge Base is not configured"}
    kb_client.retrieve.assert_not_called()
    mock_generate.assert_not_called()
