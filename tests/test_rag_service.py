import re
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.models.schemas import SESSION_ID_PATTERN, Citation, QueryRequest
from app.services import rag
from app.services.errors import UpstreamServiceError
from app.services.rag import answer_query

CITATION = Citation(content="The answer is 42.", source_uri="s3://bucket/doc.txt")


@pytest.fixture
def mocks():
    with (
        patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION]) as retrieve,
        patch("app.services.rag.bedrock_llm.generate_answer") as generate,
        patch(
            "app.services.rag.bedrock_llm.rewrite_question", return_value="standalone?"
        ) as rewrite,
    ):
        yield retrieve, generate, rewrite


def _turn(question: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def test_answer_query(mocks) -> None:
    retrieve, generate, rewrite = mocks
    generate.return_value = "42"

    result = answer_query(QueryRequest(question="What is the answer?", session_id="single"))

    assert result.answer == "42"
    assert result.citations == [CITATION]
    assert result.session_id == "single"
    retrieve.assert_called_once_with("What is the answer?")
    generate.assert_called_once_with("What is the answer?", [CITATION], history=[])
    rewrite.assert_not_called()


def test_two_turn_session_rewrites_follow_up_and_passes_history(mocks) -> None:
    retrieve, generate, rewrite = mocks
    generate.side_effect = ["It is 42.", "Because of Deep Thought."]
    session_id = "two-turn"

    first = answer_query(QueryRequest(question="What is the answer?", session_id=session_id))

    rewrite.assert_not_called()
    retrieve.assert_called_once_with("What is the answer?")
    generate.assert_called_once_with("What is the answer?", [CITATION], history=[])
    assert first.session_id == session_id

    second = answer_query(QueryRequest(question="Why is it that?", session_id=session_id))

    turn_one = _turn("What is the answer?", "It is 42.")
    rewrite.assert_called_once_with("Why is it that?", turn_one)
    assert retrieve.call_args.args == ("standalone?",)
    # The final prompt keeps the user's original wording; history precedes it.
    assert generate.call_args.args == ("Why is it that?", [CITATION])
    assert generate.call_args.kwargs == {"history": turn_one}
    assert second.answer == "Because of Deep Thought."
    assert second.session_id == session_id


def test_missing_session_id_generates_uuid(mocks) -> None:
    _, generate, _ = mocks
    generate.return_value = "ok"

    result = answer_query(QueryRequest(question="Hi?"))

    assert str(uuid.UUID(result.session_id)) == result.session_id
    assert re.fullmatch(SESSION_ID_PATTERN, result.session_id)


def test_calls_without_session_id_do_not_share_history(mocks) -> None:
    _, generate, rewrite = mocks
    generate.side_effect = ["a1", "a2"]

    first = answer_query(QueryRequest(question="q1"))
    second = answer_query(QueryRequest(question="q2"))

    assert first.session_id != second.session_id
    rewrite.assert_not_called()
    assert [c.kwargs["history"] for c in generate.call_args_list] == [[], []]


def test_different_sessions_are_isolated(mocks) -> None:
    _, generate, rewrite = mocks
    generate.side_effect = ["a1", "a2"]

    answer_query(QueryRequest(question="q1", session_id="alice"))
    answer_query(QueryRequest(question="q2", session_id="bob"))

    rewrite.assert_not_called()
    assert generate.call_args.kwargs["history"] == []


def test_history_is_trimmed_to_max_turns(mocks) -> None:
    _, generate, rewrite = mocks
    generate.side_effect = [f"a{i}" for i in range(1, 6)]

    with patch.object(settings, "history_max_turns", 2):
        for i in range(1, 6):
            answer_query(QueryRequest(question=f"q{i}", session_id="trim"))

    histories = [c.kwargs["history"] for c in generate.call_args_list]
    assert [len(h) for h in histories] == [0, 2, 4, 4, 4]
    # Turn 5 sees only the two most recent complete turns (3 and 4), oldest first.
    assert histories[-1] == _turn("q3", "a3") + _turn("q4", "a4")
    assert rewrite.call_args.args[1] == histories[-1]

    stored = rag._rag_graph.get_state({"configurable": {"thread_id": "trim"}}).values
    assert stored["history"] == _turn("q4", "a4") + _turn("q5", "a5")


def test_history_stores_only_plain_question_and_answer(mocks) -> None:
    _, generate, _ = mocks
    generate.return_value = "42 [1]"

    answer_query(QueryRequest(question="What is it?", session_id="plain"))

    stored = rag._rag_graph.get_state({"configurable": {"thread_id": "plain"}}).values
    assert stored["history"] == _turn("What is it?", "42 [1]")
    assert "passage" not in str(stored["history"])


def test_failed_turn_does_not_change_history(mocks) -> None:
    _, generate, _ = mocks
    generate.side_effect = ["a1", UpstreamServiceError("boom"), "a3"]

    answer_query(QueryRequest(question="q1", session_id="fail"))
    with pytest.raises(UpstreamServiceError):
        answer_query(QueryRequest(question="q2", session_id="fail"))
    answer_query(QueryRequest(question="q3", session_id="fail"))

    assert generate.call_args.kwargs["history"] == _turn("q1", "a1")


def test_rewrite_failure_propagates_without_retrieval(mocks) -> None:
    retrieve, generate, rewrite = mocks
    generate.return_value = "a1"
    answer_query(QueryRequest(question="q1", session_id="rewrite-fail"))
    retrieve.reset_mock()
    rewrite.side_effect = UpstreamServiceError("Model invocation failed (X)")

    with pytest.raises(UpstreamServiceError):
        answer_query(QueryRequest(question="q2", session_id="rewrite-fail"))

    retrieve.assert_not_called()


def test_logs_session_metadata_without_text(mocks, caplog) -> None:
    _, generate, _ = mocks
    generate.side_effect = ["secret answer one", "secret answer two"]

    with caplog.at_level("INFO", logger="app.services.rag"):
        answer_query(QueryRequest(question="secret question one", session_id="log-s"))
        answer_query(QueryRequest(question="secret question two", session_id="log-s"))

    messages = [r.getMessage() for r in caplog.records if r.name == "app.services.rag"]
    assert "session_id=log-s" in messages[-1]
    assert "history_messages=4" in messages[-1]
    assert "rewritten=True" in messages[-1]
    assert "rewritten=False" in messages[0]
    assert "intent=knowledge chat_tier=None chat_streak_after=0 " in messages[-1]
    assert "secret" not in caplog.text


@pytest.fixture
def stream_mocks():
    def deltas(question, citations, history=None):
        yield "It is "
        yield "42."

    with (
        patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION]) as retrieve,
        patch("app.services.rag.bedrock_llm.stream_answer", side_effect=deltas) as stream,
        patch(
            "app.services.rag.bedrock_llm.rewrite_question", return_value="standalone?"
        ) as rewrite,
        patch("app.services.rag.bedrock_llm.generate_answer") as generate,
    ):
        yield retrieve, stream, rewrite, generate


def _stored_history(session_id: str) -> list[dict] | None:
    config = {"configurable": {"thread_id": session_id}}
    return rag._rag_graph.get_state(config).values.get("history")


def test_stream_query_yields_typed_events_and_persists_turn(stream_mocks) -> None:
    _, stream, _, generate = stream_mocks

    events = list(rag.stream_query(QueryRequest(question="What?", session_id="st")))

    assert events == [
        ("citations", [CITATION]),
        ("token", "It is "),
        ("token", "42."),
        ("done", {"session_id": "st"}),
    ]
    stream.assert_called_once_with("What?", [CITATION], history=[])
    generate.assert_not_called()
    assert _stored_history("st") == _turn("What?", "It is 42.")


def test_stream_query_abandoned_midway_does_not_persist_turn(stream_mocks) -> None:
    list(rag.stream_query(QueryRequest(question="q1", session_id="abort")))

    events = rag.stream_query(QueryRequest(question="q2", session_id="abort"))
    assert next(events)[0] == "citations"
    assert next(events) == ("token", "It is ")
    events.close()  # what happens when the SSE client disconnects

    assert _stored_history("abort") == _turn("q1", "It is 42.")


def test_stream_query_logs_metadata_without_text(stream_mocks, caplog) -> None:
    with caplog.at_level("INFO", logger="app.services.rag"):
        list(rag.stream_query(QueryRequest(question="secret question", session_id="log-st")))

    messages = [r.getMessage() for r in caplog.records if r.name == "app.services.rag"]
    assert "session_id=log-st" in messages[-1]
    assert "tokens=2" in messages[-1]
    assert "history_messages=2" in messages[-1]
    assert "intent=knowledge chat_tier=None chat_streak_after=0 " in messages[-1]
    assert "secret" not in caplog.text
    assert "42" not in caplog.text


# --- Intent routing: chat turns skip retrieval ---------------------------------------------


@pytest.fixture
def chat_mocks(classify_intent):
    """Every service boundary mocked; `classify_intent` routes to chat until a test changes it."""
    classify_intent.return_value = "chat"

    def chat_deltas(question, history=None, nudge=False):
        yield "Hello "
        yield "there!"

    with (
        patch("app.services.rag.bedrock_kb.retrieve", return_value=[CITATION]) as retrieve,
        patch(
            "app.services.rag.bedrock_llm.rewrite_question", return_value="standalone?"
        ) as rewrite,
        patch("app.services.rag.bedrock_llm.generate_answer", return_value="42") as generate,
        patch("app.services.rag.bedrock_llm.stream_answer") as stream_answer,
        patch("app.services.rag.bedrock_llm.chat_reply", return_value="Hi!") as chat_reply,
        patch(
            "app.services.rag.bedrock_llm.stream_chat_reply", side_effect=chat_deltas
        ) as stream_chat_reply,
    ):
        yield SimpleNamespace(
            classify=classify_intent,
            retrieve=retrieve,
            rewrite=rewrite,
            generate=generate,
            stream_answer=stream_answer,
            chat_reply=chat_reply,
            stream_chat_reply=stream_chat_reply,
        )


def _stored_streak(session_id: str) -> int | None:
    config = {"configurable": {"thread_id": session_id}}
    return rag._rag_graph.get_state(config).values.get("chat_streak")


@pytest.mark.parametrize(
    ("streak", "tier"),
    [(0, "normal"), (2, "normal"), (3, "nudge"), (5, "nudge"), (6, "capped"), (9, "capped")],
)
def test_chat_tier_thresholds(streak, tier) -> None:
    with (
        patch.object(settings, "chat_nudge_after", 3),
        patch.object(settings, "chat_max_consecutive", 6),
    ):
        assert rag._chat_tier(streak) == tier


@pytest.mark.parametrize(
    ("intent", "streak", "tier"),
    [
        ("knowledge", 0, None),
        ("knowledge", 9, None),
        ("chat", None, "normal"),
        ("chat", 1, "nudge"),
        ("chat", 2, "capped"),
    ],
)
def test_classify_intent_node_sets_chat_tier(classify_intent, intent, streak, tier) -> None:
    classify_intent.return_value = intent
    state = rag.turn_input("q")
    if streak is not None:
        state["chat_streak"] = streak

    with (
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 2),
    ):
        assert rag.classify_intent_node(state) == {"intent": intent, "chat_tier": tier}


@pytest.mark.parametrize(
    ("intent", "tier", "node"),
    [
        ("knowledge", None, "rewrite_query"),
        ("chat", "normal", "chat"),
        ("chat", "nudge", "chat"),
        ("chat", "capped", "chat_limit"),
    ],
)
def test_route_by_intent_reads_intent_and_tier(intent, tier, node, caplog) -> None:
    with caplog.at_level("DEBUG", logger="app.services.rag"):
        assert rag.route_by_intent({"intent": intent, "chat_tier": tier}) == node

    assert caplog.records == []  # a pure routing function: no logging


def test_chat_limit_node_can_be_called_by_hand() -> None:
    state = {**rag.turn_input("hi again"), "intent": "chat", "chat_tier": "capped"}

    assert rag.chat_limit_node(state) == {"answer": settings.chat_limit_message}


def test_chat_limit_node_returns_fixed_message_under_invoke(chat_mocks) -> None:
    with (
        patch.object(settings, "chat_max_consecutive", 1),
        patch.object(settings, "chat_nudge_after", 1),
    ):
        answer_query(QueryRequest(question="hi", session_id="limit"))
        result = answer_query(QueryRequest(question="hi again", session_id="limit"))

    assert result.answer == settings.chat_limit_message
    assert result.citations == []
    chat_mocks.chat_reply.assert_called_once()


def _topology(graph) -> tuple[set[str], set[tuple[str, str, bool]]]:
    drawable = graph.get_graph()
    edges = {(edge.source, edge.target, edge.conditional) for edge in drawable.edges}
    return set(drawable.nodes), edges


def test_blocking_and_streaming_graphs_have_identical_topology() -> None:
    graph, stream_graph = rag.build_graphs(rag.new_checkpointer())

    nodes, edges = _topology(graph)

    assert _topology(stream_graph) == (nodes, edges)
    assert {"classify_intent", "chat", "chat_limit", "generate", "update_history"} <= nodes
    assert {
        ("classify_intent", "rewrite_query", True),
        ("classify_intent", "chat", True),
        ("classify_intent", "chat_limit", True),
    } <= edges


def test_chat_turn_skips_retrieval(chat_mocks) -> None:
    result = answer_query(QueryRequest(question="Hello", session_id="chat"))

    assert result.answer == "Hi!"
    assert result.citations == []
    assert result.session_id == "chat"
    chat_mocks.classify.assert_called_once_with("Hello", [])
    chat_mocks.chat_reply.assert_called_once_with("Hello", history=[], nudge=False)
    chat_mocks.retrieve.assert_not_called()
    chat_mocks.rewrite.assert_not_called()
    chat_mocks.generate.assert_not_called()
    assert _stored_history("chat") == _turn("Hello", "Hi!")
    assert _stored_streak("chat") == 1


def test_knowledge_turn_resets_streak_and_classifier_sees_history(chat_mocks) -> None:
    chat_mocks.classify.return_value = "knowledge"

    answer_query(QueryRequest(question="q1", session_id="know"))
    result = answer_query(QueryRequest(question="q2", session_id="know"))

    assert result.answer == "42"
    assert result.citations == [CITATION]
    assert chat_mocks.classify.call_args.args == ("q2", _turn("q1", "42"))
    chat_mocks.chat_reply.assert_not_called()
    assert _stored_streak("know") == 0


def test_chat_then_knowledge_follow_up_rewrites_with_history(chat_mocks) -> None:
    chat_mocks.classify.side_effect = ["chat", "knowledge"]

    answer_query(QueryRequest(question="Hello", session_id="mixed"))
    result = answer_query(QueryRequest(question="What about my final pay?", session_id="mixed"))

    turn_one = _turn("Hello", "Hi!")
    chat_mocks.rewrite.assert_called_once_with("What about my final pay?", turn_one)
    chat_mocks.retrieve.assert_called_once_with("standalone?")
    chat_mocks.generate.assert_called_once_with(
        "What about my final pay?", [CITATION], history=turn_one
    )
    assert result.citations == [CITATION]
    assert _stored_streak("mixed") == 0


def test_chat_streak_increments_and_resets_after_knowledge_turn(chat_mocks) -> None:
    chat_mocks.classify.side_effect = ["chat", "chat", "knowledge", "chat"]

    streaks = []
    for i in range(4):
        answer_query(QueryRequest(question=f"q{i}", session_id="streak"))
        streaks.append(_stored_streak("streak"))

    assert streaks == [1, 2, 0, 1]


def test_chat_tiers_nudge_then_cap_without_model_call(chat_mocks) -> None:
    with (
        patch.object(settings, "chat_nudge_after", 2),
        patch.object(settings, "chat_max_consecutive", 3),
    ):
        answers = [
            answer_query(QueryRequest(question=f"hi{i}", session_id="tiers")).answer
            for i in range(5)
        ]

    # Streaks before each turn: 0, 1 normal; 2 nudge; 3, 4 capped (no model call).
    assert [c.kwargs["nudge"] for c in chat_mocks.chat_reply.call_args_list] == [
        False,
        False,
        True,
    ]
    limit = settings.chat_limit_message
    assert answers == ["Hi!", "Hi!", "Hi!", limit, limit]
    assert _stored_streak("tiers") == 5
    # Capped turns count toward the streak but are not stored in history.
    assert _stored_history("tiers") == [
        *_turn("hi0", "Hi!"),
        *_turn("hi1", "Hi!"),
        *_turn("hi2", "Hi!"),
    ]


def test_capped_chat_turns_do_not_push_knowledge_out_of_history(chat_mocks) -> None:
    chat_mocks.classify.side_effect = ["knowledge", *["chat"] * 8]

    with (
        patch.object(settings, "history_max_turns", 3),
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 2),
    ):
        for i in range(9):
            answer_query(QueryRequest(question=f"q{i}", session_id="keep"))

    # Turns 1-2 were model chat replies; turns 3-8 were capped and left history alone.
    assert _stored_history("keep") == [*_turn("q0", "42"), *_turn("q1", "Hi!"), *_turn("q2", "Hi!")]
    assert _stored_streak("keep") == 8
    assert chat_mocks.chat_reply.call_count == 2


def test_chat_reply_receives_history(chat_mocks) -> None:
    answer_query(QueryRequest(question="Hello", session_id="chat-hist"))
    answer_query(QueryRequest(question="How are you?", session_id="chat-hist"))

    assert chat_mocks.chat_reply.call_args.kwargs["history"] == _turn("Hello", "Hi!")


def test_failed_chat_turn_leaves_streak_and_history_unchanged(chat_mocks) -> None:
    chat_mocks.chat_reply.side_effect = ["a1", UpstreamServiceError("boom"), "a3"]

    answer_query(QueryRequest(question="q1", session_id="chat-fail"))
    with pytest.raises(UpstreamServiceError):
        answer_query(QueryRequest(question="q2", session_id="chat-fail"))

    assert _stored_streak("chat-fail") == 1
    assert _stored_history("chat-fail") == _turn("q1", "a1")

    answer_query(QueryRequest(question="q3", session_id="chat-fail"))

    assert chat_mocks.chat_reply.call_args.kwargs["history"] == _turn("q1", "a1")
    assert _stored_streak("chat-fail") == 2


def test_classify_failure_propagates_without_retrieval_or_chat(chat_mocks) -> None:
    chat_mocks.classify.side_effect = UpstreamServiceError("Model invocation failed (X)")

    with pytest.raises(UpstreamServiceError):
        answer_query(QueryRequest(question="q1", session_id="classify-fail"))

    chat_mocks.retrieve.assert_not_called()
    chat_mocks.chat_reply.assert_not_called()
    assert _stored_history("classify-fail") is None


def test_stream_query_chat_path_yields_empty_citations_tokens_then_done(chat_mocks) -> None:
    events = list(rag.stream_query(QueryRequest(question="Hi", session_id="st-chat")))

    assert events == [
        ("citations", []),
        ("token", "Hello "),
        ("token", "there!"),
        ("done", {"session_id": "st-chat"}),
    ]
    chat_mocks.stream_chat_reply.assert_called_once_with("Hi", history=[], nudge=False)
    chat_mocks.chat_reply.assert_not_called()
    chat_mocks.stream_answer.assert_not_called()
    chat_mocks.retrieve.assert_not_called()
    chat_mocks.rewrite.assert_not_called()
    assert _stored_history("st-chat") == _turn("Hi", "Hello there!")
    assert _stored_streak("st-chat") == 1


def test_stream_query_capped_chat_sends_limit_message_without_model_call(chat_mocks) -> None:
    with (
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 1),
    ):
        list(rag.stream_query(QueryRequest(question="hi", session_id="st-cap")))
        events = list(rag.stream_query(QueryRequest(question="hi again", session_id="st-cap")))

    assert events == [
        ("citations", []),
        ("token", settings.chat_limit_message),
        ("done", {"session_id": "st-cap"}),
    ]
    chat_mocks.stream_chat_reply.assert_called_once()
    chat_mocks.chat_reply.assert_not_called()
    assert _stored_streak("st-cap") == 2
    assert _stored_history("st-cap") == _turn("hi", "Hello there!")


def test_chat_limit_uses_configured_message_under_invoke_and_stream(chat_mocks) -> None:
    custom = "Let's get back to your offboarding questions."

    with (
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 1),
        patch.object(settings, "chat_limit_message", custom),
    ):
        answer_query(QueryRequest(question="hi", session_id="custom-limit"))
        result = answer_query(QueryRequest(question="hi again", session_id="custom-limit"))
        events = list(rag.stream_query(QueryRequest(question="hey", session_id="custom-limit")))

    assert result.answer == custom
    assert events == [
        ("citations", []),
        ("token", custom),
        ("done", {"session_id": "custom-limit"}),
    ]
    chat_mocks.chat_reply.assert_called_once()
    chat_mocks.stream_chat_reply.assert_not_called()


def test_chat_streak_is_shared_between_blocking_and_streaming_graphs(chat_mocks) -> None:
    with patch.object(settings, "chat_nudge_after", 1):
        answer_query(QueryRequest(question="Hello", session_id="shared"))
        list(rag.stream_query(QueryRequest(question="Hi again", session_id="shared")))

    # The streaming turn saw the streak of 1 left by the blocking turn, so it nudged.
    chat_mocks.stream_chat_reply.assert_called_once_with(
        "Hi again", history=_turn("Hello", "Hi!"), nudge=True
    )
    config = {"configurable": {"thread_id": "shared"}}
    assert rag._rag_stream_graph.get_state(config).values["chat_streak"] == 2


def test_chat_logs_metadata_without_text(chat_mocks, caplog) -> None:
    chat_mocks.chat_reply.return_value = "secret answer"

    def secret_deltas(question, history=None, nudge=False):
        yield "secret streamed answer"

    chat_mocks.stream_chat_reply.side_effect = secret_deltas

    with caplog.at_level("INFO", logger="app.services.rag"):
        answer_query(QueryRequest(question="secret question one", session_id="log-chat"))
        list(rag.stream_query(QueryRequest(question="secret question two", session_id="log-chat")))

    # One summary line per turn, each correlated by session_id and carrying the chat tier.
    messages = [r.getMessage() for r in caplog.records if r.name == "app.services.rag"]
    assert len(messages) == 2
    assert messages[0].startswith("RAG query answered session_id=log-chat ")
    assert "intent=chat chat_tier=normal chat_streak_after=1 " in messages[0]
    assert "citations=0" in messages[0]
    assert messages[1].startswith("RAG query streamed session_id=log-chat ")
    assert "intent=chat chat_tier=normal chat_streak_after=2 " in messages[1]
    assert "tokens=1" in messages[1]
    assert "secret" not in caplog.text


def test_capped_chat_turn_logs_tier_without_text(chat_mocks, caplog) -> None:
    with (
        patch.object(settings, "chat_nudge_after", 1),
        patch.object(settings, "chat_max_consecutive", 1),
        caplog.at_level("INFO", logger="app.services.rag"),
    ):
        answer_query(QueryRequest(question="secret one", session_id="log-cap"))
        list(rag.stream_query(QueryRequest(question="secret two", session_id="log-cap")))
        answer_query(QueryRequest(question="secret three", session_id="log-cap"))

    messages = [r.getMessage() for r in caplog.records if r.name == "app.services.rag"]
    assert len(messages) == 3
    assert "session_id=log-cap" in messages[1]
    assert "chat_tier=capped chat_streak_after=2 " in messages[1]
    assert "tokens=1" in messages[1]
    assert "session_id=log-cap" in messages[2]
    assert "chat_tier=capped chat_streak_after=3 " in messages[2]
    assert "secret" not in caplog.text
    assert settings.chat_limit_message not in caplog.text
