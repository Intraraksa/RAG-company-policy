from unittest.mock import patch

import pytest

from app.services import bedrock_llm, rag


@pytest.fixture(autouse=True)
def fresh_rag_graph():
    """Give each test its own checkpointer so session history never leaks between tests.

    Both graphs share it, as in production, so /query and /query/stream share sessions.
    """
    graph, stream_graph = rag.build_graphs(rag.new_checkpointer())
    with (
        patch.object(rag, "_rag_graph", graph),
        patch.object(rag, "_rag_stream_graph", stream_graph),
    ):
        yield


@pytest.fixture(autouse=True)
def classify_intent():
    """Route every turn to the knowledge path unless a test says otherwise.

    Tests that need the chat path request this fixture and set `return_value` / `side_effect`.
    A module that needs the real `classify_intent` overrides this fixture by name.
    """
    with patch.object(bedrock_llm, "classify_intent", return_value="knowledge") as mock:
        yield mock
