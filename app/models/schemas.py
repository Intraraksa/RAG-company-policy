from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, StringConstraints

MAX_QUESTION_LENGTH = 4000
MAX_SESSION_ID_LENGTH = 128
SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"


class ChatTurn(TypedDict):
    """One stored conversation message (internal; persisted in the graph checkpoint)."""

    role: Literal["user", "assistant"]
    content: str


class Citation(BaseModel):
    content: str
    source_uri: str | None = None
    score: float | None = None


class QueryRequest(BaseModel):
    # Whitespace is stripped before the length checks, so blank questions are rejected (422).
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH),
    ]
    # Constrained because it is echoed back and logged (prevents log injection / huge values).
    session_id: (
        Annotated[
            str,
            StringConstraints(max_length=MAX_SESSION_ID_LENGTH, pattern=SESSION_ID_PATTERN),
        ]
        | None
    ) = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    session_id: str | None = None
