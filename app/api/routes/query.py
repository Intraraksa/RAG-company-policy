import logging
from collections.abc import Iterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.models.schemas import QueryRequest, QueryResponse
from app.services.errors import ServiceError
from app.services.rag import StreamEvent, answer_query, stream_query

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return answer_query(request)


def _to_sse(event: StreamEvent) -> ServerSentEvent:
    kind, payload = event
    if kind == "citations":
        return ServerSentEvent(event=kind, data=[c.model_dump(mode="json") for c in payload])
    if kind == "token":
        return ServerSentEvent(event=kind, data={"text": payload})
    return ServerSentEvent(event=kind, data=payload)


# Sync generator: FastAPI iterates it in a threadpool, so blocking Bedrock calls don't stall
# the event loop. Errors after the 200 response has started are sent as a final `error`
# event (no `done`), since the ServiceError handler can no longer set the status code.
@router.post("/query/stream", response_class=EventSourceResponse)
def query_stream(request: QueryRequest) -> Iterable[ServerSentEvent]:
    try:
        for event in stream_query(request):
            yield _to_sse(event)
    except ServiceError as exc:
        yield ServerSentEvent(
            event="error", data={"detail": exc.message, "status_code": exc.status_code}
        )
    except Exception:
        logger.exception("Unexpected error while streaming a query")
        yield ServerSentEvent(
            event="error", data={"detail": "Internal server error", "status_code": 500}
        )
