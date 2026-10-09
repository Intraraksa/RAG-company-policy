import logging
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.models.schemas import Citation
from app.services.errors import ServiceUnavailableError, aws_error_code, from_aws_error

logger = logging.getLogger(__name__)

_client = boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)


def retrieve(question: str) -> list[Citation]:
    kb_id = settings.bedrock_knowledge_base_id
    if not kb_id:
        logger.error("Knowledge Base retrieval requested but BEDROCK_KNOWLEDGE_BASE_ID is unset")
        raise ServiceUnavailableError("Knowledge Base is not configured")

    start = time.perf_counter()
    try:
        response = _client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                f"{settings.bedrock_kb_search_type}SearchConfiguration": {
                    "numberOfResults": settings.bedrock_kb_retrieval_results
                }
            },
        )
    except (ClientError, BotoCoreError) as exc:
        # The AWS message stays in server logs only (clients get just the code); it is the
        # only place that says what was invalid, e.g. for a ValidationException.
        aws_message = (
            exc.response.get("Error", {}).get("Message") if isinstance(exc, ClientError) else None
        )
        logger.error(
            "Knowledge Base retrieve failed kb_id=%s code=%s message=%s",
            kb_id,
            aws_error_code(exc),
            aws_message,
        )
        raise from_aws_error(exc, "Knowledge Base retrieval") from exc

    citations = []
    skipped = 0
    for result in response.get("retrievalResults", []):
        # Non-text results (IMAGE/ROW/AUDIO/VIDEO content) carry no text to ground on.
        text = (result.get("content") or {}).get("text")
        if not text:
            skipped += 1
            continue
        citations.append(
            Citation(
                content=text,
                source_uri=(result.get("location") or {}).get("s3Location", {}).get("uri"),
                score=result.get("score"),
            )
        )
    logger.info(
        "Knowledge Base retrieve kb_id=%s results=%d skipped_non_text=%d elapsed_ms=%.0f",
        kb_id,
        len(citations),
        skipped,
        (time.perf_counter() - start) * 1000,
    )
    return citations
