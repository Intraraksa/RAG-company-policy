from unittest.mock import patch

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

from app.core.config import settings
from app.services import bedrock_kb
from app.services.errors import ServiceUnavailableError, UpstreamServiceError


@pytest.fixture
def kb_settings():
    with (
        patch.object(settings, "bedrock_knowledge_base_id", "kb-test"),
        patch.object(settings, "bedrock_kb_retrieval_results", 3),
        patch.object(settings, "bedrock_kb_search_type", "vector"),
    ):
        yield


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "secret detail"}}, "Retrieve")


def test_retrieve_maps_results(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "chunk one"},
                    "location": {"s3Location": {"uri": "s3://bucket/a.txt"}},
                    "score": 0.9,
                },
                {"content": {"text": "chunk two"}},
            ]
        }
        citations = bedrock_kb.retrieve("What?")

    client.retrieve.assert_called_once_with(
        knowledgeBaseId="kb-test",
        retrievalQuery={"text": "What?"},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
    )
    assert [c.content for c in citations] == ["chunk one", "chunk two"]
    assert citations[0].source_uri == "s3://bucket/a.txt"
    assert citations[0].score == 0.9
    assert citations[1].source_uri is None
    assert citations[1].score is None


def test_retrieve_managed_kb_uses_managed_search_configuration(kb_settings) -> None:
    with (
        patch.object(settings, "bedrock_kb_search_type", "managed"),
        patch.object(bedrock_kb, "_client") as client,
    ):
        client.retrieve.return_value = {}
        bedrock_kb.retrieve("What?")

    client.retrieve.assert_called_once_with(
        knowledgeBaseId="kb-test",
        retrievalQuery={"text": "What?"},
        retrievalConfiguration={"managedSearchConfiguration": {"numberOfResults": 3}},
    )


def test_retrieve_empty_results(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.return_value = {}
        assert bedrock_kb.retrieve("What?") == []


def test_retrieve_missing_kb_id_raises_without_calling_aws() -> None:
    with (
        patch.object(settings, "bedrock_knowledge_base_id", ""),
        patch.object(bedrock_kb, "_client") as client,
        pytest.raises(ServiceUnavailableError) as exc_info,
    ):
        bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 503
    client.retrieve.assert_not_called()


def test_retrieve_client_error_is_wrapped(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.side_effect = _client_error("AccessDeniedException")
        with pytest.raises(UpstreamServiceError) as exc_info:
            bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 502
    assert "AccessDeniedException" in exc_info.value.message
    assert "secret detail" not in exc_info.value.message


def test_retrieve_throttling_maps_to_503(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.side_effect = _client_error("ThrottlingException")
        with pytest.raises(ServiceUnavailableError) as exc_info:
            bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 503


def test_retrieve_skips_non_text_results(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.return_value = {
            "retrievalResults": [
                {"content": {"type": "IMAGE", "byteContent": "data:image/png;base64,AAAA"}},
                {"content": {"type": "ROW", "row": [{"columnName": "a", "type": "STRING"}]}},
                {"content": None},
                {"location": {"type": "S3"}},
                {"content": {"type": "TEXT", "text": "kept"}, "location": None, "score": 0.5},
            ]
        }
        citations = bedrock_kb.retrieve("What?")

    assert [(c.content, c.source_uri, c.score) for c in citations] == [("kept", None, 0.5)]


def test_retrieve_quota_exceeded_maps_to_502(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.side_effect = _client_error("ServiceQuotaExceededException")
        with pytest.raises(UpstreamServiceError) as exc_info:
            bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 502


def test_retrieve_botocore_error_is_wrapped(kb_settings) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.side_effect = EndpointConnectionError(endpoint_url="https://example")
        with pytest.raises(UpstreamServiceError) as exc_info:
            bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 502


@pytest.mark.parametrize(
    "error", [NoCredentialsError(), PartialCredentialsError(provider="env", cred_var="x")]
)
def test_retrieve_missing_credentials_maps_to_503(kb_settings, error) -> None:
    with patch.object(bedrock_kb, "_client") as client:
        client.retrieve.side_effect = error
        with pytest.raises(ServiceUnavailableError) as exc_info:
            bedrock_kb.retrieve("What?")

    assert exc_info.value.status_code == 503
    assert type(error).__name__ in exc_info.value.message
