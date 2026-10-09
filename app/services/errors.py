"""Service-layer exceptions, mapped to HTTP responses by the handler in `app.main`.

AWS error mapping (see `from_aws_error`):
- botocore `ClientError` whose code is transient (throttling, service unavailable, model not
  ready) -> `ServiceUnavailableError` (503), so callers know a retry later may succeed.
- botocore `NoCredentialsError` / `PartialCredentialsError` -> `ServiceUnavailableError`
  (503): a deployment config problem, like a missing Knowledge Base id.
- Any other `ClientError` (access denied, validation, resource not found, quota exceeded,
  internal error, ...) or `BotoCoreError` (connection/timeout, ...) ->
  `UpstreamServiceError` (502).
- Exception events inside a `converse_stream` event stream arrive as `EventStreamError` (a
  `ClientError`) and follow the same rules once their code is normalized.
Messages are generic and include only the AWS error code, never raw AWS messages.
"""

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EventStreamError,
    NoCredentialsError,
    PartialCredentialsError,
)

_RETRYABLE_AWS_CODES = frozenset(
    {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
)


class ServiceError(Exception):
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ServiceUnavailableError(ServiceError):
    """Dependency temporarily unavailable or not configured (503)."""

    status_code = 503


class UpstreamServiceError(ServiceError):
    """An upstream (AWS Bedrock) call failed or returned an unusable response (502)."""

    status_code = 502


def aws_error_code(exc: BotoCoreError | ClientError) -> str:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code") or "Unknown"
        if isinstance(exc, EventStreamError):
            # Event-stream errors use the event member name as the code, e.g.
            # "throttlingException"; normalize to the usual "ThrottlingException".
            code = code[0].upper() + code[1:]
        return code
    return type(exc).__name__


def from_aws_error(exc: BotoCoreError | ClientError, operation: str) -> ServiceError:
    code = aws_error_code(exc)
    if isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
        return ServiceUnavailableError(f"{operation} is not configured ({code})")
    if isinstance(exc, ClientError) and code in _RETRYABLE_AWS_CODES:
        return ServiceUnavailableError(f"{operation} is temporarily unavailable ({code})")
    return UpstreamServiceError(f"{operation} failed ({code})")
