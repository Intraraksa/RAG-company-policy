import pytest
from pydantic import ValidationError

from app.core.config import DEFAULT_ASSISTANT_SCOPE, DEFAULT_CHAT_LIMIT_MESSAGE, Settings

_SETTINGS_ENV_VARS = (
    "AWS_REGION",
    "BEDROCK_MODEL_ID",
    "BEDROCK_KNOWLEDGE_BASE_ID",
    "BEDROCK_KB_RETRIEVAL_RESULTS",
    "BEDROCK_MAX_TOKENS",
    "BEDROCK_TEMPERATURE",
    "SYSTEM_PROMPT",
    "LOG_LEVEL",
    "HISTORY_MAX_TURNS",
    "ASSISTANT_SCOPE",
    "CHAT_NUDGE_AFTER",
    "CHAT_MAX_CONSECUTIVE",
    "CHAT_LIMIT_MESSAGE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_ignore_unrelated_env_file_entries(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ACCESS_KEY_ID=dummy-key-id\nAWS_SECRET_ACCESS_KEY=dummy-secret\n", encoding="utf-8"
    )

    settings = Settings(_env_file=env_file)

    assert settings.bedrock_knowledge_base_id == ""
    assert not hasattr(settings, "aws_access_key_id")


def test_settings_defaults(tmp_path) -> None:
    settings = Settings(_env_file=tmp_path / "missing.env")

    assert settings.bedrock_max_tokens == 1024
    assert settings.bedrock_temperature == 0.0
    assert settings.log_level == "INFO"
    assert settings.history_max_turns == 5
    assert settings.assistant_scope == DEFAULT_ASSISTANT_SCOPE
    assert settings.chat_nudge_after == 3
    assert settings.chat_max_consecutive == 6
    assert settings.chat_limit_message == DEFAULT_CHAT_LIMIT_MESSAGE


def test_default_assistant_scope_describes_offboarding_domain() -> None:
    for topic in ("left the company", "benefits", "final pay", "returning equipment"):
        assert topic in DEFAULT_ASSISTANT_SCOPE


def test_settings_router_and_chat_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ASSISTANT_SCOPE", "Custom scope.")
    monkeypatch.setenv("CHAT_NUDGE_AFTER", "2")
    monkeypatch.setenv("CHAT_MAX_CONSECUTIVE", "4")

    settings = Settings(_env_file=tmp_path / "missing.env")

    assert settings.assistant_scope == "Custom scope."
    assert settings.chat_nudge_after == 2
    assert settings.chat_max_consecutive == 4


def test_settings_chat_limit_message_env_override_is_stripped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHAT_LIMIT_MESSAGE", "  Ask me about your final pay.  ")

    settings = Settings(_env_file=tmp_path / "missing.env")

    assert settings.chat_limit_message == "Ask me about your final pay."


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_settings_chat_limit_message_rejects_blank(tmp_path, monkeypatch, value) -> None:
    monkeypatch.setenv("CHAT_LIMIT_MESSAGE", value)

    with pytest.raises(ValidationError, match="chat_limit_message"):
        Settings(_env_file=tmp_path / "missing.env")


def test_settings_chat_max_below_nudge_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError, match="chat_max_consecutive"):
        Settings(_env_file=tmp_path / "missing.env", chat_nudge_after=4, chat_max_consecutive=3)


def test_settings_chat_max_equal_to_nudge_accepted(tmp_path) -> None:
    settings = Settings(
        _env_file=tmp_path / "missing.env", chat_nudge_after=4, chat_max_consecutive=4
    )

    assert settings.chat_nudge_after == settings.chat_max_consecutive == 4


@pytest.mark.parametrize(("value", "expected"), [("debug", "DEBUG"), ("Warning", "WARNING")])
def test_settings_log_level_is_case_insensitive(tmp_path, monkeypatch, value, expected) -> None:
    monkeypatch.setenv("LOG_LEVEL", value)

    assert Settings(_env_file=tmp_path / "missing.env").log_level == expected


def test_settings_log_level_rejects_unknown(tmp_path) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=tmp_path / "missing.env", log_level="verbose")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("history_max_turns", 0),
        ("bedrock_max_tokens", 0),
        ("bedrock_temperature", -0.1),
        ("bedrock_temperature", 1.5),
        ("chat_nudge_after", 0),
        ("chat_max_consecutive", 0),
    ],
)
def test_settings_bounds(tmp_path, field, value) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=tmp_path / "missing.env", **{field: value})
