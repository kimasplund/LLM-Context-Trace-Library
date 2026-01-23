"""Tests for LCTL sensitive data redaction (lctl/core/redaction.py)."""

import re
from typing import Dict, Pattern

import pytest

from lctl.core.redaction import (
    DEFAULT_PATTERNS,
    REDACTED_VALUE,
    Redactor,
    configure_redaction,
    get_redactor,
    redact,
)
from lctl.core.session import LCTLSession


class TestDefaultPatterns:
    """Tests for default redaction patterns."""

    def test_api_key_pattern(self):
        """Test API key pattern detection."""
        redactor = Redactor()
        test_cases = [
            'api_key="sk-1234567890abcdefghij"',
            "apikey: abcdefghij1234567890ab",
            'API_KEY = "test_key_12345678901234"',
            "api-key='my-secret-api-key-here123'",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_bearer_token_pattern(self):
        """Test Bearer token pattern detection."""
        redactor = Redactor()
        test_cases = [
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "bearer abc123.def456.ghi789",
            "BEARER my_access_token_value",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_password_pattern(self):
        """Test password pattern detection."""
        redactor = Redactor()
        test_cases = [
            'password="supersecret123"',
            "passwd: mypassword",
            'pwd = "p@ssw0rd!"',
            "PASSWORD='secret'",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_secret_pattern(self):
        """Test secret pattern detection."""
        redactor = Redactor()
        test_cases = [
            'secret="abcdefghij1234"',
            "client_secret: my_client_secret_value",
            'SECRET = "verysecretvalue"',
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_token_pattern(self):
        """Test token pattern detection."""
        redactor = Redactor()
        test_cases = [
            'token="abcdefghij12345678901234567890"',
            "access_token: my_access_token_value_here",
            'refresh_token = "refresh_abcdefghij123456"',
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_aws_key_pattern(self):
        """Test AWS key pattern detection."""
        redactor = Redactor()
        test_cases = [
            "AKIAIOSFODNN7EXAMPLE",
            "ASIAISSTILLVALID1234",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_private_key_pattern(self):
        """Test private key pattern detection."""
        redactor = Redactor()
        test_cases = [
            "-----BEGIN RSA PRIVATE KEY-----",
            "-----BEGIN EC PRIVATE KEY-----",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_credit_card_pattern(self):
        """Test credit card number pattern detection."""
        redactor = Redactor()
        test_cases = [
            "4111111111111111",
            "4111-1111-1111-1111",
            "4111 1111 1111 1111",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_ssn_pattern(self):
        """Test SSN pattern detection."""
        redactor = Redactor()
        test_cases = [
            "123-45-6789",
            "SSN: 987-65-4321",
        ]
        for text in test_cases:
            result = redactor.redact_string(text)
            assert REDACTED_VALUE in result, f"Failed for: {text}"

    def test_email_pattern_off_by_default(self):
        """Test that email pattern is off by default."""
        redactor = Redactor()
        text = "Contact: user@example.com"
        result = redactor.redact_string(text)
        assert "user@example.com" in result

    def test_email_pattern_when_enabled(self):
        """Test email pattern when explicitly enabled."""
        redactor = Redactor(redact_emails=True)
        text = "Contact: user@example.com"
        result = redactor.redact_string(text)
        assert REDACTED_VALUE in result


class TestRedactorClass:
    """Tests for Redactor class functionality."""

    def test_redactor_disabled(self):
        """Test redactor when disabled."""
        redactor = Redactor(enabled=False)
        text = 'api_key="secret_value_12345678901234"'
        result = redactor.redact_string(text)
        assert result == text

    def test_redactor_empty_string(self):
        """Test redactor with empty string."""
        redactor = Redactor()
        result = redactor.redact_string("")
        assert result == ""

    def test_redactor_none_handling(self):
        """Test redactor with None-like falsy input."""
        redactor = Redactor()
        result = redactor.redact_string("")
        assert result == ""

    def test_custom_patterns(self):
        """Test redactor with custom patterns only."""
        custom_patterns = {
            "custom": re.compile(r"CUSTOM-\d{6}")
        }
        redactor = Redactor(patterns=custom_patterns)

        text = "ID: CUSTOM-123456 and api_key='should_not_be_redacted'"
        result = redactor.redact_string(text)

        assert "custom=[REDACTED]" in result
        assert "should_not_be_redacted" in result

    def test_additional_patterns(self):
        """Test redactor with additional patterns."""
        additional_patterns = {
            "custom_id": re.compile(r"ID-[A-Z]{3}-\d{4}")
        }
        redactor = Redactor(additional_patterns=additional_patterns)

        text = 'api_key="test12345678901234567890" ID-ABC-1234'
        result = redactor.redact_string(text)

        assert "api_key=[REDACTED]" in result
        assert "custom_id=[REDACTED]" in result


class TestDictRedaction:
    """Tests for dictionary redaction."""

    def test_redact_dict_sensitive_keys(self):
        """Test redaction of values for sensitive keys."""
        redactor = Redactor()
        data = {
            "password": "secret123",
            "api_key": "sk-123456",
            "token": "mytoken",
            "secret": "mysecret",
            "normal_field": "normal_value",
        }
        result = redactor.redact_dict(data)

        assert result["password"] == REDACTED_VALUE
        assert result["api_key"] == REDACTED_VALUE
        assert result["token"] == REDACTED_VALUE
        assert result["secret"] == REDACTED_VALUE
        assert result["normal_field"] == "normal_value"

    def test_redact_dict_suffix_key_match(self):
        """Test redaction with suffix key matches."""
        redactor = Redactor()
        data = {
            "user_password": "secret",
            "db_credential": "dbpass",
            "auth_token": "xyz123",
            "session_key": "sessionvalue",
        }
        result = redactor.redact_dict(data)

        assert result["user_password"] == REDACTED_VALUE
        assert result["db_credential"] == REDACTED_VALUE
        assert result["auth_token"] == REDACTED_VALUE
        assert result["session_key"] == REDACTED_VALUE

    def test_redact_dict_does_not_match_partial_keys(self):
        """Test that keys like 'tokens' are not redacted (not exact match)."""
        redactor = Redactor()
        data = {
            "tokens": {"input": 100, "output": 50},
            "authentication_status": "ok",
        }
        result = redactor.redact_dict(data)

        assert result["tokens"] == {"input": 100, "output": 50}
        assert result["authentication_status"] == "ok"

    def test_redact_dict_nested(self):
        """Test redaction of nested dictionaries."""
        redactor = Redactor()
        data = {
            "config": {
                "database": {
                    "password": "dbpassword",
                    "host": "localhost",
                }
            },
            "api": {
                "key": "normal_key",
                "api_key": "secret_key",
            }
        }
        result = redactor.redact_dict(data)

        assert result["config"]["database"]["password"] == REDACTED_VALUE
        assert result["config"]["database"]["host"] == "localhost"
        assert result["api"]["api_key"] == REDACTED_VALUE
        assert result["api"]["key"] == "normal_key"

    def test_redact_dict_with_list(self):
        """Test redaction of lists within dictionaries."""
        redactor = Redactor()
        data = {
            "users": [
                {"name": "Alice", "password": "alicepass"},
                {"name": "Bob", "password": "bobpass"},
            ],
            "messages": [
                "Contact: 123-45-6789",
                "Normal message",
            ]
        }
        result = redactor.redact_dict(data)

        assert result["users"][0]["password"] == REDACTED_VALUE
        assert result["users"][1]["password"] == REDACTED_VALUE
        assert result["users"][0]["name"] == "Alice"
        assert REDACTED_VALUE in result["messages"][0]
        assert result["messages"][1] == "Normal message"

    def test_redact_dict_max_depth(self):
        """Test redaction respects max depth."""
        redactor = Redactor()
        deep_data = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"l9": {"l10": {"l11": {"password": "deep"}}}}}}}}}}}}
        result = redactor.redact_dict(deep_data, max_depth=5)

        inner = result["l1"]["l2"]["l3"]["l4"]["l5"]["l6"]
        assert inner["l7"]["l8"]["l9"]["l10"]["l11"]["password"] == "deep"

    def test_redact_dict_preserves_non_string_values(self):
        """Test that non-string values are preserved."""
        redactor = Redactor()
        data = {
            "count": 42,
            "enabled": True,
            "ratio": 3.14,
            "items": None,
            "password": "secret",
        }
        result = redactor.redact_dict(data)

        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["ratio"] == 3.14
        assert result["items"] is None
        assert result["password"] == REDACTED_VALUE

    def test_redact_dict_disabled(self):
        """Test dictionary redaction when disabled."""
        redactor = Redactor(enabled=False)
        data = {"password": "secret"}
        result = redactor.redact_dict(data)
        assert result["password"] == "secret"


class TestRedactEventData:
    """Tests for event data redaction."""

    def test_redact_event_data(self):
        """Test redaction of event data structure."""
        redactor = Redactor()
        event_data = {
            "tool": "api_call",
            "input": {
                "url": "https://api.example.com",
                "headers": {
                    "Authorization": "Bearer secret_token",
                    "Content-Type": "application/json",
                }
            },
            "output": {
                "status": 200,
                "body": "Response with api_key='sk-12345678901234567890'"
            }
        }
        result = redactor.redact_event_data(event_data)

        assert result["tool"] == "api_call"
        assert result["input"]["headers"]["Authorization"] == REDACTED_VALUE
        assert REDACTED_VALUE in result["output"]["body"]


class TestGlobalRedactor:
    """Tests for global redactor functions."""

    def test_get_redactor_singleton(self):
        """Test get_redactor returns consistent instance."""
        import lctl.core.redaction as redaction_module
        redaction_module._default_redactor = None

        r1 = get_redactor()
        r2 = get_redactor()
        assert r1 is r2

    def test_configure_redaction(self):
        """Test configure_redaction creates new instance."""
        import lctl.core.redaction as redaction_module

        r1 = configure_redaction(enabled=True, redact_emails=False)
        r2 = configure_redaction(enabled=True, redact_emails=True)

        assert r1 is not r2
        assert "email" not in r1.patterns
        assert "email" in r2.patterns

    def test_configure_redaction_with_additional_patterns(self):
        """Test configure_redaction with additional patterns."""
        custom = {"custom": re.compile(r"CUSTOM-\d+")}
        redactor = configure_redaction(additional_patterns=custom)

        assert "custom" in redactor.patterns
        assert "api_key" in redactor.patterns

    def test_redact_convenience_function_string(self):
        """Test redact() convenience function with string."""
        import lctl.core.redaction as redaction_module
        redaction_module._default_redactor = None
        configure_redaction(enabled=True)

        result = redact('api_key="secret_12345678901234567890"')
        assert REDACTED_VALUE in result

    def test_redact_convenience_function_dict(self):
        """Test redact() convenience function with dict."""
        import lctl.core.redaction as redaction_module
        redaction_module._default_redactor = None
        configure_redaction(enabled=True)

        result = redact({"password": "secret"})
        assert result["password"] == REDACTED_VALUE

    def test_redact_convenience_function_other(self):
        """Test redact() convenience function with other types."""
        result = redact(42)
        assert result == 42

        result = redact(None)
        assert result is None

        result = redact(["a", "b"])
        assert result == ["a", "b"]


class TestSessionRedactionIntegration:
    """Tests for redaction integration with LCTLSession."""

    def test_session_redacts_tool_call_data(self):
        """Test that session redacts sensitive data in tool calls."""
        session = LCTLSession(chain_id="redact-test", redaction_enabled=True)
        session.tool_call(
            tool="api_request",
            input_data={
                "url": "https://api.example.com",
                "headers": {"Authorization": "Bearer my_secret_token"},
            },
            output_data={"status": 200},
            duration_ms=100
        )

        event = session.chain.events[0]
        assert event.data["input"]["headers"]["Authorization"] == REDACTED_VALUE

    def test_session_redacts_step_data(self):
        """Test that session redacts sensitive data in step events."""
        session = LCTLSession(chain_id="redact-test", redaction_enabled=True)
        session.step_start("agent", "call_api", "api_key='sk-12345678901234567890'")

        event = session.chain.events[0]
        assert REDACTED_VALUE in event.data["input_summary"]

    def test_session_redacts_error_data(self):
        """Test that session redacts sensitive data in error events."""
        session = LCTLSession(chain_id="redact-test", redaction_enabled=True)
        session.error(
            category="auth",
            error_type="AuthError",
            message='Invalid token: Bearer abc123.def456.ghi789',
            recoverable=False
        )

        event = session.chain.events[0]
        assert REDACTED_VALUE in event.data["message"]

    def test_session_redaction_disabled(self):
        """Test that redaction can be disabled per-session."""
        session = LCTLSession(chain_id="no-redact", redaction_enabled=False)
        session.tool_call(
            tool="test",
            input_data={"password": "mysecret"},
            output_data={},
            duration_ms=0
        )

        event = session.chain.events[0]
        assert event.data["input"]["password"] == "mysecret"

    def test_session_redaction_default_enabled(self):
        """Test that redaction is enabled by default."""
        session = LCTLSession(chain_id="default-test")
        assert session._redaction_enabled is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_redact_string_with_multiple_patterns(self):
        """Test redacting string with multiple sensitive items."""
        redactor = Redactor()
        text = 'api_key="sk-123456789012345678901234" password="secret" SSN: 123-45-6789'
        result = redactor.redact_string(text)

        assert result.count(REDACTED_VALUE) >= 3

    def test_redact_dict_with_empty_nested_structures(self):
        """Test redacting dict with empty nested structures."""
        redactor = Redactor()
        data = {
            "empty_dict": {},
            "empty_list": [],
            "nested": {
                "also_empty": {}
            }
        }
        result = redactor.redact_dict(data)

        assert result["empty_dict"] == {}
        assert result["empty_list"] == []
        assert result["nested"]["also_empty"] == {}

    def test_redact_dict_with_mixed_list(self):
        """Test redacting dict with list containing mixed types."""
        redactor = Redactor()
        data = {
            "mixed": [
                "string with 123-45-6789",
                42,
                {"password": "secret"},
                None,
                True,
                ["nested", "list"],
            ]
        }
        result = redactor.redact_dict(data)

        assert REDACTED_VALUE in result["mixed"][0]
        assert result["mixed"][1] == 42
        assert result["mixed"][2]["password"] == REDACTED_VALUE
        assert result["mixed"][3] is None
        assert result["mixed"][4] is True
        assert result["mixed"][5] == ["nested", "list"]

    def test_redact_preserves_dict_key_order(self):
        """Test that redaction preserves dictionary key order."""
        redactor = Redactor()
        data = {"z": 1, "a": 2, "m": 3, "password": "secret"}
        result = redactor.redact_dict(data)

        assert list(result.keys()) == ["z", "a", "m", "password"]
