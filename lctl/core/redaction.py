"""Sensitive data redaction for LCTL traces."""

import re
from typing import Any, Dict, List, Optional, Pattern, Set

DEFAULT_PATTERNS: Dict[str, Pattern] = {
    "api_key": re.compile(r'(?i)(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?'),
    "bearer_token": re.compile(r'(?i)bearer\s+([a-zA-Z0-9_.-]+)'),
    "password": re.compile(r'(?i)(password|passwd|pwd)["\s:=]+["\']?([^\s"\',}{]+)["\']?'),
    "secret": re.compile(r'(?i)(secret|client_secret)["\s:=]+["\']?([a-zA-Z0-9_-]{10,})["\']?'),
    "token": re.compile(r'(?i)(token|access_token|refresh_token)["\s:=]+["\']?([a-zA-Z0-9_.-]{20,})["\']?'),
    "aws_key": re.compile(r'(?i)(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}'),
    "private_key": re.compile(r'-----BEGIN [A-Z]+ PRIVATE KEY-----'),
    "credit_card": re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
}

REDACTED_VALUE = "[REDACTED]"


class Redactor:
    """Redacts sensitive data from strings and dictionaries."""

    def __init__(
        self,
        patterns: Optional[Dict[str, Pattern]] = None,
        additional_patterns: Optional[Dict[str, Pattern]] = None,
        redact_emails: bool = False,
        enabled: bool = True
    ):
        """Initialize the redactor.

        Args:
            patterns: Custom patterns to use instead of defaults.
            additional_patterns: Additional patterns to add to defaults.
            redact_emails: Whether to redact email addresses (off by default).
            enabled: Whether redaction is enabled.
        """
        self.enabled = enabled
        self.patterns = patterns or DEFAULT_PATTERNS.copy()
        if additional_patterns:
            self.patterns.update(additional_patterns)
        if not redact_emails:
            self.patterns.pop("email", None)

    def redact_string(self, text: str) -> str:
        """Redact sensitive data from a string.

        Args:
            text: The string to redact.

        Returns:
            The redacted string.
        """
        if not self.enabled or not text:
            return text
        result = text
        for name, pattern in self.patterns.items():
            result = pattern.sub(f'{name}={REDACTED_VALUE}', result)
        return result

    def redact_dict(self, data: Dict[str, Any], depth: int = 0, max_depth: int = 10) -> Dict[str, Any]:
        """Recursively redact sensitive data from a dictionary.

        Args:
            data: The dictionary to redact.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth.

        Returns:
            The redacted dictionary.
        """
        if not self.enabled or depth > max_depth:
            return data

        result = {}
        sensitive_key_exact = {'password', 'secret', 'token', 'api_key', 'apikey',
                               'authorization', 'credential', 'private_key',
                               'access_token', 'refresh_token', 'bearer_token',
                               'client_secret', 'auth_token', 'session_token'}
        sensitive_key_suffix = {'_password', '_secret', '_token', '_key', '_credential',
                                '_auth'}

        for key, value in data.items():
            key_lower = key.lower()

            is_sensitive = (
                key_lower in sensitive_key_exact or
                any(key_lower.endswith(suffix) for suffix in sensitive_key_suffix)
            )
            if is_sensitive:
                result[key] = REDACTED_VALUE
            elif isinstance(value, str):
                result[key] = self.redact_string(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value, depth + 1, max_depth)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(item, depth + 1, max_depth) if isinstance(item, dict)
                    else self.redact_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def redact_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from an LCTL event's data field.

        Args:
            event_data: The event data dictionary to redact.

        Returns:
            The redacted event data.
        """
        return self.redact_dict(event_data)


_default_redactor: Optional[Redactor] = None


def get_redactor() -> Redactor:
    """Get the global redactor instance.

    Returns:
        The global Redactor instance.
    """
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = Redactor()
    return _default_redactor


def configure_redaction(
    enabled: bool = True,
    additional_patterns: Optional[Dict[str, Pattern]] = None,
    redact_emails: bool = False
) -> Redactor:
    """Configure the global redactor.

    Args:
        enabled: Whether redaction is enabled.
        additional_patterns: Additional patterns to add to defaults.
        redact_emails: Whether to redact email addresses.

    Returns:
        The configured Redactor instance.
    """
    global _default_redactor
    _default_redactor = Redactor(
        additional_patterns=additional_patterns,
        redact_emails=redact_emails,
        enabled=enabled
    )
    return _default_redactor


def redact(data: Any) -> Any:
    """Convenience function to redact data using global redactor.

    Args:
        data: String or dictionary to redact.

    Returns:
        The redacted data.
    """
    redactor = get_redactor()
    if isinstance(data, str):
        return redactor.redact_string(data)
    elif isinstance(data, dict):
        return redactor.redact_dict(data)
    return data
