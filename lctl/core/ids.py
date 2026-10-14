"""Time-ordered identifiers (UUIDv7, RFC 9562) for events, spans and traces."""

import os
import time
import uuid

_TIMESTAMP_MASK = (1 << 48) - 1
_RAND_B_MASK = (1 << 62) - 1


def _uuid7_fallback() -> uuid.UUID:
    """UUIDv7 per RFC 9562 section 5.7, for Pythons without uuid.uuid7 (< 3.14)."""
    unix_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 62) & 0xFFF
    value = (
        (unix_ms & _TIMESTAMP_MASK) << 80
        | 0x7 << 76
        | rand_a << 64
        | 0b10 << 62
        | (rand & _RAND_B_MASK)
    )
    return uuid.UUID(int=value)


def new_id() -> str:
    """Return a new UUIDv7 as a lowercase, hyphenated string."""
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())
    return str(_uuid7_fallback())
