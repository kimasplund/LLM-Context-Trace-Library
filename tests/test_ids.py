"""Tests for LCTL's UUIDv7 id generator."""

import time
import uuid

from lctl.core.ids import _uuid7_fallback, new_id


def test_new_id_is_uuid7():
    value = new_id()
    parsed = uuid.UUID(value)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
    assert str(parsed) == value


def test_new_id_unique():
    values = {new_id() for _ in range(10_000)}
    assert len(values) == 10_000


def test_later_millisecond_sorts_later():
    first = new_id()
    time.sleep(0.002)
    second = new_id()
    assert first < second


def test_fallback_generates_uuid7():
    generated = _uuid7_fallback()
    assert generated.version == 7
    assert generated.variant == uuid.RFC_4122
    timestamp_ms = generated.int >> 80
    assert abs(timestamp_ms - time.time_ns() // 1_000_000) <= 2000


def test_fallback_used_without_stdlib_uuid7(monkeypatch):
    monkeypatch.delattr(uuid, "uuid7", raising=False)
    parsed = uuid.UUID(new_id())
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
