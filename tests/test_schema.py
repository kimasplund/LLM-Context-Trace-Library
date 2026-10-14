"""Tests for LCTL schema versioning (lctl/core/schema.py)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from lctl.core.events import Chain, Event, EventType
from lctl.core.schema import (
    CURRENT_VERSION,
    MIN_SUPPORTED_VERSION,
    VERSION_HISTORY,
    SchemaMigrator,
    SchemaVersionError,
    get_version_info,
    validate_version,
)


class TestVersionConstants:
    """Tests for version constants."""

    def test_current_version_is_string(self):
        """CURRENT_VERSION should be a string."""
        assert isinstance(CURRENT_VERSION, str)
        assert CURRENT_VERSION == "4.1"

    def test_min_supported_version_is_string(self):
        """MIN_SUPPORTED_VERSION should be a string."""
        assert isinstance(MIN_SUPPORTED_VERSION, str)
        assert MIN_SUPPORTED_VERSION == "3.0"

    def test_version_history_is_dict(self):
        """VERSION_HISTORY should be a dict with version keys."""
        assert isinstance(VERSION_HISTORY, dict)
        assert "3.0" in VERSION_HISTORY
        assert "4.0" in VERSION_HISTORY
        assert "4.1" in VERSION_HISTORY

    def test_current_version_in_history(self):
        """CURRENT_VERSION should be in VERSION_HISTORY."""
        assert CURRENT_VERSION in VERSION_HISTORY

    def test_min_supported_version_in_history(self):
        """MIN_SUPPORTED_VERSION should be in VERSION_HISTORY."""
        assert MIN_SUPPORTED_VERSION in VERSION_HISTORY


class TestSchemaVersionError:
    """Tests for SchemaVersionError exception."""

    def test_error_creation(self):
        """Test SchemaVersionError creation with versions."""
        error = SchemaVersionError("2.0", "3.0")
        assert error.found_version == "2.0"
        assert error.required_version == "3.0"
        assert "2.0" in str(error)
        assert "3.0" in str(error)

    def test_error_with_custom_message(self):
        """Test SchemaVersionError with custom message."""
        error = SchemaVersionError("2.0", "3.0", "Custom error message")
        assert str(error) == "Custom error message"

    def test_error_default_message(self):
        """Test SchemaVersionError default message format."""
        error = SchemaVersionError("1.0", "3.0")
        assert "1.0" in str(error)
        assert "incompatible" in str(error).lower()


class TestValidateVersion:
    """Tests for validate_version function."""

    def test_validate_current_version(self):
        """Current version should be valid."""
        data = {"lctl": CURRENT_VERSION}
        validate_version(data)  # Should not raise

    def test_validate_min_supported_version(self):
        """Minimum supported version should be valid."""
        data = {"lctl": MIN_SUPPORTED_VERSION}
        validate_version(data)  # Should not raise

    def test_validate_version_4_0(self):
        """Version 4.0 should be valid."""
        data = {"lctl": "4.0"}
        validate_version(data)  # Should not raise

    def test_validate_missing_version_defaults_to_3_0(self):
        """Missing version should default to 3.0 (valid)."""
        data = {"events": []}
        validate_version(data)  # Should not raise (defaults to "3.0")

    def test_validate_old_version_raises(self):
        """Version below minimum should raise SchemaVersionError."""
        data = {"lctl": "2.0"}
        with pytest.raises(SchemaVersionError) as exc_info:
            validate_version(data)
        assert exc_info.value.found_version == "2.0"
        assert exc_info.value.required_version == MIN_SUPPORTED_VERSION

    def test_validate_very_old_version_raises(self):
        """Very old version should raise SchemaVersionError."""
        data = {"lctl": "1.0"}
        with pytest.raises(SchemaVersionError) as exc_info:
            validate_version(data)
        assert exc_info.value.found_version == "1.0"


class TestSchemaMigrator:
    """Tests for SchemaMigrator class."""

    def test_migrate_current_version_no_change(self):
        """Data at current version should not be modified."""
        data = {"lctl": CURRENT_VERSION, "chain": {"id": "test"}, "events": []}
        result = SchemaMigrator.migrate(data)
        assert result["lctl"] == CURRENT_VERSION

    def test_migrate_from_3_0_to_current(self):
        """Migration from 3.0 should update to current version."""
        data = {"lctl": "3.0", "chain": {"id": "test"}, "events": []}
        result = SchemaMigrator.migrate(data)
        assert result["lctl"] == CURRENT_VERSION

    def test_migrate_from_4_0_to_current(self):
        """Migration from 4.0 should update to current version."""
        data = {"lctl": "4.0", "chain": {"id": "test"}, "events": []}
        result = SchemaMigrator.migrate(data)
        assert result["lctl"] == CURRENT_VERSION

    def test_migrate_preserves_chain_data(self):
        """Migration should preserve chain and event data."""
        data = {
            "lctl": "3.0",
            "chain": {"id": "my-chain"},
            "events": [
                {
                    "seq": 1,
                    "type": "step_start",
                    "timestamp": "2025-01-15T10:00:00",
                    "agent": "agent1",
                    "data": {"intent": "test"},
                }
            ],
        }
        result = SchemaMigrator.migrate(data)
        assert result["chain"]["id"] == "my-chain"
        assert len(result["events"]) == 1
        assert result["events"][0]["agent"] == "agent1"

    def test_migrate_to_specific_version(self):
        """Migration to specific version should stop at that version."""
        data = {"lctl": "3.0", "chain": {"id": "test"}, "events": []}
        result = SchemaMigrator.migrate(data, target_version="4.0")
        assert result["lctl"] == "4.0"

    def test_migrate_already_above_target(self):
        """Data above target version should not be modified."""
        data = {"lctl": "4.1", "chain": {"id": "test"}, "events": []}
        result = SchemaMigrator.migrate(data, target_version="4.0")
        assert result["lctl"] == "4.1"

    def test_get_migration_path_3_to_4_1(self):
        """Migration path from 3.0 to 4.1 should include all steps."""
        path = SchemaMigrator._get_migration_path("3.0", "4.1")
        assert ("3.0", "4.0") in path
        assert ("4.0", "4.1") in path

    def test_get_migration_path_4_0_to_4_1(self):
        """Migration path from 4.0 to 4.1 should be single step."""
        path = SchemaMigrator._get_migration_path("4.0", "4.1")
        assert path == [("4.0", "4.1")]

    def test_registered_migrations(self):
        """Check that expected migrations are registered."""
        migrations = SchemaMigrator.get_registered_migrations()
        assert "3.0->4.0" in migrations
        assert "4.0->4.1" in migrations


class TestGetVersionInfo:
    """Tests for get_version_info function."""

    def test_returns_dict(self):
        """get_version_info should return a dictionary."""
        info = get_version_info()
        assert isinstance(info, dict)

    def test_contains_current_version(self):
        """Info should contain current version."""
        info = get_version_info()
        assert "current" in info
        assert info["current"] == CURRENT_VERSION

    def test_contains_minimum_supported(self):
        """Info should contain minimum supported version."""
        info = get_version_info()
        assert "minimum_supported" in info
        assert info["minimum_supported"] == MIN_SUPPORTED_VERSION

    def test_contains_history(self):
        """Info should contain version history."""
        info = get_version_info()
        assert "history" in info
        assert info["history"] == VERSION_HISTORY


class TestChainLoadWithVersioning:
    """Tests for Chain loading with schema versioning."""

    def test_chain_from_dict_with_3_0(self):
        """Chain.from_dict should accept version 3.0 and migrate."""
        data = {
            "lctl": "3.0",
            "chain": {"id": "old-chain"},
            "events": [],
        }
        chain = Chain.from_dict(data)
        assert chain.id == "old-chain"
        assert chain.version == CURRENT_VERSION

    def test_chain_from_dict_with_4_0(self):
        """Chain.from_dict should accept version 4.0 and migrate."""
        data = {
            "lctl": "4.0",
            "chain": {"id": "v4-chain"},
            "events": [],
        }
        chain = Chain.from_dict(data)
        assert chain.id == "v4-chain"
        assert chain.version == CURRENT_VERSION

    def test_chain_from_dict_with_current_version(self):
        """Chain.from_dict should accept current version."""
        data = {
            "lctl": CURRENT_VERSION,
            "chain": {"id": "current-chain"},
            "events": [],
        }
        chain = Chain.from_dict(data)
        assert chain.id == "current-chain"
        assert chain.version == CURRENT_VERSION

    def test_chain_from_dict_rejects_old_version(self):
        """Chain.from_dict should reject versions below minimum."""
        data = {
            "lctl": "2.0",
            "chain": {"id": "ancient-chain"},
            "events": [],
        }
        with pytest.raises(SchemaVersionError) as exc_info:
            Chain.from_dict(data)
        assert exc_info.value.found_version == "2.0"

    def test_chain_load_file_with_old_version(self, tmp_path: Path):
        """Chain.load should handle old version files."""
        data = {
            "lctl": "3.0",
            "chain": {"id": "file-chain"},
            "events": [
                {
                    "seq": 1,
                    "type": "step_start",
                    "timestamp": "2025-01-15T10:00:00",
                    "agent": "agent1",
                    "data": {},
                }
            ],
        }
        file_path = tmp_path / "old_version.lctl.json"
        file_path.write_text(json.dumps(data))

        chain = Chain.load(file_path)
        assert chain.id == "file-chain"
        assert chain.version == CURRENT_VERSION
        assert len(chain.events) == 1

    def test_chain_load_file_rejects_unsupported_version(self, tmp_path: Path):
        """Chain.load should reject unsupported version files."""
        data = {
            "lctl": "1.5",
            "chain": {"id": "unsupported-chain"},
            "events": [],
        }
        file_path = tmp_path / "unsupported.lctl.json"
        file_path.write_text(json.dumps(data))

        with pytest.raises(SchemaVersionError):
            Chain.load(file_path)


class TestChainSaveWithVersioning:
    """Tests for Chain saving with schema versioning."""

    def test_chain_saves_with_current_version(self, tmp_path: Path):
        """New chains should save with current version."""
        chain = Chain(id="new-chain")
        file_path = tmp_path / "new.lctl.json"
        chain.save(file_path)

        content = json.loads(file_path.read_text())
        assert content["lctl"] == CURRENT_VERSION

    def test_chain_to_dict_uses_current_version(self):
        """Chain.to_dict should output current version."""
        chain = Chain(id="test")
        result = chain.to_dict()
        assert result["lctl"] == CURRENT_VERSION

    def test_chain_roundtrip_preserves_events(self, tmp_path: Path, base_timestamp: datetime):
        """Save/load roundtrip should preserve events."""
        chain = Chain(id="roundtrip")
        chain.add_event(
            Event(
                seq=1,
                type=EventType.STEP_START,
                timestamp=base_timestamp,
                agent="agent1",
                data={"intent": "test"},
            )
        )
        chain.add_event(
            Event(
                seq=2,
                type=EventType.FACT_ADDED,
                timestamp=base_timestamp,
                agent="agent1",
                data={"id": "F1", "text": "fact", "confidence": 0.9},
            )
        )

        file_path = tmp_path / "roundtrip.lctl.json"
        chain.save(file_path)
        loaded = Chain.load(file_path)

        assert loaded.id == "roundtrip"
        assert loaded.version == CURRENT_VERSION
        assert len(loaded.events) == 2
        assert loaded.events[0].type == EventType.STEP_START
        assert loaded.events[1].data["confidence"] == 0.9


class TestMigrationIntegrity:
    """Tests for migration data integrity."""

    def test_migration_preserves_complex_events(self):
        """Migration should preserve complex event data."""
        data = {
            "lctl": "3.0",
            "chain": {"id": "complex"},
            "events": [
                {
                    "seq": 1,
                    "type": "step_start",
                    "timestamp": "2025-01-15T10:00:00",
                    "agent": "planner",
                    "data": {
                        "intent": "analyze",
                        "input_summary": "complex data",
                        "metadata": {"key": "value", "nested": {"a": 1}},
                    },
                },
                {
                    "seq": 2,
                    "type": "fact_added",
                    "timestamp": "2025-01-15T10:00:01",
                    "agent": "analyzer",
                    "data": {
                        "id": "F1",
                        "text": "Important finding",
                        "confidence": 0.85,
                        "source": "analyzer",
                    },
                },
                {
                    "seq": 3,
                    "type": "error",
                    "timestamp": "2025-01-15T10:00:02",
                    "agent": "executor",
                    "data": {
                        "category": "validation",
                        "type": "ValueError",
                        "message": "Test error",
                        "recoverable": True,
                    },
                },
            ],
        }

        result = SchemaMigrator.migrate(data)

        assert result["lctl"] == CURRENT_VERSION
        assert len(result["events"]) == 3
        assert result["events"][0]["data"]["metadata"]["nested"]["a"] == 1
        assert result["events"][1]["data"]["confidence"] == 0.85
        assert result["events"][2]["data"]["recoverable"] is True

    def test_migration_does_not_modify_original(self):
        """Migration should not modify the original data."""
        original = {"lctl": "3.0", "chain": {"id": "test"}, "events": []}
        original_copy = json.loads(json.dumps(original))

        SchemaMigrator.migrate(original)

        assert original == original_copy
