"""LCTL Schema versioning and migration support."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from packaging import version

# Current schema version
CURRENT_VERSION = "4.2"

# Minimum supported version for loading
MIN_SUPPORTED_VERSION = "3.0"

# Version history
VERSION_HISTORY = {
    "3.0": "Initial public release",
    "4.0": "Added confidence tracking, fact modification",
    "4.1": "Added LRU cache, schema versioning, redaction support",
    "4.2": "Span IDs (event id, span_id, parent_span_id) and chain trace_id",
}


class SchemaVersionError(Exception):
    """Raised when schema version is incompatible."""

    def __init__(self, found_version: str, required_version: str, message: str = ""):
        self.found_version = found_version
        self.required_version = required_version
        super().__init__(
            message
            or f"Schema version {found_version} is incompatible. Required: >={required_version}"
        )


class SchemaMigrator:
    """Handles migration between schema versions."""

    _migrations: ClassVar[dict[str, Callable[[dict], dict]]] = {}

    @classmethod
    def register(cls, from_version: str, to_version: str):
        """Decorator to register a migration function."""

        def decorator(func: Callable[[dict], dict]):
            cls._migrations[f"{from_version}->{to_version}"] = func
            return func

        return decorator

    @classmethod
    def migrate(cls, data: dict[str, Any], target_version: str = CURRENT_VERSION) -> dict[str, Any]:
        """Migrate data from its current version to target version."""
        current = data.get("lctl", "3.0")

        if version.parse(current) >= version.parse(target_version):
            return data  # Already at or above target

        # Apply migrations in sequence
        result = data.copy()
        migration_path = cls._get_migration_path(current, target_version)

        for from_v, to_v in migration_path:
            key = f"{from_v}->{to_v}"
            if key in cls._migrations:
                result = cls._migrations[key](result)
                result["lctl"] = to_v

        return result

    @classmethod
    def _get_migration_path(cls, from_version: str, to_version: str) -> list[tuple]:
        """Get ordered list of migrations needed."""
        versions = sorted(VERSION_HISTORY.keys(), key=lambda v: version.parse(v))
        path = []

        in_range = False
        for i, v in enumerate(versions):
            if v == from_version:
                in_range = True
            if in_range and i + 1 < len(versions):
                next_v = versions[i + 1]
                if version.parse(next_v) <= version.parse(to_version):
                    path.append((v, next_v))
            if v == to_version:
                break

        return path

    @classmethod
    def get_registered_migrations(cls) -> list[str]:
        """Get list of registered migration keys."""
        return list(cls._migrations.keys())


def validate_version(data: dict[str, Any]) -> None:
    """Validate that schema version is supported.

    Args:
        data: Dictionary containing chain data with 'lctl' version key.

    Raises:
        SchemaVersionError: If the schema version is below MIN_SUPPORTED_VERSION.
    """
    found = data.get("lctl", "3.0")

    if version.parse(found) < version.parse(MIN_SUPPORTED_VERSION):
        raise SchemaVersionError(
            found,
            MIN_SUPPORTED_VERSION,
            f"Schema version {found} is too old. Minimum supported: {MIN_SUPPORTED_VERSION}",
        )


def get_version_info() -> dict[str, Any]:
    """Get information about schema versions."""
    return {
        "current": CURRENT_VERSION,
        "minimum_supported": MIN_SUPPORTED_VERSION,
        "history": VERSION_HISTORY,
    }


# Register migrations
@SchemaMigrator.register("3.0", "4.0")
def _migrate_3_to_4(data: dict) -> dict:
    """Migrate from 3.0 to 4.0 - add confidence tracking."""
    result = data.copy()
    # 3.0 -> 4.0: No structural changes needed, just version bump
    # Facts already supported confidence, this formalizes it
    return result


@SchemaMigrator.register("4.0", "4.1")
def _migrate_4_to_4_1(data: dict) -> dict:
    """Migrate from 4.0 to 4.1 - schema versioning support."""
    result = data.copy()
    # 4.0 -> 4.1: No structural changes, just version bump
    # Adds LRU cache and redaction support (runtime features)
    return result


@SchemaMigrator.register("4.1", "4.2")
def _migrate_4_1_to_4_2(data: dict) -> dict:
    """Migrate from 4.1 to 4.2 - optional span IDs and chain trace_id."""
    result = data.copy()
    # 4.1 -> 4.2: No structural changes; the new fields are optional and
    # older traces get their structure inferred at replay time.
    return result
