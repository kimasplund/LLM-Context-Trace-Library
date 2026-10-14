"""Verify a built LCTL wheel carries the files and metadata a git install relies on.

Usage: python scripts/check_wheel.py dist/lctl-<version>-py3-none-any.whl

Exits 0 when every check passes; otherwise prints one ``FAIL: <reason>`` line
per failed check and exits 1. Used by the CI ``package`` job.
"""

import re
import sys
import zipfile
from pathlib import Path

REQUIRED_FILES = [
    "lctl/py.typed",
    "lctl/dashboard/templates/index.html",
    "lctl/dashboard/static/app.js",
    "lctl/dashboard/static/styles.css",
]
REQUIRED_CLASSIFIERS = [
    "Programming Language :: Python :: 3.15",
    "Typing :: Typed",
]
LICENSE_EXPRESSION = "AGPL-3.0-only"


def expected_version() -> str:
    init = Path(__file__).resolve().parent.parent / "lctl" / "__init__.py"
    match = re.search(r'^__version__ = "(.+)"', init.read_text(), re.MULTILINE)
    if match is None:
        raise SystemExit(f"FAIL: no __version__ found in {init}")
    return match.group(1)


def check(wheel_path: Path) -> list[str]:
    failures = []
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        failures += [f"wheel is missing {name}" for name in REQUIRED_FILES if name not in names]

        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            return [*failures, f"expected one METADATA file, found {len(metadata_names)}"]
        lines = wheel.read(metadata_names[0]).decode().splitlines()

    version = expected_version()
    if f"Version: {version}" not in lines:
        failures.append(f"METADATA lacks 'Version: {version}'")
    if f"License-Expression: {LICENSE_EXPRESSION}" not in lines:
        failures.append(f"METADATA lacks 'License-Expression: {LICENSE_EXPRESSION}'")
    failures += [
        f"METADATA has a License classifier (PEP 639): {line}"
        for line in lines
        if line.startswith("Classifier: License ::")
    ]
    failures += [
        f"METADATA lacks classifier '{c}'"
        for c in REQUIRED_CLASSIFIERS
        if f"Classifier: {c}" not in lines
    ]
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    failures = check(Path(sys.argv[1]))
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print(f"OK: {sys.argv[1]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
