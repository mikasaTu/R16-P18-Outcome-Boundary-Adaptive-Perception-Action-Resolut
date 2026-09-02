#!/usr/bin/env python3
"""Generate or verify a fail-on-overwrite SHA256 manifest for S1 files."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path


MANIFEST_NAME = "SHA256SUMS"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, manifest: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest or path.name == MANIFEST_NAME:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.name.startswith(".") and ".tmp-" in path.name:
            continue
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in SHA256 manifest: {path}")
        result.append(path)
    return result


def _manifest_rows(root: Path, manifest: Path) -> list[tuple[str, str]]:
    rows = []
    seen: set[str] = set()
    for path in manifest.read_text(encoding="utf-8").splitlines():
        line = path.strip()
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed SHA256SUMS line: {line!r}") from exc
        if not HASH_RE.fullmatch(digest):
            raise ValueError(f"invalid digest in SHA256SUMS: {digest!r}")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"manifest path escapes root: {relative!r}")
        rel_text = rel.as_posix()
        if rel_text in seen or rel_text == MANIFEST_NAME:
            raise ValueError(f"duplicate/self manifest entry: {rel_text!r}")
        seen.add(rel_text)
        rows.append((digest, rel_text))
    return rows


def verify(root: Path, manifest: Path) -> dict[str, int | str]:
    if not manifest.is_file():
        raise FileNotFoundError(f"SHA256 manifest not found: {manifest}")
    rows = _manifest_rows(root, manifest)
    expected = {relative: digest for digest, relative in rows}
    actual_files = {path.relative_to(root).as_posix(): path for path in _files(root, manifest)}
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    mismatched = sorted(
        relative
        for relative, digest in expected.items()
        if relative in actual_files and sha256_file(actual_files[relative]) != digest
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            f"SHA256 verification failed: missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    return {"status": "PASS", "entries": len(rows), "missing": 0, "extra": 0, "mismatched": 0}


def generate(root: Path, manifest: Path) -> dict[str, int | str]:
    if manifest.exists():
        raise FileExistsError(f"fail-on-overwrite: {manifest}")
    files = _files(root, manifest)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temporary, manifest)
    # The post-write verification is part of generation, but the manifest is
    # not included in its own input set.
    return verify(root, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = (args.output or (root / MANIFEST_NAME)).resolve()
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise ValueError("SHA256 manifest must be inside --root") from exc
    result = verify(root, manifest) if args.verify else generate(root, manifest)
    print(f"SHA256S_{result['status']} entries={result['entries']}")


if __name__ == "__main__":
    main()
