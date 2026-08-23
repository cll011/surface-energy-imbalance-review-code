"""Create file-level SHA-256 manifests for the review archive."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "MANIFEST.csv"
CHECKSUMS = PROJECT_ROOT / "SHA256SUMS.txt"
EXCLUDED_NAMES = {
    "MANIFEST.csv",
    "SHA256SUMS.txt",
    "anonymous_peer_review_code.zip",
    "paths.local.json",
}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files():
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        yield path, relative


def main() -> None:
    rows = []
    for path, relative in included_files():
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["relative_path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    CHECKSUMS.write_text(
        "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Manifested {len(rows)} files")


if __name__ == "__main__":
    main()
