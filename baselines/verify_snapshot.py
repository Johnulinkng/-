from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files_sha256"]
    failures: list[str] = []
    for rel, digest in sorted(expected.items()):
        path = root / Path(rel)
        if not path.is_file():
            failures.append(f"missing:{rel}")
        elif sha256(path) != digest:
            failures.append(f"hash:{rel}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
    }
    extra = sorted(actual - set(expected))
    missing_from_tree = sorted(set(expected) - actual)
    failures.extend(f"unlisted:{rel}" for rel in extra)
    failures.extend(f"not_in_tree:{rel}" for rel in missing_from_tree)
    result = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(root),
        "checked_files": len(expected),
        "manifest_sha256": sha256(manifest_path),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
