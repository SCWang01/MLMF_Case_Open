"""Validate the files listed in the release SHA-256 manifest."""
from hashlib import sha256
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    errors = []
    for line in manifest:
        expected, name = line.split("  ", 1)
        path = root / name
        if not path.is_file():
            errors.append(f"Missing: {name}")
        elif sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Checksum differs: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
