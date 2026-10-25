"""Read-only SHA-256 duplication report for VisualQC datasets."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

ROOTS = (Path("data"), Path("data_cropped"))
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def audit(root):
    if not root.exists():
        print(f"{root}: not present")
        return

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    hashes = defaultdict(list)
    for path in files:
        digest = sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(path)

    classes = Counter(p.parent.name for p in files)
    print(f"\n{root}: {len(files)} files; {len(hashes)} unique contents")
    print("Classes:", dict(classes))
    print("Exact duplicate copies:", len(files) - len(hashes))
    for paths in hashes.values():
        if len(paths) > 1:
            print("  Duplicate:", " | ".join(map(str, paths)))


if __name__ == "__main__":
    for directory in ROOTS:
        audit(directory)
