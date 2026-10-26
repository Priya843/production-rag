"""Download the sample corpus: the Ultralytics YOLO docs (Markdown, AGPL-3.0) from GitHub.

The commit is pinned so the questions in eval/golden.jsonl keep matching the text.
Any folder of .md/.txt/.pdf files works too: put your own files in data/raw/ instead.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from rag.config import settings

REPO = "https://github.com/ultralytics/ultralytics.git"
COMMIT = "739d9c72b74b498fa674a477dc6a8190e2373350"  # main on 2026-10-26
FOLDERS = ["docs/en/modes", "docs/en/tasks", "docs/en/guides", "docs/en/integrations"]


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> None:
    dest = settings.data_dir / "ultralytics"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        repo = Path(tmp)
        git("init", "-q", cwd=repo)
        git("remote", "add", "origin", REPO, cwd=repo)
        git("sparse-checkout", "set", *FOLDERS, cwd=repo)
        git("fetch", "-q", "--depth=1", "--filter=blob:none", "origin", COMMIT, cwd=repo)
        git("checkout", "-q", "FETCH_HEAD", cwd=repo)

        if dest.exists():
            shutil.rmtree(dest)
        for folder in FOLDERS:
            shutil.copytree(
                repo / folder,
                dest / Path(folder).name,
                ignore=lambda d, names: [n for n in names if not (n.endswith(".md") or (Path(d) / n).is_dir())],
            )
    count = sum(1 for _ in dest.rglob("*.md"))
    print(f"Saved {count} Markdown files to {dest}")


if __name__ == "__main__":
    main()
