#!/usr/bin/env python3
"""
Reorganize paper_repo:
  <section>/<file>  →  <section>/<model>/<results-or-script>/<file>

Rules:
- Files matching a model keyword move into <section>/<model>/...
- Files with no model keyword stay at <section>/ (model-agnostic shared scripts)
- Results files (.json, .log, .txt, .csv) go to <model>/results/
- Script files (.py, .md) go to <model>/ directly
- Only creates model folders that will have at least one file
- Uses `git mv` to preserve history
"""

import argparse
import re
import subprocess
from pathlib import Path
from collections import defaultdict

REPO = Path("/workspace/paper_repo")

# Model classification rules — first match wins, order matters.
# Patterns are case-insensitive and match against the filename only.
MODEL_PATTERNS = [
    ("hermes3", re.compile(r"hermes3?|nous", re.I)),
    ("qwen3",   re.compile(r"qwen3?", re.I)),
    ("lfm2",    re.compile(r"lfm2|liquid", re.I)),
    ("gpt2",    re.compile(r"gpt2|gpt_?2", re.I)),
]

# File-type classification
RESULT_EXTS = {".json", ".log", ".txt", ".csv"}
SCRIPT_EXTS = {".py", ".md"}

# Files that should never be moved (stay at repo root or section root)
KEEP_AT_ROOT = {"README.md", ".gitignore"}
KEEP_AT_SECTION_ROOT = {"README.md"}

# MANIFEST_*.md stays at repo root
def is_manifest(name: str) -> bool:
    return name.startswith("MANIFEST_") and name.endswith(".md")


def classify_model(filename: str) -> str | None:
    """Return model name if filename matches one, else None (= shared)."""
    for model, pattern in MODEL_PATTERNS:
        if pattern.search(filename):
            return model
    return None


def classify_kind(filename: str) -> str:
    """Return 'results' or 'script'."""
    suffix = Path(filename).suffix.lower()
    if suffix in RESULT_EXTS:
        return "results"
    if suffix in SCRIPT_EXTS:
        return "script"
    # Default: treat unknowns as scripts (won't go to results/)
    return "script"


def plan_moves(repo: Path) -> list[tuple[Path, Path]]:
    """Return list of (src, dst) move tuples."""
    moves = []
    sections = sorted(d for d in repo.iterdir()
                      if d.is_dir() and re.match(r"\d{2}_", d.name))

    for section in sections:
        for f in sorted(section.iterdir()):
            if not f.is_file():
                continue
            if f.name in KEEP_AT_SECTION_ROOT:
                continue

            model = classify_model(f.name)
            if model is None:
                # Shared / model-agnostic — leave in place
                continue

            kind = classify_kind(f.name)
            if kind == "results":
                dst = section / model / "results" / f.name
            else:
                dst = section / model / f.name

            moves.append((f, dst))

    return moves


def run_git(args, dry: bool):
    cmd = ["git", "-C", str(REPO)] + args
    if dry:
        print(f"  [dry] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the moves (default: dry run)")
    args = parser.parse_args()

    dry = not args.apply
    moves = plan_moves(REPO)

    if not moves:
        print("Nothing to move.")
        return

    # Group by section + model for readable preview
    by_target = defaultdict(list)
    for src, dst in moves:
        target_dir = dst.parent
        by_target[target_dir].append((src.name, dst.name))

    print(f"=== {'DRY RUN' if dry else 'APPLYING'}: {len(moves)} files to move ===\n")

    for target_dir in sorted(by_target.keys()):
        rel = target_dir.relative_to(REPO)
        print(f"→ {rel}/")
        for src_name, dst_name in by_target[target_dir]:
            print(f"    {src_name}")
        print()

    # Files staying at section level (shared scripts)
    print("=== Staying at section level (model-agnostic) ===\n")
    for section in sorted(d for d in REPO.iterdir()
                          if d.is_dir() and re.match(r"\d{2}_", d.name)):
        stayers = sorted(f.name for f in section.iterdir()
                         if f.is_file()
                         and classify_model(f.name) is None
                         and f.name not in KEEP_AT_SECTION_ROOT)
        if stayers:
            print(f"→ {section.name}/")
            for name in stayers:
                print(f"    {name}")
            print()

    if dry:
        print("=" * 50)
        print("This was a dry run. To apply, re-run with --apply")
        print("=" * 50)
        return

    # Apply: create target dirs and git mv
    print("Creating directories and moving files...\n")
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Ensure parent dirs are tracked even if empty (shouldn't be, but safe)
        run_git(["mv", str(src.relative_to(REPO)), str(dst.relative_to(REPO))],
                dry=False)

    print("\nDone. Review with: git status")
    print("Then commit:")
    print('  git commit -m "Reorganize: split each section by model with results/ subfolders"')
    print("  git push")


if __name__ == "__main__":
    main()
