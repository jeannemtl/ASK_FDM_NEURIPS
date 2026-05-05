#!/usr/bin/env python3
"""
Audit paper_repo for duplicates and working-notes cleanup.

Phase 1 (this script): identify and report.
Phase 2 (later): apply manual decisions per file.

Outputs:
  - DUPLICATES.md: groups of identical files across sections
  - NOTES_PLAN.md: files to move to <section>/_notes/
  - DELETIONS_PLAN.md: files to delete (paper 2 contamination)
"""

import re
import hashlib
from pathlib import Path
from collections import defaultdict

REPO = Path("/workspace/paper_repo")

# Patterns that identify working notes (move to <section>/_notes/)
NOTES_PATTERNS = [
    re.compile(r"^ANALYSIS_AND_FIX.*\.md$"),
    re.compile(r"^check_diversity\.md$"),
    re.compile(r"^RECOVERY\.md$"),
    re.compile(r"^diagonal_.*\.txt$"),
    re.compile(r"^recompute_dirs\.log$"),
    re.compile(r"^fdm_curriculum_s\d.*\.log$"),
    re.compile(r"^fdm_curriculum_4view\.log$"),
    re.compile(r"^fdm_open_nlp\.log$"),
    re.compile(r"^multiscreen_fdm\.log$"),
    re.compile(r"^product_v\d.*\.log$"),
    re.compile(r"^benchmark_summary\.json$"),
    re.compile(r"^mqar_results\.json$"),
    re.compile(r"^prontoqa_results\.json$"),
    re.compile(r"^vocab_map\.json$"),
]

# Patterns for paper-2 contamination (delete)
DELETE_PATTERNS = [
    re.compile(r"^paper2_.*\.log$"),
]


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def matches_any(name: str, patterns: list) -> bool:
    return any(p.match(name) for p in patterns)


def all_section_files(repo: Path):
    for section in sorted(repo.iterdir()):
        if not section.is_dir() or not re.match(r"\d{2}_", section.name):
            continue
        for f in section.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                yield section.name, f


def main():
    print("Scanning repository...\n")

    by_hash = defaultdict(list)
    notes_files = []
    delete_files = []

    for section, fpath in all_section_files(REPO):
        name = fpath.name
        if matches_any(name, DELETE_PATTERNS):
            delete_files.append((section, fpath))
            continue
        if matches_any(name, NOTES_PATTERNS):
            notes_files.append((section, fpath))
            continue

        h = file_hash(fpath)
        by_hash[h].append((section, fpath))

    duplicates = {h: entries for h, entries in by_hash.items() if len(entries) > 1}

    # ---- DUPLICATES.md ----
    dup_md = REPO / "DUPLICATES.md"
    with dup_md.open("w") as f:
        f.write("# Duplicate Files Across Sections\n\n")
        f.write(f"Found **{len(duplicates)} duplicate groups** "
                f"covering **{sum(len(e) for e in duplicates.values())} files**.\n\n")
        f.write("For each group below, decide which copy to KEEP and "
                "delete the others. Mark your decisions inline with [KEEP] / [DEL].\n\n")
        f.write("---\n\n")

        sorted_groups = sorted(
            duplicates.items(),
            key=lambda kv: kv[1][0][1].name
        )

        for i, (h, entries) in enumerate(sorted_groups, 1):
            fname = entries[0][1].name
            size_kb = entries[0][1].stat().st_size / 1024
            f.write(f"## Group {i}: `{fname}` ({size_kb:.1f} KB, {len(entries)} copies)\n\n")
            for section, path in sorted(entries, key=lambda x: x[0]):
                rel = path.relative_to(REPO)
                f.write(f"- [ ] `{rel}`\n")
            f.write("\n")

    print(f"✓ Wrote {dup_md} ({len(duplicates)} duplicate groups)")

    # ---- NOTES_PLAN.md (UPDATED: notes go inside the section) ----
    notes_md = REPO / "NOTES_PLAN.md"
    with notes_md.open("w") as f:
        f.write("# Files to move to `<section>/_notes/`\n\n")
        f.write(f"**{len(notes_files)} files** matching working-notes patterns.\n\n")
        f.write("Patterns matched:\n")
        for p in NOTES_PATTERNS:
            f.write(f"- `{p.pattern}`\n")
        f.write("\n---\n\n")
        for section, fpath in sorted(notes_files):
            rel = fpath.relative_to(REPO)
            target = f"{section}/_notes/{fpath.name}"
            f.write(f"- `{rel}` → `{target}`\n")
    print(f"✓ Wrote {notes_md} ({len(notes_files)} files)")

    # ---- DELETIONS_PLAN.md ----
    del_md = REPO / "DELETIONS_PLAN.md"
    with del_md.open("w") as f:
        f.write("# Files to delete (paper 2 contamination)\n\n")
        f.write(f"**{len(delete_files)} files** matching deletion patterns.\n\n")
        f.write("Patterns matched:\n")
        for p in DELETE_PATTERNS:
            f.write(f"- `{p.pattern}`\n")
        f.write("\n---\n\n")
        for section, fpath in sorted(delete_files):
            rel = fpath.relative_to(REPO)
            f.write(f"- [ ] DELETE `{rel}`\n")
    print(f"✓ Wrote {del_md} ({len(delete_files)} files)")

    print(f"\nSummary:")
    print(f"  {len(duplicates)} duplicate groups → DUPLICATES.md")
    print(f"  {len(notes_files)} working-notes files → NOTES_PLAN.md")
    print(f"  {len(delete_files)} paper-2 files → DELETIONS_PLAN.md")
    print(f"\nNext: open the three .md files and review.")


if __name__ == "__main__":
    main()
