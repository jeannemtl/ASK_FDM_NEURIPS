#!/bin/bash
# Apply cleanup: notes, paper-2 archive, duplicates archive.
# All operations are git mv (no deletions, no content loss).

set -e
cd /workspace/paper_repo

# ─────────────────────────────────────────────
# 1. Move working notes to <section>/_notes/
# ─────────────────────────────────────────────
echo "=== 1. Moving working notes to <section>/_notes/ ==="
grep -oP '`\K[^`]+(?=` → `)' NOTES_PLAN.md > /tmp/notes_src.txt
grep -oP '→ `\K[^`]+' NOTES_PLAN.md > /tmp/notes_dst.txt

n_notes=0
paste /tmp/notes_src.txt /tmp/notes_dst.txt | while IFS=$'\t' read src dst; do
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        git mv "$src" "$dst"
        n_notes=$((n_notes + 1))
    fi
done
echo "  Moved working notes."

# ─────────────────────────────────────────────
# 2. Move paper-2 files to 02_corruption_asymmetry/_paper2_notes/
# ─────────────────────────────────────────────
echo
echo "=== 2. Moving paper-2 files to 02_corruption_asymmetry/_paper2_notes/ ==="
mkdir -p 02_corruption_asymmetry/_paper2_notes

grep -oP 'DELETE \`\K[^`]+' DELETIONS_PLAN.md | while read src; do
    if [ -f "$src" ]; then
        fname=$(basename "$src")
        dst="02_corruption_asymmetry/_paper2_notes/$fname"
        echo "  $src → $dst"
        git mv "$src" "$dst"
    fi
done

# ─────────────────────────────────────────────
# 3. Move duplicates to _duplicates/ with section-suffixed names
# ─────────────────────────────────────────────
echo
echo "=== 3. Moving duplicates to _duplicates/ ==="
mkdir -p _duplicates

# Parse DUPLICATES.md group-by-group.
# Format per group:
#   ## Group N: `filename` (...)
#   - [ ] `path1`
#   - [ ] `path2`
#   ...
#   <blank>
#
# We use awk to emit one line per file, prefixed with the group's filename:
#   <group_filename>\t<file_path>

awk '
    /^## Group / {
        # Extract filename from `name` in the heading
        match($0, /`[^`]+`/)
        fname = substr($0, RSTART+1, RLENGTH-2)
        next
    }
    /^- \[ \] `/ {
        # Extract path
        match($0, /`[^`]+`/)
        path = substr($0, RSTART+1, RLENGTH-2)
        print fname "\t" path
    }
' DUPLICATES.md > /tmp/dup_pairs.txt

# Process each duplicate group: first copy keeps original name,
# subsequent copies get __<section_number> suffix
declare -A seen_filenames

while IFS=$'\t' read -r group_fname path; do
    [ -z "$path" ] && continue
    [ ! -f "$path" ] && continue

    # Extract section number (e.g., "03_split_substrate" → "03")
    section=$(echo "$path" | cut -d'/' -f1)
    sec_num=$(echo "$section" | cut -d'_' -f1)

    # Split filename into base and ext
    base="${group_fname%.*}"
    ext="${group_fname##*.}"
    if [ "$base" = "$group_fname" ]; then
        # No extension
        ext=""
    fi

    # First time we see this group filename → no suffix
    # Subsequent times → __<sec_num> suffix
    if [ -z "${seen_filenames[$group_fname]:-}" ]; then
        if [ -n "$ext" ] && [ "$ext" != "$group_fname" ]; then
            dst_name="$base.$ext"
        else
            dst_name="$group_fname"
        fi
        seen_filenames[$group_fname]=1
    else
        if [ -n "$ext" ] && [ "$ext" != "$group_fname" ]; then
            dst_name="${base}__${sec_num}.${ext}"
        else
            dst_name="${group_fname}__${sec_num}"
        fi
    fi

    dst="_duplicates/$dst_name"
    echo "  $path → $dst"
    git mv "$path" "$dst"
done < /tmp/dup_pairs.txt

echo
echo "=============================================="
echo "Cleanup applied. Review with:"
echo "  git status"
echo "  ls _duplicates/ | head"
echo "  ls 02_corruption_asymmetry/_paper2_notes/"
echo "  ls 03_split_substrate/_notes/ 2>/dev/null"
echo "=============================================="
