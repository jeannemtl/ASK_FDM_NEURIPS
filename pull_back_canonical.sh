#!/bin/bash
# Pull canonical files back out of _duplicates/ into their proper sections.
# Each pair: <duplicate-archive-name> <canonical-section/filename>
# The "redundant" copy with __NN suffix stays in _duplicates/.

set -e
cd /workspace/paper_repo

# Helper: move from _duplicates/<name> to <target-path>
restore() {
    local src="_duplicates/$1"
    local dst="$2"
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        echo "  $src → $dst"
        git mv "$src" "$dst"
    else
        echo "  ⚠️  not found: $src"
    fi
}

echo "=== Pulling canonical files back ==="
echo

echo "── §3.1 corruption (active script)"
restore "corruption_v2_all_models_table8.py"     "02_corruption_asymmetry/corruption_v2_all_models_table8.py"
# corruption_v2_all_models_table8__02.py stays in _duplicates/

echo
echo "── §3 single-block results (Tables 1, 2)"
restore "lfm2_perchannel_eval.txt"                "01_single_block_validation/lfm2_perchannel_eval.txt"
restore "qwen3_perchannel_eval.txt"               "01_single_block_validation/qwen3_perchannel_eval.txt"
restore "lfm2_fdm_eval_results.txt"               "01_single_block_validation/lfm2_fdm_eval_results.txt"
restore "qwen3_fdm_eval_results.txt"              "01_single_block_validation/qwen3_fdm_eval_results.txt"
restore "nhop_lfm2_results.txt"                   "01_single_block_validation/nhop_lfm2_results.txt"
restore "nhop_qwen3_results.txt"                  "01_single_block_validation/nhop_qwen3_results.txt"
restore "plaintext_40ch_results.json"             "01_single_block_validation/plaintext_40ch_results.json"
# The __02 versions stay in _duplicates/

echo
echo "=== Done. Review with: git status | head -30 ==="
echo
echo "Files restored to canonical locations; redundant __<N> copies remain in _duplicates/"
