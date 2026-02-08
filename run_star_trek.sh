#!/bin/bash
# Memory Alpha (Star Trek) Wiki Graph Analysis
#
# Downloads the Memory Alpha XML dump, extracts articles to markdown,
# generates embeddings, clusters them, and creates visualizations.
#
# CPU Safety: Uses taskset + thread limits to prevent machine freeze.
# Full run (~59K articles) takes ~90 min, peak ~12GB RAM, no swap.
#
# Usage:
#   ./run_star_trek.sh                  # Full run with LLM cluster naming
#   ./run_star_trek.sh --test           # Quick test (100 articles)
#   ./run_star_trek.sh --limit 1000     # Limit to 1000 articles
#   ./run_star_trek.sh --skip-setup     # Skip download/extraction

set -e

cd "$(dirname "$0")"

SKIP_SETUP=false
EXTRA_ARGS=()

# Parse our flags vs pass-through args
for arg in "$@"; do
    if [ "$arg" = "--skip-setup" ]; then
        SKIP_SETUP=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

echo "========================================================================"
echo "Memory Alpha (Star Trek) Wiki Graph Analysis"
echo "========================================================================"

# Step 1: Download and extract wiki dump
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo "Step 1: Downloading and extracting wiki dump..."
    echo "------------------------------------------------------------------------"
    python setup.py
else
    echo ""
    echo "Step 1: Skipped (--skip-setup)"
fi

# Step 2: Generate embeddings, cluster, and visualize
# CPU safety limits prevent machine freeze on large datasets:
#   taskset -c 0-5     : Hard limit to 6 of 14 CPU cores
#   nice -n 19         : Lowest scheduling priority
#   OMP_NUM_THREADS=2  : Cap BLAS/numpy parallelism (also MKL, OpenBLAS, numexpr)
#   CUDA_VISIBLE_DEVICES=1 : Use GPU 1 only (GPU 0 reserved for VLM)
#   PYTHONUNBUFFERED=1 : Real-time output (no buffering)
echo ""
echo "Step 2: Generating embeddings, clustering, and visualizing..."
echo "------------------------------------------------------------------------"

taskset -c 0-5 nice -n 19 env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    CUDA_VISIBLE_DEVICES=1 \
    python -u generate_graphs.py \
        --wiki-name "Memory Alpha" \
        --name-clusters \
        --batch-size 4 \
        "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================================"
echo "Done! Output files are in outputs/"
echo "========================================================================"
