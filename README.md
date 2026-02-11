# MediaWiki Graph Analysis Platform

A toolkit for analyzing any MediaWiki-based wiki (Fandom/Wikia, Wikipedia, self-hosted) using semantic embeddings, knowledge graph analysis, and LLM-enhanced clustering.

## Features

- **Hybrid embeddings** combining text (Qwen3-Embedding, GPU) and graph structure (node2vec, CPU)
- **HDBSCAN clustering** with automatic cluster discovery -- `eom` for broad clusters, `leaf` for granular topic separation
- **LLM cluster naming** via Ollama (qwen3:8b) with duplicate prevention and non-English rejection
- **6 visualization types**: network graph, t-SNE/UMAP scatter, similarity heatmap, dendrogram, cluster analysis
- **SQLite embedding cache** with WAL mode -- re-runs skip already-computed embeddings
- **CPU-safe** for large datasets (59K+ articles) with thread limiting and core pinning

## Quick Start

### 1. Configure Your Wiki

Edit `setup.py` to configure your wiki:

```python
WIKI_NAME = "Memory Alpha"  # Your wiki's display name
XML_URL = "https://s3.amazonaws.com/wikia_xml_dumps/e/en/enmemoryalpha_pages_current.xml.7z"
XML_FILENAME = "enmemoryalpha_pages_current.xml"
```

### 2. Install Dependencies

```bash
pip install sentence-transformers matplotlib scikit-learn networkx umap-learn mwparserfromhell node2vec gensim adjustText seaborn hdbscan
```

For LLM cluster naming (optional):
```bash
pip install ollama
ollama pull qwen3:8b
```

System dependencies:
```bash
sudo apt install p7zip-full wget
```

### 3. Download and Extract Wiki Dump

```bash
python setup.py
```

### 4. Run Analysis

```bash
# Quick test (100 articles)
python generate_graphs.py --test --wiki-name "Memory Alpha"

# Full analysis with HDBSCAN (default, fewer large clusters)
python generate_graphs.py --wiki-name "Memory Alpha"

# Granular clusters with leaf selection (recommended for large wikis)
python generate_graphs.py --wiki-name "Memory Alpha" --cluster-selection leaf --min-cluster-size 100 --min-samples 5

# With LLM cluster naming
python generate_graphs.py --wiki-name "Memory Alpha" --name-clusters --cluster-selection leaf

# K-Means with fixed cluster count
python generate_graphs.py --wiki-name "Memory Alpha" --clustering kmeans --clusters 30
```

### 5. CPU-Safe Full Run (10K+ Articles)

For large datasets, always use the run script to prevent machine freeze:

```bash
./run_star_trek.sh                  # Full run with LLM cluster naming
./run_star_trek.sh --test           # Quick test (100 articles)
./run_star_trek.sh --limit 1000     # Limit to 1000 articles
./run_star_trek.sh --skip-setup     # Skip download/extraction step
```

The script wraps `generate_graphs.py` with `taskset -c 0-5`, `nice -n 19`, and `OMP_NUM_THREADS=2` to limit CPU usage. Full 59K articles takes ~90 min.

## Architecture

```
MediaWiki XML dump --> setup.py --> Markdown files (data/pages/)
                                       |
                              generate_graphs.py
                                       |
              +------------------------+------------------------+
              |                                                 |
    Text embeddings (Qwen3, GPU)                    Graph embeddings (node2vec, CPU)
              +------------------------+------------------------+
                                       |
                    Hybrid embeddings (weighted combination)
                                       |
                         UMAP reduction (10D for clustering)
                                       |
                    HDBSCAN (default) or K-Means clustering
                                       |
                    Cluster quality metrics (silhouette, CH, DB)
                                       |
                    LLM cluster naming (Ollama qwen3:8b, optional)
                                       |
                              Visualizations (outputs/)
```

## Clustering Modes

### HDBSCAN (default)

HDBSCAN auto-discovers cluster count from data density. Two cluster selection methods:

| Method | Flag | Behavior | 59K Articles Result |
|--------|------|----------|---------------------|
| **eom** (default) | `--cluster-selection eom` | Fewer, larger clusters. Merges sub-clusters via "excess of mass" | ~15 clusters, largest 70% |
| **leaf** | `--cluster-selection leaf` | Granular clusters at condensed tree leaves. Breaks up mega-clusters | ~108 clusters, largest 3.4% |

Key parameters:
- `--min-cluster-size N` -- smallest meaningful group (default: auto-scaled `max(5, n_articles // 200)`)
- `--min-samples N` -- lower = less noise points (default: `min(min_cluster_size, 10)`)
- `--umap-dim N` -- UMAP reduction dimensions before clustering (default: 10)

### K-Means (alternative)

```bash
python generate_graphs.py --clustering kmeans --clusters 30
```

Requires specifying cluster count. Use `--optimal-clusters` to search for best K.

## Output Files

Generated in `outputs/` with `{wiki_slug}_` prefix:
- `*_network_graph.png` - Semantic network with cluster-aware layout and wiki links
- `*_tsne_scatter.png` / `*_umap_scatter.png` - 2D projections with centroid + representative labels
- `*_similarity_heatmap.png` - Pairwise cosine similarities with cluster boundary lines
- `*_dendrogram.png` - Hierarchical clustering with colored branches and threshold line
- `*_cluster_analysis.png` - Cluster size distribution (bar chart + horizontal bar)
- `*_clusters.json` - Cluster report with keywords, metrics, LLM names, and clustering parameters

## Embedding Models

| Model | Name | Dimensions | Notes |
|-------|------|------------|-------|
| 0.6B (default) | `Qwen/Qwen3-Embedding-0.6B` | 1024 | Stable, fits on 24GB GPU |
| 4B | `Qwen/Qwen3-Embedding-4B` | 2560 | Better quality, may crash on smaller GPUs |

## Configuration

### Fandom Dump URL Pattern

For Fandom/Wikia wikis:
```
https://s3.amazonaws.com/wikia_xml_dumps/{first_letter}/{first_two_letters}/{wikiname}_pages_current.xml.7z
```

Example: `Memory Alpha` -> `enmemoryalpha` -> `e/en/enmemoryalpha_pages_current.xml.7z`

### GPU Configuration

- **GPU 0**: Reserved for VLM MCP tool (Qwen3-VL) -- do not use for embeddings
- **GPU 1**: Embedding model (Qwen3-Embedding-0.6B), 24GB VRAM (RTX 3090)
- Default `CUDA_VISIBLE_DEVICES=1` is set in the run script
- Use `--batch-size 4` for safe VRAM usage on smaller GPUs

## Directory Structure

```
wiki-graph/
├── setup.py              # Download and extract MediaWiki XML
├── generate_graphs.py    # Main analysis script
├── run_star_trek.sh      # CPU-safe wrapper script
├── embeddings.db         # SQLite embedding cache (auto-generated)
├── data/
│   ├── extracted/        # Raw XML dumps
│   └── pages/            # Extracted markdown articles
└── outputs/              # Generated visualizations and JSON
```

## Performance

- Default run with no `--limit` processes ALL articles (~59K for Memory Alpha)
- Embeddings cached in SQLite (WAL mode) -- re-runs only regenerate graph embeddings + visualizations
- Full 59K articles: ~90 min (CPU-safe mode), peak 11.6GB RAM, no swap
- Breakdown: node2vec ~35 min, spring_layout ~40 min, everything else ~15 min

## Troubleshooting

### Machine Freezes on Large Runs

Always use the run script for 10K+ articles:
```bash
./run_star_trek.sh
```
Direct `python generate_graphs.py` without CPU limits will saturate all cores.

### CUDA Out of Memory

```bash
python generate_graphs.py --batch-size 4 --model 0.6B
```

### Embedding Database Issues

```bash
# Force rebuild all embeddings
python generate_graphs.py --rebuild
```

## License & Attribution

This project uses:
- **Qwen3-Embedding:** Alibaba Cloud Qwen3 model
- **node2vec:** Arijit Sen's Python implementation
- **SentenceTransformers:** UKP Lab sentence-transformers library
