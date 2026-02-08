# MediaWiki Graph Analysis Platform

A toolkit for analyzing any MediaWiki-based wiki (Fandom/Wikia, Wikipedia, self-hosted) using semantic embeddings and knowledge graph analysis.

## Features

- **Semantic search** via text embeddings (Qwen3-Embedding models)
- **Knowledge graph extraction** from wiki link structure (node2vec)
- **Network visualization** with clustering and dimensionality reduction
- **LLM-enhanced analysis** for automatic cluster naming
- **Visual analysis** via VLM (Vision Language Models) for image interpretation

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
pip install sentence-transformers matplotlib scikit-learn networkx umap-learn mwparserfromhell node2vec gensim numpy
```

For LLM cluster naming (optional):
```bash
pip install ollama
ollama pull qwen3:8b
```

### 3. Download and Extract Wiki Dump

```bash
python setup.py
```

### 4. Run Analysis

```bash
# Full analysis with all articles
python generate_graphs.py

# Quick test (100 articles, 8 clusters)
python generate_graphs.py --test

# Process only 500 articles
python generate_graphs.py --limit 500

# Use custom wiki name
python generate_graphs.py --wiki-name "Memory Alpha"

# Use LLM for cluster naming (requires Ollama)
python generate_graphs.py --name-clusters

# Use larger 4B embedding model (may crash on small GPUs)
python generate_graphs.py --model 4B

# Change number of K-means clusters
python generate_graphs.py --clusters 30

# Adjust batch size for faster embedding generation
python generate_graphs.py --batch-size 32

# Use specific GPU (CUDA)
CUDA_VISIBLE_DEVICES=1 python generate_graphs.py
```

## Architecture

```
MediaWiki XML dump → setup.py → Markdown files (data/pages/)
                                                ↓
                                      generate_graphs.py
                                                ↓
                ┌─────────────────────────────┴─────────────────────────────┐
                ↓                                                         ↓
      Text embeddings (Qwen3, GPU)                         Graph embeddings (node2vec, CPU)
                └─────────────────────────────┬─────────────────────────────┘
                                              ↓
                              Hybrid embeddings (weighted concatenation)
                                              ↓
                                   K-means clustering
                                              ↓
                               Visualizations (outputs/)
```

## Output Files

Generated in `outputs/` with `{wiki_slug}_` prefix:
- `*_network_graph.png` - Semantic network using wiki links
- `*_tsne_scatter.png` / `*_umap_scatter.png` - 2D projections
- `*_similarity_heatmap.png` - Pairwise cosine similarities
- `*_dendrogram.png` - Hierarchical clustering
- `*_cluster_analysis.png` - Cluster size distribution
- `*_clusters.json` - Cluster report with keywords and sample titles

## Configuration

### Embedding Models

| Model | Name | Dimensions | Notes |
|-------|------|------------|-------|
| 0.6B (default) | `Qwen/Qwen3-Embedding-0.6B` | 1024 | Stable, good for most GPUs |
| 4B | `Qwen/Qwen3-Embedding-4B` | 2560 | Better quality, may crash on smaller GPUs |

### Fandom Dump URL Pattern

For Fandom/Wikia wikis:
```
https://s3.amazonaws.com/wikia_xml_dumps/{first_letter}/{first_two_letters}/{wikiname}_pages_current.xml.7z
```

Example: `Memory Alpha` → `enmemoryalpha` → `e/en/enmemoryalpha_pages_current.xml.7z`

## Directory Structure

```
wiki-graph/
├── setup.py              # Download and extract MediaWiki XML
├── generate_graphs.py    # Main analysis script
├── run_star_trek.sh      # Helper script for Star Trek wiki
├── embeddings.db         # SQLite cache (auto-generated)
├── data/
│   ├── extracted/        # Raw XML dumps (delete after extraction)
│   └── pages/            # Extracted markdown articles
├── outputs/              # Generated visualizations and JSON
└── tmp/                  # Temporary files
```

## Requirements

### Core Python Packages

```bash
pip install sentence-transformers matplotlib scikit-learn networkx umap-learn mwparserfromhell node2vec gensim numpy
```

### System Dependencies

```bash
# For setup.py
sudo apt install p7zip-full wget

# For CUDA (optional, for faster embeddings)
# NVIDIA GPU drivers + CUDA toolkit
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python generate_graphs.py --batch-size 8

# Or use smaller model
python generate_graphs.py --model 0.6B

# Or use CPU (slow)
unset CUDA_VISIBLE_DEVICES
```

### Slow Embedding Generation

```bash
python generate_graphs.py --batch-size 64
```

### Cluster Count Not Changing

```bash
python generate_graphs.py --rebuild
```

## License & Attribution

This project uses:
- **Qwen3-Embedding:** Alibaba Cloud Qwen3 model
- **node2vec:** Arijit Sen's Python implementation
- **SentenceTransformers:** UKP Lab sentence-transformers library
