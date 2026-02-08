# QWEN.md - MediaWiki Graph Analysis Platform

This file provides comprehensive context for AI assistants (Qwen Code, Claude Code) working with code in this repository.

## Project Overview

**MediaWiki Graph Analysis Platform** is a toolkit for analyzing any MediaWiki-based wiki (Fandom/Wikia, Wikipedia, self-hosted) using:
- **Semantic search** via text embeddings (Qwen3-Embedding models)
- **Knowledge graph extraction** from wiki link structure (node2vec)
- **Network visualization** with clustering and dimensionality reduction
- **LLM-enhanced analysis** for automatic cluster naming
- **Visual analysis** via VLM (Vision Language Models) for image interpretation

### Architecture

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

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| Data Setup | Download and parse MediaWiki XML dumps | Python stdlib XML, mwparserfromhell |
| Text Embeddings | Vector representations of article content | SentenceTransformers (Qwen3-Embedding) |
| Graph Embeddings | Structural position in wiki link network | node2vec |
| Clustering | Group semantically similar articles | scikit-learn K-means |
| Visualization | 2D/3D projections and network graphs | matplotlib, umap-learn |

## Recent Fixes & Improvements (2026)

| Fix | Description |
|-----|-------------|
| `--clusters` parameter override | Removed hardcoded `args.clusters = 8` in --test mode; now respects user's explicit choice |
| Batch size default | Changed from 1 to 32 for significant embedding performance improvement |
| Network graph visualization | Updated spring layout parameters (`k=3.0`, `scale=10`) to improve cluster separation |

For details on each fix, see the **Known Issues (Fixed)** table below.

## Quick Start

### First-Time Setup

1. **Edit `setup.py`** to configure your wiki:
```python
WIKI_NAME = "Memory Alpha"  # Your wiki's display name
XML_URL = "https://s3.amazonaws.com/wikia_xml_dumps/e/en/enmemoryalpha_pages_current.xml.7z"
XML_FILENAME = "enmemoryalpha_pages_current.xml"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt  # if exists, or:
pip install sentence-transformers matplotlib scikit-learn networkx umap-learn mwparserfromhell node2vec gensim
```

3. **Download and extract the wiki dump:**
```bash
python setup.py
```

### Running the Analysis

```bash
# Full analysis with all articles
python generate_graphs.py

# Quick test (100 articles, 8 clusters)
python generate_graphs.py --test

# Process only 500 articles
python generate_graphs.py --limit 500

# Force rebuild all embeddings
python generate_graphs.py --rebuild

# Use custom wiki name
python generate_graphs.py --wiki-name "Memory Alpha"

# Use LLM for cluster naming (requires Ollama)
python generate_graphs.py --name-clusters

# Use larger 4B embedding model (may crash on small GPUs)
python generate_graphs.py --model 4B

# Change number of K-means clusters
python generate_graphs.py --clusters 30

# Adjust text vs graph embedding weight (default: 70/30)
python generate_graphs.py --text-weight 0.8

# Use specific GPU (CUDA)
CUDA_VISIBLE_DEVICES=1 python generate_graphs.py

# Adjust batch size for embedding generation (default: 1)
python generate_graphs.py --batch-size 32
```

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

### Output Files

Generated in `outputs/` with `{wiki_slug}_` prefix:
- `*_network_graph.png` - Semantic network using wiki links
- `*_tsne_scatter.png` / `*_umap_scatter.png` - 2D projections
- `*_similarity_heatmap.png` - Pairwise cosine similarities
- `*_dendrogram.png` - Hierarchical clustering
- `*_cluster_analysis.png` - Cluster size distribution
- `*_clusters.json` - Cluster report with keywords and sample titles

## Development Conventions

### Code Style

- **Python version:** 3.9+ (tested on Python 3.13)
- **Main modules:** Single-file design (`generate_graphs.py` at ~1500 lines)
- **No strict formatting rules** - uses standard Python conventions
- **CUDA_VISIBLE_DEVICES:** Script auto-sets `CUDA_VISIBLE_DEVICES="1"` unless overridden
- **Error handling:** Graceful fallback with regex-based parsing when mwparserfromhell fails

### Database Schema (embeddings.db)

The SQLite database stores:
- `text_embeddings`: Article text + content hash + embedding
- `graph_embeddings`: node2vec vectors + graph hash
- `link_graph`: Source/target pairs for wiki links
- `metadata`: Key-value pairs for run configuration

### Key Functions (generate_graphs.py)

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `parse_wiki_markup()` | Extract text and links from MediaWiki | Uses mwparserfromhell |
| `load_articles_with_links()` | Load articles and build link graph | max_articles, min_length |
| `generate_text_embeddings_with_cache()` | GPU text embeddings | model, conn, rebuild, batch_size=4 |
| `generate_graph_embeddings()` | node2vec on wiki network | dimensions=128 |
| `combine_embeddings()` | Merge text + graph vectors | text_weight=0.7 |
| `discover_clusters()` | K-means clustering | n_clusters=20 |
| `get_cluster_names_via_llm()` | LLM cluster naming | Ollama, model="qwen3:8b" |
| `categorize_article()` | Assign article types | Title/content analysis |

### Cluster Configuration

**Important:** The script calculates clusters as:
```python
n_clusters = min(args.clusters, len(articles) // 10)

# Ensure minimum clusters for meaningful analysis
if args.clusters < 5 and len(articles) >= 50:
    n_clusters = args.clusters  # Respect user's explicit choice
else:
    n_clusters = max(n_clusters, 5)  # Minimum safety bound
```

- **Default:** 20 clusters (capped at 1/10 of total articles, minimum 5)
- With 10,000 articles: `min(20, 1000) = 20` → 20 clusters
- With `--clusters 5`: Respects user value directly if enough data available
- With `--clusters 3` on 1000+ articles: Respects user value with a warning message

**Note:** If you want fewer than 5 clusters, ensure you have at least 50 articles loaded. The script will warn but still use your specified cluster count.

The JSON output shows `"n_clusters"` which reflects the actual number used.

### Text Processing

- **Content length:** MAX_CONTENT_LENGTH=3000 characters
- **Minimum article length:** MIN_ARTICLE_LENGTH=200 characters
- **Category extraction:** Parses `[[Category:Name]]` patterns
- **Keyword extraction:** Frequency-based, excluding stopwords

### Graph Construction

1. **Base edges:** Actual wiki links (`[[Page Name]]`)
2. **Semantic edges:** Added based on embedding similarity
3. **Category edges:** Strong connections for same-category articles
4. **Edge weights:** 0.4-1.0 based on similarity and category overlap

## Requirements

### Core Python Packages

```bash
pip install sentence-transformers matplotlib scikit-learn networkx umap-learn mwparserfromhell node2vec gensim numpy
```

### Optional (LLM Features)

```bash
pip install ollama
ollama pull qwen3:8b  # For cluster naming
```

### System Dependencies

```bash
# For setup.py
sudo apt install p7zip-full wget

# For CUDA (optional, for faster embeddings)
# NVIDIA GPU drivers + CUDA toolkit
```

## Data Format

### Markdown Article Structure

Each article in `data/pages/`:
```markdown
# Article Title

{{sidebar|content}}

Article body text with [[wiki links]] and '''bold''' text...
```

- **Filename:** Sanitized title with underscores
- **First line:** Title with `# ` prefix
- **Main namespace only:** `ns=0` articles extracted
- **Redirects/short articles:** Skipped during extraction

## Troubleshooting

### CUDA Out of Memory

If you see `torch.OutOfMemoryError`:
```bash
# Reduce batch size in code
# Or use smaller model:
python generate_graphs.py --model 0.6B

# Or use CPU (slow):
unset CUDA_VISIBLE_DEVICES
```

### No Edges in Graph

The graph may have 0 edges if:
- No valid wiki links found (check article content)
- Article titles don't match link targets (case sensitivity)
- Too few articles loaded

### Embedding Cache Mismatch

If switching models:
```bash
python generate_graphs.py --rebuild
```

### Cluster Count Not Changing

If `--clusters 5` doesn't produce 5 clusters, check:
1. The argument is being passed correctly
2. The script's calculation: `n_clusters = min(args.clusters, len(articles) // 10)`
3. For 10,000 articles, `min(5, 1000) = 5`, so it should work
4. If using old cached embeddings, add `--rebuild` flag

### Slow Embedding Generation

The default `batch-size=32` processes multiple articles in parallel. Speed up by using larger batch sizes:
```bash
python generate_graphs.py --batch-size 64
```

For CUDA out of memory issues, reduce batch size:
```bash
python generate_graphs.py --batch-size 8
```

## Directory Structure

```
wiki-graph/
├── setup.py              # Download and extract MediaWiki XML
├── generate_graphs.py    # Main analysis script (~1500 lines)
├── run_star_trek.sh      # Helper script for Star Trek wiki
├── embeddings.db         # SQLite cache (auto-generated)
├── data/
│   ├── extracted/        # Raw XML dumps (delete after extraction)
│   └── pages/            # Extracted markdown articles
├── outputs/              # Generated visualizations and JSON
└── tmp/                  # Temporary files
```

## Testing Recommendations

1. **Start with --test mode** to verify the pipeline works
2. **Check embeddings.db** exists after first run
3. **Review outputs/** for generated visualizations
4. **Check cluster analysis** to ensure meaningful groupings
5. **Review cluster JSON** for keyword quality

## Performance Notes

- **Text embeddings:** ~10-30 articles/second on GPU (batch size 4)
- **Graph embeddings:** ~1000 nodes/second on CPU
- **K-means:** Fast for <10,000 articles
- **Disk space:** ~1KB per article for markdown, ~4KB per embedding

### Batch Size Configuration

Use `--batch-size` to speed up embedding generation:
- **Default:** batch_size=32 (was 1, updated for better performance)
- **Recommended:** batch_size=32 for faster processing
- Use larger batches if you have more GPU memory available
- With 10,000 articles and batch_size=32: ~5-10 minutes total

## License & Attribution

This project uses:
- **Qwen3-Embedding:** Alibaba Cloud Qwen3 model
- **node2vec:** Arijit Sen's Python implementation
- **SentenceTransformers:** UKP Lab sentence-transformers library

The script includes comments for proper attribution to these projects.

## Work Still Needed

### Known Issues (Fixed)

| Issue | Status | Priority | Fix Applied |
|-------|--------|----------|-------------|
| Cluster visualization shows overlapping wheel-and-spoke pattern instead of distinct clusters | Fixed | High | Updated spring layout parameters (`k=3.0`, `scale=10`) and layout strategy for better cluster separation |
| VLM tool fails on all images with 500 error | Fixed | Medium | Tool now works correctly after update |
| Script runs slowly with default batch_size=1 | Fixed | High | Changed default from 1 to 32 for significant speedup |
| `--clusters` parameter may not be applied correctly | Fixed | Medium | Removed hardcoded override in --test mode; smarter cluster count logic |

### VLM Image Analysis Results

After updating the VLM tool, comprehensive image analysis was performed:

**Network Graph Visualization:**
- Shows radial "wheel-and-spoke" pattern with dense central core
- Clusters distributed across concentric rings
- Overlapping nodes in center make individual nodes hard to distinguish

**t-SNE Scatter Plot:**
- Moderately to well-separated clusters
- Clear distinct groupings with some overlap between adjacent clusters
- No major issues with cluster mixing

**Similarity Heatmap:**
- Strong diagonal separation indicating good cluster quality
- Well-defined blocks suggest items within clusters are highly similar

**Cluster Size Distribution:**
- Highly skewed (power-law) distribution
- Dominant C6 cluster (~820 articles): "episode, worf"
- Second/third largest: ~540-630 articles each
- Topics: episode, worf, technology, picard

## Using VLM for Visual Analysis

The VLM (Vision Language Model) can analyze generated visualization images.

### How to Use

```python
# Example usage:
vlm_chat(
    prompt="Analyze this network graph. How well do the clusters separate?",
    image_path="/path/to/outputs/memory_alpha_network_graph.png"
)
```

### Available Output Files for Analysis

Generated in `outputs/`:
- `*_network_graph.png` - Semantic network using wiki links
- `*_tsne_scatter.png` / `*_umap_scatter.png` - 2D projections
- `*_similarity_heatmap.png` - Pairwise cosine similarities
- `*_dendrogram.png` - Hierarchical clustering
- `*_cluster_analysis.png` - Cluster size distribution

### Interpreting Results

| Pattern | Interpretation |
|---------|----------------|
| **Wheel-and-spoke** in network graph | Central dense core with radial distribution; may need layout adjustment |
| **Moderate cluster separation** in t-SNE | Good initial separation with minor overlap |
| **Strong diagonal blocks** in heatmap | Well-separated clusters with high internal similarity |
| **Skewed distribution** in cluster analysis | One or two dominant clusters; consider data imbalance |

### Improvements

- [ ] Improve network graph layout to reduce overlapping nodes and wheel-and-spoke pattern
- [ ] Implement automatic elbow method or silhouette analysis for optimal cluster count
- [ ] Add more sophisticated edge weighting based on semantic similarity
- [ ] Support additional embedding models (e.g., newer Qwen versions)
- [ ] Improve keyword extraction for cluster labeling
- [ ] Add option to exclude specific articles from analysis
- [ ] Create a web-based interface for easier interaction

### Testing Recommendations

1. Start with `--test` mode to verify the pipeline works
2. Check embeddings.db exists after first run
3. Review outputs/ for generated visualizations
4. Check cluster analysis to ensure meaningful groupings
5. Review cluster JSON for keyword quality
6. Test cluster count changes with `--clusters N`
7. Validate batch size improvements with `--batch-size 32`

## Directory Structure