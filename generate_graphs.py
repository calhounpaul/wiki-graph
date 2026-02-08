#!/usr/bin/env python3
"""
Generate embedding visualizations from MediaWiki articles using hybrid embeddings.

Pipeline: embed → UMAP reduce → cluster (HDBSCAN/K-Means) → evaluate → visualize

This script combines:
1. Text embeddings (Qwen3-Embedding-0.6B) from article content
2. Graph embeddings (node2vec) from wiki link structure
3. UMAP dimensionality reduction (10D) before clustering
4. HDBSCAN (default) or K-Means clustering
5. Cluster quality metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)

Usage:
    python generate_graphs.py                    # Use ALL pages (HDBSCAN default)
    python generate_graphs.py --limit 500        # Use first 500 articles
    python generate_graphs.py --test             # Quick test with 100 articles
    python generate_graphs.py --clustering kmeans --clusters 30  # K-Means mode
    python generate_graphs.py --min-cluster-size 50  # HDBSCAN with larger clusters
    python generate_graphs.py --wiki-name "My Wiki"  # Custom wiki name
    CUDA_VISIBLE_DEVICES=1 python generate_graphs.py  # Use GPU 1 instead

Embeddings are cached in a SQLite database for fast reuse.
"""

# MUST set CUDA device BEFORE any torch imports (including from dependencies)
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import re
import sys
import sqlite3
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
import colorsys

# Paths - relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(BASE_DIR, "data", "pages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EMBEDDINGS_DB = os.path.join(BASE_DIR, "embeddings.db")

# Model config - can be overridden by --model flag
MODELS = {
    "4B": {"name": "Qwen/Qwen3-Embedding-4B", "dim": 2560},
    "0.6B": {"name": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024},
}
DEFAULT_MODEL = "0.6B"  # Use smaller model by default (more stable)
MODEL_NAME = MODELS[DEFAULT_MODEL]["name"]
EMBEDDING_DIM = MODELS[DEFAULT_MODEL]["dim"]
GRAPH_EMBEDDING_DIM = 128  # node2vec embedding dimension
COMBINED_DIM = EMBEDDING_DIM + GRAPH_EMBEDDING_DIM

# Text processing config
MAX_CONTENT_LENGTH = 3000  # Use more content for better embeddings
MIN_ARTICLE_LENGTH = 200  # Lowered for more coverage

# VRAM limit for GPU 1 (in GB) - prevent exceeding this during embedding generation
GPU_VRAM_LIMIT_GB = 18  # Second GPU has 18GB VRAM


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB for CUDA device 0 (GPU 1 physically)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 * 1024)  # Convert to MB
    except ImportError:
        pass
    return 0


def check_vram_safety(batch_size, article_count):
    """
    Check if batch size is safe for VRAM limits.
    For Qwen3-Embedding-0.6B with 1024D embeddings:
    - ~500MB per 100 articles with batch_size=8
    - Scales roughly linearly with batch size and model dimensions
    
    Returns True if safe, False if should reduce batch size.
    """
    # Estimate VRAM usage: base usage + batch_size * articles * dim * scaling_factor
    # With 18GB limit, we should stay well below that during generation
    max_safe_articles_per_batch = int((GPU_VRAM_LIMIT_GB * 800) / (EMBEDDING_DIM / 1024))
    return article_count <= max(max_safe_articles_per_batch, 50)


def generate_distinct_colors(n):
    """Generate maximally distinct colors for n clusters using HLS color space."""
    import matplotlib.pyplot as plt
    if n <= 20:
        return [plt.cm.tab20(i / 20) for i in range(n)]
    # Combine tab20 + tab20b for up to 40 distinct colors
    colors = []
    for i in range(min(n, 20)):
        colors.append(plt.cm.tab20(i / 20))
    for i in range(20, n):
        # Fill remaining with HLS-spaced colors that avoid existing hues
        hue = ((i - 20) * 0.618033988749895) % 1.0  # Golden ratio spacing
        lightness = 0.45 + (i % 3) * 0.15
        saturation = 0.65 + (i % 2) * 0.2
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b, 1.0))
    return colors


def init_database(db_path: str):
    """Initialize SQLite database for embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Text embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_embeddings (
            filename TEXT PRIMARY KEY,
            title TEXT,
            content_hash TEXT,
            embedding BLOB,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Graph embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_embeddings (
            filename TEXT PRIMARY KEY,
            embedding BLOB,
            graph_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Link graph table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS link_graph (
            source TEXT,
            target TEXT,
            PRIMARY KEY (source, target)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    conn.commit()
    return conn


def get_content_hash(content: str) -> str:
    """Generate hash of content for change detection."""
    return hashlib.md5(content.encode()).hexdigest()


def parse_wiki_markup(text: str) -> Tuple[str, List[str]]:
    """
    Parse MediaWiki markup to extract clean text and wiki links.

    Returns:
        Tuple of (clean_text, list_of_links)
    """
    try:
        import mwparserfromhell
        wikicode = mwparserfromhell.parse(text)

        # Extract all wiki links
        links = []
        for link in wikicode.filter_wikilinks():
            target = str(link.title).strip()
            # Skip category, file, and other special links
            if ':' not in target or target.startswith('Document:'):
                # Normalize link target
                target = target.replace('_', ' ').strip()
                if target and len(target) > 1:
                    links.append(target)

        # Get plain text (strip all markup)
        clean_text = wikicode.strip_code()

        # Additional cleanup
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text, links

    except Exception as e:
        # Fallback to regex-based parsing
        return parse_wiki_markup_regex(text)


def parse_wiki_markup_regex(text: str) -> Tuple[str, List[str]]:
    """Fallback regex-based parser for wiki markup."""
    # Extract wiki links first
    link_pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    links = []
    for match in re.finditer(link_pattern, text):
        target = match.group(1).strip()
        if ':' not in target or target.startswith('Document:'):
            target = target.replace('_', ' ').strip()
            if target and len(target) > 1:
                links.append(target)

    # Clean text
    clean = text
    clean = re.sub(r'\{\{[^}]*\}\}', '', clean)  # Remove templates
    clean = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', clean)  # Wiki links to text
    clean = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', clean)  # External links
    clean = re.sub(r'\[https?://[^\]]+\]', '', clean)  # Bare URLs
    clean = re.sub(r'<ref[^>]*>.*?</ref>', '', clean, flags=re.DOTALL)  # References
    clean = re.sub(r'<ref[^/]*/>', '', clean)  # Self-closing refs
    clean = re.sub(r'<[^>]+>', '', clean)  # HTML tags
    clean = re.sub(r'\[\[Category:[^\]]+\]\]', '', clean)  # Categories
    clean = re.sub(r"'''?", '', clean)  # Bold/italic
    clean = re.sub(r'==+[^=]+=+', ' ', clean)  # Section headers
    clean = re.sub(r'\s+', ' ', clean).strip()

    return clean, links


def extract_categories(content: str) -> List[str]:
    """Extract category information from article content."""
    categories = re.findall(r'\[\[Category:(.*?)\]\]', content)
    return [c.strip() for c in categories if c.strip()]


def extract_keywords_from_content(text: str, min_word_length: int = 4, top_k: int = 10) -> List[str]:
    """Extract significant keywords from text content."""
    import re
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Common stopwords
    stopwords = {
        'the', 'a', 'an', 'of', 'in', 'to', 'and', 'for', 'on', 'at', 'by', 'with',
        'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'new', 'one', 'two', 'first', 'second'
    }

    word_counts = Counter(w for w in words if len(w) >= min_word_length and w not in stopwords)
    return [word for word, count in word_counts.most_common(top_k)]


def load_articles_with_links(pages_dir: str, max_articles: int = None, min_length: int = MIN_ARTICLE_LENGTH,
                              show_progress: bool = True) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    """
    Load MediaWiki articles and extract link graph.

    When max_articles is specified, randomly samples from all valid articles
    to get a representative subset (not just first N alphabetically).

    Returns:
        Tuple of (articles_list, link_graph_dict)
    """
    import random
    random.seed(42)  # Reproducible sampling

    articles = []
    link_graph = defaultdict(set)  # source -> set of targets
    pages_path = Path(pages_dir)

    if not pages_path.exists():
        print(f"Error: Pages directory not found: {pages_dir}")
        return articles, dict(link_graph)

    files = list(pages_path.glob('*.md'))
    total_files = len(files)
    print(f"Found {total_files} markdown files")

    if max_articles and max_articles < total_files:
        print(f"Randomly sampling {max_articles} articles for representative subset")
        files = random.sample(files, max_articles * 2)  # Oversample to account for filtering

    # First pass: collect all valid article titles
    print("  Pass 1: Collecting article titles...")
    valid_titles = set()
    title_to_filename = {}

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            title = first_line.lstrip('#').strip()
            if title:
                valid_titles.add(title)
                title_to_filename[title] = filepath.name
        except:
            pass

    print(f"  Found {len(valid_titles)} valid article titles")

    # Second pass: load articles and extract links
    print("  Pass 2: Loading articles and extracting links...")
    skipped_short = 0
    skipped_error = 0

    for i, filepath in enumerate(files):
        if max_articles and len(articles) >= max_articles:
            break

        if show_progress and i % 1000 == 0:
            print(f"    Processing: {i}/{total_files} files, {len(articles)} articles loaded...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n', 2)
            title = lines[0].lstrip('#').strip() if lines else filepath.stem
            body = '\n'.join(lines[2:]) if len(lines) > 2 else ''

            # Parse with mwparserfromhell
            cleaned, links = parse_wiki_markup(body)

            # Extract categories
            categories = extract_categories(content)

            if len(cleaned) >= min_length:
                # Filter links to only valid articles
                valid_links = [l for l in links if l in valid_titles and l != title]

                # Add to link graph
                for link_target in valid_links:
                    link_graph[title].add(link_target)

                # Extract keywords from content
                keywords = extract_keywords_from_content(cleaned, top_k=8)

                articles.append({
                    'title': title,
                    'content': cleaned[:MAX_CONTENT_LENGTH],
                    'summary': cleaned[:MAX_CONTENT_LENGTH],
                    'filename': filepath.name,
                    'content_hash': get_content_hash(cleaned[:MAX_CONTENT_LENGTH]),
                    'out_links': valid_links,
                    'categories': categories,
                    'keywords': keywords
                })
            else:
                skipped_short += 1
        except Exception as e:
            skipped_error += 1
            if skipped_error <= 5:
                print(f"    Warning: Error loading {filepath.name}: {e}")

    print(f"Loaded {len(articles)} articles (skipped {skipped_short} short, {skipped_error} errors)")

    # Count total links
    total_links = sum(len(targets) for targets in link_graph.values())
    print(f"Extracted {total_links} wiki links between articles")

    return articles, dict(link_graph)


def build_networkx_graph(articles: List[Dict], link_graph: Dict[str, Set[str]]):
    """Build a NetworkX graph from the link structure."""
    import networkx as nx

    G = nx.DiGraph()

    # Add all articles as nodes
    title_to_idx = {a['title']: i for i, a in enumerate(articles)}
    for i, article in enumerate(articles):
        G.add_node(i, title=article['title'], categories=article.get('categories', []))

    # Add edges from wiki links (real relationships)
    edge_count = 0
    for source_title, targets in link_graph.items():
        if source_title in title_to_idx:
            source_idx = title_to_idx[source_title]
            for target_title in targets:
                if target_title in title_to_idx:
                    target_idx = title_to_idx[target_title]
                    G.add_edge(source_idx, target_idx, weight=1.0, edge_type='link')
                    edge_count += 1

    print(f"  Built graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G, title_to_idx


def categorize_article(title: str, content: str) -> List[str]:
    """Categorize an article based on its title and content."""
    categories = []
    title_lower = title.lower()
    content_lower = content[:1000].lower() if content else ""

    # Series/Show categories
    if any(x in title_lower for x in ['star trek', 'series', 'season', 'episode', 'tv series', 'tv show']):
        categories.append('series')
    if any(x in title_lower for x in ['star trek: ', 'star trek ', 'voyager', 'enterprise', 'deep space nine', 'discovery', 'lower deck']):
        categories.append('series')

    # Character categories
    if any(x in title_lower for x in [' character', 'captain', 'commander', 'lieutenant', 'ensign', 'doctor', 'officer', 'officer', 'sisko', 'janeway', 'picard', 'kirk', 'spock', 'data', 'troi', 'geordi', 'worf', 'seven of nine', 'chakotay', ' paris', 'kim', 'b\'elanna', ' Kes', 'ro laren', 'quark', 'garak', 'odo', 'martok', 'dax', 'wrex']):
        categories.append('character')

    # Planet/Location categories
    if any(x in title_lower for x in [' planet', 'station', 'space station', 'starbase', 'moon', 'world', 'system', 'quadrant', 'sector', 'nebula', 'wormhole', 'star', 'sun', 'black hole']):
        categories.append('location')

    # Ship/Starship categories
    if any(x in title_lower for x in [' starship', ' USS ', ' vessel', 'craft', 'ship', 'enterprise', 'voyager', 'defiant', 'relativity', 'sao pa', 'titan', 'enterprise', 'defiant', 'cerritos', 'valkyrie']):
        categories.append('ship')

    # Episode categories
    if any(x in title_lower for x in ['episode', 'season', 'part ', 'part i', 'part ii', 'part iii']):
        categories.append('episode')

    # Technology/Species categories
    if any(x in title_lower for x in [' technology', 'species', 'technology ', 'device', 'weapon', 'phaser', 'torpedo', 'warp', 'transporter', 'phaser', 'borg', 'klingon', 'vulcan', 'romulan', 'ferengi', 'cardassian', 'federation', 'dominion', 'jem hadar', 'maquis', 'kazon', 'talaxian', 'ocampa', 'krenim', 'hrogen', 'malon', 'species', 'borg queen', 'founder', 'changeling']):
        categories.append('technology')

    # Episode categories
    if any(x in title_lower for x in ['episode']):
        categories.append('episode')

    # Generic categories
    if any(x in title_lower for x in ['category', 'categories', 'list', 'index', 'outline', 'portal', 'portal', 'glossary', 'index', 'timeline', 'chronology']):
        categories.append('meta')

    return list(set(categories))


def add_semantic_edges(G, articles: List[Dict], embeddings: np.ndarray,
                       title_to_idx: Dict[str, int],
                       similarity_threshold: float = 0.4,
                       max_edges_per_node: int = 20):
    """Add semantic similarity edges to improve clustering.

    Strategy:
    1. First, add edges for articles with the same category (type)
    2. Then add edges based on embedding similarity
    3. Connect related topics within Star Trek universe
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"  Adding semantic similarity edges...")

    n = len(articles)
    article_embeddings = np.array([embeddings[i] for i in range(n)])

    # Pre-compute categories for all articles
    categories_cache = {}
    for i, article in enumerate(articles):
        categories_cache[i] = categorize_article(article['title'], article['content'])

    edges_added = 0

    # Phase 1: Add category-based edges (same article type)
    # Group articles by category to avoid O(n²) all-pairs comparison
    print("  Phase 1: Adding category-based edges...")
    cat_to_articles = defaultdict(list)
    for i, cats in categories_cache.items():
        for cat in cats:
            cat_to_articles[cat].append(i)

    for cat, article_ids in cat_to_articles.items():
        # Skip very large categories — too many edges, and they're not very informative
        if len(article_ids) > 500:
            continue
        for a_idx in range(len(article_ids)):
            for b_idx in range(a_idx + 1, len(article_ids)):
                i, j = article_ids[a_idx], article_ids[b_idx]
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=0.8, edge_type='category')
                    edges_added += 1

    print(f"  Added {edges_added} category-based edges")

    # Phase 2: Add embedding-based edges for similar content
    print("  Phase 2: Adding embedding-similarity edges...")

    chunk_size = 200
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        sim_matrix = cosine_similarity(article_embeddings[i:end_i], article_embeddings)

        for local_i, idx in enumerate(range(i, end_i)):
            similarities = sim_matrix[local_i].copy()
            similarities[idx] = -1  # Exclude self

            # Get top similar articles (excluding those we already connected)
            similar_indices = np.argsort(similarities)[::-1][:max_edges_per_node]

            for sim_idx in similar_indices:
                if similarities[sim_idx] > similarity_threshold and sim_idx != idx:
                    if not G.has_edge(idx, sim_idx):
                        # Check if they share any categories
                        common_cats = set(categories_cache.get(idx, [])) & set(categories_cache.get(sim_idx, []))
                        if common_cats or 'character' in categories_cache.get(idx, []) or 'character' in categories_cache.get(sim_idx, []):
                            # Same or similar category - add edge
                            weight = 0.6 + (similarities[sim_idx] * 0.4)  # Weight by similarity
                            G.add_edge(idx, sim_idx, weight=weight, edge_type='semantic')
                            edges_added += 1

    print(f"  Added {edges_added} total semantic edges")
    return G


def generate_graph_embeddings(G, dimensions: int = GRAPH_EMBEDDING_DIM) -> np.ndarray:
    """Generate node embeddings using node2vec (CPU, parallelized)."""
    from node2vec import Node2Vec

    n_nodes = G.number_of_nodes()
    print(f"Generating graph embeddings for {n_nodes} nodes (CPU, 1 worker)...")

    if G.number_of_edges() == 0:
        print("  Warning: No edges in graph, returning zero embeddings")
        return np.zeros((n_nodes, dimensions), dtype=np.float32)

    G_undirected = G.to_undirected()

    print("  Generating random walks...")
    node2vec = Node2Vec(
        G_undirected,
        dimensions=dimensions,
        walk_length=20,
        num_walks=5,
        workers=1,
        p=1.0,
        q=1.0,
        quiet=False
    )

    print("  Training Word2Vec model...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4, workers=2)

    embeddings = np.zeros((n_nodes, dimensions), dtype=np.float32)
    for node_id in range(n_nodes):
        if str(node_id) in model.wv:
            embeddings[node_id] = model.wv[str(node_id)]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    print(f"  Graph embeddings shape: {embeddings.shape}")
    return embeddings


def load_cached_text_embedding(conn: sqlite3.Connection, filename: str, content_hash: str,
                                expected_dim: int = 1024) -> Optional[np.ndarray]:
    """Load text embedding from cache, checking dimension compatibility."""
    cursor = conn.cursor()
    cursor.execute(
        'SELECT embedding FROM text_embeddings WHERE filename = ? AND content_hash = ?',
        (filename, content_hash)
    )
    row = cursor.fetchone()
    if row:
        emb = np.frombuffer(row[0], dtype=np.float32)
        # Skip if dimension doesn't match (different model was used)
        if len(emb) != expected_dim:
            return None
        return emb
    return None


def save_text_embedding(conn: sqlite3.Connection, filename: str, title: str,
                        content_hash: str, embedding: np.ndarray, summary: str):
    """Save text embedding to database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO text_embeddings (filename, title, content_hash, embedding, summary)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, title, content_hash, embedding.astype(np.float32).tobytes(), summary[:500]))
    conn.commit()


def generate_text_embeddings_with_cache(articles: List[Dict], model, conn: sqlite3.Connection,
                                         batch_size: int = 4, rebuild: bool = False,
                                         embedding_dim: int = 1024) -> np.ndarray:
    """Generate text embeddings with database caching and error recovery."""
    import torch
    import time

    n_articles = len(articles)
    embeddings = np.zeros((n_articles, embedding_dim), dtype=np.float32)

    to_generate = []
    to_generate_indices = []
    cached_count = 0

    print(f"Checking text embedding cache (expecting {embedding_dim}D)...")
    for i, article in enumerate(articles):
        if not rebuild:
            cached = load_cached_text_embedding(conn, article['filename'], article['content_hash'],
                                                expected_dim=embedding_dim)
            if cached is not None:
                embeddings[i] = cached
                cached_count += 1
                continue
        to_generate.append(article)
        to_generate_indices.append(i)

    print(f"  Found {cached_count} cached text embeddings")
    print(f"  Need to generate {len(to_generate)} new text embeddings")

    if to_generate:
        texts = [a['summary'] for a in to_generate]
        current_batch_size = batch_size
        print(f"Generating text embeddings in batches of {current_batch_size}...")

        batch_start = 0
        while batch_start < len(texts):
            batch_end = min(batch_start + current_batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_articles = to_generate[batch_start:batch_end]
            batch_indices = to_generate_indices[batch_start:batch_end]

            try:
                batch_embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

                for idx, article, emb in zip(batch_indices, batch_articles, batch_embeddings):
                    embeddings[idx] = emb
                    save_text_embedding(conn, article['filename'], article['title'],
                                       article['content_hash'], emb, article['summary'])

                if (batch_start // current_batch_size) % 50 == 0:
                    print(f"  Generated {batch_end}/{len(texts)} text embeddings...")

                batch_start = batch_end  # Move to next batch

            except Exception as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "gpu" in error_str or "device" in error_str:
                    print(f"\n  CUDA error at batch {batch_start}: {e}")
                    print(f"  GPU has crashed. Cannot recover without restart.")
                    print(f"  Progress saved: {cached_count + batch_start} embeddings cached.")
                    print(f"  Restart script to continue from cache.")
                    sys.exit(1)
                else:
                    raise

        print(f"  Saved {len(to_generate)} new text embeddings to database")

    return embeddings


def combine_embeddings(text_embeddings: np.ndarray, graph_embeddings: np.ndarray,
                       text_weight: float = 0.7) -> np.ndarray:
    """
    Combine text and graph embeddings using weighted concatenation.

    Args:
        text_embeddings: (N, text_dim) array
        graph_embeddings: (N, graph_dim) array
        text_weight: Weight for text embeddings (graph gets 1-text_weight)
    """
    # Replace NaN/Inf with zeros
    text_embeddings = np.nan_to_num(text_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    graph_embeddings = np.nan_to_num(graph_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize both
    text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_norms[text_norms == 0] = 1  # Prevent division by zero
    text_norm = text_embeddings / text_norms

    graph_norms = np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
    graph_norms[graph_norms == 0] = 1
    graph_norm = graph_embeddings / graph_norms

    # Weight and concatenate
    combined = np.concatenate([
        text_norm * text_weight,
        graph_norm * (1 - text_weight)
    ], axis=1)

    # Re-normalize combined
    combined_norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined_norms[combined_norms == 0] = 1
    combined = combined / combined_norms

    # Final NaN check
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  NaN check: {np.isnan(combined).sum()} NaN values")

    return combined


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 10,
                      n_neighbors: int = 30, metric: str = 'cosine',
                      random_state: int = 42) -> np.ndarray:
    """Reduce embedding dimensions using UMAP before clustering.

    Research shows clustering in 5-10D UMAP space significantly outperforms
    clustering on raw high-dimensional embeddings. Use min_dist=0.0 for
    tighter clusters (better for downstream clustering algorithms).
    """
    from umap import UMAP

    n = len(embeddings)
    actual_neighbors = min(n_neighbors, n - 1)

    print(f"Reducing dimensions: {embeddings.shape[1]}D -> {n_components}D via UMAP...")
    print(f"  n_neighbors={actual_neighbors}, metric={metric}")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=actual_neighbors,
        min_dist=0.0,
        metric=metric,
        random_state=random_state,
        n_jobs=2
    )
    reduced = reducer.fit_transform(embeddings)

    print(f"  Reduced shape: {reduced.shape}")
    return reduced


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 25,
                    min_samples: int = None) -> Tuple[np.ndarray, Dict, int]:
    """Cluster using HDBSCAN with automatic cluster count discovery.

    HDBSCAN discovers the number of clusters from data density, handles
    variable-density clusters, and labels ambiguous points as noise (-1).
    Noise points are reassigned to the nearest cluster centroid.

    Args:
        embeddings: UMAP-reduced embeddings (N, reduced_dim)
        min_cluster_size: Smallest meaningful group size (auto-scaled if None)
        min_samples: Controls conservativeness; lower = less noise. Defaults to
                     min(min_cluster_size, 10).
    """
    import hdbscan

    if min_samples is None:
        min_samples = min(min_cluster_size, 10)

    print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=2
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    print(f"  Found {n_clusters_found} clusters, {noise_count} noise points ({100*noise_count/len(labels):.1f}%)")

    # Reassign noise points to nearest cluster centroid
    if noise_count > 0:
        noise_mask = labels == -1
        cluster_ids = sorted(set(labels) - {-1})
        centroids = np.array([embeddings[labels == cid].mean(axis=0) for cid in cluster_ids])

        from sklearn.metrics import pairwise_distances_argmin_min
        nearest, _ = pairwise_distances_argmin_min(embeddings[noise_mask], centroids)
        labels[noise_mask] = np.array(cluster_ids)[nearest]
        print(f"  Reassigned {noise_count} noise points to nearest clusters")

    # Relabel to contiguous 0-indexed
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    n_clusters = len(unique_labels)

    cluster_info = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_info[i] = {
            'size': int(np.sum(mask)),
            'center': embeddings[mask].mean(axis=0)
        }

    return labels, cluster_info, n_clusters


def evaluate_clusters(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate cluster quality with silhouette, Calinski-Harabasz, and Davies-Bouldin."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    n_clusters = len(set(labels))
    if n_clusters < 2:
        print("  Cannot evaluate: need at least 2 clusters")
        return {}

    metrics = {}

    try:
        metrics['silhouette'] = float(silhouette_score(embeddings, labels))
    except Exception:
        metrics['silhouette'] = float('nan')

    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, labels))
    except Exception:
        metrics['calinski_harabasz'] = float('nan')

    try:
        metrics['davies_bouldin'] = float(davies_bouldin_score(embeddings, labels))
    except Exception:
        metrics['davies_bouldin'] = float('nan')

    print(f"\n  Cluster Quality Metrics:")
    print(f"    Silhouette Score:      {metrics['silhouette']:.4f}  (range -1 to 1, >0.5 = good)")
    print(f"    Calinski-Harabasz:     {metrics['calinski_harabasz']:.1f}  (higher = better)")
    print(f"    Davies-Bouldin:        {metrics['davies_bouldin']:.4f}  (lower = better, 0 = ideal)")

    return metrics


def compute_silhouette_scores(embeddings: np.ndarray, max_clusters: int = 30) -> Dict[int, float]:
    """Compute silhouette scores for different cluster counts to find optimal K.

    Args:
        embeddings: (N, dim) array of embeddings
        max_clusters: Maximum number of clusters to try

    Returns:
        Dictionary mapping n_clusters to silhouette score
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("Computing silhouette scores for optimal cluster count...")

    scores = {}
    min_samples = len(embeddings)
    min_clusters = max(5, min(3, max_clusters))

    for n_clusters in range(min_clusters, min(max_clusters + 1, min_samples // 2)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(embeddings)

        # Skip if any cluster has only 1 member
        unique_labels = set(labels)
        if len(unique_labels) < n_clusters:
            continue

        try:
            score = silhouette_score(embeddings, labels)
            scores[n_clusters] = score
            print(f"  {n_clusters} clusters: silhouette = {score:.4f}")
        except ValueError:
            continue

    return scores


def compute_elbow_scores(embeddings: np.ndarray, max_clusters: int = 30) -> Dict[int, float]:
    """Compute inertia (WCSS) for different cluster counts (elbow method).

    Args:
        embeddings: (N, dim) array of embeddings
        max_clusters: Maximum number of clusters to try

    Returns:
        Dictionary mapping n_clusters to inertia value
    """
    from sklearn.cluster import KMeans

    print("Computing elbow method scores...")

    inertias = {}
    min_clusters = max(5, min(3, max_clusters))

    for n_clusters in range(min_clusters, min(max_clusters + 1, len(embeddings) // 2)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(embeddings)
        inertias[n_clusters] = kmeans.inertia_
        print(f"  {n_clusters} clusters: inertia = {inertias[n_clusters]:.2f}")

    return inertias


def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 30,
                         method: str = 'silhouette') -> int:
    """Find optimal number of clusters using silhouette analysis or elbow method.

    Args:
        embeddings: (N, dim) array of embeddings
        max_clusters: Maximum number of clusters to try
        method: 'silhouette' or 'elbow'

    Returns:
        Optimal number of clusters
    """
    if method == 'silhouette':
        scores = compute_silhouette_scores(embeddings, max_clusters)
        if not scores:
            return max(5, min(20, len(embeddings) // 10))
        best_k = max(scores.keys(), key=lambda k: scores[k])
        print(f"\nOptimal clusters by silhouette: {best_k} (score: {scores[best_k]:.4f})")
        return best_k
    else:
        inertias = compute_elbow_scores(embeddings, max_clusters)
        # Find elbow point using second derivative
        if len(inertias) < 3:
            return max(5, min(20, len(embeddings) // 10))

        # Calculate curvature
        clusters = sorted(inertias.keys())
        values = [inertias[c] for c in clusters]

        # Normalize
        normalized = (np.array(values) - min(values)) / (max(values) - min(values) + 1e-10)

        # Find maximum curvature
        curvature = np.abs(np.diff(normalized, 2))
        elbow_idx = np.argmax(curvature) + 1

        best_k = clusters[elbow_idx]
        print(f"\nOptimal clusters by elbow method: {best_k}")
        return best_k


def discover_clusters(embeddings: np.ndarray, n_clusters: int = 20,
                     find_optimal: bool = False) -> Tuple[np.ndarray, Dict]:
    """Discover natural clusters using K-means.

    Args:
        embeddings: (N, dim) array of embeddings
        n_clusters: Target number of clusters
        find_optimal: If True, automatically find optimal cluster count

    Returns:
        Tuple of (labels, cluster_info)
    """
    from sklearn.cluster import KMeans

    actual_n_clusters = n_clusters

    if find_optimal:
        actual_n_clusters = find_optimal_clusters(embeddings, max_clusters=min(n_clusters, 30),
                                                   method='silhouette')

    print(f"Discovering {actual_n_clusters} natural clusters...")
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)

    cluster_info = {}
    for i in range(actual_n_clusters):
        mask = labels == i
        cluster_info[i] = {
            'size': int(np.sum(mask)),
            'center': kmeans.cluster_centers_[i]
        }

    return labels, cluster_info


def get_cluster_keywords(articles: List[Dict], labels: np.ndarray, n_clusters: int,
                         top_words: int = 5) -> Dict[int, List[str]]:
    """Extract discriminating keywords for each cluster using TF-IDF weighting.

    Words that are frequent in a cluster but rare across other clusters get the
    highest scores, producing labels that actually distinguish clusters.
    """
    from collections import Counter
    import math

    # Stopwords: very common English words + wiki-wide noise terms
    stopwords = {
        'the', 'a', 'an', 'of', 'in', 'to', 'and', 'for', 'on', 'at', 'by', 'with',
        'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'can', 'need', 'used', 'this', 'that', 'these',
        'those', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'than', 'too', 'very', 'just', 'also', 'now', 'new', 'one', 'two',
        'first', 'second', 'third', 'its', 'his', 'her', 'their', 'our', 'your',
        'not', 'but', 'only', 'own', 'same', 'so', 'into', 'over', 'after',
        'before', 'between', 'under', 'during', 'while', 'about', 'then',
        'star', 'trek', 'episode', 'podcast', 'category', 'thumb',
    }

    # Phase 1: Collect word counts per cluster and globally
    cluster_word_counts = {}  # cluster_id -> Counter
    global_doc_freq = Counter()  # word -> number of clusters it appears in

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]

        words = []
        for idx in indices:
            # Title words (weighted 2x since titles are more descriptive)
            title_words = [w.lower().strip('(),"\'.:;!?') for w in articles[idx]['title'].split()]
            words.extend(title_words)
            words.extend(title_words)  # double-count titles

            # Content keywords
            words.extend(articles[idx].get('keywords', []))

        # Filter and count
        filtered = [w for w in words if len(w) > 2 and w not in stopwords
                    and not w.isdigit() and w.isalpha()]
        cluster_word_counts[cluster_id] = Counter(filtered)

        # Track which clusters each word appears in (for IDF)
        for word in set(filtered):
            global_doc_freq[word] += 1

    # Phase 2: Compute TF-IDF scores per cluster
    cluster_keywords = {}
    for cluster_id in range(n_clusters):
        counts = cluster_word_counts[cluster_id]
        if not counts:
            cluster_keywords[cluster_id] = [f'cluster {cluster_id}']
            continue

        total_words = sum(counts.values())
        scored = {}
        for word, count in counts.items():
            tf = count / total_words
            # IDF: words in fewer clusters get higher scores
            idf = math.log(n_clusters / (1 + global_doc_freq[word]))
            scored[word] = tf * idf

        # Take top words by TF-IDF score
        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        cluster_keywords[cluster_id] = [w for w, _ in top[:top_words]]

    return cluster_keywords


def get_cluster_names_via_llm(articles: List[Dict], labels: np.ndarray, n_clusters: int,
                               model: str = "qwen3:8b") -> Dict[int, str]:
    """
    Use Ollama LLM to generate meaningful category names for each cluster.

    Processes clusters largest-first. For each cluster:
    1. Gather top 30 TF-IDF distinctive words with raw occurrence counts
    2. Include 15 sample article titles for context
    3. Show the LLM all previously assigned names (to avoid duplicates)
    4. Ask for a short, specific category name

    The growing names list ensures each cluster gets a unique name.
    """
    import random
    import math
    random.seed(42)

    print(f"\nGenerating cluster names via LLM ({model} via Ollama)...")

    try:
        from ollama import chat
        test_response = chat(model=model, messages=[{"role": "user", "content": "hi"}])
        print(f"  Ollama connected, using model: {model}")
    except Exception as e:
        print(f"  Failed to connect to Ollama: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        print(f"  And model is pulled: ollama pull {model}")
        return {}

    # --- Phase 1: Build TF-IDF word lists with raw counts per cluster ---

    stopwords = {
        'the', 'a', 'an', 'of', 'in', 'to', 'and', 'for', 'on', 'at', 'by', 'with',
        'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'can', 'need', 'used', 'this', 'that', 'these',
        'those', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'than', 'too', 'very', 'just', 'also', 'now', 'new', 'one', 'two',
        'first', 'second', 'third', 'its', 'his', 'her', 'their', 'our', 'your',
        'not', 'but', 'only', 'own', 'same', 'so', 'into', 'over', 'after',
        'before', 'between', 'under', 'during', 'while', 'about', 'then',
        'star', 'trek', 'episode', 'podcast', 'category', 'thumb',
    }

    cluster_word_counts = {}
    global_doc_freq = Counter()

    for cid in range(n_clusters):
        indices = np.where(labels == cid)[0]
        words = []
        for idx in indices:
            title_words = [w.lower().strip('(),"\'.:;!?') for w in articles[idx]['title'].split()]
            words.extend(title_words)
            words.extend(title_words)  # double-weight titles
            words.extend(articles[idx].get('keywords', []))
        filtered = [w for w in words if len(w) > 2 and w not in stopwords
                    and not w.isdigit() and w.isalpha()]
        cluster_word_counts[cid] = Counter(filtered)
        for word in set(filtered):
            global_doc_freq[word] += 1

    # Get top 30 words by TF-IDF, paired with raw counts
    cluster_top_words = {}
    for cid in range(n_clusters):
        counts = cluster_word_counts[cid]
        if not counts:
            cluster_top_words[cid] = []
            continue
        total = sum(counts.values())
        scored = {}
        for word, count in counts.items():
            tf = count / total
            idf = math.log(n_clusters / (1 + global_doc_freq[word]))
            scored[word] = tf * idf
        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:30]
        cluster_top_words[cid] = [(w, counts[w]) for w, _ in top]

    # --- Phase 2: Name clusters sequentially, largest first ---

    sorted_ids = sorted(range(n_clusters),
                        key=lambda c: int(np.sum(labels == c)), reverse=True)

    assigned_names = {}  # cluster_id -> name (insertion order = assignment order)

    for cid in sorted_ids:
        size = int(np.sum(labels == cid))
        words_with_counts = cluster_top_words.get(cid, [])

        if not words_with_counts:
            assigned_names[cid] = f"Cluster {cid}"
            print(f"    C{cid} ({size} articles): Cluster {cid}  [no words]")
            continue

        # Format: "word (count), word (count), ..."
        words_str = ", ".join(f"{w} ({c})" for w, c in words_with_counts)

        # Sample articles with titles + content snippets for context
        indices = np.where(labels == cid)[0]
        sample_indices = random.sample(indices.tolist(), min(15, len(indices)))
        samples_str = ""
        for idx in sample_indices:
            title = articles[idx]['title']
            # Truncate content to first 200 chars for context
            content = articles[idx].get('content', '')[:200].replace('\n', ' ').strip()
            if content:
                samples_str += f"  - {title}: {content}...\n"
            else:
                samples_str += f"  - {title}\n"

        # Build the growing names list
        if assigned_names:
            names_list = "\n".join(f"  - {name}" for name in assigned_names.values())
            names_section = (f"\nAlready assigned category names (you MUST NOT reuse or repeat these):\n"
                           f"{names_list}\n")
        else:
            names_section = ""

        prompt = f"""I have a cluster of {size} wiki articles from a wiki encyclopedia. Give this cluster a short, specific category name (2-5 words).

RULES:
- Reply with ONLY the category name on a single line
- No quotes, no explanation, no markdown formatting
- Be SPECIFIC - do not use the wiki's general topic as a prefix (e.g. not "Star Trek X", just describe what makes this cluster unique)
- The name should distinguish this cluster from the others
{names_section}
Distinctive words for this cluster (with occurrence counts):
{words_str}

Sample articles from this cluster (title + snippet):
{samples_str}
Category name:"""

        max_attempts = 3
        name = None
        for attempt in range(max_attempts):
            try:
                response = chat(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.3 if attempt > 0 else 0},
                    think=False
                )
                raw = response.message.content.strip()

                # Safety: strip any think tags that leak through
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                if '</think>' in raw:
                    raw = raw.split('</think>')[-1].strip()

                # Take first non-empty line
                for line in raw.split('\n'):
                    line = line.strip()
                    if line:
                        raw = line
                        break

                # Strip markdown formatting
                raw = raw.replace('**', '').replace('*', '').replace('`', '')

                # Strip quotes
                raw = raw.strip('"\'').strip()

                # Reject if too long (verbose/conversational response)
                word_count = len(raw.split())
                if word_count > 8 or len(raw) > 80:
                    if attempt < max_attempts - 1:
                        continue  # retry
                    # Last attempt: take first 5 words
                    raw = ' '.join(raw.split()[:5])

                # Reject empty
                if not raw or len(raw) < 2:
                    if attempt < max_attempts - 1:
                        continue
                    raw = f"Cluster {cid}"

                name = raw.title()

                # Check for duplicates
                existing_names_lower = {n.lower() for n in assigned_names.values()}
                if name.lower() in existing_names_lower:
                    if attempt < max_attempts - 1:
                        # Add hint to prompt for retry
                        prompt += f"\n\nIMPORTANT: '{name}' is already taken. Choose a DIFFERENT name."
                        name = None
                        continue
                    # Last attempt: append disambiguator
                    name = f"{name} ({cid})"

                break  # success

            except Exception as e:
                if attempt == max_attempts - 1:
                    name = f"Cluster {cid}"
                    print(f"    C{cid} ({size} articles): Error - {e}")

        if name:
            assigned_names[cid] = name
            print(f"    C{cid} ({size} articles): {name}")

    print(f"  Named {len(assigned_names)}/{n_clusters} clusters")
    return assigned_names


def create_network_graph(articles: List[Dict], embeddings: np.ndarray, labels: np.ndarray,
                         cluster_keywords: Dict, link_graph: Dict[str, Set[str]], output_path: str,
                         wiki_name: str = "Wiki", similarity_threshold: float = 0.5,
                         top_k: int = 5, cluster_names: Dict = None):
    """Create network graph using actual wiki links + semantic similarity."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    print("Creating network graph (using wiki links + semantic similarity)...")
    n = len(articles)
    title_to_idx = {a['title']: i for i, a in enumerate(articles)}

    G = nx.Graph()

    n_clusters = len(set(labels))
    colors = generate_distinct_colors(n_clusters)
    cluster_to_color = {i: colors[i] for i in range(n_clusters)}

    for i, article in enumerate(articles):
        G.add_node(i,
                   title=article['title'][:30],
                   cluster=int(labels[i]),
                   color=cluster_to_color[labels[i]])

    # Add edges from wiki links (real relationships)
    link_edges = 0
    for source_title, targets in link_graph.items():
        if source_title in title_to_idx:
            source_idx = title_to_idx[source_title]
            for target_title in targets:
                if target_title in title_to_idx:
                    target_idx = title_to_idx[target_title]
                    if not G.has_edge(source_idx, target_idx):
                        G.add_edge(source_idx, target_idx, weight=1.0, edge_type='link')
                        link_edges += 1

    print(f"  Added {link_edges} edges from wiki links")
    print(f"  Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Visualization
    fig_size = (40, 35) if n > 1000 else (32, 28)
    fig, ax = plt.subplots(figsize=fig_size)

    print("  Computing cluster-aware layout...")

    # Use cluster-aware initial positions to get good separation
    import random as _random
    _random.seed(42)
    np.random.seed(42)

    # Place cluster centers on a circle, then jitter nodes around their center
    # Use large radius and tight jitter to keep clusters well-separated
    circle_radius = 20 if n > 1000 else 15
    jitter = 3.0 if n > 1000 else 2.5
    initial_pos = {}
    for cluster_id in range(n_clusters):
        angle = 2 * np.pi * cluster_id / n_clusters
        center_x = circle_radius * np.cos(angle)
        center_y = circle_radius * np.sin(angle)
        cluster_nodes = [i for i in range(n) if labels[i] == cluster_id]
        for node in cluster_nodes:
            initial_pos[node] = np.array([
                center_x + np.random.randn() * jitter,
                center_y + np.random.randn() * jitter
            ])

    # Spring layout: high k = strong repulsion to preserve cluster separation
    # Few iterations so layout refines edges without collapsing clusters together
    k_val = 5.0 if n > 1000 else 3.5
    iters = 15 if n > 5000 else (40 if n > 1000 else 60)
    pos = nx.spring_layout(G, pos=initial_pos, k=k_val, iterations=iters, seed=42, scale=20)

    # Draw edges with adaptive alpha based on graph density
    edges = G.edges(data=True)
    link_edge_list = [(u, v) for u, v, d in edges if d.get('edge_type') == 'link']
    semantic_edge_list = [(u, v) for u, v, d in edges if d.get('edge_type') == 'semantic']

    edge_alpha = 0.04 if n > 1000 else (0.08 if n > 500 else 0.15)
    edge_width = 0.2 if n > 1000 else 0.4

    # Draw link edges
    if link_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=link_edge_list, alpha=edge_alpha,
                              width=edge_width, edge_color='#888888', ax=ax)

    # Draw semantic edges (even more subtle)
    if semantic_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=semantic_edge_list,
                              alpha=edge_alpha * 0.3, width=edge_width * 0.5,
                              edge_color='#aaccee', ax=ax)

    node_colors = [G.nodes[nd]['color'] for nd in G.nodes()]
    node_sizes = [25 + G.degree(nd) * 2 for nd in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.75, ax=ax, edgecolors='white', linewidths=0.3)

    # Label only top hub nodes (top 1% by degree, max 60)
    degrees = dict(G.degree())
    degree_threshold = np.percentile(list(degrees.values()), 99) if degrees else 5
    degree_threshold = max(degree_threshold, 3)

    labels_dict = {nd: G.nodes[nd]['title'] for nd in G.nodes() if degrees[nd] >= degree_threshold}
    max_labels = 60 if n > 1000 else 80
    if len(labels_dict) > max_labels:
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_labels]
        labels_dict = {nd: G.nodes[nd]['title'] for nd, d in sorted_nodes}

    if labels_dict:
        nx.draw_networkx_labels(G, pos, labels_dict, font_size=9, font_weight='bold',
                               font_color='#222222', ax=ax,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                        edgecolor='#aaaaaa', alpha=0.85))

    # Legend with better formatting
    legend_patches = []
    sorted_cluster_ids = sorted(cluster_keywords.keys(),
                                key=lambda c: np.sum(labels == c), reverse=True)
    for cluster_id in sorted_cluster_ids:
        size = int(np.sum(labels == cluster_id))
        if cluster_names and cluster_id in cluster_names:
            label = f"C{cluster_id}: {cluster_names[cluster_id]} ({size})"
        else:
            kw = ', '.join(cluster_keywords[cluster_id][:3])
            label = f"C{cluster_id}: {kw} ({size})"
        legend_patches.append(mpatches.Patch(color=cluster_to_color[cluster_id], label=label))

    ncol = 2 if n_clusters <= 20 else 3
    ax.legend(handles=legend_patches, loc='upper left', fontsize=10, ncol=ncol,
             framealpha=0.95, edgecolor='#999999', fancybox=True,
             borderpad=1.0, handlelength=1.5)
    ax.set_title(f'{wiki_name} Semantic Network\n'
                 f'{n} Articles, {n_clusters} Clusters, {link_edges} Wiki Links',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved network graph to {output_path}")


def _label_cluster_centroids(ax, coords, labels, articles, cluster_keywords,
                             n_clusters, colors, cluster_names=None):
    """Label cluster centroids with name/keywords - shared by scatter plots."""
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() == 0:
            continue
        cluster_coords = coords[cluster_mask]
        centroid = cluster_coords.mean(axis=0)

        if cluster_names and cluster_id in cluster_names:
            label_text = f"{cluster_names[cluster_id]}"
        else:
            label_text = f"C{cluster_id}: {', '.join(cluster_keywords.get(cluster_id, [])[:2])}"

        ax.annotate(label_text, xy=centroid, fontsize=8, fontweight='bold',
                   ha='center', va='center', zorder=1000,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=colors[cluster_id][:3] if len(colors[cluster_id]) > 3 else colors[cluster_id],
                            alpha=0.85, linewidth=1.5))


def _label_representative_points(ax, coords, labels, articles, n_clusters, colors,
                                  labels_per_cluster=3):
    """Label a few representative points per cluster (closest to centroid)."""
    from adjustText import adjust_text

    texts = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_coords = coords[cluster_indices]
        centroid = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)
        n_labels = min(labels_per_cluster, len(cluster_indices))
        closest = cluster_indices[np.argsort(distances)[:n_labels]]
        for idx in closest:
            title = articles[idx]['title'][:22]
            t = ax.text(coords[idx, 0], coords[idx, 1], title,
                       fontsize=6, alpha=0.85,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=colors[cluster_id][:3] if len(colors[cluster_id]) > 3 else colors[cluster_id],
                                alpha=0.6, linewidth=0.5))
            texts.append(t)

    if texts:
        print(f"  Adjusting {len(texts)} labels to prevent overlap...")
        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.4, alpha=0.4),
                   force_points=(0.3, 0.3), force_text=(0.5, 0.5),
                   expand_points=(1.3, 1.3), lim=300)


def create_tsne_plot(articles: List[Dict], embeddings: np.ndarray, labels: np.ndarray,
                     cluster_keywords: Dict, output_path: str, wiki_name: str = "Wiki",
                     cluster_names: Dict = None):
    """Create t-SNE 2D visualization."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n = len(articles)
    print(f"Creating t-SNE visualization ({n} articles)...")

    perplexity = min(50, n - 1)
    print(f"  Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=2,
                learning_rate='auto', init='pca')
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(26, 22))

    n_clusters = len(set(labels))
    colors = generate_distinct_colors(n_clusters)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        size = len(indices)
        if cluster_names and cluster_id in cluster_names:
            lbl = f"C{cluster_id}: {cluster_names[cluster_id]} ({size})"
        else:
            kw = ', '.join(cluster_keywords.get(cluster_id, [])[:3])
            lbl = f"C{cluster_id}: {kw} ({size})"
        ax.scatter(coords[indices, 0], coords[indices, 1],
                  c=[colors[cluster_id]], label=lbl,
                  alpha=0.55, s=18, edgecolors='none')

    # Label cluster centroids and representative points
    _label_cluster_centroids(ax, coords, labels, articles, cluster_keywords,
                            n_clusters, colors, cluster_names)
    _label_representative_points(ax, coords, labels, articles, n_clusters, colors,
                                 labels_per_cluster=2)

    ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.22, 1), ncol=1,
             framealpha=0.9, edgecolor='#cccccc')
    ax.set_title(f'{wiki_name} t-SNE Projection\n'
                 f'{n} Articles, {n_clusters} Clusters (Hybrid Embeddings)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved t-SNE plot to {output_path}")


def create_umap_plot(articles: List[Dict], embeddings: np.ndarray, labels: np.ndarray,
                     cluster_keywords: Dict, output_path: str, wiki_name: str = "Wiki",
                     cluster_names: Dict = None):
    """Create UMAP 2D visualization."""
    try:
        from umap import UMAP
    except ImportError:
        print("  Skipping UMAP (not installed - run: pip install umap-learn)")
        return

    import matplotlib.pyplot as plt

    n = len(articles)
    print(f"Creating UMAP visualization ({n} articles)...")

    n_neighbors = min(30, n - 1)
    print(f"  Running UMAP with n_neighbors={n_neighbors}...")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42,
                   metric='cosine', n_jobs=2)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(26, 22))

    n_clusters = len(set(labels))
    colors = generate_distinct_colors(n_clusters)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        size = len(indices)
        if cluster_names and cluster_id in cluster_names:
            lbl = f"C{cluster_id}: {cluster_names[cluster_id]} ({size})"
        else:
            kw = ', '.join(cluster_keywords.get(cluster_id, [])[:3])
            lbl = f"C{cluster_id}: {kw} ({size})"
        ax.scatter(coords[indices, 0], coords[indices, 1],
                  c=[colors[cluster_id]], label=lbl,
                  alpha=0.55, s=18, edgecolors='none')

    # Label cluster centroids and representative points
    _label_cluster_centroids(ax, coords, labels, articles, cluster_keywords,
                            n_clusters, colors, cluster_names)
    _label_representative_points(ax, coords, labels, articles, n_clusters, colors,
                                 labels_per_cluster=2)

    ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.22, 1), ncol=1,
             framealpha=0.9, edgecolor='#cccccc')
    ax.set_title(f'{wiki_name} UMAP Projection\n'
                 f'{n} Articles, {n_clusters} Clusters (Hybrid Embeddings)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved UMAP plot to {output_path}")


def create_similarity_heatmap(articles: List[Dict], embeddings: np.ndarray, labels: np.ndarray,
                              output_path: str, wiki_name: str = "Wiki", max_articles: int = 200):
    """Create similarity heatmap sorted by cluster with cluster boundary lines."""
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    print("Creating similarity heatmap...")

    sorted_indices = np.argsort(labels)
    n = min(len(articles), max_articles)
    if n < len(articles):
        sample_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n, dtype=int)]
    else:
        sample_indices = sorted_indices

    print(f"  Using {len(sample_indices)} articles")

    subset_embeddings = embeddings[sample_indices]
    subset_articles = [articles[i] for i in sample_indices]
    subset_labels = labels[sample_indices]

    sim_matrix = cosine_similarity(subset_embeddings)

    fig, ax = plt.subplots(figsize=(24, 22))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1,
                   interpolation='nearest')

    # Draw cluster boundary lines
    cluster_boundaries = []
    prev_label = subset_labels[0]
    for i in range(1, len(subset_labels)):
        if subset_labels[i] != prev_label:
            cluster_boundaries.append(i)
            prev_label = subset_labels[i]

    for boundary in cluster_boundaries:
        ax.axhline(boundary - 0.5, color='black', linewidth=2.5, alpha=0.6)
        ax.axvline(boundary - 0.5, color='black', linewidth=2.5, alpha=0.6)

    n_clusters = len(set(labels))
    colors = generate_distinct_colors(n_clusters)

    # Show fewer tick labels to avoid crowding (every Nth)
    tick_step = max(1, n // 50)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [subset_articles[i]['title'][:25] for i in tick_positions]
    tick_colors = [colors[subset_labels[i]] for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    for i, (xtick, ytick) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        xtick.set_color(tick_colors[i])
        ytick.set_color(tick_colors[i])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Articles (sorted by cluster)', fontsize=12)
    ax.set_ylabel('Articles (sorted by cluster)', fontsize=12)
    ax.set_title(f'{wiki_name} Similarity Heatmap\n'
                 f'Sorted by Cluster (black lines = cluster boundaries)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved heatmap to {output_path}")


def create_dendrogram(articles: List[Dict], embeddings: np.ndarray, labels: np.ndarray,
                      output_path: str, wiki_name: str = "Wiki", max_articles: int = 200):
    """Create hierarchical clustering dendrogram with colored branches."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
    from scipy.spatial.distance import pdist

    print("Creating dendrogram...")

    n = min(len(articles), max_articles)
    if n < len(articles):
        indices = np.linspace(0, len(articles)-1, n, dtype=int)
    else:
        indices = np.arange(len(articles))

    print(f"  Using {len(indices)} articles")

    subset_embeddings = embeddings[indices]
    subset_articles = [articles[i] for i in indices]
    subset_labels = labels[indices]

    distances = pdist(subset_embeddings, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')

    # Set up colored branches
    n_clusters = len(set(labels))
    colors = generate_distinct_colors(n_clusters)
    palette_hex = [plt.cm.colors.rgb2hex(c[:3]) for c in colors]
    set_link_color_palette(palette_hex)

    # Color threshold: lower = more colored branches visible
    # 0.3 of max distance shows many distinct sub-cluster colors
    max_dist = linkage_matrix[:, 2].max()
    color_threshold = 0.3 * max_dist

    fig_width = max(32, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 16))
    dendro_labels = [f"{a['title'][:25]}" for a in subset_articles]

    dendrogram(linkage_matrix, labels=dendro_labels, leaf_rotation=90, leaf_font_size=6,
               ax=ax, color_threshold=color_threshold,
               above_threshold_color='#888888')

    # Color leaf labels by their cluster
    xlbls = ax.get_xmajorticklabels()
    leaf_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    for lbl_idx, lbl in enumerate(xlbls):
        if lbl_idx < len(leaf_order):
            original_idx = leaf_order[lbl_idx]
            if original_idx < len(subset_labels):
                cluster_id = subset_labels[original_idx]
                lbl.set_color(palette_hex[cluster_id % len(palette_hex)])
                lbl.set_fontweight('bold')

    ax.set_title(f'{wiki_name} Hierarchical Clustering Dendrogram\n'
                 f'{len(indices)} Articles, Branches Colored by Cluster',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Distance (Cosine)', fontsize=13)
    ax.tick_params(axis='y', labelsize=10)

    # Add horizontal reference line at color threshold
    ax.axhline(y=color_threshold, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate(f'color threshold = {color_threshold:.2f}',
               xy=(0.02, color_threshold), xycoords=('axes fraction', 'data'),
               fontsize=9, color='#888888', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Reset palette
    set_link_color_palette(None)
    print(f"  Saved dendrogram to {output_path}")


def create_cluster_analysis(articles: List[Dict], labels: np.ndarray,
                            cluster_keywords: Dict, output_path: str, wiki_name: str = "Wiki",
                            cluster_names: Dict = None):
    """Create cluster analysis charts (bar chart + horizontal bar or pie)."""
    import matplotlib.pyplot as plt

    print("Creating cluster analysis chart...")

    n_clusters = len(cluster_keywords)
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    sorted_clusters = sorted(range(n_clusters), key=lambda x: cluster_sizes[x], reverse=True)

    colors = generate_distinct_colors(n_clusters)

    fig_height = max(12, n_clusters * 0.45)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, fig_height))

    # Left: Vertical bar chart with value annotations
    bars = ax1.bar(range(n_clusters), [cluster_sizes[i] for i in sorted_clusters],
                   color=[colors[i] for i in sorted_clusters])
    ax1.set_xlabel('Cluster (sorted by size)', fontsize=12)
    ax1.set_ylabel('Number of Articles', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_clusters))
    ax1.set_xticklabels([f'C{i}' for i in sorted_clusters], rotation=45, fontsize=8)

    # Add value labels on top of bars
    for i, (bar, cluster_id) in enumerate(zip(bars, sorted_clusters)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                f'{cluster_sizes[cluster_id]}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Right: For many clusters, use horizontal bar chart instead of pie
    if n_clusters > 12:
        y_pos = np.arange(n_clusters)
        hbars = ax2.barh(y_pos,
                        [cluster_sizes[i] for i in sorted_clusters],
                        color=[colors[i] for i in sorted_clusters])

        # Labels for horizontal bars
        hlabels = []
        for cid in sorted_clusters:
            if cluster_names and cid in cluster_names:
                hlabels.append(f"C{cid}: {cluster_names[cid][:30]}")
            else:
                hlabels.append(f"C{cid}: {', '.join(cluster_keywords[cid][:3])}")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(hlabels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Number of Articles', fontsize=12)
        ax2.set_title('Clusters Ranked by Size', fontsize=14, fontweight='bold')

        # Add percentage labels on bars
        total = len(articles)
        for bar, cid in zip(hbars, sorted_clusters):
            width = bar.get_width()
            pct = 100 * cluster_sizes[cid] / total
            ax2.text(width + total * 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{cluster_sizes[cid]} ({pct:.1f}%)',
                    ha='left', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor='none', alpha=0.7))
    else:
        # For fewer clusters, pie chart is fine
        def get_pie_label(cid):
            if cluster_names and cid in cluster_names:
                return f"C{cid}: {cluster_names[cid][:25]}"
            return f"C{cid}: {', '.join(cluster_keywords[cid][:2])}"

        ax2.pie([cluster_sizes[i] for i in sorted_clusters],
                labels=[get_pie_label(i) for i in sorted_clusters],
                colors=[colors[i] for i in sorted_clusters],
                autopct='%1.1f%%', pctdistance=0.85, labeldistance=1.1,
                textprops={'fontsize': 8})
        ax2.set_title('Cluster Composition', fontsize=14, fontweight='bold')

    plt.suptitle(f'{wiki_name} Cluster Analysis\n'
                 f'{len(articles)} Articles, {n_clusters} Clusters (Hybrid Embeddings)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved cluster analysis to {output_path}")


def save_cluster_report(articles: List[Dict], labels: np.ndarray,
                        cluster_keywords: Dict, link_graph: Dict, output_path: str,
                        wiki_name: str = "Wiki", cluster_names: Dict = None,
                        model_name: str = None, clustering_method: str = 'hdbscan',
                        umap_dim: int = 10, cluster_metrics: Dict = None):
    """Save detailed cluster report as JSON."""
    print("Saving cluster report...")

    n_clusters = len(cluster_keywords)
    total_links = sum(len(targets) for targets in link_graph.values())

    report = {
        'wiki_name': wiki_name,
        'total_articles': len(articles),
        'n_clusters': n_clusters,
        'total_wiki_links': total_links,
        'model': model_name or MODEL_NAME,
        'embedding_type': 'hybrid (text + graph)',
        'graph_embedding_dim': GRAPH_EMBEDDING_DIM,
        'clustering_method': clustering_method,
        'umap_reduction_dim': umap_dim,
        'cluster_quality': cluster_metrics or {},
        'clusters': {}
    }

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        titles = [articles[i]['title'] for i in indices]

        cluster_data = {
            'size': len(indices),
            'keywords': cluster_keywords[cluster_id],
            'sample_titles': titles[:30]
        }
        if cluster_names and cluster_id in cluster_names:
            cluster_data['name'] = cluster_names[cluster_id]

        report['clusters'][f'cluster_{cluster_id}'] = cluster_data

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Saved cluster report to {output_path}")


def slugify(name: str) -> str:
    """Convert wiki name to lowercase slug for filenames."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def main():
    parser = argparse.ArgumentParser(
        description='Generate hybrid embedding visualizations (text + graph) for any MediaWiki.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_graphs.py                           # Use ALL pages (HDBSCAN default)
    python generate_graphs.py --limit 500               # Limit to 500 articles
    python generate_graphs.py --test                    # Quick test with 100 articles
    python generate_graphs.py --clustering kmeans --clusters 30  # K-Means with 30 clusters
    python generate_graphs.py --min-cluster-size 50     # HDBSCAN with larger clusters
    python generate_graphs.py --umap-dim 5              # 5D UMAP reduction before clustering
    python generate_graphs.py --text-weight 0.8         # Weight text embeddings higher
    python generate_graphs.py --wiki-name "Memory Alpha" # Custom wiki name
        """
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of articles to process (default: all)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode with 100 articles')
    parser.add_argument('--min-length', type=int, default=MIN_ARTICLE_LENGTH,
                        help=f'Minimum article length in characters (default: {MIN_ARTICLE_LENGTH})')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Text embedding batch size (default: 8, reduced for 18GB VRAM limit on GPU 1)')
    parser.add_argument('--clusters', type=int, default=20,
                        help='Number of clusters to discover (default: 20)')
    parser.add_argument('--optimal-clusters', action='store_true',
                        help='Automatically find optimal number of clusters using silhouette analysis')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild all embeddings from scratch')
    parser.add_argument('--text-weight', type=float, default=0.7,
                        help='Weight for text embeddings in hybrid (default: 0.7)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--pages-dir', type=str, default=PAGES_DIR,
                        help=f'Pages directory (default: {PAGES_DIR})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                        help=f'Model size: 0.6B (stable) or 4B (better but crashes GPUs) (default: {DEFAULT_MODEL})')
    parser.add_argument('--name-clusters', action='store_true',
                        help='Use LLM (Qwen3-8B) to generate meaningful cluster names')
    parser.add_argument('--wiki-name', type=str, default="Wiki",
                        help='Name of the wiki for titles and filenames (default: Wiki)')
    parser.add_argument('--clustering', type=str, default='hdbscan',
                        choices=['hdbscan', 'kmeans'],
                        help='Clustering algorithm: hdbscan (auto cluster count) or kmeans (default: hdbscan)')
    parser.add_argument('--umap-dim', type=int, default=10,
                        help='UMAP reduction dimensions before clustering (default: 10)')
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='HDBSCAN min_cluster_size (default: auto-scaled by dataset size)')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='HDBSCAN min_samples — lower = less noise (default: min(min_cluster_size, 10))')
    parser.add_argument('--semantic-threshold', type=float, default=0.5,
                        help='Cosine similarity threshold for semantic edges (default: 0.5)')
    parser.add_argument('--max-semantic-edges', type=int, default=10,
                        help='Max semantic edges per node (default: 10)')

    args = parser.parse_args()

    # Set model based on choice
    model_config = MODELS[args.model]
    model_name = model_config["name"]
    embedding_dim = model_config["dim"]

    # Generate filename prefix from wiki name
    file_prefix = slugify(args.wiki_name)

    if args.test:
        args.limit = 100
        print("Running in TEST mode with 100 articles")

    print("=" * 70)
    print(f"{args.wiki_name} Hybrid Embedding Visualization Generator")
    print(f"Text Model: {model_name} ({embedding_dim}D)")
    print(f"Graph Model: node2vec ({GRAPH_EMBEDDING_DIM}D)")
    print(f"Hybrid Weight: {args.text_weight:.0%} text, {1-args.text_weight:.0%} graph")
    print(f"Clustering: {args.clustering.upper()} (UMAP {args.umap_dim}D reduction)")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize database
    print(f"\nInitializing embeddings database: {EMBEDDINGS_DB}")
    conn = init_database(EMBEDDINGS_DB)

    # Load articles with link extraction
    print("\n1. Loading articles and extracting wiki links...")
    articles, link_graph = load_articles_with_links(args.pages_dir, max_articles=args.limit,
                                                     min_length=args.min_length)
    print(f"   Total loaded: {len(articles)} articles")

    if len(articles) < 10:
        print("Error: Not enough articles loaded!")
        sys.exit(1)

    # Build NetworkX graph
    print("\n2. Building link graph...")
    G, title_to_idx = build_networkx_graph(articles, link_graph)

    # Do GPU work FIRST (text embeddings) before CPU-heavy node2vec
    print(f"\n3. Loading {model_name} model...")
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    model = SentenceTransformer(model_name, device=device)

    # Generate text embeddings with caching
    print("\n4. Generating text embeddings (GPU)...")
    text_embeddings = generate_text_embeddings_with_cache(articles, model, conn,
                                                          batch_size=args.batch_size,
                                                          rebuild=args.rebuild,
                                                          embedding_dim=embedding_dim)
    print(f"   Text embeddings shape: {text_embeddings.shape}")

    # Free GPU memory before CPU-heavy node2vec
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Generate graph embeddings (CPU only - no GPU involvement)
    print("\n5. Generating graph embeddings (node2vec, CPU)...")
    graph_embeddings = generate_graph_embeddings(G, dimensions=GRAPH_EMBEDDING_DIM)

    # Add semantic edges to improve graph structure
    print("\n6. Adding semantic similarity edges...")
    G = add_semantic_edges(G, articles, text_embeddings, title_to_idx,
                          similarity_threshold=args.semantic_threshold,
                          max_edges_per_node=args.max_semantic_edges)

    # Combine embeddings
    print("\n7. Combining text and graph embeddings...")
    combined_embeddings = combine_embeddings(text_embeddings, graph_embeddings,
                                              text_weight=args.text_weight)
    print(f"   Combined embeddings shape: {combined_embeddings.shape}")

    # UMAP dimensionality reduction for clustering
    print("\n8. UMAP dimensionality reduction for clustering...")
    reduced_embeddings = reduce_dimensions(combined_embeddings, n_components=args.umap_dim)

    # Clustering
    print("\n9. Clustering...")
    if args.clustering == 'hdbscan':
        # Auto-scale min_cluster_size if not specified
        min_cs = args.min_cluster_size
        if min_cs is None:
            min_cs = max(5, len(articles) // 200)
            print(f"   Auto-scaled min_cluster_size={min_cs} for {len(articles)} articles")
        labels, cluster_info, n_clusters = cluster_hdbscan(
            reduced_embeddings,
            min_cluster_size=min_cs,
            min_samples=args.min_samples
        )
    else:
        n_clusters = min(args.clusters, len(articles) // 10)
        if args.clusters < 5 and len(articles) >= 50:
            print(f"   Note: --clusters={args.clusters} is below recommended minimum of 5")
            n_clusters = args.clusters
        else:
            n_clusters = max(n_clusters, 5)
        labels, cluster_info = discover_clusters(reduced_embeddings, n_clusters=n_clusters,
                                                  find_optimal=args.optimal_clusters)

    # Evaluate cluster quality
    print("\n10. Evaluating cluster quality...")
    cluster_metrics = evaluate_clusters(reduced_embeddings, labels)

    # Extract keywords
    cluster_keywords = get_cluster_keywords(articles, labels, n_clusters)

    print(f"\n   Found {n_clusters} clusters:")
    for i in sorted(cluster_info.keys(), key=lambda x: cluster_info[x]['size'], reverse=True):
        kw = ', '.join(cluster_keywords[i][:4])
        print(f"     Cluster {i}: {cluster_info[i]['size']} articles - {kw}")

    # Generate LLM-based cluster names if requested
    cluster_names = None
    if args.name_clusters:
        print("\n11. Generating cluster names via LLM...")
        cluster_names = get_cluster_names_via_llm(articles, labels, n_clusters)

    # Create visualizations
    print("\n12. Creating visualizations...")

    create_network_graph(
        articles, combined_embeddings, labels, cluster_keywords, link_graph,
        os.path.join(args.output_dir, f"{file_prefix}_network_graph.png"),
        wiki_name=args.wiki_name,
        similarity_threshold=0.45,
        top_k=5,
        cluster_names=cluster_names
    )

    create_tsne_plot(
        articles, combined_embeddings, labels, cluster_keywords,
        os.path.join(args.output_dir, f"{file_prefix}_tsne_scatter.png"),
        wiki_name=args.wiki_name,
        cluster_names=cluster_names
    )

    create_umap_plot(
        articles, combined_embeddings, labels, cluster_keywords,
        os.path.join(args.output_dir, f"{file_prefix}_umap_scatter.png"),
        wiki_name=args.wiki_name,
        cluster_names=cluster_names
    )

    create_similarity_heatmap(
        articles, combined_embeddings, labels,
        os.path.join(args.output_dir, f"{file_prefix}_similarity_heatmap.png"),
        wiki_name=args.wiki_name,
        max_articles=200
    )

    create_dendrogram(
        articles, combined_embeddings, labels,
        os.path.join(args.output_dir, f"{file_prefix}_dendrogram.png"),
        wiki_name=args.wiki_name,
        max_articles=200
    )

    create_cluster_analysis(
        articles, labels, cluster_keywords,
        os.path.join(args.output_dir, f"{file_prefix}_cluster_analysis.png"),
        wiki_name=args.wiki_name,
        cluster_names=cluster_names
    )

    save_cluster_report(
        articles, labels, cluster_keywords, link_graph,
        os.path.join(args.output_dir, f"{file_prefix}_clusters.json"),
        wiki_name=args.wiki_name,
        cluster_names=cluster_names,
        model_name=model_name,
        clustering_method=args.clustering,
        umap_dim=args.umap_dim,
        cluster_metrics=cluster_metrics
    )

    conn.close()

    print("\n" + "=" * 70)
    print("Done! All visualizations saved to:")
    print(f"  {args.output_dir}/")
    print("=" * 70)

    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(('.png', '.json')):
            fpath = os.path.join(args.output_dir, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()